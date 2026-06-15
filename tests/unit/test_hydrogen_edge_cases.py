"""Adversarial edge cases for hydrogen placement."""
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.request_builders import ingress_options
from tests.support.structure_summary import summarize_structure

from protrepair.diagnostics import (
    ValidationIssueKind,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure, read_structure_string
from protrepair.io.ingress_policy import (
    LigandHandling,
    StructureNormalizationPolicy,
)
from protrepair.state import HydrogenCoverageState, ProteinStructureObservation
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.hydrogen import core as hydrogen_core
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.rotatable import (
    RotatableHydrogenEnvironment,
    RotatableHydrogenSearch,
)
from protrepair.transformer.completion.shared import OrderedAtomPatch
from protrepair.workflow.contracts import (
    LigandPolicy,
    MutationPolicy,
    StructureIngressOptions,
)


def test_single_residue_chain_gets_only_n_terminal_backbone_hydrogens() -> None:
    """A single-residue chain should not receive a propagated backbone H atom."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:19",),
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain_site("A").residues[0]

    assert {"H1", "H2", "H3"}.issubset(set(residue.atom_site_names()))
    assert "H" not in residue.atom_site_names()


def test_proline_does_not_receive_backbone_hydrogen_from_previous_residue() -> None:
    """A residue preceding proline should not add the backbone H onto proline."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:10", "A:11"),
    )

    result = add_hydrogens(structure)
    proline = result.structure.chain_site("A").residues[1]

    assert proline.component_id == "PRO"
    assert "H" not in proline.atom_site_names()


def test_targeted_hydrogen_materialization_processes_only_target_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Targeted hydrogenation should avoid non-target chain payload work."""

    first_residue_id = ResidueId("A", 1)
    target_residue_id = ResidueId("B", 1)
    structure = build_canonical_structure(
        chains=(
            chain_payload("A", (_glycine_residue(first_residue_id, x_offset=0.0),)),
            chain_payload("B", (_glycine_residue(target_residue_id, x_offset=5.0),)),
        ),
        source_format=FileFormat.PDB,
        source_name="targeted-hydrogen-materialization",
    )
    processed_chain_ids: list[str] = []
    original_hydrogenate_chain_stage = hydrogen_core._hydrogenate_chain_stage

    def counted_hydrogenate_chain_stage(*args, **kwargs):
        processed_chain_ids.append(kwargs["chain_id"])
        return original_hydrogenate_chain_stage(*args, **kwargs)

    monkeypatch.setattr(
        hydrogen_core,
        "_hydrogenate_chain_stage",
        counted_hydrogenate_chain_stage,
    )

    result = materialize_hydrogens_core(
        structure,
        target_residue_ids=frozenset((target_residue_id,)),
    )

    first_residue = result.structure.constitution.chain("A").residues[0]
    target_residue = result.structure.constitution.chain("B").residues[0]
    assert processed_chain_ids == ["B"]
    assert not any(atom_site.element == "H" for atom_site in first_residue.atom_sites)
    assert any(atom_site.element == "H" for atom_site in target_residue.atom_sites)


@pytest.mark.parametrize(
    ("distance", "expect_hg"),
    (
        pytest.param(3.0, False, id="threshold-bond"),
        pytest.param(3.01, True, id="outside-threshold"),
    ),
)
def test_cysteine_hg_depends_on_disulfide_distance_threshold(
    distance: float, expect_hg: bool
) -> None:
    """Cysteine HG placement should flip exactly at the disulfide cutoff."""

    structure = disulfide_threshold_structure(distance)

    result = add_hydrogens(structure)
    first_residue, second_residue = result.structure.chain_site("A").residues

    assert first_residue.has_atom_site("HG") is expect_hg
    assert second_residue.has_atom_site("HG") is expect_hg


def test_disulfide_cysteine_without_hg_counts_as_complete_hydrogen_coverage() -> None:
    """Hydrogen readiness should share disulfide-CYS semantics with hydrogenation."""

    structure = disulfide_threshold_structure(3.0)

    result = add_hydrogens(structure)
    observation = ProteinStructureObservation.from_structure(result.structure)

    assert observation.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE


@pytest.mark.parametrize(
    ("count", "expected_protonated"),
    (
        pytest.param(4, 0, id="four-his-no-protonation"),
        pytest.param(5, 1, id="five-his-one-protonated"),
    ),
)
def test_histidine_protonation_threshold_is_deterministic(
    count: int, expected_protonated: int
) -> None:
    """The 20%-of-HIS rule should switch on only once the fifth HIS appears."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:41", "A:93", "A:102", "A:106", "A:124")[:count],
    )

    result = add_hydrogens(structure, protonate_histidines=True)
    residues = result.structure.chain_site("A").residues
    protonated = [residue for residue in residues if residue.has_atom_site("HD1")]

    assert len(protonated) == expected_protonated
    if expected_protonated:
        assert protonated[0].residue_id == residues[0].residue_id


def test_insertion_code_survives_class6_hydrogen_placement() -> None:
    """Insertion codes should survive contextual hydrogen placement unchanged."""

    structure = structure_from_tokens(
        Path("tests/fixtures/pdb/1aho.pdb"),
        ("A:40",),
    )
    original_residue = structure.chain_site("A").residues[0]
    residue_site, residue_geometry, formal_charge_by_atom_name = (
        residue_payload_from_structure(structure, original_residue.residue_id)
    )
    insertion_residue = (
        residue_site.with_residue_id(
            ResidueId(chain_id="A", seq_num=40, insertion_code="A")
        ),
        residue_geometry,
        formal_charge_by_atom_name,
    )
    structure = rebuild_single_chain_structure(
        structure,
        chain_id="A",
        residues=(insertion_residue,),
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain_site("A").residues[0]

    assert residue.residue_id.insertion_code == "A"
    assert residue.has_atom_site("HG")


def test_rotatable_hydrogen_falls_back_to_initial_position_on_legacy_index_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy duplicate residue numbers should fall back to the initial hydrogen."""

    candidate_hydrogens = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    energy_calls = iter((5.0, 6.0, 1.0, 2.0))

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: tuple(
            Vec3.from_iterable(candidate) for candidate in candidate_hydrogens
        ),
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: next(energy_calls),
    )

    result = RotatableHydrogenSearch(
        outer_anchor=[0.0, 0.0, 0.0],
        inner_anchor=[0.0, 0.0, 0.0],
        donor=[0.0, 0.0, 0.0],
        hydrogen=[9.0, 9.0, 9.0],
        build_bond_length=1.0,
        reproject_bond_length=1.0,
        dihedral=0.0,
        partial_charge=0.0,
        sigma=0.0,
        epsilon=0.0,
    ).optimized_coordinate(
        residue_number="7",
        environments=(
            RotatableHydrogenEnvironment(
                residue_number="7",
                atom_x=(1.0,),
                atom_y=(0.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
            RotatableHydrogenEnvironment(
                residue_number="7",
                atom_x=(0.0,),
                atom_y=(1.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
        ),
    )

    assert result == Vec3(9.0, 9.0, 9.0)


def test_ordered_patch_exposes_backbone_hydrogen_anchor_positions() -> None:
    """Backbone-H propagation should preserve the legacy positional anchor quirk."""

    payload = OrderedAtomPatch.from_atom_coordinates(
        atom_names=["N", "C", "O", "CA", "CB", "OG"],
        atom_coordinates=[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
        ],
    )

    alpha_anchor = list(payload.position("C"))
    nitrogen_anchor = list(payload.position("N"))

    assert alpha_anchor == [2.0, 2.0, 2.0]
    assert nitrogen_anchor == [1.0, 1.0, 1.0]


def test_ligand_keep_with_water_does_not_pollute_hydrogenated_structure() -> None:
    """Ligand retention should keep non-water ligands and still hydrogenate chains."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=5,
                    record_name="HETATM",
                    atom_name=" O  ",
                    residue_name="HOH",
                    chain_id="A",
                    residue_seq=2,
                    x=6.0,
                    y=6.0,
                    z=6.0,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=6,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="A",
                    residue_seq=3,
                    x=7.0,
                    y=7.0,
                    z=7.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain_site("A").residues[0]

    assert tuple(
        ligand.component_id for ligand in result.structure.constitution.ligands
    ) == ("FAD",)
    assert {"H1", "H2", "H3"}.issubset(set(residue.atom_site_names()))


def test_hydrogen_placement_is_stable_on_already_hydrogenated_input() -> None:
    """Re-running hydrogen placement should not drift the semantic structure."""

    structure = read_structure(
        Path("tests/fixtures/pdb/1aho.pdb"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )

    first_pass = add_hydrogens(structure)
    second_pass = add_hydrogens(first_pass.structure)

    assert summarize_structure(second_pass.structure) == summarize_structure(
        first_pass.structure
    )


def test_unsupported_component_skips_only_that_residue() -> None:
    """Unsupported residues should not block supported neighbors in one chain."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:19", "A:20"),
    )
    first_residue, second_residue = structure.chain_site("A").residues
    first_payload = residue_payload_from_structure(
        structure, first_residue.residue_id
    )
    second_site, second_geometry, second_formal_charge_by_atom_name = (
        residue_payload_from_structure(structure, second_residue.residue_id)
    )
    structure = rebuild_single_chain_structure(
        structure,
        chain_id="A",
        residues=(
            first_payload,
            (
                second_site.with_component_id("MLY"),
                second_geometry,
                second_formal_charge_by_atom_name,
            ),
        ),
    )

    result = add_hydrogens(structure)
    first_after, second_after = result.structure.chain_site("A").residues

    assert "H1" in first_after.atom_site_names()
    assert "H" not in second_after.atom_site_names()
    assert result.has_warnings()
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        and "parent LYS" in issue.message
        for issue in result.issues
    )


def test_unsupported_component_isolation_is_per_chain_not_global() -> None:
    """An unsupported residue in one chain should not block another chain."""

    supported_structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:19",),
    )
    unsupported_source = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:20",),
    ).chain_site("A").residues[0]
    supported_payload = residue_payload_from_structure(
        supported_structure,
        supported_structure.chain_site("A").residues[0].residue_id,
    )
    unsupported_site, unsupported_geometry, unsupported_formal_charge_by_atom_name = (
        residue_payload_from_structure(
            structure_from_tokens(
                Path("tests/fixtures/corpus/pdb1afc.ent"),
                ("A:20",),
            ),
            unsupported_source.residue_id,
        )
    )
    structure = build_canonical_structure(
        chains=(
            chain_payload("A", (supported_payload,)),
            chain_payload(
                "B",
                (
                    (
                        unsupported_site.with_component_id("MLY").with_residue_id(
                            ResidueId(
                                chain_id="B",
                                seq_num=unsupported_source.residue_id.seq_num,
                            )
                        ),
                        unsupported_geometry,
                        unsupported_formal_charge_by_atom_name,
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="mixed-chain",
    )

    result = add_hydrogens(structure)
    supported_after = result.structure.chain_site("A").residues[0]
    unsupported_after = result.structure.chain_site("B").residues[0]

    assert supported_after.has_atom_site("H1")
    assert not unsupported_after.has_atom_site("H1")
    assert result.has_warnings()
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        and "parent LYS" in issue.message
        for issue in result.issues
    )


def test_unknown_component_reports_missing_definition_during_hydrogenation() -> None:
    """Unknown components should report a missing-definition hydrogenation gap."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:19", "A:20"),
    )
    first_residue, second_residue = structure.chain_site("A").residues
    first_payload = residue_payload_from_structure(
        structure, first_residue.residue_id
    )
    second_site, second_geometry, second_formal_charge_by_atom_name = (
        residue_payload_from_structure(structure, second_residue.residue_id)
    )
    structure = rebuild_single_chain_structure(
        structure,
        chain_id="A",
        residues=(
            first_payload,
            (
                second_site.with_component_id("UNK"),
                second_geometry,
                second_formal_charge_by_atom_name,
            ),
        ),
    )

    result = add_hydrogens(structure)

    assert any(
        issue.kind is ValidationIssueKind.MISSING_COMPONENT_DEFINITION
        and "component UNK" in issue.message
        for issue in result.issues
    )


def test_blank_chain_id_survives_hydrogenation() -> None:
    """The normalized default chain id should survive the full repair pipeline."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id=" ",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    result = add_hydrogens(structure)

    assert result.structure.chain_ids() == ("_",)
    assert result.structure.chain_site("_").residues[0].has_atom_site("H1")


def test_ligand_only_input_remains_stable_without_polymer_chains() -> None:
    """A ligand-only input should not crash or manufacture polymer chains."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="A",
                    residue_seq=1,
                    x=7.0,
                    y=7.0,
                    z=7.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    result = add_hydrogens(structure)

    assert result.structure.constitution.chains == ()
    assert tuple(
        ligand.component_id for ligand in result.structure.constitution.ligands
    ) == ("FAD",)


def test_terminal_oxt_is_added_before_hydrogenation() -> None:
    """Hydrogen placement should preserve the heavy-repair OXT terminal fix."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    result = add_hydrogens(structure)

    assert result.structure.chain_site("A").residues[0].has_atom_site("OXT")


def test_mutation_policy_controls_hydrogenated_component_choice() -> None:
    """Mutation normalization should feed the chosen residue into hydrogenation."""

    pdb_text = build_pdb_text(
        [
            build_pdb_atom_line(
                serial=1,
                atom_name=" N  ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=1.0,
                y=1.0,
                z=1.0,
                occupancy=0.80,
                element="N",
            ),
            build_pdb_atom_line(
                serial=2,
                atom_name=" CA ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=2.0,
                y=1.5,
                z=1.0,
                occupancy=0.80,
                element="C",
            ),
            build_pdb_atom_line(
                serial=3,
                atom_name=" C  ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=3.0,
                y=1.0,
                z=1.5,
                occupancy=0.80,
                element="C",
            ),
            build_pdb_atom_line(
                serial=4,
                atom_name=" O  ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=3.8,
                y=1.2,
                z=2.3,
                occupancy=0.80,
                element="O",
            ),
            build_pdb_atom_line(
                serial=5,
                atom_name=" CB ",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=2.0,
                y=2.6,
                z=0.0,
                occupancy=0.80,
                element="C",
            ),
            build_pdb_atom_line(
                serial=6,
                atom_name=" N  ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=1.1,
                y=1.1,
                z=1.2,
                occupancy=0.20,
                element="N",
            ),
            build_pdb_atom_line(
                serial=7,
                atom_name=" CA ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=2.1,
                y=1.6,
                z=1.1,
                occupancy=0.20,
                element="C",
            ),
            build_pdb_atom_line(
                serial=8,
                atom_name=" C  ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=3.1,
                y=1.1,
                z=1.6,
                occupancy=0.20,
                element="C",
            ),
            build_pdb_atom_line(
                serial=9,
                atom_name=" O  ",
                residue_name="GLY",
                chain_id="A",
                residue_seq=1,
                x=3.9,
                y=1.3,
                z=2.4,
                occupancy=0.20,
                element="O",
            ),
            "END",
        ]
    )

    highest = add_hydrogens(read_structure_string(pdb_text, FileFormat.PDB))
    lowest = add_hydrogens(
        read_structure_string(
            pdb_text,
            FileFormat.PDB,
            policy=ingress_options(
                mutation_policy=MutationPolicy.LOWEST_OCCUPANCY
            ).structure_normalization_policy(),
        )
    )

    highest_residue = highest.structure.chain_site("A").residues[0]
    lowest_residue = lowest.structure.chain_site("A").residues[0]

    assert highest_residue.component_id == "ALA"
    assert "HB1" in highest_residue.atom_site_names()
    assert lowest_residue.component_id == "GLY"
    assert {"HA1", "HA2"}.issubset(set(lowest_residue.atom_site_names()))


def test_altloc_selection_survives_hydrogenation_without_duplicate_atoms() -> None:
    """Hydrogen placement should preserve the chosen heavy-atom altloc geometry."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=5,
                    atom_name=" CB ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=2.6,
                    z=0.0,
                    occupancy=0.30,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=6,
                    atom_name=" CB ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=6.0,
                    y=6.0,
                    z=6.0,
                    occupancy=0.70,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    result = add_hydrogens(structure)
    residue = result.structure.chain_site("A").residues[0]
    cb = result.structure.geometry.atom_geometry(
        result.structure.constitution.atom_index(
            AtomRef(residue.residue_id, "CB")
        )
    )

    assert cb.position == Vec3(x=6.0, y=6.0, z=6.0)
    assert len(residue.atom_site_names()) == len(set(residue.atom_site_names()))


def test_histidine_alias_is_normalized_before_hydrogenation() -> None:
    """Histidine aliases should normalize into canonical HIS before hydrogenation."""

    structure = structure_from_tokens(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        ("A:41",),
    )
    residue_site, residue_geometry, formal_charge_by_atom_name = (
        residue_payload_from_structure(
            structure,
            structure.chain_site("A").residues[0].residue_id,
        )
    )
    structure = rebuild_single_chain_structure(
        structure,
        chain_id="A",
        residues=(
            (
                residue_site.with_component_id("HSE"),
                residue_geometry,
                formal_charge_by_atom_name,
            ),
        ),
    )

    result = add_hydrogens(structure)
    normalized = result.structure.chain_site("A").residues[0]

    assert normalized.component_id == "HIS"
    assert {"HD2", "HE1", "HE2"}.issubset(set(normalized.atom_site_names()))


def test_empty_structure_survives_hydrogenation() -> None:
    """An empty structure should pass through hydrogen placement unchanged."""

    structure = read_structure_string(build_pdb_text(["END"]), FileFormat.PDB)

    result = add_hydrogens(structure)

    assert result.structure.constitution.chains == ()
    assert result.structure.constitution.ligands == ()


def test_chain_selection_and_ligand_keep_stay_aligned_through_hydrogenation() -> None:
    """Selected chains and their ligands should stay aligned after hydrogenation."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=3.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=5,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=10.0,
                    y=1.0,
                    z=1.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=6,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=11.0,
                    y=1.5,
                    z=1.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=7,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=12.0,
                    y=1.0,
                    z=1.5,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=8,
                    atom_name=" O  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=1,
                    x=12.8,
                    y=1.2,
                    z=2.3,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=9,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="B",
                    residue_seq=2,
                    x=14.0,
                    y=7.0,
                    z=7.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=StructureNormalizationPolicy(
            ligand_handling=LigandHandling.KEEP,
            selected_chain_ids=("B",),
        ),
    )

    result = add_hydrogens(structure)

    assert result.structure.chain_ids() == ("B",)
    assert result.structure.chain_site("B").residues[0].has_atom_site("H1")
    assert tuple(
        ligand.component_id for ligand in result.structure.constitution.ligands
    ) == ("FAD",)


def structure_from_tokens(
    path: Path, residue_tokens: tuple[str, ...]
) -> ProteinStructure:
    """Extract a minimal canonical structure from selected real residues."""

    structure = read_structure(
        path,
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue_id_by_token = {
        residue_site.residue_id.display_token(): residue_site.residue_id
        for chain_site in structure.constitution.chains
        for residue_site in chain_site.residues
    }
    residue_ids = tuple(residue_id_by_token[token] for token in residue_tokens)
    chain_id = residue_ids[0].chain_id
    residues = tuple(
        residue_payload_from_structure(structure, residue_id)
        for residue_id in residue_ids
    )
    return rebuild_single_chain_structure(
        structure,
        chain_id=chain_id,
        residues=residues,
    )


def disulfide_threshold_structure(distance: float) -> ProteinStructure:
    """Build a two-cysteine structure with a chosen SG-SG distance."""

    structure = structure_from_tokens(
        Path("tests/fixtures/pdb/1aho.pdb"),
        ("A:12", "A:63"),
    )
    first_residue_id, second_residue_id = tuple(
        residue_site.residue_id
        for residue_site in structure.chain_site("A").residues
    )
    first_payload = residue_payload_from_structure(structure, first_residue_id)
    second_payload = residue_payload_from_structure(structure, second_residue_id)
    first_sg = first_payload[1].position("SG")
    second_sg = second_payload[1].position("SG")
    target_sg = Vec3(
        x=first_sg.x + distance,
        y=first_sg.y,
        z=first_sg.z,
    )
    shifted_second = translate_residue_payload(
        second_payload,
        dx=target_sg.x - second_sg.x,
        dy=target_sg.y - second_sg.y,
        dz=target_sg.z - second_sg.z,
    )
    return rebuild_single_chain_structure(
        structure,
        chain_id="A",
        residues=(first_payload, shifted_second),
    )


def translate_residue_payload(
    residue: CanonicalResiduePayload,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> CanonicalResiduePayload:
    """Translate all residue atom coordinates by a fixed offset."""

    residue_site, residue_geometry, formal_charge_by_atom_name = residue
    translated_geometry = residue_geometry.with_atom_geometries(
        (
            (
                atom_name,
                atom_geometry.with_position(
                    atom_geometry.position.with_offset(dx, dy, dz)
                ),
            )
            for atom_name, atom_geometry in residue_geometry.atoms_by_name.items()
        )
    )
    return (
        residue_site,
        translated_geometry,
        formal_charge_by_atom_name,
    )


def _glycine_residue(
    residue_id: ResidueId,
    *,
    x_offset: float,
) -> CanonicalResiduePayload:
    """Return one minimal polymer glycine residue for hydrogenation tests."""

    return residue_payload(
        component_id="GLY",
        residue_id=residue_id,
        atoms=(
            atom_payload("N", "N", Vec3(x_offset, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(x_offset + 1.45, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(x_offset + 2.0, 1.3, 0.0)),
            atom_payload("O", "O", Vec3(x_offset + 2.0, 2.4, 0.0)),
        ),
    )


def residue_payload_from_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CanonicalResiduePayload:
    """Project one canonical residue payload from a structure."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    return (
        residue_site,
        residue_geometry,
        structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )


def rebuild_single_chain_structure(
    structure: ProteinStructure,
    *,
    chain_id: str,
    residues: tuple[CanonicalResiduePayload, ...],
) -> ProteinStructure:
    """Rebuild one single-chain structure from canonical residue payloads."""

    return build_canonical_structure(
        chains=(chain_payload(chain_id, residues),),
        source_format=structure.provenance.ingress.source_format,
        source_name=structure.provenance.ingress.source_name,
    )


def build_pdb_text(lines: list[str]) -> str:
    """Join fixed-width PDB records into a text payload."""

    return "\n".join(lines) + "\n"


def build_pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_seq: int,
    record_name: str = "ATOM",
    altloc: str = " ",
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    occupancy: float = 1.0,
    b_factor: float = 20.0,
    element: str = "",
) -> str:
    """Build one fixed-width PDB atom record for ingress edge tests."""

    return (
        f"{record_name:<6}{serial:>5} {atom_name}{altloc}{residue_name:>3} "
        f"{chain_id}{residue_seq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}"
        f"          {element:>2}  "
    )
