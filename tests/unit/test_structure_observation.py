"""Source observations survive repair without becoming current chemistry."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import gemmi
import pytest
from tests.support.request_builders import whole_structure_requested_goals

from protrepair.api import process_structure
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.errors import ModelInvariantError, StructureNormalizationError
from protrepair.geometry import Vec3
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.io.ingress_policy import OccupancyPolicy, StructureNormalizationPolicy
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import (
    FileFormat,
    StructureIngress,
    StructureProvenance,
)
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.topology import BondRelationshipType, StructureTopology
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.stereochemistry import (
    correct_sidechain_stereochemistry,
)
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.transformer.packing import PackingPlan, PackingScope, PackingSpec
from protrepair.transformer.packing.faspr.backend import FasprPackingBackend


def _atom_line(
    serial: int,
    name: str,
    element: str,
    position: Vec3,
    *,
    charge: str = "",
    altloc: str = "",
    occupancy: float = 1,
    component: str = "ALA",
) -> str:
    return (
        f"ATOM  {serial:5d} {name:^4}{altloc:1}{component:>3} A   1    "
        f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
        f"{occupancy:6.2f}{20:6.2f}          {element:>2}{charge:>2}\n"
    )


def _source_pdb(*, isotope: str = "H") -> str:
    return "".join(
        (
            _atom_line(1, "N", "N", Vec3(0, 0, 0), charge="1+"),
            _atom_line(2, "CA", "C", Vec3(1.458, 0, 0)),
            _atom_line(3, "C", "C", Vec3(2, 1.4, 0)),
            _atom_line(4, "O", "O", Vec3(1.5, 2.4, 0)),
            _atom_line(5, "CB", "C", Vec3(2, -0.7, 1.2)),
            _atom_line(6, "H1", isotope, Vec3(-0.7, 0.7, 0)),
            "CONECT    1    6\nEND\n",
        )
    )


def _mmcif_with_charges(pdb: str, charges: tuple[str, ...]) -> str:
    document = gemmi.read_pdb_string(pdb).make_mmcif_document()
    values = document.sole_block().find_values("_atom_site.pdbx_formal_charge")
    assert len(values) == len(charges)
    for index, token in enumerate(charges):
        values[index] = token
    return document.as_string()


@pytest.mark.parametrize(
    "token, expected", (("?", None), (".", None), ("0", 0), ("+1", 1), ("-1", -1))
)
def test_mmcif_observation_distinguishes_zero_from_unspecified_charge(
    token: str,
    expected: int | None,
) -> None:
    text = _mmcif_with_charges(_source_pdb(), (token, "?", "?", "?", "?", "?"))
    structure = read_structure_string(text, FileFormat.MMCIF)
    observation = structure.provenance.ingress.observation
    assert observation is not None
    atom_ref = AtomRef(ResidueId("A", 1), "N")
    assert observation.formal_charge(atom_ref) == expected
    assert structure.topology.formal_charge(AtomIndex(0)) == expected
    assert observation.formal_charge(AtomRef(ResidueId("A", 1), "NEW")) is None


@pytest.mark.parametrize(
    "token, expected", (("", None), ("0+", 0), ("0-", 0), ("1+", 1), ("1-", -1))
)
@pytest.mark.parametrize("output_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_charge_roundtrip_uses_current_value_not_original_observation(
    token: str,
    expected: int | None,
    output_format: FileFormat,
) -> None:
    text = _atom_line(1, "N", "N", Vec3(0, 0, 0), charge=token)
    source = read_structure_string(text, FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    ref = AtomRef(ResidueId("A", 1), "N")
    assert observation.formal_charge(ref) == expected

    original_roundtrip = read_structure_string(
        write_structure_string(source, output_format), output_format
    )
    assert original_roundtrip.topology.formal_charge(AtomIndex(0)) == expected
    changed = source.with_updated_residue_facets(
        source.constitution.residue_site_at(ResidueIndex(0)),
        residue_geometry=source.residue_geometry(ResidueIndex(0)),
        formal_charge_by_atom_name=(("N", 0),),
    )
    assert changed.provenance.ingress.observation is observation
    assert observation.formal_charge(ref) == expected
    reread = read_structure_string(
        write_structure_string(changed, output_format), output_format
    )
    assert reread.topology.formal_charge(AtomIndex(0)) == 0
    assert reread.provenance.ingress.observation is not observation


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("use_altloc", (False, True))
@pytest.mark.parametrize(
    "policy, expected", ((OccupancyPolicy.HIGHEST, 0), (OccupancyPolicy.LOWEST, None))
)
def test_charge_occurrences_follow_duplicate_atom_and_altloc_selection(
    file_format: FileFormat,
    use_altloc: bool,
    policy: OccupancyPolicy,
    expected: int | None,
) -> None:
    text = _atom_line(
        1, "N", "N", Vec3(0, 0, 0), altloc="A" if use_altloc else "", occupancy=0.4
    ) + _atom_line(
        2,
        "N",
        "N",
        Vec3(1, 0, 0),
        charge="0+",
        altloc="B" if use_altloc else "",
        occupancy=0.6,
    )
    if file_format is FileFormat.MMCIF:
        text = _mmcif_with_charges(text, ("?", "0"))
    result = read_structure_string(
        text, file_format, policy=StructureNormalizationPolicy(occupancy_policy=policy)
    )
    assert result.topology.formal_charge(AtomIndex(0)) == expected
    observation = result.provenance.ingress.observation
    assert observation is not None
    assert observation.formal_charge(AtomRef(ResidueId("A", 1), "N")) == expected
    assert len(observation.constitution.atom_slots) == 1


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_charge_observation_uses_first_model(file_format: FileFormat) -> None:
    first = _atom_line(1, "N", "N", Vec3(0, 0, 0), charge="0+")
    second = _atom_line(1, "N", "N", Vec3(1, 0, 0), charge="1+")
    text = f"MODEL        1\n{first}ENDMDL\nMODEL        2\n{second}ENDMDL\nEND\n"
    if file_format is FileFormat.MMCIF:
        text = _mmcif_with_charges(text, ("0", "1"))
    result = read_structure_string(text, file_format)
    assert result.topology.formal_charge(AtomIndex(0)) == 0
    observation = result.provenance.ingress.observation
    assert observation is not None
    assert observation.geometry.atom_geometry(AtomIndex(0)).position == Vec3(0, 0, 0)


@pytest.mark.parametrize(
    "file_format, token",
    (
        (FileFormat.PDB, "++"),
        (FileFormat.PDB, "x+"),
        (FileFormat.MMCIF, "0.5"),
        (FileFormat.MMCIF, "nan"),
    ),
)
def test_malformed_source_charge_is_not_silently_discarded(
    file_format: FileFormat, token: str
) -> None:
    text = _atom_line(
        1,
        "N",
        "N",
        Vec3(0, 0, 0),
        charge=token if file_format is FileFormat.PDB else "",
    )
    if file_format is FileFormat.MMCIF:
        text = _mmcif_with_charges(text, (token,))
    with pytest.raises(StructureNormalizationError):
        read_structure_string(text, file_format)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("targeted", (False, True))
def test_observed_hydrogens_survive_removal_heavy_and_hydrogen_rebuild(
    isotope: str, targeted: bool
) -> None:
    source = read_structure_string(_source_pdb(isotope=isotope), FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    residue_id = ResidueId("A", 1)
    h_ref = AtomRef(residue_id, "H1")
    n_ref = AtomRef(residue_id, "N")
    assert observation.hydrogen_atoms(residue_id) == (h_ref,)
    declarations = observation.bonds_for_atom(h_ref)
    assert len(declarations) == 1
    assert declarations[0].relationship_type is BondRelationshipType.UNKNOWN
    assert observation.constitution.atom_ref_at(declarations[0].atom_index_1) == n_ref
    assert observation.bonds_for_atom(AtomRef(residue_id, "CA")) == ()
    assert len(source.topology.bonds) > len(observation.topology.bonds) == 1
    original_h_index = observation.constitution.resolve_atom_index(h_ref)
    assert original_h_index is not None
    assert observation.constitution.atom_site_at(original_h_index).element == isotope
    original_h_position = observation.geometry.atom_geometry(original_h_index).position

    stripped = source.without_hydrogens()
    targets = frozenset((residue_id,)) if targeted else None
    repaired = repair_heavy_atoms_core(stripped, target_residue_ids=targets).structure
    hydrogenated = materialize_hydrogens_core(
        repaired, target_residue_ids=targets
    ).structure
    for result in (stripped, repaired, hydrogenated, add_hydrogens(source).structure):
        assert result.provenance.ingress.observation is observation
        assert observation.hydrogen_atoms(residue_id) == (h_ref,)
        assert (
            observation.geometry.atom_geometry(original_h_index).position
            == original_h_position
        )
        assert observation.formal_charge(n_ref) == 1
    assert not any(atom.is_hydrogen() for atom in stripped.constitution.atom_slots)
    assert sum(atom.is_hydrogen() for atom in hydrogenated.constitution.atom_slots) > 1


def test_observation_keeps_original_identity_after_rename_and_reorder() -> None:
    source = read_structure_string(_source_pdb(), FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    payload = CompletionResiduePayload(
        residue_site=source.constitution.residue_site_at(ResidueIndex(0)),
        residue_geometry=source.residue_geometry(ResidueIndex(0)),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(
            ResidueIndex(0)
        ),
    ).renamed_atoms({"N": "NX"})
    payload = payload.reordered(tuple(reversed(payload.atom_names())))
    moved_geometry = payload.residue_geometry.with_atom_geometry(
        "NX",
        payload.residue_geometry.atom_geometry("NX").with_position(Vec3(8, 9, 10)),
    )
    changed = source.with_updated_residue_facets_batch(
        ((payload.residue_site, moved_geometry, payload.formal_charge_by_atom_name),)
    )
    assert changed.provenance.ingress.observation is observation
    assert observation.formal_charge(AtomRef(ResidueId("A", 1), "N")) == 1
    assert (
        observation.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "NX"))
        is None
    )
    assert observation.geometry.atom_geometry(AtomIndex(0)).position == Vec3(0, 0, 0)
    assert changed.residue_geometry(ResidueIndex(0)).position("NX") == Vec3(8, 9, 10)


def test_terminal_and_faspr_projection_do_not_replace_source_observation(
    tmp_path: Path,
) -> None:
    source = read_structure_string(_source_pdb(), FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    augmented = augment_c_terminal_oxt(source).structure
    assert augmented.constitution.residue_site_at(ResidueIndex(0)).has_atom_site("OXT")
    assert augmented.provenance.ingress.observation is observation

    # Copy the backend's heavy-only input, exercising its real output reader and
    # merge without depending on native packing or a bundled rotamer library.
    executable = tmp_path / "FASPR"
    executable.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    -i) input="$2"; shift 2 ;;\n'
        '    -o) output="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'cp "$input" "$output"\n'
    )
    executable.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub")
    plan = PackingPlan.from_inputs(
        source, PackingSpec(backend_name="faspr", scope=PackingScope.FULL)
    )
    result = FasprPackingBackend(executable_path=executable).pack(plan)
    assert result.packed_structure.provenance.ingress.observation is observation
    assert observation.hydrogen_atoms(ResidueId("A", 1)) == (
        AtomRef(ResidueId("A", 1), "H1"),
    )


def test_retained_ligand_override_does_not_turn_generated_h_into_source_h() -> None:
    text = (
        _atom_line(1, "C1", "C", Vec3(0, 0, 0), component="UNK")
        + _atom_line(2, "O1", "O", Vec3(1.4, 0, 0), component="UNK")
    ).replace("ATOM  ", "HETATM")
    source = read_structure_string(text, FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    residue_id = ResidueId("A", 1)
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=ComponentLibrary(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="CO",
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        ),
    ).structure
    assert result.provenance.ingress.observation is observation
    assert observation.hydrogen_atoms(residue_id) == ()
    assert sum(atom.is_hydrogen() for atom in result.constitution.atom_slots) == 4


def test_canonical_reprocessing_does_not_capture_generated_atoms_as_original() -> None:
    source = read_structure_string(_source_pdb(), FileFormat.PDB)
    observation = source.provenance.ingress.observation
    goals = whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
    first = process_structure(source, requested_goals=goals).structure
    second = process_structure(first, requested_goals=goals).structure
    assert observation is not None
    assert first.provenance.ingress.observation is observation
    assert second.provenance.ingress.observation is observation
    assert observation.hydrogen_atoms(ResidueId("A", 1)) == (
        AtomRef(ResidueId("A", 1), "H1"),
    )
    assert sum(atom.is_hydrogen() for atom in second.constitution.atom_slots) > 1


def test_observation_rejects_misaligned_or_inferred_topology() -> None:
    source = read_structure_string(_source_pdb(), FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    with pytest.raises(ModelInvariantError, match="source declarations"):
        replace(observation, topology=source.topology)
    stripped = source.without_hydrogens()
    with pytest.raises(ModelInvariantError, match="geometry"):
        replace(observation, geometry=stripped.geometry)
    with pytest.raises(ModelInvariantError, match="topology"):
        replace(
            observation,
            topology=StructureTopology.empty(constitution=stripped.constitution),
        )
    with pytest.raises(FrozenInstanceError):
        observation.__setattr__("geometry", stripped.geometry)
    assert observation.hydrogen_atoms(ResidueId("Z", 99)) == ()
    assert observation.bonds_for_atom(AtomRef(ResidueId("A", 1), "MISSING")) == ()


def test_path_ingress_and_manual_observation_construction(tmp_path: Path) -> None:
    path = tmp_path / "input.pdb"
    path.write_text(_source_pdb())
    source = read_structure(path)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert source.provenance.ingress.source_name == "input.pdb"
    assert (
        StructureObservation.from_source_facets(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=source.topology,
        )
        == observation
    )


def test_observation_does_not_capture_template_filled_connection_semantics() -> None:
    source = read_structure_string(
        _source_pdb().replace("CONECT    1    6", "CONECT    1    2"),
        FileFormat.PDB,
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    n_ref = AtomRef(ResidueId("A", 1), "N")
    (observed_bond,) = observation.bonds_for_atom(n_ref)
    current_bond = source.topology.bond_between(AtomIndex(0), AtomIndex(1))
    assert current_bond is not None
    assert current_bond.relationship_type is BondRelationshipType.COVALENT
    assert current_bond.order == 1
    assert observed_bond.relationship_type is BondRelationshipType.UNKNOWN
    assert observed_bond.order is None


@pytest.mark.parametrize(
    ("connection_type", "relationship"),
    [
        (gemmi.ConnectionType.Covale, BondRelationshipType.COVALENT),
        (gemmi.ConnectionType.MetalC, BondRelationshipType.METAL_COORDINATION),
        (gemmi.ConnectionType.Hydrog, BondRelationshipType.HYDROGEN_BOND),
    ],
)
def test_observation_preserves_typed_source_connections(
    connection_type: gemmi.ConnectionType, relationship: BondRelationshipType
) -> None:
    raw = gemmi.read_pdb_string(_source_pdb())
    connection = gemmi.Connection()
    connection.name = "source-relationship"
    connection.type = connection_type
    for partner, atom_name in (
        (connection.partner1, "N"),
        (connection.partner2, "H1"),
    ):
        partner.chain_name = "A"
        partner.res_id.seqid = gemmi.SeqId(1, " ")
        partner.res_id.name = "ALA"
        partner.atom_name = atom_name
    raw.connections.append(connection)
    source = read_structure_string(
        raw.make_mmcif_document().as_string(), FileFormat.MMCIF
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    (bond,) = observation.bonds_for_atom(AtomRef(ResidueId("A", 1), "N"))
    assert bond.relationship_type is relationship
    assert bond.order is None


def test_mmcif_quoted_atom_identity_and_explicit_zero_are_decoded() -> None:
    raw = gemmi.read_pdb_string(_source_pdb())
    document = raw.make_mmcif_document()
    block = document.sole_block()
    for name in ("label_atom_id", "auth_atom_id"):
        column = block.find_values(f"_atom_site.{name}")
        if column:
            column[0] = "'N'"
    block.find_values("_atom_site.pdbx_formal_charge")[0] = "0"
    source = read_structure_string(document.as_string(), FileFormat.MMCIF)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert source.topology.formal_charge(AtomIndex(0)) == 0
    assert observation.formal_charge(AtomRef(ResidueId("A", 1), "N")) == 0


def test_microheterogeneous_selection_keeps_selected_component_charge() -> None:
    source = read_structure_string(
        _atom_line(1, "N", "N", Vec3(0, 0, 0), charge="1+", occupancy=0.4)
        + _atom_line(
            2, "N", "N", Vec3(1, 0, 0), charge="0+", occupancy=0.6, component="GLY"
        ),
        FileFormat.PDB,
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert source.constitution.residue_site_at(ResidueIndex(0)).component_id == "GLY"
    assert observation.formal_charge(AtomRef(ResidueId("A", 1), "N")) == 0


def test_manual_structure_without_observations_is_not_resnapshotted() -> None:
    source = read_structure_string(_source_pdb(), FileFormat.PDB)
    manual = type(source).from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=source.topology,
        provenance=StructureProvenance(ingress=StructureIngress(FileFormat.PDB)),
    )
    result = process_structure(
        manual,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure.provenance.ingress.observation is None
    assert result.structure.topology.formal_charge(AtomIndex(0)) == 1


def test_stereo_rebuild_preserves_original_observation() -> None:
    fixture = Path("tests/fixtures/corpus/pdb1afc.ent").read_text()
    source = read_structure_string(
        "\n".join(
            line
            for line in fixture.splitlines()
            if line.startswith("ATOM  ")
            and line[21] == "A"
            and line[22:26].strip() == "30"
        ),
        FileFormat.PDB,
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    original_geometry = source.residue_geometry(ResidueIndex(0))
    swapped = original_geometry.with_atom_geometries(
        (
            (
                "OG1",
                original_geometry.atom_geometry("OG1").with_position(
                    original_geometry.position("CG2")
                ),
            ),
            (
                "CG2",
                original_geometry.atom_geometry("CG2").with_position(
                    original_geometry.position("OG1")
                ),
            ),
        )
    )
    inverted = source.with_updated_residue_facets(
        source.constitution.residue_site_at(ResidueIndex(0)), residue_geometry=swapped
    )
    result = correct_sidechain_stereochemistry(inverted)
    assert result.repairs
    assert result.structure.provenance.ingress.observation is observation
    assert observation.geometry is source.geometry
    assert source.residue_geometry(ResidueIndex(0)) == original_geometry
