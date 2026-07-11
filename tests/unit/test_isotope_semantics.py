"""Cross-path regressions for hydrogen isotope element semantics."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.clashes import ClashPolicy, detect_clashes
from protrepair.diagnostics.parser_readability import (
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.errors import StructureNormalizationError
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure_string, write_structure_string
from protrepair.io.gemmi_normalization import gemmi
from protrepair.state import (
    HydrogenCoverageState,
    StructureChemistryReadinessFacts,
    StructureIntrinsicGeometryFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.discrete.axis_rotation import (
    score_discrete_correction_candidate,
)

HYDROGEN_ISOTOPE_SYMBOLS = ("H", "D", "T")


def test_hydrogen_isotopes_share_clash_scope_and_overlap() -> None:
    """Clash detection should classify H, D, and T identically."""

    component_library = build_default_component_library()
    included_overlaps: list[float] = []
    for isotope_symbol in HYDROGEN_ISOTOPE_SYMBOLS:
        structure = _inter_residue_isotope_contact(isotope_symbol)
        included = detect_clashes(
            structure,
            component_library=component_library,
            policy=ClashPolicy(include_hydrogens=True),
        )
        excluded = detect_clashes(
            structure,
            component_library=component_library,
            policy=ClashPolicy(include_hydrogens=False),
        )

        assert len(included.clashes) == 1
        assert excluded.clashes == ()
        included_overlaps.append(included.clashes[0].overlap_angstrom)

    assert included_overlaps == pytest.approx((1.9, 1.9, 1.9))


def test_hydrogen_isotopes_share_discrete_candidate_clash_ranking() -> None:
    """Heavy-clash candidate axes should exclude H, D, and T identically."""

    component_library = build_default_component_library()
    scores = tuple(
        score_discrete_correction_candidate(
            structure=_inter_residue_isotope_contact(isotope_symbol),
            residue_id=ResidueId("A", 1),
            component_library=component_library,
            moved_atom_indices=(),
        ).score
        for isotope_symbol in HYDROGEN_ISOTOPE_SYMBOLS
    )

    assert scores[0].focus_clash_count == 1
    assert scores[0].focus_heavy_clash_count == 0
    assert scores[0].focus_heavy_fractional_clash_overlap_sum == 0.0
    assert scores[0].focus_heavy_clash_overlap_sum_angstrom == 0.0
    assert scores[1:] == scores[:1] * 2


def test_hydrogen_isotopes_share_aggregate_heavy_clash_projection() -> None:
    """All-atom aggregate facts should project H, D, and T out of heavy clashes."""

    component_library = build_default_component_library()
    observed_counts = []
    for isotope_symbol in HYDROGEN_ISOTOPE_SYMBOLS:
        structure = _inter_residue_isotope_contact(isotope_symbol)
        readiness = StructureChemistryReadinessFacts.from_structure(
            structure,
            component_library=component_library,
        )
        facts = StructureIntrinsicGeometryFacts.from_structure(
            structure,
            component_library=component_library,
            chemistry_readiness_facts=replace(
                readiness,
                hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
            ),
        )

        observed_counts.append(
            (
                facts.protein_self_clash_count,
                facts.observed_hydrogen_inclusive_self_clash_count,
                facts.observed_heavy_atom_self_clash_count,
            )
        )

    assert observed_counts == [(1, 1, 0)] * 3


@pytest.mark.parametrize("isotope_symbol", HYDROGEN_ISOTOPE_SYMBOLS)
def test_hydrogen_isotope_pairs_follow_hydrogen_hydrogen_policy(
    isotope_symbol: str,
) -> None:
    """D-D and T-T contacts should not bypass the H-H pair policy."""

    structure = _isotope_pair_structure(isotope_symbol)
    component_library = build_default_component_library()

    excluded = detect_clashes(
        structure,
        component_library=component_library,
    )
    included = detect_clashes(
        structure,
        component_library=component_library,
        policy=ClashPolicy(include_hydrogen_hydrogen=True),
    )

    assert excluded.clashes == ()
    assert len(included.clashes) == 1


def test_hydrogen_isotopes_share_polymer_readiness_state() -> None:
    """Hydrogen coverage/readiness must not depend on isotope source spelling."""

    states = tuple(
        StructureChemistryReadinessFacts.from_structure(
            _polymer_isotope_structure(isotope_symbol),
            component_library=build_default_component_library(),
        ).hydrogen_coverage_state
        for isotope_symbol in HYDROGEN_ISOTOPE_SYMBOLS
    )

    assert states == (HydrogenCoverageState.PARTIAL,) * 3


def test_hydrogen_isotopes_share_parser_readability_probe() -> None:
    """Parser probing should not skip D/T structures as hydrogen-free."""

    metrics = tuple(
        probe_rdkit_no_conect_parser_readability(
            _polymer_isotope_structure(isotope_symbol),
            component_library=build_default_component_library(),
        ).metrics()
        for isotope_symbol in HYDROGEN_ISOTOPE_SYMBOLS
    )

    assert metrics[0].sanitize_readable is True
    assert metrics[1:] == metrics[:1] * 2


def test_hydrogen_isotopes_share_retained_non_polymer_readiness() -> None:
    """Retained chemistry should count source D/T as present hydrogens."""

    states = tuple(
        StructureChemistryReadinessFacts.from_structure(
            _retained_isotope_structure(isotope_symbol),
            component_library=build_default_component_library(),
        ).retained_non_polymer_facts[0].hydrogen_coverage_state
        for isotope_symbol in HYDROGEN_ISOTOPE_SYMBOLS
    )

    assert states == (HydrogenCoverageState.COMPLETE,) * 3


@pytest.mark.parametrize(
    "target_residue_ids",
    (None, frozenset({ResidueId("A", 1)})),
    ids=("whole-structure", "targeted-residue"),
)
def test_hydrogen_isotopes_share_completion_behavior(
    target_residue_ids: frozenset[ResidueId] | None,
) -> None:
    """Hydrogen completion should not reinterpret D/T as heavy atoms."""

    output_atom_names: list[tuple[str, ...]] = []
    output_chemical_symbols: list[tuple[str, ...]] = []
    retained_source_symbols: list[str] = []
    for isotope_symbol in HYDROGEN_ISOTOPE_SYMBOLS:
        result = add_hydrogens(
            _polymer_isotope_structure(isotope_symbol),
            target_residue_ids=target_residue_ids,
        )
        residue = result.structure.constitution.residue_or_ligand(ResidueId("A", 1))

        assert not result.has_errors()
        assert residue is not None
        output_atom_names.append(residue.atom_site_names())
        output_chemical_symbols.append(
            tuple(
                atom_site.element_identity.chemical_symbol
                for atom_site in residue.atom_sites
            )
        )
        retained_source_symbols.append(residue.atom_site("HA2").element)

    assert output_atom_names[1:] == output_atom_names[:1] * 2
    assert output_chemical_symbols[1:] == output_chemical_symbols[:1] * 2
    assert retained_source_symbols == list(HYDROGEN_ISOTOPE_SYMBOLS)


def test_structure_hydrogen_removal_includes_explicit_isotopes() -> None:
    """Structure-level hydrogen stripping should remove H, D, and T alike."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNX",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("D1", "D", Vec3(0.0, 1.0, 0.0)),
                            atom_payload("T1", "T", Vec3(0.0, 0.0, 1.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    stripped = structure.without_hydrogens()
    residue = stripped.constitution.residue_or_ligand(ResidueId("A", 1))

    assert residue is not None
    assert tuple(atom_site.name for atom_site in residue.atom_sites) == ("C1",)


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_isotope_source_symbols_survive_coordinate_egress(
    file_format: FileFormat,
) -> None:
    """Serialization should not collapse canonical D/T source identity to X/H."""

    structure = _two_isotope_structure(file_format)
    serialized = write_structure_string(structure, file_format)

    if file_format is FileFormat.PDB:
        element_by_atom_name = {
            line[12:16].strip(): line[76:78].strip()
            for line in serialized.splitlines()
            if line.startswith(("ATOM  ", "HETATM"))
        }
        assert element_by_atom_name == {"D1": "D", "T1": "T"}
    else:
        document = gemmi.cif.read_string(serialized)
        block = document.sole_block()
        assert tuple(block.find_loop("_atom_site.type_symbol")) == ("D", "T")
        assert tuple(block.find_loop("_atom_type.symbol")) == ("D", "T")

    reread = read_structure_string(serialized, file_format)
    residue = reread.constitution.residue_or_ligand(ResidueId("A", 1))

    assert residue is not None
    assert tuple(atom_site.element for atom_site in residue.atom_sites) == ("D", "T")
    assert tuple(
        atom_site.element_identity.isotope_mass_number
        for atom_site in residue.atom_sites
    ) == (2, 3)


def test_pdb_isotope_ingress_follows_selected_altloc_identity() -> None:
    """The selected source altloc should retain its own isotope symbol."""

    structure = read_structure_string(
        "\n".join(
            (
                _pdb_atom_line(
                    serial=1,
                    atom_name=" H1 ",
                    altloc="A",
                    occupancy=0.25,
                    element="D",
                ),
                _pdb_atom_line(
                    serial=2,
                    atom_name=" H1 ",
                    altloc="B",
                    occupancy=0.75,
                    element="T",
                ),
                "END",
                "",
            )
        ),
        FileFormat.PDB,
    )
    residue = structure.constitution.residue_or_ligand(ResidueId("A", 1))

    assert residue is not None
    assert residue.atom_site("H1").element == "T"
    residue_index = structure.constitution.residue_index(ResidueId("A", 1))
    assert structure.residue_geometry(residue_index).atom_geometry("H1").altloc == "B"


def test_pdb_isotope_ingress_ignores_conflicting_later_model() -> None:
    """A later model must not contaminate first-model isotope identity."""

    structure = read_structure_string(
        "\n".join(
            (
                "MODEL       17",
                _pdb_atom_line(serial=1, atom_name=" H1 ", element="D"),
                "ENDMDL",
                "MODEL       42",
                _pdb_atom_line(serial=1, atom_name=" H1 ", element="T"),
                "ENDMDL",
                "END",
                "",
            )
        ),
        FileFormat.PDB,
    )
    residue = structure.constitution.residue_or_ligand(ResidueId("A", 1))

    assert residue is not None
    assert residue.atom_site("H1").element == "D"


def test_pdb_isotope_ingress_rejects_conflicting_source_identity() -> None:
    """One source atom identity cannot simultaneously denote D and T."""

    payload = "\n".join(
        (
            _pdb_atom_line(serial=1, atom_name=" H1 ", element="D"),
            _pdb_atom_line(serial=2, atom_name=" H1 ", element="T"),
            "END",
            "",
        )
    )

    with pytest.raises(
        StructureNormalizationError,
        match="conflicting isotope symbols",
    ):
        read_structure_string(payload, FileFormat.PDB)


def test_mmcif_isotope_ingress_matches_author_identity_and_first_model() -> None:
    """mmCIF isotope lowering should match Gemmi's author-space identity."""

    structure = read_structure_string(
        """data_isotopes
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_model_num
ATOM 1 D D_LABEL . GLY LABEL_A 1 1 ? 0 0 0 1 20 H1 GLY A 1 7
ATOM 2 T T_LABEL . GLY LABEL_A 1 1 ? 9 9 9 1 20 H1 GLY A 1 8
#
""",
        FileFormat.MMCIF,
    )
    residue = structure.constitution.residue_or_ligand(ResidueId("A", 1))

    assert residue is not None
    assert residue.atom_site("H1").element == "D"


def test_mmcif_isotope_ingress_falls_back_from_null_author_identity() -> None:
    """A null author identity should coalesce to the row's label identity."""

    structure = read_structure_string(
        """data_isotope_label_fallback
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_model_num
ATOM 1 T T1 . GLY A 1 1 ? 0 0 0 1 20 ? ? ? ? 1
#
""",
        FileFormat.MMCIF,
    )
    residue = structure.constitution.residue_or_ligand(ResidueId("A", 1))

    assert residue is not None
    assert residue.atom_site("T1").element == "T"


def _polymer_isotope_structure(isotope_symbol: str) -> ProteinStructure:
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.9, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.6, 1.0, 0.0)),
                            atom_payload(
                                "HA2",
                                isotope_symbol,
                                Vec3(1.45, 1.0, 0.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _retained_isotope_structure(isotope_symbol: str) -> ProteinStructure:
    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="NAD",
                residue_id=ResidueId("A", 401),
                atoms=(
                    atom_payload("C5N", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("H001", isotope_symbol, Vec3(1.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _inter_residue_isotope_contact(isotope_symbol: str) -> ProteinStructure:
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNX",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("HX", isotope_symbol, Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="UNY",
                        residue_id=ResidueId("B", 1),
                        atoms=(atom_payload("C1", "C", Vec3(1.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _isotope_pair_structure(isotope_symbol: str) -> ProteinStructure:
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNX",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("H1", isotope_symbol, Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="UNY",
                        residue_id=ResidueId("B", 1),
                        atoms=(
                            atom_payload("H2", isotope_symbol, Vec3(0.5, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _two_isotope_structure(file_format: FileFormat) -> ProteinStructure:
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("D1", "D", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("T1", "T", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=file_format,
    )


def _pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    element: str,
    altloc: str = " ",
    occupancy: float = 1.0,
) -> str:
    """Build one fixed-width isotope-bearing PDB atom record."""

    return (
        f"{'ATOM':<6}{serial:>5} {atom_name}{altloc}{'GLY':>3} A{1:>4}    "
        f"{0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{occupancy:>6.2f}{20.0:>6.2f}"
        f"          {element:>2}  "
    )
