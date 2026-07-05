"""Analysis runtime tests over the canonical structure model."""

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.analysis.kinds import AnalysisKind
from protrepair.analysis.ramachandran import _ramachandran_category
from protrepair.analysis.runtime import build_analysis_bundle
from protrepair.analysis.secondary_structure import _secondary_structure_label
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


def test_build_analysis_bundle_returns_requested_outputs_only() -> None:
    """Analysis runtime should populate only explicitly requested outputs."""

    structure = analysis_fixture_structure()

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset(
            {
                AnalysisKind.SECONDARY_STRUCTURE,
                AnalysisKind.RAMACHANDRAN,
            }
        ),
    )

    assert bundle.secondary_structure is not None
    assert bundle.ramachandran is not None
    assert bundle.has(AnalysisKind.SECONDARY_STRUCTURE)
    assert bundle.has(AnalysisKind.RAMACHANDRAN)


def test_build_analysis_bundle_computes_backbone_torsions() -> None:
    """Ramachandran output should include torsions for internal residues."""

    structure = analysis_fixture_structure()

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.RAMACHANDRAN}),
    )

    assert bundle.secondary_structure is None
    assert bundle.ramachandran is not None
    first_point, middle_point, last_point = bundle.ramachandran.points
    assert first_point.phi_degrees is None
    assert first_point.psi_degrees is not None
    assert middle_point.phi_degrees is not None
    assert middle_point.psi_degrees is not None
    assert middle_point.category in {"helix", "beta", "left_handed", "other"}
    assert last_point.phi_degrees is not None
    assert last_point.psi_degrees is None


def test_build_analysis_bundle_assigns_coarse_secondary_structure_labels() -> None:
    """Secondary-structure output should assign one coarse label per residue."""

    structure = analysis_fixture_structure()

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.SECONDARY_STRUCTURE}),
    )

    assert bundle.secondary_structure is not None
    assert bundle.ramachandran is None
    assert len(bundle.secondary_structure.assignments) == 3
    assert {
        assignment.label for assignment in bundle.secondary_structure.assignments
    } <= {"H", "E", "C"}


def test_ramachandran_torsions_accept_sane_chain_slot_numbering_gap() -> None:
    """Residue-numbering gaps do not break sane chain-slot peptide neighbors."""

    structure = analysis_fixture_structure(seq_nums=(1, 3, 4))

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.RAMACHANDRAN}),
    )

    assert bundle.ramachandran is not None
    first_point, middle_point, last_point = bundle.ramachandran.points
    assert first_point.psi_degrees is not None
    assert middle_point.phi_degrees is not None
    assert middle_point.psi_degrees is not None
    assert middle_point.category in {"helix", "beta", "left_handed", "other"}
    assert last_point.phi_degrees is not None


def test_ramachandran_torsions_accept_insertion_code_peptide_neighbor() -> None:
    """Insertion-code residue ids still support torsions when chain geometry is sane."""

    structure = analysis_fixture_structure(
        seq_nums=(100, 100, 101),
        insertion_codes=(None, "A", None),
    )

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.RAMACHANDRAN}),
    )

    assert bundle.ramachandran is not None
    first_point, middle_point, last_point = bundle.ramachandran.points
    assert first_point.psi_degrees is not None
    assert middle_point.phi_degrees is not None
    assert middle_point.psi_degrees is not None
    assert middle_point.category in {"helix", "beta", "left_handed", "other"}
    assert last_point.phi_degrees is not None


def test_ramachandran_torsions_require_sane_peptide_cn_geometry() -> None:
    """Immediate sequence ids are not enough when C-N geometry is implausible."""

    structure = analysis_fixture_structure(residue2_n=Vec3(20.0, 20.0, 20.0))

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.RAMACHANDRAN}),
    )

    assert bundle.ramachandran is not None
    first_point, middle_point, _last_point = bundle.ramachandran.points
    assert first_point.psi_degrees is None
    assert middle_point.phi_degrees is None
    assert middle_point.category is None


def test_ramachandran_torsions_accept_sane_topology_covalent_gap() -> None:
    """Canonical topology can authorize a numbering gap when geometry is sane."""

    structure = _with_peptide_topology_bond(
        analysis_fixture_structure(seq_nums=(1, 3, 4)),
        left_residue_id=ResidueId(chain_id="A", seq_num=1),
        right_residue_id=ResidueId(chain_id="A", seq_num=3),
    )

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.RAMACHANDRAN}),
    )

    assert bundle.ramachandran is not None
    first_point, middle_point, _last_point = bundle.ramachandran.points
    assert first_point.psi_degrees is not None
    assert middle_point.phi_degrees is not None
    assert middle_point.psi_degrees is not None
    assert middle_point.category in {"helix", "beta", "left_handed", "other"}


def test_ramachandran_torsions_reject_implausible_non_covalent_topology_gap() -> None:
    """Non-covalent topology must not authorize implausible peptide geometry."""

    structure = _with_peptide_topology_bond(
        analysis_fixture_structure(
            seq_nums=(1, 3, 4),
            residue2_n=Vec3(20.0, 20.0, 20.0),
        ),
        left_residue_id=ResidueId(chain_id="A", seq_num=1),
        right_residue_id=ResidueId(chain_id="A", seq_num=3),
        relationship_type=BondRelationshipType.METAL_COORDINATION,
    )

    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.RAMACHANDRAN}),
    )

    assert bundle.ramachandran is not None
    first_point, middle_point, _last_point = bundle.ramachandran.points
    assert first_point.psi_degrees is None
    assert middle_point.phi_degrees is None
    assert middle_point.psi_degrees is not None
    assert middle_point.category is None


@pytest.mark.parametrize(
    ("phi_degrees", "psi_degrees", "expected_category"),
    (
        (-60.0, -45.0, "helix"),
        (-120.0, 120.0, "beta"),
        (-120.0, -130.0, "beta"),
        (60.0, 60.0, "left_handed"),
        (-90.0, -90.0, "helix"),
        (-20.0, 45.0, "helix"),
        (-180.0, 90.0, "beta"),
        (-40.0, 180.0, "beta"),
        (-180.0, -120.0, "beta"),
        (-40.0, -180.0, "beta"),
        (20.0, -20.0, "left_handed"),
        (120.0, 120.0, "left_handed"),
        (0.0, 0.0, "other"),
        (None, -45.0, None),
        (-60.0, None, None),
    ),
)
def test_ramachandran_category_projection_is_exact(
    phi_degrees: float | None,
    psi_degrees: float | None,
    expected_category: str | None,
) -> None:
    """Documented coarse Ramachandran bins should stay exact."""

    assert (
        _ramachandran_category(
            phi_degrees=phi_degrees,
            psi_degrees=psi_degrees,
        )
        == expected_category
    )


@pytest.mark.parametrize(
    ("category", "expected_label"),
    (
        ("helix", "H"),
        ("beta", "E"),
        ("left_handed", "C"),
        ("other", "C"),
        (None, "C"),
    ),
)
def test_secondary_structure_label_projection_is_exact(
    category: str | None,
    expected_label: str,
) -> None:
    """Secondary structure should project only documented coarse labels."""

    assert _secondary_structure_label(category) == expected_label


def analysis_fixture_structure(
    *,
    seq_nums: tuple[int, int, int] = (1, 2, 3),
    insertion_codes: tuple[str | None, str | None, str | None] = (None, None, None),
    residue2_n: Vec3 | None = None,
) -> ProteinStructure:
    """Return one small non-degenerate backbone-only structure."""

    resolved_residue2_n = (
        Vec3(2.9, 1.0, 1.2)
        if residue2_n is None
        else residue2_n
    )
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(
                            chain_id="A",
                            seq_num=seq_nums[0],
                            insertion_code=insertion_codes[0],
                        ),
                        atoms=(
                            atom("N", "N", 0.0, 0.0, 0.0),
                            atom("CA", "C", 1.1, 0.1, 0.0),
                            atom("C", "C", 1.8, 1.2, 0.4),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(
                            chain_id="A",
                            seq_num=seq_nums[1],
                            insertion_code=insertion_codes[1],
                        ),
                        atoms=(
                            atom_payload(
                                name="N",
                                element="N",
                                position=resolved_residue2_n,
                            ),
                            atom("CA", "C", 3.8, 1.8, 1.6),
                            atom("C", "C", 4.9, 1.2, 0.9),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(
                            chain_id="A",
                            seq_num=seq_nums[2],
                            insertion_code=insertion_codes[2],
                        ),
                        atoms=(
                            atom("N", "N", 5.9, 1.9, 1.3),
                            atom("CA", "C", 6.8, 1.5, 0.4),
                            atom("C", "C", 7.6, 2.5, -0.2),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="analysis-fixture",
    )


def _with_peptide_topology_bond(
    structure: ProteinStructure,
    *,
    left_residue_id: ResidueId,
    right_residue_id: ResidueId,
    relationship_type: BondRelationshipType = BondRelationshipType.COVALENT,
) -> ProteinStructure:
    left_carbon = structure.constitution.atom_index(AtomRef(left_residue_id, "C"))
    right_nitrogen = structure.constitution.atom_index(AtomRef(right_residue_id, "N"))
    topology = StructureTopology(
        constitution=structure.constitution,
        atom_topologies=structure.topology.atom_topologies,
        bonds=(
            *structure.topology.bonds,
            TopologyBond(
                atom_index_1=left_carbon,
                atom_index_2=right_nitrogen,
                relationship_type=relationship_type,
                provenance=BondProvenance.EVIDENCE_RESOLVED,
            ),
        ),
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def atom(
    atom_name: str,
    element: str,
    x: float,
    y: float,
    z: float,
) -> CanonicalAtomPayload:
    """Return one canonical atom for analysis tests."""

    return atom_payload(
        name=atom_name,
        element=element,
        position=Vec3(x=x, y=y, z=z),
    )
