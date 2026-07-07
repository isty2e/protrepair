"""Cross-family scientific geometry regression sentinels."""

import numpy as np
import pytest
from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.canonical_builders import (
    completion_payload as build_completion_payload,
)

from protrepair.analysis.kinds import AnalysisKind
from protrepair.analysis.runtime import build_analysis_bundle
from protrepair.chemistry import (
    covalent_radius_angstrom,
    van_der_waals_radius_angstrom,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import ClashPolicy, detect_clashes
from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.completion.atom.backbone import (
    PeptideCarbonylFrame,
    backbone_psi_degrees,
)
from protrepair.transformer.completion.atom.internal_coordinates import (
    InternalCoordinatePlacementTransformer,
)
from protrepair.transformer.completion.shared.domain import (
    CompletionResiduePayload,
    ResidueBackboneNeighborhood,
    ResidueCompletionSite,
)
from protrepair.transformer.context import ProteinTransformationContext

_DEFAULT_O_POSITION = Vec3(0.0, 1.0, 0.0)
_DEFAULT_CB_POSITION = Vec3(1.8, -0.75, 1.25)


def test_scientific_regression_backbone_oxygen_uses_current_context() -> None:
    """Backbone O repair must use the current residue psi, not previous context."""

    previous_residue = _ala_payload(
        seq_num=1,
        n=Vec3(-3.0, 0.0, 0.0),
        ca=Vec3(-2.0, 1.0, 0.0),
        c=Vec3(-1.0, 0.0, 0.0),
    )
    residue = _ala_payload(
        seq_num=2,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        o=None,
    )
    next_residue = _ala_payload(
        seq_num=3,
        n=Vec3(1.6, 0.0, 1.0),
        ca=Vec3(2.6, 0.4, 1.4),
        c=Vec3(3.4, 1.4, 1.2),
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_completion_payloads(
            (previous_residue, residue, next_residue)
        )
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(1),
        original_payload=residue,
    )

    repaired = InternalCoordinatePlacementTransformer(site).transform(
        ProteinTransformationContext.from_snapshot_atom_input(
            snapshot,
            site.atom_input(snapshot),
        )
    )
    repaired_residue = site.payload(repaired)
    assert repaired_residue is not None

    next_nitrogen = next_residue.position("N")
    current_psi = backbone_psi_degrees(
        (
            residue.position("N"),
            residue.position("CA"),
            residue.position("C"),
            next_nitrogen,
        )
    )
    stale_psi = backbone_psi_degrees(
        (
            previous_residue.position("N"),
            previous_residue.position("CA"),
            previous_residue.position("C"),
            residue.position("N"),
        )
    )

    actual_position = repaired_residue.position("O")
    assert current_psi == pytest.approx(96.70756579005402)
    assert stale_psi == pytest.approx(180.0)

    expected_position = Vec3(
        3.4240205679990066,
        1.1855360556074688,
        -0.6812287946260066,
    )
    stale_position = Vec3(
        2.0678769411724334,
        2.2591497601785058,
        -0.5299019336715382,
    )

    np.testing.assert_allclose(
        _vec_array(actual_position),
        _vec_array(expected_position),
        atol=1e-12,
    )
    assert np.linalg.norm(
        _vec_array(actual_position) - _vec_array(stale_position)
    ) > 1.0


def test_scientific_regression_ramachandran_accepts_sane_numbering_gap() -> None:
    """Residue numbering gaps are allowed when chain order and C-N geometry agree."""

    structure = _analysis_structure(seq_nums=(1, 3, 4))
    bundle = build_analysis_bundle(
        structure,
        requested_analyses=frozenset({AnalysisKind.RAMACHANDRAN}),
    )

    assert bundle.ramachandran is not None
    first_point, middle_point, last_point = bundle.ramachandran.points
    assert first_point.psi_degrees is not None
    assert middle_point.phi_degrees is not None
    assert middle_point.psi_degrees is not None
    assert last_point.phi_degrees is not None


def test_scientific_regression_torsion_preserves_iupac_sign() -> None:
    """Internal-coordinate torsions must distinguish mirrored signed angles."""

    assert InternalCoordinateFrame.torsion(
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 1.0, 1.0),
    ) == pytest.approx(-90.0)
    assert InternalCoordinateFrame.torsion(
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 1.0, -1.0),
    ) == pytest.approx(90.0)


def test_scientific_regression_radii_cover_selenium_and_common_metals() -> None:
    """Geometry/clash fallbacks must not treat MSE or common metals as carbon."""

    assert van_der_waals_radius_angstrom("SE") == pytest.approx(1.90)
    assert covalent_radius_angstrom("SE") == pytest.approx(1.20)
    assert van_der_waals_radius_angstrom("FE") == pytest.approx(2.05)
    assert covalent_radius_angstrom("ZN") == pytest.approx(1.22)


def test_scientific_regression_hbond_suppression_requires_sane_angle() -> None:
    """Acute donor-H...acceptor contacts must remain steric clashes."""

    report = detect_clashes(
        _hydrogen_bond_candidate_structure(
            donor_position=Vec3(1.8, 1.2, 0.0),
            hydrogen_position=Vec3(1.8, 0.0, 0.0),
            acceptor_position=Vec3(0.0, 0.0, 0.0),
        ),
        component_library=build_standard_component_library(),
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=2.0),
    )

    assert len(report.clashes) == 1
    assert {report.clashes[0].left_atom_name, report.clashes[0].right_atom_name} == {
        "H",
        "O",
    }


def test_scientific_regression_degenerate_placement_raises_typed_error() -> None:
    """Undefined placement frames should not leak raw math exceptions."""

    with pytest.raises(GeometryPlacementError, match="distinct B/C anchors"):
        InternalCoordinateFrame(
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        ).place(
            bond_length=1.25,
            bond_angle_degrees=122.5,
            dihedral_degrees=180.0,
        )


def _ala_payload(
    *,
    seq_num: int,
    n: Vec3,
    ca: Vec3,
    c: Vec3,
    o: Vec3 | None = _DEFAULT_O_POSITION,
    cb: Vec3 = _DEFAULT_CB_POSITION,
) -> CompletionResiduePayload:
    atoms = [
        atom_payload("N", "N", n),
        atom_payload("CA", "C", ca),
        atom_payload("C", "C", c),
        atom_payload("CB", "C", cb),
    ]
    if o is not None:
        atoms.append(atom_payload("O", "O", o))

    return build_completion_payload(
        component_id="ALA",
        residue_id=ResidueId(chain_id="A", seq_num=seq_num),
        atoms=tuple(atoms),
    )


def _structure_from_completion_payloads(
    residues: tuple[CompletionResiduePayload, ...],
) -> ProteinStructure:
    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    (
                        residue.residue_site,
                        residue.residue_geometry,
                        residue.formal_charge_by_atom_name,
                    )
                    for residue in residues
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="geometry-scientific-backbone-regression",
    )


def _completion_site(
    snapshot: ProteinStructureSnapshot,
    *,
    residue_index: ResidueIndex,
    original_payload: CompletionResiduePayload,
) -> ResidueCompletionSite:
    return ResidueCompletionSite(
        residue_index=residue_index,
        template=build_standard_component_library().require("ALA"),
        original_payload=original_payload,
        neighborhood=ResidueBackboneNeighborhood.from_linear_residue_slots(
            residue_index,
            residue_count=len(snapshot.structure.constitution.residue_slots),
        ),
    )


def _oxygen_position(
    residue: CompletionResiduePayload,
    next_nitrogen: Vec3,
    psi_degrees: float,
) -> Vec3:
    return PeptideCarbonylFrame(
        nitrogen=residue.position("N"),
        alpha_carbon=residue.position("CA"),
        carbonyl_carbon=residue.position("C"),
    ).backbone_oxygen(
        psi_degrees=psi_degrees,
        clash_reference=next_nitrogen,
    )


def _analysis_structure(*, seq_nums: tuple[int, int, int]) -> ProteinStructure:
    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=seq_nums[0]),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.1, 0.1, 0.0)),
                            atom_payload("C", "C", Vec3(1.8, 1.2, 0.4)),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=seq_nums[1]),
                        atoms=(
                            atom_payload("N", "N", Vec3(2.9, 1.0, 1.2)),
                            atom_payload("CA", "C", Vec3(3.8, 1.8, 1.6)),
                            atom_payload("C", "C", Vec3(4.9, 1.2, 0.9)),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=seq_nums[2]),
                        atoms=(
                            atom_payload("N", "N", Vec3(5.9, 1.9, 1.3)),
                            atom_payload("CA", "C", Vec3(6.8, 1.5, 0.4)),
                            atom_payload("C", "C", Vec3(7.6, 2.5, -0.2)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="geometry-scientific-ramachandran-regression",
    )


def _hydrogen_bond_candidate_structure(
    *,
    donor_position: Vec3,
    hydrogen_position: Vec3,
    acceptor_position: Vec3,
) -> ProteinStructure:
    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("O", "O", acceptor_position),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="B", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", donor_position),
                            atom_payload("H", "H", hydrogen_position),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="geometry-scientific-hbond-angle-regression",
    )


def _vec_array(vector: Vec3) -> np.ndarray:
    return np.asarray(tuple(vector), dtype=np.float64)
