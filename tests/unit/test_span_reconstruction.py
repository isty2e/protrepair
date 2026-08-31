"""Donor-backed span reconstruction numerical contract tests."""

from typing import cast

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.span_reconstruction import (
    SpanClosureSettings,
    SpanReconstructionFailure,
    SpanReconstructionFailureKind,
    reconstruct_donor_span,
)

BackboneCoordinates = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def test_span_closure_settings_reject_non_numeric_tolerances() -> None:
    """Numerical bounds must reject bools and non-numeric values explicitly."""

    with pytest.raises(TypeError, match="must be numeric"):
        SpanClosureSettings(endpoint_rmsd_tolerance_angstrom=True)
    with pytest.raises(TypeError, match="must be numeric"):
        SpanClosureSettings(convergence_delta_angstrom=cast(float, "small"))


def test_span_reconstruction_rejects_mismatched_donor_mapping() -> None:
    """The numerical kernel must reject malformed direct-call mappings."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_backbone_residue("A", 1, _NORMAL_BACKBONE),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _backbone_residue("X", 1, _NORMAL_BACKBONE),
                    _backbone_residue("X", 2, _NORMAL_BACKBONE),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )

    with pytest.raises(ValueError, match="one donor residue"):
        reconstruct_donor_span(
            source_structure,
            scope=AbsentResidueSpanScope(
                preceding_residue_id=ResidueId("A", 1),
                absent_residue_ids=(ResidueId("A", 2), ResidueId("A", 3)),
            ),
            donor_structure=donor_structure,
            donor_residue_ids=(ResidueId("X", 2),),
            donor_preceding_residue_id=ResidueId("X", 1),
            donor_following_residue_id=None,
        )


@pytest.mark.filterwarnings("error")
def test_span_anchor_frame_rejects_finite_coordinate_overflow() -> None:
    """Extreme finite anchors should fail without leaking NumPy warnings."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_backbone_residue("A", 1, _NORMAL_BACKBONE),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _backbone_residue(
                        "X",
                        1,
                        (
                            (0.0, 0.0, 0.0),
                            (-1.0e308, 0.0, 0.0),
                            (1.0e308, 0.0, 0.0),
                            (1.0e308, 1.0, 0.0),
                        ),
                    ),
                    _backbone_residue(
                        "X",
                        2,
                        (
                            (1.0e307, 0.0, 0.0),
                            (1.0e307, 1.0, 0.0),
                            (1.0e307, 2.0, 0.0),
                            (1.0e307, 2.0, 1.0),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )

    outcome = reconstruct_donor_span(
        source_structure,
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId("A", 1),
            absent_residue_ids=(ResidueId("A", 2),),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 2),),
        donor_preceding_residue_id=ResidueId("X", 1),
        donor_following_residue_id=None,
    )

    assert isinstance(outcome, SpanReconstructionFailure)
    assert outcome.kind is SpanReconstructionFailureKind.DEGENERATE_ANCHOR_FRAME


@pytest.mark.filterwarnings("error")
def test_span_axis_rotation_classifies_finite_arithmetic_overflow() -> None:
    """Finite inputs that overflow CCD arithmetic should remain a typed failure."""

    base = 1.0e154
    step = 1.0e140
    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _backbone_residue("A", 1, _NORMAL_BACKBONE),
                    _backbone_residue(
                        "A",
                        3,
                        _square_backbone(0.9 * base, step=step),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _backbone_residue("X", 1, _NORMAL_BACKBONE),
                    _backbone_residue(
                        "X",
                        2,
                        _square_backbone(0.2 * base, step=step),
                    ),
                    _backbone_residue(
                        "X",
                        3,
                        _square_backbone(base, step=step),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )

    outcome = reconstruct_donor_span(
        source_structure,
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId("A", 1),
            absent_residue_ids=(ResidueId("A", 2),),
            following_residue_id=ResidueId("A", 3),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 2),),
        donor_preceding_residue_id=ResidueId("X", 1),
        donor_following_residue_id=ResidueId("X", 3),
    )

    assert isinstance(outcome, SpanReconstructionFailure)
    assert outcome.kind is SpanReconstructionFailureKind.NON_FINITE_COORDINATES


_NORMAL_BACKBONE: BackboneCoordinates = (
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
)


def _square_backbone(base: float, *, step: float) -> BackboneCoordinates:
    return (
        (base, base, 0.0),
        (base + step, base, 0.0),
        (base + step, base + step, 0.0),
        (base, base + step, 0.0),
    )


def _backbone_residue(
    chain_id: str,
    seq_num: int,
    coordinates: BackboneCoordinates,
) -> CanonicalResiduePayload:
    return residue_payload(
        component_id="ALA",
        residue_id=ResidueId(chain_id, seq_num),
        atoms=tuple(
            atom_payload(
                atom_name,
                "N" if atom_name == "N" else "O" if atom_name == "O" else "C",
                Vec3(*position),
            )
            for atom_name, position in zip(
                ("N", "CA", "C", "O"),
                coordinates,
                strict=True,
            )
        ),
    )
