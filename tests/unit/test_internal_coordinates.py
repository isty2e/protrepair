"""Direct regression tests for internal-coordinate torsion primitives."""

from math import acos, degrees

import numpy as np
import pytest

from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3


def test_torsion_returns_180_for_collinear_trans_outer_bonds() -> None:
    """Collinear trans-like outer bonds should preserve a finite trans torsion."""

    torsion = InternalCoordinateFrame.torsion(
        Vec3(-1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(2.0, 0.0, 0.0),
    )

    assert torsion == 180.0


def test_torsion_returns_0_for_collinear_cis_outer_bonds() -> None:
    """Collinear cis-like outer bonds should preserve a finite cis torsion."""

    torsion = InternalCoordinateFrame.torsion(
        Vec3(-1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.5, 0.0, 0.0),
    )

    assert torsion == 0.0


def test_place_returns_finite_coordinate_for_collinear_anchors() -> None:
    """Collinear placement anchors should still yield a finite constrained point."""

    point_c = Vec3(2.0, 0.0, 0.0)
    placed = InternalCoordinateFrame(
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        point_c,
    ).place(
        bond_length=1.25,
        bond_angle_degrees=122.5,
        dihedral_degrees=180.0,
    )

    placed_array = placed.to_array()
    point_c_array = point_c.to_array()
    point_b_array = Vec3(1.0, 0.0, 0.0).to_array()
    placed_bond = placed_array - point_c_array
    reference_bond = point_b_array - point_c_array
    cosine = float(np.dot(placed_bond, reference_bond)) / (
        float(np.linalg.norm(placed_bond)) * float(np.linalg.norm(reference_bond))
    )
    bond_angle_degrees = degrees(acos(min(1.0, max(-1.0, cosine))))

    assert np.isfinite(placed_array).all()
    assert InternalCoordinateFrame.distance(placed, point_c) == pytest.approx(1.25)
    assert bond_angle_degrees == pytest.approx(122.5)


def test_place_raises_structured_error_for_coincident_b_c_anchors() -> None:
    """Coincident B/C anchors make placement undefined, not merely inaccurate."""

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
