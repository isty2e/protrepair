"""Direct regression tests for polyhedral placement primitives."""

import numpy as np
import pytest

from protrepair.geometry import TetrahedralCenter, Vec3


def test_tetrahedral_pair_returns_finite_points_for_collinear_anchors() -> None:
    """Collinear tetrahedral anchors should still yield a finite symmetric pair."""

    center = Vec3(1.0, 0.0, 0.0)
    first, second = TetrahedralCenter(
        anchor_a=Vec3(2.0, 0.0, 0.0),
        anchor_b=Vec3(0.0, 0.0, 0.0),
        center=center,
    ).pair(bond_length=1.09)

    midpoint = 0.5 * (first.to_array() + second.to_array())

    assert np.isfinite(first.to_array()).all()
    assert np.isfinite(second.to_array()).all()
    assert first.distance_to(center) == pytest.approx(1.09)
    assert second.distance_to(center) == pytest.approx(1.09)
    assert midpoint == pytest.approx(center.to_array())


def test_tetrahedral_remaining_returns_finite_point_for_collinear_anchors() -> None:
    """Collinear tetrahedral anchors should still yield a finite remaining vertex."""

    center = Vec3(1.0, 0.0, 0.0)
    placed = TetrahedralCenter(
        anchor_a=Vec3(2.0, 0.0, 0.0),
        anchor_b=Vec3(0.0, 0.0, 0.0),
        center=center,
    ).remaining(bond_length=1.09)

    assert np.isfinite(placed.to_array()).all()
    assert placed.distance_to(center) == pytest.approx(1.09)
