"""Direct regression tests for polyhedral placement primitives."""

import numpy as np
import pytest

from protrepair.geometry import (
    GeometryPlacementError,
    PlanarCenter,
    TetrahedralCenter,
    Vec3,
)
from protrepair.geometry.placement_vector import PLACEMENT_VECTOR_NORM_EPSILON


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


@pytest.mark.filterwarnings("error")
def test_tetrahedral_pair_rejects_non_finite_anchor_without_runtime_warning() -> None:
    """Non-finite tetrahedral input should fail before midpoint arithmetic."""

    with pytest.raises(GeometryPlacementError, match="finite coordinates"):
        TetrahedralCenter(
            anchor_a=Vec3(float("inf"), 0.0, 0.0),
            anchor_b=Vec3(0.0, 0.0, 0.0),
            center=Vec3(1.0, 0.0, 0.0),
        ).pair()


@pytest.mark.filterwarnings("error")
def test_tetrahedral_pair_rejects_finite_coordinate_overflow() -> None:
    """Finite inputs whose intermediate arithmetic overflows should fail cleanly."""

    with pytest.raises(GeometryPlacementError, match="finite coordinates"):
        TetrahedralCenter(
            anchor_a=Vec3(-1.0e308, 0.0, 0.0),
            anchor_b=Vec3(-1.0e308, 0.0, 0.0),
            center=Vec3(1.0e308, 1.0, 0.0),
        ).pair()


def test_planar_projected_raises_structured_error_for_coincident_anchor() -> None:
    """Planar projection should reject undefined angles before producing NaNs."""

    with pytest.raises(GeometryPlacementError, match="non-zero vectors"):
        PlanarCenter(
            anchor_a=Vec3(0.0, 0.0, 0.0),
            center=Vec3(0.0, 0.0, 0.0),
            anchor_b=Vec3(1.0, 0.0, 0.0),
        ).projected(bond_length=1.01)


def test_planar_projected_rejects_axis_at_shared_degenerate_threshold() -> None:
    """Planar projection should use the closed shared norm threshold."""

    with pytest.raises(GeometryPlacementError, match="non-zero vectors"):
        PlanarCenter(
            anchor_a=Vec3(PLACEMENT_VECTOR_NORM_EPSILON, 0.0, 0.0),
            center=Vec3(0.0, 0.0, 0.0),
            anchor_b=Vec3(0.0, 1.0, 0.0),
        ).projected(bond_length=1.01)


def test_planar_projected_accepts_axis_above_shared_degenerate_threshold() -> None:
    """The next representable planar axis should remain projectable."""

    separation = float(np.nextafter(PLACEMENT_VECTOR_NORM_EPSILON, np.inf))
    placed = PlanarCenter(
        anchor_a=Vec3(separation, 0.0, 0.0),
        center=Vec3(0.0, 0.0, 0.0),
        anchor_b=Vec3(0.0, 1.0, 0.0),
    ).projected(bond_length=1.01)

    assert np.isfinite(placed.to_array()).all()
    assert placed.distance_to(Vec3(0.0, 0.0, 0.0)) == pytest.approx(1.01)


@pytest.mark.filterwarnings("error")
def test_planar_projected_rejects_non_finite_anchor() -> None:
    """Non-finite planar input should raise rather than emit NaN coordinates."""

    with pytest.raises(GeometryPlacementError, match="finite coordinates"):
        PlanarCenter(
            anchor_a=Vec3(float("inf"), 0.0, 0.0),
            center=Vec3(0.0, 0.0, 0.0),
            anchor_b=Vec3(1.0, 0.0, 0.0),
        ).projected(bond_length=1.01)
