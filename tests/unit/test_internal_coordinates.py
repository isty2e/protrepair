"""Direct regression tests for internal-coordinate torsion primitives."""

from math import acos, degrees, sqrt
from typing import cast

import numpy as np
import pytest

from protrepair import ProtrepairError
from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3
from protrepair.geometry.internal_coordinates import (
    TORSION_PLANE_NORM_SQUARED_EPSILON,
)
from protrepair.geometry.placement_vector import PLACEMENT_VECTOR_NORM_EPSILON


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


@pytest.mark.filterwarnings("error")
def test_torsion_rejects_non_finite_coordinates_without_runtime_warning() -> None:
    """Non-finite torsion input must not become a fake cis/trans fallback."""

    with pytest.raises(GeometryPlacementError, match="finite coordinates"):
        InternalCoordinateFrame.torsion(
            Vec3(float("inf"), 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
        )


def test_torsion_uses_closed_squared_plane_threshold() -> None:
    """A plane norm exactly at the squared threshold should use the fallback."""

    plane_offset = sqrt(TORSION_PLANE_NORM_SQUARED_EPSILON)

    torsion = InternalCoordinateFrame.torsion(
        Vec3(0.0, plane_offset, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 1.0),
    )

    assert torsion == 0.0


def test_torsion_accepts_plane_above_squared_threshold() -> None:
    """The next representable squared plane norm should retain its torsion."""

    plane_norm_squared = float(
        np.nextafter(TORSION_PLANE_NORM_SQUARED_EPSILON, np.inf)
    )
    plane_offset = sqrt(plane_norm_squared)

    torsion = InternalCoordinateFrame.torsion(
        Vec3(0.0, plane_offset, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 1.0),
    )

    assert torsion == pytest.approx(90.0)


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


def test_place_rejects_b_c_axis_at_shared_degenerate_threshold() -> None:
    """Internal-coordinate placement should use the closed shared threshold."""

    with pytest.raises(GeometryPlacementError, match="distinct B/C anchors"):
        InternalCoordinateFrame(
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(PLACEMENT_VECTOR_NORM_EPSILON, 0.0, 0.0),
        ).place(
            bond_length=1.0,
            bond_angle_degrees=109.5,
            dihedral_degrees=60.0,
        )


def test_place_accepts_b_c_axis_above_shared_degenerate_threshold() -> None:
    """The next representable B/C separation should remain placeable."""

    separation = float(np.nextafter(PLACEMENT_VECTOR_NORM_EPSILON, np.inf))
    placed = InternalCoordinateFrame(
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(separation, 0.0, 0.0),
    ).place(
        bond_length=1.0,
        bond_angle_degrees=109.5,
        dihedral_degrees=60.0,
    )

    assert np.isfinite(placed.to_array()).all()
    assert placed.distance_to(Vec3(separation, 0.0, 0.0)) == pytest.approx(1.0)


@pytest.mark.filterwarnings("error")
def test_place_rejects_non_finite_b_c_axis_without_runtime_warning() -> None:
    """A non-finite placement axis should project into the domain error."""

    with pytest.raises(GeometryPlacementError, match="distinct B/C anchors"):
        InternalCoordinateFrame(
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(float("inf"), 0.0, 0.0),
        ).place(
            bond_length=1.25,
            bond_angle_degrees=122.5,
            dihedral_degrees=180.0,
        )


@pytest.mark.filterwarnings("error")
def test_place_rejects_non_finite_projected_anchor_basis() -> None:
    """A non-finite outer anchor must not become a fallback placement basis."""

    with pytest.raises(GeometryPlacementError, match="finite anchor basis"):
        InternalCoordinateFrame(
            Vec3(float("inf"), 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
        ).place(
            bond_length=1.25,
            bond_angle_degrees=122.5,
            dihedral_degrees=180.0,
        )


@pytest.mark.filterwarnings("error")
def test_place_rejects_finite_projection_overflow_without_runtime_warning() -> None:
    """Finite anchors whose projection overflows should fail cleanly."""

    with pytest.raises(GeometryPlacementError, match="finite anchor basis"):
        InternalCoordinateFrame(
            Vec3(1.0e154, 1.0e154, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(9.0e153, 9.0e153, 0.0),
        ).place(
            bond_length=1.25,
            bond_angle_degrees=122.5,
            dihedral_degrees=180.0,
        )


def test_geometry_placement_error_is_package_domain_error() -> None:
    """Undefined placement geometry should be catchable as a package error."""

    assert issubclass(GeometryPlacementError, ProtrepairError)
    assert issubclass(GeometryPlacementError, ValueError)

    with pytest.raises(ProtrepairError, match="distinct B/C anchors"):
        InternalCoordinateFrame(
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        ).place(
            bond_length=1.25,
            bond_angle_degrees=122.5,
            dihedral_degrees=180.0,
        )


def test_invalid_coordinate_payload_is_not_geometry_placement_error() -> None:
    """Malformed coordinate inputs should stay distinct from degenerate geometry."""

    with pytest.raises(TypeError):
        InternalCoordinateFrame(
            cast(Vec3, object()),
            Vec3(1.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
        ).place(
            bond_length=1.25,
            bond_angle_degrees=122.5,
            dihedral_degrees=180.0,
        )
