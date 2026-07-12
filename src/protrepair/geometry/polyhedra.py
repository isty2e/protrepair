"""Pure polyhedral geometry primitives shared across placement families."""

from dataclasses import dataclass
from math import degrees, pi

import numpy as np

from protrepair.geometry.exceptions import GeometryPlacementError
from protrepair.geometry.placement_vector import (
    PLACEMENT_VECTOR_NORM_EPSILON,
    PlacementVector,
    angle_between_vectors_radians_or_none,
    deterministic_unit_orthogonal_or_none,
    normalized_vector_or_none,
    scaled_point_from_origin_or_none,
    vector_norm,
)
from protrepair.geometry.vector import CoordinateLike, Vec3


def _require_finite_points(*points: PlacementVector) -> None:
    """Reject non-finite polyhedral coordinates before vector arithmetic."""

    if not all(np.isfinite(point).all() for point in points):
        raise GeometryPlacementError(
            "polyhedral placement requires finite coordinates"
        )


def _scale_bond(
    origin: PlacementVector,
    candidate: PlacementVector,
    bond_length: float,
) -> Vec3:
    """Return one point scaled to the desired bond length from the origin."""

    scaled = scaled_point_from_origin_or_none(origin, candidate, bond_length)
    if scaled is None:
        raise GeometryPlacementError(
            "polyhedral placement produced a degenerate bond vector"
        )

    return Vec3.from_iterable(scaled)


def _unit_orthogonal_vector(axis: PlacementVector) -> PlacementVector:
    """Return a deterministic unit vector orthogonal to one axis."""

    orthogonal = deterministic_unit_orthogonal_or_none(axis)
    if orthogonal is None:
        raise GeometryPlacementError("polyhedral orthogonal basis is undefined")

    return orthogonal


def _tetrahedral_offset(
    *,
    point_a: PlacementVector,
    point_b: PlacementVector,
    point_center: PlacementVector,
    midpoint: PlacementVector,
) -> PlacementVector:
    """Return a finite offset direction for tetrahedral completion."""

    center_axis = point_center - midpoint
    center_axis_norm = vector_norm(center_axis)
    if center_axis_norm > PLACEMENT_VECTOR_NORM_EPSILON:
        offset = np.asarray(
            np.cross(point_a - midpoint, center_axis) / center_axis_norm,
            dtype=np.float64,
        )
        offset_norm = vector_norm(offset)
        if offset_norm > PLACEMENT_VECTOR_NORM_EPSILON:
            return offset

    anchor_axis = point_a - point_b
    anchor_axis_norm = vector_norm(anchor_axis)
    if anchor_axis_norm <= PLACEMENT_VECTOR_NORM_EPSILON:
        anchor_axis = point_a - point_center
        anchor_axis_norm = vector_norm(anchor_axis)

    if anchor_axis_norm <= PLACEMENT_VECTOR_NORM_EPSILON:
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)

    return _unit_orthogonal_vector(anchor_axis / anchor_axis_norm)


@dataclass(frozen=True, slots=True)
class TetrahedralCenter:
    """Three-atom tetrahedral frame around one placement center."""

    anchor_a: CoordinateLike
    anchor_b: CoordinateLike
    center: CoordinateLike

    def pair(self, *, bond_length: float = 1.09) -> tuple[Vec3, Vec3]:
        """Return the two remaining tetrahedral vertices."""

        point_a = Vec3.coerce(self.anchor_a).to_array()
        point_b = Vec3.coerce(self.anchor_b).to_array()
        point_center = Vec3.coerce(self.center).to_array()
        _require_finite_points(point_a, point_b, point_center)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            midpoint = (0.5 * point_a) + (0.5 * point_b)
            reflected = midpoint + 2 * (point_center - midpoint)
        _require_finite_points(midpoint, reflected)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            offset = _tetrahedral_offset(
                point_a=point_a,
                point_b=point_b,
                point_center=point_center,
                midpoint=midpoint,
            )
            candidate_1 = reflected - offset
            candidate_2 = reflected + offset
        _require_finite_points(midpoint, reflected, offset, candidate_1, candidate_2)
        return (
            _scale_bond(point_center, candidate_1, bond_length),
            _scale_bond(point_center, candidate_2, bond_length),
        )

    def remaining(self, *, bond_length: float = 1.09) -> Vec3:
        """Return the last tetrahedral vertex implied by the current frame."""

        point_a = Vec3.coerce(self.anchor_a).to_array()
        point_b = Vec3.coerce(self.anchor_b).to_array()
        point_center = Vec3.coerce(self.center).to_array()
        _require_finite_points(point_a, point_b, point_center)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            midpoint = (0.5 * point_a) + (0.5 * point_b)
            reflected = midpoint + 2 * (point_center - midpoint)
        _require_finite_points(midpoint, reflected)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate = reflected + _tetrahedral_offset(
                point_a=point_a,
                point_b=point_b,
                point_center=point_center,
                midpoint=midpoint,
            )
        _require_finite_points(midpoint, reflected, candidate)
        return _scale_bond(point_center, candidate, bond_length)


@dataclass(frozen=True, slots=True)
class PlanarCenter:
    """Three-atom planar frame around one placement center."""

    anchor_a: CoordinateLike
    center: CoordinateLike
    anchor_b: CoordinateLike

    def bisector(self, *, bond_length: float = 1.01) -> Vec3:
        """Return the planar bisector point used by amide-like donors."""

        point_a = Vec3.coerce(self.anchor_a).to_array()
        point_center = Vec3.coerce(self.center).to_array()
        point_b = Vec3.coerce(self.anchor_b).to_array()
        _require_finite_points(point_a, point_center, point_b)
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = (
                (point_a - point_b) + (point_center - point_b)
            ) + point_center
        _require_finite_points(candidate)
        return _scale_bond(point_center, candidate, bond_length)

    def projected(self, *, bond_length: float) -> Vec3:
        """Return the rotated planar point implied by this frame."""

        point_a = Vec3.coerce(self.anchor_a).to_array()
        point_center = Vec3.coerce(self.center).to_array()
        point_b = Vec3.coerce(self.anchor_b).to_array()
        _require_finite_points(point_a, point_center, point_b)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate = (
                (point_center - point_a) + (point_center - point_b)
            ) + point_center

            bond_b = point_b - point_center
            bond_candidate = candidate - point_center
            bond_a = point_a - point_center
            angle_b = _angle_between_vectors_radians(bond_b, bond_candidate)
            average_angle = (
                degrees(
                    angle_b
                    + _angle_between_vectors_radians(bond_a, bond_candidate)
                )
            ) / 2
            rotate = average_angle - degrees(angle_b)

            normal = np.asarray(
                np.cross(bond_b, bond_candidate),
                dtype=np.float64,
            )
            unit_normal = _unit_vector(
                normal,
                error_message="polyhedral placement requires a rotation plane",
            )
            rotated = (
                point_center
                + bond_candidate * np.cos(rotate * pi / 180.0)
                + np.cross(unit_normal, bond_candidate)
                * np.sin(rotate * pi / 180.0)
                + unit_normal
                * np.dot(unit_normal, bond_candidate)
                * (1 - np.cos(rotate * pi / 180.0))
            )
        _require_finite_points(candidate, rotated)
        return _scale_bond(point_center, rotated, bond_length)


def _angle_between_vectors_radians(
    left_vector: PlacementVector,
    right_vector: PlacementVector,
) -> float:
    """Return the finite angle between two non-zero vectors in radians."""

    angle = angle_between_vectors_radians_or_none(left_vector, right_vector)
    if angle is None:
        raise GeometryPlacementError(
            "polyhedral placement angle requires non-zero vectors"
        )

    return angle


def _unit_vector(
    vector: PlacementVector,
    *,
    error_message: str,
) -> PlacementVector:
    """Return one unit vector or raise when the vector is degenerate."""

    normalized = normalized_vector_or_none(vector)
    if normalized is None:
        raise GeometryPlacementError(error_message)

    return normalized
