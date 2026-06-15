"""Pure polyhedral geometry primitives shared across placement families."""

from dataclasses import dataclass
from math import acos, degrees, pi

import numpy as np
import numpy.typing as npt

from protrepair.geometry.vector import CoordinateLike, Vec3

Vector = npt.NDArray[np.float64]
POLYHEDRA_DEGENERATE_NORM_EPSILON = 1e-12
UNIT_X_AXIS = np.array((1.0, 0.0, 0.0), dtype=np.float64)
UNIT_Y_AXIS = np.array((0.0, 1.0, 0.0), dtype=np.float64)
UNIT_Z_AXIS = np.array((0.0, 0.0, 1.0), dtype=np.float64)


def _scale_bond(origin: Vector, candidate: Vector, bond_length: float) -> Vec3:
    """Return one point scaled to the desired bond length from the origin."""

    scaled = (
        (candidate - origin) * (bond_length / np.linalg.norm(candidate - origin))
    ) + origin
    return Vec3.from_iterable(scaled)


def _unit_orthogonal_vector(axis: Vector) -> Vector:
    """Return a deterministic unit vector orthogonal to one axis."""

    reference_axis = min(
        (UNIT_X_AXIS, UNIT_Y_AXIS, UNIT_Z_AXIS),
        key=lambda candidate: abs(float(np.dot(axis, candidate))),
    )
    orthogonal = np.cross(axis, reference_axis)
    return np.asarray(orthogonal / np.linalg.norm(orthogonal), dtype=np.float64)


def _tetrahedral_offset(
    *,
    point_a: Vector,
    point_b: Vector,
    point_center: Vector,
    midpoint: Vector,
) -> Vector:
    """Return a finite offset direction for tetrahedral completion."""

    center_axis = point_center - midpoint
    center_axis_norm = float(np.linalg.norm(center_axis))
    if center_axis_norm > POLYHEDRA_DEGENERATE_NORM_EPSILON:
        offset = np.cross(point_a - midpoint, center_axis) / center_axis_norm
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm > POLYHEDRA_DEGENERATE_NORM_EPSILON:
            return np.asarray(offset, dtype=np.float64)

    anchor_axis = point_a - point_b
    anchor_axis_norm = float(np.linalg.norm(anchor_axis))
    if anchor_axis_norm <= POLYHEDRA_DEGENERATE_NORM_EPSILON:
        anchor_axis = point_a - point_center
        anchor_axis_norm = float(np.linalg.norm(anchor_axis))

    if anchor_axis_norm <= POLYHEDRA_DEGENERATE_NORM_EPSILON:
        return UNIT_Z_AXIS.copy()

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

        midpoint = 0.5 * (point_a + point_b)
        reflected = midpoint + 2 * (point_center - midpoint)
        offset = _tetrahedral_offset(
            point_a=point_a,
            point_b=point_b,
            point_center=point_center,
            midpoint=midpoint,
        )

        candidate_1 = reflected - offset
        candidate_2 = reflected + offset
        return (
            _scale_bond(point_center, candidate_1, bond_length),
            _scale_bond(point_center, candidate_2, bond_length),
        )

    def remaining(self, *, bond_length: float = 1.09) -> Vec3:
        """Return the last tetrahedral vertex implied by the current frame."""

        point_a = Vec3.coerce(self.anchor_a).to_array()
        point_b = Vec3.coerce(self.anchor_b).to_array()
        point_center = Vec3.coerce(self.center).to_array()

        midpoint = 0.5 * (point_a + point_b)
        reflected = midpoint + 2 * (point_center - midpoint)
        candidate = reflected + _tetrahedral_offset(
            point_a=point_a,
            point_b=point_b,
            point_center=point_center,
            midpoint=midpoint,
        )
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
        candidate = ((point_a - point_b) + (point_center - point_b)) + point_center
        return _scale_bond(point_center, candidate, bond_length)

    def projected(self, *, bond_length: float) -> Vec3:
        """Return the rotated planar point implied by this frame."""

        point_a = Vec3.coerce(self.anchor_a).to_array()
        point_center = Vec3.coerce(self.center).to_array()
        point_b = Vec3.coerce(self.anchor_b).to_array()
        candidate = ((point_center - point_a) + (point_center - point_b)) + point_center

        bond_b = point_b - point_center
        bond_candidate = candidate - point_center
        bond_a = point_a - point_center
        angle_b = acos(
            np.dot(bond_b, bond_candidate)
            / (np.linalg.norm(bond_b) * np.linalg.norm(bond_candidate))
        )
        average_angle = (
            degrees(
                acos(
                    np.dot(bond_b, bond_candidate)
                    / (np.linalg.norm(bond_b) * np.linalg.norm(bond_candidate))
                )
                + acos(
                    np.dot(bond_a, bond_candidate)
                    / (np.linalg.norm(bond_a) * np.linalg.norm(bond_candidate))
                )
            )
        ) / 2
        rotate = average_angle - degrees(angle_b)

        normal = np.cross(bond_b, bond_candidate)
        unit_normal = normal / np.linalg.norm(normal)
        rotated = (
            point_center
            + bond_candidate * np.cos(rotate * pi / 180.0)
            + np.cross(unit_normal, bond_candidate) * np.sin(rotate * pi / 180.0)
            + unit_normal
            * np.dot(unit_normal, bond_candidate)
            * (1 - np.cos(rotate * pi / 180.0))
        )
        return _scale_bond(point_center, rotated, bond_length)
