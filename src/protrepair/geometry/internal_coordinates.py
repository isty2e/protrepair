"""Pure internal-coordinate geometry primitives."""

from dataclasses import dataclass
from math import acos, degrees, pi, sqrt

import numpy as np
import numpy.typing as npt

from protrepair.geometry.exceptions import GeometryPlacementError
from protrepair.geometry.vector import CoordinateLike, Vec3

Vector = npt.NDArray[np.float64]
TORSION_DEGENERATE_NORM_EPSILON = 1e-12
UNIT_X_AXIS = np.array((1.0, 0.0, 0.0), dtype=np.float64)
UNIT_Y_AXIS = np.array((0.0, 1.0, 0.0), dtype=np.float64)
UNIT_Z_AXIS = np.array((0.0, 0.0, 1.0), dtype=np.float64)


class InternalCoordinatePlacementError(GeometryPlacementError):
    """Raised when an internal-coordinate frame is geometrically undefined."""


@dataclass(frozen=True, slots=True)
class InternalCoordinateFrame:
    """Three-anchor frame for internal-coordinate placement."""

    atom_a: CoordinateLike
    atom_b: CoordinateLike
    atom_c: CoordinateLike

    def place(
        self,
        *,
        bond_length: float,
        bond_angle_degrees: float,
        dihedral_degrees: float,
    ) -> Vec3:
        """Return the fourth point implied by this internal-coordinate frame."""

        point_a = Vec3.coerce(self.atom_a).to_array()
        point_b = Vec3.coerce(self.atom_b).to_array()
        point_c = Vec3.coerce(self.atom_c).to_array()
        dihedral_radians = dihedral_degrees * pi / 180.0

        axis_bc = point_c - point_b
        axis_ba = point_a - point_b
        axis_bc_norm = _vector_norm(axis_bc)
        if axis_bc_norm <= TORSION_DEGENERATE_NORM_EPSILON:
            raise InternalCoordinatePlacementError(
                "internal-coordinate placement requires distinct B/C anchors"
            )

        unit_axis_bc = axis_bc / axis_bc_norm
        projected = axis_ba - (
            np.dot(axis_ba, axis_bc) / (axis_bc_norm * axis_bc_norm)
        ) * axis_bc
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm <= TORSION_DEGENERATE_NORM_EPSILON:
            unit_projected = InternalCoordinateFrame._unit_orthogonal_vector(
                unit_axis_bc
            )
        else:
            unit_projected = projected / projected_norm

        unit_perpendicular = _unit_vector(
            np.asarray(np.cross(unit_axis_bc, unit_projected), dtype=np.float64),
            error_message="internal-coordinate placement requires non-collinear basis",
        )

        temp_point = point_b + (
            unit_projected * np.cos(dihedral_radians)
            + unit_perpendicular * np.sin(dihedral_radians)
        )

        bond_cb = point_b - point_c
        bond_ct = temp_point - point_c
        angle_bct = _angle_between_vectors_degrees(bond_cb, bond_ct)
        rotate = bond_angle_degrees - angle_bct

        normal = np.asarray(np.cross(bond_cb, bond_ct), dtype=np.float64)
        unit_normal = _unit_vector(
            normal,
            error_message="internal-coordinate placement requires a rotation plane",
        )
        rotated = (
            point_c
            + bond_ct * np.cos(rotate * pi / 180.0)
            + np.cross(unit_normal, bond_ct) * np.sin(rotate * pi / 180.0)
            + unit_normal
            * np.dot(unit_normal, bond_ct)
            * (1 - np.cos(rotate * pi / 180.0))
        )

        scaled = _scale_from_origin(point_c, rotated, bond_length)
        return Vec3.from_iterable(scaled)

    @staticmethod
    def torsion(
        coord_1: CoordinateLike,
        coord_2: CoordinateLike,
        coord_3: CoordinateLike,
        coord_4: CoordinateLike,
    ) -> float:
        """Return the signed torsion angle defined by four points."""

        point_1 = Vec3.coerce(coord_1).to_array()
        point_2 = Vec3.coerce(coord_2).to_array()
        point_3 = Vec3.coerce(coord_3).to_array()
        point_4 = Vec3.coerce(coord_4).to_array()

        bond_12 = point_1 - point_2
        bond_32 = point_3 - point_2
        bond_43 = point_4 - point_3

        plane_13 = np.cross(bond_12, bond_32)
        plane_24 = np.cross(bond_43, bond_32)
        plane_13_norm_sq = float(np.dot(plane_13, plane_13))
        plane_24_norm_sq = float(np.dot(plane_24, plane_24))

        if (
            plane_13_norm_sq <= TORSION_DEGENERATE_NORM_EPSILON
            or plane_24_norm_sq <= TORSION_DEGENERATE_NORM_EPSILON
        ):
            return InternalCoordinateFrame._degenerate_torsion_degrees(
                bond_12=bond_12,
                bond_43=bond_43,
            )

        cosine = np.dot(plane_13, plane_24) / sqrt(
            plane_13_norm_sq * plane_24_norm_sq
        )
        clamped = min(1.0, max(-1.0, float(cosine)))
        angle = acos(clamped)

        if np.dot(plane_13, np.cross(plane_24, bond_32)) < 0:
            angle = -angle

        return degrees(angle)

    @staticmethod
    def _degenerate_torsion_degrees(*, bond_12: Vector, bond_43: Vector) -> float:
        """Return a finite fallback torsion for collinear outer-bond geometry."""

        bond_12_norm = float(np.linalg.norm(bond_12))
        bond_43_norm = float(np.linalg.norm(bond_43))
        if (
            bond_12_norm <= TORSION_DEGENERATE_NORM_EPSILON
            or bond_43_norm <= TORSION_DEGENERATE_NORM_EPSILON
        ):
            return 0.0

        outer_alignment = float(np.dot(bond_12, bond_43)) / (
            bond_12_norm * bond_43_norm
        )
        return 180.0 if outer_alignment < 0.0 else 0.0

    @staticmethod
    def _unit_orthogonal_vector(axis: Vector) -> Vector:
        """Return a deterministic unit vector orthogonal to one unit axis."""

        reference_axis = min(
            (UNIT_X_AXIS, UNIT_Y_AXIS, UNIT_Z_AXIS),
            key=lambda candidate: abs(float(np.dot(axis, candidate))),
        )
        orthogonal = np.asarray(np.cross(axis, reference_axis), dtype=np.float64)
        return _unit_vector(
            orthogonal,
            error_message="internal-coordinate orthogonal basis is undefined",
        )

    @staticmethod
    def distance(coord_1: CoordinateLike, coord_2: CoordinateLike) -> float:
        """Return the Euclidean distance between two coordinates."""

        point_1 = Vec3.coerce(coord_1).to_array()
        point_2 = Vec3.coerce(coord_2).to_array()
        delta_x = float(point_1[0]) - float(point_2[0])
        delta_y = float(point_1[1]) - float(point_2[1])
        delta_z = float(point_1[2]) - float(point_2[2])
        return sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z))


def _angle_between_vectors_degrees(left_vector: Vector, right_vector: Vector) -> float:
    """Return the finite angle between two non-zero vectors."""

    left_norm = _vector_norm(left_vector)
    right_norm = _vector_norm(right_vector)
    if (
        left_norm <= TORSION_DEGENERATE_NORM_EPSILON
        or right_norm <= TORSION_DEGENERATE_NORM_EPSILON
    ):
        raise InternalCoordinatePlacementError(
            "internal-coordinate angle requires non-zero vectors"
        )

    cosine = float(np.dot(left_vector, right_vector)) / (left_norm * right_norm)
    clamped = min(1.0, max(-1.0, cosine))
    return degrees(acos(clamped))


def _unit_vector(vector: Vector, *, error_message: str) -> Vector:
    """Return one unit vector or raise when the vector is degenerate."""

    norm = _vector_norm(vector)
    if norm <= TORSION_DEGENERATE_NORM_EPSILON:
        raise InternalCoordinatePlacementError(error_message)

    return np.asarray(vector / norm, dtype=np.float64)


def _scale_from_origin(origin: Vector, candidate: Vector, bond_length: float) -> Vector:
    """Return a candidate scaled to the requested distance from the origin."""

    direction = candidate - origin
    direction_norm = _vector_norm(direction)
    if direction_norm <= TORSION_DEGENERATE_NORM_EPSILON:
        raise InternalCoordinatePlacementError(
            "internal-coordinate placement produced a degenerate bond vector"
        )

    return origin + (direction * (bond_length / direction_norm))


def _vector_norm(vector: Vector) -> float:
    """Return the Euclidean norm for one vector."""

    return float(np.linalg.norm(vector))
