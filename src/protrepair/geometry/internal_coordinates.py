"""Pure internal-coordinate geometry primitives."""

from dataclasses import dataclass
from math import acos, atan2, degrees, isfinite, pi, sqrt

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

TORSION_PLANE_NORM_SQUARED_EPSILON = 1e-12


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
        unit_axis_bc = normalized_vector_or_none(axis_bc)
        if unit_axis_bc is None:
            raise InternalCoordinatePlacementError(
                "internal-coordinate placement requires distinct B/C anchors"
            )
        if not np.isfinite(axis_ba).all():
            raise InternalCoordinatePlacementError(
                "internal-coordinate placement requires a finite anchor basis"
            )

        axis_bc_norm = vector_norm(axis_bc)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = (
                axis_ba
                - (np.dot(axis_ba, axis_bc) / (axis_bc_norm * axis_bc_norm)) * axis_bc
            )
        if not np.isfinite(projected).all():
            raise InternalCoordinatePlacementError(
                "internal-coordinate placement requires a finite anchor basis"
            )
        projected_norm = vector_norm(projected)
        if projected_norm <= PLACEMENT_VECTOR_NORM_EPSILON:
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

    def bond_closure_candidates(
        self,
        *,
        bond_length: float,
        bond_angle_degrees: float,
        preferred_dihedral_degrees: float,
        closing_anchor: CoordinateLike,
        closing_bond_length: float,
    ) -> tuple[Vec3, ...]:
        """Close a second bond without moving the materialized anchors.

        Parameters
        ----------
        bond_length : float
            Distance from anchor C to the target, in angstroms.
        bond_angle_degrees : float
            Preferred B-C-target angle in degrees.
        preferred_dihedral_degrees : float
            Torsion used to order equivalent closure solutions.
        closing_anchor : CoordinateLike
            Materialized atom forming the second target bond.
        closing_bond_length : float
            Second bond's dictionary distance, in angstroms.

        Returns
        -------
        tuple[Vec3, ...]
            Up to two fixed-angle solutions, ordered by preferred torsion. An
            infeasible or non-discrete circle intersection instead returns one
            nearest two-sphere solution.

        Raises
        ------
        InternalCoordinatePlacementError
            The circle/sphere intersection is infeasible or undefined.
        """

        if (
            not isfinite(bond_length)
            or bond_length <= 0.0
            or not isfinite(bond_angle_degrees)
            or not 0.0 < bond_angle_degrees <= 180.0
            or not isfinite(closing_bond_length)
            or closing_bond_length <= 0.0
            or not isfinite(preferred_dihedral_degrees)
        ):
            raise InternalCoordinatePlacementError(
                "bond closure requires finite positive lengths and a valid angle"
            )

        circle_points = tuple(
            self.place(
                bond_length=bond_length,
                bond_angle_degrees=bond_angle_degrees,
                dihedral_degrees=angle,
            ).to_array()
            for angle in (0.0, 90.0, 180.0)
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            center = 0.5 * circle_points[0] + 0.5 * circle_points[2]
            cosine_basis = circle_points[0] - center
            sine_basis = circle_points[1] - center
            offset = center - Vec3.coerce(closing_anchor).to_array()
            cosine_coefficient = 2.0 * float(np.dot(offset, cosine_basis))
            sine_coefficient = 2.0 * float(np.dot(offset, sine_basis))
            amplitude = float(np.hypot(cosine_coefficient, sine_coefficient))
            constant = (
                closing_bond_length * closing_bond_length
                - float(np.dot(offset, offset))
                - float(np.dot(cosine_basis, cosine_basis))
            )
        if not isfinite(amplitude) or not isfinite(constant):
            raise InternalCoordinatePlacementError(
                "bond closure requires a finite non-degenerate torsion circle"
            )

        ratio = constant / amplitude if amplitude > 0.0 else float("inf")
        if abs(ratio) > 1.0 + 16.0 * np.finfo(np.float64).eps:
            return (
                self._place_on_bond_intersection(
                    bond_length=bond_length,
                    bond_angle_degrees=bond_angle_degrees,
                    preferred_dihedral_degrees=preferred_dihedral_degrees,
                    closing_anchor=closing_anchor,
                    closing_bond_length=closing_bond_length,
                ),
            )

        phase = atan2(sine_coefficient, cosine_coefficient)
        separation = acos(max(-1.0, min(1.0, ratio)))
        solutions = (degrees(phase + separation), degrees(phase - separation))
        ordered = sorted(
            solutions,
            key=lambda angle: abs(
                (angle - preferred_dihedral_degrees + 180.0) % 360.0 - 180.0
            ),
        )
        return tuple(
            self.place(
                bond_length=bond_length,
                bond_angle_degrees=bond_angle_degrees,
                dihedral_degrees=angle,
            )
            for angle in ordered
        )

    def _place_on_bond_intersection(
        self,
        *,
        bond_length: float,
        bond_angle_degrees: float,
        preferred_dihedral_degrees: float,
        closing_anchor: CoordinateLike,
        closing_bond_length: float,
    ) -> Vec3:
        parent = Vec3.coerce(self.atom_c).to_array()
        closing = Vec3.coerce(closing_anchor).to_array()
        axis = closing - parent
        unit_axis = normalized_vector_or_none(axis)
        if unit_axis is None:
            raise InternalCoordinatePlacementError(
                "bond closure requires distinct finite bond anchors"
            )

        distance = vector_norm(axis)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            axial_distance = (
                bond_length * bond_length
                - closing_bond_length * closing_bond_length
                + distance * distance
            ) / (2.0 * distance)
            radius_squared = bond_length * bond_length - axial_distance * axial_distance
        roundoff = 16.0 * np.finfo(np.float64).eps * bond_length * bond_length
        if not isfinite(radius_squared) or radius_squared < -roundoff:
            raise InternalCoordinatePlacementError(
                "bond closure is incompatible with the materialized anchors"
            )

        center = parent + axial_distance * unit_axis
        preferred = self.place(
            bond_length=bond_length,
            bond_angle_degrees=bond_angle_degrees,
            dihedral_degrees=preferred_dihedral_degrees,
        ).to_array()
        radial = preferred - center
        radial -= np.dot(radial, unit_axis) * unit_axis
        unit_radial = normalized_vector_or_none(radial)
        if unit_radial is None:
            unit_radial = deterministic_unit_orthogonal_or_none(unit_axis)
        if unit_radial is None:
            raise InternalCoordinatePlacementError(
                "bond closure requires a finite radial basis"
            )

        return Vec3.from_iterable(center + sqrt(max(0.0, radius_squared)) * unit_radial)

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
        if not all(
            np.isfinite(point).all() for point in (point_1, point_2, point_3, point_4)
        ):
            raise InternalCoordinatePlacementError(
                "internal-coordinate torsion requires finite coordinates"
            )

        bond_12 = point_1 - point_2
        bond_32 = point_3 - point_2
        bond_43 = point_4 - point_3

        plane_13 = np.cross(bond_12, bond_32)
        plane_24 = np.cross(bond_43, bond_32)
        plane_13_norm_sq = float(np.dot(plane_13, plane_13))
        plane_24_norm_sq = float(np.dot(plane_24, plane_24))

        if (
            plane_13_norm_sq <= TORSION_PLANE_NORM_SQUARED_EPSILON
            or plane_24_norm_sq <= TORSION_PLANE_NORM_SQUARED_EPSILON
        ):
            return InternalCoordinateFrame._degenerate_torsion_degrees(
                bond_12=bond_12,
                bond_43=bond_43,
            )

        cosine = np.dot(plane_13, plane_24) / sqrt(plane_13_norm_sq * plane_24_norm_sq)
        clamped = min(1.0, max(-1.0, float(cosine)))
        angle = acos(clamped)

        if np.dot(plane_13, np.cross(plane_24, bond_32)) < 0:
            angle = -angle

        return degrees(angle)

    @staticmethod
    def _degenerate_torsion_degrees(
        *,
        bond_12: PlacementVector,
        bond_43: PlacementVector,
    ) -> float:
        """Return a finite fallback torsion for collinear outer-bond geometry."""

        bond_12_norm = vector_norm(bond_12)
        bond_43_norm = vector_norm(bond_43)
        if (
            bond_12_norm <= PLACEMENT_VECTOR_NORM_EPSILON
            or bond_43_norm <= PLACEMENT_VECTOR_NORM_EPSILON
        ):
            return 0.0

        outer_alignment = float(np.dot(bond_12, bond_43)) / (
            bond_12_norm * bond_43_norm
        )
        return 180.0 if outer_alignment < 0.0 else 0.0

    @staticmethod
    def _unit_orthogonal_vector(axis: PlacementVector) -> PlacementVector:
        """Return a deterministic unit vector orthogonal to one unit axis."""

        orthogonal = deterministic_unit_orthogonal_or_none(axis)
        if orthogonal is None:
            raise InternalCoordinatePlacementError(
                "internal-coordinate orthogonal basis is undefined"
            )

        return orthogonal

    @staticmethod
    def distance(coord_1: CoordinateLike, coord_2: CoordinateLike) -> float:
        """Return the Euclidean distance between two coordinates."""

        point_1 = Vec3.coerce(coord_1).to_array()
        point_2 = Vec3.coerce(coord_2).to_array()
        delta_x = float(point_1[0]) - float(point_2[0])
        delta_y = float(point_1[1]) - float(point_2[1])
        delta_z = float(point_1[2]) - float(point_2[2])
        return sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z))


def _angle_between_vectors_degrees(
    left_vector: PlacementVector,
    right_vector: PlacementVector,
) -> float:
    """Return the finite angle between two non-zero vectors."""

    angle_radians = angle_between_vectors_radians_or_none(
        left_vector,
        right_vector,
    )
    if angle_radians is None:
        raise InternalCoordinatePlacementError(
            "internal-coordinate angle requires non-zero vectors"
        )

    return degrees(angle_radians)


def _unit_vector(
    vector: PlacementVector,
    *,
    error_message: str,
) -> PlacementVector:
    """Return one unit vector or raise when the vector is degenerate."""

    normalized = normalized_vector_or_none(vector)
    if normalized is None:
        raise InternalCoordinatePlacementError(error_message)

    return normalized


def _scale_from_origin(
    origin: PlacementVector,
    candidate: PlacementVector,
    bond_length: float,
) -> PlacementVector:
    """Return a candidate scaled to the requested distance from the origin."""

    scaled = scaled_point_from_origin_or_none(origin, candidate, bond_length)
    if scaled is None:
        raise InternalCoordinatePlacementError(
            "internal-coordinate placement produced a degenerate bond vector"
        )

    return scaled
