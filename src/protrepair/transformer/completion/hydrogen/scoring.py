"""Neutral rotatable-hydrogen search and scoring primitives."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import acos, degrees, pi, sqrt

import numpy as np
from numpy.typing import NDArray

from protrepair.chemistry import ElementRadiusLookup, RadiusKind, prepare_radius_lookup
from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3

CoordinateLike = Vec3 | Sequence[float] | NDArray[np.float64]
Vector = NDArray[np.float64]
ROTATABLE_HYDROGEN_DEGENERATE_NORM_EPSILON = 1e-12
ROTATABLE_HYDROGEN_STERIC_CUTOFF_SQ_ANGSTROM = 6.25
ROTATABLE_HYDROGEN_CLASH_PENALTY_SCALE = 100.0
ROTATABLE_HYDROGEN_HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM = 1.6
ROTATABLE_HYDROGEN_HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM = 2.4
ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS = 1
ROTATABLE_HYDROGEN_DONOR_ELEMENTS = frozenset({"N", "O", "S"})
ROTATABLE_HYDROGEN_ACCEPTOR_ELEMENTS = frozenset({"N", "O", "S"})
ROTATABLE_HYDROGEN_OVERLAP_TOLERANCE_ANGSTROM = 0.90


@dataclass(frozen=True, slots=True)
class RotatableHydrogenLocalSite:
    """Same-residue heavy-atom site used in rotatable-hydrogen scoring."""

    atom_name: str
    element: str
    x: float
    y: float
    z: float
    bond_hops_from_donor: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_name", self.atom_name.strip().upper())
        object.__setattr__(self, "element", self.element.strip().upper())


@dataclass(frozen=True, slots=True)
class RotatableHydrogenEnvironment:
    """Packed heavy-atom interaction data for one residue-number environment."""

    residue_number: str
    atom_x: tuple[float, ...]
    atom_y: tuple[float, ...]
    atom_z: tuple[float, ...]
    elements: tuple[str, ...]
    charges: tuple[float, ...]
    sigmas_nm: tuple[float, ...]
    epsilons_kj_mol: tuple[float, ...]
    local_sites: tuple[RotatableHydrogenLocalSite, ...] = ()
    van_der_waals_radius_lookup: ElementRadiusLookup = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        radius_lookup = prepare_radius_lookup(
            (
                "H",
                *self.elements,
                *(local_site.element for local_site in self.local_sites),
            ),
            RadiusKind.VAN_DER_WAALS,
        )
        if radius_lookup.has_unresolved_elements():
            raise radius_lookup.unresolved_radius_error(
                "rotatable hydrogen scoring environment"
            )

        object.__setattr__(self, "van_der_waals_radius_lookup", radius_lookup)

    def van_der_waals_radius(self, element: str) -> float:
        """Return a prepared vdW radius for one scoring element."""

        return self.van_der_waals_radius_lookup.radius_angstrom(element)


@dataclass(frozen=True, slots=True)
class RotatableHydrogenSearch:
    """Inputs required to score a rotatable hydrogen over a 360-degree scan."""

    outer_anchor: CoordinateLike
    inner_anchor: CoordinateLike
    donor: CoordinateLike
    hydrogen: CoordinateLike
    build_bond_length: float
    reproject_bond_length: float
    dihedral: float
    partial_charge: float
    sigma: float
    epsilon: float
    donor_element: str = "O"

    def __post_init__(self) -> None:
        object.__setattr__(self, "donor_element", self.donor_element.strip().upper())

    def candidate_positions(self) -> tuple[Vec3, ...]:
        """Return the six candidate coordinates from the hydroxyl torsion scan."""

        candidates: list[Vec3] = []
        current_dihedral = self.dihedral
        for increment in range(0, 360, 60):
            # The legacy rotatable-H scan uses the potential-energy helper's
            # parameter ordering: dihedral first, scanned torsion second.
            try:
                candidate = InternalCoordinateFrame(
                    self.outer_anchor,
                    self.inner_anchor,
                    self.donor,
                ).place(
                    bond_length=self.build_bond_length,
                    bond_angle_degrees=current_dihedral,
                    dihedral_degrees=109.5,
                )
                adjusted = recalculate_coordinate(
                    self.inner_anchor,
                    self.donor,
                    candidate,
                    self.reproject_bond_length,
                )
            except GeometryPlacementError:
                current_dihedral += increment
                continue

            candidates.append(adjusted)
            current_dihedral += increment

        return tuple(candidates)

    def potential_energy(
        self,
        hydrogen: CoordinateLike,
        environment: RotatableHydrogenEnvironment,
    ) -> float:
        """Return the nonbonded energy for one rotatable-hydrogen candidate."""

        return hydrogen_potential_energy(
            hydrogen,
            environment,
            self,
        )

    def steric_penalty(
        self,
        hydrogen: CoordinateLike,
        environment: RotatableHydrogenEnvironment,
    ) -> float:
        """Return clash-aware penalties for one rotatable-hydrogen candidate."""

        return hydrogen_steric_penalty(
            hydrogen,
            environment,
            self,
        )

    def candidate_score(
        self,
        hydrogen: CoordinateLike,
        environment: RotatableHydrogenEnvironment,
    ) -> float:
        """Return the total score for one rotatable-hydrogen candidate."""

        return self.potential_energy(
            hydrogen,
            environment,
        ) + self.steric_penalty(
            hydrogen,
            environment,
        )

    def optimized_coordinate(
        self,
        *,
        residue_number: str,
        environments: Sequence[RotatableHydrogenEnvironment],
    ) -> Vec3:
        """Return the lowest-energy hydrogen coordinate from a six-step scan."""

        candidate_hydrogens = self.candidate_positions()
        best_candidate: Vec3 | None = None
        best_score: float | None = None

        for environment in environments:
            if environment.residue_number != residue_number:
                continue

            for candidate in candidate_hydrogens:
                candidate_score = self.candidate_score(candidate, environment)
                if best_score is None or candidate_score < best_score:
                    best_score = candidate_score
                    best_candidate = candidate

        if best_candidate is None:
            return Vec3.coerce(self.hydrogen)

        return Vec3.coerce(best_candidate)


def hydrogen_potential_energy(
    hydrogen: CoordinateLike,
    environment: RotatableHydrogenEnvironment,
    search: RotatableHydrogenSearch,
) -> float:
    """Return the nonbonded energy between a candidate H and nearby heavy atoms."""

    total_energy = 0.0
    hydrogen_vector = Vec3.coerce(hydrogen).to_array()
    hydrogen_x = float(hydrogen_vector[0])
    hydrogen_y = float(hydrogen_vector[1])
    hydrogen_z = float(hydrogen_vector[2])
    hydrogen_charge = search.partial_charge
    hydrogen_sigma = search.sigma
    hydrogen_epsilon = search.epsilon
    electrostatic_constant = 138.94

    for atom_x, atom_y, atom_z, atom_charge, atom_sigma, atom_epsilon in zip(
        environment.atom_x,
        environment.atom_y,
        environment.atom_z,
        environment.charges,
        environment.sigmas_nm,
        environment.epsilons_kj_mol,
        strict=True,
    ):
        delta_x = hydrogen_x - atom_x
        delta_y = hydrogen_y - atom_y
        delta_z = hydrogen_z - atom_z
        separation_sq_angstrom = (
            (delta_x * delta_x)
            + (delta_y * delta_y)
            + (delta_z * delta_z)
        )
        if separation_sq_angstrom > 6.25:
            continue

        separation = sqrt(separation_sq_angstrom) / 10.0
        coulomb = (
            electrostatic_constant * (hydrogen_charge * atom_charge)
        ) / separation
        mixed_epsilon = 4 * sqrt(hydrogen_epsilon * atom_epsilon)
        mixed_sigma = ((hydrogen_sigma + atom_sigma) * 0.5) / separation
        mixed_sigma_sq = mixed_sigma * mixed_sigma
        mixed_sigma_six = mixed_sigma_sq * mixed_sigma_sq * mixed_sigma_sq
        lennard_jones = (mixed_sigma_six * mixed_sigma_six) - mixed_sigma_six
        total_energy += coulomb + (mixed_epsilon * lennard_jones)

    return total_energy


def hydrogen_steric_penalty(
    hydrogen: CoordinateLike,
    environment: RotatableHydrogenEnvironment,
    search: RotatableHydrogenSearch,
) -> float:
    """Return clash-aware penalties for one rotatable-hydrogen candidate."""

    penalty = 0.0
    hydrogen_vector = Vec3.coerce(hydrogen).to_array()
    hydrogen_x = float(hydrogen_vector[0])
    hydrogen_y = float(hydrogen_vector[1])
    hydrogen_z = float(hydrogen_vector[2])

    for local_site in environment.local_sites:
        if (
            local_site.bond_hops_from_donor is not None
            and local_site.bond_hops_from_donor
            <= ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS
        ):
            continue

        penalty += hydrogen_steric_penalty_against_site(
            hydrogen_x=hydrogen_x,
            hydrogen_y=hydrogen_y,
            hydrogen_z=hydrogen_z,
            site_x=local_site.x,
            site_y=local_site.y,
            site_z=local_site.z,
            site_element=local_site.element,
            hydrogen_vdw_radius=environment.van_der_waals_radius("H"),
            site_vdw_radius=environment.van_der_waals_radius(local_site.element),
            donor_element=search.donor_element,
            allow_hydrogen_bond=False,
        )

    for site_x, site_y, site_z, site_element in zip(
        environment.atom_x,
        environment.atom_y,
        environment.atom_z,
        environment.elements,
        strict=True,
    ):
        penalty += hydrogen_steric_penalty_against_site(
            hydrogen_x=hydrogen_x,
            hydrogen_y=hydrogen_y,
            hydrogen_z=hydrogen_z,
            site_x=site_x,
            site_y=site_y,
            site_z=site_z,
            site_element=site_element,
            hydrogen_vdw_radius=environment.van_der_waals_radius("H"),
            site_vdw_radius=environment.van_der_waals_radius(site_element),
            donor_element=search.donor_element,
            allow_hydrogen_bond=True,
        )

    return penalty


def hydrogen_steric_penalty_against_site(
    *,
    hydrogen_x: float,
    hydrogen_y: float,
    hydrogen_z: float,
    site_x: float,
    site_y: float,
    site_z: float,
    site_element: str,
    hydrogen_vdw_radius: float,
    site_vdw_radius: float,
    donor_element: str,
    allow_hydrogen_bond: bool,
) -> float:
    """Return the steric penalty for one H-heavy pair."""

    delta_x = hydrogen_x - site_x
    delta_y = hydrogen_y - site_y
    delta_z = hydrogen_z - site_z
    separation_sq_angstrom = (
        (delta_x * delta_x)
        + (delta_y * delta_y)
        + (delta_z * delta_z)
    )
    if separation_sq_angstrom > ROTATABLE_HYDROGEN_STERIC_CUTOFF_SQ_ANGSTROM:
        return 0.0

    separation = sqrt(separation_sq_angstrom)
    if allow_hydrogen_bond and probable_rotatable_hydrogen_bond(
        donor_element=donor_element,
        acceptor_element=site_element,
        separation_angstrom=separation,
    ):
        return 0.0

    allowed_distance = (
        hydrogen_vdw_radius
        + site_vdw_radius
        - ROTATABLE_HYDROGEN_OVERLAP_TOLERANCE_ANGSTROM
    )
    if separation >= allowed_distance:
        return 0.0

    overlap = allowed_distance - separation
    return ROTATABLE_HYDROGEN_CLASH_PENALTY_SCALE * overlap * overlap


def probable_rotatable_hydrogen_bond(
    *,
    donor_element: str,
    acceptor_element: str,
    separation_angstrom: float,
) -> bool:
    """Return whether one H-heavy contact is plausibly a hydrogen bond."""

    return (
        donor_element in ROTATABLE_HYDROGEN_DONOR_ELEMENTS
        and acceptor_element in ROTATABLE_HYDROGEN_ACCEPTOR_ELEMENTS
        and ROTATABLE_HYDROGEN_HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM
        <= separation_angstrom
        <= ROTATABLE_HYDROGEN_HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM
    )


def rotatable_hydrogen_vdw_radius_angstrom(element: str) -> float:
    """Return the van der Waals radius used by rotatable-H scoring."""

    return prepare_radius_lookup((element,), RadiusKind.VAN_DER_WAALS).radius_angstrom(
        element
    )


def recalculate_coordinate(
    atom_b: CoordinateLike,
    atom_c: CoordinateLike,
    atom_d: CoordinateLike,
    bond_length: float,
) -> Vec3:
    """Re-normalize a rotated hydrogen coordinate to the expected bond length."""

    point_b = Vec3.coerce(atom_b).to_array()
    point_c = Vec3.coerce(atom_c).to_array()
    point_d = Vec3.coerce(atom_d).to_array()
    theta = 107.0

    bond_cb = point_b - point_c
    bond_dc = point_d - point_c
    angle_bcd = _angle_between_vectors_radians(bond_cb, bond_dc)
    rotate = theta - degrees(angle_bcd)

    normal = np.asarray(np.cross(bond_cb, bond_dc), dtype=np.float64)
    unit_normal = _unit_vector(
        normal,
        error_message="rotatable-hydrogen reprojection requires a rotation plane",
    )
    rotated = (
        point_c
        + bond_dc * np.cos(rotate * pi / 180.0)
        + np.cross(unit_normal, bond_dc) * np.sin(rotate * pi / 180.0)
        + unit_normal
        * np.dot(unit_normal, bond_dc)
        * (1 - np.cos(rotate * pi / 180.0))
    )

    scaled = _scale_from_origin(point_c, rotated, bond_length)
    return Vec3.from_iterable(scaled)


def _angle_between_vectors_radians(left_vector: Vector, right_vector: Vector) -> float:
    """Return the finite angle between two non-zero vectors in radians."""

    left_norm = _vector_norm(left_vector)
    right_norm = _vector_norm(right_vector)
    if (
        left_norm <= ROTATABLE_HYDROGEN_DEGENERATE_NORM_EPSILON
        or right_norm <= ROTATABLE_HYDROGEN_DEGENERATE_NORM_EPSILON
    ):
        raise GeometryPlacementError(
            "rotatable-hydrogen reprojection requires non-zero vectors"
        )

    cosine = float(np.dot(left_vector, right_vector)) / (left_norm * right_norm)
    clamped = min(1.0, max(-1.0, cosine))
    return acos(clamped)


def _unit_vector(vector: Vector, *, error_message: str) -> Vector:
    """Return one unit vector or raise when the vector is degenerate."""

    norm = _vector_norm(vector)
    if norm <= ROTATABLE_HYDROGEN_DEGENERATE_NORM_EPSILON:
        raise GeometryPlacementError(error_message)

    return np.asarray(vector / norm, dtype=np.float64)


def _scale_from_origin(origin: Vector, candidate: Vector, bond_length: float) -> Vector:
    """Return a candidate scaled to the requested distance from the origin."""

    direction = candidate - origin
    direction_norm = _vector_norm(direction)
    if direction_norm <= ROTATABLE_HYDROGEN_DEGENERATE_NORM_EPSILON:
        raise GeometryPlacementError(
            "rotatable-hydrogen reprojection produced a degenerate bond vector"
        )

    return origin + (direction * (bond_length / direction_norm))


def _vector_norm(vector: Vector) -> float:
    """Return the Euclidean norm for one vector."""

    return float(np.linalg.norm(vector))


__all__ = [
    "CoordinateLike",
    "ROTATABLE_HYDROGEN_ACCEPTOR_ELEMENTS",
    "ROTATABLE_HYDROGEN_CLASH_PENALTY_SCALE",
    "ROTATABLE_HYDROGEN_DONOR_ELEMENTS",
    "ROTATABLE_HYDROGEN_HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM",
    "ROTATABLE_HYDROGEN_HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM",
    "ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS",
    "ROTATABLE_HYDROGEN_OVERLAP_TOLERANCE_ANGSTROM",
    "ROTATABLE_HYDROGEN_STERIC_CUTOFF_SQ_ANGSTROM",
    "RotatableHydrogenEnvironment",
    "RotatableHydrogenLocalSite",
    "RotatableHydrogenSearch",
    "Vector",
    "hydrogen_potential_energy",
    "hydrogen_steric_penalty",
    "hydrogen_steric_penalty_against_site",
    "probable_rotatable_hydrogen_bond",
    "recalculate_coordinate",
    "rotatable_hydrogen_vdw_radius_angstrom",
]
