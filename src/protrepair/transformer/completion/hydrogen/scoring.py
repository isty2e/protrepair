"""Neutral rotatable-hydrogen search and scoring primitives."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import inf, isfinite, isnan, sqrt

import numpy as np
from numpy.typing import NDArray

from protrepair.chemistry import ElementRadiusLookup, RadiusKind, prepare_radius_lookup
from protrepair.diagnostics.clash_topology_rules import (
    probable_hydrogen_bond_geometry,
)
from protrepair.diagnostics.clashes import ClashPolicy
from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3
from protrepair.structure.labels import ResidueId

CoordinateLike = Vec3 | Sequence[float] | NDArray[np.float64]
# PRAS Server 1.2.1 limits its AMBER nonbonded scan to 2.5 angstrom.
_ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_HORIZON_ANGSTROM = 2.5
_ROTATABLE_HYDROGEN_MIN_SEPARATION_SQ_ANGSTROM = 1e-12
_COULOMB_CONSTANT_KJ_MOL_NM_E2 = 138.94
ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_CUTOFF_SQ_ANGSTROM = (
    _ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_HORIZON_ANGSTROM**2
)
ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS = 1
_ROTATABLE_HYDROGEN_CLASH_POLICY = ClashPolicy()


@dataclass(frozen=True, order=True, slots=True)
class RotatableHydrogenClashBurden:
    """Lexicographic steric burden for one candidate hydrogen position."""

    clash_count: int
    total_overlap_angstrom: float
    worst_overlap_angstrom: float

    def __post_init__(self) -> None:
        if isinstance(self.clash_count, bool) or not isinstance(self.clash_count, int):
            raise TypeError("rotatable-hydrogen clash count must be an integer")
        if self.clash_count < 0:
            raise ValueError("rotatable-hydrogen clash count must be non-negative")
        if (
            not isfinite(self.total_overlap_angstrom)
            or self.total_overlap_angstrom < 0.0
        ):
            raise ValueError(
                "rotatable-hydrogen total overlap must be finite and non-negative"
            )
        if (
            not isfinite(self.worst_overlap_angstrom)
            or self.worst_overlap_angstrom < 0.0
        ):
            raise ValueError(
                "rotatable-hydrogen worst overlap must be finite and non-negative"
            )
        if self.clash_count == 0 and (
            self.total_overlap_angstrom != 0.0
            or self.worst_overlap_angstrom != 0.0
        ):
            raise ValueError("zero clash count requires zero overlap burden")
        if self.clash_count > 0 and (
            self.total_overlap_angstrom <= 0.0
            or self.worst_overlap_angstrom <= 0.0
        ):
            raise ValueError("positive clash count requires positive overlap burden")
        if self.worst_overlap_angstrom > self.total_overlap_angstrom:
            raise ValueError("worst overlap cannot exceed total overlap")

    @classmethod
    def from_positive_overlaps(
        cls,
        overlaps_angstrom: Iterable[float],
    ) -> "RotatableHydrogenClashBurden":
        """Aggregate one iterable of finite positive clash overlaps."""

        clash_count = 0
        total_overlap_angstrom = 0.0
        worst_overlap_angstrom = 0.0
        for overlap in overlaps_angstrom:
            if not isfinite(overlap) or overlap <= 0.0:
                raise ValueError(
                    "rotatable-hydrogen clash overlaps must be finite and positive"
                )
            clash_count += 1
            total_overlap_angstrom += overlap
            worst_overlap_angstrom = max(worst_overlap_angstrom, overlap)
        return cls(
            clash_count=clash_count,
            total_overlap_angstrom=total_overlap_angstrom,
            worst_overlap_angstrom=worst_overlap_angstrom,
        )


@dataclass(frozen=True, order=True, slots=True)
class RotatableHydrogenCandidateRank:
    """Complete deterministic rank for one rotatable-hydrogen candidate."""

    clash_burden: RotatableHydrogenClashBurden
    potential_energy_kj_mol: float
    scan_order: int

    def __post_init__(self) -> None:
        if isnan(self.potential_energy_kj_mol) or self.potential_energy_kj_mol == -inf:
            raise ValueError(
                "candidate potential energy must be finite or positive infinity"
            )
        if isinstance(self.scan_order, bool) or not isinstance(self.scan_order, int):
            raise TypeError("candidate scan order must be an integer")
        if self.scan_order < 0:
            raise ValueError("candidate scan order must be non-negative")


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
    """Packed heavy-atom interaction data for one canonical residue."""

    residue_id: ResidueId
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
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError("rotatable hydrogen environment requires a ResidueId")

        packed_field_lengths = {
            len(self.atom_x),
            len(self.atom_y),
            len(self.atom_z),
            len(self.elements),
            len(self.charges),
            len(self.sigmas_nm),
            len(self.epsilons_kj_mol),
        }
        if len(packed_field_lengths) != 1:
            raise ValueError(
                "rotatable hydrogen environment packed fields must have equal lengths"
            )

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
    bond_length: float
    bond_angle_degrees: float
    dihedral: float
    partial_charge: float
    sigma: float
    epsilon: float
    donor_element: str = "O"

    def __post_init__(self) -> None:
        if not isfinite(self.bond_length) or self.bond_length <= 0.0:
            raise ValueError("bond_length must be finite and positive")
        if (
            not isfinite(self.bond_angle_degrees)
            or self.bond_angle_degrees <= 0.0
            or self.bond_angle_degrees >= 180.0
        ):
            raise ValueError("bond_angle_degrees must be between 0 and 180 degrees")
        if not isfinite(self.dihedral):
            raise ValueError("dihedral must be finite")
        object.__setattr__(self, "donor_element", self.donor_element.strip().upper())

    def candidate_positions(self) -> tuple[Vec3, ...]:
        """Return the six candidate coordinates from the hydroxyl torsion scan."""

        frame = InternalCoordinateFrame(
            self.outer_anchor,
            self.inner_anchor,
            self.donor,
        )
        candidates: list[Vec3] = []
        for increment in range(0, 360, 60):
            try:
                candidate = frame.place(
                    bond_length=self.bond_length,
                    bond_angle_degrees=self.bond_angle_degrees,
                    dihedral_degrees=self.dihedral + increment,
                )
            except GeometryPlacementError:
                continue

            candidates.append(candidate)

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

    def clash_burden(
        self,
        hydrogen: CoordinateLike,
        environment: RotatableHydrogenEnvironment,
    ) -> RotatableHydrogenClashBurden:
        """Return the steric burden for one rotatable-hydrogen candidate."""

        return hydrogen_clash_burden(
            hydrogen,
            environment,
            self,
        )

    def candidate_rank(
        self,
        hydrogen: CoordinateLike,
        environment: RotatableHydrogenEnvironment,
        *,
        scan_order: int,
    ) -> RotatableHydrogenCandidateRank:
        """Return the deterministic lexicographic rank for one candidate."""

        potential_energy = self.potential_energy(
            hydrogen,
            environment,
        )
        if not isfinite(potential_energy):
            potential_energy = inf
        return RotatableHydrogenCandidateRank(
            clash_burden=self.clash_burden(hydrogen, environment),
            potential_energy_kj_mol=potential_energy,
            scan_order=scan_order,
        )

    def optimized_coordinate(
        self,
        environment: RotatableHydrogenEnvironment,
    ) -> Vec3:
        """Return the lowest-energy hydrogen coordinate from a six-step scan."""

        candidate_hydrogens = self.candidate_positions()
        best_candidate: Vec3 | None = None
        best_rank: RotatableHydrogenCandidateRank | None = None

        for scan_order, candidate in enumerate(candidate_hydrogens):
            candidate_rank = self.candidate_rank(
                candidate,
                environment,
                scan_order=scan_order,
            )
            if best_rank is None or candidate_rank < best_rank:
                best_rank = candidate_rank
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
    hydrogen_position = Vec3.coerce(hydrogen)
    hydrogen_x = hydrogen_position.x
    hydrogen_y = hydrogen_position.y
    hydrogen_z = hydrogen_position.z
    hydrogen_charge = search.partial_charge
    hydrogen_sigma = search.sigma
    hydrogen_epsilon = search.epsilon

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
            (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z)
        )
        if (
            separation_sq_angstrom
            > ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_CUTOFF_SQ_ANGSTROM
        ):
            continue
        if separation_sq_angstrom <= _ROTATABLE_HYDROGEN_MIN_SEPARATION_SQ_ANGSTROM:
            return float("inf")

        separation = sqrt(separation_sq_angstrom) / 10.0
        coulomb = (
            _COULOMB_CONSTANT_KJ_MOL_NM_E2 * (hydrogen_charge * atom_charge)
        ) / separation
        mixed_epsilon = 4 * sqrt(hydrogen_epsilon * atom_epsilon)
        mixed_sigma = ((hydrogen_sigma + atom_sigma) * 0.5) / separation
        mixed_sigma_sq = mixed_sigma * mixed_sigma
        mixed_sigma_six = mixed_sigma_sq * mixed_sigma_sq * mixed_sigma_sq
        lennard_jones = (mixed_sigma_six * mixed_sigma_six) - mixed_sigma_six
        total_energy += coulomb + (mixed_epsilon * lennard_jones)

    return total_energy


def hydrogen_clash_burden(
    hydrogen: CoordinateLike,
    environment: RotatableHydrogenEnvironment,
    search: RotatableHydrogenSearch,
) -> RotatableHydrogenClashBurden:
    """Return the physical vdW-overlap burden for one candidate hydrogen."""

    overlaps: list[float] = []
    hydrogen_position = Vec3.coerce(hydrogen)
    hydrogen_x = hydrogen_position.x
    hydrogen_y = hydrogen_position.y
    hydrogen_z = hydrogen_position.z
    donor_vector = Vec3.coerce(search.donor)
    hydrogen_vdw_radius = environment.van_der_waals_radius("H")

    for local_site in environment.local_sites:
        if (
            local_site.bond_hops_from_donor is not None
            and local_site.bond_hops_from_donor
            <= ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS
        ):
            continue

        overlap = hydrogen_steric_overlap_against_site(
            hydrogen_x=hydrogen_x,
            hydrogen_y=hydrogen_y,
            hydrogen_z=hydrogen_z,
            site_x=local_site.x,
            site_y=local_site.y,
            site_z=local_site.z,
            site_element=local_site.element,
            hydrogen_vdw_radius=hydrogen_vdw_radius,
            site_vdw_radius=environment.van_der_waals_radius(local_site.element),
            donor_x=donor_vector.x,
            donor_y=donor_vector.y,
            donor_z=donor_vector.z,
            donor_element=search.donor_element,
            allow_hydrogen_bond=False,
        )
        if overlap > 0.0:
            overlaps.append(overlap)

    for site_x, site_y, site_z, site_element in zip(
        environment.atom_x,
        environment.atom_y,
        environment.atom_z,
        environment.elements,
        strict=True,
    ):
        overlap = hydrogen_steric_overlap_against_site(
            hydrogen_x=hydrogen_x,
            hydrogen_y=hydrogen_y,
            hydrogen_z=hydrogen_z,
            site_x=site_x,
            site_y=site_y,
            site_z=site_z,
            site_element=site_element,
            hydrogen_vdw_radius=hydrogen_vdw_radius,
            site_vdw_radius=environment.van_der_waals_radius(site_element),
            donor_x=donor_vector.x,
            donor_y=donor_vector.y,
            donor_z=donor_vector.z,
            donor_element=search.donor_element,
            allow_hydrogen_bond=True,
        )
        if overlap > 0.0:
            overlaps.append(overlap)

    return RotatableHydrogenClashBurden.from_positive_overlaps(overlaps)


def hydrogen_steric_overlap_against_site(
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
    donor_x: float,
    donor_y: float,
    donor_z: float,
    donor_element: str,
    allow_hydrogen_bond: bool,
) -> float:
    """Return physical vdW overlap for one admitted H-heavy clash."""

    delta_x = hydrogen_x - site_x
    delta_y = hydrogen_y - site_y
    delta_z = hydrogen_z - site_z
    allowed_distance = rotatable_hydrogen_steric_cutoff_angstrom(
        hydrogen_vdw_radius=hydrogen_vdw_radius,
        site_vdw_radius=site_vdw_radius,
    )
    if allowed_distance <= 0.0:
        return 0.0

    separation_sq_angstrom = (
        (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z)
    )
    if separation_sq_angstrom >= allowed_distance * allowed_distance:
        return 0.0

    separation = sqrt(separation_sq_angstrom)
    if allow_hydrogen_bond and probable_hydrogen_bond_geometry(
        donor_element=donor_element,
        acceptor_element=site_element,
        hydrogen_acceptor_distance_angstrom=separation,
        donor_x=donor_x,
        donor_y=donor_y,
        donor_z=donor_z,
        hydrogen_x=hydrogen_x,
        hydrogen_y=hydrogen_y,
        hydrogen_z=hydrogen_z,
        acceptor_x=site_x,
        acceptor_y=site_y,
        acceptor_z=site_z,
    ):
        return 0.0

    if separation >= allowed_distance:
        return 0.0

    return hydrogen_vdw_radius + site_vdw_radius - separation


def rotatable_hydrogen_vdw_radius_angstrom(element: str) -> float:
    """Return the van der Waals radius used by rotatable-H scoring."""

    return prepare_radius_lookup((element,), RadiusKind.VAN_DER_WAALS).radius_angstrom(
        element
    )


def rotatable_hydrogen_steric_cutoff_angstrom(
    *,
    hydrogen_vdw_radius: float,
    site_vdw_radius: float,
) -> float:
    """Return the H-heavy distance below which steric scoring can be non-zero."""

    return _ROTATABLE_HYDROGEN_CLASH_POLICY.allowed_distance_angstrom(
        left_van_der_waals_radius_angstrom=hydrogen_vdw_radius,
        right_van_der_waals_radius_angstrom=site_vdw_radius,
        left_is_hydrogen=True,
        right_is_hydrogen=False,
    )


def max_rotatable_hydrogen_steric_cutoff_angstrom(
    site_elements: Iterable[str],
) -> float:
    """Return the maximum H-heavy steric scoring horizon for site elements."""

    site_element_tuple = tuple(site_elements)
    radius_lookup = prepare_radius_lookup(
        ("H", *site_element_tuple),
        RadiusKind.VAN_DER_WAALS,
    )
    radius_lookup.require_complete("rotatable hydrogen scoring radius")
    hydrogen_vdw_radius = radius_lookup.radius_angstrom("H")
    return max(
        (
            rotatable_hydrogen_steric_cutoff_angstrom(
                hydrogen_vdw_radius=hydrogen_vdw_radius,
                site_vdw_radius=radius_lookup.radius_angstrom(site_element),
            )
            for site_element in site_element_tuple
        ),
        default=0.0,
    )


def max_rotatable_hydrogen_interaction_horizon_angstrom(
    site_elements: Iterable[str],
) -> float:
    """Return the maximum candidate-H horizon across every active score term."""

    return max(
        _ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_HORIZON_ANGSTROM,
        max_rotatable_hydrogen_steric_cutoff_angstrom(site_elements),
    )


__all__ = [
    "CoordinateLike",
    "ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS",
    "ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_CUTOFF_SQ_ANGSTROM",
    "RotatableHydrogenCandidateRank",
    "RotatableHydrogenClashBurden",
    "RotatableHydrogenEnvironment",
    "RotatableHydrogenLocalSite",
    "RotatableHydrogenSearch",
    "hydrogen_clash_burden",
    "hydrogen_potential_energy",
    "hydrogen_steric_overlap_against_site",
    "max_rotatable_hydrogen_interaction_horizon_angstrom",
    "max_rotatable_hydrogen_steric_cutoff_angstrom",
    "rotatable_hydrogen_steric_cutoff_angstrom",
    "rotatable_hydrogen_vdw_radius_angstrom",
]
