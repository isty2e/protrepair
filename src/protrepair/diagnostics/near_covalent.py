"""Derived diagnostics for severe nonbonded contacts that look near-covalent."""

from collections.abc import Iterable
from dataclasses import dataclass

from protrepair.chemistry import ElementRadiusLookup, RadiusKind, prepare_radius_lookup
from protrepair.diagnostics.clashes import StericClash
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId


@dataclass(frozen=True, slots=True)
class NearCovalentContactPolicy:
    """Policy for classifying clash pairs as near-covalent nonbonded contacts."""

    minimum_overlap_angstrom: float = 1.0
    covalent_distance_margin_angstrom: float = 0.45
    ignore_same_residue: bool = True

    def __post_init__(self) -> None:
        if self.minimum_overlap_angstrom <= 0.0:
            raise ValueError("minimum_overlap_angstrom must be positive")

        if self.covalent_distance_margin_angstrom < 0.0:
            raise ValueError(
                "covalent_distance_margin_angstrom must be non-negative"
            )


def detect_near_covalent_contacts(
    structure: ProteinStructure,
    *,
    clashes: Iterable[StericClash],
    policy: NearCovalentContactPolicy | None = None,
) -> tuple[StericClash, ...]:
    """Return severe nonbonded contacts short enough to look near-covalent."""

    active_policy = NearCovalentContactPolicy() if policy is None else policy
    clash_tuple = tuple(clashes)
    covalent_radius_lookup = prepare_radius_lookup(
        _clash_endpoint_elements(structure, clash_tuple),
        RadiusKind.COVALENT,
    )
    covalent_radius_lookup.require_complete("near-covalent contact detection")
    return tuple(
        clash
        for clash in clash_tuple
        if is_near_covalent_contact(
            structure,
            clash=clash,
            policy=active_policy,
            covalent_radius_lookup=covalent_radius_lookup,
        )
    )


def is_near_covalent_contact(
    structure: ProteinStructure,
    *,
    clash: StericClash,
    policy: NearCovalentContactPolicy,
    covalent_radius_lookup: ElementRadiusLookup | None = None,
) -> bool:
    """Return whether one clash pair is severe enough to resemble one extra bond."""

    if policy.ignore_same_residue and clash.left_residue_id == clash.right_residue_id:
        return False

    if clash.overlap_angstrom < policy.minimum_overlap_angstrom:
        return False

    left_element = _atom_element(
        structure,
        residue_id=clash.left_residue_id,
        atom_name=clash.left_atom_name,
    )
    right_element = _atom_element(
        structure,
        residue_id=clash.right_residue_id,
        atom_name=clash.right_atom_name,
    )
    active_radius_lookup = (
        prepare_radius_lookup((left_element, right_element), RadiusKind.COVALENT)
        if covalent_radius_lookup is None
        else covalent_radius_lookup
    )
    active_radius_lookup.require_complete("near-covalent contact")
    covalent_distance_cutoff = (
        active_radius_lookup.radius_angstrom(left_element)
        + active_radius_lookup.radius_angstrom(right_element)
        + policy.covalent_distance_margin_angstrom
    )
    return clash.distance_angstrom <= covalent_distance_cutoff


def _clash_endpoint_elements(
    structure: ProteinStructure,
    clashes: tuple[StericClash, ...],
) -> tuple[str, ...]:
    """Return endpoint elements used by a near-covalent clash batch."""

    elements: list[str] = []
    for clash in clashes:
        elements.append(
            _atom_element(
                structure,
                residue_id=clash.left_residue_id,
                atom_name=clash.left_atom_name,
            )
        )
        elements.append(
            _atom_element(
                structure,
                residue_id=clash.right_residue_id,
                atom_name=clash.right_atom_name,
            )
        )

    return tuple(elements)


def _atom_element(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    atom_name: str,
) -> str:
    """Return the element symbol for one canonical atom reference."""

    atom_index = structure.constitution.atom_index(
        AtomRef(residue_id=residue_id, atom_name=atom_name)
    )
    return structure.constitution.atom_site_at(atom_index).element
