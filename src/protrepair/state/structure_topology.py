"""Whole-structure topology-evidence resolution facts."""

from dataclasses import dataclass
from enum import Enum

from protrepair.diagnostics.topology import (
    AmbiguousDisulfideFinding,
    LikelyDisulfideBond,
    detect_disulfide_topology,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    TopologyBond,
    is_covalent_like_relationship,
)


class DisulfideTopologyConflictReason(str, Enum):
    """Reason geometric disulfide evidence cannot update canonical topology."""

    EXISTING_PAIR_RELATIONSHIP = "existing_pair_relationship"
    ENDPOINT_HAS_OTHER_COVALENT_PARTNER = "endpoint_has_other_covalent_partner"


@dataclass(frozen=True, slots=True)
class DisulfideTopologyConflict:
    """One likely disulfide blocked by contradictory canonical topology."""

    candidate: LikelyDisulfideBond
    reason: DisulfideTopologyConflictReason
    conflicting_atom_ref_pairs: tuple[tuple[AtomRef, AtomRef], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reason, DisulfideTopologyConflictReason):
            raise TypeError(
                "disulfide topology conflicts require a typed conflict reason"
            )
        pairs = tuple(dict.fromkeys(self.conflicting_atom_ref_pairs))
        if not pairs:
            raise ValueError(
                "disulfide topology conflicts require conflicting topology evidence"
            )
        if any(left_ref >= right_ref for left_ref, right_ref in pairs):
            raise ValueError(
                "conflicting disulfide atom-reference pairs must be canonical"
            )
        object.__setattr__(self, "conflicting_atom_ref_pairs", pairs)

    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Return all residues participating in the contradiction."""

        return tuple(
            dict.fromkeys(
                (
                    *self.candidate.residue_pair(),
                    *(
                        atom_ref.residue_id
                        for atom_ref_pair in self.conflicting_atom_ref_pairs
                        for atom_ref in atom_ref_pair
                    ),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class StructureDisulfideTopologyFacts:
    """Planner-readable disulfide evidence resolved against canonical topology."""

    carrier: ProteinStructure
    promotable_candidates: tuple[LikelyDisulfideBond, ...]
    conflicts: tuple[DisulfideTopologyConflict, ...]
    ambiguous_findings: tuple[AmbiguousDisulfideFinding, ...]

    def __post_init__(self) -> None:
        promotable_candidates = tuple(self.promotable_candidates)
        conflicts = tuple(self.conflicts)
        ambiguous_findings = tuple(self.ambiguous_findings)
        promotable_pairs = tuple(
            candidate.residue_pair() for candidate in promotable_candidates
        )
        conflict_pairs = tuple(
            conflict.candidate.residue_pair() for conflict in conflicts
        )
        if len(set(promotable_pairs)) != len(promotable_pairs):
            raise ValueError("promotable disulfide candidates must not repeat")
        promotable_residue_ids = tuple(
            residue_id
            for candidate in promotable_candidates
            for residue_id in candidate.residue_pair()
        )
        if len(set(promotable_residue_ids)) != len(promotable_residue_ids):
            raise ValueError(
                "promotable disulfide candidates must not share endpoints"
            )
        if len(set(conflict_pairs)) != len(conflict_pairs):
            raise ValueError("disulfide topology conflicts must not repeat")
        if set(promotable_pairs).intersection(conflict_pairs):
            raise ValueError(
                "disulfide candidates cannot be both promotable and conflicting"
            )
        object.__setattr__(self, "promotable_candidates", promotable_candidates)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "ambiguous_findings", ambiguous_findings)

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
    ) -> "StructureDisulfideTopologyFacts":
        """Resolve geometric disulfide evidence against canonical topology."""

        likely_candidates, ambiguous_findings = detect_disulfide_topology(structure)
        promotable_candidates: list[LikelyDisulfideBond] = []
        conflicts: list[DisulfideTopologyConflict] = []
        for candidate in likely_candidates:
            conflict = _candidate_topology_conflict(structure, candidate)
            if conflict is not None:
                conflicts.append(conflict)
                continue
            if _candidate_is_already_resolved(structure, candidate):
                continue
            promotable_candidates.append(candidate)

        return cls(
            carrier=structure,
            promotable_candidates=tuple(promotable_candidates),
            conflicts=tuple(conflicts),
            ambiguous_findings=ambiguous_findings,
        )

    def has_promotable_candidates(self) -> bool:
        """Return whether explicit topology resolution can make progress."""

        return bool(self.promotable_candidates)


def _candidate_topology_conflict(
    structure: ProteinStructure,
    candidate: LikelyDisulfideBond,
) -> DisulfideTopologyConflict | None:
    """Return canonical topology blocking one likely candidate, if any."""

    left_index, right_index = _candidate_atom_indices(structure, candidate)
    existing_pair_bond = structure.topology.bond_between(left_index, right_index)
    if existing_pair_bond is not None:
        if is_covalent_like_relationship(existing_pair_bond):
            return None
        return DisulfideTopologyConflict(
            candidate=candidate,
            reason=DisulfideTopologyConflictReason.EXISTING_PAIR_RELATIONSHIP,
            conflicting_atom_ref_pairs=(
                _bond_atom_ref_pair(structure, existing_pair_bond),
            ),
        )

    competing_bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if is_covalent_like_relationship(bond)
        and (bond.involves(left_index) or bond.involves(right_index))
        and _bond_is_inter_residue(structure, bond)
    )
    if not competing_bonds:
        return None

    return DisulfideTopologyConflict(
        candidate=candidate,
        reason=(
            DisulfideTopologyConflictReason.ENDPOINT_HAS_OTHER_COVALENT_PARTNER
        ),
        conflicting_atom_ref_pairs=tuple(
            _bond_atom_ref_pair(structure, bond) for bond in competing_bonds
        ),
    )


def _candidate_is_already_resolved(
    structure: ProteinStructure,
    candidate: LikelyDisulfideBond,
) -> bool:
    """Return whether canonical topology already resolves this candidate pair."""

    left_index, right_index = _candidate_atom_indices(structure, candidate)
    existing_bond = structure.topology.bond_between(left_index, right_index)
    return existing_bond is not None and is_covalent_like_relationship(existing_bond)


def _candidate_atom_indices(
    structure: ProteinStructure,
    candidate: LikelyDisulfideBond,
) -> tuple[AtomIndex, AtomIndex]:
    """Return canonical SG atom slots for one geometry candidate."""

    return (
        structure.constitution.atom_index(
            AtomRef(candidate.left_residue_id, "SG")
        ),
        structure.constitution.atom_index(
            AtomRef(candidate.right_residue_id, "SG")
        ),
    )


def _bond_atom_ref_pair(
    structure: ProteinStructure,
    bond: TopologyBond,
) -> tuple[AtomRef, AtomRef]:
    """Return one topology bond as a canonical atom-reference pair."""

    left_ref = structure.constitution.atom_ref_at(bond.atom_index_1)
    right_ref = structure.constitution.atom_ref_at(bond.atom_index_2)
    return (left_ref, right_ref) if left_ref < right_ref else (right_ref, left_ref)


def _bond_is_inter_residue(
    structure: ProteinStructure,
    bond: TopologyBond,
) -> bool:
    """Return whether one topology bond crosses residue identities."""

    left_ref, right_ref = _bond_atom_ref_pair(structure, bond)
    return left_ref.residue_id != right_ref.residue_id
