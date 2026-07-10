"""Whole-structure topology-evidence resolution facts."""

from dataclasses import dataclass
from enum import Enum

from protrepair.diagnostics.topology import (
    AmbiguousDisulfideFinding,
    LikelyDisulfideBond,
    detect_disulfide_topology,
    detect_unassigned_disulfide_evidence,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.disulfide import disulfide_atom_ref_pairs
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    TopologyBond,
    is_covalent_like_relationship,
)

_CYSTEINE_THIOL_HYDROGEN_ATOM_NAMES = frozenset(("HG", "DG", "TG"))


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
class DisulfideEndpointMultiplicityContradiction:
    """One cysteine sulfur assigned to multiple canonical disulfide pairs."""

    sulfur_atom_ref: AtomRef
    disulfide_atom_ref_pairs: tuple[tuple[AtomRef, AtomRef], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sulfur_atom_ref, AtomRef):
            raise TypeError(
                "disulfide endpoint contradictions require an AtomRef endpoint"
            )
        if self.sulfur_atom_ref.atom_name != "SG":
            raise ValueError(
                "disulfide endpoint contradictions require an SG endpoint"
            )

        raw_pairs = tuple(self.disulfide_atom_ref_pairs)
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in raw_pairs):
            raise TypeError(
                "disulfide endpoint contradictions require two-AtomRef tuples"
            )
        if any(
            not all(isinstance(atom_ref, AtomRef) for atom_ref in pair)
            for pair in raw_pairs
        ):
            raise TypeError(
                "disulfide endpoint contradictions require AtomRef relationships"
            )

        pairs = tuple(sorted(dict.fromkeys(raw_pairs)))
        if len(pairs) < 2:
            raise ValueError(
                "disulfide endpoint contradictions require multiple relationships"
            )
        if any(
            pair[0] >= pair[1]
            for pair in pairs
        ):
            raise ValueError(
                "disulfide endpoint contradictions require canonical AtomRef pairs"
            )
        if any(self.sulfur_atom_ref not in pair for pair in pairs):
            raise ValueError(
                "every disulfide relationship must involve the contradictory endpoint"
            )
        if any(atom_ref.atom_name != "SG" for pair in pairs for atom_ref in pair):
            raise ValueError(
                "disulfide endpoint contradictions require SG-SG relationships"
            )

        object.__setattr__(self, "disulfide_atom_ref_pairs", pairs)

    @classmethod
    def all_from_structure(
        cls,
        structure: ProteinStructure,
    ) -> tuple["DisulfideEndpointMultiplicityContradiction", ...]:
        """Derive all endpoint-multiplicity contradictions from canonical topology."""

        pairs_by_endpoint: dict[AtomRef, list[tuple[AtomRef, AtomRef]]] = {}
        for atom_ref_pair in sorted(disulfide_atom_ref_pairs(structure)):
            for atom_ref in atom_ref_pair:
                pairs_by_endpoint.setdefault(atom_ref, []).append(atom_ref_pair)

        return tuple(
            cls(
                sulfur_atom_ref=sulfur_atom_ref,
                disulfide_atom_ref_pairs=tuple(atom_ref_pairs),
            )
            for sulfur_atom_ref, atom_ref_pairs in sorted(pairs_by_endpoint.items())
            if len(atom_ref_pairs) > 1
        )

    def partner_atom_refs(self) -> tuple[AtomRef, ...]:
        """Return the distinct SG partners attached to the shared endpoint."""

        return tuple(
            sorted(
                {
                    atom_ref
                    for pair in self.disulfide_atom_ref_pairs
                    for atom_ref in pair
                    if atom_ref != self.sulfur_atom_ref
                }
            )
        )

    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Return the shared endpoint residue followed by partner residues."""

        return (
            self.sulfur_atom_ref.residue_id,
            *(atom_ref.residue_id for atom_ref in self.partner_atom_refs()),
        )

    def projected_pair_count(
        self,
        residue_ids: frozenset[ResidueId],
    ) -> int:
        """Return how many contradictory pairs survive one residue projection."""

        return sum(
            1
            for pair in self.disulfide_atom_ref_pairs
            if all(atom_ref.residue_id in residue_ids for atom_ref in pair)
        )

    def is_contradictory_in_residue_projection(
        self,
        residue_ids: frozenset[ResidueId],
    ) -> bool:
        """Return whether endpoint multiplicity survives one residue projection."""

        return self.projected_pair_count(residue_ids) > 1


@dataclass(frozen=True, slots=True)
class StructureDisulfideTopologyFacts:
    """Planner-readable disulfide evidence and canonical contradictions."""

    carrier: ProteinStructure
    promotable_candidates: tuple[LikelyDisulfideBond, ...]
    conflicts: tuple[DisulfideTopologyConflict, ...]
    ambiguous_findings: tuple[AmbiguousDisulfideFinding, ...]
    endpoint_multiplicity_contradictions: tuple[
        DisulfideEndpointMultiplicityContradiction,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        promotable_candidates = tuple(self.promotable_candidates)
        conflicts = tuple(self.conflicts)
        ambiguous_findings = tuple(self.ambiguous_findings)
        raw_endpoint_multiplicity_contradictions = tuple(
            self.endpoint_multiplicity_contradictions
        )
        if any(
            not isinstance(
                contradiction,
                DisulfideEndpointMultiplicityContradiction,
            )
            for contradiction in raw_endpoint_multiplicity_contradictions
        ):
            raise TypeError(
                "endpoint multiplicity contradictions require typed values"
            )
        endpoint_multiplicity_contradictions = tuple(
            sorted(
                raw_endpoint_multiplicity_contradictions,
                key=lambda contradiction: contradiction.sulfur_atom_ref,
            )
        )
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
        contradictory_endpoints = tuple(
            contradiction.sulfur_atom_ref
            for contradiction in endpoint_multiplicity_contradictions
        )
        if len(set(contradictory_endpoints)) != len(contradictory_endpoints):
            raise ValueError(
                "disulfide endpoint multiplicity contradictions must not repeat"
            )
        object.__setattr__(self, "promotable_candidates", promotable_candidates)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "ambiguous_findings", ambiguous_findings)
        object.__setattr__(
            self,
            "endpoint_multiplicity_contradictions",
            endpoint_multiplicity_contradictions,
        )

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
    ) -> "StructureDisulfideTopologyFacts":
        """Resolve geometric disulfide evidence against canonical topology."""

        raw_likely_candidates, _ = detect_disulfide_topology(structure)
        likely_candidates, ambiguous_findings = (
            detect_unassigned_disulfide_evidence(structure)
        )
        promotable_candidates: list[LikelyDisulfideBond] = []
        conflicts_by_pair: dict[
            tuple[ResidueId, ResidueId],
            DisulfideTopologyConflict,
        ] = {}
        candidates_by_pair = {
            candidate.residue_pair(): candidate
            for candidate in (*raw_likely_candidates, *likely_candidates)
        }
        for candidate in candidates_by_pair.values():
            conflict = _candidate_topology_conflict(structure, candidate)
            if conflict is not None:
                conflicts_by_pair[candidate.residue_pair()] = conflict

        for candidate in likely_candidates:
            if candidate.residue_pair() in conflicts_by_pair:
                continue
            if _candidate_is_already_resolved(structure, candidate):
                continue
            promotable_candidates.append(candidate)

        return cls(
            carrier=structure,
            promotable_candidates=tuple(promotable_candidates),
            conflicts=tuple(conflicts_by_pair.values()),
            ambiguous_findings=ambiguous_findings,
            endpoint_multiplicity_contradictions=(
                DisulfideEndpointMultiplicityContradiction.all_from_structure(
                    structure
                )
            ),
        )

    def has_promotable_candidates(self) -> bool:
        """Return whether explicit topology resolution can make progress."""

        return bool(self.promotable_candidates)

    def has_endpoint_multiplicity_contradictions(self) -> bool:
        """Return whether any canonical disulfide endpoint has multiple partners."""

        return bool(self.endpoint_multiplicity_contradictions)


@dataclass(frozen=True, slots=True)
class DisulfideHydrogenContradiction:
    """Forbidden thiol hydrogens observed on one canonical disulfide pair."""

    disulfide_atom_ref_pair: tuple[AtomRef, AtomRef]
    forbidden_hydrogen_atom_refs: tuple[AtomRef, ...]

    def __post_init__(self) -> None:
        disulfide_pair = tuple(self.disulfide_atom_ref_pair)
        raw_forbidden_atom_refs = tuple(self.forbidden_hydrogen_atom_refs)
        if any(not isinstance(atom_ref, AtomRef) for atom_ref in disulfide_pair):
            raise TypeError(
                "disulfide hydrogen contradictions require AtomRef endpoints"
            )
        if any(
            not isinstance(atom_ref, AtomRef) for atom_ref in raw_forbidden_atom_refs
        ):
            raise TypeError(
                "disulfide hydrogen contradictions require AtomRef forbidden atoms"
            )
        forbidden_atom_refs = tuple(sorted(dict.fromkeys(raw_forbidden_atom_refs)))
        if len(disulfide_pair) != 2 or disulfide_pair[0] >= disulfide_pair[1]:
            raise ValueError(
                "disulfide hydrogen contradictions require one canonical SG pair"
            )
        if any(atom_ref.atom_name != "SG" for atom_ref in disulfide_pair):
            raise ValueError(
                "disulfide hydrogen contradictions require CYS SG endpoints"
            )
        if not forbidden_atom_refs:
            raise ValueError(
                "disulfide hydrogen contradictions require forbidden atoms"
            )
        endpoint_residue_ids = {atom_ref.residue_id for atom_ref in disulfide_pair}
        if any(
            atom_ref.residue_id not in endpoint_residue_ids
            for atom_ref in forbidden_atom_refs
        ):
            raise ValueError(
                "forbidden disulfide hydrogens must belong to bond endpoints"
            )
        object.__setattr__(self, "disulfide_atom_ref_pair", disulfide_pair)
        object.__setattr__(
            self,
            "forbidden_hydrogen_atom_refs",
            forbidden_atom_refs,
        )

    def affected_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return disulfide endpoints that carry forbidden hydrogens."""

        forbidden_residue_ids = {
            atom_ref.residue_id for atom_ref in self.forbidden_hydrogen_atom_refs
        }
        return tuple(
            atom_ref.residue_id
            for atom_ref in self.disulfide_atom_ref_pair
            if atom_ref.residue_id in forbidden_residue_ids
        )

    def present_endpoint_count(self) -> int:
        """Return how many disulfide endpoints carry forbidden hydrogens."""

        return len(self.affected_residue_ids())


@dataclass(frozen=True, slots=True, init=False)
class StructureDisulfideHydrogenFacts:
    """Structure-local forbidden hydrogen facts over canonical disulfides."""

    carrier: ProteinStructure
    contradictions: tuple[DisulfideHydrogenContradiction, ...]

    def __init__(self, *, carrier: ProteinStructure) -> None:
        if not isinstance(carrier, ProteinStructure):
            raise TypeError("disulfide hydrogen facts require a ProteinStructure")
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(
            self,
            "contradictions",
            type(self)._contradictions_for_structure(carrier),
        )

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
    ) -> "StructureDisulfideHydrogenFacts":
        """Derive forbidden thiol hydrogens from topology and atom inventory."""

        return cls(carrier=structure)

    @classmethod
    def _contradictions_for_structure(
        cls,
        structure: ProteinStructure,
    ) -> tuple[DisulfideHydrogenContradiction, ...]:
        """Derive the complete canonical contradiction set."""

        contradictions: list[DisulfideHydrogenContradiction] = []
        for disulfide_pair in sorted(disulfide_atom_ref_pairs(structure)):
            forbidden_atom_refs = tuple(
                atom_ref
                for sulfur_atom_ref in disulfide_pair
                for atom_ref in cls._forbidden_atom_refs_for_sulfur(
                    structure,
                    sulfur_atom_ref,
                )
            )
            if forbidden_atom_refs:
                contradictions.append(
                    DisulfideHydrogenContradiction(
                        disulfide_atom_ref_pair=disulfide_pair,
                        forbidden_hydrogen_atom_refs=forbidden_atom_refs,
                    )
                )

        return tuple(contradictions)

    @staticmethod
    def _forbidden_atom_refs_for_sulfur(
        structure: ProteinStructure,
        sulfur_atom_ref: AtomRef,
    ) -> tuple[AtomRef, ...]:
        """Return thiol H/D/T atoms incompatible with one disulfide sulfur."""

        residue_index = structure.constitution.residue_index(
            sulfur_atom_ref.residue_id
        )
        residue_site = structure.constitution.residue_site_at(residue_index)
        forbidden_atom_refs = {
            AtomRef(residue_site.residue_id, atom_site.name)
            for atom_site in residue_site.atom_sites
            if atom_site.is_hydrogen()
            and atom_site.name in _CYSTEINE_THIOL_HYDROGEN_ATOM_NAMES
        }
        sulfur_atom_index = structure.constitution.atom_index(sulfur_atom_ref)
        for bond in structure.topology.bonds:
            if not bond.involves(
                sulfur_atom_index
            ) or not is_covalent_like_relationship(bond):
                continue
            other_atom_index = (
                bond.atom_index_2
                if bond.atom_index_1 == sulfur_atom_index
                else bond.atom_index_1
            )
            other_atom_ref = structure.constitution.atom_ref_at(other_atom_index)
            if other_atom_ref.residue_id != sulfur_atom_ref.residue_id:
                continue
            if structure.constitution.atom_site_at(other_atom_index).is_hydrogen():
                forbidden_atom_refs.add(other_atom_ref)

        return tuple(sorted(forbidden_atom_refs))

    def forbidden_hydrogen_atom_refs(self) -> tuple[AtomRef, ...]:
        """Return all forbidden disulfide-bound hydrogen atom identities."""

        return tuple(
            sorted(
                {
                    atom_ref
                    for contradiction in self.contradictions
                    for atom_ref in contradiction.forbidden_hydrogen_atom_refs
                }
            )
        )

    def has_contradictions(self) -> bool:
        """Return whether canonical disulfides retain thiol hydrogens."""

        return bool(self.contradictions)


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
