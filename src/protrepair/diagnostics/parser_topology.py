"""Cross-diagnostics between parser witnesses and topology ambiguity."""

from collections import defaultdict
from dataclasses import dataclass

from protrepair.diagnostics.events import EventScope, ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.diagnostics.topology import (
    detect_unassigned_disulfide_evidence,
    residue_sort_key,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId


@dataclass(frozen=True, slots=True)
class AmbiguousDisulfideParserWitnessBlocker:
    """Parser-witness burden that ordinary FF repair must not resolve."""

    possible_disulfide_residue_ids: tuple[ResidueId, ...]
    witnesses: tuple[RDKitProximityBondWitness, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "possible_disulfide_residue_ids",
            tuple(
                sorted(
                    self.possible_disulfide_residue_ids,
                    key=residue_sort_key,
                )
            ),
        )
        object.__setattr__(
            self,
            "witnesses",
            tuple(
                sorted(
                    self.witnesses,
                    key=lambda witness: witness.display_token(),
                )
            ),
        )


def ambiguous_disulfide_partner_residue_ids_by_residue(
    structure: ProteinStructure,
) -> dict[ResidueId, frozenset[ResidueId]]:
    """Return ambiguous disulfide neighborhoods keyed by member residue."""

    _, ambiguous_disulfides = detect_unassigned_disulfide_evidence(structure)
    partner_residue_ids_by_residue: dict[ResidueId, frozenset[ResidueId]] = {}
    for finding in ambiguous_disulfides:
        finding_residue_ids = frozenset(
            (
                finding.residue_id,
                *tuple(candidate.residue_id for candidate in finding.candidates),
            )
        )
        for residue_id in finding_residue_ids:
            partner_residue_ids_by_residue[residue_id] = frozenset(
                (
                    *partner_residue_ids_by_residue.get(residue_id, frozenset()),
                    *finding_residue_ids,
                )
            )

    return partner_residue_ids_by_residue


def parser_witness_cluster_is_ambiguous_disulfide_blocker(
    cluster: RDKitProximityBondCluster,
    *,
    ambiguous_disulfide_partners: dict[ResidueId, frozenset[ResidueId]],
) -> bool:
    """Return whether a parser-witness cluster overlaps ambiguous S-S topology."""

    return any(
        _parser_witness_is_ambiguous_disulfide_blocker(
            witness,
            ambiguous_disulfide_partners=ambiguous_disulfide_partners,
        )
        for witness in cluster.bonds
    )


def ambiguous_disulfide_parser_witness_blocker_issues(
    structure: ProteinStructure,
    *,
    clusters: tuple[RDKitProximityBondCluster, ...],
) -> tuple[ValidationIssue, ...]:
    """Return typed issues for parser witnesses blocked by disulfide ambiguity."""

    blockers = ambiguous_disulfide_parser_witness_blockers(
        structure,
        clusters=clusters,
    )
    return tuple(_ambiguous_disulfide_blocker_issue(blocker) for blocker in blockers)


def ambiguous_disulfide_parser_witness_blockers(
    structure: ProteinStructure,
    *,
    clusters: tuple[RDKitProximityBondCluster, ...],
) -> tuple[AmbiguousDisulfideParserWitnessBlocker, ...]:
    """Return parser/topology blockers grouped by ambiguous disulfide set."""

    ambiguous_disulfide_partners = ambiguous_disulfide_partner_residue_ids_by_residue(
        structure
    )
    if not ambiguous_disulfide_partners:
        return ()

    witnesses_by_possible_disulfide_set: dict[
        frozenset[ResidueId],
        list[RDKitProximityBondWitness],
    ] = defaultdict(list)
    for cluster in clusters:
        for witness in cluster.bonds:
            possible_disulfide_residue_ids = (
                _parser_witness_ambiguous_disulfide_partner_residue_ids(
                    witness,
                    ambiguous_disulfide_partners=ambiguous_disulfide_partners,
                )
            )
            if possible_disulfide_residue_ids is None:
                continue

            witnesses_by_possible_disulfide_set[
                possible_disulfide_residue_ids
            ].append(witness)

    return tuple(
        AmbiguousDisulfideParserWitnessBlocker(
            possible_disulfide_residue_ids=tuple(possible_disulfide_residue_ids),
            witnesses=tuple(dict.fromkeys(witnesses)),
        )
        for possible_disulfide_residue_ids, witnesses in sorted(
            witnesses_by_possible_disulfide_set.items(),
            key=lambda item: tuple(
                residue_sort_key(residue_id) for residue_id in item[0]
            ),
        )
    )


def _ambiguous_disulfide_blocker_issue(
    blocker: AmbiguousDisulfideParserWitnessBlocker,
) -> ValidationIssue:
    """Return one validation issue from a structured parser/topology blocker."""

    display_witnesses = ", ".join(
        witness.display_token()
        for witness in blocker.witnesses[:3]
    )
    partner_text = ", ".join(
        residue_id.display_token()
        for residue_id in blocker.possible_disulfide_residue_ids
    )
    message = (
        "no-CONECT RDKit parser witness overlaps ambiguous disulfide topology; "
        "ordinary parser-witness local FF repair was skipped for possible "
        f"disulfide set {partner_text}; witness extra bond(s): {display_witnesses}"
    )
    if len(blocker.witnesses) > 3:
        message += f" (+{len(blocker.witnesses) - 3} more)"

    return ValidationIssue(
        kind=ValidationIssueKind.AMBIGUOUS_DISULFIDE,
        severity=IssueSeverity.WARNING,
        scope=EventScope.for_residue_set(blocker.possible_disulfide_residue_ids),
        message=message,
    )


def _parser_witness_is_ambiguous_disulfide_blocker(
    witness: RDKitProximityBondWitness,
    *,
    ambiguous_disulfide_partners: dict[ResidueId, frozenset[ResidueId]],
) -> bool:
    """Return whether one parser witness is explained by ambiguous S-S topology."""

    return (
        _parser_witness_ambiguous_disulfide_partner_residue_ids(
            witness,
            ambiguous_disulfide_partners=ambiguous_disulfide_partners,
        )
        is not None
    )


def _parser_witness_ambiguous_disulfide_partner_residue_ids(
    witness: RDKitProximityBondWitness,
    *,
    ambiguous_disulfide_partners: dict[ResidueId, frozenset[ResidueId]],
) -> frozenset[ResidueId] | None:
    """Return the ambiguous disulfide set explaining one parser witness."""

    matching_partner_sets: list[frozenset[ResidueId]] = []
    left_partner_residue_ids = _ambiguous_sulfur_partner_residue_ids(
        witness,
        endpoint=1,
        ambiguous_disulfide_partners=ambiguous_disulfide_partners,
    )
    if (
        left_partner_residue_ids is not None
        and witness.atom_ref_2.residue_id in left_partner_residue_ids
    ):
        matching_partner_sets.append(left_partner_residue_ids)

    right_partner_residue_ids = _ambiguous_sulfur_partner_residue_ids(
        witness,
        endpoint=2,
        ambiguous_disulfide_partners=ambiguous_disulfide_partners,
    )
    if (
        right_partner_residue_ids is not None
        and witness.atom_ref_1.residue_id in right_partner_residue_ids
    ):
        matching_partner_sets.append(right_partner_residue_ids)

    if not matching_partner_sets:
        return None

    return frozenset(
        residue_id
        for partner_set in matching_partner_sets
        for residue_id in partner_set
    )


def _ambiguous_sulfur_partner_residue_ids(
    witness: RDKitProximityBondWitness,
    *,
    endpoint: int,
    ambiguous_disulfide_partners: dict[ResidueId, frozenset[ResidueId]],
) -> frozenset[ResidueId] | None:
    """Return the ambiguous disulfide partner set for one sulfur endpoint."""

    if endpoint == 1:
        atom_ref = witness.atom_ref_1
        element = witness.element_1
    elif endpoint == 2:
        atom_ref = witness.atom_ref_2
        element = witness.element_2
    else:
        raise ValueError("parser-witness endpoint must be 1 or 2")

    if element != "S" or atom_ref.atom_name.strip().upper() != "SG":
        return None

    return ambiguous_disulfide_partners.get(atom_ref.residue_id)
