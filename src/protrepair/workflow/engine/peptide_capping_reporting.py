"""Attribute a verified cap junction to the branch that actually survived."""

from dataclasses import dataclass, field

from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.bound_peptide_ports import BoundPeptidePorts
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType
from protrepair.transformer.completion.terminal.cap_placement import (
    CapPortAssessment,
    PortCondition,
)
from protrepair.transformer.completion.terminal.cap_product import neighbors
from protrepair.transformer.refinement.speculative_planning import (
    SpeculativePlanningNodeId,
)
from protrepair.workflow.actions.peptide_capping import (
    PeptideCappingTransformer,
    PeptideCapRefusal,
)
from protrepair.workflow.contracts.peptide_capping import (
    PeptideCapOutcome,
    PeptideCappingReport,
    PeptideCapStatus,
    PeptideEnd,
    PeptidePort,
)
from protrepair.workflow.planning.peptide_capping import (
    PeptideCapSchedule,
    PeptidePortReadiness,
)


def _current_cap_partner(
    structure: ProteinStructure, port: PeptidePort
) -> AtomRef | None:
    """Describe a unique edge present in the result, even when the source vetoes its use."""
    c = structure.constitution
    matches = []
    for other, edge in neighbors(structure, port.atom_ref):
        if (
            edge.relationship_type is not BondRelationshipType.COVALENT
            or other.atom_name != ("C" if port.end is PeptideEnd.N else "N")
        ):
            continue
        index = c.residue_index(other.residue_id)
        if (
            c.role_at(index) is ResidueRole.LIGAND
            and c.residue_site_at(index).component_id == port.expected_component
        ):
            matches.append(other)
    return matches[0] if len(matches) == 1 else None


@dataclass(slots=True)
class PeptideCappingHistory:
    """Per-port attempts and budget stops; no candidate coordinates are retained."""

    binding: BoundPeptidePorts
    parents: dict[SpeculativePlanningNodeId, SpeculativePlanningNodeId | None] = field(
        default_factory=dict, init=False
    )
    _written: set[tuple[AtomRef, ResidueId]] = field(default_factory=set, init=False)
    _failures: dict[tuple[SpeculativePlanningNodeId, AtomRef], PeptideCapRefusal] = (
        field(default_factory=dict, init=False)
    )
    _budget: dict[tuple[SpeculativePlanningNodeId, AtomRef], str] = field(
        default_factory=dict, init=False
    )

    def record(
        self,
        parent: SpeculativePlanningNodeId,
        action: PeptideCappingTransformer,
        repairs: tuple[RepairEvent, ...],
        issues: tuple[ValidationIssue, ...],
    ) -> None:
        """Record attempts before pruning; a trial alone never establishes a cap."""
        if any(e.kind is RepairEventKind.PEPTIDE_CAP_PLACED for e in repairs):
            self._written.add((action.port, action.cap_id))
        for issue in issues:
            if isinstance(issue, PeptideCapRefusal):
                self._failures[parent, action.port] = issue

    def record_budget(
        self, node: SpeculativePlanningNodeId, schedule: PeptideCapSchedule
    ) -> None:
        for port, _, readiness in schedule.entries:
            if readiness in (
                PeptidePortReadiness.PENDING_GRAFT,
                PeptidePortReadiness.PENDING_HEAVY_REPAIR,
            ):
                self._budget[node, port] = readiness.value

    def report(
        self,
        structure: ProteinStructure,
        repairs: tuple[RepairEvent, ...],
        selected: SpeculativePlanningNodeId,
        *,
        limited: bool,
    ) -> PeptideCappingReport:
        """Check actual chemistry after finalization and attribute only selected writes."""
        ancestors = []
        node: SpeculativePlanningNodeId | None = selected
        while node is not None:
            ancestors.append(node)
            node = self.parents[node]
        new_cap_ids = {
            impact.residue_id
            for event in repairs
            if event.kind is RepairEventKind.PEPTIDE_CAP_PLACED
            for impact in event.residue_impacts
            if impact.component_id in ("ACE", "NME")
        }
        reconciled_ids = {
            impact.residue_id
            for event in repairs
            if event.kind is RepairEventKind.PEPTIDE_CAP_TOPOLOGY_RESOLVED
            for impact in event.residue_impacts
            if impact.component_id in ("ACE", "NME")
        }
        outcomes = []
        for ref in self.binding.selected:
            port = PeptidePort(ref.residue_id, PeptideEnd(ref.atom_name))
            state = CapPortAssessment.observe(self.binding, structure, ref)
            reason = state.reason
            reserved = self.binding.cap_ids.get(ref)
            partner = _current_cap_partner(structure, port)
            cap_id = partner.residue_id if partner is not None else None
            if state.condition is PortCondition.OMITTED:
                cap_id = state.cap_id
            actual_partner = partner or state.current_partner
            actual = (ref, actual_partner) if actual_partner is not None else None
            failure = next(
                (
                    self._failures[n, ref]
                    for n in ancestors
                    if (n, ref) in self._failures
                ),
                None,
            )
            detail = ""
            rejected = ()
            if state.condition is PortCondition.COMPLETE:
                if cap_id in new_cap_ids:
                    status, reason = PeptideCapStatus.APPLIED, "placed"
                else:
                    status = PeptideCapStatus.ALREADY_CAPPED
                    if cap_id in reconciled_ids:
                        reason = "reconciled_existing"
            elif state.condition is PortCondition.OMITTED:
                status = PeptideCapStatus.OMITTED
            elif state.condition is PortCondition.OCCUPIED:
                status = PeptideCapStatus.OCCUPIED
            elif state.condition is PortCondition.REFUSED:
                status = (
                    PeptideCapStatus.ATTACHED_UNVERIFIED
                    if actual is not None
                    else PeptideCapStatus.REFUSED
                )
            elif failure is not None:
                status = (
                    PeptideCapStatus.ATTACHED_UNVERIFIED
                    if actual is not None
                    else PeptideCapStatus.REFUSED
                )
                reason, detail, rejected = (
                    failure.reason,
                    failure.message,
                    failure.refusals,
                )
            elif limited and (selected, ref) in self._budget:
                status, reason = (
                    PeptideCapStatus.SEARCH_LIMIT,
                    self._budget[selected, ref],
                )
            elif reserved is not None and (ref, reserved) in self._written:
                status, reason = PeptideCapStatus.NOT_RETAINED, "placement_not_retained"
            elif limited:
                status = PeptideCapStatus.SEARCH_LIMIT
                reason = "search_limit"
            else:
                status, reason = PeptideCapStatus.NOT_ATTEMPTED, "not_attempted"
            if (
                state.condition is PortCondition.RECONCILE
                and actual is not None
                and status is PeptideCapStatus.NOT_ATTEMPTED
            ):
                status, reason = PeptideCapStatus.ATTACHED_UNVERIFIED, state.reason
            outcomes.append(
                PeptideCapOutcome(
                    port,
                    status,
                    reason,
                    self.binding.facts.get(ref),
                    self.binding.cap_ids.get(ref),
                    None if state.condition is PortCondition.OPEN else cap_id,
                    actual,
                    detail,
                    rejected,
                )
            )
        return PeptideCappingReport(tuple(outcomes))
