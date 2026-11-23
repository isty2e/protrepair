"""Attribute realized site chemistry to the branch that actually supplied it."""

from dataclasses import dataclass, field

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.bound_protonation import BoundSiteProtonations
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.requested_sites import RequestedSiteAssessment
from protrepair.chemistry.microstate.selection import SiteProtonation
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.refinement.speculative_planning import (
    SpeculativePlanningNodeId,
)
from protrepair.workflow.contracts.site_protonation import (
    SiteProtonationOutcome,
    SiteProtonationReport,
    SiteProtonationStatus,
)


@dataclass(slots=True)
class SiteProtonationHistory:
    """Keep trial evidence separate from conformance on each retained lineage."""

    binding: BoundSiteProtonations
    library: ComponentLibrary
    _written: set[tuple[SpeculativePlanningNodeId, SiteProtonation]] = field(
        default_factory=set, init=False
    )
    _failures: dict[tuple[SpeculativePlanningNodeId, SiteProtonation], str] = field(
        default_factory=dict, init=False
    )
    _parents: dict[SpeculativePlanningNodeId, SpeculativePlanningNodeId | None] = field(
        default_factory=dict, init=False
    )
    _lineages: dict[
        SpeculativePlanningNodeId, dict[SiteProtonation, SiteProtonationStatus]
    ] = field(default_factory=dict, init=False)

    def record(
        self,
        parent: SpeculativePlanningNodeId,
        request: SiteProtonation,
        repairs: tuple[RepairEvent, ...],
        issues: tuple[ValidationIssue, ...],
    ) -> None:
        if any(
            event.kind is RepairEventKind.SITE_PROTONATION_APPLIED for event in repairs
        ):
            self._written.add((parent, request))
        else:
            for issue in issues:
                if issue.kind is ValidationIssueKind.SITE_PROTONATION_FAILED:
                    self._failures[parent, request] = issue.message

    def retain(
        self,
        node: SpeculativePlanningNodeId,
        parent: SpeculativePlanningNodeId | None,
        structure: ProteinStructure,
        writer: SiteProtonation | None,
    ) -> None:
        context = PolymerMicrostateContext(structure)
        previous = self._lineages[parent] if parent is not None else {}
        self._parents[node] = parent
        current = {}
        for request in self.binding.requests.values():
            if (
                RequestedSiteAssessment.observe(
                    self.binding, context, self.library, request
                ).realized
                is None
            ):
                continue
            if request in previous:
                current[request] = previous[request]
            elif parent is None:
                current[request] = SiteProtonationStatus.ALREADY_SATISFIED
            elif writer == request and (parent, request) in self._written:
                current[request] = SiteProtonationStatus.APPLIED
            else:
                current[request] = SiteProtonationStatus.SATISFIED_BY_OTHER_WORK
        self._lineages[node] = current

    def report(
        self,
        structure: ProteinStructure,
        selected: SpeculativePlanningNodeId,
        *,
        search_limited: bool,
    ) -> SiteProtonationReport:
        context = PolymerMicrostateContext(structure)
        outcomes = []
        lineage = self._lineages[selected]
        trial_requests = {request for _, request in (*self._written, *self._failures)}
        ancestors = []
        node: SpeculativePlanningNodeId | None = selected
        while node is not None:
            ancestors.append(node)
            node = self._parents[node]
        for request in self.binding.requests.values():
            state = RequestedSiteAssessment.observe(
                self.binding, context, self.library, request
            )
            resolution = state.resolution
            details = resolution.details
            if state.realized is not None:
                status = lineage.get(
                    request, SiteProtonationStatus.SATISFIED_BY_OTHER_WORK
                )
            elif resolution.graph is None:
                status = SiteProtonationStatus.REFUSED
            elif (
                failure := next(
                    (
                        self._failures[parent, request]
                        for parent in ancestors
                        if (parent, request) in self._failures
                    ),
                    None,
                )
            ) is not None:
                status = SiteProtonationStatus.PLACEMENT_FAILED
                details = (*details, failure)
            elif request in trial_requests:
                status = SiteProtonationStatus.NOT_RETAINED
            else:
                status = (
                    SiteProtonationStatus.SEARCH_LIMIT
                    if search_limited
                    else SiteProtonationStatus.NOT_ATTEMPTED
                )
            classes = tuple(
                sorted({graph.protonation_key() for graph in resolution.candidates})
            )
            outcomes.append(
                SiteProtonationOutcome(
                    request,
                    resolution.status,
                    classes,
                    status,
                    details,
                    reason=resolution.reason,
                )
            )
        return SiteProtonationReport(tuple(outcomes))
