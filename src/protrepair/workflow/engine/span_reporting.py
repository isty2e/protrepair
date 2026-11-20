"""Observe external span trials without changing the workflow search."""

from dataclasses import dataclass, replace
from typing import TypeVar

from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.errors import ModelInvariantError
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.refinement.speculative_planning import (
    SpeculativePlanningNodeId,
    SpeculativePlanningTrace,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.contracts.external_reference import (
    ExternalSpanReconstructionSpec,
)
from protrepair.workflow.contracts.span_report import (
    ExternalSpanReconstructionOutcome,
    ExternalSpanReconstructionReport,
    ExternalSpanReconstructionStatus,
)
from protrepair.workflow.planning.action.registry import WorkflowStateAction

_NodeStateT = TypeVar("_NodeStateT")


def _missing_positions(
    structure: ProteinStructure, scope: AbsentResidueSpanScope
) -> tuple[ResidueId, ...]:
    return tuple(
        residue_id
        for residue_id in scope.absent_residue_ids
        if structure.constitution.residue_or_ligand(residue_id) is None
    )


@dataclass(frozen=True, slots=True)
class _SpanRequestTrials:
    scope: AbsentResidueSpanScope
    initially_present: bool
    attempt_count: int = 0
    successful_attempt_count: int = 0
    last_failure: ValidationIssue | None = None

    def status(
        self, *, missing: bool, graft_on_selected_lineage: bool
    ) -> ExternalSpanReconstructionStatus:
        if not missing:
            if graft_on_selected_lineage:
                return ExternalSpanReconstructionStatus.APPLIED
            if self.initially_present:
                return ExternalSpanReconstructionStatus.ALREADY_PRESENT
            return ExternalSpanReconstructionStatus.PRESENT_WITHOUT_THIS_REQUEST
        if self.successful_attempt_count:
            return ExternalSpanReconstructionStatus.NOT_RETAINED
        if self.attempt_count:
            return ExternalSpanReconstructionStatus.FAILED
        return ExternalSpanReconstructionStatus.NOT_ATTEMPTED


@dataclass(frozen=True, slots=True)
class _SpanAncestryStep:
    node_id: SpeculativePlanningNodeId
    parent: SpeculativePlanningNodeId | None
    grafted_request_index: int | None


@dataclass(frozen=True, slots=True)
class SpanReconstructionEvidence:
    """Span trial totals and compact ancestry, without candidate structures."""

    _requests: tuple[_SpanRequestTrials, ...]
    _ancestry: tuple[_SpanAncestryStep, ...]

    def report_for(
        self, structure: ProteinStructure, *, selected_node: SpeculativePlanningNodeId
    ) -> ExternalSpanReconstructionReport:
        """Project final coverage and donor attribution for the selected node."""

        steps = {step.node_id: step for step in self._ancestry}
        selected_grafts: set[int] = set()
        node: SpeculativePlanningNodeId | None = selected_node
        while node is not None:
            step = steps[node]
            if step.grafted_request_index is not None:
                selected_grafts.add(step.grafted_request_index)
            node = step.parent

        outcomes = []
        for index, trials in enumerate(self._requests):
            missing = _missing_positions(structure, trials.scope)
            outcomes.append(
                ExternalSpanReconstructionOutcome(
                    request_index=index,
                    scope=trials.scope,
                    status=trials.status(
                        missing=bool(missing),
                        graft_on_selected_lineage=index in selected_grafts,
                    ),
                    missing_residue_ids=missing,
                    attempt_count=trials.attempt_count,
                    successful_attempt_count=trials.successful_attempt_count,
                    last_failure=trials.last_failure,
                )
            )
        return ExternalSpanReconstructionReport(tuple(outcomes))


class SpanReconstructionRecorder:
    """Collect one invocation's span trials before sibling pruning."""

    def __init__(
        self,
        source: ProteinStructure,
        requests: tuple[ExternalSpanReconstructionSpec, ...],
    ) -> None:
        self._index_by_identity = {
            id(spec): index for index, spec in enumerate(requests)
        }
        self._requests = [
            _SpanRequestTrials(
                scope=spec.scope,
                initially_present=not _missing_positions(source, spec.scope),
            )
            for spec in requests
        ]
        self._successful_trials: set[tuple[SpeculativePlanningNodeId, int]] = set()

    def record(
        self,
        parent: SpeculativePlanningNodeId,
        *,
        transformer: ExternalSpanReconstructionTransformer,
        repairs: tuple[RepairEvent, ...],
        issues: tuple[ValidationIssue, ...],
    ) -> None:
        """Observe only the new records appended by this execution."""

        index = self._index_by_identity[id(transformer.reconstruction)]
        trials = self._requests[index]
        target = trials.scope.absent_residue_ids
        if (
            len(repairs) == 1
            and not issues
            and repairs[0].kind is RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED
            and repairs[0].scope.residue_ids == target
        ):
            self._successful_trials.add((parent, index))
            self._requests[index] = replace(
                trials,
                attempt_count=trials.attempt_count + 1,
                successful_attempt_count=trials.successful_attempt_count + 1,
            )
        elif (
            not repairs
            and len(issues) == 1
            and issues[0].kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
            and issues[0].scope.residue_ids == target
        ):
            self._requests[index] = replace(
                trials, attempt_count=trials.attempt_count + 1, last_failure=issues[0]
            )
        else:
            raise ModelInvariantError("span execution must append one graft or refusal")

    def finish(
        self,
        trace: SpeculativePlanningTrace[
            _NodeStateT, WorkflowStateAction, TransformationResult, TransformationResult
        ],
    ) -> SpanReconstructionEvidence:
        """Bind successful trials to the incoming edges actually retained."""

        ancestry = []
        for node in trace.nodes:
            parent = node.parent_node_id
            decision = node.adopted_decision
            graft = None
            if (
                parent is not None
                and decision is not None
                and decision.candidate is not None
            ):
                transformer = decision.candidate.execution.proposal
                if isinstance(transformer, ExternalSpanReconstructionTransformer):
                    index = self._index_by_identity[id(transformer.reconstruction)]
                    if (parent, index) in self._successful_trials:
                        graft = index
            ancestry.append(_SpanAncestryStep(node.node_id, parent, graft))
        return SpanReconstructionEvidence(tuple(self._requests), tuple(ancestry))
