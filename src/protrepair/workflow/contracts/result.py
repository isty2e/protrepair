"""Structured processing result contracts for the redesigned ProtRepair package."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import Self

from protrepair.analysis.results import (
    AnalysisBundle,
    RamachandranAnalysis,
    RamachandranCategory,
    RamachandranPoint,
    SecondaryStructureAnalysis,
    SecondaryStructureAssignment,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.scope import Scope
from protrepair.state.scoped import ScopedState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.refinement.result import RefinementTransformationResult
from protrepair.transformer.refinement.speculative_planning import SpeculativeStopReason
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
    WorkflowPlanningPhase,
)
from protrepair.workflow.contracts.request import (
    WorkflowGoal,
    WorkflowGoalStateValue,
    is_workflow_goal_state_value,
    validate_workflow_goal_scope,
)

if TYPE_CHECKING:
    from protrepair.transformer.refinement.speculative_planning import (
        SpeculativePlanningNodeId,
    )

PreferredValueT = TypeVar("PreferredValueT")

BranchPreferenceKey = tuple[int | float, ...]


class RequestedGoalStatus(str, Enum):
    """Goal-oriented outcome status for one requested scoped proposition."""

    ALREADY_SATISFIED = "already_satisfied"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"
    UNMET = "unmet"


class RequestedGoalCompletionVerdict(str, Enum):
    """Top-level requested-goal completion verdict for one workflow run."""

    NOT_REQUESTED = "not_requested"
    ACHIEVED = "achieved"
    PARTIALLY_ACHIEVED = "partially_achieved"
    UNACHIEVED = "unachieved"


class WorkflowPhaseStatus(str, Enum):
    """Final reporting status for one workflow phase."""

    CLEAR = "clear"
    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class WorkflowBranchStatus(str, Enum):
    """Whether a retained candidate finished planning before search stopped."""

    TERMINAL = "terminal"
    FRONTIER = "frontier"


@dataclass(frozen=True, slots=True)
class RequestedGoalOutcome:
    """Outcome for one requested scoped goal after one workflow run."""

    requested_goal: WorkflowGoal
    status: RequestedGoalStatus
    blocking_scopes: tuple[Scope, ...] = ()
    blocking_phases: tuple[WorkflowPlanningPhase, ...] = ()
    observed_state: WorkflowGoalStateValue | None = None
    details: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.requested_goal, ScopedState):
            raise TypeError("requested goal outcomes require a WorkflowGoal value")
        if not is_workflow_goal_state_value(self.requested_goal.value):
            raise TypeError(
                "requested goal outcomes require a supported requested-goal value"
            )
        validate_workflow_goal_scope(self.requested_goal)
        if not isinstance(self.status, RequestedGoalStatus):
            raise TypeError(
                "requested goal outcomes require a RequestedGoalStatus value"
            )
        if self.observed_state is not None and not is_workflow_goal_state_value(
            self.observed_state
        ):
            raise TypeError(
                "requested goal outcome observed_state must be a supported "
                "requested-goal value or None"
            )
        for scope in self.blocking_scopes:
            if not isinstance(scope, Scope):
                raise TypeError(
                    "requested goal outcome blocking_scopes must contain Scope values"
                )
        for phase in self.blocking_phases:
            if not isinstance(phase, WorkflowPlanningPhase):
                raise TypeError(
                    "requested goal outcome blocking_phases must contain "
                    "WorkflowPlanningPhase values"
                )

        object.__setattr__(self, "blocking_scopes", tuple(self.blocking_scopes))
        object.__setattr__(self, "blocking_phases", tuple(self.blocking_phases))

    def is_satisfied(self) -> bool:
        """Return whether this requested goal ended in a satisfied status."""

        return self.status in {
            RequestedGoalStatus.ALREADY_SATISFIED,
            RequestedGoalStatus.SATISFIED,
        }


@dataclass(frozen=True, slots=True)
class WorkflowPhaseOutcome:
    """Final reporting outcome for one explicit workflow phase."""

    phase: WorkflowPlanningPhase
    status: WorkflowPhaseStatus
    blocking_scopes: tuple[Scope, ...] = ()
    details: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, WorkflowPlanningPhase):
            raise TypeError(
                "workflow phase outcomes require a WorkflowPlanningPhase value"
            )
        if not isinstance(self.status, WorkflowPhaseStatus):
            raise TypeError(
                "workflow phase outcomes require a WorkflowPhaseStatus value"
            )
        for scope in self.blocking_scopes:
            if not isinstance(scope, Scope):
                raise TypeError(
                    "workflow phase outcome blocking_scopes must contain Scope values"
                )

        object.__setattr__(self, "blocking_scopes", tuple(self.blocking_scopes))


@dataclass(frozen=True, slots=True, init=False)
class WorkflowPhaseReport:
    """Structured report over explicit workflow phases for one branch."""

    _outcomes: tuple[WorkflowPhaseOutcome, ...] = field(repr=False)

    def __init__(self, outcomes: tuple[WorkflowPhaseOutcome, ...]) -> None:
        normalized_outcomes = tuple(outcomes)
        for outcome in normalized_outcomes:
            if not isinstance(outcome, WorkflowPhaseOutcome):
                raise TypeError(
                    "workflow phase reports require WorkflowPhaseOutcome values"
                )

        object.__setattr__(self, "_outcomes", normalized_outcomes)

    @property
    def outcomes(self) -> tuple[WorkflowPhaseOutcome, ...]:
        """Return phase outcomes for this already-evaluated report."""

        return self._outcomes

    def outcome_for(
        self,
        phase: WorkflowPlanningPhase,
    ) -> WorkflowPhaseOutcome | None:
        """Return the outcome for one specific workflow phase."""

        if not isinstance(phase, WorkflowPlanningPhase):
            raise TypeError("phase must be a WorkflowPlanningPhase value")

        for outcome in self.outcomes:
            if outcome.phase is phase:
                return outcome

        return None


@dataclass(frozen=True, slots=True)
class RequestedGoalReport:
    """Structured report over requested-goal satisfaction for one run."""

    outcomes: tuple[RequestedGoalOutcome, ...]

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        for outcome in outcomes:
            if not isinstance(outcome, RequestedGoalOutcome):
                raise TypeError(
                    "requested goal reports require RequestedGoalOutcome values"
                )

        object.__setattr__(self, "outcomes", outcomes)

    def outcome_for(
        self,
        goal: WorkflowGoal,
    ) -> RequestedGoalOutcome | None:
        """Return the outcome for one exact requested goal if present."""

        for outcome in self.outcomes:
            if outcome.requested_goal == goal:
                return outcome

        return None

    def status_count(
        self,
        status: RequestedGoalStatus,
    ) -> int:
        """Return the number of outcomes with one specific status."""

        if not isinstance(status, RequestedGoalStatus):
            raise TypeError("status must be a RequestedGoalStatus value")

        return sum(1 for outcome in self.outcomes if outcome.status is status)

    def satisfied_count(self) -> int:
        """Return the number of outcomes satisfied before or after execution."""

        return self.status_count(
            RequestedGoalStatus.ALREADY_SATISFIED
        ) + self.status_count(RequestedGoalStatus.SATISFIED)

    def unmet_count(self) -> int:
        """Return the number of outcomes left unmet after execution."""

        return self.status_count(RequestedGoalStatus.UNMET)

    def blocked_count(self) -> int:
        """Return the number of outcomes blocked during planning."""

        return self.status_count(RequestedGoalStatus.BLOCKED)

    def unsupported_count(self) -> int:
        """Return the number of unsupported requested goals."""

        return self.status_count(RequestedGoalStatus.UNSUPPORTED)

    def completion_verdict(self) -> RequestedGoalCompletionVerdict:
        """Return the top-level completion verdict for this requested-goal set."""

        if not self.outcomes:
            return RequestedGoalCompletionVerdict.NOT_REQUESTED

        if self.is_fully_satisfied():
            return RequestedGoalCompletionVerdict.ACHIEVED

        if self.satisfied_count() > 0:
            return RequestedGoalCompletionVerdict.PARTIALLY_ACHIEVED

        return RequestedGoalCompletionVerdict.UNACHIEVED

    def has_failures(self) -> bool:
        """Return whether any requested goal was unmet, blocked, or unsupported."""

        return (
            self.unmet_count() > 0
            or self.blocked_count() > 0
            or self.unsupported_count() > 0
        )

    def is_fully_satisfied(self) -> bool:
        """Return whether every requested goal ended in a satisfied status."""

        return not self.has_failures()

    def preference_key(self) -> tuple[int, int, int, int]:
        """Return the goal-first preference key for this requested-goal report."""

        return (
            -self.satisfied_count(),
            self.unmet_count(),
            self.blocked_count(),
            self.unsupported_count(),
        )

    def branch_preference_key(
        self,
        *,
        error_count: int,
        warning_count: int,
        issue_count: int,
        tie_breaker: int = 0,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        """Return one workflow branch preference key for this report."""

        return (
            *self.preference_key(),
            error_count,
            warning_count,
            issue_count,
            tie_breaker,
        )


@dataclass(frozen=True, slots=True)
class WorkflowBranchQualityScore:
    """Canonical branch-preference projection over observed workflow outcomes."""

    satisfied_goal_count: int = 0
    unmet_goal_count: int = 0
    blocked_goal_count: int = 0
    unsupported_goal_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    issue_count: int = 0
    parser_incompatible: int = 0
    parser_extra_heavy_bond_count: int = 0
    parser_extra_bond_count: int = 0
    protein_self_clash_count: int = 0
    ligand_aware_clash_count: int = 0
    ligand_aware_worst_overlap_angstrom: float = 0.0
    ligand_aware_total_overlap_angstrom: float = 0.0
    stereochemistry_violation_count: int = 0
    severe_geometry_residue_count: int = 0
    search_depth: int = 0

    @classmethod
    def from_requested_goal_report(
        cls,
        requested_goal_report: RequestedGoalReport,
        *,
        error_count: int = 0,
        warning_count: int = 0,
        issue_count: int = 0,
        parser_incompatible: int = 0,
        parser_extra_heavy_bond_count: int = 0,
        parser_extra_bond_count: int = 0,
        protein_self_clash_count: int = 0,
        ligand_aware_clash_count: int = 0,
        ligand_aware_worst_overlap_angstrom: float = 0.0,
        ligand_aware_total_overlap_angstrom: float = 0.0,
        stereochemistry_violation_count: int = 0,
        severe_geometry_residue_count: int = 0,
        search_depth: int = 0,
    ) -> "WorkflowBranchQualityScore":
        """Build a branch score from requested goals plus observed burdens."""

        return cls(
            satisfied_goal_count=requested_goal_report.satisfied_count(),
            unmet_goal_count=requested_goal_report.unmet_count(),
            blocked_goal_count=requested_goal_report.blocked_count(),
            unsupported_goal_count=requested_goal_report.unsupported_count(),
            error_count=error_count,
            warning_count=warning_count,
            issue_count=issue_count,
            parser_incompatible=parser_incompatible,
            parser_extra_heavy_bond_count=parser_extra_heavy_bond_count,
            parser_extra_bond_count=parser_extra_bond_count,
            protein_self_clash_count=protein_self_clash_count,
            ligand_aware_clash_count=ligand_aware_clash_count,
            ligand_aware_worst_overlap_angstrom=(ligand_aware_worst_overlap_angstrom),
            ligand_aware_total_overlap_angstrom=(ligand_aware_total_overlap_angstrom),
            stereochemistry_violation_count=stereochemistry_violation_count,
            severe_geometry_residue_count=severe_geometry_residue_count,
            search_depth=search_depth,
        )

    def order_key(
        self,
        policy: WorkflowBranchPreferencePolicy,
        *,
        tie_breaker: int = 0,
    ) -> BranchPreferenceKey:
        """Return a sortable branch-preference tuple under one axis policy."""

        key_parts: list[int | float] = []
        for axis in policy.axes:
            key_parts.extend(self._axis_key(axis))
        key_parts.append(tie_breaker)
        return tuple(key_parts)

    def _axis_key(
        self,
        axis: WorkflowBranchQualityAxis,
    ) -> BranchPreferenceKey:
        """Return this score's order tuple for one preference axis."""

        if axis is WorkflowBranchQualityAxis.HARD_ERRORS:
            return (self.error_count,)
        if axis is WorkflowBranchQualityAxis.PARSER_COMPATIBILITY:
            return (
                self.parser_incompatible,
                self.parser_extra_heavy_bond_count,
                self.parser_extra_bond_count,
            )
        if axis is WorkflowBranchQualityAxis.INTRINSIC_CLASH:
            return (self.protein_self_clash_count,)
        if axis is WorkflowBranchQualityAxis.INTERACTION_CLASH:
            return (
                self.ligand_aware_clash_count,
                self.ligand_aware_worst_overlap_angstrom,
                self.ligand_aware_total_overlap_angstrom,
            )
        if axis is WorkflowBranchQualityAxis.STEREOCHEMISTRY:
            return (self.stereochemistry_violation_count,)
        if axis is WorkflowBranchQualityAxis.SEVERE_GEOMETRY:
            return (self.severe_geometry_residue_count,)
        if axis is WorkflowBranchQualityAxis.REQUESTED_GOALS:
            return (
                -self.satisfied_goal_count,
                self.unmet_goal_count,
                self.blocked_goal_count,
                self.unsupported_goal_count,
            )
        if axis is WorkflowBranchQualityAxis.ISSUE_BURDEN:
            return (self.warning_count, self.issue_count)
        if axis is WorkflowBranchQualityAxis.SEARCH_DEPTH:
            return (self.search_depth,)
        raise ValueError(f"unknown workflow branch quality axis: {axis!r}")


@dataclass(frozen=True, slots=True)
class WorkflowBranchOutcome:
    """Observed goals and quality for one retained workflow candidate.

    Parameters
    ----------
    node_id : SpeculativePlanningNodeId
        Identity within this search invocation.
    requested_goal_report : RequestedGoalReport
        Goal observations for this candidate, not inferred from termination.
    phase_report : WorkflowPhaseReport
        Phase observations, which may be evaluated lazily.
    branch_quality_score : WorkflowBranchQualityScore
        Policy-independent quality axes for selection.
    error_count, warning_count, issue_count : int
        Diagnostic counts before post-selection final diagnostics.
    status : WorkflowBranchStatus
        Planner-terminal or still on the frontier; neither implies goal attainment.

    Raises
    ------
    TypeError
        Status is not a canonical branch-status value.
    """

    node_id: "SpeculativePlanningNodeId"
    requested_goal_report: RequestedGoalReport
    phase_report: WorkflowPhaseReport
    branch_quality_score: WorkflowBranchQualityScore
    error_count: int
    warning_count: int
    issue_count: int
    status: WorkflowBranchStatus = WorkflowBranchStatus.TERMINAL
    relaxation_executions: tuple[ContinuousRelaxationExecution, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, WorkflowBranchStatus):
            raise TypeError("workflow branch status must be a WorkflowBranchStatus")
        object.__setattr__(
            self,
            "relaxation_executions",
            ContinuousRelaxationExecution.deduplicated(self.relaxation_executions),
        )

    def preference_key(
        self,
        policy: WorkflowBranchPreferencePolicy,
    ) -> BranchPreferenceKey:
        """Return the configured preference key with a stable node-ID tie-breaker.

        Parameters
        ----------
        policy : WorkflowBranchPreferencePolicy
            Ordering of observed quality axes.

        Returns
        -------
        tuple[int or float, ...]
            Ascending preference key.
        """

        return self.branch_quality_score.order_key(
            policy,
            tie_breaker=self.node_id.value,
        )


@dataclass(frozen=True, slots=True)
class WorkflowBranchReport:
    """Select among retained candidates without equating exhaustion with success.

    Parameters
    ----------
    preferred_node_id : SpeculativePlanningNodeId
        Selected node; must agree with the configured quality policy.
    outcomes : tuple[WorkflowBranchOutcome, ...]
        Nonempty, uniquely identified retained candidates.
    branch_preference_policy : WorkflowBranchPreferencePolicy
        Ordering shared with workflow candidate assessment.
    stop_reason : SpeculativeStopReason
        Why search stopped, independent of the selected candidate's goal report.

    Raises
    ------
    TypeError
        Outcomes or the stop reason are noncanonical.
    ValueError
        Candidates are absent, duplicated, inconsistent with termination, or the
        preferred node disagrees with policy.
    """

    preferred_node_id: "SpeculativePlanningNodeId"
    outcomes: tuple[WorkflowBranchOutcome, ...]
    branch_preference_policy: WorkflowBranchPreferencePolicy = field(
        default_factory=WorkflowBranchPreferencePolicy
    )
    stop_reason: SpeculativeStopReason = SpeculativeStopReason.NO_PROPOSALS_AVAILABLE

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("workflow branch report requires at least one outcome")
        if not all(isinstance(outcome, WorkflowBranchOutcome) for outcome in outcomes):
            raise TypeError(
                "workflow branch reports require WorkflowBranchOutcome values"
            )
        if not isinstance(self.stop_reason, SpeculativeStopReason):
            raise TypeError(
                "workflow branch report stop reason must be a SpeculativeStopReason"
            )
        node_ids = tuple(outcome.node_id for outcome in outcomes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("workflow branch report must not repeat node IDs")
        if not self.search_exhausted() and any(
            outcome.status is WorkflowBranchStatus.FRONTIER for outcome in outcomes
        ):
            raise ValueError("frontier outcomes require exhausted workflow search")
        preferred = min(
            outcomes,
            key=lambda outcome: outcome.preference_key(self.branch_preference_policy),
        )
        if self.preferred_node_id != preferred.node_id:
            raise ValueError(
                "preferred workflow branch must match the configured policy"
            )
        object.__setattr__(self, "outcomes", outcomes)

    @classmethod
    def from_outcomes(
        cls,
        outcomes: tuple[WorkflowBranchOutcome, ...],
        *,
        branch_preference_policy: WorkflowBranchPreferencePolicy | None = None,
        stop_reason: SpeculativeStopReason = (
            SpeculativeStopReason.NO_PROPOSALS_AVAILABLE
        ),
    ) -> "WorkflowBranchReport":
        """Choose the preferred candidate under one canonical quality ordering.

        Parameters
        ----------
        outcomes : tuple[WorkflowBranchOutcome, ...]
            Assessed terminal candidates and, on exhaustion, retained frontier nodes.
        branch_preference_policy : WorkflowBranchPreferencePolicy or None
            Axis ordering, or the default policy.
        stop_reason : SpeculativeStopReason
            Actual search termination cause.

        Returns
        -------
        WorkflowBranchReport
            Candidate observations and deterministic selection.

        Raises
        ------
        TypeError
            Outcomes or the stop reason are noncanonical.
        ValueError
            Outcomes are empty, duplicated, or inconsistent with termination.
        """

        if not outcomes:
            raise ValueError("workflow branch report requires at least one outcome")

        active_policy = (
            WorkflowBranchPreferencePolicy()
            if branch_preference_policy is None
            else branch_preference_policy
        )
        preferred_outcome = min(
            outcomes,
            key=lambda outcome: outcome.preference_key(active_policy),
        )
        return cls(
            preferred_node_id=preferred_outcome.node_id,
            outcomes=outcomes,
            branch_preference_policy=active_policy,
            stop_reason=stop_reason,
        )

    def search_exhausted(self) -> bool:
        """Return whether unexplored work remained at the node limit.

        Returns
        -------
        bool
            Whether search stopped at its configured node limit.
        """

        return self.stop_reason is SpeculativeStopReason.ITERATION_LIMIT_REACHED

    def require_selected_native_history(
        self, executions: tuple[ContinuousRelaxationExecution, ...]
    ) -> None:
        """Require known native returns to remain in the selected history.

        Parameters
        ----------
        executions : tuple[ContinuousRelaxationExecution, ...]
            Facts already observed for the selected structural carrier.

        Raises
        ------
        ValueError
            Replacing a report would silently discard known invocation facts.
        """

        if not frozenset(executions).issubset(
            self.preferred_outcome().relaxation_executions
        ):
            raise ValueError(
                "workflow report must retain known native execution history"
            )

    def preferred_outcome(self) -> WorkflowBranchOutcome:
        """Return the selected candidate's observation, not a success verdict.

        Returns
        -------
        WorkflowBranchOutcome
            The outcome selected by this report's policy.
        """

        for outcome in self.outcomes:
            if outcome.node_id == self.preferred_node_id:
                return outcome

        raise ValueError("workflow branch report preferred node is missing")

    def require_preferred_value(
        self,
        values_by_node_id: Mapping[
            "SpeculativePlanningNodeId",
            PreferredValueT,
        ],
    ) -> PreferredValueT:
        """Return the value mapped to the selected candidate.

        Parameters
        ----------
        values_by_node_id : Mapping[SpeculativePlanningNodeId, PreferredValueT]
            Candidate carriers belonging to this invocation.

        Returns
        -------
        PreferredValueT
            Carrier with this report's preferred node identity.

        Raises
        ------
        ValueError
            The selected candidate has no mapped carrier.
        """

        try:
            return values_by_node_id[self.preferred_node_id]
        except KeyError as error:
            raise ValueError(
                "workflow branch report preferred node is missing"
            ) from error


@dataclass(frozen=True, slots=True)
class ProcessResult(TransformationResult):
    """Returned structure and observations, including exhausted workflow search.

    Parameters
    ----------
    structure : ProteinStructure
        Selected retained structure, which may still have unresolved defects.
    repairs : tuple[RepairEvent, ...]
        Repair events belonging to the selected carrier.
    issues : tuple[ValidationIssue, ...]
        Transformation and final diagnostics, including search exhaustion.
    analyses : AnalysisBundle or None
        Requested analysis output.
    requested_goal_report : RequestedGoalReport or None
        Observed requested-goal outcomes.
    branch_report : WorkflowBranchReport or None
        Retained candidate selection and search termination evidence.

    Notes
    -----
    Native history includes observed FF returns from sibling batches along the
    selected ancestry, even when their coordinates were rejected. Returns from
    other descendant subtrees are excluded; it is not a workflow-wide execution log.
    """

    analyses: AnalysisBundle | None = None
    requested_goal_report: RequestedGoalReport | None = None
    branch_report: WorkflowBranchReport | None = None

    @property
    def relaxation_executions(self) -> tuple[ContinuousRelaxationExecution, ...]:
        """Return native calls from sibling batches along the selected ancestry.

        Returns
        -------
        tuple[ContinuousRelaxationExecution, ...]
            Selected-ancestry batch history, including unselected siblings, or
            no evidence when no branch report is attached.
            Empty evidence does not prove that no native work ran.
        """

        return (
            ()
            if self.branch_report is None
            else self.branch_report.preferred_outcome().relaxation_executions
        )

    @classmethod
    def from_transformation_result(
        cls,
        result: TransformationResult,
        *,
        analyses: AnalysisBundle | None = None,
        requested_goal_report: RequestedGoalReport | None = None,
        branch_report: WorkflowBranchReport | None = None,
    ) -> "ProcessResult":
        """Build a public workflow result from a transformer-owned carrier.

        Parameters
        ----------
        result : TransformationResult
            Selected structure, repairs, diagnostics and native execution evidence.
        analyses : AnalysisBundle or None
            Requested analysis output.
        requested_goal_report : RequestedGoalReport or None
            Observed goal outcomes for the selected structure.
        branch_report : WorkflowBranchReport or None
            Retained candidate observations and search termination evidence.

        Returns
        -------
        ProcessResult
            Carrier preserving the selected result's transformation evidence.

        Raises
        ------
        ValueError
            Known native returns are missing from the selected branch history.
        """

        observed = (
            result.relaxation_executions
            if isinstance(result, ProcessResult)
            else RefinementTransformationResult.observed_executions(result)
        )
        if observed and branch_report is None:
            raise ValueError(
                "workflow report must retain known native execution history"
            )
        if branch_report is not None:
            branch_report.require_selected_native_history(observed)

        return cls(
            structure=result.structure,
            repairs=result.repairs,
            issues=result.issues,
            analyses=analyses,
            requested_goal_report=requested_goal_report,
            branch_report=branch_report,
        )

    def requested_goal_completion_verdict(
        self,
    ) -> RequestedGoalCompletionVerdict:
        """Return the top-level requested-goal completion verdict."""

        if self.requested_goal_report is None:
            return RequestedGoalCompletionVerdict.NOT_REQUESTED

        return self.requested_goal_report.completion_verdict()

    def requested_goals_fully_satisfied(self) -> bool | None:
        """Return whether the requested-goal report is fully satisfied."""

        completion_verdict = self.requested_goal_completion_verdict()
        if completion_verdict is RequestedGoalCompletionVerdict.NOT_REQUESTED:
            return None

        return completion_verdict is RequestedGoalCompletionVerdict.ACHIEVED

    def with_appended_issues(
        self,
        issues: tuple[ValidationIssue, ...],
    ) -> Self:
        """Return a copy with additional validation issues appended."""

        if not issues:
            return self

        return replace(self, issues=self.issues + tuple(issues))

    def with_workflow_reporting(
        self,
        *,
        requested_goal_report: RequestedGoalReport,
        branch_report: WorkflowBranchReport,
    ) -> Self:
        """Return a copy with workflow requested-goal and branch reports.

        Parameters
        ----------
        requested_goal_report : RequestedGoalReport
            Observed requested-goal outcomes.
        branch_report : WorkflowBranchReport
            Retained candidate selection and search termination evidence.

        Returns
        -------
        Self
            Reported result with unchanged structure and transformation evidence.

        Raises
        ------
        ValueError
            The replacement report discards known native invocation facts.
        """

        branch_report.require_selected_native_history(self.relaxation_executions)
        return replace(
            self,
            requested_goal_report=requested_goal_report,
            branch_report=branch_report,
        )

    def with_analyses(
        self,
        analyses: AnalysisBundle | None,
    ) -> Self:
        """Return a copy with structured analysis outputs attached."""

        return replace(self, analyses=analyses)

    def with_structure(self, structure: ProteinStructure) -> Self:
        """Return a copy with an updated structure."""

        return replace(self, structure=structure)


__all__ = [
    "AnalysisBundle",
    "ProcessResult",
    "RamachandranAnalysis",
    "RamachandranCategory",
    "RamachandranPoint",
    "RequestedGoalCompletionVerdict",
    "RequestedGoalOutcome",
    "RequestedGoalReport",
    "RequestedGoalStatus",
    "SecondaryStructureAnalysis",
    "SecondaryStructureAssignment",
    "WorkflowPhaseOutcome",
    "WorkflowPhaseReport",
    "WorkflowPhaseStatus",
    "WorkflowBranchQualityScore",
    "WorkflowBranchOutcome",
    "WorkflowBranchStatus",
    "WorkflowBranchReport",
]
