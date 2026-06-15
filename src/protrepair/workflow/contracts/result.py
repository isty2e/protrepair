"""Structured processing result contracts for the redesigned ProtRepair package."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import Self

from protrepair.analysis.results import (
    AnalysisBundle,
    RamachandranAnalysis,
    RamachandranPoint,
    SecondaryStructureAnalysis,
    SecondaryStructureAssignment,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.scope import Scope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
    WorkflowPlanningPhase,
)
from protrepair.workflow.contracts.request import WorkflowGoal, WorkflowGoalStateValue

if TYPE_CHECKING:
    from protrepair.transformer.refinement.speculative_planning import (
        SpeculativePlanningNodeId,
    )

PreferredValueT = TypeVar("PreferredValueT")

__all__ = [
    "AnalysisBundle",
    "ProcessResult",
    "RamachandranAnalysis",
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
    "WorkflowTerminalBranchOutcome",
    "WorkflowTerminalBranchReport",
]

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
        object.__setattr__(self, "blocking_scopes", tuple(self.blocking_scopes))


@dataclass(frozen=True, slots=True, init=False)
class WorkflowPhaseReport:
    """Structured report over explicit workflow phases for one branch."""

    _outcomes: tuple[WorkflowPhaseOutcome, ...] = field(repr=False)

    def __init__(self, outcomes: tuple[WorkflowPhaseOutcome, ...]) -> None:
        object.__setattr__(self, "_outcomes", tuple(outcomes))

    @property
    def outcomes(self) -> tuple[WorkflowPhaseOutcome, ...]:
        """Return phase outcomes for this already-evaluated report."""

        return self._outcomes

    def outcome_for(
        self,
        phase: WorkflowPlanningPhase,
    ) -> WorkflowPhaseOutcome | None:
        """Return the outcome for one specific workflow phase."""

        for outcome in self.outcomes:
            if outcome.phase is phase:
                return outcome

        return None


@dataclass(frozen=True, slots=True)
class RequestedGoalReport:
    """Structured report over requested-goal satisfaction for one run."""

    outcomes: tuple[RequestedGoalOutcome, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))

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
            ligand_aware_worst_overlap_angstrom=(
                ligand_aware_worst_overlap_angstrom
            ),
            ligand_aware_total_overlap_angstrom=(
                ligand_aware_total_overlap_angstrom
            ),
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
class WorkflowTerminalBranchOutcome:
    """One terminal workflow branch plus its requested-goal evaluation."""

    node_id: "SpeculativePlanningNodeId"
    requested_goal_report: RequestedGoalReport
    phase_report: WorkflowPhaseReport
    branch_quality_score: WorkflowBranchQualityScore
    error_count: int
    warning_count: int
    issue_count: int

    def preference_key(
        self,
        policy: WorkflowBranchPreferencePolicy,
    ) -> BranchPreferenceKey:
        """Return the configured preference key for one terminal branch."""

        return self.branch_quality_score.order_key(
            policy,
            tie_breaker=self.node_id.value,
        )


@dataclass(frozen=True, slots=True)
class WorkflowTerminalBranchReport:
    """Branch-aware report over terminal workflow frontier outcomes."""

    preferred_node_id: "SpeculativePlanningNodeId"
    outcomes: tuple[WorkflowTerminalBranchOutcome, ...]
    branch_preference_policy: WorkflowBranchPreferencePolicy = field(
        default_factory=WorkflowBranchPreferencePolicy
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))

    @classmethod
    def from_outcomes(
        cls,
        outcomes: tuple[WorkflowTerminalBranchOutcome, ...],
        *,
        branch_preference_policy: WorkflowBranchPreferencePolicy | None = None,
    ) -> "WorkflowTerminalBranchReport":
        """Build one terminal-branch report and choose its preferred outcome."""

        if not outcomes:
            raise ValueError(
                "workflow terminal branch report requires at least one outcome"
            )

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
        )

    def preferred_outcome(self) -> WorkflowTerminalBranchOutcome:
        """Return the preferred terminal branch outcome."""

        for outcome in self.outcomes:
            if outcome.node_id == self.preferred_node_id:
                return outcome

        raise ValueError("workflow terminal branch report preferred node is missing")

    def require_preferred_value(
        self,
        values_by_node_id: Mapping[
            "SpeculativePlanningNodeId",
            PreferredValueT,
        ],
    ) -> PreferredValueT:
        """Return the value mapped to the preferred terminal branch."""

        try:
            return values_by_node_id[self.preferred_node_id]
        except KeyError as error:
            raise ValueError(
                "workflow terminal branch report preferred node is missing"
            ) from error


@dataclass(frozen=True, slots=True)
class ProcessResult(TransformationResult):
    """Public structured result of a completed workflow processing run."""

    analyses: AnalysisBundle | None = None
    requested_goal_report: RequestedGoalReport | None = None
    terminal_branch_report: WorkflowTerminalBranchReport | None = None

    @classmethod
    def from_transformation_result(
        cls,
        result: TransformationResult,
        *,
        analyses: AnalysisBundle | None = None,
        requested_goal_report: RequestedGoalReport | None = None,
        terminal_branch_report: WorkflowTerminalBranchReport | None = None,
    ) -> "ProcessResult":
        """Build a public workflow result from a transformer-owned carrier."""

        return cls(
            structure=result.structure,
            repairs=result.repairs,
            issues=result.issues,
            analyses=analyses,
            requested_goal_report=requested_goal_report,
            terminal_branch_report=terminal_branch_report,
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

        return type(self)(
            structure=self.structure,
            repairs=self.repairs,
            issues=self.issues + tuple(issues),
            analyses=self.analyses,
            requested_goal_report=self.requested_goal_report,
            terminal_branch_report=self.terminal_branch_report,
        )

    def with_workflow_reporting(
        self,
        *,
        requested_goal_report: RequestedGoalReport,
        terminal_branch_report: WorkflowTerminalBranchReport,
    ) -> Self:
        """Return a copy with workflow requested-goal and branch reports."""

        return type(self)(
            structure=self.structure,
            repairs=self.repairs,
            issues=self.issues,
            analyses=self.analyses,
            requested_goal_report=requested_goal_report,
            terminal_branch_report=terminal_branch_report,
        )

    def with_analyses(
        self,
        analyses: AnalysisBundle | None,
    ) -> Self:
        """Return a copy with structured analysis outputs attached."""

        return type(self)(
            structure=self.structure,
            repairs=self.repairs,
            issues=self.issues,
            analyses=analyses,
            requested_goal_report=self.requested_goal_report,
            terminal_branch_report=self.terminal_branch_report,
        )

    def with_structure(self, structure: ProteinStructure) -> Self:
        """Return a copy with an updated structure."""

        return type(self)(
            structure=structure,
            repairs=self.repairs,
            issues=self.issues,
            analyses=self.analyses,
            requested_goal_report=self.requested_goal_report,
            terminal_branch_report=self.terminal_branch_report,
        )
