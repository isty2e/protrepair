"""Unit tests for workflow branch quality score projection."""

from protrepair.transformer.refinement.speculative_planning import (
    SpeculativePlanningNodeId,
)
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
)
from protrepair.workflow.contracts.result import (
    RequestedGoalReport,
    WorkflowBranchQualityScore,
    WorkflowPhaseReport,
    WorkflowTerminalBranchOutcome,
    WorkflowTerminalBranchReport,
)


def test_terminal_branch_selection_uses_parser_quality_before_tie_breaker() -> None:
    """Default branch scoring should prefer parser-clean branches."""

    worse_branch = _terminal_outcome(
        node_id=0,
        score=WorkflowBranchQualityScore(parser_incompatible=1),
    )
    better_branch = _terminal_outcome(
        node_id=1,
        score=WorkflowBranchQualityScore(parser_incompatible=0),
    )

    report = WorkflowTerminalBranchReport.from_outcomes(
        (worse_branch, better_branch)
    )

    assert report.preferred_node_id == SpeculativePlanningNodeId(1)


def test_terminal_branch_selection_policy_can_reorder_axes() -> None:
    """Branch quality policy should tune the sortable score projection."""

    earlier_branch = _terminal_outcome(
        node_id=0,
        score=WorkflowBranchQualityScore(parser_incompatible=1, search_depth=0),
    )
    later_branch = _terminal_outcome(
        node_id=1,
        score=WorkflowBranchQualityScore(parser_incompatible=0, search_depth=1),
    )
    search_first_policy = WorkflowBranchPreferencePolicy(
        axes=(
            WorkflowBranchQualityAxis.SEARCH_DEPTH,
            WorkflowBranchQualityAxis.PARSER_COMPATIBILITY,
        )
    )

    report = WorkflowTerminalBranchReport.from_outcomes(
        (earlier_branch, later_branch),
        branch_preference_policy=search_first_policy,
    )

    assert report.preferred_node_id == SpeculativePlanningNodeId(0)


def _terminal_outcome(
    *,
    node_id: int,
    score: WorkflowBranchQualityScore,
) -> WorkflowTerminalBranchOutcome:
    """Return one minimal terminal outcome for branch preference tests."""

    return WorkflowTerminalBranchOutcome(
        node_id=SpeculativePlanningNodeId(node_id),
        requested_goal_report=RequestedGoalReport(()),
        phase_report=WorkflowPhaseReport(()),
        branch_quality_score=score,
        error_count=score.error_count,
        warning_count=score.warning_count,
        issue_count=score.issue_count,
    )
