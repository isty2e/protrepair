"""Terminal workflow result finalization."""

from dataclasses import replace

from protrepair.analysis.kinds import AnalysisKind
from protrepair.analysis.runtime import build_analysis_bundle
from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.parser_readability import (
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.diagnostics.parser_topology import (
    ambiguous_disulfide_parser_witness_blocker_issues,
)
from protrepair.io.pdb_projection import prepare_rdkit_no_conect_pdb_block_projector
from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.contracts.planning import WorkflowPlanningContext
from protrepair.workflow.contracts.request import RequestedGoalSet, WorkflowGoal
from protrepair.workflow.contracts.result import (
    ProcessResult,
    WorkflowTerminalBranchReport,
)
from protrepair.workflow.engine.reporting import evaluate_terminal_branch_outcome
from protrepair.workflow.engine.runtime import WorkflowTerminalBranch


def finalize_workflow_result(
    *,
    terminal_branches: tuple[WorkflowTerminalBranch, ...],
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary,
    initially_satisfied_requested_goals: tuple[WorkflowGoal, ...],
    requested_analyses: frozenset[AnalysisKind],
    preliminary_issues: tuple[ValidationIssue, ...] = (),
) -> ProcessResult:
    """Select the preferred terminal branch and attach terminal artifacts."""

    terminal_branch_report = _terminal_branch_report(
        terminal_branches=terminal_branches,
        requested_goals=requested_goals,
        planning_context=planning_context,
        component_library=component_library,
        initially_satisfied_requested_goals=initially_satisfied_requested_goals,
    )
    preferred_terminal_branch = terminal_branch_report.require_preferred_value(
        {branch.node_id: branch for branch in terminal_branches}
    )
    result = ProcessResult.from_transformation_result(
        preferred_terminal_branch.result,
        requested_goal_report=(
            terminal_branch_report.preferred_outcome().requested_goal_report
        ),
        terminal_branch_report=terminal_branch_report,
    )
    parser_readability_issues = _final_parser_readability_issues(
        result.structure,
        component_library=component_library,
    )
    if parser_readability_issues:
        result = result.with_appended_issues(parser_readability_issues)
    if preliminary_issues:
        result = result.with_appended_issues(preliminary_issues)
    return _attach_requested_analyses(
        result,
        requested_analyses=requested_analyses,
    )


def _terminal_branch_report(
    *,
    terminal_branches: tuple[WorkflowTerminalBranch, ...],
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary,
    initially_satisfied_requested_goals: tuple[WorkflowGoal, ...],
) -> WorkflowTerminalBranchReport:
    """Build the terminal branch preference report for runtime branches."""

    return WorkflowTerminalBranchReport.from_outcomes(
        tuple(
            evaluate_terminal_branch_outcome(
                node_id=branch.node_id,
                result=branch.result,
                requested_goals=requested_goals.goals,
                planning_context=planning_context,
                component_library=component_library,
                unsupported_requested_goals=(
                    branch.planning_outcome.unsupported_requested_goals
                ),
                blocked_requested_goal_blockers=(
                    branch.planning_outcome.blocked_requested_goal_blockers()
                ),
                already_satisfied_requested_goals=(
                    initially_satisfied_requested_goals
                ),
            )
            for branch in terminal_branches
        ),
        branch_preference_policy=planning_context.branch_preference_policy,
    )


def _attach_requested_analyses(
    result: ProcessResult,
    *,
    requested_analyses: frozenset[AnalysisKind],
) -> ProcessResult:
    """Attach structured analyses to one completed workflow result."""

    if not requested_analyses:
        return result

    return result.with_analyses(
        build_analysis_bundle(
            result.structure,
            requested_analyses=requested_analyses,
        )
    )


def _final_parser_readability_issues(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
) -> tuple[ValidationIssue, ...]:
    """Return final parser-readability issues plus topology-specific blockers."""

    pdb_block_projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    parser_probe = probe_rdkit_no_conect_parser_readability(
        structure,
        component_library=component_library,
        pdb_block_projector=pdb_block_projector,
    )
    parser_readability_issues = parser_probe.issues()
    if not parser_readability_issues:
        return ()

    parser_witness_clusters = parser_probe.extra_proximity_bond_clusters()
    ambiguous_disulfide_blockers = ambiguous_disulfide_parser_witness_blocker_issues(
        structure,
        clusters=parser_witness_clusters,
    )
    return (
        _linked_parser_readability_blocker_issues(
            parser_readability_issues,
            ambiguous_disulfide_blockers=ambiguous_disulfide_blockers,
        )
        + ambiguous_disulfide_blockers
    )


def _linked_parser_readability_blocker_issues(
    parser_readability_issues: tuple[ValidationIssue, ...],
    *,
    ambiguous_disulfide_blockers: tuple[ValidationIssue, ...],
) -> tuple[ValidationIssue, ...]:
    """Return parser-readability issues annotated with terminal blockers."""

    if not ambiguous_disulfide_blockers:
        return parser_readability_issues

    return tuple(
        (
            replace(
                issue,
                message=(
                    f"{issue.message}; parser-readability repair is blocked by "
                    "ambiguous disulfide topology, so ordinary parser-witness local "
                    "FF repair was not proposed"
                ),
            )
            if _issue_overlaps_any_blocker(
                issue,
                blockers=ambiguous_disulfide_blockers,
            )
            else issue
        )
        for issue in parser_readability_issues
    )


def _issue_overlaps_any_blocker(
    issue: ValidationIssue,
    *,
    blockers: tuple[ValidationIssue, ...],
) -> bool:
    """Return whether one issue shares structure/residue scope with blockers."""

    if not issue.scope.residue_ids:
        return True

    issue_residue_ids = frozenset(issue.scope.residue_ids)
    return any(
        not blocker.scope.residue_ids
        or bool(issue_residue_ids.intersection(blocker.scope.residue_ids))
        for blocker in blockers
    )
