"""Behavior of retained candidates when workflow search reaches its node limit."""

import json
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.request_builders import whole_structure_requested_goals
from tests.support.workflow_proposals import workflow_proposals

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.events import EventScope, RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.state import (
    HydrogenCoverageState,
    StructureProjectionStateFacts,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.hydrogen.protonation import (
    DisabledHistidineProtonationRequest,
)
from protrepair.transformer.continuous.binding import ManualContinuousRelaxationBinding
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.continuous.shared import apply_position_updates
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.result import RefinementTransformationResult
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.refinement.speculative_planning import (
    SpeculativePlanningNodeId,
    SpeculativeStopReason,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
    WorkflowPlanningContext,
)
from protrepair.workflow.contracts.policies import OrphanFragmentPolicy
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowTransformRequests,
)
from protrepair.workflow.contracts.result import (
    RequestedGoalReport,
    RequestedGoalStatus,
    WorkflowBranchOutcome,
    WorkflowBranchQualityScore,
    WorkflowBranchReport,
    WorkflowBranchStatus,
    WorkflowPhaseReport,
)
from protrepair.workflow.engine.finalization import finalize_workflow_result
from protrepair.workflow.engine.runtime import (
    WorkflowBranch,
    WorkflowBranchEvaluator,
    WorkflowRuntimeState,
    execute_iterative_workflow,
)
from protrepair.workflow.planning.action.proposals import WorkflowProposalSelectionMode
from protrepair.workflow.planning.action.registry import WorkflowStateAction
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    WorkflowPlanningOutcome,
)
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)


def test_zero_action_exhaustion_returns_unmodified_incomplete_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    execute = Mock(side_effect=AssertionError("no action fits a one-node trace"))
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer", execute
    )
    result = process_structure(
        source,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=1),
    )

    assert result.structure.geometry == source.geometry
    assert result.structure.constitution == source.constitution
    assert result.structure.topology == source.topology
    assert result.repairs == ()
    assert result.relaxation_executions == ()
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[-1].status is RequestedGoalStatus.UNMET
    assert result.requested_goals_fully_satisfied() is False
    assert result.branch_report is not None
    assert result.branch_report.search_exhausted()
    assert (
        result.branch_report.preferred_outcome().status is WorkflowBranchStatus.FRONTIER
    )
    assert result.branch_report.preferred_node_id == SpeculativePlanningNodeId(0)
    (issue,) = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.WORKFLOW_SEARCH_EXHAUSTED
    )
    assert issue.severity is IssueSeverity.WARNING
    execute.assert_not_called()

    payload = json.loads(json.dumps(asdict(result.branch_report)))
    assert payload["stop_reason"] == "iteration_limit_reached"
    assert payload["outcomes"][0]["status"] == "frontier"
    assert (
        payload["outcomes"][0]["requested_goal_report"]["outcomes"][-1]["status"]
        == "unmet"
    )
    written = write_structure_string(result.structure, file_format=FileFormat.MMCIF)
    restored = read_structure_string(written, file_format=FileFormat.MMCIF)
    assert restored.constitution == result.structure.constitution
    assert tuple(
        (bond.endpoint_pair(), bond.order, bond.aromatic, bond.relationship_type)
        for bond in restored.topology.bonds
    ) == tuple(
        (bond.endpoint_pair(), bond.order, bond.aromatic, bond.relationship_type)
        for bond in result.structure.topology.bonds
    )


@pytest.mark.parametrize(
    "terminal_labels", ((), ("first",), ("last",), ("first", "last"))
)
@pytest.mark.parametrize("native_returned", (False, True))
def test_exhaustion_selects_retained_branch_not_last_action_output(
    monkeypatch: pytest.MonkeyPatch,
    terminal_labels: tuple[str, ...],
    native_returned: bool,
) -> None:
    source = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    residue_ids = tuple(residue.residue_id for residue in source.iter_residue_sites())
    first = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=residue_ids[:1])
    )
    second = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=residue_ids)
    )
    labels = ("first", "last")
    outputs: list[RefinementTransformationResult] = []
    planned_labels: list[str | None] = []

    def plan(current_structure: ProteinStructure, **kwargs) -> WorkflowPlanningOutcome:
        del kwargs
        label = current_structure.provenance.ingress.source_name
        planned_labels.append(label)
        terminal = label in terminal_labels
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            proposals=workflow_proposals(() if terminal else (first, second)),
        )

    def execute(
        result: TransformationResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> TransformationResult:
        del transformer, execution_context
        output = replace(
            RefinementTransformationResult.with_observed_executions(result, ()),
            repairs=(
                RepairEvent(
                    kind=RepairEventKind.LOCAL_REFINEMENT_APPLIED,
                    scope=EventScope.for_structure(),
                    details=labels[len(outputs)],
                ),
            ),
            relaxation_executions=(
                ContinuousRelaxationExecution(
                    settings=ContinuousRelaxationSettings(
                        profile=ContinuousRelaxationProfile.RDKIT_UFF
                    ),
                    movable_scope=AtomSetScope(
                        (
                            source.constitution.atom_ref_at(
                                next(iter(source.geometry.iter_positions()))[0]
                            ),
                        )
                    ),
                    minimizer_status=1,
                    backend_version="test-return",
                ),
            )
            if native_returned
            else (),
            structure=result.structure.with_provenance(
                replace(
                    result.structure.provenance,
                    ingress=replace(
                        result.structure.provenance.ingress,
                        source_name=labels[len(outputs)],
                    ),
                ),
            ),
        )
        outputs.append(output)
        return output

    assessed_plans = []
    assess = WorkflowBranchEvaluator.branch_outcome

    def observed_assessment(self, **kwargs):
        assessed_plans.append(kwargs.get("planning_outcome"))
        return assess(self, **kwargs)

    monkeypatch.setattr(WorkflowBranchEvaluator, "branch_outcome", observed_assessment)
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions", plan
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer", execute
    )
    context = WorkflowPlanningContext(max_speculative_nodes=3)
    runtime = execute_iterative_workflow(
        source,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_default_component_library(),
        planning_context=context,
        reference_structure=None,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
        histidine_protonation=DisabledHistidineProtonationRequest(),
    )
    assert len(outputs) == 2
    assert len(runtime.branches) == 2
    observed = tuple(
        execution for output in outputs for execution in output.relaxation_executions
    )
    assert all(
        branch.outcome.relaxation_executions == observed for branch in runtime.branches
    )
    exhausted = not all(label in terminal_labels for label in labels)
    assert runtime.stop_reason is (
        SpeculativeStopReason.ITERATION_LIMIT_REACHED
        if exhausted
        else SpeculativeStopReason.NO_PROPOSALS_AVAILABLE
    )
    assert tuple(branch.node_id.value for branch in runtime.branches) == (1, 2)
    assert runtime.branches[0].outcome.status is (
        WorkflowBranchStatus.TERMINAL
        if "first" in terminal_labels
        else WorkflowBranchStatus.FRONTIER
    )
    assert runtime.branches[1].outcome.status is (
        WorkflowBranchStatus.TERMINAL
        if "last" in terminal_labels
        else WorkflowBranchStatus.FRONTIER
    )
    # Each retained state is assessed once under the complete sibling history.
    assert planned_labels.count("first") == (2 if native_returned else 1)
    assert planned_labels.count("last") == (2 if native_returned else 1)
    assert all(plan is not None for plan in assessed_plans)
    result = finalize_workflow_result(
        branches=runtime.branches,
        stop_reason=runtime.stop_reason,
        planning_context=context,
        component_library=build_default_component_library(),
        requested_analyses=frozenset(),
    )
    assert result.structure is outputs[0].structure
    assert result.branch_report is not None
    assert result.branch_report.search_exhausted() is exhausted
    assert (
        any(
            issue.kind is ValidationIssueKind.WORKFLOW_SEARCH_EXHAUSTED
            for issue in result.issues
        )
        is exhausted
    )
    assert result.branch_report.preferred_node_id.value == 1
    assert result.repairs == outputs[0].repairs
    assert result.relaxation_executions == tuple(
        execution for output in outputs for execution in output.relaxation_executions
    )
    if native_returned:
        assert result.relaxation_executions[0] is outputs[0].relaxation_executions[0]
        assert (
            result.relaxation_executions[0] is not outputs[-1].relaxation_executions[0]
        )


@pytest.mark.parametrize("readable", (False, True))
def test_exhaustion_preserves_final_parser_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    readable: bool,
) -> None:
    source = add_hydrogens(
        load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    ).structure
    if not readable:
        source = apply_position_updates(
            source,
            moved_positions={
                index: Vec3(0.3 * index.value, 0.0, 0.0)
                for index, _ in source.geometry.iter_positions()
            },
        )
    action = TerminalAugmentationTransformer(
        scope=ResidueSetScope(
            residue_ids=tuple(
                residue.residue_id for residue in source.iter_residue_sites()
            )
        )
    )

    def plan(current_structure: ProteinStructure, **kwargs) -> WorkflowPlanningOutcome:
        del kwargs
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            proposals=workflow_proposals((action,)),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions", plan
    )
    result = process_structure(
        source,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=1),
    )
    assert result.branch_report is not None and result.branch_report.search_exhausted()
    assert result.requested_goals_fully_satisfied() is True
    parser_issues = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.PARSER_READABILITY
    )
    assert bool(parser_issues) is not readable
    assert (
        result.branch_report.preferred_outcome().branch_quality_score.parser_incompatible
        == int(not readable)
    )


def _outcome(
    node_id: int, clash_count: int, status: WorkflowBranchStatus
) -> WorkflowBranchOutcome:
    return WorkflowBranchOutcome(
        node_id=SpeculativePlanningNodeId(node_id),
        requested_goal_report=RequestedGoalReport(()),
        phase_report=WorkflowPhaseReport(()),
        branch_quality_score=WorkflowBranchQualityScore(
            protein_self_clash_count=clash_count
        ),
        error_count=0,
        warning_count=0,
        issue_count=0,
        status=status,
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_mixed_candidates_use_configured_quality_not_terminal_preference(
    reverse: bool,
) -> None:
    terminal = _outcome(1, 8, WorkflowBranchStatus.TERMINAL)
    frontier = _outcome(2, 1, WorkflowBranchStatus.FRONTIER)
    outcomes = (frontier, terminal) if reverse else (terminal, frontier)
    report = WorkflowBranchReport.from_outcomes(
        outcomes, stop_reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED
    )
    assert report.preferred_outcome() is frontier
    search_first = WorkflowBranchPreferencePolicy(
        axes=(WorkflowBranchQualityAxis.SEARCH_DEPTH,)
    )
    tied = replace(
        frontier,
        branch_quality_score=replace(frontier.branch_quality_score, search_depth=10),
    )
    report = WorkflowBranchReport.from_outcomes(
        (terminal, tied),
        branch_preference_policy=search_first,
        stop_reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED,
    )
    assert report.preferred_outcome() is terminal


@pytest.mark.parametrize("reverse", (False, True))
def test_incomplete_frontier_can_beat_hydrogenated_terminal_with_more_clashes(reverse):
    source = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    prepared = add_hydrogens(source)
    library = build_default_component_library()
    context = WorkflowPlanningContext()
    evaluator = WorkflowBranchEvaluator(
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        component_library=library,
        planning_context=context,
        already_satisfied_requested_goals=(),
    )
    branches = []
    for node_id, candidate, status in (
        (1, prepared, WorkflowBranchStatus.TERMINAL),
        (2, TransformationResult(source, (), ()), WorkflowBranchStatus.FRONTIER),
    ):
        branches.append(
            WorkflowBranch(
                result=candidate,
                outcome=evaluator.branch_outcome(
                    node_id=SpeculativePlanningNodeId(node_id),
                    state=WorkflowRuntimeState(candidate, WorkflowPlannerMemory()),
                    result=candidate,
                    status=status,
                ),
            )
        )
    terminal, frontier = branches
    assert terminal.outcome.requested_goal_report.unmet_count() == 0
    assert frontier.outcome.requested_goal_report.unmet_count() == 1
    assert (
        frontier.outcome.branch_quality_score.protein_self_clash_count
        < terminal.outcome.branch_quality_score.protein_self_clash_count
    )
    result = finalize_workflow_result(
        branches=tuple(reversed(branches)) if reverse else tuple(branches),
        planning_context=context,
        component_library=library,
        requested_analyses=frozenset(),
        stop_reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED,
    )
    assert result.structure is source
    assert not result.requested_goals_fully_satisfied()
    assert result.branch_report is not None
    assert result.branch_report.preferred_outcome() is frontier.outcome
    assert any(
        issue.kind is ValidationIssueKind.WORKFLOW_SEARCH_EXHAUSTED
        for issue in result.issues
    )


def test_report_rejects_absent_and_conflicting_candidate_evidence() -> None:
    frontier = _outcome(1, 1, WorkflowBranchStatus.FRONTIER)
    with pytest.raises(ValueError, match="at least one"):
        WorkflowBranchReport.from_outcomes(())
    with pytest.raises(ValueError, match="frontier outcomes require"):
        WorkflowBranchReport.from_outcomes((frontier,))
    with pytest.raises(ValueError, match="repeat node IDs"):
        WorkflowBranchReport.from_outcomes(
            (frontier, frontier),
            stop_reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED,
        )
    with pytest.raises(ValueError, match="configured policy"):
        WorkflowBranchReport(
            preferred_node_id=SpeculativePlanningNodeId(2),
            outcomes=(frontier,),
            stop_reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED,
        )


def test_backend_failure_is_not_reclassified_as_search_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    execute = Mock(side_effect=RuntimeError("native backend failed"))
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer", execute
    )
    with pytest.raises(RuntimeError, match="native backend failed"):
        process_structure(
            source,
            requested_goals=whole_structure_requested_goals(
                HydrogenCoverageState.COMPLETE
            ),
            planning_context=WorkflowPlanningContext(max_speculative_nodes=2),
        )
    execute.assert_called_once()


def test_exhaustion_keeps_regression_retained_parent_and_native_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    atom_ref = source.constitution.atom_ref_at(
        next(iter(source.geometry.iter_positions()))[0]
    )
    execution = ContinuousRelaxationExecution(
        settings=ContinuousRelaxationSettings(
            profile=ContinuousRelaxationProfile.RDKIT_UFF
        ),
        movable_scope=AtomSetScope((atom_ref,)),
        minimizer_status=1,
        backend_version="test-return",
    )
    transformer = LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_atoms((atom_ref,)),
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationProfile.RDKIT_UFF
            ),
        )
    )
    memories = []

    def plan(
        current_structure: ProteinStructure, *, planner_memory, **kwargs
    ) -> WorkflowPlanningOutcome:
        del kwargs
        memories.append(planner_memory)
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            proposals=workflow_proposals(
                (transformer,),
                selection_mode=WorkflowProposalSelectionMode.QUALITY_ALTERNATIVE,
            ),
        )

    def execute(result: TransformationResult, **kwargs) -> TransformationResult:
        del kwargs
        return replace(
            RefinementTransformationResult.with_observed_executions(result, ()),
            issues=(
                ValidationIssue(
                    kind=ValidationIssueKind.INVALID_GEOMETRY,
                    severity=IssueSeverity.ERROR,
                    message="regressed child",
                ),
            ),
            relaxation_executions=(execution,),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions", plan
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer", execute
    )
    context = WorkflowPlanningContext(max_speculative_nodes=2)
    result = process_structure(source, planning_context=context)
    assert result.structure.geometry is source.geometry
    assert result.repairs == ()
    assert all(issue.message != "regressed child" for issue in result.issues)
    assert result.relaxation_executions == (execution,)
    assert result.relaxation_executions[0] is execution
    assert result.branch_report is not None and result.branch_report.search_exhausted()
    assert memories[-1].attempted_transformers == (transformer,)
    assert memories[-1].adopted_transformers == ()
