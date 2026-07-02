"""Frontier runtime for workflow planning execution."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.refinement.speculative_planning import (
    EvaluatedSpeculativeProposal,
    SpeculativeAdoptedChild,
    SpeculativeAdoptionDecision,
    SpeculativeExecution,
    SpeculativePlanningNodeId,
    SpeculativePlanningTrace,
    SpeculativeStopReason,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
    WorkflowPlanningContext,
)
from protrepair.workflow.contracts.policies import OrphanFragmentPolicy
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowGoal,
    WorkflowTransformRequests,
)
from protrepair.workflow.engine.parser_witness_repair import (
    execute_bounded_parser_witness_repair_loop,
)
from protrepair.workflow.engine.reporting import (
    evaluate_requested_goal_report,
    evaluate_workflow_branch_quality_score,
)
from protrepair.workflow.planning.action.registry import WorkflowStateAction
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    WorkflowPlanningOutcome,
    plan_workflow_actions,
)


@dataclass(frozen=True, slots=True)
class WorkflowRuntimeState:
    """Workflow runtime state stored at one speculative planning node."""

    result: TransformationResult
    planner_memory: WorkflowPlannerMemory


@dataclass(frozen=True, slots=True)
class WorkflowTerminalBranch:
    """One terminal workflow branch captured before final branch selection."""

    node_id: SpeculativePlanningNodeId
    result: TransformationResult
    planning_outcome: WorkflowPlanningOutcome


@dataclass(frozen=True, slots=True)
class WorkflowRuntimeResult:
    """Terminal branches plus root-requested-goal facts from one runtime run."""

    terminal_branches: tuple[WorkflowTerminalBranch, ...]
    initially_satisfied_requested_goals: tuple[WorkflowGoal, ...]


def execute_iterative_workflow(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    transform_requests: WorkflowTransformRequests,
    component_library: ComponentLibrary,
    planning_context: WorkflowPlanningContext,
    reference_structure: ProteinStructure | None,
    orphan_fragment_policy: OrphanFragmentPolicy,
    protonate_histidines: bool,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence, ...
    ] = (),
    initial_issues: tuple[ValidationIssue, ...] = (),
) -> WorkflowRuntimeResult:
    """Execute workflow planning through a frontier-based iterative driver."""

    trace = SpeculativePlanningTrace.root(
        initial_state=WorkflowRuntimeState(
            result=TransformationResult(
                structure=structure,
                repairs=(),
                issues=initial_issues,
            ),
            planner_memory=WorkflowPlannerMemory(),
        )
    )
    terminal_branches: list[WorkflowTerminalBranch] = []
    initial_already_satisfied_requested_goals: tuple[WorkflowGoal, ...] = ()
    while True:
        if not trace.frontier.active_nodes:
            trace = trace.stop(reason=SpeculativeStopReason.NO_PROPOSALS_AVAILABLE)
            break

        active_node = trace.frontier.select_next_active_node()
        runtime_state = active_node.state
        planning_outcome = plan_workflow_actions(
            runtime_state.result.structure,
            requested_goals=requested_goals,
            transform_requests=transform_requests,
            component_library=component_library,
            planner_memory=runtime_state.planner_memory,
            planning_context=planning_context,
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
        )
        if not runtime_state.planner_memory.adopted_transformers:
            initial_already_satisfied_requested_goals = (
                planning_outcome.already_satisfied_requested_goals
            )
        if planning_outcome.is_terminal():
            terminal_branches.append(
                WorkflowTerminalBranch(
                    node_id=active_node.node_id,
                    result=runtime_state.result.with_appended_issues(
                        planning_outcome.blocker_issues(
                            runtime_state.result.structure,
                            component_library=component_library,
                        )
                    ),
                    planning_outcome=planning_outcome,
                )
            )
            trace = trace.expand_active_node(
                parent_node=active_node,
                adopted_children=(),
            )
            continue
        if len(trace.nodes) >= planning_context.max_speculative_nodes:
            trace = trace.stop(
                reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED
            )
            break

        adopted_children: list[
            SpeculativeAdoptedChild[
                WorkflowRuntimeState,
                WorkflowStateAction,
                TransformationResult,
                TransformationResult,
            ]
        ] = []
        execution_context = TransformerExecutionContext(
            component_library=component_library,
            original_structure=structure,
            orphan_fragment_policy=orphan_fragment_policy,
            reference_structure=reference_structure,
            protonate_histidines=protonate_histidines,
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
        )
        for transformer in planning_outcome.current_proposal_batch():
            execution_outcome = execute_workflow_transformer(
                runtime_state.result,
                transformer=transformer,
                execution_context=execution_context,
            )
            adopted_transformers: tuple[WorkflowStateAction, ...] = (transformer,)
            if transform_requests.repair_refinement is None and not isinstance(
                transformer, LocalRefinementTransformer
            ):
                parser_witness_repair = execute_bounded_parser_witness_repair_loop(
                    execution_outcome,
                    execution_context=execution_context,
                    execute_transformer=execute_workflow_transformer,
                )
                execution_outcome = parser_witness_repair.result
                adopted_transformers = (
                    adopted_transformers + parser_witness_repair.adopted_transformers
                )

            planner_memory = runtime_state.planner_memory
            for adopted_transformer in adopted_transformers:
                planner_memory = planner_memory.with_adopted_transformer(
                    adopted_transformer
                )

            adopted_children.append(
                SpeculativeAdoptedChild(
                    state=WorkflowRuntimeState(
                        result=execution_outcome,
                        planner_memory=planner_memory,
                    ),
                    adopted_decision=SpeculativeAdoptionDecision.adopt(
                        EvaluatedSpeculativeProposal(
                            execution=SpeculativeExecution(
                                proposal=transformer,
                                outcome=execution_outcome,
                            ),
                            evaluation=execution_outcome,
                        )
                    ),
                ),
            )
        adopted_children = list(
            _adopted_workflow_children(
                adopted_children=tuple(adopted_children),
                requested_goals=requested_goals,
                component_library=component_library,
                planning_context=planning_context,
                already_satisfied_requested_goals=(
                    initial_already_satisfied_requested_goals
                ),
            )
        )
        adopted_children = list(
            _workflow_children_with_regression_retention(
                current_branch_state=runtime_state,
                attempted_transformers=planning_outcome.current_proposal_batch(),
                transform_requests=transform_requests,
                retained_children=tuple(adopted_children),
                requested_goals=requested_goals,
                component_library=component_library,
                planning_context=planning_context,
                already_satisfied_requested_goals=(
                    initial_already_satisfied_requested_goals
                ),
            )
        )
        trace = trace.expand_active_node(
            parent_node=active_node,
            adopted_children=tuple(adopted_children),
        )

    if not terminal_branches:
        if trace.stop_reason is SpeculativeStopReason.ITERATION_LIMIT_REACHED:
            raise ValueError(
                "iterative workflow execution reached "
                f"max_speculative_nodes={planning_context.max_speculative_nodes} "
                "before any terminal branch"
            )
        raise ValueError(
            "iterative workflow execution requires at least one terminal branch"
        )

    return WorkflowRuntimeResult(
        terminal_branches=tuple(terminal_branches),
        initially_satisfied_requested_goals=initial_already_satisfied_requested_goals,
    )


def execute_workflow_transformer(
    result: TransformationResult,
    *,
    transformer: WorkflowStateAction,
    execution_context: TransformerExecutionContext,
) -> TransformationResult:
    """Execute one workflow transformer through transformer-owned behavior."""

    return transformer.execute(
        result,
        context=execution_context,
    )


def _adopted_workflow_children(
    *,
    adopted_children: tuple[
        SpeculativeAdoptedChild[
            WorkflowRuntimeState,
            WorkflowStateAction,
            TransformationResult,
            TransformationResult,
        ],
        ...,
    ],
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    planning_context: WorkflowPlanningContext,
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...],
) -> tuple[
    SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    ...,
]:
    """Return non-dominated sibling branches under the goal-first key."""

    if not adopted_children:
        return ()

    if len(adopted_children) == 1:
        return adopted_children

    child_keys = tuple(
        _workflow_child_quality_key(
            child,
            requested_goals=requested_goals,
            component_library=component_library,
            planning_context=planning_context,
            already_satisfied_requested_goals=already_satisfied_requested_goals,
        )
        for child in adopted_children
    )
    best_quality_key = min(preference_key for _, preference_key in child_keys)
    return tuple(
        child
        for child, preference_key in child_keys
        if preference_key == best_quality_key
    )


def _workflow_children_with_regression_retention(
    *,
    current_branch_state: WorkflowRuntimeState,
    attempted_transformers: tuple[WorkflowStateAction, ...],
    transform_requests: WorkflowTransformRequests,
    retained_children: tuple[
        SpeculativeAdoptedChild[
            WorkflowRuntimeState,
            WorkflowStateAction,
            TransformationResult,
            TransformationResult,
        ],
        ...,
    ],
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    planning_context: WorkflowPlanningContext,
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...],
) -> tuple[
    SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    ...,
]:
    """Retain the current branch as a no-op child when all children regress."""

    if not retained_children:
        return ()

    if not _proposal_batch_allows_regression_retention(
        attempted_transformers,
        transform_requests=transform_requests,
    ):
        return retained_children

    regression_policy = _branch_regression_preference_policy(planning_context)
    if regression_policy is None:
        return retained_children

    current_branch_quality_key = _workflow_result_quality_key(
        current_branch_state.result,
        planner_memory=current_branch_state.planner_memory,
        requested_goals=requested_goals,
        component_library=component_library,
        planning_context=planning_context,
        branch_preference_policy=regression_policy,
        already_satisfied_requested_goals=already_satisfied_requested_goals,
    )
    child_quality_keys = tuple(
        _workflow_result_quality_key(
            child.state.result,
            planner_memory=child.state.planner_memory,
            requested_goals=requested_goals,
            component_library=component_library,
            planning_context=planning_context,
            branch_preference_policy=regression_policy,
            already_satisfied_requested_goals=already_satisfied_requested_goals,
        )
        for child in retained_children
    )
    if min(child_quality_keys) <= current_branch_quality_key:
        return retained_children

    planner_memory = current_branch_state.planner_memory
    for transformer in attempted_transformers:
        planner_memory = planner_memory.with_attempted_transformer(transformer)

    if planner_memory == current_branch_state.planner_memory:
        return retained_children

    return retained_children + (
        SpeculativeAdoptedChild(
            state=WorkflowRuntimeState(
                result=current_branch_state.result,
                planner_memory=planner_memory,
            ),
            adopted_decision=SpeculativeAdoptionDecision.reject(
                reason=(
                    "current branch retained because all children "
                    "regressed under branch quality policy"
                ),
            ),
        ),
    )


def _proposal_batch_allows_regression_retention(
    attempted_transformers: tuple[WorkflowStateAction, ...],
    *,
    transform_requests: WorkflowTransformRequests,
) -> bool:
    """Return whether a proposal batch is exploratory enough for retention."""

    if not attempted_transformers:
        return False

    for transformer in attempted_transformers:
        if isinstance(transformer, LocalRefinementTransformer):
            if transform_requests.repair_refinement is not None:
                return False
            continue

        if isinstance(transformer, CommittedPackingTransformer):
            if transform_requests.committed_sidechain_packing is not None:
                return False
            continue

        return False

    return True


def _branch_regression_preference_policy(
    planning_context: WorkflowPlanningContext,
) -> WorkflowBranchPreferencePolicy | None:
    """Return the branch-quality policy used for parent-child regression checks."""

    axes = tuple(
        axis
        for axis in planning_context.branch_preference_policy.axes
        if axis is not WorkflowBranchQualityAxis.SEARCH_DEPTH
    )
    if not axes:
        return None

    return WorkflowBranchPreferencePolicy(axes=axes)


def _workflow_child_quality_key(
    child: SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    *,
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    planning_context: WorkflowPlanningContext,
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...],
) -> tuple[
    SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    tuple[int | float, ...],
]:
    """Return one adopted child paired with its configured branch score key."""

    return (
        child,
        _workflow_result_quality_key(
            child.state.result,
            planner_memory=child.state.planner_memory,
            requested_goals=requested_goals,
            component_library=component_library,
            planning_context=planning_context,
            branch_preference_policy=planning_context.branch_preference_policy,
            already_satisfied_requested_goals=already_satisfied_requested_goals,
        ),
    )


def _workflow_result_quality_key(
    result: TransformationResult,
    *,
    planner_memory: WorkflowPlannerMemory,
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    planning_context: WorkflowPlanningContext,
    branch_preference_policy: WorkflowBranchPreferencePolicy,
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...],
) -> tuple[int | float, ...]:
    """Return one result's configured branch score key."""

    requested_goal_report = evaluate_requested_goal_report(
        result.structure,
        requested_goals=requested_goals.goals,
        component_library=component_library,
        already_satisfied_requested_goals=already_satisfied_requested_goals,
    )
    score = evaluate_workflow_branch_quality_score(
        result,
        requested_goal_report=requested_goal_report,
        planning_context=planning_context,
        component_library=component_library,
        search_depth=len(planner_memory.adopted_transformers),
    )
    return score.order_key(branch_preference_policy)
