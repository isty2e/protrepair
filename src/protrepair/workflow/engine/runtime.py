"""Frontier runtime for workflow planning execution."""

from dataclasses import dataclass, field, replace

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.completion.hydrogen.protonation import (
    HistidineProtonationRequest,
)
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
from protrepair.workflow.contracts.result import (
    RequestedGoalReport,
    WorkflowBranchQualityScore,
    WorkflowTerminalBranchOutcome,
)
from protrepair.workflow.engine.parser_witness_repair import (
    execute_bounded_parser_witness_repair_loop,
)
from protrepair.workflow.engine.reporting import (
    evaluate_requested_goal_report,
    evaluate_terminal_branch_outcome,
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
class _WorkflowBranchEvaluation:
    """One runtime branch's goal report and policy-independent quality score."""

    requested_goal_report: RequestedGoalReport
    quality_score: WorkflowBranchQualityScore


@dataclass(slots=True)
class _WorkflowBranchEvaluator:
    """Memoize branch evaluations for one immutable workflow runtime."""

    requested_goals: RequestedGoalSet
    component_library: ComponentLibrary
    planning_context: WorkflowPlanningContext
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...]
    _evaluations_by_result_depth: dict[
        tuple[int, int],
        tuple[TransformationResult, _WorkflowBranchEvaluation],
    ] = field(default_factory=dict, init=False, repr=False)

    def evaluation(self, state: WorkflowRuntimeState) -> _WorkflowBranchEvaluation:
        """Return the cached evaluation for one exact result and search depth."""

        search_depth = len(state.planner_memory.adopted_transformers)
        cache_key = (id(state.result), search_depth)
        cached_entry = self._evaluations_by_result_depth.get(cache_key)
        if cached_entry is not None and cached_entry[0] is state.result:
            return cached_entry[1]

        requested_goal_report = evaluate_requested_goal_report(
            state.result.structure,
            requested_goals=self.requested_goals.goals,
            component_library=self.component_library,
            already_satisfied_requested_goals=(
                self.already_satisfied_requested_goals
            ),
        )
        evaluation = self._evaluation_for_report(
            state.result,
            requested_goal_report=requested_goal_report,
            search_depth=search_depth,
        )
        self._evaluations_by_result_depth[cache_key] = (state.result, evaluation)
        return evaluation

    def quality_key(
        self,
        state: WorkflowRuntimeState,
        *,
        policy: WorkflowBranchPreferencePolicy,
    ) -> tuple[int | float, ...]:
        """Return one cached branch evaluation under the selected axis order."""

        return self.evaluation(state).quality_score.order_key(policy)

    def retain_states(self, states: tuple[WorkflowRuntimeState, ...]) -> None:
        """Discard evaluations that no longer belong to the live frontier."""

        retained_keys = {
            (
                id(state.result),
                len(state.planner_memory.adopted_transformers),
            )
            for state in states
        }
        self._evaluations_by_result_depth = {
            cache_key: entry
            for cache_key, entry in self._evaluations_by_result_depth.items()
            if cache_key in retained_keys
        }

    def _evaluation_for_report(
        self,
        result: TransformationResult,
        *,
        requested_goal_report: RequestedGoalReport,
        search_depth: int,
    ) -> _WorkflowBranchEvaluation:
        """Pair one goal report with its policy-independent structure score."""

        return _WorkflowBranchEvaluation(
            requested_goal_report=requested_goal_report,
            quality_score=evaluate_workflow_branch_quality_score(
                result,
                requested_goal_report=requested_goal_report,
                planning_context=self.planning_context,
                component_library=self.component_library,
                search_depth=search_depth,
            ),
        )

    def terminal_outcome(
        self,
        *,
        node_id: SpeculativePlanningNodeId,
        state: WorkflowRuntimeState,
        result: TransformationResult,
        planning_outcome: WorkflowPlanningOutcome,
    ) -> WorkflowTerminalBranchOutcome:
        """Return terminal reporting facts, reusing selection facts when exact."""

        blocked_requested_goal_blockers = (
            planning_outcome.blocked_requested_goal_blockers()
        )
        unsupported_requested_goals = planning_outcome.unsupported_requested_goals
        if (
            result is state.result
            and not unsupported_requested_goals
            and not blocked_requested_goal_blockers
        ):
            selection_evaluation = self.evaluation(state)
            evaluation = _WorkflowBranchEvaluation(
                requested_goal_report=selection_evaluation.requested_goal_report,
                quality_score=replace(
                    selection_evaluation.quality_score,
                    search_depth=node_id.value,
                ),
            )
        else:
            requested_goal_report = evaluate_requested_goal_report(
                result.structure,
                requested_goals=self.requested_goals.goals,
                component_library=self.component_library,
                unsupported_requested_goals=unsupported_requested_goals,
                blocked_requested_goal_blockers=blocked_requested_goal_blockers,
                already_satisfied_requested_goals=(
                    self.already_satisfied_requested_goals
                ),
            )
            evaluation = self._evaluation_for_report(
                result,
                requested_goal_report=requested_goal_report,
                search_depth=node_id.value,
            )

        return evaluate_terminal_branch_outcome(
            node_id=node_id,
            result=result,
            requested_goal_report=evaluation.requested_goal_report,
            branch_quality_score=evaluation.quality_score,
            planning_context=self.planning_context,
            component_library=self.component_library,
            blockers=tuple(
                blocker
                for _, blockers in blocked_requested_goal_blockers
                for blocker in blockers
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkflowTerminalBranch:
    """One terminal workflow branch captured before final branch selection."""

    result: TransformationResult
    outcome: WorkflowTerminalBranchOutcome

    @property
    def node_id(self) -> SpeculativePlanningNodeId:
        """Return the identity owned by this branch's terminal outcome."""

        return self.outcome.node_id


@dataclass(frozen=True, slots=True)
class WorkflowRuntimeResult:
    """Evaluated terminal branches from one workflow runtime run."""

    terminal_branches: tuple[WorkflowTerminalBranch, ...]


def execute_iterative_workflow(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    transform_requests: WorkflowTransformRequests,
    component_library: ComponentLibrary,
    planning_context: WorkflowPlanningContext,
    reference_structure: ProteinStructure | None,
    orphan_fragment_policy: OrphanFragmentPolicy,
    histidine_protonation: HistidineProtonationRequest,
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
    branch_evaluator: _WorkflowBranchEvaluator | None = None
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
        if branch_evaluator is None:
            branch_evaluator = _WorkflowBranchEvaluator(
                requested_goals=requested_goals,
                component_library=component_library,
                planning_context=planning_context,
                already_satisfied_requested_goals=(
                    planning_outcome.already_satisfied_requested_goals
                ),
            )
        if planning_outcome.is_terminal():
            terminal_result = runtime_state.result.with_appended_issues(
                planning_outcome.blocker_issues(
                    runtime_state.result.structure,
                    component_library=component_library,
                )
            )
            terminal_branches.append(
                WorkflowTerminalBranch(
                    result=terminal_result,
                    outcome=branch_evaluator.terminal_outcome(
                        node_id=active_node.node_id,
                        state=runtime_state,
                        result=terminal_result,
                        planning_outcome=planning_outcome,
                    ),
                )
            )
            trace = trace.expand_active_node(
                parent_node=active_node,
                adopted_children=(),
            )
            branch_evaluator.retain_states(
                tuple(node.state for node in trace.frontier.active_nodes)
            )
            continue
        remaining_child_budget = planning_context.max_speculative_nodes - len(
            trace.nodes
        )
        if remaining_child_budget <= 0:
            trace = trace.stop(reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED)
            break
        proposal_batch = planning_outcome.current_proposal_batch()[
            :remaining_child_budget
        ]

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
            histidine_protonation=histidine_protonation,
            allow_retained_non_polymer_rdkit_fallback=(
                transform_requests.allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
        )
        for transformer in proposal_batch:
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
                branch_evaluator=branch_evaluator,
            )
        )
        adopted_children = list(
            _workflow_children_with_regression_retention(
                current_branch_state=runtime_state,
                attempted_transformers=proposal_batch,
                transform_requests=transform_requests,
                retained_children=tuple(adopted_children),
                branch_evaluator=branch_evaluator,
            )
        )
        adopted_children = list(
            _workflow_children_within_node_budget(
                children=tuple(adopted_children),
                child_budget=remaining_child_budget,
                branch_evaluator=branch_evaluator,
            )
        )
        trace = trace.expand_active_node(
            parent_node=active_node,
            adopted_children=tuple(adopted_children),
        )
        branch_evaluator.retain_states(
            tuple(node.state for node in trace.frontier.active_nodes)
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
    branch_evaluator: _WorkflowBranchEvaluator,
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
            branch_evaluator=branch_evaluator,
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
    branch_evaluator: _WorkflowBranchEvaluator,
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

    regression_policy = _branch_regression_preference_policy(
        branch_evaluator.planning_context
    )
    if regression_policy is None:
        return retained_children

    current_branch_quality_key = branch_evaluator.quality_key(
        current_branch_state,
        policy=regression_policy,
    )
    child_quality_keys = tuple(
        branch_evaluator.quality_key(
            child.state,
            policy=regression_policy,
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


def _workflow_children_within_node_budget(
    *,
    children: tuple[
        SpeculativeAdoptedChild[
            WorkflowRuntimeState,
            WorkflowStateAction,
            TransformationResult,
            TransformationResult,
        ],
        ...,
    ],
    child_budget: int,
    branch_evaluator: _WorkflowBranchEvaluator,
) -> tuple[
    SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    ...,
]:
    """Return the best child branches that fit the remaining trace budget."""

    if child_budget <= 0 or not children:
        return ()

    if len(children) <= child_budget:
        return children

    child_keys = tuple(
        _workflow_child_quality_key(
            child,
            branch_evaluator=branch_evaluator,
        )
        for child in children
    )
    ranked_child_keys = sorted(child_keys, key=lambda child_key: child_key[1])
    return tuple(child for child, _ in ranked_child_keys[:child_budget])


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
    branch_evaluator: _WorkflowBranchEvaluator,
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
        branch_evaluator.quality_key(
            child.state,
            policy=branch_evaluator.planning_context.branch_preference_policy,
        ),
    )
