"""Frontier runtime for workflow planning execution."""

from dataclasses import dataclass, field, replace

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.completion.hydrogen.protonation import (
    HistidineProtonationRequest,
)
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.refinement.result import RefinementTransformationResult
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
    WorkflowBranchOutcome,
    WorkflowBranchQualityScore,
    WorkflowBranchStatus,
)
from protrepair.workflow.engine.parser_witness_repair import (
    execute_bounded_parser_witness_repair_loop,
)
from protrepair.workflow.engine.reporting import (
    evaluate_requested_goal_report,
    evaluate_workflow_branch_outcome,
    evaluate_workflow_branch_quality_score,
)
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
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
class WorkflowPlanningSession:
    """Reuse exact-state plans within one workflow's fixed planning context."""

    requested_goals: RequestedGoalSet
    transform_requests: WorkflowTransformRequests
    component_library: ComponentLibrary
    planning_context: WorkflowPlanningContext
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence, ...
    ]
    _outcomes_by_state: dict[
        int, tuple[WorkflowRuntimeState, WorkflowPlanningOutcome]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def known_outcome(
        self, state: WorkflowRuntimeState
    ) -> WorkflowPlanningOutcome | None:
        """Return an existing plan without doing new planning work."""

        cached = self._outcomes_by_state.get(id(state))
        return cached[1] if cached is not None and cached[0] is state else None

    def plan(self, state: WorkflowRuntimeState) -> WorkflowPlanningOutcome:
        """Plan once for this exact immutable structure and history carrier."""

        outcome = self.known_outcome(state)
        if outcome is None:
            outcome = plan_workflow_actions(
                state.result.structure,
                requested_goals=self.requested_goals,
                transform_requests=self.transform_requests,
                component_library=self.component_library,
                planner_memory=state.planner_memory,
                planning_context=self.planning_context,
                retained_non_polymer_chemistry_evidence=(
                    self.retained_non_polymer_chemistry_evidence
                ),
            )
            self._outcomes_by_state[id(state)] = (state, outcome)
        return outcome

    def retain_states(self, states: tuple[WorkflowRuntimeState, ...]) -> None:
        """Release plans and their strong input references outside the frontier."""

        retained_ids = {id(state) for state in states}
        for state_id in tuple(self._outcomes_by_state):
            if state_id not in retained_ids:
                del self._outcomes_by_state[state_id]


@dataclass(frozen=True, slots=True)
class WorkflowBranchEvaluation:
    """One runtime branch's goal report and policy-independent quality score."""

    requested_goal_report: RequestedGoalReport
    quality_score: WorkflowBranchQualityScore


@dataclass(slots=True)
class WorkflowBranchEvaluator:
    """Memoize branch evaluations for one immutable workflow runtime."""

    requested_goals: RequestedGoalSet
    component_library: ComponentLibrary
    planning_context: WorkflowPlanningContext
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...]
    _evaluations_by_result_depth: dict[
        tuple[int, int],
        tuple[TransformationResult, WorkflowBranchEvaluation],
    ] = field(default_factory=dict, init=False, repr=False)

    def evaluation(self, state: WorkflowRuntimeState) -> WorkflowBranchEvaluation:
        """Return the cached evaluation for one exact result and search depth."""

        search_depth = len(state.planner_memory.adopted_transformers)
        cache_key = (id(state.result), search_depth)
        cached_entry = self._evaluations_by_result_depth.get(cache_key)
        if cached_entry is not None and cached_entry[0] is state.result:
            return cached_entry[1]

        observation = StructureChemistryObservation(
            state.result.structure, component_library=self.component_library
        )
        requested_goal_report = evaluate_requested_goal_report(
            state.result.structure,
            requested_goals=self.requested_goals.goals,
            component_library=self.component_library,
            already_satisfied_requested_goals=(self.already_satisfied_requested_goals),
            chemistry_observation=observation,
        )
        evaluation = self._evaluation_for_report(
            state.result,
            requested_goal_report=requested_goal_report,
            search_depth=search_depth,
            chemistry_observation=observation,
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
        chemistry_observation: StructureChemistryObservation,
    ) -> WorkflowBranchEvaluation:
        """Pair one goal report with its policy-independent structure score."""

        return WorkflowBranchEvaluation(
            requested_goal_report=requested_goal_report,
            quality_score=evaluate_workflow_branch_quality_score(
                result,
                requested_goal_report=requested_goal_report,
                planning_context=self.planning_context,
                component_library=self.component_library,
                search_depth=search_depth,
                chemistry_observation=chemistry_observation,
            ),
        )

    def branch_outcome(
        self,
        *,
        node_id: SpeculativePlanningNodeId,
        state: WorkflowRuntimeState,
        result: TransformationResult,
        planning_outcome: WorkflowPlanningOutcome | None = None,
        status: WorkflowBranchStatus = WorkflowBranchStatus.TERMINAL,
    ) -> WorkflowBranchOutcome:
        """Assess a retained candidate, reusing exact selection observations."""

        blocked_requested_goal_blockers = (
            ()
            if planning_outcome is None
            else planning_outcome.blocked_requested_goal_blockers()
        )
        unsupported_requested_goals = (
            ()
            if planning_outcome is None
            else planning_outcome.unsupported_requested_goals
        )
        if (
            result is state.result
            and not unsupported_requested_goals
            and not blocked_requested_goal_blockers
        ):
            selection_evaluation = self.evaluation(state)
            evaluation = WorkflowBranchEvaluation(
                requested_goal_report=selection_evaluation.requested_goal_report,
                quality_score=replace(
                    selection_evaluation.quality_score,
                    search_depth=node_id.value,
                ),
            )
        else:
            observation = StructureChemistryObservation(
                result.structure, component_library=self.component_library
            )
            requested_goal_report = evaluate_requested_goal_report(
                result.structure,
                requested_goals=self.requested_goals.goals,
                component_library=self.component_library,
                unsupported_requested_goals=unsupported_requested_goals,
                blocked_requested_goal_blockers=blocked_requested_goal_blockers,
                already_satisfied_requested_goals=(
                    self.already_satisfied_requested_goals
                ),
                chemistry_observation=observation,
            )
            evaluation = self._evaluation_for_report(
                result,
                requested_goal_report=requested_goal_report,
                search_depth=node_id.value,
                chemistry_observation=observation,
            )

        return evaluate_workflow_branch_outcome(
            node_id=node_id,
            result=result,
            requested_goal_report=evaluation.requested_goal_report,
            branch_quality_score=evaluation.quality_score,
            planning_context=self.planning_context,
            component_library=self.component_library,
            status=status,
            relaxation_executions=state.planner_memory.relaxation_executions,
            blockers=tuple(
                blocker
                for _, blockers in blocked_requested_goal_blockers
                for blocker in blockers
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkflowBranch:
    """One assessed retained candidate captured before final branch selection."""

    result: TransformationResult
    outcome: WorkflowBranchOutcome

    @property
    def node_id(self) -> SpeculativePlanningNodeId:
        """Return the identity owned by this branch's assessed outcome."""

        return self.outcome.node_id


@dataclass(frozen=True, slots=True)
class WorkflowRuntimeResult:
    """Assessed retained branches and the actual search stop cause."""

    branches: tuple[WorkflowBranch, ...]
    stop_reason: SpeculativeStopReason = SpeculativeStopReason.NO_PROPOSALS_AVAILABLE


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
    branches: list[WorkflowBranch] = []
    planning_session = WorkflowPlanningSession(
        requested_goals=requested_goals,
        transform_requests=transform_requests,
        component_library=component_library,
        planning_context=planning_context,
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
    )
    branch_evaluator: WorkflowBranchEvaluator | None = None
    while True:
        if not trace.frontier.active_nodes:
            trace = trace.stop(reason=SpeculativeStopReason.NO_PROPOSALS_AVAILABLE)
            break

        active_node = trace.frontier.select_next_active_node()
        runtime_state = active_node.state
        planning_outcome = planning_session.plan(runtime_state)
        if branch_evaluator is None:
            branch_evaluator = WorkflowBranchEvaluator(
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
            branches.append(
                WorkflowBranch(
                    result=terminal_result,
                    outcome=branch_evaluator.branch_outcome(
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
            planning_session.retain_states(
                tuple(node.state for node in trace.frontier.active_nodes)
            )
            continue
        remaining_child_budget = planning_context.max_speculative_nodes - len(
            trace.nodes
        )
        if remaining_child_budget <= 0:
            for retained_node in trace.frontier.active_nodes:
                retained_state = retained_node.state
                retained_plan = planning_session.plan(retained_state)
                retained_result = retained_state.result.with_appended_issues(
                    retained_plan.blocker_issues(
                        retained_state.result.structure,
                        component_library=component_library,
                    )
                )
                branches.append(
                    WorkflowBranch(
                        result=retained_result,
                        outcome=branch_evaluator.branch_outcome(
                            node_id=retained_node.node_id,
                            state=retained_state,
                            result=retained_result,
                            planning_outcome=retained_plan,
                            status=(
                                WorkflowBranchStatus.TERMINAL
                                if retained_plan.is_terminal()
                                else WorkflowBranchStatus.FRONTIER
                            ),
                        ),
                    )
                )
            trace = trace.stop(reason=SpeculativeStopReason.ITERATION_LIMIT_REACHED)
            break
        proposal_batch = planning_outcome.proposals[:remaining_child_budget]

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
        for proposal in proposal_batch:
            transformer = proposal.transformer
            execution_outcome = execute_workflow_transformer(
                runtime_state.result,
                transformer=transformer,
                execution_context=execution_context,
            )
            planner_memory = runtime_state.planner_memory.with_adopted_transformer(
                transformer, input_structure=runtime_state.result.structure
            ).with_returned_relaxations(
                RefinementTransformationResult.observed_executions(execution_outcome)
            )
            child_state = WorkflowRuntimeState(
                result=execution_outcome,
                planner_memory=planner_memory,
            )
            if (
                transform_requests.repair_refinement is None
                and not isinstance(transformer, LocalRefinementTransformer)
                and _next_batch_is_local_refinement(
                    planning_session.plan(child_state),
                )
            ):
                parser_witness_repair = execute_bounded_parser_witness_repair_loop(
                    execution_outcome,
                    execution_context=execution_context,
                    execute_transformer=execute_workflow_transformer,
                )
                execution_outcome = parser_witness_repair.result
                for attempted in parser_witness_repair.attempted_transformers:
                    planner_memory = planner_memory.with_attempted_transformer(
                        attempted
                    )
                for adopted in parser_witness_repair.adopted_transformers:
                    planner_memory = planner_memory.with_adopted_transformer(adopted)
                planner_memory = planner_memory.with_returned_relaxations(
                    RefinementTransformationResult.observed_executions(
                        execution_outcome
                    )
                )
                if (
                    execution_outcome is not child_state.result
                    or planner_memory != child_state.planner_memory
                ):
                    child_state = WorkflowRuntimeState(
                        result=execution_outcome,
                        planner_memory=planner_memory,
                    )

            adopted_children.append(
                SpeculativeAdoptedChild(
                    state=child_state,
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
        batch_executions = ContinuousRelaxationExecution.deduplicated(
            tuple(
                execution
                for child in adopted_children
                for execution in child.state.planner_memory.relaxation_executions
            )
        )
        adopted_children = [
            child
            if child.state.planner_memory.relaxation_executions == batch_executions
            else replace(
                child,
                state=replace(
                    child.state,
                    planner_memory=replace(
                        child.state.planner_memory,
                        relaxation_executions=batch_executions,
                    ),
                ),
            )
            for child in adopted_children
        ]
        adopted_children = list(
            adopted_workflow_children(
                adopted_children=tuple(adopted_children),
                branch_evaluator=branch_evaluator,
            )
        )
        adopted_children = list(
            workflow_children_with_regression_retention(
                current_branch_state=runtime_state,
                attempted_proposals=proposal_batch,
                retained_children=tuple(adopted_children),
                branch_evaluator=branch_evaluator,
            )
        )
        adopted_children = list(
            workflow_children_within_node_budget(
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
        planning_session.retain_states(
            tuple(node.state for node in trace.frontier.active_nodes)
        )

    if not branches:
        raise ValueError(
            "iterative workflow execution requires at least one assessed branch"
        )

    return WorkflowRuntimeResult(
        branches=tuple(branches),
        stop_reason=trace.stop_reason or SpeculativeStopReason.NO_PROPOSALS_AVAILABLE,
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


def adopted_workflow_children(
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
    branch_evaluator: WorkflowBranchEvaluator,
) -> tuple[
    SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    ...,
]:
    """Keep sibling branches tied at the best goal-first quality key.

    Parameters
    ----------
    adopted_children : tuple[SpeculativeAdoptedChild, ...]
        Candidate workflow branches after transformation.
    branch_evaluator : WorkflowBranchEvaluator
        Evaluation cache belonging to this workflow invocation.

    Returns
    -------
    tuple[SpeculativeAdoptedChild, ...]
        Best-quality siblings in their original order.
    """

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


def workflow_children_with_regression_retention(
    *,
    current_branch_state: WorkflowRuntimeState,
    attempted_proposals: tuple[WorkflowActionProposal, ...],
    retained_children: tuple[
        SpeculativeAdoptedChild[
            WorkflowRuntimeState,
            WorkflowStateAction,
            TransformationResult,
            TransformationResult,
        ],
        ...,
    ],
    branch_evaluator: WorkflowBranchEvaluator,
) -> tuple[
    SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    ...,
]:
    """Retain the parent when permitted and every child regresses.

    Parameters
    ----------
    current_branch_state : WorkflowRuntimeState
        Parent result and action history.
    attempted_proposals : tuple[WorkflowActionProposal, ...]
        Proposals retaining explicit-request and selection semantics.
    retained_children : tuple[SpeculativeAdoptedChild, ...]
        Candidate children already selected for further search.
    branch_evaluator : WorkflowBranchEvaluator
        This invocation's evaluator and quality policy context.

    Returns
    -------
    tuple[SpeculativeAdoptedChild, ...]
        Children, optionally followed by the unchanged parent with updated attempt
        history. Explicit-request and regression policies still govern retention.
    """

    if not retained_children:
        return ()

    if not attempted_proposals or not all(
        proposal.allows_parent_retention() for proposal in attempted_proposals
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
    for child in retained_children:
        planner_memory = planner_memory.with_returned_relaxations(
            child.state.planner_memory.relaxation_executions
        )
    for proposal in attempted_proposals:
        planner_memory = planner_memory.with_attempted_transformer(
            proposal.transformer,
            input_structure=current_branch_state.result.structure,
        )

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


def workflow_children_within_node_budget(
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
    branch_evaluator: WorkflowBranchEvaluator,
) -> tuple[
    SpeculativeAdoptedChild[
        WorkflowRuntimeState,
        WorkflowStateAction,
        TransformationResult,
        TransformationResult,
    ],
    ...,
]:
    """Select best-quality children within the existing trace budget.

    Parameters
    ----------
    children : tuple[SpeculativeAdoptedChild, ...]
        Branches eligible for trace insertion.
    child_budget : int
        Remaining node capacity; non-positive values retain no children.
    branch_evaluator : WorkflowBranchEvaluator
        This invocation's quality evaluator.

    Returns
    -------
    tuple[SpeculativeAdoptedChild, ...]
        All children when they fit, otherwise a stable best-quality prefix.
    """

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


def _next_batch_is_local_refinement(outcome: WorkflowPlanningOutcome) -> bool:
    """Keep attached parser work behind the planner's current prerequisites."""

    return bool(outcome.proposals) and all(
        isinstance(proposal.transformer, LocalRefinementTransformer)
        for proposal in outcome.proposals
    )


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
    branch_evaluator: WorkflowBranchEvaluator,
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
