"""Generic workflow planning over requested goals and transformer invocations."""

from dataclasses import dataclass, replace

from protrepair.chemistry import (
    ComponentLibrary,
    build_default_component_library,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
    evidence_by_residue_id,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.scope import Scope
from protrepair.state import (
    HydrogenCoverageState,
    StructureBoundaryStateFacts,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.structure_topology import (
    StructureDisulfideHydrogenFacts,
    StructureDisulfideTopologyFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.local.projection import (
    LocalContinuousExecutionResidueProjection,
)
from protrepair.workflow.contracts.planning import (
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
    WorkflowSpanDonorAvailability,
)
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowGoal,
    WorkflowTransformRequests,
)
from protrepair.workflow.planning.action.attempt import WorkflowActionAttempt
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.action.registry import (
    WorkflowActionAdoption,
    WorkflowActionRegistryEntry,
    WorkflowStateAction,
    action_adoption_for_registry,
    action_registry_capabilities,
)
from protrepair.workflow.planning.assessment.blockers import WorkflowBlocker
from protrepair.workflow.planning.assessment.deficits import (
    WorkflowStateDeficit,
)
from protrepair.workflow.planning.assessment.goal_support import (
    already_satisfied_requested_goals,
    blocked_requested_goal_projections,
    unsupported_requested_goals,
)
from protrepair.workflow.planning.context_projection import (
    planning_context_is_holo_for_structure,
)
from protrepair.workflow.planning.intrinsic_geometry import (
    derive_structure_intrinsic_geometry_facts,
)
from protrepair.workflow.planning.policy import WorkflowPlanningPolicy
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)


@dataclass(frozen=True, slots=True)
class BlockedRequestedGoal:
    """One requested goal plus the blockers that prevented planning it."""

    requested_goal: WorkflowGoal
    blockers: tuple[WorkflowBlocker, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(self.blockers))

    def blocking_scopes(self) -> tuple[Scope, ...]:
        """Return the semantic scopes responsible for this blocked goal."""

        return tuple(blocker.scope for blocker in self.blockers)

    def blocking_phases(self) -> tuple[WorkflowPlanningPhase, ...]:
        """Return workflow phases responsible for this blocked goal."""

        return tuple(dict.fromkeys(blocker.phase for blocker in self.blockers))


@dataclass(frozen=True, slots=True)
class WorkflowPlanningOutcome:
    """Selected proposals and observations from one current-state assessment.

    Parameters
    ----------
    structure_planning_signature : StructurePlanningSignature
        Current chemistry, coverage and boundary signature.
    proposals : tuple[WorkflowActionProposal, ...]
        Selected family batch, retaining per-proposal selection semantics.
    blockers : tuple[WorkflowBlocker, ...]
        Known obstacles to requested goals.
    unsupported_requested_goals : tuple[WorkflowGoal, ...]
        Goals outside the active registry's capabilities.
    blocked_requested_goals : tuple[BlockedRequestedGoal, ...]
        Supported goals with current blockers.
    already_satisfied_requested_goals : tuple[WorkflowGoal, ...]
        Goals met before this planning step.
    state_deficit : WorkflowStateDeficit or None
        Current assessed burden, when available.

    Raises
    ------
    TypeError
        A batch member is not a canonical proposal.
    """

    structure_planning_signature: StructurePlanningSignature
    proposals: tuple[WorkflowActionProposal, ...]
    blockers: tuple[WorkflowBlocker, ...] = ()
    unsupported_requested_goals: tuple[
        WorkflowGoal,
        ...,
    ] = ()
    blocked_requested_goals: tuple[BlockedRequestedGoal, ...] = ()
    already_satisfied_requested_goals: tuple[
        WorkflowGoal,
        ...,
    ] = ()
    state_deficit: WorkflowStateDeficit | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposals", tuple(self.proposals))
        if any(
            not isinstance(proposal, WorkflowActionProposal)
            for proposal in self.proposals
        ):
            raise TypeError("planning outcome requires canonical proposals")
        object.__setattr__(self, "blockers", tuple(self.blockers))
        object.__setattr__(
            self,
            "unsupported_requested_goals",
            tuple(self.unsupported_requested_goals),
        )
        object.__setattr__(
            self,
            "blocked_requested_goals",
            tuple(self.blocked_requested_goals),
        )
        object.__setattr__(
            self,
            "already_satisfied_requested_goals",
            tuple(self.already_satisfied_requested_goals),
        )

    @classmethod
    def from_candidates(
        cls,
        *,
        structure_planning_signature: StructurePlanningSignature,
        candidate_proposals: list[WorkflowActionProposal],
        blockers: tuple[WorkflowBlocker, ...] = (),
        unsupported_requested_goals: tuple[
            WorkflowGoal,
            ...,
        ] = (),
        blocked_requested_goals: tuple[BlockedRequestedGoal, ...] = (),
        already_satisfied_requested_goals: tuple[
            WorkflowGoal,
            ...,
        ] = (),
        state_deficit: WorkflowStateDeficit | None = None,
    ) -> "WorkflowPlanningOutcome":
        """Select the first ranked family without discarding proposal semantics.

        Parameters
        ----------
        structure_planning_signature : StructurePlanningSignature
            Current assessed signature.
        candidate_proposals : list[WorkflowActionProposal]
            Already ranked proposals; order and duplicates remain meaningful.
        blockers : tuple[WorkflowBlocker, ...]
            Current goal obstacles.
        unsupported_requested_goals : tuple[WorkflowGoal, ...]
            Goals outside registry support.
        blocked_requested_goals : tuple[BlockedRequestedGoal, ...]
            Goals with current blockers.
        already_satisfied_requested_goals : tuple[WorkflowGoal, ...]
            Goals already met.
        state_deficit : WorkflowStateDeficit or None
            Current burden, when assessed.

        Returns
        -------
        WorkflowPlanningOutcome
            First-family batch with original proposal identity and metadata.
        """

        return cls(
            structure_planning_signature=structure_planning_signature,
            proposals=cls._current_proposal_batch(candidate_proposals),
            blockers=blockers,
            unsupported_requested_goals=unsupported_requested_goals,
            blocked_requested_goals=blocked_requested_goals,
            already_satisfied_requested_goals=(already_satisfied_requested_goals),
            state_deficit=state_deficit,
        )

    @staticmethod
    def _current_proposal_batch(
        candidate_proposals: list[WorkflowActionProposal],
    ) -> tuple[WorkflowActionProposal, ...]:
        """Return the current frontier proposal batch from ordered candidates."""

        if not candidate_proposals:
            return ()

        first_family = candidate_proposals[0].transformer.proposal_family()
        return tuple(
            proposal
            for proposal in candidate_proposals
            if proposal.transformer.proposal_family() is first_family
        )

    @property
    def transformers(self) -> tuple[WorkflowStateAction, ...]:
        """Project the selected proposals into executable actions.

        Returns
        -------
        tuple[WorkflowStateAction, ...]
            Actions in proposal order; selection semantics remain on proposals.
        """
        return tuple(proposal.transformer for proposal in self.proposals)

    def has_transformers(self) -> bool:
        """Return whether the current planning outcome proposes any transformers."""

        return bool(self.proposals)

    def is_terminal(self) -> bool:
        """Return whether the workflow reached a terminal planning state."""

        return not self.has_transformers()

    def current_proposal_batch(self) -> tuple[WorkflowStateAction, ...]:
        """Return the executable transformer batch for the current frontier depth."""

        return self.transformers

    def blocked_requested_goal_scopes(
        self,
    ) -> tuple[
        tuple[WorkflowGoal, tuple[Scope, ...]],
        ...,
    ]:
        """Return blocked requested goals projected into blocking scopes."""

        return tuple(
            (
                blocked_requested_goal.requested_goal,
                blocked_requested_goal.blocking_scopes(),
            )
            for blocked_requested_goal in self.blocked_requested_goals
        )

    def blocked_requested_goal_blockers(
        self,
    ) -> tuple[
        tuple[
            WorkflowGoal,
            tuple[WorkflowBlocker, ...],
        ],
        ...,
    ]:
        """Return blocked requested goals projected with full blocker payloads."""

        return tuple(
            (
                blocked_requested_goal.requested_goal,
                blocked_requested_goal.blockers,
            )
            for blocked_requested_goal in self.blocked_requested_goals
        )

    def blocker_issues(
        self,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
    ) -> tuple[ValidationIssue, ...]:
        """Return workflow issues projected from the current blockers."""

        return tuple(
            blocker.issue_for(
                structure,
                component_library=component_library,
            )
            for blocker in self.blockers
        )


@dataclass(frozen=True, slots=True)
class WorkflowPlannerMemory:
    """Retain action and native evidence under one branch's fixed execution context.

    Parameters
    ----------
    adopted_transformers : tuple[WorkflowStateAction, ...]
        Historical actions whose outputs entered this path.
    attempts : tuple[WorkflowActionAttempt, ...]
        Evaluations, including input evidence for reapplicable preparation.
    relaxation_executions : tuple[ContinuousRelaxationExecution, ...]
        Actual native returns, independently of coordinate adoption.

    Notes
    -----
    Library, reference and chemistry-policy inputs are fixed within this path.
    This memory is not reusable across workflows with different execution contexts.
    """

    adopted_transformers: tuple[WorkflowStateAction, ...] = ()
    attempts: tuple[WorkflowActionAttempt, ...] = ()
    relaxation_executions: tuple[ContinuousRelaxationExecution, ...] = ()

    def __post_init__(self) -> None:
        adopted_transformers_list: list[WorkflowStateAction] = []
        for adopted_transformer in self.adopted_transformers:
            if adopted_transformer not in adopted_transformers_list:
                adopted_transformers_list.append(adopted_transformer)

        adopted_transformers = tuple(adopted_transformers_list)
        attempts: list[WorkflowActionAttempt] = []
        for attempt in self.attempts:
            if attempt not in attempts:
                attempts.append(attempt)
        for transformer in adopted_transformers:
            if not any(attempt.transformer == transformer for attempt in attempts):
                attempts.append(WorkflowActionAttempt(transformer))
        object.__setattr__(self, "adopted_transformers", adopted_transformers)
        object.__setattr__(self, "attempts", tuple(attempts))
        object.__setattr__(
            self,
            "relaxation_executions",
            ContinuousRelaxationExecution.deduplicated(self.relaxation_executions),
        )

    def has_adopted_transformer(
        self,
        transformer: WorkflowStateAction,
    ) -> bool:
        """Return whether one exact workflow transformer was already adopted."""

        return transformer in self.adopted_transformers

    def has_attempted_transformer(
        self,
        transformer: WorkflowStateAction,
    ) -> bool:
        """Return whether one exact workflow transformer was already evaluated."""

        return any(attempt.transformer == transformer for attempt in self.attempts)

    @property
    def attempted_transformers(self) -> tuple[WorkflowStateAction, ...]:
        """Return historical action membership, independent of current applicability.

        Returns
        -------
        tuple[WorkflowStateAction, ...]
            Actions in first-observed order without duplicating input-bound attempts.
        """
        actions: list[WorkflowStateAction] = []
        for attempt in self.attempts:
            if attempt.transformer not in actions:
                actions.append(attempt.transformer)
        return tuple(actions)

    def blocks_transformer_attempt(
        self, transformer: WorkflowStateAction, *, structure: ProteinStructure
    ) -> bool:
        """Check current-input suppression without changing historical facts.

        Parameters
        ----------
        transformer : WorkflowStateAction
            Proposed action.
        structure : ProteinStructure
            Canonical current input under this branch's fixed execution context.

        Returns
        -------
        bool
            Whether historical attempts cover this action's reapplication context.
        """
        return any(attempt.blocks(transformer, structure) for attempt in self.attempts)

    def with_adopted_transformer(
        self,
        transformer: WorkflowStateAction,
        *,
        input_structure: ProteinStructure | None = None,
    ) -> "WorkflowPlannerMemory":
        """Record adoption without losing a repeated preparation's input.

        Parameters
        ----------
        transformer : WorkflowStateAction
            Adopted action.
        input_structure : ProteinStructure or None
            Carrier before execution. None retains only historical action evidence.

        Returns
        -------
        WorkflowPlannerMemory
            Updated attempt evidence and deduplicated historical adoption membership.
        """

        memory = self.with_attempted_transformer(
            transformer, input_structure=input_structure
        )
        if memory.has_adopted_transformer(transformer):
            return memory

        return replace(
            memory,
            adopted_transformers=memory.adopted_transformers + (transformer,),
        )

    def with_attempted_transformer(
        self,
        transformer: WorkflowStateAction,
        *,
        input_structure: ProteinStructure | None = None,
    ) -> "WorkflowPlannerMemory":
        """Record evaluation separately from output adoption.

        Parameters
        ----------
        transformer : WorkflowStateAction
            Evaluated action, whether or not its output was retained.
        input_structure : ProteinStructure or None
            Carrier before execution; branch-only actions do not retain it.

        Returns
        -------
        WorkflowPlannerMemory
            History containing the action and its declared reapplication context.
        """

        attempt = WorkflowActionAttempt(
            transformer,
            None
            if input_structure is None
            else transformer.reapplication_context(input_structure),
        )
        if attempt in self.attempts:
            return self

        return replace(
            self,
            attempts=self.attempts + (attempt,),
        )

    def with_returned_relaxations(
        self,
        executions: tuple[ContinuousRelaxationExecution, ...],
    ) -> "WorkflowPlannerMemory":
        """Remember native returns without claiming action or coordinate adoption.

        Parameters
        ----------
        executions : tuple[ContinuousRelaxationExecution, ...]
            Observed returns transported from this branch's execution.

        Returns
        -------
        WorkflowPlannerMemory
            Memory preserving invocation identity, order and existing action facts.
        """

        combined = ContinuousRelaxationExecution.deduplicated(
            (*self.relaxation_executions, *executions)
        )
        if combined == self.relaxation_executions:
            return self

        return replace(self, relaxation_executions=combined)

    def adoption_records(
        self,
        *,
        registry: tuple[WorkflowActionRegistryEntry[WorkflowActionDomain], ...],
    ) -> tuple[WorkflowActionAdoption, ...]:
        """Return planner-readable adoption records for the current memory."""

        adoption_records: list[WorkflowActionAdoption] = []
        for transformer in self.adopted_transformers:
            adoption = action_adoption_for_registry(transformer, registry=registry)
            if adoption is None or adoption in adoption_records:
                continue

            adoption_records.append(adoption)

        return tuple(adoption_records)


def plan_workflow_actions(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    transform_requests: WorkflowTransformRequests,
    component_library: ComponentLibrary | None = None,
    planner_memory: WorkflowPlannerMemory | None = None,
    planning_context: WorkflowPlanningContext | None = None,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    action_registry: tuple[
        WorkflowActionRegistryEntry[WorkflowActionDomain],
        ...,
    ]
    | None = None,
    chemistry_observation: StructureChemistryObservation | None = None,
) -> WorkflowPlanningOutcome:
    """Plan from current facts without transferring proposal-time authority.

    Parameters
    ----------
    structure : ProteinStructure
        Exact current carrier.
    requested_goals : RequestedGoalSet
        Normalized target states.
    transform_requests : WorkflowTransformRequests
        Explicit transformation requests and preparation policy.
    component_library : ComponentLibrary or None
        Active definitions, or the default library.
    planner_memory : WorkflowPlannerMemory or None
        Prior attempts and adopted actions, or an empty history.
    planning_context : WorkflowPlanningContext or None
        Search and donor constraints, or defaults inferred from the request.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit RetainedNonPolymerChemistryEvidence values.
    action_registry : tuple or None
        Available WorkflowActionRegistryEntry families, or the default registry.
    chemistry_observation : StructureChemistryObservation or None
        Observation bound to the exact carrier, library, canonical evidence and
        default observation policy. None derives one independently.

    Returns
    -------
    WorkflowPlanningOutcome
        Selected proposals and current blockers.

    Raises
    ------
    ValueError
        The observation has different interpretation inputs, or required donor
        availability is absent from the planning context.
    """

    active_action_registry = (
        _default_workflow_action_registry()
        if action_registry is None
        else action_registry
    )
    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    active_planner_memory = (
        WorkflowPlannerMemory() if planner_memory is None else planner_memory
    )
    active_planning_context = (
        WorkflowPlanningContext(
            span_donor_availability=(
                WorkflowSpanDonorAvailability.AVAILABLE
                if transform_requests.requests_external_span_reconstruction()
                else WorkflowSpanDonorAvailability.NONE
            )
        )
        if planning_context is None
        else planning_context
    )
    if (
        transform_requests.requests_external_span_reconstruction()
        and not active_planning_context.allows_span_reconstruction()
    ):
        raise ValueError(
            "workflow planning context must advertise span-donor availability "
            "when explicit external span reconstructions are requested"
        )

    if chemistry_observation is None:
        chemistry_observation = StructureChemistryObservation(
            structure,
            component_library=active_component_library,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
        )
    else:
        chemistry_observation.require_context(
            structure,
            component_library=active_component_library,
            retained_non_polymer_chemistry_evidence=tuple(
                evidence_by_residue_id(retained_non_polymer_chemistry_evidence).values()
            ),
        )
    coverage_facts, chemistry_readiness_facts = (
        chemistry_observation.whole_structure_facts()
    )
    boundary_facts = StructureBoundaryStateFacts.from_structure(structure)
    disulfide_topology_facts = StructureDisulfideTopologyFacts.from_structure(structure)
    disulfide_hydrogen_facts = StructureDisulfideHydrogenFacts.from_structure(structure)
    intrinsic_geometry_facts: StructureIntrinsicGeometryFacts | None = None
    parser_compatibility_facts: StructureParserCompatibilityFacts | None = None
    interaction_facts: StructureInteractionFacts | None = None
    if transform_requests.repair_refinement is not None:
        intrinsic_geometry_facts = derive_structure_intrinsic_geometry_facts(
            structure,
            component_library=active_component_library,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )
        if planning_context_is_holo_for_structure(
            active_planning_context,
            structure,
        ):
            interaction_facts = StructureInteractionFacts.from_structure(
                structure,
                component_library=active_component_library,
                chemistry_readiness_facts=chemistry_readiness_facts,
            )

    if (
        chemistry_readiness_facts.hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    ):
        parser_compatibility_facts = StructureParserCompatibilityFacts.from_structure(
            structure,
            component_library=active_component_library,
        )

    structure_planning_signature = (
        StructurePlanningSignature.from_coverage_chemistry_and_boundary(
            coverage_facts=coverage_facts,
            chemistry_readiness_facts=chemistry_readiness_facts,
            boundary_facts=boundary_facts,
        )
    )
    registry_capabilities = action_registry_capabilities(active_action_registry)
    unsupported_goal_set = unsupported_requested_goals(
        requested_goals,
        capabilities=registry_capabilities,
    )
    state_deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        disulfide_topology_facts=disulfide_topology_facts,
        disulfide_hydrogen_facts=disulfide_hydrogen_facts,
        requested_goals=requested_goals,
        planning_context=active_planning_context,
        intrinsic_geometry_facts=intrinsic_geometry_facts,
        parser_compatibility_facts=parser_compatibility_facts,
        interaction_facts=interaction_facts,
        backbone_window_refinement_specs=(
            transform_requests.backbone_window_refinements
        ),
    )
    explicit_repair_refinement_execution_projection = None
    if transform_requests.repair_refinement is not None:
        try:
            projection = LocalContinuousExecutionResidueProjection.from_scope_spec(
                ProteinStructureSnapshot.from_structure(structure),
                transform_requests.repair_refinement.resolved_execution_scope_spec(),
                context_radius_angstrom=(
                    transform_requests.repair_refinement.config.context_radius_angstrom
                ),
                component_library=active_component_library,
            )
            explicit_repair_refinement_execution_projection = projection
        except ValueError:
            # An earlier transformer may materialize this scope. Keep the request;
            # execution owns rejection if it still cannot bind on the next state.
            pass
    action_domain = WorkflowActionDomain(
        chemistry_observation=chemistry_observation,
        requested_goals=requested_goals,
        transform_requests=transform_requests,
        planning_context=active_planning_context,
        state_deficit=state_deficit,
        boundary_facts=boundary_facts,
        disulfide_topology_facts=disulfide_topology_facts,
        disulfide_hydrogen_facts=disulfide_hydrogen_facts,
        intrinsic_geometry_facts=intrinsic_geometry_facts,
        parser_compatibility_facts=parser_compatibility_facts,
        interaction_facts=interaction_facts,
        explicit_repair_refinement_execution_projection=(
            explicit_repair_refinement_execution_projection
        ),
        adopted_actions=active_planner_memory.adoption_records(
            registry=active_action_registry,
        ),
        relaxation_executions=active_planner_memory.relaxation_executions,
        attempts=active_planner_memory.attempts,
    )
    candidate_proposals = [
        proposal
        for registry_entry in active_action_registry
        if registry_entry.is_admissible(action_domain)
        for proposal in registry_entry.proposals(action_domain)
        if not active_planner_memory.blocks_transformer_attempt(
            proposal.transformer, structure=structure
        )
    ]
    blocked_goal_projections = blocked_requested_goal_projections(
        requested_goals,
        state_deficit=state_deficit,
        planning_context=active_planning_context,
        capabilities=registry_capabilities,
    )
    blockers = tuple(
        dict.fromkeys(
            blocker
            for blocked_goal_projection in blocked_goal_projections
            for blocker in blocked_goal_projection.blockers
        )
    )
    blocked_requested_goals = tuple(
        BlockedRequestedGoal(
            requested_goal=blocked_goal_projection.goal,
            blockers=blocked_goal_projection.blockers,
        )
        for blocked_goal_projection in blocked_goal_projections
    )
    ranked_candidate_proposals = WorkflowPlanningPolicy().rank_candidates(
        candidate_proposals,
        domain=action_domain,
    )

    return WorkflowPlanningOutcome.from_candidates(
        structure_planning_signature=structure_planning_signature,
        candidate_proposals=list(ranked_candidate_proposals),
        blockers=blockers,
        unsupported_requested_goals=unsupported_goal_set,
        blocked_requested_goals=blocked_requested_goals,
        already_satisfied_requested_goals=already_satisfied_requested_goals(
            structure,
            requested_goals=tuple(requested_goals),
            component_library=active_component_library,
            chemistry_observation=chemistry_observation,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
        ),
        state_deficit=state_deficit,
    )


def _default_workflow_action_registry() -> tuple[
    WorkflowActionRegistryEntry[WorkflowActionDomain],
    ...,
]:
    """Return the default concrete workflow action registry."""

    from protrepair.workflow.planning.default_action_registry import (
        WORKFLOW_ACTION_REGISTRY,
    )

    return WORKFLOW_ACTION_REGISTRY
