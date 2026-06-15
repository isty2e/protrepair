"""Generic workflow planning over requested goals and transformer invocations."""

from dataclasses import dataclass

from protrepair.chemistry import (
    ComponentLibrary,
    build_default_component_library,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.scope import Scope
from protrepair.state import (
    HydrogenCoverageState,
    StructureBoundaryStateFacts,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
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
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
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
    """Generic workflow planning result over the current proposal batch."""

    structure_planning_signature: StructurePlanningSignature
    transformers: tuple[WorkflowStateAction, ...]
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
        object.__setattr__(self, "transformers", tuple(self.transformers))
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
        candidate_transformers: list[WorkflowStateAction],
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
        """Build one outcome from ordered candidate actions and planner facts."""

        return cls(
            structure_planning_signature=structure_planning_signature,
            transformers=cls._current_proposal_batch(candidate_transformers),
            blockers=blockers,
            unsupported_requested_goals=unsupported_requested_goals,
            blocked_requested_goals=blocked_requested_goals,
            already_satisfied_requested_goals=(already_satisfied_requested_goals),
            state_deficit=state_deficit,
        )

    @staticmethod
    def _current_proposal_batch(
        candidate_transformers: list[WorkflowStateAction],
    ) -> tuple[WorkflowStateAction, ...]:
        """Return the current frontier proposal batch from ordered candidates."""

        if not candidate_transformers:
            return ()

        first_family = candidate_transformers[0].proposal_family()
        return tuple(
            transformer
            for transformer in candidate_transformers
            if transformer.proposal_family() is first_family
        )

    def has_transformers(self) -> bool:
        """Return whether the current planning outcome proposes any transformers."""

        return bool(self.transformers)

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
    """Planner memory for one single-branch workflow planning path."""

    adopted_transformers: tuple[WorkflowStateAction, ...] = ()
    attempted_transformers: tuple[WorkflowStateAction, ...] = ()

    def __post_init__(self) -> None:
        adopted_transformers_list: list[WorkflowStateAction] = []
        for adopted_transformer in self.adopted_transformers:
            if adopted_transformer not in adopted_transformers_list:
                adopted_transformers_list.append(adopted_transformer)

        adopted_transformers = tuple(adopted_transformers_list)
        attempted_transformers_list: list[WorkflowStateAction] = []
        for attempted_transformer in (
            *self.attempted_transformers,
            *adopted_transformers,
        ):
            if attempted_transformer not in attempted_transformers_list:
                attempted_transformers_list.append(attempted_transformer)

        attempted_transformers = tuple(attempted_transformers_list)
        object.__setattr__(self, "adopted_transformers", adopted_transformers)
        object.__setattr__(self, "attempted_transformers", attempted_transformers)

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

        return transformer in self.attempted_transformers

    def with_adopted_transformer(
        self,
        transformer: WorkflowStateAction,
    ) -> "WorkflowPlannerMemory":
        """Return one planning state with one additional adopted transformer."""

        if self.has_adopted_transformer(transformer):
            return self

        return type(self)(
            adopted_transformers=self.adopted_transformers + (transformer,),
            attempted_transformers=self.attempted_transformers,
        )

    def with_attempted_transformer(
        self,
        transformer: WorkflowStateAction,
    ) -> "WorkflowPlannerMemory":
        """Return one planning state with one additional attempted transformer."""

        if self.has_attempted_transformer(transformer):
            return self

        return type(self)(
            adopted_transformers=self.adopted_transformers,
            attempted_transformers=self.attempted_transformers + (transformer,),
        )

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
) -> WorkflowPlanningOutcome:
    """Return the generic workflow planning outcome for one canonical structure."""

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

    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=active_component_library,
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
        )
    )
    boundary_facts = StructureBoundaryStateFacts.from_structure(structure)
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
        explicit_repair_refinement_execution_projection = (
            LocalContinuousExecutionResidueProjection.from_scope_spec(
                ProteinStructureSnapshot.from_structure(structure),
                transform_requests.repair_refinement.resolved_execution_scope_spec(),
                context_radius_angstrom=(
                    transform_requests.repair_refinement.config.context_radius_angstrom
                ),
                component_library=active_component_library,
            )
        )
    action_domain = WorkflowActionDomain(
        structure=structure,
        requested_goals=requested_goals,
        transform_requests=transform_requests,
        component_library=active_component_library,
        planning_context=active_planning_context,
        state_deficit=state_deficit,
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        boundary_facts=boundary_facts,
        intrinsic_geometry_facts=intrinsic_geometry_facts,
        parser_compatibility_facts=parser_compatibility_facts,
        interaction_facts=interaction_facts,
        explicit_repair_refinement_execution_projection=(
            explicit_repair_refinement_execution_projection
        ),
        adopted_actions=active_planner_memory.adoption_records(
            registry=active_action_registry,
        ),
    )
    candidate_proposals = [
        proposal
        for registry_entry in active_action_registry
        if registry_entry.is_admissible(action_domain)
        for proposal in registry_entry.proposals(action_domain)
        if not active_planner_memory.has_attempted_transformer(proposal.transformer)
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
        candidate_transformers=[
            proposal.transformer for proposal in ranked_candidate_proposals
        ],
        blockers=blockers,
        unsupported_requested_goals=unsupported_goal_set,
        blocked_requested_goals=blocked_requested_goals,
        already_satisfied_requested_goals=already_satisfied_requested_goals(
            structure,
            requested_goals=tuple(requested_goals),
            component_library=active_component_library,
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
