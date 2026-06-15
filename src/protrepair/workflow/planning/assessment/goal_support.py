"""Capability- and deficit-driven goal support projection for workflow planning."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.scope import (
    ResidueBoundaryScope,
    ResidueSetScope,
    WholeStructureScope,
)
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    ClashPresenceState,
    ClashState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    OxtPresenceState,
    SidechainHeavyAtomCompletenessState,
    StructureBoundaryStateFacts,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
)
from protrepair.workflow.contracts.planning import WorkflowPlanningContext
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowGoal,
    WorkflowGoalStateValue,
)
from protrepair.workflow.planning.assessment.blockers import WorkflowBlocker
from protrepair.workflow.planning.assessment.deficits import (
    WorkflowDeficitDisposition,
    WorkflowStateDeficit,
)
from protrepair.workflow.planning.capability import (
    WorkflowActionCapability,
    WorkflowCapabilityDeficitFamily,
)


@dataclass(frozen=True, slots=True)
class BlockedWorkflowGoalProjection:
    """One goal plus capability-derived blockers that currently stop it."""

    goal: WorkflowGoal
    blockers: tuple[WorkflowBlocker, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(self.blockers))


def unsupported_requested_goals(
    requested_goals: RequestedGoalSet,
    *,
    capabilities: tuple[WorkflowActionCapability, ...],
) -> tuple[WorkflowGoal, ...]:
    """Return requested goals unsupported by the available goal-driven capabilities."""

    return tuple(
        goal
        for goal in requested_goals
        if not _supporting_capabilities(goal=goal, capabilities=capabilities)
    )


def blocked_requested_goal_projections(
    requested_goals: RequestedGoalSet,
    *,
    state_deficit: WorkflowStateDeficit,
    planning_context: WorkflowPlanningContext,
    capabilities: tuple[WorkflowActionCapability, ...],
) -> tuple[BlockedWorkflowGoalProjection, ...]:
    """Return blocked-goal payloads derived from capabilities and current deficits."""

    blocked_goal_projections: list[BlockedWorkflowGoalProjection] = []
    for goal in requested_goals:
        supporting_capabilities = _supporting_capabilities(
            goal=goal,
            capabilities=capabilities,
        )
        if not supporting_capabilities:
            continue

        goal_blockers = tuple(
            dict.fromkeys(
                blocker
                for capability in supporting_capabilities
                for blocker in _blockers_for_capability(
                    capability=capability,
                    state_deficit=state_deficit,
                    planning_context=planning_context,
                )
            )
        )
        if goal_blockers:
            blocked_goal_projections.append(
                BlockedWorkflowGoalProjection(
                    goal=goal,
                    blockers=goal_blockers,
                )
            )

    return tuple(blocked_goal_projections)


def _supporting_capabilities(
    *,
    goal: WorkflowGoal,
    capabilities: tuple[WorkflowActionCapability, ...],
) -> tuple[WorkflowActionCapability, ...]:
    """Return goal-driven capabilities that can advance one requested goal."""

    return tuple(
        capability
        for capability in capabilities
        if capability.supports_proposition(
            scope=goal.scope,
            value=goal.value,
        )
    )


def _blockers_for_capability(
    *,
    capability: WorkflowActionCapability,
    state_deficit: WorkflowStateDeficit,
    planning_context: WorkflowPlanningContext,
) -> tuple[WorkflowBlocker, ...]:
    """Return current blockers visible to one specific workflow capability."""

    del planning_context

    blockers: list[WorkflowBlocker] = []
    for deficit_family in dict.fromkeys(
        (
            *capability.reducible_deficit_families,
            *capability.blocking_deficit_families,
        )
    ):
        blockers.extend(
            _blockers_for_deficit_family(
                deficit_family=deficit_family,
                state_deficit=state_deficit,
            )
        )

    return tuple(dict.fromkeys(blockers))


def _blockers_for_deficit_family(
    *,
    deficit_family: WorkflowCapabilityDeficitFamily,
    state_deficit: WorkflowStateDeficit,
) -> tuple[WorkflowBlocker, ...]:
    """Return blockers projected from one specific deficit family."""

    if deficit_family is WorkflowCapabilityDeficitFamily.ATOM_COVERAGE:
        return tuple(
            WorkflowBlocker.unsupported_component(
                deficit_family=deficit_family,
                scope=ResidueSetScope(residue_ids=(atom_deficit.residue_id,)),
            )
            for atom_deficit in state_deficit.coverage.atom_deficits
            if atom_deficit.disposition is WorkflowDeficitDisposition.BLOCKED
            and atom_deficit.blocked_by_component_support
        )

    if deficit_family is WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS:
        return tuple(
            WorkflowBlocker.unsupported_component(
                deficit_family=deficit_family,
                scope=ResidueSetScope(residue_ids=(residue_id,)),
            )
            for residue_id in dict.fromkeys(
                (
                    *state_deficit.chemistry_readiness.unsupported_component_residue_ids,
                    *state_deficit.chemistry_readiness.hydrogen_blocked_residue_ids,
                )
            )
        )

    return ()


def already_satisfied_requested_goals(
    structure: ProteinStructure,
    *,
    requested_goals: tuple[WorkflowGoal, ...],
    component_library: ComponentLibrary | None = None,
) -> tuple[WorkflowGoal, ...]:
    """Return requested goals already satisfied by the current structure."""

    snapshot: ProteinStructureSnapshot | None = None
    coverage_facts: StructureCoverageFacts | None = None
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None
    boundary_facts: StructureBoundaryStateFacts | None = None
    satisfied_goals: list[WorkflowGoal] = []
    for goal in requested_goals:
        (
            observed_state,
            snapshot,
            coverage_facts,
            chemistry_readiness_facts,
            boundary_facts,
        ) = observed_requested_goal_value(
            structure,
            snapshot=snapshot,
            coverage_facts=coverage_facts,
            chemistry_readiness_facts=chemistry_readiness_facts,
            boundary_facts=boundary_facts,
            goal=goal,
            component_library=component_library,
        )
        if requested_goal_is_satisfied(
            goal,
            observed_state=observed_state,
        ):
            satisfied_goals.append(goal)

    return tuple(satisfied_goals)


def observed_requested_goal_value(
    structure: ProteinStructure,
    *,
    snapshot: ProteinStructureSnapshot | None,
    coverage_facts: StructureCoverageFacts | None,
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None,
    boundary_facts: StructureBoundaryStateFacts | None,
    goal: WorkflowGoal,
    component_library: ComponentLibrary | None = None,
) -> tuple[
    WorkflowGoalStateValue | None,
    ProteinStructureSnapshot | None,
    StructureCoverageFacts | None,
    StructureChemistryReadinessFacts | None,
    StructureBoundaryStateFacts | None,
]:
    """Return the observed value for one requested goal when evaluable."""

    scope = goal.scope
    value = goal.value
    if scope == WholeStructureScope():
        active_coverage_facts = coverage_facts
        active_chemistry_readiness_facts = chemistry_readiness_facts
        if (
            isinstance(
                value,
                (
                    BackboneHeavyAtomCompletenessState,
                    SidechainHeavyAtomCompletenessState,
                ),
            )
            and active_coverage_facts is None
        ):
            active_coverage_facts = StructureCoverageFacts.from_structure(
                structure,
                component_library=component_library,
            )
        if (
            isinstance(value, HydrogenCoverageState)
            and active_chemistry_readiness_facts is None
        ):
            active_chemistry_readiness_facts = (
                StructureChemistryReadinessFacts.from_structure(
                    structure,
                    component_library=component_library,
                )
            )
        if (
            isinstance(
                value,
                (
                    BackboneHeavyAtomCompletenessState,
                    SidechainHeavyAtomCompletenessState,
                    HydrogenCoverageState,
                    ClashPresenceState,
                ),
            )
        ):
            if (
                isinstance(value, ClashPresenceState)
                and active_chemistry_readiness_facts is None
            ):
                active_chemistry_readiness_facts = (
                    StructureChemistryReadinessFacts.from_structure(
                        structure,
                        component_library=component_library,
                    )
                )
            return (
                _whole_structure_goal_observed_state_value(
                    structure=structure,
                    coverage_facts=active_coverage_facts,
                    chemistry_readiness_facts=active_chemistry_readiness_facts,
                    goal_state_value=value,
                    component_library=component_library,
                ),
                snapshot,
                active_coverage_facts,
                active_chemistry_readiness_facts,
                boundary_facts,
            )
    if isinstance(value, OxtPresenceState):
        if not isinstance(scope, ResidueBoundaryScope):
            return (
                None,
                snapshot,
                coverage_facts,
                chemistry_readiness_facts,
                boundary_facts,
            )

        active_boundary_facts = boundary_facts
        if active_boundary_facts is None:
            active_boundary_facts = StructureBoundaryStateFacts.from_structure(
                structure,
            )
        return (
            active_boundary_facts.terminal_boundary_observation.oxt_presence_for_scope(
                scope
            ),
            snapshot,
            coverage_facts,
            chemistry_readiness_facts,
            active_boundary_facts,
        )

    observed_state, active_snapshot = _lowered_scope_observed_state_value(
        structure,
        snapshot=snapshot,
        goal=goal,
        component_library=component_library,
    )
    return (
        observed_state,
        active_snapshot,
        coverage_facts,
        chemistry_readiness_facts,
        boundary_facts,
    )


def requested_goal_is_satisfied(
    goal: WorkflowGoal,
    *,
    observed_state: WorkflowGoalStateValue | None,
) -> bool:
    """Return whether one observed value satisfies the requested goal."""

    return observed_state == goal.value


def _lowered_scope_observed_state_value(
    structure: ProteinStructure,
    *,
    snapshot: ProteinStructureSnapshot | None,
    goal: WorkflowGoal,
    component_library: ComponentLibrary | None = None,
) -> tuple[WorkflowGoalStateValue | None, ProteinStructureSnapshot | None]:
    """Return one lowered-scope observed value plus the snapshot used to derive it."""

    scope = goal.scope
    value = goal.value
    if not OBSERVED_ATOM_SCOPE_LOWERING.supports(scope):
        return None, snapshot

    active_snapshot = snapshot
    if active_snapshot is None:
        active_snapshot = ProteinStructureSnapshot.from_structure(structure)

    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        scope,
        carrier=active_snapshot,
    )
    atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
        active_snapshot,
        atom_scope,
        component_library=component_library,
    )
    selected_scope_facts = atom_scope_facts.selected_scope_facts
    component_support_state = (
        selected_scope_facts.structure_facts.component_support_fact.value
    )
    if not component_support_state.is_fully_supported() and isinstance(
        value,
        (
            BackboneHeavyAtomCompletenessState,
            SidechainHeavyAtomCompletenessState,
            HydrogenCoverageState,
        ),
    ):
        return None, active_snapshot
    if isinstance(value, ClashState):
        return selected_scope_facts.clash_fact.value, active_snapshot
    if isinstance(value, BackboneHeavyAtomCompletenessState):
        return (
            selected_scope_facts.structure_facts.backbone_heavy_atom_completeness_fact.value,
            active_snapshot,
        )
    if isinstance(value, SidechainHeavyAtomCompletenessState):
        return (
            selected_scope_facts.structure_facts.sidechain_heavy_atom_completeness_fact.value,
            active_snapshot,
        )
    if isinstance(value, HydrogenCoverageState):
        return (
            selected_scope_facts.structure_facts.hydrogen_coverage_fact.value,
            active_snapshot,
        )

    return None, active_snapshot


def _whole_structure_goal_observed_state_value(
    *,
    structure: ProteinStructure,
    coverage_facts: StructureCoverageFacts | None,
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None,
    goal_state_value: WorkflowGoalStateValue,
    component_library: ComponentLibrary | None = None,
) -> WorkflowGoalStateValue | None:
    """Return the observed whole-structure value from narrow fact owners."""

    if isinstance(
        goal_state_value,
        (
            BackboneHeavyAtomCompletenessState,
            SidechainHeavyAtomCompletenessState,
        ),
    ):
        if coverage_facts is None:
            return None
        if isinstance(goal_state_value, BackboneHeavyAtomCompletenessState):
            return coverage_facts.backbone_heavy_atom_completeness_state
        return coverage_facts.sidechain_heavy_atom_completeness_state

    if isinstance(goal_state_value, HydrogenCoverageState):
        if chemistry_readiness_facts is None:
            return None
        return _whole_structure_hydrogen_goal_observed_state(
            chemistry_readiness_facts
        )

    if isinstance(goal_state_value, ClashPresenceState):
        if chemistry_readiness_facts is None:
            return None
        return _whole_structure_clash_goal_observed_state(
            structure,
            component_library=component_library,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )

    return None


def _whole_structure_clash_goal_observed_state(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
) -> ClashPresenceState:
    """Return aggregate whole-structure intrinsic and ligand-aware clash presence."""

    intrinsic_geometry_facts = StructureIntrinsicGeometryFacts.from_structure(
        structure,
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )
    if (
        intrinsic_geometry_facts.protein_self_clash_state
        is ClashPresenceState.PRESENT
    ):
        return ClashPresenceState.PRESENT

    if not structure.constitution.ligands:
        return ClashPresenceState.NONE

    interaction_facts = StructureInteractionFacts.from_structure(
        structure,
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )
    if interaction_facts.ligand_aware_clash_state is ClashPresenceState.PRESENT:
        return ClashPresenceState.PRESENT

    return ClashPresenceState.NONE


def _whole_structure_hydrogen_goal_observed_state(
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
) -> HydrogenCoverageState | None:
    """Return one whole-structure hydrogen state over retained components."""

    if not chemistry_readiness_facts.component_support_state.is_fully_supported():
        return None
    if any(
        not retained_fact.is_supported()
        for retained_fact in chemistry_readiness_facts.retained_non_polymer_facts
    ):
        return None

    applicable_coverage_states: list[HydrogenCoverageState] = []
    if (
        chemistry_readiness_facts.hydrogen_applicability_state
        is not HydrogenApplicabilityState.NOT_APPLICABLE
    ):
        applicable_coverage_states.append(
            chemistry_readiness_facts.hydrogen_coverage_state
        )

    applicable_coverage_states.extend(
        retained_fact.hydrogen_coverage_state
        for retained_fact in chemistry_readiness_facts.retained_non_polymer_facts
        if (
            retained_fact.hydrogen_applicability_state
            is not HydrogenApplicabilityState.NOT_APPLICABLE
        )
    )
    if not applicable_coverage_states:
        return HydrogenCoverageState.COMPLETE
    if all(
        coverage_state is HydrogenCoverageState.COMPLETE
        for coverage_state in applicable_coverage_states
    ):
        return HydrogenCoverageState.COMPLETE
    if all(
        coverage_state is HydrogenCoverageState.NONE
        for coverage_state in applicable_coverage_states
    ):
        return HydrogenCoverageState.NONE
    return HydrogenCoverageState.PARTIAL
