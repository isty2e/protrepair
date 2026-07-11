"""Workflow reporting evaluators over terminal structures and planner context."""

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics.geometry import (
    detect_heavy_geometry,
    severe_intrinsic_geometry_residues,
)
from protrepair.diagnostics.parser_readability import (
    rdkit_no_conect_extra_proximity_bond_clusters,
)
from protrepair.diagnostics.parser_topology import (
    ambiguous_disulfide_parser_witness_blockers,
)
from protrepair.scope import ResidueSetScope, Scope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    ClashPresenceState,
    HydrogenCoverageState,
    OrientationCorrectionEligibilityState,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
    StructureBoundaryStateFacts,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
)
from protrepair.state.structure_topology import StructureDisulfideHydrogenFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.refinement.speculative_planning import (
    SpeculativePlanningNodeId,
)
from protrepair.workflow.contracts.planning import (
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
)
from protrepair.workflow.contracts.request import WorkflowGoal
from protrepair.workflow.contracts.result import (
    RequestedGoalOutcome,
    RequestedGoalReport,
    RequestedGoalStatus,
    TransformationResult,
    WorkflowBranchQualityScore,
    WorkflowPhaseOutcome,
    WorkflowPhaseReport,
    WorkflowPhaseStatus,
    WorkflowTerminalBranchOutcome,
)
from protrepair.workflow.planning.assessment.blockers import WorkflowBlocker
from protrepair.workflow.planning.assessment.goal_support import (
    observed_requested_goal_value,
    requested_goal_is_satisfied,
)
from protrepair.workflow.planning.context_projection import (
    planning_context_is_holo_for_structure,
)
from protrepair.workflow.planning.intrinsic_geometry import (
    derive_structure_intrinsic_geometry_facts,
)


class _DeferredWorkflowPhaseReport(WorkflowPhaseReport):
    """Engine-owned lazy phase report that keeps contracts free of evaluators."""

    __slots__ = (
        "_blockers",
        "_cached_outcomes",
        "_component_library",
        "_planning_context",
        "_structure",
    )

    def __init__(
        self,
        structure: ProteinStructure,
        *,
        planning_context: WorkflowPlanningContext,
        component_library: ComponentLibrary | None,
        blockers: tuple[WorkflowBlocker, ...],
    ) -> None:
        object.__setattr__(self, "_outcomes", ())
        object.__setattr__(self, "_structure", structure)
        object.__setattr__(self, "_planning_context", planning_context)
        object.__setattr__(self, "_component_library", component_library)
        object.__setattr__(self, "_blockers", tuple(blockers))
        object.__setattr__(self, "_cached_outcomes", None)

    @property
    def outcomes(self) -> tuple[WorkflowPhaseOutcome, ...]:
        """Return phase outcomes, computing engine projections only on demand."""

        cached_outcomes = self._cached_outcomes
        if cached_outcomes is not None:
            return cached_outcomes

        resolved_outcomes = evaluate_workflow_phase_outcomes(
            self._structure,
            planning_context=self._planning_context,
            component_library=self._component_library,
            blockers=self._blockers,
        )
        object.__setattr__(self, "_cached_outcomes", resolved_outcomes)
        return resolved_outcomes


def deferred_workflow_phase_report(
    structure: ProteinStructure,
    *,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary | None,
    blockers: tuple[WorkflowBlocker, ...],
) -> WorkflowPhaseReport:
    """Create a lazily evaluated phase report from engine-owned context."""

    return _DeferredWorkflowPhaseReport(
        structure,
        planning_context=planning_context,
        component_library=component_library,
        blockers=blockers,
    )


def evaluate_workflow_phase_report(
    structure: ProteinStructure,
    *,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary | None,
    blockers: tuple[WorkflowBlocker, ...],
) -> WorkflowPhaseReport:
    """Eagerly evaluate one workflow phase report."""

    return WorkflowPhaseReport(
        outcomes=evaluate_workflow_phase_outcomes(
            structure,
            planning_context=planning_context,
            component_library=component_library,
            blockers=blockers,
        )
    )


def evaluate_workflow_phase_outcomes(
    structure: ProteinStructure,
    *,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary | None,
    blockers: tuple[WorkflowBlocker, ...],
) -> tuple[WorkflowPhaseOutcome, ...]:
    """Evaluate phase outcomes from narrow fact owners and blocker families."""

    coverage_facts = StructureCoverageFacts.from_structure(
        structure,
        component_library=component_library,
    )
    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure,
        component_library=component_library,
    )
    disulfide_hydrogen_facts = StructureDisulfideHydrogenFacts.from_structure(
        structure
    )
    intrinsic_geometry_facts = derive_structure_intrinsic_geometry_facts(
        structure,
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )
    parser_compatibility_facts = StructureParserCompatibilityFacts.from_structure(
        structure,
        component_library=component_library,
    )
    interaction_facts = (
        StructureInteractionFacts.from_structure(
            structure,
            component_library=component_library,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )
        if planning_context_is_holo_for_structure(planning_context, structure)
        else None
    )
    return (
        _coverage_phase_outcome(
            coverage_facts=coverage_facts,
            blockers=blockers,
        ),
        _chemistry_phase_outcome(
            chemistry_readiness_facts=chemistry_readiness_facts,
            disulfide_hydrogen_facts=disulfide_hydrogen_facts,
            blockers=blockers,
        ),
        _intrinsic_geometry_phase_outcome(
            structure=structure,
            intrinsic_geometry_facts=intrinsic_geometry_facts,
            parser_compatibility_facts=parser_compatibility_facts,
            component_library=component_library,
            blockers=blockers,
        ),
        _interaction_phase_outcome(
            interaction_facts=interaction_facts,
            blockers=blockers,
        ),
    )


def evaluate_requested_goal_report(
    structure: ProteinStructure,
    *,
    requested_goals: tuple[WorkflowGoal, ...],
    component_library: ComponentLibrary | None = None,
    unsupported_requested_goals: tuple[WorkflowGoal, ...] = (),
    blocked_requested_goal_blockers: tuple[
        tuple[
            WorkflowGoal,
            tuple[WorkflowBlocker, ...],
        ],
        ...,
    ] = (),
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...] = (),
) -> RequestedGoalReport:
    """Evaluate requested-goal outcomes against one final structure."""

    unsupported_goal_set = set(unsupported_requested_goals)
    blockers_by_goal = {
        goal: blockers for goal, blockers in blocked_requested_goal_blockers
    }
    already_satisfied_goal_set = set(already_satisfied_requested_goals)
    coverage_facts: StructureCoverageFacts | None = None
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None
    boundary_facts: StructureBoundaryStateFacts | None = None
    snapshot: ProteinStructureSnapshot | None = None

    outcomes: list[RequestedGoalOutcome] = []
    for goal in requested_goals:
        if goal in unsupported_goal_set:
            outcomes.append(
                RequestedGoalOutcome(
                    requested_goal=goal,
                    status=RequestedGoalStatus.UNSUPPORTED,
                )
            )
            continue

        blockers = blockers_by_goal.get(goal)
        if blockers is not None:
            outcomes.append(
                RequestedGoalOutcome(
                    requested_goal=goal,
                    status=RequestedGoalStatus.BLOCKED,
                    blocking_scopes=tuple(blocker.scope for blocker in blockers),
                    blocking_phases=tuple(
                        dict.fromkeys(blocker.phase for blocker in blockers)
                    ),
                )
            )
            continue

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
        if goal in already_satisfied_goal_set:
            outcomes.append(
                RequestedGoalOutcome(
                    requested_goal=goal,
                    status=RequestedGoalStatus.ALREADY_SATISFIED,
                    observed_state=observed_state,
                )
            )
            continue

        outcomes.append(
            RequestedGoalOutcome(
                requested_goal=goal,
                status=(
                    RequestedGoalStatus.SATISFIED
                    if requested_goal_is_satisfied(
                        goal,
                        observed_state=observed_state,
                    )
                    else RequestedGoalStatus.UNMET
                ),
                observed_state=observed_state,
            )
        )

    return RequestedGoalReport(outcomes=tuple(outcomes))


def evaluate_terminal_branch_outcome(
    *,
    node_id: SpeculativePlanningNodeId,
    result: TransformationResult,
    requested_goals: tuple[WorkflowGoal, ...],
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary,
    unsupported_requested_goals: tuple[WorkflowGoal, ...] = (),
    blocked_requested_goal_blockers: tuple[
        tuple[
            WorkflowGoal,
            tuple[WorkflowBlocker, ...],
        ],
        ...,
    ] = (),
    already_satisfied_requested_goals: tuple[WorkflowGoal, ...] = (),
) -> WorkflowTerminalBranchOutcome:
    """Evaluate one terminal branch outcome against requested goals."""

    requested_goal_report = evaluate_requested_goal_report(
        result.structure,
        requested_goals=requested_goals,
        component_library=component_library,
        unsupported_requested_goals=unsupported_requested_goals,
        blocked_requested_goal_blockers=blocked_requested_goal_blockers,
        already_satisfied_requested_goals=already_satisfied_requested_goals,
    )
    return WorkflowTerminalBranchOutcome(
        node_id=node_id,
        requested_goal_report=requested_goal_report,
        phase_report=deferred_workflow_phase_report(
            result.structure,
            planning_context=planning_context,
            component_library=component_library,
            blockers=tuple(
                blocker
                for _, blockers in blocked_requested_goal_blockers
                for blocker in blockers
            ),
        ),
        branch_quality_score=evaluate_workflow_branch_quality_score(
            result,
            requested_goal_report=requested_goal_report,
            planning_context=planning_context,
            component_library=component_library,
            search_depth=node_id.value,
        ),
        error_count=result.error_count(),
        warning_count=result.warning_count(),
        issue_count=result.issue_count(),
    )


def evaluate_workflow_branch_quality_score(
    result: TransformationResult,
    *,
    requested_goal_report: RequestedGoalReport,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary | None,
    search_depth: int = 0,
) -> WorkflowBranchQualityScore:
    """Return the canonical branch score projection for one workflow result."""

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        result.structure,
        component_library=active_component_library,
    )
    intrinsic_facts = derive_structure_intrinsic_geometry_facts(
        result.structure,
        component_library=active_component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )
    parser_facts = None
    if (
        chemistry_readiness_facts.hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    ):
        parser_facts = StructureParserCompatibilityFacts.from_structure(
            result.structure,
            component_library=active_component_library,
        )
    interaction_facts = None
    if planning_context_is_holo_for_structure(
        planning_context,
        result.structure,
    ):
        interaction_facts = StructureInteractionFacts.from_structure(
            result.structure,
            component_library=active_component_library,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )
    geometry_report = detect_heavy_geometry(
        result.structure,
        component_library=active_component_library,
    )
    severe_geometry_residue_count = len(
        severe_intrinsic_geometry_residues(geometry_report)
    )
    return WorkflowBranchQualityScore.from_requested_goal_report(
        requested_goal_report,
        error_count=result.error_count(),
        warning_count=result.warning_count(),
        issue_count=result.issue_count(),
        parser_incompatible=(
            0
            if parser_facts is None
            or not parser_facts.compatibility_state.is_incompatible()
            else 1
        ),
        parser_extra_heavy_bond_count=(
            0 if parser_facts is None else parser_facts.extra_heavy_proximity_bond_count
        ),
        parser_extra_bond_count=(
            0 if parser_facts is None else parser_facts.extra_proximity_bond_count
        ),
        protein_self_clash_count=intrinsic_facts.protein_self_clash_count,
        ligand_aware_clash_count=(
            0
            if interaction_facts is None
            else interaction_facts.ligand_aware_clash_count
        ),
        ligand_aware_worst_overlap_angstrom=(
            0.0
            if interaction_facts is None
            else interaction_facts.ligand_aware_worst_overlap_angstrom
        ),
        ligand_aware_total_overlap_angstrom=(
            0.0
            if interaction_facts is None
            else interaction_facts.ligand_aware_total_overlap_angstrom
        ),
        stereochemistry_violation_count=(
            0
            if intrinsic_facts.stereochemistry_state
            is not StereochemistryState.VIOLATED
            else 1
        ),
        severe_geometry_residue_count=severe_geometry_residue_count,
        search_depth=search_depth,
    )


def _blocking_scopes_for_phase(
    blockers: tuple[WorkflowBlocker, ...],
    *,
    phase: WorkflowPlanningPhase,
) -> tuple[Scope, ...]:
    """Return blocking scopes for one workflow phase in first-seen order."""

    return tuple(
        dict.fromkeys(
            blocker.scope for blocker in blockers if blocker.phase is phase
        )
    )


def _coverage_phase_outcome(
    *,
    coverage_facts: StructureCoverageFacts,
    blockers: tuple[WorkflowBlocker, ...],
) -> WorkflowPhaseOutcome:
    """Return final reporting outcome for the coverage phase."""

    blocking_scopes = _blocking_scopes_for_phase(
        blockers,
        phase=WorkflowPlanningPhase.COVERAGE,
    )
    if blocking_scopes:
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.COVERAGE,
            status=WorkflowPhaseStatus.BLOCKED,
            blocking_scopes=blocking_scopes,
            details="coverage planning stopped on unresolved residue support",
        )
    if (
        coverage_facts.coverage_gaps
        or coverage_facts.backbone_heavy_atom_completeness_state
        is BackboneHeavyAtomCompletenessState.INCOMPLETE
        or coverage_facts.sidechain_heavy_atom_completeness_state
        is SidechainHeavyAtomCompletenessState.INCOMPLETE
    ):
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.COVERAGE,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details="coverage defects remain after terminal planning",
        )

    return WorkflowPhaseOutcome(
        phase=WorkflowPlanningPhase.COVERAGE,
        status=WorkflowPhaseStatus.CLEAR,
    )


def _chemistry_phase_outcome(
    *,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
    disulfide_hydrogen_facts: StructureDisulfideHydrogenFacts,
    blockers: tuple[WorkflowBlocker, ...],
) -> WorkflowPhaseOutcome:
    """Return final reporting outcome for chemistry normalization."""

    blocking_scopes = _blocking_scopes_for_phase(
        blockers,
        phase=WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,
    )
    if blocking_scopes:
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,
            status=WorkflowPhaseStatus.BLOCKED,
            blocking_scopes=blocking_scopes,
            details="chemistry normalization stopped on unresolved support",
        )
    if disulfide_hydrogen_facts.has_contradictions():
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details="disulfide-incompatible thiol hydrogens remain present",
        )
    if not chemistry_readiness_facts.component_support_state.is_fully_supported():
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details="chemistry readiness remains unsupported",
        )
    if (
        chemistry_readiness_facts.hydrogen_coverage_state
        is not HydrogenCoverageState.COMPLETE
    ):
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details="hydrogen completion remains incomplete",
        )

    return WorkflowPhaseOutcome(
        phase=WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,
        status=WorkflowPhaseStatus.CLEAR,
    )


def _intrinsic_geometry_phase_outcome(
    *,
    structure: ProteinStructure,
    intrinsic_geometry_facts: StructureIntrinsicGeometryFacts,
    parser_compatibility_facts: StructureParserCompatibilityFacts,
    component_library: ComponentLibrary | None,
    blockers: tuple[WorkflowBlocker, ...],
) -> WorkflowPhaseOutcome:
    """Return final reporting outcome for intrinsic-geometry correction."""

    blocking_scopes = _blocking_scopes_for_phase(
        blockers,
        phase=WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
    )
    if blocking_scopes:
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
            status=WorkflowPhaseStatus.BLOCKED,
            blocking_scopes=blocking_scopes,
            details="intrinsic geometry correction is blocked",
        )
    if (
        intrinsic_geometry_facts.protein_self_clash_state
        is ClashPresenceState.PRESENT
        or intrinsic_geometry_facts.orientation_correction_eligibility_state
        is not OrientationCorrectionEligibilityState.NOT_ELIGIBLE
        or intrinsic_geometry_facts.stereochemistry_state
        is not StereochemistryState.CONSISTENT
    ):
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details="intrinsic geometry defects remain after terminal planning",
        )
    if parser_compatibility_facts.compatibility_state.is_incompatible():
        parser_compatibility_blocking_scopes = _parser_compatibility_blocking_scopes(
            structure,
            component_library=component_library,
            parser_compatibility_facts=parser_compatibility_facts,
        )
        if parser_compatibility_blocking_scopes:
            return WorkflowPhaseOutcome(
                phase=WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
                status=WorkflowPhaseStatus.BLOCKED,
                blocking_scopes=parser_compatibility_blocking_scopes,
                details=(
                    "parser-visible proximity defects are blocked by topology "
                    "ambiguity"
                ),
            )

        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details=(
                "parser-visible proximity defects remain after terminal planning"
            ),
        )

    return WorkflowPhaseOutcome(
        phase=WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
        status=WorkflowPhaseStatus.CLEAR,
    )


def _parser_compatibility_blocking_scopes(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None,
    parser_compatibility_facts: StructureParserCompatibilityFacts,
) -> tuple[Scope, ...]:
    """Return topology blockers that fully explain parser incompatibility."""

    if not parser_compatibility_facts.compatibility_state.is_incompatible():
        return ()
    if parser_compatibility_facts.extra_heavy_proximity_bond_count <= 0:
        return ()

    parser_witness_clusters = rdkit_no_conect_extra_proximity_bond_clusters(
        structure,
        component_library=component_library,
    )
    if not parser_witness_clusters:
        return ()

    ambiguous_disulfide_blockers = ambiguous_disulfide_parser_witness_blockers(
        structure,
        clusters=parser_witness_clusters,
    )
    if not ambiguous_disulfide_blockers:
        return ()

    blocked_witnesses = {
        witness
        for blocker in ambiguous_disulfide_blockers
        for witness in blocker.witnesses
    }
    all_parser_witnesses = {
        witness for cluster in parser_witness_clusters for witness in cluster.bonds
    }
    if blocked_witnesses != all_parser_witnesses:
        return ()

    return tuple(
        ResidueSetScope(
            residue_ids=blocker.possible_disulfide_residue_ids,
        )
        for blocker in ambiguous_disulfide_blockers
    )


def _interaction_phase_outcome(
    *,
    interaction_facts: StructureInteractionFacts | None,
    blockers: tuple[WorkflowBlocker, ...],
) -> WorkflowPhaseOutcome:
    """Return final reporting outcome for interaction-aware correction."""

    if interaction_facts is None:
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION,
            status=WorkflowPhaseStatus.NOT_APPLICABLE,
        )

    blocking_scopes = _blocking_scopes_for_phase(
        blockers,
        phase=WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION,
    )
    if blocking_scopes:
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION,
            status=WorkflowPhaseStatus.BLOCKED,
            blocking_scopes=blocking_scopes,
            details="interaction-aware correction is blocked",
        )
    if interaction_facts.ligand_aware_clash_state is ClashPresenceState.PRESENT:
        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details="ligand-aware interaction defects remain after terminal planning",
        )

    return WorkflowPhaseOutcome(
        phase=WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION,
        status=WorkflowPhaseStatus.CLEAR,
    )
