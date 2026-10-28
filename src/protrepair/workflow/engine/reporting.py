"""Workflow reporting evaluators over terminal structures and planner context."""

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.geometry import (
    detect_heavy_geometry,
    severe_intrinsic_geometry_residues,
)
from protrepair.diagnostics.kinds import IssueSeverity
from protrepair.diagnostics.parser_readability import (
    rdkit_no_conect_extra_proximity_bond_clusters,
)
from protrepair.diagnostics.parser_topology import (
    ambiguous_disulfide_parser_witness_blockers,
)
from protrepair.scope import ResidueSetScope, Scope, WholeStructureScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
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
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.structure_topology import StructureDisulfideHydrogenFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.refinement.spec import RepairRefinementSpec
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
    WorkflowBranchOutcome,
    WorkflowBranchQualityScore,
    WorkflowBranchStatus,
    WorkflowPhaseOutcome,
    WorkflowPhaseReport,
    WorkflowPhaseStatus,
)
from protrepair.workflow.engine.hydrogen import project_hydrogen_issues
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


class _DeferredWorkflowPhaseReport(WorkflowPhaseReport):  # pyrefly: ignore[bad-class-definition]  # Explicit slots are valid on this undecorated subclass.
    """Engine-owned lazy phase report that keeps contracts free of evaluators."""

    __slots__ = (
        "_blockers",
        "_cached_outcomes",
        "_component_library",
        "_planning_context",
        "_structure",
    )

    _blockers: tuple[WorkflowBlocker, ...]
    _cached_outcomes: tuple[WorkflowPhaseOutcome, ...] | None
    _component_library: ComponentLibrary | None
    _planning_context: WorkflowPlanningContext
    _structure: ProteinStructure

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

    observation = StructureChemistryObservation(
        structure, component_library=component_library
    )
    coverage_facts, chemistry_readiness_facts = observation.whole_structure_facts()
    disulfide_hydrogen_facts = StructureDisulfideHydrogenFacts.from_structure(structure)
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
    chemistry_observation: StructureChemistryObservation | None = None,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence, ...
    ] = (),
) -> RequestedGoalReport:
    """Evaluate requested-goal outcomes against one final structure.

    Parameters
    ----------
    structure : ProteinStructure
        Snapshot being observed.
    requested_goals : tuple[WorkflowGoal, ...]
        Goals in reporting order.
    component_library : ComponentLibrary or None
        Active definitions, or the default library.
    unsupported_requested_goals : tuple[WorkflowGoal, ...]
        Goals to report as unsupported without observing their state.
    blocked_requested_goal_blockers : tuple
        Goals blocked by the listed scopes and phases.
    already_satisfied_requested_goals : tuple[WorkflowGoal, ...]
        Initially satisfied goals, still checked against the final state.
    chemistry_observation : StructureChemistryObservation or None
        Shared observation with matching snapshot, library and evidence.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit chemistry used for both goal observation and generation.

    Returns
    -------
    RequestedGoalReport
        Observed outcomes with blocker and initial-satisfaction distinctions.

    Raises
    ------
    ValueError
        The observation has different interpretation inputs.
    """

    observation = chemistry_observation
    if observation is None:
        observation = StructureChemistryObservation(
            structure,
            component_library=component_library,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
        )
    else:
        observation.require_context(
            structure,
            component_library=component_library,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
        )

    unsupported_goal_set = set(unsupported_requested_goals)
    blockers_by_goal = {
        goal: blockers for goal, blockers in blocked_requested_goal_blockers
    }
    already_satisfied_goal_set = set(already_satisfied_requested_goals)
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

        observed_state, snapshot, boundary_facts = observed_requested_goal_value(
            observation,
            snapshot=snapshot,
            boundary_facts=boundary_facts,
            goal=goal,
        )
        satisfied = requested_goal_is_satisfied(goal, observed_state=observed_state)
        if satisfied and goal in already_satisfied_goal_set:
            outcomes.append(
                RequestedGoalOutcome(
                    requested_goal=goal,
                    status=RequestedGoalStatus.ALREADY_SATISFIED,
                    observed_state=observed_state,
                )
            )
            continue

        blocking_scopes: tuple[Scope, ...] = ()
        details = None
        if not satisfied and isinstance(goal.value, HydrogenCoverageState):
            if isinstance(goal.scope, WholeStructureScope):
                hydrogen_facts = observation.whole_structure_facts()[1]
            elif OBSERVED_ATOM_SCOPE_LOWERING.supports(goal.scope):
                if snapshot is None:
                    snapshot = ProteinStructureSnapshot.from_structure(structure)
                atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
                    goal.scope, carrier=snapshot
                )
                hydrogen_facts = observation.projected_facts(
                    tuple(dict.fromkeys(ref.residue_id for ref in atom_scope.atom_refs))
                )[1]
            else:
                hydrogen_facts = None
            failures = (
                ()
                if hydrogen_facts is None
                else tuple(
                    fact
                    for fact in hydrogen_facts.retained_non_polymer_facts
                    if fact.chemistry_failure is not None
                    and not fact.has_complete_reference_hydrogens()
                )
            )
            blocking_scopes = tuple(
                ResidueSetScope((fact.residue_id,)) for fact in failures
            )
            polymer_failures = (
                ()
                if hydrogen_facts is None
                else tuple(
                    (fact.residue_id, assessment.failure)
                    for fact in hydrogen_facts.residue_facts
                    for assessment in (
                        observation.hydrogen_expectation_model.polymer_reference_assessments.get(
                            fact.residue_id
                        ),
                    )
                    if assessment is not None and assessment.failure is not None
                )
            )
            blocking_scopes += tuple(
                ResidueSetScope((residue_id,)) for residue_id, _ in polymer_failures
            )
            details = (
                "; ".join(
                    f"{fact.residue_id.display_token()} {fact.component_id}: "
                    f"{fact.chemistry_failure.detail}"
                    for fact in failures
                    if fact.chemistry_failure is not None
                )
                or None
            )
            if polymer_failures:
                details = "; ".join(
                    filter(
                        None,
                        (
                            details,
                            *(
                                f"{residue_id.display_token()}: {failure.detail}"
                                for residue_id, failure in polymer_failures
                            ),
                        ),
                    )
                )
        outcomes.append(
            RequestedGoalOutcome(
                requested_goal=goal,
                status=(
                    RequestedGoalStatus.SATISFIED
                    if satisfied
                    else RequestedGoalStatus.UNMET
                ),
                observed_state=observed_state,
                blocking_scopes=blocking_scopes,
                details=details,
            )
        )

    return RequestedGoalReport(outcomes=tuple(outcomes))


def evaluate_workflow_branch_outcome(
    *,
    node_id: SpeculativePlanningNodeId,
    result: TransformationResult,
    requested_goal_report: RequestedGoalReport,
    branch_quality_score: WorkflowBranchQualityScore,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary,
    blockers: tuple[WorkflowBlocker, ...] = (),
    status: WorkflowBranchStatus = WorkflowBranchStatus.TERMINAL,
    relaxation_executions: tuple[ContinuousRelaxationExecution, ...] = (),
) -> WorkflowBranchOutcome:
    """Build one retained candidate outcome from already-evaluated branch facts."""

    return WorkflowBranchOutcome(
        node_id=node_id,
        requested_goal_report=requested_goal_report,
        phase_report=deferred_workflow_phase_report(
            result.structure,
            planning_context=planning_context,
            component_library=component_library,
            blockers=blockers,
        ),
        branch_quality_score=branch_quality_score,
        error_count=branch_quality_score.error_count,
        warning_count=branch_quality_score.warning_count,
        issue_count=branch_quality_score.issue_count,
        status=status,
        relaxation_executions=relaxation_executions,
    )


def evaluate_workflow_branch_quality_score(
    result: TransformationResult,
    *,
    requested_goal_report: RequestedGoalReport,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary | None,
    search_depth: int = 0,
    chemistry_observation: StructureChemistryObservation | None = None,
    repair_refinement: RepairRefinementSpec | None = None,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence, ...
    ] = (),
) -> WorkflowBranchQualityScore:
    """Return the canonical branch score projection for one workflow result.

    Parameters
    ----------
    result : TransformationResult
        Result whose structure and diagnostics are scored.
    requested_goal_report : RequestedGoalReport
        Goal observations for this result.
    planning_context : WorkflowPlanningContext
        Interaction context for assessment.
    component_library : ComponentLibrary or None
        Active definitions, or the default library.
    search_depth : int
        Branch depth used by the preference policy.
    chemistry_observation : StructureChemistryObservation or None
        Shared observation with matching snapshot, library and explicit evidence.
    repair_refinement : RepairRefinementSpec or None
        Local request whose included region requires current H diagnostics.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit chemistry shared by observation and generation.

    Returns
    -------
    WorkflowBranchQualityScore
        Policy-independent quality axes.

    Raises
    ------
    ValueError
        The observation has different interpretation inputs.
    """

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    observation = chemistry_observation
    if observation is None:
        observation = StructureChemistryObservation(
            result.structure,
            component_library=active_component_library,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
        )
    else:
        observation.require_context(
            result.structure,
            component_library=active_component_library,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
        )
    _, chemistry_readiness_facts = observation.whole_structure_facts()
    # Eligibility searches prospective corrections but is not a score axis.
    # Keep these facts local; planning and phase reports derive eligibility.
    intrinsic_facts = StructureIntrinsicGeometryFacts.from_structure(
        result.structure,
        component_library=active_component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )
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
    issues = project_hydrogen_issues(
        result,
        requested_goal_report=requested_goal_report,
        component_library=active_component_library,
        repair_refinement=repair_refinement,
        preparation=observation.hydrogen_expectation_model.polymer_preparation,
        chemistry_observation=observation,
    )
    return WorkflowBranchQualityScore.from_requested_goal_report(
        requested_goal_report,
        error_count=sum(issue.severity is IssueSeverity.ERROR for issue in issues),
        warning_count=sum(issue.severity is IssueSeverity.WARNING for issue in issues),
        issue_count=len(issues),
        parser_incompatible=(
            0 if not parser_facts.compatibility_state.is_incompatible() else 1
        ),
        parser_extra_heavy_bond_count=(parser_facts.extra_heavy_proximity_bond_count),
        parser_extra_bond_count=(parser_facts.extra_proximity_bond_count),
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
        dict.fromkeys(blocker.scope for blocker in blockers if blocker.phase is phase)
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
        intrinsic_geometry_facts.protein_self_clash_state is ClashPresenceState.PRESENT
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
                    "parser-visible proximity defects are blocked by topology ambiguity"
                ),
            )

        return WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
            status=WorkflowPhaseStatus.UNRESOLVED,
            details=("parser-visible proximity defects remain after terminal planning"),
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
