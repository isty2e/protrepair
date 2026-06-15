"""Direct regression tests for canonical transformation legality decisions."""

from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    CarrierScopedState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureProjectionStateFacts,
)
from protrepair.state.domain import (
    AtomScopeStateFacts,
    ClashState,
    ContinuousBondRealizabilityFacts,
    ContinuousBondRealizabilityState,
    ContinuousRegionReadinessFacts,
    HydrogenAttachmentResolutionFacts,
    HydrogenAttachmentResolutionObservation,
    HydrogenAttachmentResolutionState,
    ResidueTopologyAvailabilityFact,
    SelectedAtomScopeFacts,
    TopologyAvailabilityAspect,
    TopologyAvailabilityFacts,
    TopologyAvailabilityState,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.workflow.planning.completion import (
    WorkflowAnchorAtomPairExecutionScope,
    WorkflowCompletionPartitionKind,
    WorkflowCompletionStageKind,
    WorkflowCompositeExecutionScope,
    WorkflowExecutionScopeKind,
    WorkflowExecutionStage,
    WorkflowResidueSetExecutionScope,
    workflow_legal_completion_plans,
)
from protrepair.workflow.planning.transformation.legality import (
    LocalPreparationReason,
    LocalTransformationStratum,
    selected_region_legal_transformations,
    selected_region_termination_decision,
)
from protrepair.workflow.planning.transformation.runtime import (
    LocalTransformationFamily,
    TransformationPlanningMemory,
    TransformationTerminationReason,
)


def _manual_legality_state_facts(
    *,
    selected_scope: ResidueSetScope,
    atom_count: int,
    hydrogen_atom_count: int,
    component_support_state: ComponentSupportState,
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState,
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState,
    hydrogen_applicability_state: HydrogenApplicabilityState,
    hydrogen_coverage_state: HydrogenCoverageState,
    stereochemistry_state: StereochemistryState,
    clash_state: ClashState,
    heavy_atom_topology_state: TopologyAvailabilityState,
    hydrogen_topology_state: TopologyAvailabilityState,
    hydrogen_attachment_resolution_state: HydrogenAttachmentResolutionState
    | None = None,
    continuous_bond_realizability_state: ContinuousBondRealizabilityState | None = None,
    continuous_bond_realizability_blocker: str | None = None,
) -> tuple[StructureProjectionStateFacts, AtomScopeStateFacts]:
    """Return primitive state facts for one manual legality fixture."""

    carrier = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="manual-legality-state",
    )
    structure_facts = StructureProjectionStateFacts(
        chain_count=1,
        residue_count=len(selected_scope.residue_ids),
        ligand_count=0,
        atom_count=atom_count,
        hydrogen_atom_count=hydrogen_atom_count,
        component_support_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=component_support_state,
        ),
        backbone_heavy_atom_completeness_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=backbone_heavy_atom_completeness_state,
        ),
        sidechain_heavy_atom_completeness_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=sidechain_heavy_atom_completeness_state,
        ),
        hydrogen_applicability_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=hydrogen_applicability_state,
        ),
        hydrogen_coverage_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=hydrogen_coverage_state,
        ),
        stereochemistry_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=stereochemistry_state,
        ),
    )
    resolved_hydrogen_attachment_state = hydrogen_attachment_resolution_state
    if resolved_hydrogen_attachment_state is None:
        if hydrogen_topology_state is TopologyAvailabilityState.PRESENT:
            resolved_hydrogen_attachment_state = (
                HydrogenAttachmentResolutionState.TEMPLATE_RESOLVED
            )
        elif hydrogen_topology_state is TopologyAvailabilityState.NOT_APPLICABLE:
            resolved_hydrogen_attachment_state = (
                HydrogenAttachmentResolutionState.NOT_APPLICABLE
            )
        else:
            resolved_hydrogen_attachment_state = (
                HydrogenAttachmentResolutionState.ABSENT
            )

    resolved_continuous_bond_realizability_state = continuous_bond_realizability_state
    if resolved_continuous_bond_realizability_state is None:
        if heavy_atom_topology_state is TopologyAvailabilityState.PRESENT and (
            hydrogen_topology_state
            in {
                TopologyAvailabilityState.PRESENT,
                TopologyAvailabilityState.NOT_APPLICABLE,
            }
        ):
            resolved_continuous_bond_realizability_state = (
                ContinuousBondRealizabilityState.REALIZABLE
            )
        else:
            resolved_continuous_bond_realizability_state = (
                ContinuousBondRealizabilityState.UNREALIZABLE
            )

    resolved_continuous_bond_realizability_blocker = (
        continuous_bond_realizability_blocker
    )
    if (
        resolved_continuous_bond_realizability_state
        is ContinuousBondRealizabilityState.REALIZABLE
    ):
        resolved_continuous_bond_realizability_blocker = None
    elif resolved_continuous_bond_realizability_blocker is None:
        resolved_continuous_bond_realizability_blocker = (
            "continuous relaxation requires a realizable selected-scope bond "
            "graph before any force field can be bound"
        )

    return structure_facts, AtomScopeStateFacts(
        selected_scope_facts=SelectedAtomScopeFacts(
            atom_count=atom_count,
            residue_count=len(selected_scope.residue_ids),
            hydrogen_atom_count=hydrogen_atom_count,
            structure_facts=structure_facts,
            clash_fact=CarrierScopedState(
                carrier=carrier,
                scope=selected_scope,
                value=clash_state,
            ),
            topology_availability_facts=TopologyAvailabilityFacts(
                carrier=carrier,
                scope=selected_scope,
                residue_facts=tuple(
                    residue_fact
                    for residue_id in selected_scope.residue_ids
                    for residue_fact in (
                        ResidueTopologyAvailabilityFact(
                            residue_id=residue_id,
                            aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                            state=heavy_atom_topology_state,
                        ),
                        ResidueTopologyAvailabilityFact(
                            residue_id=residue_id,
                            aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                            state=hydrogen_topology_state,
                        ),
                    )
                ),
            ),
            hydrogen_attachment_resolution_facts=HydrogenAttachmentResolutionFacts(
                carrier=carrier,
                scope=selected_scope,
                residue_facts=HydrogenAttachmentResolutionObservation.uniform(
                    selected_scope.residue_ids,
                    state=resolved_hydrogen_attachment_state,
                ).residue_facts,
            ),
            continuous_bond_realizability_facts=ContinuousBondRealizabilityFacts(
                carrier=carrier,
                scope=selected_scope,
                state=resolved_continuous_bond_realizability_state,
                blocker=resolved_continuous_bond_realizability_blocker,
            ),
        ),
        continuous_region_readiness_facts=ContinuousRegionReadinessFacts(
            coverage_facts=StructureCoverageFacts(
                carrier=carrier,
                residue_facts=(),
                blueprint_coverages=(),
                coverage_gaps=(),
                backbone_heavy_atom_completeness_state=(
                    structure_facts.backbone_heavy_atom_completeness_fact.value
                ),
                sidechain_heavy_atom_completeness_state=(
                    structure_facts.sidechain_heavy_atom_completeness_fact.value
                ),
            ),
            chemistry_readiness_facts=StructureChemistryReadinessFacts(
                carrier=carrier,
                residue_facts=(),
                retained_non_polymer_facts=(),
                component_support_state=structure_facts.component_support_fact.value,
                heavy_atom_topology_availability_state=heavy_atom_topology_state,
                hydrogen_topology_availability_state=hydrogen_topology_state,
                hydrogen_applicability_state=(
                    structure_facts.hydrogen_applicability_fact.value
                ),
                hydrogen_coverage_state=structure_facts.hydrogen_coverage_fact.value,
            ),
        ),
    )


def test_workflow_legal_transformations_route_hydrogen_only_plan() -> None:
    """Hydrogen-only continuation should be explicit for heavy-complete input."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-hydrogen-legality",
    )

    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=build_default_component_library(),
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=True,
    )

    assert not legal_plans.is_empty()
    assert legal_plans.plans == (
        legal_plans.plan_for_execution_stages(
            (WorkflowCompletionStageKind.HYDROGEN_COMPLETION,)
        ),
    )
    assert legal_plans.plans[0].partition_kinds() == (
        WorkflowCompletionPartitionKind.HYDROGEN_ONLY,
    )


def test_workflow_legal_transformations_route_heavy_then_hydrogen_plan() -> None:
    """Heavy-incomplete hydrogenation should still expose the composite plan."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-heavy-then-hydrogen-legality",
    )

    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=build_default_component_library(),
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=True,
    )

    assert not legal_plans.is_empty()
    assert legal_plans.plans == (
        legal_plans.plan_for_execution_stages(
            (
                WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,
                WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
            )
        ),
    )
    assert legal_plans.plans[0].partition_kinds() == (
        WorkflowCompletionPartitionKind.HEAVY_THEN_HYDROGEN,
    )


def test_workflow_legal_transformations_partition_heterogeneous_subsets() -> None:
    """Heterogeneous structures should expose partitioned subset semantics."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("UNK", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-heterogeneous-legality",
    )

    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=build_default_component_library(),
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=True,
    )

    assert not legal_plans.is_empty()
    assert legal_plans.plans[0].execution_stages() == (
        WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
    )
    assert legal_plans.plans[0].partition_kinds() == (
        WorkflowCompletionPartitionKind.HYDROGEN_ONLY,
        WorkflowCompletionPartitionKind.UNSUPPORTED_STOP,
    )


def test_workflow_legal_transformations_order_stage_union_by_workflow_spine() -> None:
    """Mixed subset plans should project onto canonical workflow stage order."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-stage-union-order",
    )

    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=build_default_component_library(),
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=True,
    )

    assert not legal_plans.is_empty()
    assert legal_plans.plans[0].execution_stages() == (
        WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,
        WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
    )
    assert legal_plans.plans[0].execution_plan() == (
        WorkflowExecutionStage(
            kind=WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,
            scope=WorkflowResidueSetExecutionScope(
                residue_ids=(ResidueId(chain_id="A", seq_num=2),)
            ),
        ),
        WorkflowExecutionStage(
            kind=WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
            scope=WorkflowResidueSetExecutionScope(
                residue_ids=(
                    ResidueId(chain_id="A", seq_num=1),
                    ResidueId(chain_id="A", seq_num=2),
                )
            ),
        ),
    )
    assert legal_plans.plans[0].partition_kinds() == (
        WorkflowCompletionPartitionKind.HYDROGEN_ONLY,
        WorkflowCompletionPartitionKind.HEAVY_THEN_HYDROGEN,
    )


def test_workflow_execution_stage_supports_relational_scope_variants() -> None:
    """Workflow execution stages should admit richer future scope nouns."""

    stage = WorkflowExecutionStage(
        kind=WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
        scope=WorkflowAnchorAtomPairExecutionScope(
            left_anchor_atom_ref=AtomRef(
                residue_id=ResidueId(chain_id="A", seq_num=10),
                atom_name="C",
            ),
            right_anchor_atom_ref=AtomRef(
                residue_id=ResidueId(chain_id="A", seq_num=11),
                atom_name="N",
            ),
        ),
    )

    assert stage.scope.kind is WorkflowExecutionScopeKind.ANCHOR_ATOM_PAIR
    assert stage.scope.display_tokens() == ("A:10.C", "A:11.N")


def test_workflow_execution_stage_supports_composite_scope_variants() -> None:
    """Workflow execution stages should preserve mixed primitive scope roles."""

    stage = WorkflowExecutionStage(
        kind=WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,
        scope=WorkflowCompositeExecutionScope(
            scopes=(
                WorkflowResidueSetExecutionScope(
                    residue_ids=(ResidueId(chain_id="A", seq_num=10),)
                ),
                WorkflowAnchorAtomPairExecutionScope(
                    left_anchor_atom_ref=AtomRef(
                        residue_id=ResidueId(chain_id="A", seq_num=10),
                        atom_name="C",
                    ),
                    right_anchor_atom_ref=AtomRef(
                        residue_id=ResidueId(chain_id="A", seq_num=13),
                        atom_name="N",
                    ),
                ),
            )
        ),
    )

    assert stage.scope.kind is WorkflowExecutionScopeKind.COMPOSITE
    assert stage.scope.display_tokens() == (
        "residue_set:A:10",
        "anchor_atom_pair:A:10.C",
        "anchor_atom_pair:A:13.N",
    )


def test_selected_region_termination_detects_cycle_on_revisited_exhausted_state() -> (
    None
):
    """Repeatedly exhausting the same observed signature should terminate as cycle."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_legality_state_facts(
        selected_scope=selected_scope,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.PRESENT,
    )
    first_legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=False,
        discrete_seeding_applicable=False,
    )
    continuous_signature = first_legal_transformations.signature_for_family(
        LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION
    )
    revisiting_run_state = (
        TransformationPlanningMemory.initial(max_steps=3)
        .with_observed_signature(first_legal_transformations.planning_signature)
        .with_attempted_family(continuous_signature)
    )

    revisited_legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=revisiting_run_state,
        discrete_preparation_applicable=False,
        discrete_seeding_applicable=False,
    )
    termination = selected_region_termination_decision(
        planning_memory=revisiting_run_state,
        legal_transformations=revisited_legal_transformations,
    )

    assert revisited_legal_transformations.is_empty()
    assert revisited_legal_transformations.is_exhausted_by_run_history()
    assert termination.reason is TransformationTerminationReason.CYCLE_DETECTED


def test_selected_region_preparation_remains_legal_before_relaxation_ready() -> None:
    """Topology and candidate preparation should remain legal before relax."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_legality_state_facts(
        selected_scope=selected_scope,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.PRESENT,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.ABSENT,
    )

    legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=True,
        discrete_seeding_applicable=True,
    )

    assert legal_transformations.contains_stratum(
        LocalTransformationStratum.PREPARATION
    )
    assert legal_transformations.contains_stratum(
        LocalTransformationStratum.CANDIDATE_CONSTRUCTION
    )
    assert not legal_transformations.contains_stratum(
        LocalTransformationStratum.RELAXATION
    )
    assert legal_transformations.contains_family(
        LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
    )
    assert legal_transformations.contains_family(
        LocalTransformationFamily.BRANCHED_SIDECHAIN_SEED
    )
    assert not legal_transformations.contains_family(
        LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION
    )


def test_selected_region_topology_preparation_tracks_unrealizable_bond_graph() -> None:
    """Topology-preparation should track unrealizable bond planning semantics."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_legality_state_facts(
        selected_scope=selected_scope,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.PRESENT,
        continuous_bond_realizability_state=(
            ContinuousBondRealizabilityState.UNREALIZABLE
        ),
        continuous_bond_realizability_blocker=(
            "joint selected scope cannot realize one continuous bond graph"
        ),
    )

    legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=True,
        discrete_seeding_applicable=True,
    )

    assert legal_transformations.contains_stratum(
        LocalTransformationStratum.PREPARATION
    )
    assert not legal_transformations.contains_stratum(
        LocalTransformationStratum.RELAXATION
    )
    assert (
        legal_transformations.family_record_for(
            LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
        ).stratum
        is LocalTransformationStratum.PREPARATION
    )
    assert (
        legal_transformations.family_record_for(
            LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
        ).preparation_reason
        is LocalPreparationReason.TOPOLOGY_PRECONDITION
    )


def test_selected_region_ready_state_separates_chemistry_and_seed() -> None:
    """Coordinate-inferred local chemistry should remain chemistry preparation."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_legality_state_facts(
        selected_scope=selected_scope,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_attachment_resolution_state=(
            HydrogenAttachmentResolutionState.COORDINATE_INFERRED
        ),
    )

    legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=True,
        discrete_seeding_applicable=True,
    )

    assert legal_transformations.contains_stratum(
        LocalTransformationStratum.PREPARATION
    )
    assert legal_transformations.contains_stratum(
        LocalTransformationStratum.CANDIDATE_CONSTRUCTION
    )
    assert legal_transformations.contains_stratum(LocalTransformationStratum.RELAXATION)
    assert (
        legal_transformations.family_record_for(
            LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
        ).stratum
        is LocalTransformationStratum.PREPARATION
    )
    assert (
        legal_transformations.family_record_for(
            LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
        ).preparation_reason
        is LocalPreparationReason.CHEMISTRY_PREPARATION
    )
    assert (
        legal_transformations.family_record_for(
            LocalTransformationFamily.BRANCHED_SIDECHAIN_SEED
        ).stratum
        is LocalTransformationStratum.CANDIDATE_CONSTRUCTION
    )


def test_selected_region_ready_geometry_repair_remains_distinct_from_chemistry() -> (
    None
):
    """Ready-state discrete repair should remain preparation via geometry reason."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_legality_state_facts(
        selected_scope=selected_scope,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_attachment_resolution_state=(
            HydrogenAttachmentResolutionState.TEMPLATE_RESOLVED
        ),
    )

    legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=True,
        discrete_seeding_applicable=True,
    )

    assert legal_transformations.contains_stratum(
        LocalTransformationStratum.PREPARATION
    )
    assert (
        legal_transformations.family_record_for(
            LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
        ).preparation_reason
        is LocalPreparationReason.LOCAL_GEOMETRY
    )


def test_selected_region_stereochemistry_violation_blocks_relaxation() -> None:
    """Stereo-invalid selected scope should not be sent to force-field relaxation."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_legality_state_facts(
        selected_scope=selected_scope,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.VIOLATED,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.PRESENT,
    )

    legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=False,
        discrete_seeding_applicable=False,
    )

    assert legal_transformations.is_empty()
    assert not legal_transformations.contains_stratum(
        LocalTransformationStratum.RELAXATION
    )


def test_selected_region_termination_separates_fresh_exhaustion_from_cycle() -> None:
    """Freshly exhausted state should terminate as no-legal, not cycle."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_legality_state_facts(
        selected_scope=selected_scope,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.PRESENT,
    )
    first_legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=False,
        discrete_seeding_applicable=False,
    )
    continuous_signature = first_legal_transformations.signature_for_family(
        LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION
    )
    first_try_run_state = TransformationPlanningMemory.initial(
        max_steps=3
    ).with_attempted_family(continuous_signature)

    exhausted_legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        atom_scope_facts=atom_scope_facts,
        planning_memory=first_try_run_state,
        discrete_preparation_applicable=False,
        discrete_seeding_applicable=False,
    )
    termination = selected_region_termination_decision(
        planning_memory=first_try_run_state,
        legal_transformations=exhausted_legal_transformations,
    )

    assert exhausted_legal_transformations.is_empty()
    assert exhausted_legal_transformations.is_exhausted_by_run_history()
    assert (
        termination.reason is TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
    )


def build_atom_ref(chain_id: str, seq_num: int, atom_name: str) -> AtomRef:
    """Return one canonical atom reference for legality fixtures."""

    return AtomRef(
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atom_name=atom_name,
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
) -> CanonicalResiduePayload:
    """Return one canonical residue for direct legality tests."""

    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=tuple(
            atom_payload(
                name=atom_name,
                element="H" if atom_name.startswith("H") else atom_name[0],
                position=Vec3(float(index), 0.0, 0.0),
            )
            for index, atom_name in enumerate(atom_names, start=1)
        ),
    )
