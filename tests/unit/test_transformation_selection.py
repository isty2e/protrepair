"""Direct regression tests for explainable transformation selection policy."""

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
    WorkflowCompletionPartitionKind,
    WorkflowCompletionSelectionReason,
    WorkflowCompletionStageKind,
    choose_workflow_completion_plan,
    workflow_legal_completion_plans,
)
from protrepair.workflow.planning.transformation.legality import (
    LegalTransformationFamilySet,
    TerminationDecision,
    selected_region_legal_transformations,
)
from protrepair.workflow.planning.transformation.runtime import (
    LocalTransformationFamily,
    TransformationPlanningMemory,
    TransformationPlanningSignature,
    TransformationTerminationReason,
)
from protrepair.workflow.planning.transformation.selection import (
    TransformationSelectionReason,
    choose_next_transformation,
)


def _manual_selection_state_facts(
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


def test_choose_next_transformation_prefers_hydrogen_only_plan() -> None:
    """Workflow completion selection should explain hydrogen continuation."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="selection-workflow-hydrogen",
    )
    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=build_default_component_library(),
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=True,
    )

    planning_decision = choose_workflow_completion_plan(legal_plans)

    assert planning_decision.plan.execution_stages() == (
        WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
    )
    assert planning_decision.plan.partition_kinds() == (
        WorkflowCompletionPartitionKind.HYDROGEN_ONLY,
    )
    assert planning_decision.reason is (
        WorkflowCompletionSelectionReason.REQUESTED_HYDROGEN_POPULATION_REQUIRES_HYDROGEN_CONTINUATION
    )


def test_choose_next_transformation_prefers_heavy_then_hydrogen_plan() -> None:
    """Workflow completion selection should still explain composite planning."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="selection-workflow-heavy-then-hydrogen",
    )
    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=build_default_component_library(),
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=True,
    )

    planning_decision = choose_workflow_completion_plan(legal_plans)

    assert planning_decision.plan.execution_stages() == (
        WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,
        WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
    )
    assert planning_decision.plan.partition_kinds() == (
        WorkflowCompletionPartitionKind.HEAVY_THEN_HYDROGEN,
    )
    assert planning_decision.reason is (
        WorkflowCompletionSelectionReason.REQUESTED_HYDROGEN_POPULATION_REQUIRES_HEAVY_AND_HYDROGEN_STAGES
    )


def test_choose_workflow_completion_plan_prefers_partitioned_heterogeneous_plan() -> (
    None
):
    """Workflow selection should explain heterogeneous subset planning."""

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
        source_name="selection-workflow-heterogeneous",
    )
    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=build_default_component_library(),
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=True,
    )

    planning_decision = choose_workflow_completion_plan(legal_plans)

    assert planning_decision.plan.execution_stages() == (
        WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
    )
    assert planning_decision.plan.partition_kinds() == (
        WorkflowCompletionPartitionKind.HYDROGEN_ONLY,
        WorkflowCompletionPartitionKind.UNSUPPORTED_STOP,
    )
    assert planning_decision.reason is (
        WorkflowCompletionSelectionReason.HETEROGENEOUS_SUBSETS_REQUIRE_PARTITIONED_COMPLETION
    )


def test_choose_next_transformation_prefers_discrete_preparation() -> None:
    """Coordinate-inferred local chemistry should outrank relax and seeding."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_selection_state_facts(
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

    planning_decision = choose_next_transformation(
        legal_transformations=legal_transformations,
    )

    assert (
        planning_decision.selected_family()
        is LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
    )
    assert planning_decision.selected_transformation_family is not None
    assert planning_decision.selected_transformation_family.reason is (
        TransformationSelectionReason.CHEMISTRY_PREPARATION_REPAIRS_LOCAL_STATE
    )


def test_choose_next_transformation_prefers_prep_before_illegal_relaxation() -> None:
    """Preparation should still be selectable before relaxation readiness."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_selection_state_facts(
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

    planning_decision = choose_next_transformation(
        legal_transformations=legal_transformations,
    )

    assert (
        planning_decision.selected_family()
        is LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
    )
    assert planning_decision.selected_transformation_family is not None
    assert planning_decision.selected_transformation_family.reason is (
        TransformationSelectionReason.PREPARATION_REPAIRS_RELAXATION_TOPOLOGY_PRECONDITIONS
    )


def test_choose_next_transformation_prefers_topology_prep_for_unrealizable_graph() -> (
    None
):
    """Selection should explain unrealizable bond graphs as topology preparation."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_selection_state_facts(
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

    planning_decision = choose_next_transformation(
        legal_transformations=legal_transformations,
    )

    assert (
        planning_decision.selected_family()
        is LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
    )
    assert planning_decision.selected_transformation_family is not None
    assert planning_decision.selected_transformation_family.reason is (
        TransformationSelectionReason.PREPARATION_REPAIRS_RELAXATION_TOPOLOGY_PRECONDITIONS
    )


def test_choose_next_transformation_explains_geometry_only_preparation() -> None:
    """Ready-state discrete repair should surface a geometry preparation reason."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_selection_state_facts(
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

    planning_decision = choose_next_transformation(
        legal_transformations=legal_transformations,
    )

    assert (
        planning_decision.selected_family()
        is LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
    )
    assert planning_decision.selected_transformation_family is not None
    assert planning_decision.selected_transformation_family.reason is (
        TransformationSelectionReason.GEOMETRY_PREPARATION_REPAIRS_LOCAL_STATE
    )


def test_choose_next_transformation_returns_terminal_decision() -> None:
    """Selection policy should preserve explicit terminal decisions."""

    empty_legal_transformations = LegalTransformationFamilySet(
        planning_signature=build_planning_signature(),
    )

    planning_decision = choose_next_transformation(
        legal_transformations=empty_legal_transformations,
        termination=TerminationDecision(TransformationTerminationReason.CYCLE_DETECTED),
    )

    assert planning_decision.is_terminal()
    assert planning_decision.selected_transformation_family is None
    assert planning_decision.termination is not None
    assert (
        planning_decision.termination.reason
        is TransformationTerminationReason.CYCLE_DETECTED
    )


def build_planning_signature() -> TransformationPlanningSignature:
    """Return one canonical planning signature for selection tests."""

    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts, atom_scope_facts = _manual_selection_state_facts(
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
    return TransformationPlanningSignature.from_state_facts(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )


def build_atom_ref(chain_id: str, seq_num: int, atom_name: str) -> AtomRef:
    """Return one canonical atom reference for selection fixtures."""

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
    """Return one canonical residue for selection fixtures."""

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
