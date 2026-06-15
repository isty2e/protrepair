"""Fixture-driven regression tests for correction-state and route semantics."""

import pytest
from tests.support.correction_state_dsl import CorrectionStateCase
from tests.support.correction_state_registry import (
    CORRECTION_STATE_CASES,
    REQUIRED_CORRECTION_COVERAGE,
    collected_correction_coverage_tags,
)

from protrepair.chemistry import build_default_component_library
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import StructureProjectionStateFacts
from protrepair.structure.snapshot import (
    ProteinStructureSnapshot,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.readiness import (
    atom_scope_facts_supports_continuous_relaxation,
    derive_atom_scope_continuous_relaxation_facts,
    derive_atom_scope_continuous_relaxation_observation,
)
from protrepair.transformer.discrete import (
    BranchedSidechainSeedTransformer,
    DiscretePreRefinementCorrectionTransformer,
)
from protrepair.transformer.local import atom_input_from_local_scope_spec
from protrepair.workflow.planning.completion import (
    choose_workflow_completion_plan,
    workflow_legal_completion_plans,
)
from protrepair.workflow.planning.transformation.legality import (
    selected_region_legal_transformations,
    selected_region_termination_decision,
)
from protrepair.workflow.planning.transformation.runtime import (
    TransformationPlanningMemory,
)


def test_correction_state_corpus_covers_required_route_preconditions() -> None:
    """The fixture corpus should document the required route-precondition axes."""

    assert REQUIRED_CORRECTION_COVERAGE <= collected_correction_coverage_tags()


@pytest.mark.parametrize(
    "case",
    tuple(
        case for case in CORRECTION_STATE_CASES.values() if case.workflow is not None
    ),
    ids=lambda case: case.case_id,
)
def test_correction_state_cases_preserve_workflow_classification(
    case: CorrectionStateCase,
) -> None:
    """Whole-workflow completion planning should stay stable per fixture."""

    assert case.workflow is not None
    component_library = build_default_component_library()
    structure = case.build_structure(component_library)
    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=component_library,
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=(case.workflow.requests_hydrogen_population),
    )

    assert not legal_plans.is_empty()

    planning_decision = choose_workflow_completion_plan(legal_plans)
    assert (
        tuple(stage.value for stage in planning_decision.plan.execution_stages())
        == case.workflow.execution_stage_values
    )
    assert (
        tuple(
            partition_kind.value
            for partition_kind in planning_decision.plan.partition_kinds()
        )
        == case.workflow.partition_kind_values
    )


@pytest.mark.parametrize(
    "case",
    tuple(case for case in CORRECTION_STATE_CASES.values() if case.local is not None),
    ids=lambda case: case.case_id,
)
def test_correction_state_cases_preserve_local_classification(
    case: CorrectionStateCase,
) -> None:
    """Local state truth and legality should stay stable per fixture."""

    assert case.local is not None
    component_library = build_default_component_library()
    structure = case.build_structure(component_library)
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        case.local.scope_spec,
        component_library=component_library,
    )
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        atom_input.as_scope(),
        carrier=snapshot,
    )
    atom_scope_observation = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        atom_scope,
        component_library=component_library,
    )
    atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=component_library,
    )

    if case.local.component_support_state is not None:
        assert (
            atom_scope_observation.component_support_state
            is case.local.component_support_state
        )
    if case.local.backbone_heavy_atom_completeness_state is not None:
        assert (
            atom_scope_observation.backbone_heavy_atom_completeness_state
            is case.local.backbone_heavy_atom_completeness_state
        )
    if case.local.sidechain_heavy_atom_completeness_state is not None:
        assert (
            atom_scope_observation.sidechain_heavy_atom_completeness_state
            is case.local.sidechain_heavy_atom_completeness_state
        )
    if case.local.hydrogen_applicability_state is not None:
        assert (
            atom_scope_observation.hydrogen_applicability_state
            is case.local.hydrogen_applicability_state
        )
    if case.local.hydrogen_coverage_state is not None:
        assert (
            atom_scope_observation.hydrogen_coverage_state
            is case.local.hydrogen_coverage_state
        )
    if case.local.clash_state is not None:
        assert atom_scope_observation.clash_state is case.local.clash_state
    if case.local.continuous_relaxation_ready is not None:
        assert (
            atom_scope_facts_supports_continuous_relaxation(atom_scope_facts)
            is case.local.continuous_relaxation_ready
        )

    for topology_expectation in case.local.topology_expectations:
        assert (
            atom_scope_observation.topology_availability.state_for(
                topology_expectation.residue_id,
                aspect=topology_expectation.aspect,
            )
            is topology_expectation.state
        )

    if case.local.validate_discrete_preparation_detector:
        assert (
            DiscretePreRefinementCorrectionTransformer(component_library).is_applicable(
                ProteinTransformationContext.from_snapshot_atom_input(
                    snapshot,
                    atom_input,
                )
            )
            is case.local.discrete_preparation_applicable
        )
    if case.local.validate_discrete_seeding_detector:
        assert (
            BranchedSidechainSeedTransformer(component_library).is_applicable(
                ProteinTransformationContext.from_snapshot_atom_input(
                    snapshot,
                    atom_input,
                )
            )
            is case.local.discrete_seeding_applicable
        )

    structure_facts = StructureProjectionStateFacts.from_structure(
        structure,
        component_library=component_library,
    )
    legal_transformations = selected_region_legal_transformations(
        structure_facts=structure_facts,
        selected_scope=atom_input.as_scope(),
        atom_scope_facts=atom_scope_facts,
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        discrete_preparation_applicable=(case.local.discrete_preparation_applicable),
        discrete_seeding_applicable=case.local.discrete_seeding_applicable,
    )

    assert (
        tuple(legal_family.family for legal_family in legal_transformations.families)
        == case.local.legal_families
    )
    assert (
        tuple(legal_family.stratum for legal_family in legal_transformations.families)
        == case.local.legal_strata
    )

    termination = selected_region_termination_decision(
        planning_memory=TransformationPlanningMemory.initial(max_steps=3),
        legal_transformations=legal_transformations,
    )
    assert termination.reason is case.local.termination_reason
