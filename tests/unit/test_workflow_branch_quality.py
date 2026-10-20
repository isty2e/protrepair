"""Unit tests for workflow branch quality score projection."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.request_builders import whole_structure_requested_goals

from protrepair.chemistry import (
    ComponentLibrary,
    UnknownElementRadiusError,
    build_default_component_library,
)
from protrepair.geometry import Vec3
from protrepair.state import (
    ClashObservationMode,
    HydrogenCoverageState,
    OrientationCorrectionEligibilityState,
    StructureIntrinsicGeometryFacts,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.refinement.speculative_planning import (
    SpeculativePlanningNodeId,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
    WorkflowLigandContextMode,
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
)
from protrepair.workflow.contracts.result import (
    RequestedGoalReport,
    WorkflowBranchOutcome,
    WorkflowBranchQualityScore,
    WorkflowBranchReport,
    WorkflowPhaseReport,
    WorkflowPhaseStatus,
)
from protrepair.workflow.engine.reporting import (
    deferred_workflow_phase_report,
    evaluate_requested_goal_report,
    evaluate_workflow_branch_quality_score,
)
from protrepair.workflow.planning import intrinsic_geometry


@pytest.mark.parametrize("custom_library", (False, True))
@pytest.mark.parametrize(
    ("case_id", "complete_hydrogens"),
    (
        (None, False),
        ("3g8l-asn182", False),
        ("3g8l-asn182", True),
        ("2q6f-cys143-pje", False),
    ),
)
def test_branch_score_preserves_all_axes_without_searching_for_actions(
    case_id: str | None, complete_hydrogens: bool, custom_library: bool
) -> None:
    structure = (
        build_structure(chains=(), source_format=FileFormat.PDB)
        if case_id is None
        else load_case_structure(REFINEMENT_BENCHMARK_CASES[case_id])
    )
    if complete_hydrogens:
        structure = add_hydrogens(structure).structure
    library = (
        ComponentLibrary() if custom_library else build_default_component_library()
    )
    result = TransformationResult(structure, repairs=(), issues=())
    context = WorkflowPlanningContext(
        ligand_context_mode=WorkflowLigandContextMode.CONSIDER_IF_PRESENT
    )
    goals = evaluate_requested_goal_report(
        structure,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
        component_library=library,
    )
    full_facts = intrinsic_geometry.derive_structure_intrinsic_geometry_facts(
        structure, component_library=library
    )
    if complete_hydrogens and not custom_library:
        assert full_facts.protein_self_clash_observation_mode is (
            ClashObservationMode.ALL_ATOM_COMPLETE
        )
    if case_id == "3g8l-asn182" and not custom_library and not complete_hydrogens:
        assert full_facts.orientation_correction_eligibility_state is (
            OrientationCorrectionEligibilityState.ELIGIBLE
        )

    with patch.object(
        StructureIntrinsicGeometryFacts, "from_structure", return_value=full_facts
    ):
        reference = evaluate_workflow_branch_quality_score(
            result,
            requested_goal_report=goals,
            planning_context=context,
            component_library=library,
            search_depth=2,
        )
    with patch.object(
        intrinsic_geometry,
        "derive_orientation_correction_eligibility_state",
        side_effect=AssertionError("branch scoring must not search for actions"),
    ):
        score = evaluate_workflow_branch_quality_score(
            result,
            requested_goal_report=goals,
            planning_context=context,
            component_library=library,
            search_depth=2,
        )

    assert score == reference
    competing_parent = _terminal_outcome(
        node_id=0,
        score=replace(
            reference,
            protein_self_clash_count=reference.protein_self_clash_count + 1,
            search_depth=0,
        ),
    )
    for policy in (
        WorkflowBranchPreferencePolicy(),
        WorkflowBranchPreferencePolicy(axes=tuple(reversed(WorkflowBranchQualityAxis))),
        *(
            WorkflowBranchPreferencePolicy(axes=(axis,))
            for axis in WorkflowBranchQualityAxis
        ),
    ):
        expected = WorkflowBranchReport.from_outcomes(
            (competing_parent, _terminal_outcome(node_id=1, score=reference)),
            branch_preference_policy=policy,
        )
        actual = WorkflowBranchReport.from_outcomes(
            (competing_parent, _terminal_outcome(node_id=1, score=score)),
            branch_preference_policy=policy,
        )
        assert actual == expected


def test_scoring_keeps_deferred_phase_orientation_search() -> None:
    structure = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    library = build_default_component_library()
    context = WorkflowPlanningContext()
    report = deferred_workflow_phase_report(
        structure, planning_context=context, component_library=library, blockers=()
    )
    with patch.object(
        intrinsic_geometry,
        "derive_orientation_correction_eligibility_state",
        wraps=intrinsic_geometry.derive_orientation_correction_eligibility_state,
    ) as eligibility:
        evaluate_workflow_branch_quality_score(
            TransformationResult(structure, repairs=(), issues=()),
            requested_goal_report=RequestedGoalReport(()),
            planning_context=context,
            component_library=library,
        )
        eligibility.assert_not_called()
        phase = report.outcome_for(WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION)
        eligibility.assert_called_once_with(structure, component_library=library)

    assert phase is not None
    assert phase.status is WorkflowPhaseStatus.UNRESOLVED
    assert intrinsic_geometry.derive_structure_intrinsic_geometry_facts(
        structure, component_library=library
    ).orientation_correction_eligibility_state is (
        OrientationCorrectionEligibilityState.ELIGIBLE
    )


def test_branch_score_keeps_unknown_radius_rejection() -> None:
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(atom_payload("CA", "XX", Vec3(0, 0, 0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    with pytest.raises(UnknownElementRadiusError):
        evaluate_workflow_branch_quality_score(
            TransformationResult(structure, repairs=(), issues=()),
            requested_goal_report=RequestedGoalReport(()),
            planning_context=WorkflowPlanningContext(),
            component_library=build_default_component_library(),
        )


def test_branch_score_keeps_supplied_observation_library_rejection() -> None:
    structure = build_structure(chains=(), source_format=FileFormat.PDB)
    observation = StructureChemistryObservation(
        structure, component_library=build_default_component_library()
    )
    with pytest.raises(ValueError, match="interpretation inputs"):
        evaluate_workflow_branch_quality_score(
            TransformationResult(structure, repairs=(), issues=()),
            requested_goal_report=RequestedGoalReport(()),
            planning_context=WorkflowPlanningContext(),
            component_library=ComponentLibrary(),
            chemistry_observation=observation,
        )


def test_terminal_branch_selection_uses_parser_quality_before_tie_breaker() -> None:
    """Default branch scoring should prefer parser-clean branches."""

    worse_branch = _terminal_outcome(
        node_id=0,
        score=WorkflowBranchQualityScore(parser_incompatible=1),
    )
    better_branch = _terminal_outcome(
        node_id=1,
        score=WorkflowBranchQualityScore(parser_incompatible=0),
    )

    report = WorkflowBranchReport.from_outcomes((worse_branch, better_branch))

    assert report.preferred_node_id == SpeculativePlanningNodeId(1)


def test_terminal_branch_selection_policy_can_reorder_axes() -> None:
    """Branch quality policy should tune the sortable score projection."""

    earlier_branch = _terminal_outcome(
        node_id=0,
        score=WorkflowBranchQualityScore(parser_incompatible=1, search_depth=0),
    )
    later_branch = _terminal_outcome(
        node_id=1,
        score=WorkflowBranchQualityScore(parser_incompatible=0, search_depth=1),
    )
    search_first_policy = WorkflowBranchPreferencePolicy(
        axes=(
            WorkflowBranchQualityAxis.SEARCH_DEPTH,
            WorkflowBranchQualityAxis.PARSER_COMPATIBILITY,
        )
    )

    report = WorkflowBranchReport.from_outcomes(
        (earlier_branch, later_branch),
        branch_preference_policy=search_first_policy,
    )

    assert report.preferred_node_id == SpeculativePlanningNodeId(0)


def _terminal_outcome(
    *,
    node_id: int,
    score: WorkflowBranchQualityScore,
) -> WorkflowBranchOutcome:
    """Return one minimal terminal outcome for branch preference tests."""

    return WorkflowBranchOutcome(
        node_id=SpeculativePlanningNodeId(node_id),
        requested_goal_report=RequestedGoalReport(()),
        phase_report=WorkflowPhaseReport(()),
        branch_quality_score=score,
        error_count=score.error_count,
        warning_count=score.warning_count,
        issue_count=score.issue_count,
    )
