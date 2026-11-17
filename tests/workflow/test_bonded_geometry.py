"""Public workflow contracts for residual bonded geometry."""

from pathlib import Path

import pytest

from protrepair import RepairRefinementSpec, process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.geometry import detect_heavy_geometry
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.scope import WholeStructureScope
from protrepair.state import BondedGeometryState, HydrogenCoverageState
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.shared import apply_position_updates
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.workflow.contracts import (
    RequestedGoalStatus,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
    requested_process_goal,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pdb"
_LEU = ResidueId("A", 8)


def _short_carbonyl() -> ProteinStructure:
    structure = read_structure(_FIXTURES / "span-reconstruction" / "1ubq.pdb")
    carbon = structure.constitution.atom_index(AtomRef(_LEU, "C"))
    oxygen = structure.constitution.atom_index(AtomRef(_LEU, "O"))
    c = structure.geometry.atom_geometry(carbon).position
    o = structure.geometry.atom_geometry(oxygen).position
    compressed = Vec3.from_iterable(
        left + (right - left) * 0.8 for left, right in zip(c, o, strict=True)
    )
    return apply_position_updates(structure, moved_positions={oxygen: compressed})


@pytest.mark.parametrize("request_hydrogens", [False, True])
def test_carbonyl_report_does_not_authorize_motion(request_hydrogens: bool) -> None:
    """A report-only or H-completion run keeps the input carbonyl coordinates."""

    source = _short_carbonyl()
    h_goal = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    result = process_structure(
        source, requested_goals=(h_goal,) if request_hydrogens else ()
    )
    for name in ("C", "O"):
        ref = AtomRef(_LEU, name)
        before = source.geometry.atom_geometry(source.constitution.atom_index(ref))
        after = result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(ref)
        )
        assert before.position == after.position

    expected = detect_heavy_geometry(
        result.structure, component_library=build_default_component_library()
    ).to_issues()
    assert any(issue.scope.residue_ids == (_LEU,) for issue in expected)
    actual = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.INVALID_GEOMETRY
    )
    assert actual == expected


def test_geometry_goal_does_not_stage_missing_hydrogens() -> None:
    """A geometry goal observes an unready site but does not prepare it."""

    source = _short_carbonyl()
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=BondedGeometryState.WITHIN_REFERENCE
    )
    result = process_structure(source, requested_goals=(goal,))

    assert result.structure == source
    report = result.requested_goal_report
    assert report is not None
    outcome = report.outcome_for(goal)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.UNMET
    assert outcome.observed_state is BondedGeometryState.OUTLIERS
    assert outcome.details and "hydrogens" in outcome.details
    assert not result.relaxation_executions


def test_geometry_goal_repairs_a_readable_carbonyl_after_requested_hydrogenation() -> (
    None
):
    """Independent goals can prepare H and repair bonded geometry in one run."""

    source = _short_carbonyl()
    h_goal = requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    )
    bonded_goal = requested_process_goal(
        scope=WholeStructureScope(), value=BondedGeometryState.WITHIN_REFERENCE
    )
    h_only = process_structure(source, requested_goals=(h_goal,))
    result = process_structure(
        source,
        requested_goals=(bonded_goal, h_goal),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=12),
    )

    assert not h_only.has_errors()
    assert result.structure.constitution == h_only.structure.constitution
    assert result.structure.topology == h_only.structure.topology
    report = result.requested_goal_report
    assert report is not None
    repaired = report.outcome_for(bonded_goal)
    hydrogenated = report.outcome_for(h_goal)
    assert repaired is not None and repaired.is_satisfied()
    assert hydrogenated is not None and hydrogenated.is_satisfied()
    assert not any(
        issue.kind is ValidationIssueKind.INVALID_GEOMETRY
        and issue.scope.residue_ids == (_LEU,)
        for issue in result.issues
    )
    assert any(
        AtomRef(_LEU, "O") in record.movable_scope.atom_refs
        for record in result.relaxation_executions
    )
    repeated = process_structure(result.structure, requested_goals=(bonded_goal,))
    repeated_report = repeated.requested_goal_report
    assert repeated_report is not None
    repeated_outcome = repeated_report.outcome_for(bonded_goal)
    assert repeated_outcome is not None
    assert repeated_outcome.status is RequestedGoalStatus.ALREADY_SATISFIED
    assert not repeated.relaxation_executions
    assert repeated.structure == result.structure


def test_rejected_geometry_attempt_uses_a_single_workflow_node() -> None:
    """An unavailable solve retains one parent path and exposes budget exhaustion."""

    source = _short_carbonyl()
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=BondedGeometryState.WITHIN_REFERENCE
    )
    result = process_structure(
        source,
        requested_goals=(goal,),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=2),
    )
    assert result.structure == source
    assert not result.repairs
    assert result.branch_report is not None
    assert result.branch_report.search_exhausted()
    assert len(result.branch_report.outcomes) == 1
    report = result.requested_goal_report
    assert report is not None
    outcome = report.outcome_for(goal)
    assert outcome is not None and outcome.status is RequestedGoalStatus.UNMET
    assert (
        outcome.details
        and "1 bonded-geometry attempt(s), 1 not retained" in outcome.details
    )


def test_budget_can_retain_partial_progress_without_satisfying_goal() -> None:
    """Adopt safe progress before issue burden or shallow depth can hide it."""

    h_goal = requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    )
    source = process_structure(_short_carbonyl(), requested_goals=(h_goal,)).structure
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=BondedGeometryState.WITHIN_REFERENCE
    )
    result = process_structure(
        source,
        requested_goals=(goal,),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=2),
    )

    assert result.branch_report is not None and result.branch_report.search_exhausted()
    assert len(result.relaxation_executions) == 1
    assert result.structure.constitution == source.constitution
    assert result.structure.topology == source.topology
    assert not any(
        issue.kind is ValidationIssueKind.INVALID_GEOMETRY
        and issue.scope.residue_ids == (_LEU,)
        for issue in result.issues
    )
    report = result.requested_goal_report
    assert report is not None
    outcome = report.outcome_for(goal)
    assert outcome is not None and outcome.status is RequestedGoalStatus.UNMET


def test_explicit_local_scope_does_not_suppress_bonded_geometry_goal() -> None:
    """Keep the explicit action and the independent geometry action distinct."""

    h_goal = requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    )
    source = process_structure(_short_carbonyl(), requested_goals=(h_goal,)).structure
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=BondedGeometryState.WITHIN_REFERENCE
    )
    manual = AtomRef(ResidueId("A", 12), "O")
    result = process_structure(
        source,
        requested_goals=(goal,),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_atoms((manual,)),
                binding=RecommendedContinuousRelaxationBinding(),
            )
        ),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=12),
    )
    returned_scopes = tuple(
        record.movable_scope.atom_refs for record in result.relaxation_executions
    )
    assert (manual,) in returned_scopes
    assert any(AtomRef(_LEU, "O") in refs for refs in returned_scopes)
    assert not any(
        issue.kind is ValidationIssueKind.INVALID_GEOMETRY
        and issue.scope.residue_ids == (_LEU,)
        for issue in result.issues
    )
