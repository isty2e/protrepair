"""Public DTO enum contracts should reject raw string values."""

from typing import cast

import pytest

from protrepair.diagnostics import (
    EventScope,
    EventScopeKind,
    IssueSeverity,
    RepairEvent,
    RepairEventKind,
    ResidueAtomImpact,
    ValidationIssue,
    ValidationIssueKind,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import ClashState, HydrogenCoverageState
from protrepair.state.scoped import ScopedState
from protrepair.structure.labels import ResidueId
from protrepair.workflow.contracts import (
    RequestedGoalOutcome,
    RequestedGoalReport,
    RequestedGoalStatus,
    WorkflowPhaseOutcome,
    WorkflowPhaseReport,
    WorkflowPhaseStatus,
    WorkflowPlanningPhase,
    requested_process_goal,
)


def _hydrogen_completion_goal():
    """Return one valid requested goal for enum contract tests."""

    return requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )


def test_requested_process_goal_rejects_raw_state_value_strings() -> None:
    """Requested-goal helpers must not build non-canonical string state."""

    with pytest.raises(TypeError, match="unsupported requested-goal value"):
        requested_process_goal(
            scope=WholeStructureScope(),
            value=cast(
                HydrogenCoverageState,
                HydrogenCoverageState.COMPLETE.value,
            ),
        )


def test_requested_goal_outcome_rejects_raw_status_strings() -> None:
    """Status helpers use enum identity and must reject raw enum values."""

    with pytest.raises(TypeError, match="RequestedGoalStatus"):
        RequestedGoalOutcome(
            requested_goal=_hydrogen_completion_goal(),
            status=cast(RequestedGoalStatus, RequestedGoalStatus.SATISFIED.value),
        )


def test_requested_goal_outcome_rejects_raw_observed_state_strings() -> None:
    """Observed requested-goal state must stay on the closed state enum axis."""

    with pytest.raises(TypeError, match="observed_state"):
        RequestedGoalOutcome(
            requested_goal=_hydrogen_completion_goal(),
            status=RequestedGoalStatus.SATISFIED,
            observed_state=cast(
                HydrogenCoverageState,
                HydrogenCoverageState.COMPLETE.value,
            ),
        )


def test_requested_goal_outcome_rejects_incompatible_goal_scope_axes() -> None:
    """Outcome DTOs should carry the same goal contract as request DTOs."""

    with pytest.raises(ValueError, match="whole-structure clash goals"):
        RequestedGoalOutcome(
            requested_goal=ScopedState(
                scope=WholeStructureScope(),
                value=ClashState.NONE,
            ),
            status=RequestedGoalStatus.SATISFIED,
        )


def test_requested_goal_report_rejects_non_outcome_members() -> None:
    """Report containers should not defer malformed members to helper calls."""

    with pytest.raises(TypeError, match="RequestedGoalOutcome values"):
        RequestedGoalReport(
            outcomes=(cast(RequestedGoalOutcome, RequestedGoalStatus.SATISFIED.value),)
        )


def test_requested_goal_report_rejects_raw_status_count_strings() -> None:
    """Status-count helpers should fail loudly on raw enum strings."""

    report = RequestedGoalReport(
        outcomes=(
            RequestedGoalOutcome(
                requested_goal=_hydrogen_completion_goal(),
                status=RequestedGoalStatus.SATISFIED,
            ),
        )
    )

    with pytest.raises(TypeError, match="RequestedGoalStatus"):
        report.status_count(
            cast(RequestedGoalStatus, RequestedGoalStatus.SATISFIED.value)
        )


def test_workflow_phase_outcome_rejects_raw_phase_and_status_strings() -> None:
    """Workflow phase reports should carry closed phase/status enums."""

    with pytest.raises(TypeError, match="WorkflowPlanningPhase"):
        WorkflowPhaseOutcome(
            phase=cast(WorkflowPlanningPhase, WorkflowPlanningPhase.COVERAGE.value),
            status=WorkflowPhaseStatus.CLEAR,
        )

    with pytest.raises(TypeError, match="WorkflowPhaseStatus"):
        WorkflowPhaseOutcome(
            phase=WorkflowPlanningPhase.COVERAGE,
            status=cast(WorkflowPhaseStatus, WorkflowPhaseStatus.CLEAR.value),
        )


def test_workflow_phase_report_rejects_non_outcome_members() -> None:
    """Phase report containers should reject malformed outcome members."""

    with pytest.raises(TypeError, match="WorkflowPhaseOutcome values"):
        WorkflowPhaseReport(
            outcomes=(cast(WorkflowPhaseOutcome, WorkflowPhaseStatus.CLEAR.value),)
        )


def test_workflow_phase_report_rejects_raw_phase_lookup_strings() -> None:
    """Phase lookup should not silently miss raw enum strings."""

    report = WorkflowPhaseReport(
        outcomes=(
            WorkflowPhaseOutcome(
                phase=WorkflowPlanningPhase.COVERAGE,
                status=WorkflowPhaseStatus.CLEAR,
            ),
        )
    )

    with pytest.raises(TypeError, match="WorkflowPlanningPhase"):
        report.outcome_for(
            cast(WorkflowPlanningPhase, WorkflowPlanningPhase.COVERAGE.value)
        )


def test_diagnostic_event_scope_rejects_raw_kind_strings() -> None:
    """Event scopes should reject raw kind strings at construction."""

    with pytest.raises(TypeError, match="EventScopeKind"):
        EventScope(kind=cast(EventScopeKind, EventScopeKind.STRUCTURE.value))


def test_diagnostic_event_scope_rejects_raw_residue_ids() -> None:
    """Event scopes should not accept raw residue-id strings."""

    with pytest.raises(TypeError, match="ResidueId"):
        EventScope(
            kind=EventScopeKind.RESIDUE,
            residue_ids=(cast(ResidueId, "A:1"),),
        )

    with pytest.raises(TypeError, match="ResidueId"):
        EventScope(
            kind=EventScopeKind.RESIDUE,
            residue_ids=(cast(ResidueId, {"chain_id": "A", "seq_num": 1}),),
        )


def test_residue_atom_impact_rejects_raw_residue_ids() -> None:
    """Residue impacts should carry canonical residue identity."""

    with pytest.raises(TypeError, match="ResidueId"):
        ResidueAtomImpact(
            residue_id=cast(ResidueId, "A:1"),
            atom_names=("CA",),
        )


def test_residue_atom_impact_rejects_malformed_component_and_atom_names() -> None:
    """Residue impacts should normalize only typed atom-impact payloads."""

    residue_id = ResidueId("A", 1)

    with pytest.raises(TypeError, match="component_id"):
        ResidueAtomImpact(
            residue_id=residue_id,
            component_id=cast(str, 1),
            atom_names=("CA",),
        )

    with pytest.raises(TypeError, match="atom_names"):
        ResidueAtomImpact(
            residue_id=residue_id,
            atom_names=(cast(str, 1),),
        )


def test_repair_event_rejects_raw_kind_strings() -> None:
    """Repair events should not accept raw repair-kind strings."""

    with pytest.raises(TypeError, match="RepairEventKind"):
        RepairEvent(
            kind=cast(RepairEventKind, RepairEventKind.HEAVY_ATOMS_ADDED.value),
            scope=EventScope.for_structure(),
        )


def test_repair_event_rejects_non_impact_members() -> None:
    """Repair events should not hide malformed impact members."""

    with pytest.raises(TypeError, match="ResidueAtomImpact"):
        RepairEvent(
            kind=RepairEventKind.HEAVY_ATOMS_ADDED,
            scope=EventScope.for_structure(),
            residue_impacts=(cast(ResidueAtomImpact, "not-an-impact"),),
        )


def test_validation_issue_rejects_raw_kind_and_severity_strings() -> None:
    """Issue helpers use enum identity, so kind and severity must be typed."""

    with pytest.raises(TypeError, match="ValidationIssueKind"):
        ValidationIssue(
            kind=cast(ValidationIssueKind, ValidationIssueKind.STERIC_CLASH.value),
            severity=IssueSeverity.ERROR,
            message="clash",
        )

    with pytest.raises(TypeError, match="IssueSeverity"):
        ValidationIssue(
            kind=ValidationIssueKind.STERIC_CLASH,
            severity=cast(IssueSeverity, IssueSeverity.ERROR.value),
            message="clash",
        )
