"""Regression tests for diagnosis-driven workflow route benchmarks."""

import pytest
from tests.support.correction_route_benchmarks import (
    WORKFLOW_ROUTE_BENCHMARK_CASES,
    WorkflowRouteBenchmarkCase,
    run_workflow_route_benchmark_case,
)


@pytest.mark.parametrize(
    "case",
    tuple(WORKFLOW_ROUTE_BENCHMARK_CASES.values()),
    ids=lambda case: case.case_id,
)
def test_workflow_route_benchmark_outputs_preserve_expected_choice(
    case: WorkflowRouteBenchmarkCase,
) -> None:
    """Benchmark outputs should include diagnosed state and chosen route."""

    result = run_workflow_route_benchmark_case(case)
    serialized = result.as_serializable_dict()

    assert result.selection_reason == case.expected.selection_reason
    assert result.partition_kind_values == case.expected.partition_kind_values
    assert result.execution_plan == case.expected.execution_plan
    assert result.route_quality.selected_plan_is_legal
    assert result.route_quality.route_selection_matches_expected
    assert (
        tuple(serialized["partition_kind_values"])
        == case.expected.partition_kind_values
    )
    assert tuple(stage["kind"] for stage in serialized["execution_plan"]) == tuple(
        stage.kind for stage in case.expected.execution_plan
    )
    assert tuple(
        stage["scope_kind"] for stage in serialized["execution_plan"]
    ) == tuple(stage.scope_kind for stage in case.expected.execution_plan)
    assert "diagnosed_state_before" in serialized
    assert "diagnosed_state_after" in serialized
    assert "route_quality" in serialized


def test_composite_workflow_route_benchmark_reports_stage_progress() -> None:
    """Composite heavy+hydrogen benchmark should show both progress signals."""

    result = run_workflow_route_benchmark_case(
        WORKFLOW_ROUTE_BENCHMARK_CASES["heavy-then-hydrogen-workflow"]
    )

    assert result.route_quality.heavy_completion_cleared is True
    assert result.route_quality.hydrogen_gap_cleared is True
    assert result.route_quality.unsupported_stop_reported is None


def test_heterogeneous_workflow_route_benchmark_reports_unsupported_stop() -> None:
    """Partitioned heterogeneous benchmark should project unsupported-stop issues."""

    result = run_workflow_route_benchmark_case(
        WORKFLOW_ROUTE_BENCHMARK_CASES["heterogeneous-workflow-partition"]
    )

    assert result.route_quality.hydrogen_gap_cleared is False
    assert result.route_quality.unsupported_stop_reported is True
    assert "missing_component_definition" in result.issue_kind_values
