"""Tests for perturb-and-recover refinement benchmark outcomes."""

import pytest
from tests.support.refinement_benchmarks import (
    REFINEMENT_BENCHMARK_PROFILES,
    REFINEMENT_BENCHMARK_TRACKS,
    REFINEMENT_STRATEGIES,
)
from tests.support.refinement_perturbation_benchmarks import (
    PERTURBATION_BENCHMARK_SCENARIOS,
    REFINEMENT_PERTURBATION_SPECS,
    SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES,
    run_refinement_perturbation_benchmark_case,
)

from protrepair.transformer.continuous.rdkit import Chem, rdBase, rdForceFieldHelpers

RDKIT_AVAILABLE = (
    Chem is not None and rdBase is not None and rdForceFieldHelpers is not None
)
pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    "scenario_id",
    tuple(PERTURBATION_BENCHMARK_SCENARIOS),
)
def test_perturbation_benchmark_scenario_recovers_toward_reference(
    scenario_id: str,
) -> None:
    """Perturbation scenarios should satisfy their declared outcome contract."""

    scenario = PERTURBATION_BENCHMARK_SCENARIOS[scenario_id]
    result = run_refinement_perturbation_benchmark_case(
        scenario.case,
        scenario.strategy,
        perturbation=scenario.perturbation,
        track=scenario.track,
        execution_profile=scenario.execution_profile,
        repeats=1,
    )

    assert scenario.expected_outcome.matches_status(result.status)
    if result.status == "success":
        assert result.error_type is None
        assert result.error_message is None
    assert result.metrics_after is not None
    assert result.metrics_before.reference_aligned_focus_atom_rmsd_angstrom is not None
    assert result.metrics_after.reference_aligned_focus_atom_rmsd_angstrom is not None
    assert result.metrics_before.reference_aligned_focus_atom_rmsd_angstrom > 0.0
    if scenario.expected_outcome.expects_reference_rmsd_recovery():
        assert (
            result.metrics_after.reference_aligned_focus_atom_rmsd_angstrom
            <= result.metrics_before.reference_aligned_focus_atom_rmsd_angstrom
        )
    assert result.metrics_after.whole_structure_rdkit_sanitize_readable is True
    assert result.fixed_context_max_displacement_angstrom == 0.0
    assert result.moved_atom_count is not None
    assert result.moved_atom_count > 0
    assert result.runtime_breakdown is not None
    assert result.runtime_breakdown.problem_build_runtime_ms >= 0.0
    assert result.runtime_breakdown.snapshot_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.atom_input_lowering_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.readiness_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.settings_bind_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.request_normalization_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.candidate_construction_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.backend_execution_stage_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.assessment_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.selection_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.materialization_runtime_median_ms >= 0.0


def test_perturbation_benchmark_surfaces_no_op_refinement() -> None:
    """No-op refinement should not be reported as a successful recovery."""

    result = run_refinement_perturbation_benchmark_case(
        SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES["1jd0-gln92-self-reference"],
        REFINEMENT_STRATEGIES["sidechain-local"],
        perturbation=REFINEMENT_PERTURBATION_SPECS["clash_shove"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
        repeats=1,
    )

    assert result.status == "no_op"
    assert result.error_type == "NoOpRefinement"
    assert result.error_message is not None
    assert "selected-region quality regressed" in result.error_message
    assert result.moved_atom_count == 0
    assert result.metrics_after is not None
    assert (
        result.metrics_after.reference_aligned_focus_atom_rmsd_angstrom
        == result.metrics_before.reference_aligned_focus_atom_rmsd_angstrom
    )
