"""Tests for executable refinement benchmark outcomes."""

import pytest
from tests.support.refinement_benchmarks import (
    REFINEMENT_BENCHMARK_TRACKS,
    REFINEMENT_STRATEGIES,
    RefinementBenchmarkExpectedOutcome,
    run_refinement_benchmark_case,
)
from tests.support.refinement_cases import (
    EXPLORATORY_REFINEMENT_BENCHMARK_CASES,
    REFINEMENT_BENCHMARK_CASES,
)

from protrepair.transformer.continuous.rdkit import Chem, rdBase, rdForceFieldHelpers
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField

RDKIT_AVAILABLE = (
    Chem is not None and rdBase is not None and rdForceFieldHelpers is not None
)
pytestmark = pytest.mark.benchmark


def test_negative_refinement_benchmark_case_surfaces_error() -> None:
    """The covalent-link negative control should remain unsupported."""

    case = REFINEMENT_BENCHMARK_CASES["2q6f-cys143-pje"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["atom-tight"],
        track=REFINEMENT_BENCHMARK_TRACKS["heavy-only"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "error"
    assert result.metrics_after is None
    assert result.error_type is not None
    assert result.expected_outcome is RefinementBenchmarkExpectedOutcome.EXPECTED_ERROR
    assert result.expected_success is False
    assert result.expectation_matched is True


def test_low_resolution_sidechain_gap_stays_stable_no_op() -> None:
    """Low-resolution sidechain fixtures should expose missing operator support."""

    case = EXPLORATORY_REFINEMENT_BENCHMARK_CASES["5k12-tyr471-sidechain"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["sidechain-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "no_op"
    assert (
        result.expected_outcome
        is RefinementBenchmarkExpectedOutcome.EXPECTED_OPERATOR_SUPPORT_GAP
    )
    assert result.expectation_matched is True
    assert result.metrics_after is not None
    assert result.metrics_before.focus_clash_count == 0
    assert result.metrics_after.focus_clash_count == 0
    assert result.error_message is not None
    assert "geometry 0->" in result.error_message


def test_hydrogenated_negative_refinement_benchmark_case_surfaces_topology_blocker(
) -> None:
    """Hydrogenating retained chemistry should not mask the PJE linkage blocker."""

    case = REFINEMENT_BENCHMARK_CASES["2q6f-cys143-pje"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "error"
    assert result.metrics_after is None
    assert result.error_type == "RefinementError"
    assert result.error_message == (
        "continuous relaxation requires local bond-planning support for all "
        "included residues: PJE (D:5)"
    )
    assert result.expected_outcome is RefinementBenchmarkExpectedOutcome.EXPECTED_ERROR
    assert result.expected_success is False
    assert result.expectation_matched is True


def test_reference_rmsd_is_context_aligned_for_paired_fixture() -> None:
    """Paired reference RMSD should be local-context aligned, not raw-frame offset."""

    case = REFINEMENT_BENCHMARK_CASES["1xgo-leu253"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["atom-tight"],
        track=REFINEMENT_BENCHMARK_TRACKS["heavy-only"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.metrics_before.reference_aligned_focus_atom_rmsd_angstrom is not None
    assert result.metrics_before.reference_aligned_focus_atom_rmsd_angstrom < 5.0


def test_positive_refinement_benchmark_case_runs_with_fixed_context_preserved() -> None:
    """One positive benchmark case should run and keep fixed context stationary."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.metrics_after is not None
    assert result.fixed_context_max_displacement_angstrom == 0.0
    assert result.moved_atom_count is not None
    assert result.moved_atom_count > 0
    assert result.preparation_runtime_ms >= 0.0
    assert result.backend_runtime_median_ms >= 0.0
    assert result.refinement_runtime_median_ms >= 0.0
    assert result.runtime_breakdown is not None
    assert result.runtime_breakdown.problem_build_runtime_ms >= 0.0
    assert result.runtime_breakdown.snapshot_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.atom_input_lowering_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.readiness_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.settings_bind_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.request_normalization_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.candidate_construction_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.execution_candidate_count_median >= 1.0
    assert result.runtime_breakdown.continuous_backend_candidate_count_median >= 1.0
    assert result.runtime_breakdown.discrete_only_candidate_count_median >= 0.0
    assert result.runtime_breakdown.backend_execution_stage_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.assessment_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.selection_runtime_median_ms >= 0.0
    assert result.runtime_breakdown.materialization_runtime_median_ms >= 0.0
    assert (
        result.metrics_before.focus_geometry_outlier_count
        == result.metrics_before.focus_restraint_backed_geometry_outlier_count
        + result.metrics_before.focus_fallback_geometry_outlier_count
    )
    assert (
        result.metrics_after.focus_geometry_outlier_count
        == result.metrics_after.focus_restraint_backed_geometry_outlier_count
        + result.metrics_after.focus_fallback_geometry_outlier_count
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "4pep-sep68",
        "8pyr-tpo170",
        "4jfx-ptr8",
    ),
)
def test_hydrogenated_retained_non_polymer_benchmark_cases_run_under_uff(
    case_id: str,
) -> None:
    """Hydrogenated benchmark prep should realize retained chemistry for UFF."""

    case = REFINEMENT_BENCHMARK_CASES[case_id]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.error_type is None
    assert result.error_message is None
    assert result.metrics_after is not None
    assert (
        result.expected_outcome
        is RefinementBenchmarkExpectedOutcome.RUNNABLE_SUCCESS
    )
    assert result.expected_success is True
    assert result.expectation_matched is True


@pytest.mark.parametrize("case_id", ("1jd0-gln92", "1ywr-asn155"))
def test_metric_clean_benchmark_cases_can_expect_no_op(
    case_id: str,
) -> None:
    """Metric-clean local benchmarks should not masquerade as improvements."""

    case = REFINEMENT_BENCHMARK_CASES[case_id]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "no_op"
    assert result.error_type == "NoOpRefinement"
    assert result.metrics_after is not None
    assert result.moved_atom_count == 0
    assert result.expected_outcome is RefinementBenchmarkExpectedOutcome.EXPECTED_NO_OP
    assert result.expected_success is True
    assert result.expectation_matched is True


@pytest.mark.parametrize("case_id", ("4pep-sep68", "8pyr-tpo170", "4jfx-ptr8"))
def test_phosphorylated_benchmark_cases_run_under_mmff(
    case_id: str,
) -> None:
    """Phosphorylated retained-region benchmarks should run under MMFF."""

    case = REFINEMENT_BENCHMARK_CASES[case_id]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.MMFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.force_field == "mmff"
    assert result.metrics_after is not None
    assert result.error_type is None
    assert result.error_message is None
    assert result.expected_success is True
    assert result.expectation_matched is True
    assert (
        result.metrics_after.focus_clash_count
        <= result.metrics_before.focus_clash_count
    )
    assert (
        result.metrics_after.focus_geometry_outlier_count
        <= result.metrics_before.focus_geometry_outlier_count
    )
    assert result.metrics_after.whole_structure_rdkit_sanitize_readable is True
    assert result.fixed_context_max_displacement_angstrom == 0.0


def test_asn_benchmark_case_errors_when_clashes_block_residue_local_track() -> None:
    """Hydrogenated 3G8L Asn182 should now run and relieve local clashes."""

    case = REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.error_type is None
    assert result.error_message is None
    assert result.metrics_after is not None
    assert (
        result.metrics_after.focus_clash_count < result.metrics_before.focus_clash_count
    )
    assert result.fixed_context_max_displacement_angstrom == 0.0
    assert result.moved_atom_count is not None
    assert result.moved_atom_count > 0


@pytest.mark.parametrize("case_id", ("1bkr-his42", "1ywr-his148", "1lxa-his125"))
def test_histidine_benchmark_cases_report_non_regressing_quality_metrics(
    case_id: str,
) -> None:
    """Histidine cases should expose richer quality metrics without regressing."""

    case = REFINEMENT_BENCHMARK_CASES[case_id]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.metrics_after is not None
    assert (
        result.metrics_after.focus_clash_overlap_sum_angstrom
        <= result.metrics_before.focus_clash_overlap_sum_angstrom
    )
    assert (
        result.metrics_after.focus_total_near_covalent_overlap_angstrom
        <= result.metrics_before.focus_total_near_covalent_overlap_angstrom
    )
    assert result.metrics_after.whole_structure_rdkit_sanitize_readable is True
    assert result.focus_atom_rms_displacement_angstrom is not None
    assert result.focus_atom_max_displacement_angstrom is not None
    assert result.fixed_context_max_displacement_angstrom == 0.0


def test_glutamine_benchmark_case_reports_non_regressing_quality_metrics() -> None:
    """Hydrogenated 1UAQ Gln55 should expose benchmark quality metrics cleanly."""

    case = REFINEMENT_BENCHMARK_CASES["1uaq-gln55"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.error_type is None
    assert result.error_message is None
    assert result.metrics_after is not None
    assert (
        result.metrics_after.focus_clash_overlap_sum_angstrom
        <= result.metrics_before.focus_clash_overlap_sum_angstrom
    )
    assert (
        result.metrics_after.focus_total_near_covalent_overlap_angstrom
        <= result.metrics_before.focus_total_near_covalent_overlap_angstrom
    )
    assert result.metrics_after.whole_structure_rdkit_sanitize_readable is True
    assert result.fixed_context_max_displacement_angstrom == 0.0


@pytest.mark.parametrize(
    "case_id",
    (
        "3g8l-asn182",
        "1uaq-gln55",
    ),
)
def test_clashy_hydrogenated_benchmark_cases_error_before_refinement(
    case_id: str,
) -> None:
    """Clashy hydrogenated local cases should run without worsening clash burden."""

    case = REFINEMENT_BENCHMARK_CASES[case_id]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.error_type is None
    assert result.error_message is None
    assert result.metrics_after is not None
    assert (
        result.metrics_after.focus_clash_overlap_sum_angstrom
        <= result.metrics_before.focus_clash_overlap_sum_angstrom
    )
    assert (
        result.metrics_after.focus_total_near_covalent_overlap_angstrom
        <= result.metrics_before.focus_total_near_covalent_overlap_angstrom
    )


def test_leu_benchmark_case_improves_hydrogenated_residue_local_track() -> None:
    """Hydrogenated Leu253 residue-local track should benefit from rotamer seeding."""

    case = REFINEMENT_BENCHMARK_CASES["1xgo-leu253"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.UFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.metrics_after is not None
    assert result.metrics_after.focus_clash_count == 0
    assert result.metrics_after.focus_geometry_outlier_count <= 1
    assert result.metrics_after.reference_aligned_focus_atom_rmsd_angstrom is not None
    assert result.metrics_before.reference_aligned_focus_atom_rmsd_angstrom is not None
    assert result.metrics_after.reference_aligned_focus_atom_rmsd_angstrom < 5.0
    assert result.fixed_context_max_displacement_angstrom == 0.0
    assert result.moved_atom_count is not None
    assert result.moved_atom_count > 0


def test_positive_refinement_benchmark_case_runs_under_mmff() -> None:
    """One positive benchmark case should also run under MMFF."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    result = run_refinement_benchmark_case(
        case,
        REFINEMENT_STRATEGIES["residue-local"],
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        force_field=ContinuousRelaxationForceField.MMFF,
        repeats=1,
    )

    assert result.status == "success"
    assert result.force_field == "mmff"
    assert result.metrics_after is not None
    assert result.fixed_context_max_displacement_angstrom == 0.0
    assert result.error_type is None
