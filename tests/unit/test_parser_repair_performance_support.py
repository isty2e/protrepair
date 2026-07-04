import inspect

from tests.support.parser_repair_performance import (
    PARSER_REPAIR_PERFORMANCE_SCHEMA_VERSION,
    ParserBurdenSignal,
    ParserRepairClusterSummary,
    ParserRepairFocusQuality,
    ParserRepairPerformanceResult,
    ParserRepairProbeMode,
    ParserRepairQualitySignal,
    RuntimeTrackingSignal,
    run_first_parser_cluster_repair_probe,
)


def test_parser_repair_probe_does_not_expose_pickle_cache_axis() -> None:
    """Parser-repair probes should not expose unsafe object-cache controls."""

    parameters = inspect.signature(run_first_parser_cluster_repair_probe).parameters

    assert "use_hydrogen_cache" not in parameters
    assert "hydrogen_cache_path" not in parameters


def test_parser_repair_probe_mode_rejects_unknown_value() -> None:
    """Parser-repair probe modes should be a closed benchmark axis."""

    assert (
        ParserRepairProbeMode.from_value("production_like")
        is ParserRepairProbeMode.PRODUCTION_LIKE
    )

    try:
        ParserRepairProbeMode.from_value("full")
    except ValueError as error:
        assert "production_like" in str(error)
        assert "diagnostic_all_candidates" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown parser-repair probe mode should fail")


def test_parser_repair_result_serializes_schema_and_mode() -> None:
    """Parser-repair JSON output should identify the benchmark mode."""

    result = _parser_repair_result()

    payload = result.as_serializable_dict()

    assert payload["schema_version"] == PARSER_REPAIR_PERFORMANCE_SCHEMA_VERSION
    assert payload["mode"] == "production_like"
    assert payload["executed_count"] == 1
    assert "hydrogen_cache_used" not in payload
    assert payload["signal"] == {
        "parser_burden": "present",
        "repair_quality": "improved",
        "runtime_tracking": "tracking_only",
    }


def test_parser_repair_signal_keeps_zero_burden_out_of_repair_success() -> None:
    """Zero-burden parser smoke should not be reported as repair success."""

    result = _parser_repair_result(
        cluster_extra_bond_count=0,
        before_witness_count=0,
        before_heavy_witness_count=0,
        after_witness_count=0,
        after_heavy_witness_count=0,
    )

    signal = result.probe_signal()

    assert signal.parser_burden is ParserBurdenSignal.ABSENT
    assert signal.repair_quality is ParserRepairQualitySignal.NO_PARSER_BURDEN


def test_parser_repair_signal_marks_scaling_cases_tracking_only() -> None:
    """Exploratory scaling cases should remain runtime tracking-only."""

    result = _parser_repair_result(case_id="3ja8-whole-structure")

    assert result.probe_signal().runtime_tracking is RuntimeTrackingSignal.TRACKING_ONLY


def test_parser_repair_signal_marks_must_pass_cases_threshold_eligible() -> None:
    """Must-pass cases may be interpreted against runtime thresholds."""

    result = _parser_repair_result(case_id="1afc-hydrogen-his-protonated")

    assert (
        result.probe_signal().runtime_tracking
        is RuntimeTrackingSignal.GATING_THRESHOLD_ELIGIBLE
    )


def _parser_repair_result(
    *,
    case_id: str = "3ja8-whole-structure",
    cluster_extra_bond_count: int = 1,
    before_witness_count: int = 1,
    before_heavy_witness_count: int = 1,
    after_witness_count: int = 0,
    after_heavy_witness_count: int = 0,
) -> ParserRepairPerformanceResult:
    """Return a lightweight parser-repair performance result fixture."""

    return ParserRepairPerformanceResult(
        schema_version=PARSER_REPAIR_PERFORMANCE_SCHEMA_VERSION,
        case_id=case_id,
        mode=ParserRepairProbeMode.PRODUCTION_LIKE.value,
        timings_sec={"candidate_construction": 1.0},
        cluster=ParserRepairClusterSummary(
            residue_ids=("A:1",),
            extra_bond_count=cluster_extra_bond_count,
            extra_bonds=("A:1.N-A:1.CA",) if cluster_extra_bond_count else (),
        ),
        candidate_count=2,
        executed_count=1,
        execution_error_count=0,
        candidate_summaries=(),
        before_witness_count=before_witness_count,
        before_heavy_witness_count=before_heavy_witness_count,
        after_witness_count=after_witness_count,
        after_heavy_witness_count=after_heavy_witness_count,
        before_focus=ParserRepairFocusQuality(
            rdkit_sanitize_readable=False,
            focus_clash_count=1,
            focus_geometry_outlier_count=0,
        ),
        after_focus=ParserRepairFocusQuality(
            rdkit_sanitize_readable=True,
            focus_clash_count=0,
            focus_geometry_outlier_count=0,
        ),
        backend_name="discrete_preconditioning",
        issue_count=0,
    )
