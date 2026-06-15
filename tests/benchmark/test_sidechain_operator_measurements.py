"""Tests for benchmark-only side-chain operator measurements."""

import pytest
from tests.support.refinement_benchmarks import REFINEMENT_BENCHMARK_TRACKS
from tests.support.refinement_cases import EXPLORATORY_REFINEMENT_BENCHMARK_CASES
from tests.support.sidechain_operator_measurements import (
    SIDECHAIN_ROTAMER_GRID_DEFINITIONS,
    SidechainOperatorSignal,
    run_sidechain_operator_measurement,
    sidechain_rotamer_grid_angles,
)

pytestmark = pytest.mark.benchmark


def test_sidechain_rotamer_grid_definitions_cover_gap_cases() -> None:
    """Exploratory side-chain gaps should have benchmark rotamer-grid coverage."""

    assert set(SIDECHAIN_ROTAMER_GRID_DEFINITIONS) == {"GLN", "TYR"}
    assert len(sidechain_rotamer_grid_angles(SIDECHAIN_ROTAMER_GRID_DEFINITIONS["GLN"]))
    assert len(sidechain_rotamer_grid_angles(SIDECHAIN_ROTAMER_GRID_DEFINITIONS["TYR"]))


def test_sidechain_operator_measurement_classifies_5k12_signal() -> None:
    """The TYR gap measurement should run without requiring packaged FASPR assets."""

    result = run_sidechain_operator_measurement(
        EXPLORATORY_REFINEMENT_BENCHMARK_CASES["5k12-tyr471-sidechain"],
        track=REFINEMENT_BENCHMARK_TRACKS["heavy-only"],
    )

    assert result.case_id == "5k12-tyr471-sidechain"
    assert result.component_id == "TYR"
    assert result.candidate_count > 0
    assert result.best_candidate is not None
    assert result.signal in {
        SidechainOperatorSignal.PRODUCTION_PLANNING_SIGNAL,
        SidechainOperatorSignal.TRACKING_ONLY_DIAGNOSTIC_SIGNAL,
        SidechainOperatorSignal.UNSUITABLE,
    }
