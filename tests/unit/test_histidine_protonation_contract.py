"""Contract tests for histidine protonation request resolution artifacts."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import get_args

import pytest

from protrepair.structure.slots import ResidueIndex
from protrepair.transformer.completion.hydrogen.protonation import (
    DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO,
    DisabledHistidineProtonationRequest,
    HistidineDeltaProtonationAssignment,
    HistidineProtonationRequest,
    PrasRatioHistidineProtonationRequest,
)


def test_disabled_histidine_protonation_request_is_a_request() -> None:
    assert get_args(HistidineProtonationRequest) == (
        DisabledHistidineProtonationRequest,
        PrasRatioHistidineProtonationRequest,
    )


def test_pras_ratio_histidine_protonation_request_normalizes_ratio() -> None:
    request = PrasRatioHistidineProtonationRequest()

    assert request.ratio == DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO
    assert PrasRatioHistidineProtonationRequest(ratio=0).ratio == 0.0
    assert PrasRatioHistidineProtonationRequest(ratio=1).ratio == 1.0


@pytest.mark.parametrize(
    "ratio",
    (True, False, "0.2", 0.2 + 0j, Fraction(1, 5), object()),
)
def test_pras_ratio_histidine_request_rejects_non_numeric_ratio(ratio) -> None:
    with pytest.raises(TypeError, match="ratio must be an int or float"):
        PrasRatioHistidineProtonationRequest(ratio=ratio)  # type: ignore[arg-type]


@pytest.mark.parametrize("ratio", (math.nan, math.inf, -math.inf))
def test_pras_ratio_histidine_request_rejects_non_finite_ratio(
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match="ratio must be finite"):
        PrasRatioHistidineProtonationRequest(ratio=ratio)


@pytest.mark.parametrize("ratio", (-0.001, 1.001))
def test_pras_ratio_histidine_request_rejects_out_of_range_ratio(
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match=r"ratio must be in \[0\.0, 1\.0\]"):
        PrasRatioHistidineProtonationRequest(ratio=ratio)


def test_histidine_delta_assignment_records_residue() -> None:
    assignment = HistidineDeltaProtonationAssignment(ResidueIndex(3))

    assert assignment.residue_index == ResidueIndex(3)


def test_histidine_delta_assignment_rejects_raw_residue_index() -> None:
    with pytest.raises(TypeError, match="residue_index must be a ResidueIndex"):
        HistidineDeltaProtonationAssignment(3)  # type: ignore[arg-type]


def test_histidine_protonation_contract_values_are_immutable() -> None:
    request = PrasRatioHistidineProtonationRequest()
    assignment = HistidineDeltaProtonationAssignment(ResidueIndex(3))

    with pytest.raises(FrozenInstanceError):
        request.ratio = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        assignment.residue_index = ResidueIndex(4)  # type: ignore[misc]
