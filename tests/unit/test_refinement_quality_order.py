from dataclasses import replace
from itertools import product

import pytest

from protrepair.transformer.refinement.acceptance import (
    FocusRefinementQualityMetrics,
    RefinementAcceptanceMetrics,
    WholeStructureParserCompatibilityMetrics,
    refinement_metrics_order_key,
    refinement_metrics_regressed,
    refinement_metrics_rejected,
)


def _metrics(
    *,
    readable: bool | None = False,
    clashes: int = 11,
    overlap: float = 5.0,
    proxy_overlap: float = 0.1852622871,
    extra: int = 1,
    heavy: int = 1,
) -> RefinementAcceptanceMetrics:
    return RefinementAcceptanceMetrics(
        focus_quality=FocusRefinementQualityMetrics(
            clash_count=clashes,
            clash_overlap_sum_angstrom=overlap,
            geometry_outlier_count=0,
            near_covalent_contact_count=1,
            worst_near_covalent_overlap_angstrom=proxy_overlap,
            total_near_covalent_overlap_angstrom=proxy_overlap,
        ),
        parser_compatibility=WholeStructureParserCompatibilityMetrics(
            rdkit_sanitize_readable=readable,
            extra_proximity_bond_count=extra,
            extra_heavy_proximity_bond_count=heavy,
        ),
    )


def test_unreadable_proxy_relief_does_not_buy_three_extra_clashes() -> None:
    endpoint = _metrics()
    intermediate = _metrics(clashes=14, proxy_overlap=0.1782656726)

    assert refinement_metrics_order_key(endpoint) < refinement_metrics_order_key(
        intermediate
    )
    assert refinement_metrics_rejected(endpoint, intermediate)
    assert not refinement_metrics_rejected(intermediate, endpoint)


@pytest.mark.parametrize("extra,heavy", [(2, 0), (1, 1)])
def test_actual_parser_bond_relief_precedes_local_contact_proxies(
    extra: int, heavy: int
) -> None:
    parent = _metrics(extra=2, heavy=1)
    candidate = _metrics(
        extra=extra, heavy=heavy, clashes=14, overlap=8.0, proxy_overlap=0.5
    )

    assert refinement_metrics_order_key(candidate) < refinement_metrics_order_key(
        parent
    )
    assert not refinement_metrics_rejected(parent, candidate)
    assert refinement_metrics_rejected(candidate, parent)


def test_clash_relief_can_progress_without_resolving_remote_parser_failure() -> None:
    parent = _metrics()
    candidate = _metrics(clashes=10, proxy_overlap=0.5)

    assert refinement_metrics_order_key(candidate) < refinement_metrics_order_key(
        parent
    )
    assert not refinement_metrics_rejected(parent, candidate)


@pytest.mark.parametrize("readable", [False, True, None])
def test_equal_observed_quality_is_not_regression(readable: bool | None) -> None:
    metrics = _metrics(readable=readable)

    assert not refinement_metrics_regressed(metrics, metrics)
    assert not refinement_metrics_rejected(metrics, metrics)
    assert metrics.parser_compatibility.rdkit_sanitize_readable is readable


def test_readability_recovery_precedes_remaining_contact_quality() -> None:
    unreadable = _metrics(clashes=0, overlap=0, extra=0, heavy=0)
    readable = _metrics(readable=True, clashes=12, extra=0, heavy=0)

    assert refinement_metrics_order_key(readable) < refinement_metrics_order_key(
        unreadable
    )
    assert not refinement_metrics_rejected(unreadable, readable)
    assert refinement_metrics_rejected(readable, unreadable)


def test_unobserved_extra_bonds_do_not_override_clashes() -> None:
    readable = None
    parent = _metrics(readable=readable, extra=2)
    candidate = _metrics(readable=readable, extra=0, heavy=0, clashes=14)

    assert refinement_metrics_order_key(parent) < refinement_metrics_order_key(
        candidate
    )
    assert refinement_metrics_rejected(parent, candidate)


def test_readable_hydrogen_attachment_relief_precedes_clashes() -> None:
    parent = _metrics(readable=True, extra=1, heavy=0, clashes=0)
    candidate = _metrics(readable=True, extra=0, heavy=0, clashes=1)

    assert refinement_metrics_order_key(candidate) < refinement_metrics_order_key(
        parent
    )
    assert not refinement_metrics_rejected(parent, candidate)
    assert refinement_metrics_rejected(candidate, parent)


@pytest.mark.parametrize(
    "field",
    [
        "stereochemistry_violation_count",
        "severe_restraint_backed_bond_length_outlier_count",
    ],
)
def test_parser_recovery_does_not_bypass_chemical_vetoes(field: str) -> None:
    parent = _metrics()
    candidate = _metrics(readable=True, clashes=0, extra=0, heavy=0)
    candidate = replace(
        candidate, focus_quality=replace(candidate.focus_quality, **{field: 1})
    )

    assert refinement_metrics_order_key(candidate) < refinement_metrics_order_key(
        parent
    )
    assert refinement_metrics_rejected(parent, candidate)


def test_acceptance_and_ranking_share_one_order_without_chemical_vetoes() -> None:
    observations = tuple(
        _metrics(readable=readable, clashes=clashes, extra=extra, heavy=0)
        for readable, clashes, extra in product([False, True, None], [0, 2], [0, 1])
    )
    for parent, candidate in product(observations, repeat=2):
        worse = refinement_metrics_order_key(candidate) > refinement_metrics_order_key(
            parent
        )
        assert refinement_metrics_regressed(parent, candidate) is worse
        assert refinement_metrics_rejected(parent, candidate) is worse
