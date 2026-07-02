"""Tests for local-refinement proposal policy ownership."""

import pytest

from protrepair.workflow.planning.local_refinement_policy import (
    LocalRefinementProposalPolicy,
)


def test_local_refinement_proposal_policy_owns_default_search_knobs() -> None:
    """Default local-refinement search knobs should live outside the registry."""

    policy = LocalRefinementProposalPolicy()

    assert policy.severe_geometry_fallback_limit == 1
    assert policy.joint_scope_fallback_limit == 1
    assert policy.severe_geometry_context_radius_angstrom == 3.0


@pytest.mark.parametrize("invalid_limit", [-1, True, 1.5, "1"])
def test_local_refinement_proposal_policy_rejects_invalid_candidate_limits(
    invalid_limit: object,
) -> None:
    """Candidate caps should be explicit non-negative integer policy values."""

    with pytest.raises((TypeError, ValueError)):
        LocalRefinementProposalPolicy(
            severe_geometry_fallback_limit=invalid_limit,  # type: ignore[arg-type]
        )


def test_local_refinement_proposal_policy_allows_zero_candidate_limits() -> None:
    """A zero cap should explicitly disable that fallback family."""

    policy = LocalRefinementProposalPolicy(
        severe_geometry_fallback_limit=0,
        joint_scope_fallback_limit=0,
    )

    assert policy.severe_geometry_fallback_limit == 0
    assert policy.joint_scope_fallback_limit == 0


@pytest.mark.parametrize(
    "invalid_radius",
    [0.0, -1.0, float("inf"), float("nan"), True, "3.0"],
)
def test_local_refinement_proposal_policy_rejects_invalid_context_radius(
    invalid_radius: object,
) -> None:
    """Fallback context radius should be an explicit positive numeric value."""

    with pytest.raises((TypeError, ValueError)):
        LocalRefinementProposalPolicy(
            severe_geometry_context_radius_angstrom=invalid_radius,  # type: ignore[arg-type]
        )
