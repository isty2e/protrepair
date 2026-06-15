"""Tests for generic speculative-planning contracts."""

import pytest

from protrepair.transformer.refinement.speculative_planning import (
    EvaluatedSpeculativeProposal,
    SpeculativeAdoptedChild,
    SpeculativeAdoptionDecision,
    SpeculativeAdoptionDecisionKind,
    SpeculativeExecution,
    SpeculativePlanningFrontier,
    SpeculativePlanningNodeId,
    SpeculativePlanningTrace,
    SpeculativeStopReason,
)


def test_speculative_adoption_decision_requires_candidate_for_adopt() -> None:
    """Adopted speculative decisions should expose their candidate."""

    candidate = EvaluatedSpeculativeProposal(
        execution=SpeculativeExecution(
            proposal="seed-a",
            outcome="relaxed-a",
        ),
        evaluation=1.0,
    )

    decision = SpeculativeAdoptionDecision.adopt(candidate)

    assert decision.kind is SpeculativeAdoptionDecisionKind.ADOPT
    assert decision.require_candidate() is candidate


def test_speculative_adoption_decision_reject_has_no_candidate() -> None:
    """Rejected speculative decisions should not expose a candidate."""

    decision = SpeculativeAdoptionDecision.reject(reason="no proposals improved state")

    assert decision.kind is SpeculativeAdoptionDecisionKind.REJECT
    assert decision.reason == "no proposals improved state"
    with pytest.raises(ValueError, match="has no candidate"):
        decision.require_candidate()


def test_speculative_planning_trace_appends_single_active_child() -> None:
    """Adopted child nodes should become the sole active frontier node."""

    trace = SpeculativePlanningTrace[str, str, str, float].root(
        initial_state="root-state"
    )
    parent_node = trace.frontier.require_single_active_node()
    decision = SpeculativeAdoptionDecision.adopt(
        EvaluatedSpeculativeProposal(
            execution=SpeculativeExecution(
                proposal="proposal-a",
                outcome="outcome-a",
            ),
            evaluation=1.0,
        )
    )

    next_trace = trace.append_adopted_child(
        parent_node=parent_node,
        child_state="child-state",
        adopted_decision=decision,
    )

    active_node = next_trace.frontier.require_single_active_node()
    assert active_node.node_id == SpeculativePlanningNodeId(1)
    assert active_node.parent_node_id == SpeculativePlanningNodeId(0)
    assert active_node.state == "child-state"


def test_speculative_planning_trace_expands_active_node_with_multiple_children(
) -> None:
    """Active-node expansion should retire the parent and append child worklist."""

    trace = SpeculativePlanningTrace[str, str, str, float].root(
        initial_state="root-state"
    )
    parent_node = trace.frontier.require_single_active_node()
    next_trace = trace.expand_active_node(
        parent_node=parent_node,
        adopted_children=(
            SpeculativeAdoptedChild(
                state="child-a",
                adopted_decision=SpeculativeAdoptionDecision.adopt(
                    EvaluatedSpeculativeProposal(
                        execution=SpeculativeExecution(
                            proposal="proposal-a",
                            outcome="outcome-a",
                        ),
                        evaluation=1.0,
                    )
                ),
            ),
            SpeculativeAdoptedChild(
                state="child-b",
                adopted_decision=SpeculativeAdoptionDecision.adopt(
                    EvaluatedSpeculativeProposal(
                        execution=SpeculativeExecution(
                            proposal="proposal-b",
                            outcome="outcome-b",
                        ),
                        evaluation=2.0,
                    )
                ),
            ),
        ),
    )

    assert tuple(node.node_id for node in next_trace.frontier.active_nodes) == (
        SpeculativePlanningNodeId(1),
        SpeculativePlanningNodeId(2),
    )
    assert next_trace.frontier.select_next_active_node().state == "child-a"


def test_speculative_planning_trace_stop_clears_frontier() -> None:
    """Stopped speculative traces should clear their active frontier."""

    trace = SpeculativePlanningTrace[str, str, str, float].root(
        initial_state="root-state"
    )

    stopped_trace = trace.stop(
        reason=SpeculativeStopReason.NO_PROPOSALS_AVAILABLE
    )

    assert stopped_trace.stop_reason is SpeculativeStopReason.NO_PROPOSALS_AVAILABLE
    assert stopped_trace.frontier == SpeculativePlanningFrontier.empty()
