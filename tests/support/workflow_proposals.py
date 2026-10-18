"""Proposal fixtures for runtime tests that replace the concrete planner."""

from protrepair.workflow.planning.action.proposals import (
    WorkflowActionProposal,
    WorkflowProposalSelectionMode,
)
from protrepair.workflow.planning.action.registry import WorkflowStateAction
from protrepair.workflow.planning.default_action_registry import (
    WORKFLOW_ACTION_REGISTRY,
)


def workflow_proposals(
    transformers: tuple[WorkflowStateAction, ...],
    *,
    selection_mode: WorkflowProposalSelectionMode = (
        WorkflowProposalSelectionMode.REQUIRED
    ),
    explicitly_requested: bool = False,
) -> tuple[WorkflowActionProposal, ...]:
    """Pair fixture actions with registered capabilities and explicit test intent."""
    proposals = []
    for transformer in transformers:
        entry = next(
            entry
            for entry in WORKFLOW_ACTION_REGISTRY
            if entry.matches_action(transformer)
        )
        proposals.append(
            WorkflowActionProposal(
                transformer=transformer,
                capability=entry.capability,
                explicitly_requested=explicitly_requested,
                selection_mode=selection_mode,
            )
        )
    return tuple(proposals)
