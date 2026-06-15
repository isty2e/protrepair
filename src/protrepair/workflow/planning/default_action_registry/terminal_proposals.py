"""Terminal-boundary workflow action proposal adapters."""

from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    TERMINAL_AUGMENTATION_CAPABILITY,
)

__all__ = [
    "terminal_augmentation_is_admissible",
    "terminal_augmentation_proposals",
]


def terminal_augmentation_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether terminal augmentation is admissible in the active domain."""

    return (
        TerminalAugmentationTransformer.planned_candidate(
            domain.structure,
            boundary_facts=domain.boundary_facts,
            requested_goals=domain.requested_goals,
            component_library=domain.component_library,
        )
        is not None
    )


def terminal_augmentation_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return concrete terminal-augmentation proposals for the active domain."""

    terminal_transformer = TerminalAugmentationTransformer.planned_candidate(
        domain.structure,
        boundary_facts=domain.boundary_facts,
        requested_goals=domain.requested_goals,
        component_library=domain.component_library,
    )
    if terminal_transformer is None:
        return ()

    return (
        WorkflowActionProposal(
            transformer=terminal_transformer,
            capability=TERMINAL_AUGMENTATION_CAPABILITY,
        ),
    )
