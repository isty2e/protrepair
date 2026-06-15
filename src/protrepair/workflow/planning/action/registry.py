"""Concrete-free workflow action registry contracts."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeAlias, TypeVar

from protrepair.workflow.contracts.request import WorkflowGoal
from protrepair.workflow.planning.action.proposals import (
    WorkflowActionProposal,
    WorkflowExecutableAction,
    WorkflowProposalAction,
)
from protrepair.workflow.planning.capability import WorkflowActionCapability

DomainT = TypeVar("DomainT")

WorkflowActionFamily: TypeAlias = type[WorkflowProposalAction]
WorkflowStateAction: TypeAlias = WorkflowExecutableAction
WorkflowStateActionType: TypeAlias = WorkflowActionFamily

WorkflowActionAdmissibility: TypeAlias = Callable[[DomainT], bool]
WorkflowActionProposalFactory = Callable[
    [DomainT],
    tuple[WorkflowActionProposal, ...],
]


@dataclass(frozen=True, slots=True)
class WorkflowActionAdoption:
    """Planner memory fact for one adopted action family."""

    action_family: WorkflowStateActionType
    capability: WorkflowActionCapability


@dataclass(frozen=True, slots=True)
class WorkflowActionRegistryEntry(Generic[DomainT]):
    """Registry entry pairing one action family with planning adapters."""

    action_type: WorkflowStateActionType
    capability: WorkflowActionCapability
    is_admissible_for_domain: WorkflowActionAdmissibility
    proposals_for_domain: WorkflowActionProposalFactory

    def supports_goal(
        self,
        goal: WorkflowGoal,
    ) -> bool:
        """Return whether this registered action can advance one requested goal."""

        return self.capability.supports_proposition(
            scope=goal.scope,
            value=goal.value,
        )

    def supports_goal_value(
        self,
        goal: WorkflowGoal,
    ) -> bool:
        """Return whether this entry supports one enum-valued goal."""

        if not isinstance(goal.value, Enum):
            return False

        return self.supports_goal(goal)

    def is_admissible(
        self,
        domain: DomainT,
    ) -> bool:
        """Return whether this action family is admissible in the active domain."""

        return self.is_admissible_for_domain(domain)

    def proposals(
        self,
        domain: DomainT,
    ) -> tuple[WorkflowActionProposal, ...]:
        """Return concrete planner proposals from this action family."""

        return self.proposals_for_domain(domain)

    def matches_action(
        self,
        action: WorkflowStateAction,
    ) -> bool:
        """Return whether one action belongs to this registry family."""

        return action.proposal_family() is self.action_type

    def adoption_for(
        self,
        action: WorkflowStateAction,
    ) -> WorkflowActionAdoption | None:
        """Return the planner-memory adoption record for one action if matched."""

        if not self.matches_action(action):
            return None

        return WorkflowActionAdoption(
            action_family=self.action_type,
            capability=self.capability,
        )


def action_adoption_for_registry(
    action: WorkflowStateAction,
    *,
    registry: tuple[WorkflowActionRegistryEntry[DomainT], ...],
) -> WorkflowActionAdoption | None:
    """Return the registered adoption record for one action."""

    for entry in registry:
        adoption = entry.adoption_for(action)
        if adoption is not None:
            return adoption

    return None


def action_registry_capabilities(
    registry: tuple[WorkflowActionRegistryEntry[DomainT], ...],
) -> tuple[WorkflowActionCapability, ...]:
    """Return planner capabilities from one registry."""

    return tuple(entry.capability for entry in registry)


def action_registry_entry_for_type(
    action_type: WorkflowStateActionType,
    *,
    registry: tuple[WorkflowActionRegistryEntry[DomainT], ...],
) -> WorkflowActionRegistryEntry[DomainT] | None:
    """Return the registry entry for one action family."""

    for entry in registry:
        if entry.action_type is action_type:
            return entry

    return None
