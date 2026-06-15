"""Planner-visible concrete workflow action proposals."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from protrepair.workflow.planning.capability import WorkflowActionCapability

if TYPE_CHECKING:
    from protrepair.transformer.result import TransformationResult
    from protrepair.workflow.actions.context import TransformerExecutionContext


class WorkflowProposalAction(Protocol):
    """Planner-visible action-family identity."""

    def proposal_family(self) -> type["WorkflowProposalAction"]:
        """Return the frontier proposal family key for this action."""

        ...


class WorkflowExecutableAction(WorkflowProposalAction, Protocol):
    """Workflow action that can materialize a process-result transition."""

    def execute(
        self,
        carrier: "TransformationResult",
        *,
        context: "TransformerExecutionContext",
    ) -> "TransformationResult":
        """Execute this action against one workflow result and context."""

        ...


@dataclass(frozen=True, slots=True)
class WorkflowActionProposal:
    """One planner-visible proposal emitted by one workflow action family."""

    transformer: WorkflowExecutableAction
    capability: WorkflowActionCapability
    explicitly_requested: bool = False
