"""Planner-visible concrete workflow action proposals."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from protrepair.scope import Scope, ScopeKind
from protrepair.workflow.planning.capability import (
    WorkflowActionCapability,
    WorkflowActionLocality,
)

if TYPE_CHECKING:
    from protrepair.structure.aggregate import ProteinStructure
    from protrepair.transformer.result import TransformationResult
    from protrepair.workflow.actions.context import TransformerExecutionContext


class WorkflowProposalAction(Protocol):
    """Planner-visible action-family identity."""

    def proposal_family(self) -> type["WorkflowProposalAction"]:
        """Return the frontier proposal family key for this action."""

        ...


class WorkflowExecutableAction(WorkflowProposalAction, Protocol):
    """Workflow action that can materialize a process-result transition."""

    def reapplication_context(
        self, structure: "ProteinStructure"
    ) -> "ProteinStructure | None":
        """Declare the context in which an earlier attempt suppresses reapplication.

        Parameters
        ----------
        structure : ProteinStructure
            Current canonical input under the branch's fixed execution settings.

        Returns
        -------
        ProteinStructure or None
            Carrier for exact-input suppression, or None for branch-only execution.
        """

        ...

    def execute(
        self,
        carrier: "TransformationResult",
        *,
        context: "TransformerExecutionContext",
    ) -> "TransformationResult":
        """Execute this action against one workflow result and context."""

        ...


@runtime_checkable
class _ScopedWorkflowAction(Protocol):
    @property
    def workflow_scope(self) -> Scope: ...


class WorkflowProposalSelectionMode(str, Enum):
    """Whether a transition is required or competes with its unchanged parent."""

    REQUIRED = "required"
    QUALITY_ALTERNATIVE = "quality_alternative"


@dataclass(frozen=True, slots=True)
class WorkflowActionProposal:
    """A concrete action and its planner-owned selection semantics.

    Parameters
    ----------
    transformer : WorkflowExecutableAction
        Action bound to the current canonical state.
    capability : WorkflowActionCapability
        Declared physical effects and supported planning facts.
    explicitly_requested : bool
        Whether the caller requires this action rather than automatic selection.
    selection_mode : WorkflowProposalSelectionMode
        Required transition or quality alternative. Explicit requests always
        prevent quality rollback, independently of this mode or physical effect.

    Raises
    ------
    TypeError
        Selection mode is not a canonical enum value.
    """

    transformer: WorkflowExecutableAction
    capability: WorkflowActionCapability
    explicitly_requested: bool = False
    selection_mode: WorkflowProposalSelectionMode = (
        WorkflowProposalSelectionMode.REQUIRED
    )

    def __post_init__(self) -> None:
        if not isinstance(self.selection_mode, WorkflowProposalSelectionMode):
            raise TypeError("proposal selection mode must be canonical")

    def allows_parent_retention(self) -> bool:
        """Return whether quality regression may keep the unchanged parent.

        Returns
        -------
        bool
            True only for an alternative not explicitly requested by the caller.
        """
        return (
            self.selection_mode is WorkflowProposalSelectionMode.QUALITY_ALTERNATIVE
            and not self.explicitly_requested
        )

    def target_locality(self) -> WorkflowActionLocality | None:
        """Project a supported residue-set or whole-structure target locality.

        Returns
        -------
        WorkflowActionLocality or None
            Concrete target locality, or None for unscoped, unmapped or
            capability-unsupported targets. This does not describe native motion.
        """

        if not isinstance(self.transformer, _ScopedWorkflowAction):
            return None

        scope = self.transformer.workflow_scope
        if not isinstance(scope, Scope):
            return None

        locality = {
            ScopeKind.RESIDUE_SET: WorkflowActionLocality.RESIDUE_SET,
            ScopeKind.WHOLE_STRUCTURE: WorkflowActionLocality.WHOLE_STRUCTURE,
        }.get(scope.kind)
        if locality is None or not self.capability.supports_locality(locality):
            return None

        return locality
