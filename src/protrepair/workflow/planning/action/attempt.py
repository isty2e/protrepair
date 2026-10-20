"""Observed workflow attempts and their canonical input evidence."""

from dataclasses import dataclass

from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.planning.action.proposals import WorkflowExecutableAction


@dataclass(frozen=True, slots=True)
class WorkflowActionAttempt:
    """An action evaluation with optional evidence of its input structure.

    Parameters
    ----------
    transformer : WorkflowExecutableAction
        Evaluated action, independently of output adoption.
    input_structure : ProteinStructure or None
        Canonical input for reapplicable actions. None records historical action
        membership without claiming an evaluation on a particular structure.
    """

    transformer: WorkflowExecutableAction
    input_structure: ProteinStructure | None = None

    def blocks(
        self, transformer: WorkflowExecutableAction, structure: ProteinStructure
    ) -> bool:
        """Check whether this attempt suppresses the current proposal.

        Parameters
        ----------
        transformer : WorkflowExecutableAction
            Current proposed action.
        structure : ProteinStructure
            Its canonical input.

        Returns
        -------
        bool
            True for a matching branch-only action, or for equal evidenced inputs
            of a reapplicable action. Unbound history cannot prove input equality.
        """
        if self.transformer != transformer:
            return False

        current_input = transformer.reapplication_context(structure)
        if current_input is None:
            return True

        return self.input_structure is current_input or (
            self.input_structure is not None and self.input_structure == current_input
        )
