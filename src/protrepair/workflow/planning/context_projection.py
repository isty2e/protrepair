"""Structure-conditioned projections derived from workflow planning inputs."""

from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.contracts.planning import (
    WorkflowLigandContextMode,
    WorkflowPlanningContext,
)


def planning_context_is_holo_for_structure(
    planning_context: WorkflowPlanningContext,
    structure: ProteinStructure,
) -> bool:
    """Return whether planning should treat the given structure as holo."""

    if not isinstance(planning_context, WorkflowPlanningContext):
        raise TypeError(
            "planning_context_is_holo_for_structure requires a "
            "WorkflowPlanningContext"
        )
    if not isinstance(structure, ProteinStructure):
        raise TypeError(
            "planning_context_is_holo_for_structure requires a ProteinStructure"
        )
    return (
        planning_context.ligand_context_mode
        is WorkflowLigandContextMode.CONSIDER_IF_PRESENT
        and bool(structure.constitution.ligands)
    )
