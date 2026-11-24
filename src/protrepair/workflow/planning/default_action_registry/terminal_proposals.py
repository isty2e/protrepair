"""Terminal-boundary workflow action proposal adapters."""

from protrepair.structure.bound_peptide_ports import active_peptide_ports
from protrepair.transformer.completion.terminal.cap_placement import CapPortAssessment
from protrepair.workflow.actions.peptide_capping import PeptideCappingTransformer
from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import (
    WorkflowActionProposal,
    WorkflowProposalSelectionMode,
)
from protrepair.workflow.planning.default_action_registry.capabilities import (
    PEPTIDE_CAPPING_CAPABILITY,
    TERMINAL_AUGMENTATION_CAPABILITY,
)


def terminal_augmentation_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether terminal augmentation is admissible in the active domain."""

    return (
        TerminalAugmentationTransformer.planned_candidate(
            domain.structure,
            boundary_facts=domain.boundary_facts,
            requested_goals=domain.requested_goals,
            component_library=domain.component_library,
            required_residue_ids=domain.explicit_repair.prerequisite_residue_ids(),
            observed_preparation=(
                domain.observed_preparation
                if domain.hydrogen_preparation_is_requested()
                else None
            ),
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
        required_residue_ids=domain.explicit_repair.prerequisite_residue_ids(),
        observed_preparation=(
            domain.observed_preparation
            if domain.hydrogen_preparation_is_requested()
            else None
        ),
    )
    if terminal_transformer is None:
        return ()

    return (
        WorkflowActionProposal(
            transformer=terminal_transformer,
            capability=TERMINAL_AUGMENTATION_CAPABILITY,
        ),
    )


def peptide_cap_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Offer checked ports; prerequisite and attempt filtering select at most one."""
    binding = active_peptide_ports()
    if binding is None:
        return ()
    parent = domain.structure
    proposals = []
    for port, address in sorted(
        (dict(binding.attached) | dict(binding.cap_ids)).items()
    ):
        if CapPortAssessment.observe(binding, parent, port).requires_action():
            proposals.append(
                WorkflowActionProposal(
                    transformer=PeptideCappingTransformer(
                        port, address, parent, binding
                    ),
                    capability=PEPTIDE_CAPPING_CAPABILITY,
                    selection_mode=WorkflowProposalSelectionMode.QUALITY_ALTERNATIVE,
                )
            )
    return tuple(proposals)


__all__ = [
    "peptide_cap_proposals",
    "terminal_augmentation_is_admissible",
    "terminal_augmentation_proposals",
]
