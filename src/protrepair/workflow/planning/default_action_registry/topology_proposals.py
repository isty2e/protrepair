"""Disulfide topology-resolution workflow proposal adapters."""

from protrepair.workflow.actions.disulfide_hydrogen import (
    DisulfideHydrogenNormalizationTransformer,
)
from protrepair.workflow.actions.disulfide_topology import (
    DisulfideTopologyResolutionTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    DISULFIDE_HYDROGEN_NORMALIZATION_CAPABILITY,
    DISULFIDE_TOPOLOGY_RESOLUTION_CAPABILITY,
)

__all__ = [
    "disulfide_hydrogen_normalization_is_admissible",
    "disulfide_hydrogen_normalization_proposals",
    "disulfide_topology_resolution_is_admissible",
    "disulfide_topology_resolution_proposals",
]


def disulfide_hydrogen_normalization_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether forbidden disulfide hydrogens can be removed."""

    return domain.disulfide_hydrogen_facts.has_contradictions()


def disulfide_hydrogen_normalization_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return one atomic proposal for all forbidden disulfide hydrogens."""

    if not disulfide_hydrogen_normalization_is_admissible(domain):
        return ()

    return (
        WorkflowActionProposal(
            transformer=DisulfideHydrogenNormalizationTransformer(
                forbidden_hydrogen_atom_refs=(
                    domain.disulfide_hydrogen_facts.forbidden_hydrogen_atom_refs()
                )
            ),
            capability=DISULFIDE_HYDROGEN_NORMALIZATION_CAPABILITY,
        ),
    )


def disulfide_topology_resolution_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether canonical topology can resolve likely disulfides."""

    return domain.disulfide_topology_facts.has_promotable_candidates()


def disulfide_topology_resolution_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return one atomic proposal for all independent likely disulfides."""

    if not disulfide_topology_resolution_is_admissible(domain):
        return ()

    return (
        WorkflowActionProposal(
            transformer=DisulfideTopologyResolutionTransformer(
                candidates=domain.disulfide_topology_facts.promotable_candidates
            ),
            capability=DISULFIDE_TOPOLOGY_RESOLUTION_CAPABILITY,
        ),
    )
