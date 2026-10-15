"""Evidence-based topology-resolution workflow proposal adapters."""

from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.scope import ResidueSetScope
from protrepair.state import HydrogenCoverageState, TopologyAvailabilityState
from protrepair.workflow.actions.disulfide_hydrogen import (
    DisulfideHydrogenNormalizationTransformer,
)
from protrepair.workflow.actions.disulfide_topology import (
    DisulfideTopologyResolutionTransformer,
)
from protrepair.workflow.actions.hydrogen_topology import (
    PolymerHydrogenTopologyResolutionTransformer,
)
from protrepair.workflow.actions.retained_non_polymer_topology import (
    RetainedNonPolymerTopologyResolutionTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    DISULFIDE_HYDROGEN_NORMALIZATION_CAPABILITY,
    DISULFIDE_TOPOLOGY_RESOLUTION_CAPABILITY,
    POLYMER_HYDROGEN_TOPOLOGY_RESOLUTION_CAPABILITY,
    RETAINED_NON_POLYMER_TOPOLOGY_RESOLUTION_CAPABILITY,
)


def retained_non_polymer_topology_resolution_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Select topology-only ligand prerequisites from owned readiness facts.

    Parameters
    ----------
    domain : WorkflowActionDomain
        Current chemistry facts and requested completion/refinement goals.

    Returns
    -------
    tuple[WorkflowActionProposal, ...]
        A batch of graph-only edits, or no proposal if none is required.
    """
    whole_structure = (
        domain.requested_goals.requests_whole_structure_hydrogen_population()
    )
    required = domain.explicit_repair.prerequisite_residue_ids()
    selected = tuple(
        fact.residue_id
        for fact in domain.chemistry_readiness_facts.retained_non_polymer_facts
        if (whole_structure or fact.residue_id in required)
        and fact.requires_topology_resolution()
    )
    if not selected:
        return ()
    return (
        WorkflowActionProposal(
            transformer=RetainedNonPolymerTopologyResolutionTransformer(
                scope=ResidueSetScope(residue_ids=selected)
            ),
            capability=RETAINED_NON_POLYMER_TOPOLOGY_RESOLUTION_CAPABILITY,
        ),
    )


def polymer_hydrogen_topology_resolution_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether H readiness is requested directly or for local repair.

    Parameters
    ----------
    domain : WorkflowActionDomain
        Current goals and explicit repair prerequisites.

    Returns
    -------
    bool
        Whether preserved H topology resolution may contribute to the request.
    """
    return (
        domain.requested_goals.requests_whole_structure_hydrogen_population()
        or bool(domain.explicit_repair.prerequisite_residue_ids())
    )


def polymer_hydrogen_topology_resolution_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Propose topology-only repair for complete H inventories with missing bonds.

    Parameters
    ----------
    domain : WorkflowActionDomain
        Assessed structure, chemistry, goals, and active component library.

    Returns
    -------
    tuple[WorkflowActionProposal, ...]
        One batch of supported fixed-H edits, or no proposal. Missing H atoms and
        unrealized coupled sites remain owned by hydrogen completion, which also
        resolves present fixed-H bonds.
    """
    whole_structure = (
        domain.requested_goals.requests_whole_structure_hydrogen_population()
    )
    required = domain.explicit_repair.prerequisite_residue_ids()
    candidates = tuple(
        fact.residue_id
        for fact in domain.chemistry_readiness_facts.residue_facts
        if (whole_structure or fact.residue_id in required)
        and fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
        and not fact.requires_microstate_realization()
        and fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )
    if not candidates:
        return ()
    prepared = PolymerMicrostatePreparation(
        PolymerMicrostateContext(domain.structure), domain.component_library
    )
    selected = []
    for residue_id in candidates:
        try:
            if prepared.fixed_hydrogen_topology_updates(residue_id):
                selected.append(residue_id)
        except ValueError:
            continue
    if not selected:
        return ()
    return (
        WorkflowActionProposal(
            transformer=PolymerHydrogenTopologyResolutionTransformer(
                scope=ResidueSetScope(residue_ids=tuple(selected))
            ),
            capability=POLYMER_HYDROGEN_TOPOLOGY_RESOLUTION_CAPABILITY,
        ),
    )


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


__all__ = [
    "disulfide_hydrogen_normalization_is_admissible",
    "disulfide_hydrogen_normalization_proposals",
    "disulfide_topology_resolution_is_admissible",
    "disulfide_topology_resolution_proposals",
    "polymer_hydrogen_topology_resolution_is_admissible",
    "polymer_hydrogen_topology_resolution_proposals",
]
