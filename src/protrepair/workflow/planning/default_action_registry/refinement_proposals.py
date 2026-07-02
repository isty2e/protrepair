"""Local and backbone-window refinement workflow proposal adapters."""

from protrepair.state import StereochemistryState
from protrepair.workflow.actions.backbone_window_refinement import (
    BackboneWindowRefinementTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
    LOCAL_REFINEMENT_CAPABILITY,
)
from protrepair.workflow.planning.local_refinement_policy import (
    LocalRefinementProposalPolicy,
)

__all__ = [
    "backbone_window_refinement_is_admissible",
    "backbone_window_refinement_proposals",
    "local_refinement_is_admissible",
    "local_refinement_proposals",
]

DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY = LocalRefinementProposalPolicy()


def local_refinement_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether local refinement is admissible in the active domain."""

    if (
        domain.intrinsic_geometry_facts is not None
        and domain.intrinsic_geometry_facts.stereochemistry_state
        is StereochemistryState.VIOLATED
    ):
        return False

    if domain.explicit_repair.is_requested():
        return (
            domain.burden.has_intrinsic_geometry_burden()
            or domain.burden.has_parser_compatibility_burden()
            or domain.burden.has_interaction_burden()
            or domain.memory.has_prior_augmentation_adoption()
        )

    return DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY.automatic_is_admissible(domain)


def backbone_window_refinement_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether requested backbone-window operator work is admissible."""

    return bool(domain.state_deficit.backbone_window_operator)


def backbone_window_refinement_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit backbone-window refinement proposals."""

    return tuple(
        WorkflowActionProposal(
            transformer=BackboneWindowRefinementTransformer.from_window_spec(
                operator_deficit.window_spec
            ),
            capability=BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
            explicitly_requested=True,
        )
        for operator_deficit in domain.state_deficit.backbone_window_operator
    )


def local_refinement_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit or automatic local-refinement proposals."""

    repair_refinement = domain.transform_requests.repair_refinement
    if repair_refinement is None:
        return tuple(
            WorkflowActionProposal(
                transformer=LocalRefinementTransformer.from_repair_refinement(
                    repair_refinement_spec
                ),
                capability=LOCAL_REFINEMENT_CAPABILITY,
                explicitly_requested=False,
            )
            for repair_refinement_spec in (
                DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY.automatic_specs(domain)
            )
        )

    return tuple(
        WorkflowActionProposal(
            transformer=LocalRefinementTransformer.from_repair_refinement(
                repair_refinement_spec
            ),
            capability=LOCAL_REFINEMENT_CAPABILITY,
            explicitly_requested=True,
        )
        for repair_refinement_spec in (
            DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY.explicit_specs(
                domain,
                repair_refinement=repair_refinement,
            )
        )
    )
