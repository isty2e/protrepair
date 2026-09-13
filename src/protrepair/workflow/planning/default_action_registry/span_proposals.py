"""External span-reconstruction workflow action proposal adapters."""

from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    EXTERNAL_SPAN_RECONSTRUCTION_CAPABILITY,
)


def external_span_reconstruction_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether donor span reconstruction is admissible."""

    return domain.span_reconstruction.allows_reconstruction()


def external_span_reconstruction_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit donor-span proposals for the active domain."""

    return tuple(
        WorkflowActionProposal(
            transformer=ExternalSpanReconstructionTransformer(
                reconstruction=reconstruction
            ),
            capability=EXTERNAL_SPAN_RECONSTRUCTION_CAPABILITY,
            explicitly_requested=True,
        )
        for reconstruction in domain.span_reconstruction.pending_reconstructions()
    )


__all__ = [
    "external_span_reconstruction_is_admissible",
    "external_span_reconstruction_proposals",
]
