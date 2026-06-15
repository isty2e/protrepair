"""Stereochemistry-correction workflow action proposal adapters."""

from protrepair.diagnostics.stereochemistry import detect_sidechain_stereochemistry
from protrepair.scope import ResidueSetScope
from protrepair.state import StereochemistryState
from protrepair.transformer.completion.stereochemistry.batch import (
    StereochemistryCorrectionBatch,
)
from protrepair.workflow.actions.stereochemistry import (
    StereochemistryCorrectionTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    STEREOCHEMISTRY_CORRECTION_CAPABILITY,
)

__all__ = [
    "stereochemistry_correction_is_admissible",
    "stereochemistry_correction_proposals",
]


def stereochemistry_correction_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether stereochemistry correction is admissible."""

    intrinsic_geometry_facts = domain.intrinsic_geometry_facts
    if intrinsic_geometry_facts is None:
        return False

    return (
        intrinsic_geometry_facts.stereochemistry_state
        is StereochemistryState.VIOLATED
    )


def stereochemistry_correction_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return stereochemistry-correction proposals for the active domain."""

    if not stereochemistry_correction_is_admissible(domain):
        return ()

    report = detect_sidechain_stereochemistry(
        domain.structure,
        component_library=domain.component_library,
    )
    if report.is_empty():
        return ()

    correction_batch = StereochemistryCorrectionBatch.from_violations(
        report.violations
    )
    residue_ids = tuple(sorted(correction_batch.corrected_residue_ids()))
    if not residue_ids:
        return ()

    return (
        WorkflowActionProposal(
            transformer=StereochemistryCorrectionTransformer(
                scope=ResidueSetScope(residue_ids=residue_ids)
            ),
            capability=STEREOCHEMISTRY_CORRECTION_CAPABILITY,
        ),
    )
