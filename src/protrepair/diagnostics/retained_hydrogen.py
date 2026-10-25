"""Current H-preparation diagnostics for reference-backed retained components."""

from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryFailure,
)
from protrepair.structure.labels import ResidueId


def unresolved_retained_hydrogen_issue(
    residue_id: ResidueId,
    component_id: str,
    failure: RetainedNonPolymerChemistryFailure,
) -> ValidationIssue:
    """Describe unresolved preparation without claiming a user opt-out.

    Parameters
    ----------
    residue_id : ResidueId
        Current retained component identity.
    component_id : str
        Active component identifier.
    failure : RetainedNonPolymerChemistryFailure
        Canonical unresolved-preparation cause.

    Returns
    -------
    ValidationIssue
        Residue-local current-state warning, not an execution-history event.
    """
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN,
        severity=IssueSeverity.WARNING,
        residue_id=residue_id,
        component_id=component_id,
        message=f"{residue_id.display_token()} {component_id}: {failure.detail}; "
        "leaving source atoms unchanged",
    )
