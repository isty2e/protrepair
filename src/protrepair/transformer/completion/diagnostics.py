"""Canonical diagnostics for completion-stage outcomes."""

from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload


def skipped_geometry_placement_issue(
    residue: CompletionResiduePayload,
    *,
    atom_names: tuple[str, ...],
    reason: str,
) -> ValidationIssue:
    """Return one atom-scoped warning for a recoverable placement failure."""

    normalized_atom_names = tuple(
        dict.fromkeys(atom_name.strip().upper() for atom_name in atom_names)
    )
    if not normalized_atom_names or any(
        not atom_name for atom_name in normalized_atom_names
    ):
        raise ValueError("skipped geometry placement requires named atoms")
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("skipped geometry placement requires a failure reason")
    atom_summary = ", ".join(normalized_atom_names)
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.GEOMETRY_PLACEMENT_SKIPPED,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue.residue_id.display_token()} skipped geometry placement "
            f"for {atom_summary} ({normalized_reason}); leaving those atoms absent"
        ),
        residue_id=residue.residue_id,
        component_id=residue.component_id,
        atom_names=normalized_atom_names,
    )
