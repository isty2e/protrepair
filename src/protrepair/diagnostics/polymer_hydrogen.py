"""Diagnostic projection of unresolved fixed polymer H inventories."""

from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType


def unresolved_fixed_hydrogen_issue(
    preparation: AminoAcidPreparation,
    residue_id: ResidueId,
    *,
    reference_failure_detail: str | None = None,
) -> ValidationIssue | None:
    """Report unresolved fixed H without reinterpreting its inventory.

    Parameters
    ----------
    preparation : AminoAcidPreparation
        Selected chemistry bound to the current snapshot and library.
    residue_id : ResidueId
        Polymer identity to report.
    reference_failure_detail : str or None
        Additional current failure from read-only reference H assessment.

    Returns
    -------
    ValidationIssue or None
        Parent-local reason, any independent inventory failure and untyped-link
        guidance, or no unresolved parent.

    Raises
    ------
    ValueError
        The residue is absent from the preparation snapshot.
    """
    fixed = preparation.fixed_hydrogen_context(residue_id)
    if (
        fixed is None or not fixed.unresolved_parent_names
    ) and reference_failure_detail is None:
        return None

    context = preparation.context
    residue = context.source.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    if fixed is None or not fixed.unresolved_parent_names:
        return ValidationIssue.for_residue(
            kind=ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN,
            severity=IssueSeverity.WARNING,
            residue_id=residue_id,
            component_id=residue.component_id,
            message=f"polymer H coverage is indeterminate: {reference_failure_detail}",
        )
    inventory_failure = ""
    try:
        preparation.fixed_hydrogen_atom_sites(residue_id)
    except ValueError as error:
        inventory_failure = f"; other fixed H inventory could not be selected: {error}"

    untyped = any(
        bond.relationship_type is BondRelationshipType.UNKNOWN
        for parent in fixed.unresolved_parent_names
        for bond in context.external_heavy_bonds(AtomRef(residue_id, parent))
    )
    guidance = (
        "; an untyped connection record does not establish covalent or metal "
        "chemistry; use a chemically appropriate typed LINK or mmCIF _struct_conn "
        "record to resolve relationship ambiguity"
        if untyped
        else ""
    )
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN,
        severity=IssueSeverity.WARNING,
        residue_id=residue_id,
        component_id=residue.component_id,
        message=(
            f"linked polymer H at {', '.join(sorted(fixed.unresolved_parent_names))} "
            f"left unresolved: {fixed.failure_reason}{guidance}{inventory_failure}"
            + (f"; {reference_failure_detail}" if reference_failure_detail else "")
        ),
    )
