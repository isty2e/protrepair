"""Diagnostics for unresolved polymer hydrogen inventories and peptide linkages."""

from dataclasses import dataclass

from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import input_declares_thiolate
from protrepair.chemistry.microstate.preparation import (
    AminoAcidPreparation,
    AminoAcidSitePreparation,
)
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.chemistry.peptide_caps import LinkedPeptideCap
from protrepair.diagnostics.events import EventScope, ResidueAtomImpact, ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.structure.bound_peptide_ports import active_peptide_ports
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingCapLinkageIssue(ValidationIssue):
    """An H-stage warning that a verified cap at the same port can resolve.

    Parameters
    ----------
    port : AtomRef
        Polymer N or C whose linkage was unknown during H preparation.
    """

    port: AtomRef

    @classmethod
    def for_unresolved_target(
        cls, target: AminoAcidSitePreparation, message: str
    ) -> "PendingCapLinkageIssue | None":
        """Identify a pending cap without reclassifying other placement failures.

        Parameters
        ----------
        target : AminoAcidSitePreparation
            Backbone or side-chain chemistry assessed by the H stage.
        message : str
            Original placement diagnostic, retained if the port stays unresolved.

        Returns
        -------
        PendingCapLinkageIssue or None
            A port-specific warning only for insufficient chemistry at a reserved
            backbone end with unknown linkage.
        """
        binding = active_peptide_ports()
        if (
            binding is None
            or target.site.kind is PolymerChemicalSite.SIDECHAIN
            or target.site.linkage is not PeptideLinkage.UNKNOWN
            or target.resolution.status is not MicrostateResolutionStatus.INSUFFICIENT
        ):
            return None
        port = AtomRef(
            target.residue_id,
            "N" if target.site.kind is PolymerChemicalSite.BACKBONE_N else "C",
        )
        if not binding.reserves(target.context.source, port):
            return None

        return cls(
            port=port,
            kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
            severity=IssueSeverity.WARNING,
            message=message,
            scope=EventScope.for_residue(port.residue_id),
            residue_impacts=(
                ResidueAtomImpact(
                    port.residue_id,
                    target.site.template.component_id,
                    tuple(sorted(target.controlled_parent_names())),
                ),
            ),
        )


def linked_cap_issue(cap: LinkedPeptideCap, detail: str) -> ValidationIssue:
    """Report why an occupied cap was left unprepared."""
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
        severity=IssueSeverity.WARNING,
        residue_id=cap.residue_id,
        component_id=cap.component_id,
        message=f"Linked cap hydrogenation unavailable: {detail}",
    )


def declared_thiolate_conflict(
    target: AminoAcidSitePreparation,
) -> ValidationIssue | None:
    """Report a conflicting sulfur assignment even if no hydrogen needs adding.

    Parameters
    ----------
    target : AminoAcidSitePreparation
        Site bound to the structure being diagnosed.

    Returns
    -------
    ValidationIssue or None
        The contradiction on a declared cysteine thiolate, if present.
    """
    if (
        target.resolution.status is not MicrostateResolutionStatus.CONFLICT
        or target.site.kind is not PolymerChemicalSite.SIDECHAIN
        or target.site.template.component_id != "CYS"
        or not input_declares_thiolate(target.context.source, target.residue_id)
    ):
        return None
    detail = "; ".join(target.resolution.details)
    reason = f": {detail}" if detail else ""
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
        severity=IssueSeverity.WARNING,
        residue_id=target.residue_id,
        component_id="CYS",
        atom_names=("SG",),
        message=f"Cysteine SG has charge -1 but its site evidence is incompatible with a thiolate{reason}; left unchanged.",
    )


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
        guidance, or None when no unresolved inventory is found.

    Raises
    ------
    ValueError
        The residue is absent from the preparation snapshot.
    """
    cap = preparation.linked_peptide_cap(residue_id)
    if cap is not None:
        return None if cap.refusal is None else linked_cap_issue(cap, cap.refusal)
    fixed = preparation.fixed_hydrogen_context(residue_id)
    unresolved = fixed is not None and bool(fixed.unresolved_parent_names)
    inventory_failure = None
    if unresolved or any(t.is_requested() for t in preparation.targets_for(residue_id)):
        try:
            preparation.fixed_hydrogen_atom_sites(residue_id)
        except ValueError as error:
            inventory_failure = str(error)
    if (
        not unresolved
        and inventory_failure is None
        and reference_failure_detail is None
    ):
        return None

    context = preparation.context
    residue = context.source.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    if fixed is None or not fixed.unresolved_parent_names:
        details = "; ".join(
            d for d in (reference_failure_detail, inventory_failure) if d
        )
        return ValidationIssue.for_residue(
            kind=ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN,
            severity=IssueSeverity.WARNING,
            residue_id=residue_id,
            component_id=residue.component_id,
            message=f"polymer H coverage is indeterminate: {details}",
        )

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
            f"left unresolved: {fixed.failure_reason}{guidance}"
            + (
                f"; other fixed H inventory could not be selected: {inventory_failure}"
                if inventory_failure
                else ""
            )
            + (f"; {reference_failure_detail}" if reference_failure_detail else "")
        ),
    )
