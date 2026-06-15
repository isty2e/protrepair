"""Structured component-support diagnostics over canonical component libraries."""

from dataclasses import dataclass

from protrepair.chemistry import IdealGeometryHeavyAtomSemantics
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.structure.constitution import ResidueSite


@dataclass(frozen=True, slots=True)
class ComponentSupportDiagnosis:
    """Structured support status for one canonical component identifier."""

    component_id: str
    parent_standard_id: str | None
    has_component_definition: bool
    supports_template_repair: bool
    supports_hydrogenation: bool
    repair_requires_local_refinement: bool = False

    def descriptor(self) -> str:
        """Return a short human-readable descriptor for issue messages."""

        if self.parent_standard_id is None:
            return f"component {self.component_id}"

        return f"component {self.component_id} (parent {self.parent_standard_id})"


def diagnose_component_support(
    component_id: str,
    component_library: ComponentLibrary,
) -> ComponentSupportDiagnosis:
    """Return the structured support diagnosis for one component identifier."""

    normalized_component_id = component_library.normalize_component_id(component_id)
    template = component_library.get(normalized_component_id)
    heavy_atom_semantics = None if template is None else template.heavy_atom_semantics
    return ComponentSupportDiagnosis(
        component_id=normalized_component_id,
        parent_standard_id=(
            None if template is None else template.lineage_parent_component_id
        ),
        has_component_definition=template is not None,
        supports_template_repair=(
            False if template is None else template.can_repair_heavy_atoms()
        ),
        supports_hydrogenation=(
            False if template is None else template.can_add_hydrogens()
        ),
        repair_requires_local_refinement=(
            isinstance(heavy_atom_semantics, IdealGeometryHeavyAtomSemantics)
            and heavy_atom_semantics.requires_local_refinement()
        ),
    )


def missing_component_definition_issue(
    residue: ResidueSite,
    *,
    diagnosis: ComponentSupportDiagnosis,
    action: str,
) -> ValidationIssue:
    """Return a typed issue for a missing component definition."""

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.MISSING_COMPONENT_DEFINITION,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue.residue_id.display_token()} has no definition for "
            f"{diagnosis.descriptor()}; {action}"
        ),
        residue_id=residue.residue_id,
    )


def unsupported_template_repair_issue(
    residue: ResidueSite,
    *,
    diagnosis: ComponentSupportDiagnosis,
) -> ValidationIssue:
    """Return a typed issue for a known component without heavy repair semantics."""

    if diagnosis.repair_requires_local_refinement:
        message = (
            f"{residue.residue_id.display_token()} uses recognized "
            f"{diagnosis.descriptor()}, but heavy-atom template repair requires "
            "explicit local refinement; leaving residue unchanged"
        )
    else:
        message = (
            f"{residue.residue_id.display_token()} uses recognized "
            f"{diagnosis.descriptor()}, but heavy-atom template repair is "
            "not implemented; leaving residue unchanged"
        )

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR,
        severity=IssueSeverity.WARNING,
        message=message,
        residue_id=residue.residue_id,
    )


def unsupported_hydrogenation_issue(
    residue: ResidueSite,
    *,
    diagnosis: ComponentSupportDiagnosis,
) -> ValidationIssue:
    """Return a typed issue for a known component without hydrogenation support."""

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue.residue_id.display_token()} uses recognized "
            f"{diagnosis.descriptor()}, but hydrogen placement is not implemented; "
            "leaving residue unchanged during hydrogenation"
        ),
        residue_id=residue.residue_id,
    )
