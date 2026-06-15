"""Diagnostics adapters for contradictory chemistry evidence across sources."""

from typing import Protocol

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.source_microstate import (
    adjudicate_microstate_evidence,
    collect_microstate_evidence,
    validation_issue_from_microstate_decision,
)
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry


class _RetainedNonPolymerPayloadLike(Protocol):
    """Payload shape accepted by the compatibility diagnostics adapter."""

    residue_site: ResidueSite
    residue_geometry: ResidueGeometry
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...]


def diagnose_retained_non_polymer_template_chemistry_contradictions(
    residue_site: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    source_formal_charge_by_atom_name: dict[str, int | None],
    template: ResidueTemplate,
) -> tuple[ValidationIssue, ...]:
    """Return typed issues for contradictory retained non-polymer chemistry."""

    if not isinstance(residue_site, ResidueSite):
        raise TypeError(
            "retained non-polymer chemistry diagnostics require a ResidueSite"
        )
    if not isinstance(residue_geometry, ResidueGeometry):
        raise TypeError(
            "retained non-polymer chemistry diagnostics require a ResidueGeometry"
        )

    evidence = collect_microstate_evidence(
        residue_site,
        residue_geometry=residue_geometry,
        source_formal_charge_by_atom_name=source_formal_charge_by_atom_name,
        standard_component_library=build_standard_component_library(),
        component_library=ComponentLibrary(
            templates={template.component_id: template}
        ),
    )
    decision_record = adjudicate_microstate_evidence(evidence)
    issue = validation_issue_from_microstate_decision(decision_record)
    if issue is None:
        return ()

    return (issue,)


def diagnose_retained_non_polymer_payload_template_chemistry_contradictions(
    payload: _RetainedNonPolymerPayloadLike,
    *,
    template: ResidueTemplate,
) -> tuple[ValidationIssue, ...]:
    """Compatibility adapter for payload-shaped retained non-polymer diagnostics."""

    return diagnose_retained_non_polymer_template_chemistry_contradictions(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
        source_formal_charge_by_atom_name=dict(payload.formal_charge_by_atom_name),
        template=template,
    )
