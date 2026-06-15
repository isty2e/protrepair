"""Source-microstate contradiction adjudication transformer."""

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.source_microstate import (
    MicrostateDecision,
    adjudicate_microstate_evidence,
    collect_microstate_evidence,
    validation_issue_from_microstate_decision,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.slots import ResidueIndex


def adjudicate_source_microstate_contradictions(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
) -> tuple[ProteinStructure, tuple[ValidationIssue, ...]]:
    """Return one structure with source-microstate adjudications applied."""

    active_component_library = (
        build_standard_component_library()
        if component_library is None
        else component_library
    )
    standard_component_library = build_standard_component_library()
    updated_residue_facets: list[
        tuple[
            ResidueSite,
            ResidueGeometry,
            tuple[tuple[str, int | None], ...],
        ]
    ] = []
    issues: list[ValidationIssue] = []
    for residue_index_value, residue_site in enumerate(
        structure.constitution.residue_slots
    ):
        residue_index = ResidueIndex(residue_index_value)
        residue_geometry = structure.residue_geometry(residue_index)
        source_formal_charge_by_atom_name = (
            structure.residue_formal_charge_by_atom_name(residue_index)
        )
        evidence = collect_microstate_evidence(
            residue_site,
            residue_geometry=residue_geometry,
            source_formal_charge_by_atom_name=dict(source_formal_charge_by_atom_name),
            standard_component_library=standard_component_library,
            component_library=active_component_library,
        )
        decision_record = adjudicate_microstate_evidence(evidence)
        if (
            decision_record.decision is MicrostateDecision.ADJUDICATE
            and decision_record.demoted_atom_names
        ):
            updated_charge_payload = tuple(
                (atom_name, formal_charge)
                for atom_name, formal_charge in source_formal_charge_by_atom_name
                if atom_name not in decision_record.demoted_atom_names
            )
            updated_residue_facets.append(
                (
                    residue_site,
                    residue_geometry,
                    updated_charge_payload,
                )
            )

        issue = validation_issue_from_microstate_decision(decision_record)
        if issue is not None:
            issues.append(issue)

    adjudicated_structure = (
        structure.with_updated_residue_facets_batch(updated_residue_facets)
        if updated_residue_facets
        else structure
    )
    return adjudicated_structure, tuple(issues)
