"""Canonical batch model for side-chain stereochemistry correction."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics import (
    SidechainStereochemistryViolation,
    StereochemistryReport,
    detect_sidechain_stereochemistry,
)
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.errors import ResidueNotFoundError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.result import TransformationResult


@dataclass(frozen=True, slots=True)
class StereochemistryCorrectionBatch:
    """Canonical batch of localized stereochemistry corrections."""

    violations_by_residue: Mapping[
        ResidueId, tuple[SidechainStereochemistryViolation, ...]
    ]

    @classmethod
    def from_violations(
        cls,
        violations: tuple[SidechainStereochemistryViolation, ...],
    ) -> "StereochemistryCorrectionBatch":
        """Group stereochemistry violations into one canonical correction batch."""

        grouped: dict[ResidueId, list[SidechainStereochemistryViolation]] = (
            defaultdict(list)
        )
        for violation in violations:
            grouped[violation.residue_id].append(violation)

        return cls(
            violations_by_residue={
                residue_id: tuple(grouped_violations)
                for residue_id, grouped_violations in grouped.items()
            }
        )

    def is_empty(self) -> bool:
        """Return whether this batch contains no correction work."""

        return not self.violations_by_residue

    def corrected_residue_ids(self) -> frozenset[ResidueId]:
        """Return residue ids selected for localized stereochemistry rebuild."""

        return frozenset(self.violations_by_residue)

    def prepared_structure(
        self,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
    ) -> ProteinStructure:
        """Drop correction subtrees so heavy-atom completion can rebuild them."""

        prepared_residue_facets = []
        for residue_id in self.corrected_residue_ids():
            prepared_residue = self.prepared_payload(
                _completion_payload_for_structure(structure, residue_id),
                component_library=component_library,
            )
            prepared_residue_facets.append(
                (
                    prepared_residue.residue_site,
                    prepared_residue.residue_geometry,
                    prepared_residue.formal_charge_by_atom_name,
                )
            )

        return structure.with_updated_residue_facets_batch(prepared_residue_facets)

    def prepared_payload(
        self,
        residue: CompletionResiduePayload,
        *,
        component_library: ComponentLibrary,
    ) -> CompletionResiduePayload:
        """Drop the local correction subtree for one residue if selected."""

        violations = self.violations_by_residue.get(residue.residue_id)
        if violations is None:
            return residue

        corrected_atom_names = self.corrected_atom_names_for_residue(
            residue,
            component_library=component_library,
            violations=violations,
        )
        if not corrected_atom_names:
            return residue

        return residue.without_atom_sites(corrected_atom_names)

    def merged_structure(
        self,
        *,
        original_structure: ProteinStructure,
        corrected_heavy_structure: ProteinStructure,
    ) -> ProteinStructure:
        """Merge corrected heavy residues back onto the original structure."""

        corrected_residue_ids = self.corrected_residue_ids()
        if not corrected_residue_ids:
            return original_structure

        corrected_residue_facets = []
        for residue_id in corrected_residue_ids:
            corrected_residue = _completion_payload_for_structure(
                corrected_heavy_structure,
                residue_id,
            )
            corrected_residue_facets.append(
                (
                    corrected_residue.residue_site,
                    corrected_residue.residue_geometry,
                    corrected_residue.formal_charge_by_atom_name,
                )
            )

        return original_structure.with_updated_residue_facets_batch(
            corrected_residue_facets
        )

    def completed_result(
        self,
        *,
        original_structure: ProteinStructure,
        heavy_repair_result: TransformationResult,
        component_library: ComponentLibrary,
    ) -> TransformationResult:
        """Return the finalized stereochemistry-correction result."""

        merged_structure = self.merged_structure(
            original_structure=original_structure,
            corrected_heavy_structure=heavy_repair_result.structure,
        )
        remaining_report = detect_sidechain_stereochemistry(
            merged_structure,
            component_library=component_library,
            residue_ids=self.corrected_residue_ids(),
        )
        remaining_issues = self.remaining_issues(remaining_report)
        correction_repairs = self.correction_repairs(
            corrected_heavy_structure=heavy_repair_result.structure,
            component_library=component_library,
            remaining_issues=remaining_issues,
        )
        return TransformationResult(
            structure=merged_structure,
            repairs=(*heavy_repair_result.repairs, *correction_repairs),
            issues=(*heavy_repair_result.issues, *remaining_issues),
        )

    def remaining_issues(
        self,
        report: StereochemistryReport,
    ) -> tuple[ValidationIssue, ...]:
        """Return unresolved stereochemistry issues for residues in this batch."""

        corrected_residue_ids = self.corrected_residue_ids()
        return tuple(
            issue
            for issue in report.to_issues()
            if issue.residue_id in corrected_residue_ids
        )

    def correction_repairs(
        self,
        *,
        corrected_heavy_structure: ProteinStructure,
        component_library: ComponentLibrary,
        remaining_issues: tuple[ValidationIssue, ...],
    ) -> tuple[RepairEvent, ...]:
        """Return successful stereochemistry correction events for this batch."""

        failed_residue_ids = {
            issue.residue_id
            for issue in remaining_issues
            if issue.residue_id is not None
        }
        repairs: list[RepairEvent] = []
        for residue_id, violations in self.violations_by_residue.items():
            if residue_id in failed_residue_ids:
                continue

            corrected_residue = _completion_payload_for_structure(
                corrected_heavy_structure,
                residue_id,
            )
            repairs.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.STEREOCHEMISTRY_CORRECTED,
                    residue_id=residue_id,
                    component_id=corrected_residue.component_id,
                    atom_names=self.corrected_atom_names_for_residue(
                        corrected_residue,
                        component_library=component_library,
                        violations=violations,
                    ),
                    details=(
                        "localized side-chain chirality correction"
                        if len(violations) == 1
                        else (
                            "localized side-chain chirality correction "
                            "at multiple centers"
                        )
                    ),
                )
            )

        return tuple(repairs)

    def corrected_atom_names_for_residue(
        self,
        residue: CompletionResiduePayload,
        *,
        component_library: ComponentLibrary,
        violations: tuple[SidechainStereochemistryViolation, ...],
    ) -> tuple[str, ...]:
        """Return subtree atoms that should be rebuilt for one corrected residue."""

        template = component_library.get(residue.component_id)
        if template is None:
            return ()

        atom_names_to_correct: set[str] = set()
        for violation in violations:
            center_semantics = template.tetrahedral_center(
                center_atom_name=violation.center_atom_name,
                ordered_neighbor_atom_names=violation.ordered_neighbor_atom_names,
            )
            if center_semantics is None:
                continue

            atom_names_to_correct.update(
                template.correction_subtree_atom_names(center_semantics)
            )

        return tuple(
            atom_name
            for atom_name in template.ordered_atom_names()
            if atom_name in atom_names_to_correct
        )


def _completion_payload_for_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CompletionResiduePayload:
    """Resolve one facet-native residue payload from a structure."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    if residue_site is None:
        raise ResidueNotFoundError(
            f"structure has no residue {residue_id.display_token()}"
        )

    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=residue_index,
    )

    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=(
            structure.topology.residue_formal_charge_by_atom_name(
                constitution=structure.constitution,
                residue_index=residue_index,
            )
        ),
    )
