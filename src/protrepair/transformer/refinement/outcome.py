"""Refinement-stage outcome helpers for structure transformers."""

from dataclasses import dataclass, replace
from enum import Enum

from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.result import TransformationResult


class RefinementStage(str, Enum):
    """Internal transformer stages that can apply local refinement."""

    REPAIR = "repair-stage"


@dataclass(frozen=True, slots=True)
class RefinementStageOutcome:
    """Stage-local refinement outcome before merging into a transform result."""

    structure: ProteinStructure
    repairs: tuple[RepairEvent, ...]
    issues: tuple[ValidationIssue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "repairs", tuple(self.repairs))
        object.__setattr__(self, "issues", tuple(self.issues))

    @classmethod
    def from_result(
        cls,
        refinement_result: RegionTransformationResult,
        *,
        stage: RefinementStage,
    ) -> "RefinementStageOutcome":
        """Return one stage-local outcome from a successful refinement result."""

        constitution = refinement_result.refined_structure.constitution
        moved_atom_names_by_residue: dict[ResidueId, list[str]] = {}
        for moved_atom in refinement_result.delta.moved_atoms:
            atom_site = constitution.atom_site_at(moved_atom.after_atom_index)
            residue_id = constitution.residue_site_at(
                constitution.residue_index_for_atom_index(
                    moved_atom.after_atom_index
                )
            ).residue_id
            moved_atom_names_by_residue.setdefault(residue_id, []).append(
                atom_site.name
            )

        repairs: list[RepairEvent] = []
        for residue in refinement_result.refined_structure.iter_residue_sites(
            include_ligands=True
        ):
            moved_atom_names = moved_atom_names_by_residue.get(residue.residue_id)
            if not moved_atom_names:
                continue

            repairs.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.LOCAL_REFINEMENT_APPLIED,
                    residue_id=residue.residue_id,
                    component_id=residue.component_id,
                    atom_names=tuple(moved_atom_names),
                    details=(
                        f"{stage.value} local refinement via "
                        f"{refinement_result.backend_name}"
                    ),
                )
            )

        return cls(
            structure=refinement_result.refined_structure,
            repairs=tuple(repairs),
            issues=refinement_result.issues,
        )

    @classmethod
    def rejected(
        cls,
        structure: ProteinStructure,
        *,
        stage: RefinementStage,
        message: str,
        residue_id: ResidueId | None,
    ) -> "RefinementStageOutcome":
        """Return one rejected stage-local refinement outcome."""

        return cls(
            structure=structure,
            repairs=(),
            issues=(
                (
                    ValidationIssue.for_residue(
                        kind=ValidationIssueKind.REFINEMENT_REJECTED,
                        severity=IssueSeverity.WARNING,
                        residue_id=residue_id,
                        message=(
                            f"{stage.value} local refinement was skipped: "
                            f"{message}"
                        ),
                    )
                    if residue_id is not None
                    else ValidationIssue(
                        kind=ValidationIssueKind.REFINEMENT_REJECTED,
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"{stage.value} local refinement was skipped: "
                            f"{message}"
                        ),
                    )
                ),
            ),
        )


def merge_refinement_stage_outcome(
    result: TransformationResult,
    outcome: RefinementStageOutcome,
) -> TransformationResult:
    """Return one transform result with one stage-local outcome merged in."""

    return replace(
        result,
        structure=outcome.structure,
        repairs=result.repairs + outcome.repairs,
        issues=result.issues + outcome.issues,
    )
