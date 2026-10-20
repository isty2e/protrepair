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
from protrepair.transformer.base import ProjectedCodomainState
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.refinement.result import (
    LocalRefinementResult,
    RefinementTransformationResult,
)
from protrepair.transformer.result import TransformationResult


class RefinementStage(str, Enum):
    """Internal transformer stages that can apply local refinement."""

    REPAIR = "repair-stage"
    BACKBONE_WINDOW = "backbone-window"


@dataclass(frozen=True, slots=True)
class RefinementStageOutcome:
    """Stage-local refinement output, independent of coordinate adoption.

    Parameters
    ----------
    structure : ProteinStructure
        Selected stage output, possibly its unchanged parent.
    repairs : tuple[RepairEvent, ...]
        Stage repair events.
    issues : tuple[ValidationIssue, ...]
        Stage diagnostics.
    relaxation_executions : tuple[ContinuousRelaxationExecution, ...]
        Observed native returns, including unselected candidates.
    """

    structure: ProteinStructure
    repairs: tuple[RepairEvent, ...]
    issues: tuple[ValidationIssue, ...]
    relaxation_executions: tuple[ContinuousRelaxationExecution, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "repairs", tuple(self.repairs))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(
            self,
            "relaxation_executions",
            ContinuousRelaxationExecution.deduplicated(self.relaxation_executions),
        )

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
                constitution.residue_index_for_atom_index(moved_atom.after_atom_index)
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
            relaxation_executions=LocalRefinementResult.observed_executions(
                refinement_result
            ),
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
                            f"{stage.value} local refinement was skipped: {message}"
                        ),
                    )
                    if residue_id is not None
                    else ValidationIssue(
                        kind=ValidationIssueKind.REFINEMENT_REJECTED,
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"{stage.value} local refinement was skipped: {message}"
                        ),
                    )
                ),
            ),
        )


def merge_refinement_stage_outcome(
    result: TransformationResult,
    outcome: RefinementStageOutcome,
) -> RefinementTransformationResult:
    """Return one transform result with one stage-local outcome merged in."""

    return replace(
        RefinementTransformationResult.with_observed_executions(
            result, outcome.relaxation_executions
        ),
        structure=outcome.structure,
        repairs=result.repairs + outcome.repairs,
        issues=result.issues + outcome.issues,
    )


@dataclass(frozen=True, slots=True)
class RefinementCodomainState(ProjectedCodomainState[ProteinStructure]):
    """Refinement projection retaining native facts through carrier lift.

    Parameters
    ----------
    scope : Scope
        Semantic action scope, not a substitute for native movable atoms.
    state : ProteinStructure
        Selected structural output.
    repairs : tuple[RepairEvent, ...]
        Newly produced repair events.
    issues : tuple[ValidationIssue, ...]
        Newly produced diagnostics.
    relaxation_executions : tuple[ContinuousRelaxationExecution, ...]
        Native returns observed while producing this output.
    """

    relaxation_executions: tuple[ContinuousRelaxationExecution, ...] = ()

    def __post_init__(self) -> None:
        ProjectedCodomainState.__post_init__(self)
        object.__setattr__(
            self,
            "relaxation_executions",
            ContinuousRelaxationExecution.deduplicated(self.relaxation_executions),
        )

    def lift(self, carrier: TransformationResult) -> RefinementTransformationResult:
        """Merge projected structure, diagnostics and native returns into a carrier.

        Parameters
        ----------
        carrier : TransformationResult
            Accumulated input result.

        Returns
        -------
        RefinementTransformationResult
            Accumulated diagnostics and native facts with the selected structure.
        """

        return merge_refinement_stage_outcome(
            carrier,
            RefinementStageOutcome(
                structure=self.state,
                repairs=self.repairs,
                issues=self.issues,
                relaxation_executions=self.relaxation_executions,
            ),
        )
