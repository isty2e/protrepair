"""Explainable action-selection policy over canonical legal transformation sets."""

from dataclasses import dataclass
from enum import Enum

from protrepair.workflow.planning.transformation.legality import (
    LegalTransformationFamily,
    LegalTransformationFamilySet,
    LocalPreparationReason,
    LocalTransformationStratum,
    TerminationDecision,
)
from protrepair.workflow.planning.transformation.runtime import (
    LocalTransformationFamily,
    TransformationTerminationReason,
)


class TransformationSelectionReason(str, Enum):
    """Closed explanations for why one legal transformation was selected."""

    PREPARATION_REPAIRS_RELAXATION_TOPOLOGY_PRECONDITIONS = (
        "preparation_repairs_relaxation_topology_preconditions"
    )
    CHEMISTRY_PREPARATION_REPAIRS_LOCAL_STATE = (
        "chemistry_preparation_repairs_local_state"
    )
    GEOMETRY_PREPARATION_REPAIRS_LOCAL_STATE = (
        "geometry_preparation_repairs_local_state"
    )
    CANDIDATE_CONSTRUCTION_PREPARES_RELAXATION_CANDIDATES = (
        "candidate_construction_prepares_relaxation_candidates"
    )
    CONTINUOUS_RELAXATION_ONLY_REMAINING_ACTION = (
        "continuous_relaxation_only_remaining_action"
    )


@dataclass(frozen=True, slots=True)
class SelectedTransformationFamily:
    """One chosen legal transformation family with a selection reason."""

    legal_family: LegalTransformationFamily
    reason: TransformationSelectionReason


@dataclass(frozen=True, slots=True)
class TransformationPlanningDecision:
    """One explainable planning decision over a canonical legal-family set."""

    selected_transformation_family: SelectedTransformationFamily | None = None
    termination: TerminationDecision | None = None

    def __post_init__(self) -> None:
        if (
            self.selected_transformation_family is None
            and self.termination is None
        ):
            raise ValueError(
                "planning decisions require either a selected transformation "
                "or a termination decision"
            )

        if (
            self.selected_transformation_family is not None
            and self.termination is not None
            and self.termination.is_terminal()
        ):
            raise ValueError(
                "planning decisions cannot select an action and terminate "
                "at the same time"
            )

    def is_terminal(self) -> bool:
        """Return whether the planning decision is terminal."""

        return self.termination is not None and self.termination.is_terminal()

    def selected_family(self) -> LocalTransformationFamily | None:
        """Return the chosen transformation family when one family was selected."""

        if self.selected_transformation_family is None:
            return None

        return self.selected_transformation_family.legal_family.family


def choose_next_transformation(
    *,
    legal_transformations: LegalTransformationFamilySet,
    termination: TerminationDecision | None = None,
) -> TransformationPlanningDecision:
    """Choose the next family from a canonical legal-family set."""

    if termination is not None and termination.is_terminal():
        return TransformationPlanningDecision(termination=termination)

    if legal_transformations.contains_family(
        LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
    ):
        legal_family = legal_transformations.family_record_for(
            LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION
        )
        return TransformationPlanningDecision(
            selected_transformation_family=SelectedTransformationFamily(
                legal_family=legal_family,
                reason=_preparation_selection_reason(legal_family),
            )
        )

    if legal_transformations.contains_family(
        LocalTransformationFamily.BRANCHED_SIDECHAIN_SEED
    ):
        return TransformationPlanningDecision(
            selected_transformation_family=SelectedTransformationFamily(
                legal_family=legal_transformations.family_record_for(
                    LocalTransformationFamily.BRANCHED_SIDECHAIN_SEED
                ),
                reason=(
                    TransformationSelectionReason
                    .CANDIDATE_CONSTRUCTION_PREPARES_RELAXATION_CANDIDATES
                ),
            )
        )

    if legal_transformations.contains_stratum(
        LocalTransformationStratum.RELAXATION
    ):
        return TransformationPlanningDecision(
            selected_transformation_family=SelectedTransformationFamily(
                legal_family=legal_transformations.family_record_for(
                    LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION
                ),
                reason=(
                    TransformationSelectionReason
                    .CONTINUOUS_RELAXATION_ONLY_REMAINING_ACTION
                ),
            )
        )

    terminal_reason = (
        termination
        if termination is not None
        else TerminationDecision(
            TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
        )
    )
    return TransformationPlanningDecision(termination=terminal_reason)


def _preparation_selection_reason(
    legal_family: LegalTransformationFamily,
) -> TransformationSelectionReason:
    """Return the explainable selection reason for one preparation-stage action."""

    if legal_family.preparation_reason is LocalPreparationReason.TOPOLOGY_PRECONDITION:
        return (
            TransformationSelectionReason
            .PREPARATION_REPAIRS_RELAXATION_TOPOLOGY_PRECONDITIONS
        )

    if legal_family.preparation_reason is LocalPreparationReason.CHEMISTRY_PREPARATION:
        return TransformationSelectionReason.CHEMISTRY_PREPARATION_REPAIRS_LOCAL_STATE

    if legal_family.preparation_reason is LocalPreparationReason.LOCAL_GEOMETRY:
        return TransformationSelectionReason.GEOMETRY_PREPARATION_REPAIRS_LOCAL_STATE

    raise ValueError("preparation-stage selection requires one preparation reason")
