"""Explicit workflow-planning inputs separate from requests and state."""

from dataclasses import dataclass, field
from enum import Enum

from protrepair.workflow.contracts.span_policy import (
    ExternalSpanGapSelectionPolicy,
)


class WorkflowTargetIntent(str, Enum):
    """High-level workflow target that shapes planning policy."""

    INSPECTION = "inspection"
    DOCKING = "docking"
    MD_READY = "md_ready"


class WorkflowPlanningPhase(str, Enum):
    """Closed workflow phases used by planning and reporting surfaces."""

    COVERAGE = "coverage"
    CHEMISTRY_AUGMENTATION = "chemistry_augmentation"
    INTRINSIC_GEOMETRY_CORRECTION = "intrinsic_geometry_correction"
    INTERACTION_AWARE_CORRECTION = "interaction_aware_correction"


class WorkflowSpanDonorAvailability(str, Enum):
    """Availability of donor-backed missing-residue span reconstruction."""

    NONE = "none"
    AVAILABLE = "available"


class WorkflowLigandContextMode(str, Enum):
    """Whether planning should use present ligands as contextual burden."""

    IGNORE = "ignore"
    CONSIDER_IF_PRESENT = "consider_if_present"


class WorkflowBranchQualityAxis(str, Enum):
    """One tunable axis in workflow branch preference projection."""

    HARD_ERRORS = "hard_errors"
    PARSER_COMPATIBILITY = "parser_compatibility"
    INTRINSIC_CLASH = "intrinsic_clash"
    INTERACTION_CLASH = "interaction_clash"
    STEREOCHEMISTRY = "stereochemistry"
    SEVERE_GEOMETRY = "severe_geometry"
    REQUESTED_GOALS = "requested_goals"
    ISSUE_BURDEN = "issue_burden"
    SEARCH_DEPTH = "search_depth"


@dataclass(frozen=True, slots=True)
class WorkflowBranchPreferencePolicy:
    """Tunable axis ordering for workflow branch score projection."""

    axes: tuple[WorkflowBranchQualityAxis, ...] = (
        WorkflowBranchQualityAxis.HARD_ERRORS,
        WorkflowBranchQualityAxis.PARSER_COMPATIBILITY,
        WorkflowBranchQualityAxis.INTRINSIC_CLASH,
        WorkflowBranchQualityAxis.INTERACTION_CLASH,
        WorkflowBranchQualityAxis.STEREOCHEMISTRY,
        WorkflowBranchQualityAxis.SEVERE_GEOMETRY,
        WorkflowBranchQualityAxis.REQUESTED_GOALS,
        WorkflowBranchQualityAxis.ISSUE_BURDEN,
        WorkflowBranchQualityAxis.SEARCH_DEPTH,
    )

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes:
            raise ValueError("workflow branch preference policy requires axes")
        for axis in axes:
            if not isinstance(axis, WorkflowBranchQualityAxis):
                raise TypeError(
                    "workflow branch preference policy axes must be "
                    "WorkflowBranchQualityAxis values"
                )
        if len(set(axes)) != len(axes):
            raise ValueError(
                "workflow branch preference policy must not repeat axes"
            )
        object.__setattr__(self, "axes", axes)


@dataclass(frozen=True, slots=True)
class WorkflowPlanningContext:
    """Explicit planning inputs that should not be inferred piecemeal."""

    ligand_context_mode: WorkflowLigandContextMode = (
        WorkflowLigandContextMode.IGNORE
    )
    target_intent: WorkflowTargetIntent = WorkflowTargetIntent.INSPECTION
    span_donor_availability: WorkflowSpanDonorAvailability = (
        WorkflowSpanDonorAvailability.NONE
    )
    external_span_gap_selection_policy: ExternalSpanGapSelectionPolicy = field(
        default_factory=ExternalSpanGapSelectionPolicy.internal_only
    )
    branch_preference_policy: WorkflowBranchPreferencePolicy = field(
        default_factory=WorkflowBranchPreferencePolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.ligand_context_mode, WorkflowLigandContextMode):
            raise TypeError(
                "workflow planning contexts require a "
                "WorkflowLigandContextMode value"
            )
        if not isinstance(self.target_intent, WorkflowTargetIntent):
            raise TypeError(
                "workflow planning contexts require a WorkflowTargetIntent value"
            )
        if not isinstance(
            self.span_donor_availability,
            WorkflowSpanDonorAvailability,
        ):
            raise TypeError(
                "workflow planning contexts require a "
                "WorkflowSpanDonorAvailability value"
            )
        if not isinstance(
            self.external_span_gap_selection_policy,
            ExternalSpanGapSelectionPolicy,
        ):
            raise TypeError(
                "workflow planning contexts require an "
                "ExternalSpanGapSelectionPolicy value"
            )
        if not isinstance(
            self.branch_preference_policy,
            WorkflowBranchPreferencePolicy,
        ):
            raise TypeError(
                "workflow planning contexts require a "
                "WorkflowBranchPreferencePolicy value"
            )

    def considers_ligand_context(self) -> bool:
        """Return whether planning may use present ligands as interaction context."""

        return (
            self.ligand_context_mode
            is WorkflowLigandContextMode.CONSIDER_IF_PRESENT
        )

    def allows_span_reconstruction(self) -> bool:
        """Return whether donor-backed span reconstruction is available."""

        return self.span_donor_availability is WorkflowSpanDonorAvailability.AVAILABLE
