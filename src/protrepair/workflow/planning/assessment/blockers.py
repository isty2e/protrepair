"""Workflow blocker values over semantic scopes."""

from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.component_support import (
    diagnose_component_support,
    missing_component_definition_issue,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.workflow.contracts.planning import WorkflowPlanningPhase
from protrepair.workflow.planning.capability import WorkflowCapabilityDeficitFamily


class WorkflowBlockerKind(str, Enum):
    """Closed non-executable blocker kinds over workflow scopes."""

    UNSUPPORTED_COMPONENT = "unsupported_component"


@dataclass(frozen=True, slots=True)
class WorkflowBlocker:
    """One workflow blocker projected over a semantic scope."""

    phase: WorkflowPlanningPhase
    deficit_family: WorkflowCapabilityDeficitFamily
    kind: WorkflowBlockerKind
    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.phase, WorkflowPlanningPhase):
            raise TypeError(
                "workflow blockers require a WorkflowPlanningPhase value"
            )
        if not isinstance(
            self.deficit_family,
            WorkflowCapabilityDeficitFamily,
        ):
            raise TypeError(
                "workflow blockers require a WorkflowCapabilityDeficitFamily value"
            )
        if not isinstance(self.kind, WorkflowBlockerKind):
            raise TypeError("workflow blockers require a WorkflowBlockerKind value")
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError("workflow blockers currently require a residue-set scope")
        if (
            self.kind is WorkflowBlockerKind.UNSUPPORTED_COMPONENT
            and len(self.scope.residue_ids) != 1
        ):
            raise ValueError(
                "unsupported-component blockers currently require exactly one "
                "blocked residue"
            )

    @classmethod
    def unsupported_component(
        cls,
        *,
        deficit_family: WorkflowCapabilityDeficitFamily,
        scope: ResidueSetScope,
    ) -> "WorkflowBlocker":
        """Return one unsupported-component blocker for a specific deficit family."""

        return cls(
            phase=workflow_planning_phase_hint(deficit_family),
            deficit_family=deficit_family,
            kind=WorkflowBlockerKind.UNSUPPORTED_COMPONENT,
            scope=scope,
        )

    def residue_ids(self) -> frozenset[ResidueId]:
        """Return residue ids directly covered by this blocker scope."""

        return frozenset(self.scope.residue_ids)

    def issue_for(
        self,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
    ) -> ValidationIssue:
        """Return one workflow issue projected from this blocker."""

        if self.kind is not WorkflowBlockerKind.UNSUPPORTED_COMPONENT:
            raise NotImplementedError(
                "workflow blocker issue projection currently supports only "
                "unsupported-component blockers"
        )

        residue_ids = self.residue_ids()
        residue_id = next(iter(residue_ids))
        residue = structure.constitution.residue_site_at(
            structure.constitution.residue_index(residue_id)
        )
        diagnosis = diagnose_component_support(
            residue.component_id,
            component_library=component_library,
        )
        return missing_component_definition_issue(
            residue,
            diagnosis=diagnosis,
            action="leaving residue unchanged",
        )


def workflow_planning_phase_hint(
    deficit_family: WorkflowCapabilityDeficitFamily,
) -> WorkflowPlanningPhase:
    """Return one reporting-phase hint for a blocker's originating deficit family."""

    if deficit_family in (
        WorkflowCapabilityDeficitFamily.COVERAGE_GAP,
        WorkflowCapabilityDeficitFamily.ATOM_COVERAGE,
    ):
        return WorkflowPlanningPhase.COVERAGE
    if deficit_family in (
        WorkflowCapabilityDeficitFamily.CHEMISTRY_CONTRADICTION,
        WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
    ):
        return WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION
    if deficit_family is WorkflowCapabilityDeficitFamily.BACKBONE_WINDOW_OPERATOR:
        return WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION
    if deficit_family is WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY:
        return WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION
    if deficit_family is WorkflowCapabilityDeficitFamily.PARSER_COMPATIBILITY:
        return WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION
    if deficit_family is WorkflowCapabilityDeficitFamily.INTERACTION:
        return WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION

    raise NotImplementedError(
        "workflow blocker phase hint requires one covered deficit family"
    )
