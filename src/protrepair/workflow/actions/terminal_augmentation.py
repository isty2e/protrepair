"""Planner-visible terminal-augmentation transformer invocations."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.scope import ResidueBoundaryScope, ResidueBoundarySide, ResidueSetScope
from protrepair.state import (
    BoundaryAuthenticityState,
    OxtPresenceState,
    StructureBoundaryStateFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.completion.heavy.policy import should_add_terminal_oxt
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import ResidueSetWorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
)


@dataclass(frozen=True, slots=True)
class TerminalAugmentationTransformer(
    ResidueSetWorkflowStructureTransformer
):
    """Workflow-visible terminal OXT augmentation transformer."""

    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError(
                "terminal augmentation transformers require a residue-set scope"
            )

    @classmethod
    def planned_candidate(
        cls,
        structure: ProteinStructure,
        *,
        boundary_facts: StructureBoundaryStateFacts,
        requested_goals: RequestedGoalSet,
        component_library: ComponentLibrary,
    ) -> "TerminalAugmentationTransformer | None":
        """Return one terminal-augmentation candidate when one is warranted."""

        boundary_goals = requested_goals.boundary_goals(
            OxtPresenceState,
            side=ResidueBoundarySide.C_TERMINUS,
        )
        if not boundary_goals:
            return None

        target_residue_ids: list[ResidueId] = []
        for goal in boundary_goals:
            requested_scope = goal.scope
            if not isinstance(requested_scope, ResidueBoundaryScope):
                continue

            authenticity = (
                boundary_facts.terminal_boundary_observation.authenticity_for_scope(
                    requested_scope
                )
            )
            if authenticity is not BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT:
                continue

            residue_id = requested_scope.residue_id
            residue_site = structure.constitution.residue_or_ligand(residue_id)
            if residue_site is None:
                continue
            if residue_site.has_atom_site("OXT"):
                continue
            if should_add_terminal_oxt(
                residue_site,
                component_library=component_library,
            ):
                target_residue_ids.append(residue_id)

        if not target_residue_ids:
            return None

        return cls(
            scope=ResidueSetScope(
                residue_ids=tuple(dict.fromkeys(target_residue_ids))
            )
        )


    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one terminal-augmentation domain into its codomain."""

        del carrier
        terminal_result = augment_c_terminal_oxt(
            projected_domain.state,
            component_library=context.component_library,
            target_residue_ids=self.covered_residue_ids(),
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=terminal_result.structure,
            repairs=terminal_result.repairs,
            issues=terminal_result.issues,
        )
