"""Planner-visible heavy-atom completion transformer invocations."""

from dataclasses import dataclass

from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import ResidueSetWorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class HeavyAtomCompletionTransformer(
    ResidueSetWorkflowStructureTransformer
):
    """Workflow-visible heavy-atom completion transformer."""

    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError(
                "heavy-atom completion transformers require a residue-set scope"
            )

    @classmethod
    def from_completion_scope(
        cls,
        scope: ResidueSetScope,
    ) -> "HeavyAtomCompletionTransformer":
        """Build one heavy-completion transformer from a residue-set scope."""

        return cls(scope=scope)

    def is_completion_transformer(self) -> bool:
        """Return whether this transformer belongs to the completion family."""

        return True

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one heavy-completion projected domain into its codomain."""

        del carrier
        stage_result = repair_heavy_atoms_core(
            projected_domain.state,
            component_library=context.component_library,
            reference_structure=context.reference_structure,
            augment_c_terminal_oxt=False,
            target_residue_ids=self.covered_residue_ids(),
            orphan_fragment_policy=context.orphan_fragment_policy,
            local_refinement=None,
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=stage_result.structure.with_ligand_facets_from(
                context.original_structure
            ),
            repairs=stage_result.repairs,
            issues=stage_result.issues,
        )
