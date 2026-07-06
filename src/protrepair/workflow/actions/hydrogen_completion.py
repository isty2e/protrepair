"""Planner-visible hydrogen-completion transformer invocations."""

from dataclasses import dataclass

from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import ResidueSetWorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class HydrogenCompletionTransformer(ResidueSetWorkflowStructureTransformer):
    """Workflow-visible hydrogen completion transformer."""

    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError(
                "hydrogen completion transformers require a residue-set scope"
            )

    @classmethod
    def from_completion_scope(
        cls,
        scope: ResidueSetScope,
    ) -> "HydrogenCompletionTransformer":
        """Build one hydrogen-completion transformer from a residue-set scope."""

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
        """Transform one hydrogen-completion projected domain into its codomain."""

        del carrier
        stage_result = add_hydrogens(
            projected_domain.state,
            component_library=context.component_library,
            reference_structure=context.reference_structure,
            prepare_heavy_atoms=False,
            target_residue_ids=self.covered_residue_ids(),
            orphan_fragment_policy=context.orphan_fragment_policy,
            histidine_protonation=context.histidine_protonation,
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
