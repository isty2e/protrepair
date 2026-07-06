"""Workflow-visible retained non-polymer hydrogen completion actions."""

from dataclasses import dataclass

from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import ResidueSetWorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerHydrogenCompletionTransformer(
    ResidueSetWorkflowStructureTransformer
):
    """Workflow-visible retained non-polymer hydrogen completion transformer."""

    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError(
                "retained non-polymer hydrogen completion requires "
                "a residue-set scope"
            )

    @classmethod
    def from_completion_scope(
        cls,
        scope: ResidueSetScope,
    ) -> "RetainedNonPolymerHydrogenCompletionTransformer":
        """Build one retained non-polymer hydrogen completion transformer."""

        return cls(scope=scope)

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one retained non-polymer hydrogen domain into its codomain."""

        del carrier
        stage_result = add_retained_non_polymer_hydrogens(
            projected_domain.state,
            component_library=context.component_library,
            target_residue_ids=self.covered_residue_ids(),
            chemistry_evidence=(
                context.retained_non_polymer_chemistry_evidence
            ),
            allow_retained_non_polymer_rdkit_fallback=(
                context.allow_retained_non_polymer_rdkit_fallback
            ),
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=stage_result.structure,
            repairs=stage_result.repairs,
            issues=stage_result.issues,
        )
