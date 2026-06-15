"""Planner-visible stereochemistry-correction transformer invocations."""

from dataclasses import dataclass

from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.completion.stereochemistry.correction import (
    correct_sidechain_stereochemistry,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import ResidueSetWorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class StereochemistryCorrectionTransformer(
    ResidueSetWorkflowStructureTransformer
):
    """Workflow-visible side-chain stereochemistry correction transformer."""

    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError(
                "stereochemistry correction transformers require a residue-set scope"
            )

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one stereo-correction projected domain into its codomain."""

        del carrier
        stage_result = correct_sidechain_stereochemistry(
            projected_domain.state,
            component_library=context.component_library,
            target_residue_ids=self.covered_residue_ids(),
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=stage_result.structure,
            repairs=stage_result.repairs,
            issues=stage_result.issues,
        )
