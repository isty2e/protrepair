"""Planner-visible backbone-window refinement transformer."""

from dataclasses import dataclass, field

from protrepair.errors import RefinementError
from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.refinement.backbone_window import (
    execute_backbone_window_refinement,
)
from protrepair.transformer.refinement.outcome import (
    RefinementStage,
    RefinementStageOutcome,
)
from protrepair.transformer.refinement.spec import BackboneWindowRefinementSpec
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


def _default_backbone_window_settings() -> ContinuousRelaxationSettings:
    """Return the current backbone-window continuous-relaxation profile."""

    return ContinuousRelaxationSettings(
        profile=ContinuousRelaxationProfile.RDKIT_UFF,
        max_iterations=250,
    )


@dataclass(frozen=True, slots=True)
class BackboneWindowRefinementTransformer(WorkflowStructureTransformer):
    """Workflow-visible backbone-window refinement transformer."""

    window_spec: BackboneWindowRefinementSpec
    settings: ContinuousRelaxationSettings = field(
        default_factory=_default_backbone_window_settings
    )

    def __post_init__(self) -> None:
        if not isinstance(self.window_spec, BackboneWindowRefinementSpec):
            raise TypeError(
                "backbone-window refinement transformers require a "
                "BackboneWindowRefinementSpec"
            )
        if not isinstance(self.settings, ContinuousRelaxationSettings):
            raise TypeError(
                "backbone-window refinement transformers require "
                "ContinuousRelaxationSettings"
            )

    @property
    def scope(self) -> ResidueSetScope:
        """Return the semantic residue-window scope."""

        return self.window_spec.as_residue_scope()

    @property
    def workflow_scope(self) -> ResidueSetScope:
        """Return the backbone-window scope this action transforms."""

        return self.scope

    @classmethod
    def from_window_spec(
        cls,
        window_spec: BackboneWindowRefinementSpec,
    ) -> "BackboneWindowRefinementTransformer":
        """Build one workflow transformer from a backbone-window contract."""

        return cls(window_spec=window_spec)

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one backbone-window domain into its refined codomain."""

        del carrier
        try:
            refinement_result = execute_backbone_window_refinement(
                projected_domain.state,
                self.window_spec,
                spec=self.settings,
                component_library=context.component_library,
            )
        except (RefinementError, ValueError) as error:
            outcome = RefinementStageOutcome.rejected(
                projected_domain.state,
                stage=RefinementStage.REPAIR,
                residue_id=None,
                message=str(error),
            )
        else:
            outcome = RefinementStageOutcome.from_result(
                refinement_result,
                stage=RefinementStage.REPAIR,
            )

        return ProjectedCodomainState(
            scope=self.scope,
            state=outcome.structure,
            repairs=outcome.repairs,
            issues=outcome.issues,
        )
