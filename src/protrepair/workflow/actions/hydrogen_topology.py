"""Planner action for preserved polymer H with incomplete topology."""

from dataclasses import dataclass

from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.completion.hydrogen.topology import (
    resolve_fixed_hydrogen_topology,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import ResidueSetWorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class PolymerHydrogenTopologyResolutionTransformer(
    ResidueSetWorkflowStructureTransformer
):
    """Resolve fixed-H bonds without invoking coordinate placement.

    Parameters
    ----------
    scope : ResidueSetScope
        Polymer residues whose present H attachments need materialization.
    """

    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError("hydrogen topology resolution requires a residue-set scope")

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Revalidate current chemistry and materialize supported H attachments.

        Parameters
        ----------
        projected_domain : ProjectedDomainState[ProteinStructure]
            Current structure, not a cached proposal-time graph.
        carrier : TransformationResult
            Existing workflow result; accumulated events are lifted by the base.
        context : TransformerExecutionContext
            Active component library.

        Returns
        -------
        ProjectedCodomainState[ProteinStructure]
            Scoped topology changes, with contradictory residues left untouched.
        """
        del carrier
        result = resolve_fixed_hydrogen_topology(
            projected_domain.state,
            context.component_library,
            target_residue_ids=self.covered_residue_ids(),
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=result.structure,
            repairs=result.repairs,
            issues=result.issues,
        )
