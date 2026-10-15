"""Planner action for retained chemistry with an already complete atom inventory."""

from dataclasses import dataclass

from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import ResidueSetWorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerTopologyResolutionTransformer(
    ResidueSetWorkflowStructureTransformer
):
    """Publish current ligand chemistry without changing atom payloads.

    Parameters
    ----------
    scope : ResidueSetScope
        Retained residues whose bond chemistry needs resolution.
    """

    scope: ResidueSetScope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidueSetScope):
            raise TypeError("retained topology resolution requires a residue-set scope")

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Revalidate chemistry and publish compatible bonds on the current carrier.

        Parameters
        ----------
        projected_domain : ProjectedDomainState[ProteinStructure]
            Current structure rather than a proposal-time chemistry graph.
        carrier : TransformationResult
            Existing workflow result, accumulated by the base transformer.
        context : TransformerExecutionContext
            Component, override and fallback policy for this execution.

        Returns
        -------
        ProjectedCodomainState[ProteinStructure]
            Topology edits and residue-local refusal diagnostics.
        """
        del carrier
        result = resolve_retained_non_polymer_topology(
            projected_domain.state,
            context.component_library,
            target_residue_ids=self.covered_residue_ids(),
            chemistry_evidence=context.retained_non_polymer_chemistry_evidence,
            allow_retained_non_polymer_rdkit_fallback=context.allow_retained_non_polymer_rdkit_fallback,
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=result.structure,
            repairs=result.repairs,
            issues=result.issues,
        )
