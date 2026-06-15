"""Workflow-visible structure-transformer action contracts."""

from abc import abstractmethod

from protrepair.scope import ResidueSetScope, Scope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
    ProteinTransformer,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.contracts.request import WorkflowGoalStateValue


class WorkflowStructureTransformer(
    ProteinTransformer[
        TransformerExecutionContext,
        TransformationResult,
        ProteinStructure,
        ProteinStructure,
        WorkflowGoalStateValue,
    ]
):
    """Workflow action that transforms the current structure result carrier."""

    @property
    @abstractmethod
    def workflow_scope(self) -> Scope:
        """Return the semantic scope this workflow action transforms."""

    def project_domain_state(
        self,
        carrier: TransformationResult,
        *,
        context: TransformerExecutionContext,
    ) -> ProjectedDomainState[ProteinStructure]:
        """Project the current workflow result into this action's structure domain."""

        del context
        return ProjectedDomainState(
            scope=self.workflow_scope,
            state=carrier.structure,
        )

    def lift_projected_codomain(
        self,
        carrier: TransformationResult,
        projected_codomain: ProjectedCodomainState[ProteinStructure],
        *,
        context: TransformerExecutionContext,
    ) -> TransformationResult:
        """Lift one structure codomain back into the workflow result carrier."""

        del context
        return TransformationResult(
            structure=projected_codomain.state,
            repairs=carrier.repairs + projected_codomain.repairs,
            issues=carrier.issues + projected_codomain.issues,
        )


class ResidueSetWorkflowStructureTransformer(WorkflowStructureTransformer):
    """Workflow structure action whose semantic scope is a residue set."""

    scope: ResidueSetScope

    @property
    def workflow_scope(self) -> ResidueSetScope:
        """Return the residue-set scope this workflow action transforms."""

        return self.scope

    def covered_residue_ids(self) -> frozenset[ResidueId]:
        """Return residue ids directly covered by this transformer's scope."""

        return frozenset(self.workflow_scope.residue_ids)
