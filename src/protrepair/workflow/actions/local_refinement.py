"""Planner-visible local-refinement transformer invocations."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.repair_stage import (
    apply_repair_stage_local_refinement,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class LocalRefinementTransformer(WorkflowStructureTransformer):
    """Workflow-visible local-refinement transformer."""

    scope: ResidueSetScope | AtomSetScope
    repair_refinement: RepairRefinementSpec

    @property
    def workflow_scope(self) -> ResidueSetScope | AtomSetScope:
        """Return the local-refinement scope this action transforms."""

        return self.scope

    def __post_init__(self) -> None:
        if not isinstance(
            self.scope,
            (ResidueSetScope, AtomSetScope),
        ):
            raise TypeError(
                "local refinement transformers require a residue-set or atom-set "
                "scope"
            )
        if not isinstance(self.repair_refinement, RepairRefinementSpec):
            raise TypeError(
                "local refinement transformers require a RepairRefinementSpec payload"
            )

    @classmethod
    def from_repair_refinement(
        cls,
        repair_refinement: RepairRefinementSpec,
    ) -> "LocalRefinementTransformer":
        """Build one local-refinement transformer from a repair request."""

        refinement_scope = repair_refinement.scope_spec.scope
        if isinstance(refinement_scope, ResidueSetScope):
            scope: ResidueSetScope | AtomSetScope = refinement_scope
        elif isinstance(refinement_scope, AtomSetScope):
            scope = refinement_scope
        else:
            raise NotImplementedError(
                "workflow local refinement currently requires a residue-set or "
                "atom-set scope"
            )

        return cls(
            scope=scope,
            repair_refinement=repair_refinement,
        )

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one local-refinement projected domain into its codomain."""

        local_refinement = materialize_workflow_local_refinement(
            ProteinStructureSnapshot.from_structure(projected_domain.state),
            self.repair_refinement,
            component_library=context.component_library,
        )
        stage_result = apply_repair_stage_local_refinement(
            carrier,
            local_refinement=local_refinement,
            component_library=context.component_library,
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=stage_result.structure,
            repairs=stage_result.repairs[len(carrier.repairs) :],
            issues=stage_result.issues[len(carrier.issues) :],
        )


def materialize_workflow_local_refinement(
    snapshot: ProteinStructureSnapshot,
    repair_refinement: RepairRefinementSpec,
    *,
    component_library: ComponentLibrary,
) -> RepairLocalRefinementDirective:
    """Return one canonical workflow-local refinement directive."""

    del snapshot
    del component_library
    return RepairLocalRefinementDirective(
        scope_spec=repair_refinement.scope_spec,
        execution_scope_spec=repair_refinement.execution_scope_spec,
        config=repair_refinement.config,
        binding=repair_refinement.binding,
    )
