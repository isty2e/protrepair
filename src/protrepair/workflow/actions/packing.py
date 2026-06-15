"""Planner-visible committed packing transformer invocations."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.scope import ResidueSetScope, Scope, WholeStructureScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.packing.models import PackingSpec
from protrepair.transformer.packing.planning import planned_committed_packing_spec
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.contracts.request import (
    WorkflowTransformRequests,
)
from protrepair.workflow.engine.packing.committed import (
    execute_committed_workflow_packing,
)


@dataclass(frozen=True, slots=True)
class CommittedPackingTransformer(WorkflowStructureTransformer):
    """Workflow-visible committed side-chain packing transformer."""

    scope: WholeStructureScope | ResidueSetScope
    packing_spec: PackingSpec

    @property
    def workflow_scope(self) -> WholeStructureScope | ResidueSetScope:
        """Return the committed-packing scope this action transforms."""

        return self.scope

    def __post_init__(self) -> None:
        if not isinstance(
            self.scope,
            (WholeStructureScope, ResidueSetScope),
        ):
            raise TypeError(
                "committed packing transformers require a whole-structure or "
                "residue-set scope"
            )
        if not isinstance(self.packing_spec, PackingSpec):
            raise TypeError(
                "committed packing transformers require a PackingSpec payload"
            )

    @classmethod
    def from_planned_spec(
        cls,
        packing_spec: PackingSpec,
    ) -> "CommittedPackingTransformer":
        """Build one committed-packing transformer from a planned packing spec."""

        if packing_spec.mutable_residue_ids is None:
            scope: Scope = WholeStructureScope()
        else:
            scope = ResidueSetScope(
                residue_ids=packing_spec.mutable_residue_ids
            )

        return cls(scope=scope, packing_spec=packing_spec)

    @classmethod
    def planned_candidate(
        cls,
        structure: ProteinStructure,
        *,
        transform_requests: WorkflowTransformRequests,
        component_library: ComponentLibrary,
    ) -> "CommittedPackingTransformer | None":
        """Return one committed-packing candidate when the request is legal."""

        if transform_requests.committed_sidechain_packing is None:
            return None

        planned_spec = planned_committed_packing_spec(
            structure,
            transform_requests.committed_sidechain_packing,
            component_library=component_library,
        )
        if planned_spec is None:
            return None

        return cls.from_planned_spec(planned_spec)

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one packing projected domain into its codomain."""

        del context, carrier
        packing_result = execute_committed_workflow_packing(
            projected_domain.state,
            self.packing_spec,
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=packing_result.packed_structure.with_ligand_facets_from(
                projected_domain.state
            ),
            issues=packing_result.issues,
        )
