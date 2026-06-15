"""Workflow completion legal-plan construction from whole-structure state."""

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.scope import WholeStructureScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import ComponentSupportState, StructureProjectionStateFacts
from protrepair.state.domain import AtomScopeStateFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
)
from protrepair.workflow.planning.completion.plan import (
    WorkflowCompletionPartition,
    WorkflowCompletionPartitionKind,
    WorkflowCompletionPlanSet,
)
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)


def workflow_completion_state(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
) -> tuple[StructureProjectionStateFacts, AtomScopeStateFacts]:
    """Return primitive whole-structure truth for workflow completion planning."""

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    workflow_snapshot = ProteinStructureSnapshot.from_structure(structure)
    workflow_atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        WholeStructureScope(),
        carrier=workflow_snapshot,
    )
    structure_facts = StructureProjectionStateFacts.from_structure(
        structure,
        component_library=active_component_library,
    )
    atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
        workflow_snapshot,
        workflow_atom_scope,
        component_library=active_component_library,
    )
    return structure_facts, atom_scope_facts


def workflow_legal_completion_plans(
    structure: ProteinStructure,
    *,
    requests_heavy_atom_completion: bool,
    requests_hydrogen_population: bool,
    component_library: ComponentLibrary | None = None,
) -> WorkflowCompletionPlanSet:
    """Return the legal staged workflow-completion plans for one structure."""

    structure_facts, _ = workflow_completion_state(
        structure,
        component_library=component_library,
    )
    structure_planning_signature = StructurePlanningSignature.from_facts(
        structure_facts
    )
    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    partitions = _workflow_completion_partitions(
        structure,
        requests_heavy_atom_completion=requests_heavy_atom_completion,
        requests_hydrogen_population=requests_hydrogen_population,
        component_library=active_component_library,
    )
    if not partitions:
        return WorkflowCompletionPlanSet(
            structure_planning_signature=structure_planning_signature,
            plans=(),
        )

    return WorkflowCompletionPlanSet.from_partition_sets(
        structure_planning_signature=structure_planning_signature,
        partition_sets=(partitions,),
    )


def _workflow_completion_partitions(
    structure: ProteinStructure,
    *,
    requests_heavy_atom_completion: bool,
    requests_hydrogen_population: bool,
    component_library: ComponentLibrary,
) -> tuple[WorkflowCompletionPartition, ...]:
    """Return canonical residue-subset partitions for one workflow structure."""

    residue_ids_by_kind: dict[WorkflowCompletionPartitionKind, list[ResidueId]] = {}
    partition_order: list[WorkflowCompletionPartitionKind] = []
    for residue in structure.constitution.iter_residues(include_ligands=False):
        partition_kind = _workflow_completion_partition_kind(
            residue,
            source_structure=structure,
            requests_heavy_atom_completion=requests_heavy_atom_completion,
            requests_hydrogen_population=requests_hydrogen_population,
            component_library=component_library,
        )
        if partition_kind is None:
            continue
        if partition_kind not in residue_ids_by_kind:
            residue_ids_by_kind[partition_kind] = []
            partition_order.append(partition_kind)
        residue_ids_by_kind[partition_kind].append(residue.residue_id)

    return tuple(
        WorkflowCompletionPartition(
            residue_ids=tuple(residue_ids_by_kind[partition_kind]),
            kind=partition_kind,
        )
        for partition_kind in partition_order
    )


def _workflow_completion_partition_kind(
    residue: ResidueSite,
    *,
    source_structure: ProteinStructure,
    requests_heavy_atom_completion: bool,
    requests_hydrogen_population: bool,
    component_library: ComponentLibrary,
) -> WorkflowCompletionPartitionKind | None:
    """Return canonical completion semantics for one observed residue."""

    residue_facts = StructureProjectionStateFacts.from_projection(
        context_structure=source_structure,
        residues=(residue,),
        component_library=component_library,
    )
    if (
        residue_facts.component_support_fact.value
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    ):
        return WorkflowCompletionPartitionKind.UNSUPPORTED_STOP
    if (
        requests_hydrogen_population
        and residue_facts.hydrogen_coverage_fact.value.needs_hydrogenation()
        and (
            residue_facts.backbone_heavy_atom_completeness_fact.value.requires_completion()
            or (
                residue_facts.sidechain_heavy_atom_completeness_fact.value.requires_completion()
            )
        )
    ):
        return WorkflowCompletionPartitionKind.HEAVY_THEN_HYDROGEN
    if (
        requests_hydrogen_population
        and residue_facts.hydrogen_coverage_fact.value.needs_hydrogenation()
    ):
        return WorkflowCompletionPartitionKind.HYDROGEN_ONLY

    if requests_heavy_atom_completion and (
        residue_facts.backbone_heavy_atom_completeness_fact.value.requires_completion()
        or (
            residue_facts.sidechain_heavy_atom_completeness_fact.value.requires_completion()
        )
    ):
        return WorkflowCompletionPartitionKind.HEAVY_ONLY

    return None
