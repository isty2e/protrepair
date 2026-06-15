"""Workflow completion plan ontology."""

from dataclasses import dataclass
from enum import Enum

from protrepair.structure.labels import ResidueId
from protrepair.workflow.planning.completion.scope import (
    WorkflowAbsentResidueSpanExecutionScope,
    WorkflowAnchorAtomPairExecutionScope,
    WorkflowCompositeExecutionScope,
    WorkflowExecutionScope,
    WorkflowResidueSetExecutionScope,
)
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)


class WorkflowCompletionStageKind(str, Enum):
    """Closed workflow-completion stages executed over one whole structure."""

    HEAVY_ATOM_REPAIR = "heavy_atom_repair"
    HYDROGEN_COMPLETION = "hydrogen_completion"


WORKFLOW_COMPLETION_STAGE_PRECEDENCE: tuple[WorkflowCompletionStageKind, ...] = (
    WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,
    WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
)


class WorkflowCompletionPartitionKind(str, Enum):
    """Closed completion semantics for one heterogeneous residue subset."""

    HEAVY_ONLY = "heavy_only"
    HYDROGEN_ONLY = "hydrogen_only"
    HEAVY_THEN_HYDROGEN = "heavy_then_hydrogen"
    UNSUPPORTED_STOP = "unsupported_stop"

    def semantic_stages(self) -> tuple[WorkflowCompletionStageKind, ...]:
        """Return the semantic stage sequence for one subset kind."""

        if self is WorkflowCompletionPartitionKind.HEAVY_ONLY:
            return (WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,)
        if self is WorkflowCompletionPartitionKind.HYDROGEN_ONLY:
            return (WorkflowCompletionStageKind.HYDROGEN_COMPLETION,)
        if self is WorkflowCompletionPartitionKind.HEAVY_THEN_HYDROGEN:
            return (
                WorkflowCompletionStageKind.HEAVY_ATOM_REPAIR,
                WorkflowCompletionStageKind.HYDROGEN_COMPLETION,
            )
        return ()

    def is_terminal_stop(self) -> bool:
        """Return whether the subset semantics represent no completion route."""

        return self is WorkflowCompletionPartitionKind.UNSUPPORTED_STOP


@dataclass(frozen=True, slots=True)
class WorkflowCompletionPartition:
    """One disjoint residue subset plus its canonical completion semantics."""

    residue_ids: tuple[ResidueId, ...]
    kind: WorkflowCompletionPartitionKind

    def __post_init__(self) -> None:
        residue_ids = tuple(dict.fromkeys(self.residue_ids))
        if not residue_ids:
            raise ValueError(
                "workflow completion partitions require at least one residue"
            )
        if not isinstance(self.kind, WorkflowCompletionPartitionKind):
            raise TypeError(
                "workflow completion partitions require a "
                "WorkflowCompletionPartitionKind value"
            )

        object.__setattr__(self, "residue_ids", residue_ids)

    def contains_residue(self, residue_id: ResidueId) -> bool:
        """Return whether one residue belongs to this completion partition."""

        return residue_id in self.residue_ids


@dataclass(frozen=True, slots=True)
class WorkflowExecutionStage:
    """One executable workflow stage plus its canonical execution scope."""

    kind: WorkflowCompletionStageKind
    scope: WorkflowExecutionScope

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WorkflowCompletionStageKind):
            raise TypeError(
                "workflow execution stages require a WorkflowCompletionStageKind value"
            )
        if not isinstance(
            self.scope,
            (
                WorkflowResidueSetExecutionScope,
                WorkflowAbsentResidueSpanExecutionScope,
                WorkflowAnchorAtomPairExecutionScope,
                WorkflowCompositeExecutionScope,
            ),
        ):
            raise TypeError(
                "workflow execution stages require one canonical "
                "WorkflowExecutionScope value"
            )


@dataclass(frozen=True, slots=True)
class WorkflowCompletionPlan:
    """One legal staged completion plan for a whole-workflow repair run."""

    structure_planning_signature: StructurePlanningSignature
    partitions: tuple[WorkflowCompletionPartition, ...]

    def __post_init__(self) -> None:
        if not self.partitions:
            raise ValueError("workflow completion plans require at least one partition")
        residue_ids = tuple(
            residue_id
            for partition in self.partitions
            for residue_id in partition.residue_ids
        )
        if len(set(residue_ids)) != len(residue_ids):
            raise ValueError(
                "workflow completion plans must not repeat residue ids across "
                "partitions"
            )

    def contains_stage(self, stage: WorkflowCompletionStageKind) -> bool:
        """Return whether the plan executes one specific completion stage."""

        return any(
            execution_stage.kind is stage for execution_stage in self.execution_plan()
        )

    def execution_plan(self) -> tuple[WorkflowExecutionStage, ...]:
        """Return executable workflow stages with canonical residue targets."""

        execution_plan: list[WorkflowExecutionStage] = []
        for stage in WORKFLOW_COMPLETION_STAGE_PRECEDENCE:
            residue_ids = tuple(
                residue_id
                for partition in self.partitions
                if stage in partition.kind.semantic_stages()
                for residue_id in partition.residue_ids
            )
            if residue_ids:
                execution_plan.append(
                    WorkflowExecutionStage(
                        kind=stage,
                        scope=WorkflowResidueSetExecutionScope(residue_ids=residue_ids),
                    )
                )

        return tuple(execution_plan)

    def execution_stages(self) -> tuple[WorkflowCompletionStageKind, ...]:
        """Return the ordered stage union needed by the current workflow spine."""

        return tuple(stage.kind for stage in self.execution_plan())

    def residue_ids_for_stage(
        self,
        stage: WorkflowCompletionStageKind,
    ) -> tuple[ResidueId, ...]:
        """Return residue ids for one currently residue-set-backed stage."""

        for execution_stage in self.execution_plan():
            if execution_stage.kind is stage:
                if isinstance(
                    execution_stage.scope,
                    WorkflowResidueSetExecutionScope,
                ):
                    return execution_stage.scope.residue_ids

                raise RuntimeError(
                    "workflow completion plan stage does not target an existing "
                    "residue-set scope"
                )

        return ()

    def partition_kinds(self) -> tuple[WorkflowCompletionPartitionKind, ...]:
        """Return the canonical subset semantics represented in this plan."""

        return tuple(partition.kind for partition in self.partitions)

    def is_heterogeneous(self) -> bool:
        """Return whether the plan spans more than one subset semantics."""

        return len(set(self.partition_kinds())) > 1

    def partitions_of_kind(
        self,
        kind: WorkflowCompletionPartitionKind,
    ) -> tuple[WorkflowCompletionPartition, ...]:
        """Return partitions that carry one specific subset semantic."""

        return tuple(
            partition for partition in self.partitions if partition.kind is kind
        )

    def unsupported_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residues blocked by unsupported-stop completion semantics."""

        return tuple(
            residue_id
            for partition in self.partitions_of_kind(
                WorkflowCompletionPartitionKind.UNSUPPORTED_STOP
            )
            for residue_id in partition.residue_ids
        )


@dataclass(frozen=True, slots=True)
class WorkflowCompletionPlanSet:
    """Legal staged workflow-completion plans under one structure planning signature."""

    structure_planning_signature: StructurePlanningSignature
    plans: tuple[WorkflowCompletionPlan, ...]

    @classmethod
    def from_partition_sets(
        cls,
        *,
        structure_planning_signature: StructurePlanningSignature,
        partition_sets: tuple[tuple[WorkflowCompletionPartition, ...], ...],
    ) -> "WorkflowCompletionPlanSet":
        """Build one legal plan set from canonical partition sets."""

        return cls(
            structure_planning_signature=structure_planning_signature,
            plans=tuple(
                WorkflowCompletionPlan(
                    structure_planning_signature=structure_planning_signature,
                    partitions=partitions,
                )
                for partitions in partition_sets
            ),
        )

    def is_empty(self) -> bool:
        """Return whether no staged completion plan is legal."""

        return not self.plans

    def plan_for_execution_stages(
        self,
        stages: tuple[WorkflowCompletionStageKind, ...],
    ) -> WorkflowCompletionPlan:
        """Return the legal plan matching one exact ordered execution sequence."""

        for plan in self.plans:
            if plan.execution_stages() == stages:
                return plan

        stage_tokens = ", ".join(stage.value for stage in stages)
        raise KeyError(
            "workflow completion plan with stages "
            f"({stage_tokens}) is not present in the current legal set"
        )
