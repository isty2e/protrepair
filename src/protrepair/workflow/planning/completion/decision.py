"""Workflow completion plan selection decisions."""

from dataclasses import dataclass
from enum import Enum

from protrepair.workflow.planning.completion.plan import (
    WorkflowCompletionPartitionKind,
    WorkflowCompletionPlan,
    WorkflowCompletionPlanSet,
)


class WorkflowCompletionSelectionReason(str, Enum):
    """Closed explanations for staged workflow-completion plan choice."""

    HETEROGENEOUS_SUBSETS_REQUIRE_PARTITIONED_COMPLETION = (
        "heterogeneous_subsets_require_partitioned_completion"
    )
    REQUESTED_HYDROGEN_POPULATION_REQUIRES_HYDROGEN_CONTINUATION = (
        "requested_hydrogen_population_requires_hydrogen_continuation"
    )
    REQUESTED_HYDROGEN_POPULATION_REQUIRES_HEAVY_AND_HYDROGEN_STAGES = (
        "requested_hydrogen_population_requires_heavy_and_hydrogen_stages"
    )
    REQUESTED_HEAVY_COMPLETION_REQUIRES_HEAVY_REPAIR = (
        "requested_heavy_completion_requires_heavy_repair"
    )


@dataclass(frozen=True, slots=True)
class WorkflowCompletionDecision:
    """One explainable staged workflow-completion decision."""

    plan: WorkflowCompletionPlan
    reason: WorkflowCompletionSelectionReason


def choose_workflow_completion_plan(
    legal_plans: WorkflowCompletionPlanSet,
) -> WorkflowCompletionDecision:
    """Choose one explainable staged completion plan from legal plans."""

    if legal_plans.is_empty():
        raise RuntimeError(
            "workflow completion required at least one legal staged plan"
        )

    plan = legal_plans.plans[0]
    if plan.is_heterogeneous():
        return WorkflowCompletionDecision(
            plan=plan,
            reason=(
                WorkflowCompletionSelectionReason.HETEROGENEOUS_SUBSETS_REQUIRE_PARTITIONED_COMPLETION
            ),
        )

    partition_kinds = plan.partition_kinds()
    if partition_kinds == (WorkflowCompletionPartitionKind.HYDROGEN_ONLY,):
        return WorkflowCompletionDecision(
            plan=plan,
            reason=(
                WorkflowCompletionSelectionReason.REQUESTED_HYDROGEN_POPULATION_REQUIRES_HYDROGEN_CONTINUATION
            ),
        )
    if partition_kinds == (WorkflowCompletionPartitionKind.HEAVY_THEN_HYDROGEN,):
        return WorkflowCompletionDecision(
            plan=plan,
            reason=(
                WorkflowCompletionSelectionReason.REQUESTED_HYDROGEN_POPULATION_REQUIRES_HEAVY_AND_HYDROGEN_STAGES
            ),
        )

    return WorkflowCompletionDecision(
        plan=plan,
        reason=(
            WorkflowCompletionSelectionReason.REQUESTED_HEAVY_COMPLETION_REQUIRES_HEAVY_REPAIR
        ),
    )
