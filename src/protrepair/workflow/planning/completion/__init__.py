"""Workflow completion planning contracts and legal-plan construction."""

from protrepair.workflow.planning.completion.decision import (
    WorkflowCompletionDecision,
    WorkflowCompletionSelectionReason,
    choose_workflow_completion_plan,
)
from protrepair.workflow.planning.completion.plan import (
    WORKFLOW_COMPLETION_STAGE_PRECEDENCE,
    WorkflowCompletionPartition,
    WorkflowCompletionPartitionKind,
    WorkflowCompletionPlan,
    WorkflowCompletionPlanSet,
    WorkflowCompletionStageKind,
    WorkflowExecutionStage,
)
from protrepair.workflow.planning.completion.planning import (
    workflow_completion_state,
    workflow_legal_completion_plans,
)
from protrepair.workflow.planning.completion.scope import (
    WorkflowAbsentResidueSpanExecutionScope,
    WorkflowAnchorAtomPairExecutionScope,
    WorkflowCompositeExecutionScope,
    WorkflowExecutionScope,
    WorkflowExecutionScopeKind,
    WorkflowResidueSetExecutionScope,
)

__all__ = [
    "WORKFLOW_COMPLETION_STAGE_PRECEDENCE",
    "WorkflowAbsentResidueSpanExecutionScope",
    "WorkflowAnchorAtomPairExecutionScope",
    "WorkflowCompletionDecision",
    "WorkflowCompletionPartition",
    "WorkflowCompletionPartitionKind",
    "WorkflowCompletionPlan",
    "WorkflowCompletionPlanSet",
    "WorkflowCompletionSelectionReason",
    "WorkflowCompletionStageKind",
    "WorkflowCompositeExecutionScope",
    "WorkflowExecutionScope",
    "WorkflowExecutionScopeKind",
    "WorkflowExecutionStage",
    "WorkflowResidueSetExecutionScope",
    "choose_workflow_completion_plan",
    "workflow_completion_state",
    "workflow_legal_completion_plans",
]
