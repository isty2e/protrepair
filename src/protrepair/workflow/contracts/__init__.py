"""Workflow contracts for requests, results, and policy enums."""

from protrepair.analysis.results import (
    AnalysisBundle,
    RamachandranAnalysis,
    RamachandranPoint,
    SecondaryStructureAnalysis,
    SecondaryStructureAssignment,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.transformer.completion.hydrogen.protonation import (
    DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO,
    DisabledHistidineProtonationRequest,
    HistidineProtonationRequest,
    PrasRatioHistidineProtonationRequest,
)
from protrepair.workflow.contracts.external_reference import (
    ExternalSpanReconstructionSpec,
    build_alphafold_span_reconstruction_specs,
)
from protrepair.workflow.contracts.planning import (
    WorkflowLigandContextMode,
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
    WorkflowSpanDonorAvailability,
    WorkflowTargetIntent,
)
from protrepair.workflow.contracts.policies import (
    CTerminalOxtPolicy,
    HydrogenPolicy,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
    OrphanFragmentPolicy,
)
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    StructureIngressOptions,
    WorkflowGoal,
    WorkflowGoalStateValue,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.result import (
    ProcessResult,
    RamachandranCategory,
    RequestedGoalCompletionVerdict,
    RequestedGoalOutcome,
    RequestedGoalReport,
    RequestedGoalStatus,
    WorkflowPhaseOutcome,
    WorkflowPhaseReport,
    WorkflowPhaseStatus,
    WorkflowTerminalBranchOutcome,
    WorkflowTerminalBranchReport,
)
from protrepair.workflow.contracts.span_policy import ExternalSpanGapSelectionPolicy

__all__ = [
    "AnalysisBundle",
    "CTerminalOxtPolicy",
    "DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO",
    "DisabledHistidineProtonationRequest",
    "ExternalSpanGapSelectionPolicy",
    "ExternalSpanReconstructionSpec",
    "HistidineProtonationRequest",
    "HydrogenPolicy",
    "LigandPolicy",
    "MutationPolicy",
    "OccupancyPolicy",
    "OrphanFragmentPolicy",
    "ProcessResult",
    "PrasRatioHistidineProtonationRequest",
    "RamachandranAnalysis",
    "RamachandranCategory",
    "RamachandranPoint",
    "RequestedGoalCompletionVerdict",
    "RetainedNonPolymerChemistryOverride",
    "WorkflowGoal",
    "WorkflowGoalStateValue",
    "RequestedGoalSet",
    "RequestedGoalOutcome",
    "RequestedGoalReport",
    "RequestedGoalStatus",
    "SecondaryStructureAnalysis",
    "SecondaryStructureAssignment",
    "StructureIngressOptions",
    "WorkflowPhaseOutcome",
    "WorkflowPhaseReport",
    "WorkflowPhaseStatus",
    "WorkflowTerminalBranchOutcome",
    "WorkflowTerminalBranchReport",
    "WorkflowPlanningContext",
    "WorkflowLigandContextMode",
    "WorkflowPlanningPhase",
    "WorkflowSpanDonorAvailability",
    "WorkflowTargetIntent",
    "WorkflowTransformRequests",
    "build_alphafold_span_reconstruction_specs",
    "requested_process_goal",
]
