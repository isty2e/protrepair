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
from protrepair.transformer.packing import PackingMode, PackingScope, PackingSpec
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
    WorkflowBranchOutcome,
    WorkflowBranchReport,
    WorkflowBranchStatus,
    WorkflowPhaseOutcome,
    WorkflowPhaseReport,
    WorkflowPhaseStatus,
)
from protrepair.workflow.contracts.span_policy import ExternalSpanGapSelectionPolicy

__all__ = [
    "DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO",
    "AnalysisBundle",
    "DisabledHistidineProtonationRequest",
    "ExternalSpanGapSelectionPolicy",
    "ExternalSpanReconstructionSpec",
    "HistidineProtonationRequest",
    "LigandPolicy",
    "MutationPolicy",
    "OccupancyPolicy",
    "OrphanFragmentPolicy",
    "PackingMode",
    "PackingScope",
    "PackingSpec",
    "PrasRatioHistidineProtonationRequest",
    "ProcessResult",
    "RamachandranAnalysis",
    "RamachandranCategory",
    "RamachandranPoint",
    "RequestedGoalCompletionVerdict",
    "RequestedGoalOutcome",
    "RequestedGoalReport",
    "RequestedGoalSet",
    "RequestedGoalStatus",
    "RetainedNonPolymerChemistryOverride",
    "SecondaryStructureAnalysis",
    "SecondaryStructureAssignment",
    "StructureIngressOptions",
    "WorkflowBranchOutcome",
    "WorkflowBranchReport",
    "WorkflowBranchStatus",
    "WorkflowGoal",
    "WorkflowGoalStateValue",
    "WorkflowLigandContextMode",
    "WorkflowPhaseOutcome",
    "WorkflowPhaseReport",
    "WorkflowPhaseStatus",
    "WorkflowPlanningContext",
    "WorkflowPlanningPhase",
    "WorkflowSpanDonorAvailability",
    "WorkflowTargetIntent",
    "WorkflowTransformRequests",
    "build_alphafold_span_reconstruction_specs",
    "requested_process_goal",
]
