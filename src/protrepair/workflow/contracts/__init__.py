"""Workflow contracts for requests, results, and policy enums."""

from protrepair.analysis.results import (
    AnalysisBundle,
    RamachandranAnalysis,
    RamachandranPoint,
    SecondaryStructureAnalysis,
    SecondaryStructureAssignment,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.resolution import SiteProtonationReason
from protrepair.chemistry.microstate.selection import SiteProtonation
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
from protrepair.workflow.contracts.hydrogen_lengths import (
    HydrogenLengthRepairOutcome,
    HydrogenLengthRepairStatus,
    HydrogenLengthReport,
    HydrogenLengthRequest,
)
from protrepair.workflow.contracts.peptide_capping import (
    PeptideCapOutcome,
    PeptideCappingReport,
    PeptideCappingRequest,
    PeptideCapStatus,
    PeptideEnd,
    PeptidePort,
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
from protrepair.workflow.contracts.site_protonation import (
    SiteProtonationOutcome,
    SiteProtonationReport,
    SiteProtonationStatus,
)
from protrepair.workflow.contracts.span_policy import ExternalSpanGapSelectionPolicy
from protrepair.workflow.contracts.span_report import (
    ExternalSpanReconstructionOutcome,
    ExternalSpanReconstructionReport,
    ExternalSpanReconstructionStatus,
)

__all__ = [
    "DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO",
    "AnalysisBundle",
    "DisabledHistidineProtonationRequest",
    "ExternalSpanGapSelectionPolicy",
    "ExternalSpanReconstructionOutcome",
    "ExternalSpanReconstructionReport",
    "ExternalSpanReconstructionSpec",
    "ExternalSpanReconstructionStatus",
    "HistidineProtonationRequest",
    "HydrogenLengthRepairOutcome",
    "HydrogenLengthRepairStatus",
    "HydrogenLengthReport",
    "HydrogenLengthRequest",
    "LigandPolicy",
    "MutationPolicy",
    "OccupancyPolicy",
    "OrphanFragmentPolicy",
    "PackingMode",
    "PackingScope",
    "PackingSpec",
    "PeptideCapOutcome",
    "PeptideCapStatus",
    "PeptideCappingReport",
    "PeptideCappingRequest",
    "PeptideEnd",
    "PeptidePort",
    "PolymerChemicalSite",
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
    "SiteProtonation",
    "SiteProtonationOutcome",
    "SiteProtonationReason",
    "SiteProtonationReport",
    "SiteProtonationStatus",
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
