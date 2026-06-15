"""Canonical local-refinement execution over one selected atom domain."""

from protrepair.transformer.refinement.local_pipeline.assessment import (
    assess_refinement_candidate,
    assess_refinement_candidate_batch,
    cached_before_acceptance_metrics,
    discrete_parser_preconditioning_short_circuits,
    execute_and_assess_refinement_candidate_batch,
    execute_refinement_candidate_batch,
    materialize_selected_refinement_candidate,
    select_refinement_candidate,
)
from protrepair.transformer.refinement.local_pipeline.backend import (
    resolve_continuous_relaxation_backend,
)
from protrepair.transformer.refinement.local_pipeline.candidates import (
    AssessedRefinementBatch,
    AssessedRefinementCandidate,
    BeforeMetricsCache,
    ExecutedRefinementBatch,
    ExecutedRefinementCandidate,
    RefinementExecutionBatch,
    RefinementExecutionCandidate,
    RefinementExecutionMode,
    SelectedRefinementCandidate,
)
from protrepair.transformer.refinement.local_pipeline.construction import (
    CandidateConstructionStageKind,
    CandidateConstructionStep,
    PreparedRefinementCandidateBase,
    RefinementCandidateLineage,
    build_refinement_execution_batch,
    prepare_refinement_candidate_base,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
    normalize_local_refinement_request,
)
from protrepair.transformer.refinement.local_pipeline.runtime import (
    LocalRefinementRuntimeProfile,
    execute_local_refinement_pipeline,
    execute_local_refinement_pipeline_profiled,
    execute_local_transformation,
    execute_local_transformation_profiled,
)

__all__ = [
    "AssessedRefinementBatch",
    "AssessedRefinementCandidate",
    "BeforeMetricsCache",
    "CandidateConstructionStageKind",
    "CandidateConstructionStep",
    "ExecutedRefinementBatch",
    "ExecutedRefinementCandidate",
    "LocalRefinementRequest",
    "LocalRefinementRuntimeProfile",
    "PreparedRefinementCandidateBase",
    "RefinementCandidateLineage",
    "RefinementExecutionBatch",
    "RefinementExecutionCandidate",
    "RefinementExecutionMode",
    "SelectedRefinementCandidate",
    "assess_refinement_candidate",
    "assess_refinement_candidate_batch",
    "build_refinement_execution_batch",
    "cached_before_acceptance_metrics",
    "discrete_parser_preconditioning_short_circuits",
    "execute_and_assess_refinement_candidate_batch",
    "execute_local_refinement_pipeline",
    "execute_local_refinement_pipeline_profiled",
    "execute_local_transformation",
    "execute_local_transformation_profiled",
    "execute_refinement_candidate_batch",
    "materialize_selected_refinement_candidate",
    "normalize_local_refinement_request",
    "prepare_refinement_candidate_base",
    "resolve_continuous_relaxation_backend",
    "select_refinement_candidate",
]
