"""Runtime orchestration for the local refinement pipeline."""

from dataclasses import dataclass
from time import perf_counter

from protrepair.chemistry import ComponentLibrary, RestraintLibrary
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.settings import ContinuousRelaxationSettings
from protrepair.transformer.refinement.local_pipeline.assessment import (
    execute_and_assess_refinement_candidate_batch,
    materialize_selected_refinement_candidate,
    select_refinement_candidate,
)
from protrepair.transformer.refinement.local_pipeline.candidates import (
    RefinementExecutionMode,
)
from protrepair.transformer.refinement.local_pipeline.construction import (
    build_refinement_execution_batch,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
    normalize_local_refinement_request,
)


@dataclass(frozen=True, slots=True)
class LocalRefinementRuntimeProfile:
    """Stage timing breakdown for one local-refinement pipeline execution."""

    request_normalization_runtime_ms: float
    candidate_construction_runtime_ms: float
    execution_candidate_count: int
    continuous_backend_candidate_count: int
    discrete_only_candidate_count: int
    backend_execution_runtime_ms: float
    assessment_runtime_ms: float
    selection_runtime_ms: float
    materialization_runtime_ms: float


def execute_local_transformation(
    context: ProteinTransformationContext,
    *,
    spec: ContinuousRelaxationSettings,
    component_library: ComponentLibrary | None = None,
    restraint_library: RestraintLibrary | None = None,
) -> RegionTransformationResult:
    """Run one canonical selected-region local-refinement transformation."""

    result, _ = execute_local_transformation_profiled(
        context,
        spec=spec,
        component_library=component_library,
        restraint_library=restraint_library,
    )
    return result


def execute_local_transformation_profiled(
    context: ProteinTransformationContext,
    *,
    spec: ContinuousRelaxationSettings,
    component_library: ComponentLibrary | None = None,
    restraint_library: RestraintLibrary | None = None,
) -> tuple[RegionTransformationResult, LocalRefinementRuntimeProfile]:
    """Run one canonical local-refinement transformation with stage timings."""

    start = perf_counter()
    request = normalize_local_refinement_request(
        context,
        spec=spec,
        component_library=component_library,
        restraint_library=restraint_library,
    )
    request_normalization_runtime_ms = (perf_counter() - start) * 1000.0

    result, runtime_profile = execute_local_refinement_pipeline_profiled(request)
    return (
        result,
        LocalRefinementRuntimeProfile(
            request_normalization_runtime_ms=request_normalization_runtime_ms,
            candidate_construction_runtime_ms=(
                runtime_profile.candidate_construction_runtime_ms
            ),
            execution_candidate_count=runtime_profile.execution_candidate_count,
            continuous_backend_candidate_count=(
                runtime_profile.continuous_backend_candidate_count
            ),
            discrete_only_candidate_count=runtime_profile.discrete_only_candidate_count,
            backend_execution_runtime_ms=runtime_profile.backend_execution_runtime_ms,
            assessment_runtime_ms=runtime_profile.assessment_runtime_ms,
            selection_runtime_ms=runtime_profile.selection_runtime_ms,
            materialization_runtime_ms=runtime_profile.materialization_runtime_ms,
        ),
    )


def execute_local_refinement_pipeline(
    request: LocalRefinementRequest,
) -> RegionTransformationResult:
    """Execute the one-way local-refinement pipeline over one canonical request."""

    result, _ = execute_local_refinement_pipeline_profiled(request)
    return result


def execute_local_refinement_pipeline_profiled(
    request: LocalRefinementRequest,
) -> tuple[RegionTransformationResult, LocalRefinementRuntimeProfile]:
    """Execute the local-refinement pipeline and return stage timings."""

    start = perf_counter()
    execution_batch = build_refinement_execution_batch(request)
    candidate_construction_runtime_ms = (perf_counter() - start) * 1000.0

    (
        assessed_batch,
        backend_execution_runtime_ms,
        assessment_runtime_ms,
    ) = execute_and_assess_refinement_candidate_batch(
        execution_batch,
        request=request,
    )

    start = perf_counter()
    selected_candidate = select_refinement_candidate(assessed_batch)
    selection_runtime_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    result = materialize_selected_refinement_candidate(
        selected_candidate,
        request=request,
    )
    materialization_runtime_ms = (perf_counter() - start) * 1000.0

    return (
        result,
        LocalRefinementRuntimeProfile(
            request_normalization_runtime_ms=0.0,
            candidate_construction_runtime_ms=candidate_construction_runtime_ms,
            execution_candidate_count=len(execution_batch.candidates),
            continuous_backend_candidate_count=(
                execution_batch.candidate_count_for_mode(
                    RefinementExecutionMode.CONTINUOUS_RELAXATION,
                )
            ),
            discrete_only_candidate_count=execution_batch.candidate_count_for_mode(
                RefinementExecutionMode.DISCRETE_ONLY,
            ),
            backend_execution_runtime_ms=backend_execution_runtime_ms,
            assessment_runtime_ms=assessment_runtime_ms,
            selection_runtime_ms=selection_runtime_ms,
            materialization_runtime_ms=materialization_runtime_ms,
        ),
    )
