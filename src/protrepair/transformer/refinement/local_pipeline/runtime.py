"""Runtime orchestration for the local refinement pipeline."""

from dataclasses import dataclass, replace
from time import perf_counter

from protrepair.chemistry import ComponentLibrary, RestraintLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
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
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    chemistry_observation: StructureChemistryObservation | None = None,
) -> RegionTransformationResult:
    """Run a selected-region local refinement.

    Parameters
    ----------
    context : ProteinTransformationContext
        Source carrier and selected atom domain.
    spec : ContinuousRelaxationSettings
        Force field, execution region and search settings.
    component_library : ComponentLibrary or None
        Active chemistry definitions, or defaults.
    restraint_library : RestraintLibrary or None
        Active geometric restraints, or defaults.
    allow_retained_non_polymer_rdkit_fallback : bool
        Whether retained-component inference is permitted.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit retained-component evidence.
    chemistry_observation : StructureChemistryObservation or None
        Chemistry reuse bound to the exact source and chemistry inputs.

    Returns
    -------
    RegionTransformationResult
        Selected result after candidate execution and assessment.

    Raises
    ------
    RefinementError
        Readiness, execution or candidate selection fails.
    ValueError
        A supplied observation has different derivation inputs.
    """

    result, _ = execute_local_transformation_profiled(
        context,
        spec=spec,
        component_library=component_library,
        restraint_library=restraint_library,
        allow_retained_non_polymer_rdkit_fallback=(
            allow_retained_non_polymer_rdkit_fallback
        ),
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        chemistry_observation=chemistry_observation,
    )
    return result


def execute_local_transformation_profiled(
    context: ProteinTransformationContext,
    *,
    spec: ContinuousRelaxationSettings,
    component_library: ComponentLibrary | None = None,
    restraint_library: RestraintLibrary | None = None,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    chemistry_observation: StructureChemistryObservation | None = None,
) -> tuple[RegionTransformationResult, LocalRefinementRuntimeProfile]:
    """Run local refinement and measure its stages.

    Parameters
    ----------
    context : ProteinTransformationContext
        Source carrier and selected atom domain.
    spec : ContinuousRelaxationSettings
        Force field, execution region and search settings.
    component_library : ComponentLibrary or None
        Active chemistry definitions, or defaults.
    restraint_library : RestraintLibrary or None
        Active geometric restraints, or defaults.
    allow_retained_non_polymer_rdkit_fallback : bool
        Whether retained-component inference is permitted.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit retained-component evidence.
    chemistry_observation : StructureChemistryObservation or None
        Chemistry reuse bound to the exact source and chemistry inputs.

    Returns
    -------
    tuple[RegionTransformationResult, LocalRefinementRuntimeProfile]
        Selected result, elapsed stage times and candidate counts.

    Raises
    ------
    RefinementError
        Readiness, execution or candidate selection fails.
    ValueError
        A supplied observation has different derivation inputs.
    """

    start = perf_counter()
    request = LocalRefinementRequest.from_context(
        context,
        spec=spec,
        component_library=component_library,
        restraint_library=restraint_library,
        allow_retained_non_polymer_rdkit_fallback=(
            allow_retained_non_polymer_rdkit_fallback
        ),
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        chemistry_observation=chemistry_observation,
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
    result = replace(result, relaxation_executions=assessed_batch.relaxation_executions)
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
