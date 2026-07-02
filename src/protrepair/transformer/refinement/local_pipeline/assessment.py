"""Assessment, selection, and materialization for local refinement."""

from time import perf_counter

from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.dependent_hydrogen import (
    revalidate_dependent_hydrogens_after_refinement,
)
from protrepair.transformer.refinement.acceptance import (
    RefinementAcceptanceMetrics,
    assess_refinement_result_with_before_metrics,
    measure_refinement_acceptance_metrics_for_scope,
)
from protrepair.transformer.refinement.candidate_selection import (
    assessed_refinement_candidate_order_key,
    materialize_assessed_refinement_candidate,
)
from protrepair.transformer.refinement.local_pipeline.candidates import (
    AssessedRefinementBatch,
    AssessedRefinementCandidate,
    BeforeMetricsCache,
    ExecutedRefinementBatch,
    ExecutedRefinementCandidate,
    RefinementExecutionBatch,
    SelectedRefinementCandidate,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
)
from protrepair.transformer.refinement.speculative_planning import (
    EvaluatedSpeculativeProposal,
    SpeculativeAdoptionDecision,
    SpeculativeEvaluationBatch,
)


def execute_refinement_candidate_batch(
    batch: RefinementExecutionBatch,
    *,
    request: LocalRefinementRequest,
) -> ExecutedRefinementBatch:
    """Execute one full candidate batch through the continuous backend stage."""

    return batch.execute(request=request)


def execute_and_assess_refinement_candidate_batch(
    batch: RefinementExecutionBatch,
    *,
    request: LocalRefinementRequest,
) -> tuple[AssessedRefinementBatch, float, float]:
    """Execute and assess candidates, short-circuiting parser preconditioning."""

    backend_execution_runtime_ms = 0.0
    assessment_runtime_ms = 0.0
    discrete_candidate = batch.discrete_parser_preconditioning_candidate()
    if discrete_candidate is None:
        start = perf_counter()
        executed_batch = execute_refinement_candidate_batch(
            batch,
            request=request,
        )
        backend_execution_runtime_ms = (perf_counter() - start) * 1000.0

        start = perf_counter()
        assessed_batch = assess_refinement_candidate_batch(
            executed_batch,
            request=request,
        )
        assessment_runtime_ms = (perf_counter() - start) * 1000.0
        return assessed_batch, backend_execution_runtime_ms, assessment_runtime_ms

    start = perf_counter()
    discrete_execution = discrete_candidate.execute_discrete_only()
    backend_execution_runtime_ms += (perf_counter() - start) * 1000.0

    start = perf_counter()
    before_metrics_cache: BeforeMetricsCache = {}
    before_metrics = cached_before_acceptance_metrics(
        discrete_execution,
        request=request,
        cache=before_metrics_cache,
    )
    discrete_assessment = assess_refinement_candidate(
        discrete_execution,
        request=request,
        before_metrics=before_metrics,
    )
    assessment_runtime_ms += (perf_counter() - start) * 1000.0
    if discrete_parser_preconditioning_short_circuits(discrete_assessment):
        return (
            SpeculativeEvaluationBatch(
                evaluated_proposals=(discrete_assessment,),
            ),
            backend_execution_runtime_ms,
            assessment_runtime_ms,
        )

    remaining_batch = batch.without_candidate(discrete_candidate)
    start = perf_counter()
    remaining_executed_batch = execute_refinement_candidate_batch(
        remaining_batch,
        request=request,
    )
    backend_execution_runtime_ms += (perf_counter() - start) * 1000.0

    start = perf_counter()
    remaining_assessed_batch = assess_refinement_candidate_batch(
        remaining_executed_batch,
        request=request,
        before_metrics_cache=before_metrics_cache,
    )
    assessment_runtime_ms += (perf_counter() - start) * 1000.0
    return (
        SpeculativeEvaluationBatch(
            evaluated_proposals=(
                discrete_assessment,
                *remaining_assessed_batch.evaluated_proposals,
            ),
            execution_errors=remaining_assessed_batch.execution_errors,
        ),
        backend_execution_runtime_ms,
        assessment_runtime_ms,
    )


def discrete_parser_preconditioning_short_circuits(
    assessment: AssessedRefinementCandidate,
) -> bool:
    """Return whether a discrete parser repair makes FF execution unnecessary."""

    if not assessment.evaluation.is_accepted():
        return False

    before_metrics = assessment.evaluation.before_metrics
    after_metrics = assessment.evaluation.after_metrics
    return before_metrics.whole_structure_rdkit_sanitize_readable is False and (
        after_metrics.whole_structure_rdkit_sanitize_readable is not False
        or after_metrics.whole_structure_parser_extra_heavy_proximity_bond_count == 0
    )


def assess_refinement_candidate(
    candidate: ExecutedRefinementCandidate,
    *,
    request: LocalRefinementRequest,
    before_metrics: RefinementAcceptanceMetrics,
) -> AssessedRefinementCandidate:
    """Assess one executed candidate without performing final materialization."""

    return EvaluatedSpeculativeProposal(
        execution=candidate,
        evaluation=assess_refinement_result_with_before_metrics(
            candidate.proposal.context.atom_input.as_scope(),
            request.component_library,
            request.restraint_library,
            candidate.outcome,
            before_metrics=before_metrics,
            clash_basis=request.clash_basis,
        ),
    )


def assess_refinement_candidate_batch(
    batch: ExecutedRefinementBatch,
    *,
    request: LocalRefinementRequest,
    before_metrics_cache: BeforeMetricsCache | None = None,
) -> AssessedRefinementBatch:
    """Assess all successfully executed candidates as one batch."""

    active_before_metrics_cache = (
        {} if before_metrics_cache is None else before_metrics_cache
    )

    return SpeculativeEvaluationBatch(
        evaluated_proposals=tuple(
            assess_refinement_candidate(
                candidate,
                request=request,
                before_metrics=cached_before_acceptance_metrics(
                    candidate,
                    request=request,
                    cache=active_before_metrics_cache,
                ),
            )
            for candidate in batch.executions
        ),
        execution_errors=batch.errors,
    )


def cached_before_acceptance_metrics(
    candidate: ExecutedRefinementCandidate,
    *,
    request: LocalRefinementRequest,
    cache: BeforeMetricsCache,
) -> RefinementAcceptanceMetrics:
    """Return cached fallback acceptance metrics for one candidate source/scope."""

    selected_scope = candidate.proposal.context.atom_input.as_scope()
    cache_key = (id(candidate.proposal.fallback_structure), selected_scope)
    cached_metrics = cache.get(cache_key)
    if cached_metrics is not None:
        return cached_metrics

    measured_metrics = measure_refinement_acceptance_metrics_for_scope(
        candidate.proposal.fallback_structure,
        selected_scope=selected_scope,
        component_library=request.component_library,
        restraint_library=request.restraint_library,
        clash_basis=request.clash_basis,
    )
    cache[cache_key] = measured_metrics
    return measured_metrics


def select_refinement_candidate(
    batch: AssessedRefinementBatch,
) -> SelectedRefinementCandidate:
    """Return the best assessed candidate or raise the first execution error."""

    from protrepair.errors import RefinementError

    if not batch.evaluated_proposals:
        if batch.execution_errors:
            raise batch.execution_errors[0]

        raise RefinementError("local refinement produced no executable candidates")

    best_candidate = min(
        batch.evaluated_proposals,
        key=lambda candidate: assessed_refinement_candidate_order_key(
            candidate.evaluation
        ),
    )
    return SpeculativeAdoptionDecision.adopt(best_candidate)


def materialize_selected_refinement_candidate(
    selected_candidate: SelectedRefinementCandidate,
    *,
    request: LocalRefinementRequest,
) -> RegionTransformationResult:
    """Materialize the selected assessed candidate into one final result."""

    candidate = selected_candidate.require_candidate()
    materialized_result = materialize_assessed_refinement_candidate(
        candidate.evaluation,
        original_structure=candidate.execution.proposal.fallback_structure,
        pre_backend_moved_atom_indices=(
            candidate.execution.proposal.lineage.moved_atom_indices()
        ),
    )
    if not candidate.evaluation.is_accepted():
        return materialized_result

    return revalidate_dependent_hydrogens_after_refinement(
        materialized_result,
        selected_scope=candidate.execution.proposal.context.atom_input.as_scope(),
        component_library=request.component_library,
        restraint_library=request.restraint_library,
        current_metrics=candidate.evaluation.after_metrics,
        clash_basis=request.clash_basis,
    )
