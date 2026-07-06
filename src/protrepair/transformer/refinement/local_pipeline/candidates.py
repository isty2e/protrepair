"""Execution candidates for local refinement."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from protrepair.errors import RefinementError
from protrepair.scope import Scope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.artifacts.patch import StructureDelta
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.refinement.acceptance import (
    AssessedRefinementResult,
    RefinementAcceptanceMetrics,
)
from protrepair.transformer.refinement.local_pipeline.lineage import (
    CandidateConstructionStageKind,
    RefinementCandidateLineage,
)
from protrepair.transformer.refinement.speculative_planning import (
    EvaluatedSpeculativeProposal,
    SpeculativeAdoptionDecision,
    SpeculativeEvaluationBatch,
    SpeculativeExecution,
    SpeculativeExecutionBatch,
)

if TYPE_CHECKING:
    from protrepair.transformer.refinement.local_pipeline.request import (
        LocalRefinementRequest,
    )


class RefinementExecutionMode(str, Enum):
    """Closed execution modes for prepared refinement candidates."""

    CONTINUOUS_RELAXATION = "continuous_relaxation"
    DISCRETE_ONLY = "discrete_only"


@dataclass(frozen=True, slots=True)
class RefinementExecutionCandidate:
    """One canonical candidate ready for a refinement execution mode."""

    context: ProteinTransformationContext
    lineage: "RefinementCandidateLineage"
    fallback_structure: ProteinStructure
    execution_mode: RefinementExecutionMode = (
        RefinementExecutionMode.CONTINUOUS_RELAXATION
    )

    def execute(
        self,
        *,
        request: "LocalRefinementRequest",
    ) -> "ExecutedRefinementCandidate":
        """Execute this candidate according to its execution mode."""

        if self.execution_mode is RefinementExecutionMode.DISCRETE_ONLY:
            return self.execute_discrete_only()
        if self.execution_mode is RefinementExecutionMode.CONTINUOUS_RELAXATION:
            return self.execute_continuous(request=request)

        raise RefinementError(
            f"unknown refinement execution mode {self.execution_mode!r}"
        )

    def execute_continuous(
        self,
        *,
        request: "LocalRefinementRequest",
    ) -> "ExecutedRefinementCandidate":
        """Execute this candidate through the continuous backend stage."""

        atom_scope = self.context.atom_input.observed_atom_scope(
            self.context.source_snapshot
        )
        require_atom_scope_continuous_relaxation_execution(
            derive_atom_scope_continuous_relaxation_facts(
                self.context.source_snapshot,
                atom_scope,
                component_library=request.component_library,
                context_radius_angstrom=request.spec.context_radius_angstrom,
                allow_retained_non_polymer_rdkit_fallback=(
                    request.allow_retained_non_polymer_rdkit_fallback
                ),
                retained_non_polymer_chemistry_evidence=(
                    request.retained_non_polymer_chemistry_evidence
                ),
                hydrogen_expectation_model=request.hydrogen_expectation_model,
            )
        )

        problem = ContinuousRelaxationProblem.from_inputs(
            self.context.source_snapshot,
            self.context.atom_input,
            spec=request.spec,
            component_library=request.component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                request.allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                request.retained_non_polymer_chemistry_evidence
            ),
            retained_non_polymer_chemistry_resolution_by_residue_id=(
                None
                if request.hydrogen_expectation_model is None
                else (
                    request.hydrogen_expectation_model
                    .retained_non_polymer_resolution_by_residue_id
                )
            ),
        )
        return SpeculativeExecution(
            proposal=self,
            outcome=request.backend.relax(
                problem,
                restraint_library=request.restraint_library,
            ),
        )

    def execute_discrete_only(self) -> "ExecutedRefinementCandidate":
        """Materialize this pre-backend discrete candidate without FF execution."""

        if not self.lineage.moved_atom_indices():
            raise RefinementError(
                "discrete-only refinement candidates require pre-backend movement"
            )

        refined_structure = self.context.source_snapshot.structure
        return SpeculativeExecution(
            proposal=self,
            outcome=RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=self.fallback_structure.constitution,
                    after_constitution=refined_structure.constitution,
                ),
                issues=(),
                backend_name="discrete_preconditioning",
            ),
        )

    def is_parser_preconditioning_candidate(self) -> bool:
        """Return whether this is the parser-witness discrete preconditioner."""

        return (
            self.execution_mode is RefinementExecutionMode.DISCRETE_ONLY
            and self.lineage.has_step_kind(
                CandidateConstructionStageKind.PARSER_WITNESS_PRE_UNTANGLE
            )
        )


@dataclass(frozen=True, slots=True)
class RefinementExecutionBatch:
    """Candidate-construction output ready for backend execution."""

    candidates: tuple[RefinementExecutionCandidate, ...]

    def candidate_count_for_mode(self, mode: RefinementExecutionMode) -> int:
        """Return how many candidates in this batch use one execution mode."""

        return sum(
            1
            for candidate in self.candidates
            if candidate.execution_mode is mode
        )

    def discrete_parser_preconditioning_candidate(
        self,
    ) -> RefinementExecutionCandidate | None:
        """Return the parser-witness discrete-only candidate when present."""

        if not self.candidates:
            return None

        candidate = self.candidates[0]
        if candidate.is_parser_preconditioning_candidate():
            return candidate

        return None

    def without_candidate(
        self,
        excluded_candidate: RefinementExecutionCandidate,
    ) -> "RefinementExecutionBatch":
        """Return this batch without one candidate identity."""

        return type(self)(
            candidates=tuple(
                candidate
                for candidate in self.candidates
                if candidate is not excluded_candidate
            )
        )

    def execute(
        self,
        *,
        request: "LocalRefinementRequest",
    ) -> "ExecutedRefinementBatch":
        """Execute this full candidate batch through the backend stage."""

        executed_candidates: list[ExecutedRefinementCandidate] = []
        execution_errors: list[RefinementError] = []
        for candidate in self.candidates:
            try:
                executed_candidates.append(candidate.execute(request=request))
            except RefinementError as exc:
                execution_errors.append(exc)

        return SpeculativeExecutionBatch(
            executions=tuple(executed_candidates),
            errors=tuple(execution_errors),
        )

ExecutedRefinementCandidate: TypeAlias = SpeculativeExecution[
    RefinementExecutionCandidate,
    RegionTransformationResult,
]
ExecutedRefinementBatch: TypeAlias = SpeculativeExecutionBatch[
    RefinementExecutionCandidate,
    RegionTransformationResult,
    RefinementError,
]
AssessedRefinementCandidate: TypeAlias = EvaluatedSpeculativeProposal[
    RefinementExecutionCandidate,
    RegionTransformationResult,
    AssessedRefinementResult,
]
AssessedRefinementBatch: TypeAlias = SpeculativeEvaluationBatch[
    RefinementExecutionCandidate,
    RegionTransformationResult,
    AssessedRefinementResult,
    RefinementError,
]
SelectedRefinementCandidate: TypeAlias = SpeculativeAdoptionDecision[
    RefinementExecutionCandidate,
    RegionTransformationResult,
    AssessedRefinementResult,
]
BeforeMetricsCache: TypeAlias = dict[tuple[int, Scope], RefinementAcceptanceMetrics]
