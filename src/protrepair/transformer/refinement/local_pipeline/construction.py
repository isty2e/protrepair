"""Candidate construction for local refinement."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.clashes import ClashDetectionBasis
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.discrete import (
    BranchedSidechainSeedTransformer,
    DiscretePreRefinementCorrectionTransformer,
    ParserWitnessPreUntangleTransformer,
)
from protrepair.transformer.refinement.local_pipeline.candidates import (
    RefinementExecutionBatch,
    RefinementExecutionCandidate,
    RefinementExecutionMode,
)

if TYPE_CHECKING:
    from protrepair.transformer.refinement.local_pipeline.request import (
        LocalRefinementRequest,
    )


class CandidateConstructionStageKind(str, Enum):
    """Closed internal stages that can construct one refinement candidate."""

    DISCRETE_PREPARATION = "discrete_preparation"
    PARSER_WITNESS_PRE_UNTANGLE = "parser_witness_pre_untangle"
    DISCRETE_SEEDING = "discrete_seeding"


@dataclass(frozen=True, slots=True)
class CandidateConstructionStep:
    """One internal candidate-construction stage plus the atoms it moved."""

    kind: CandidateConstructionStageKind
    moved_atom_indices: tuple[AtomIndex, ...] = ()

    def __post_init__(self) -> None:
        ordered_atom_indices: list[AtomIndex] = []
        seen_atom_indices: set[AtomIndex] = set()
        for atom_index in self.moved_atom_indices:
            if atom_index in seen_atom_indices:
                continue

            ordered_atom_indices.append(atom_index)
            seen_atom_indices.add(atom_index)

        object.__setattr__(self, "moved_atom_indices", tuple(ordered_atom_indices))


@dataclass(frozen=True, slots=True)
class RefinementCandidateLineage:
    """Internal lineage of discrete stages applied before backend execution."""

    steps: tuple[CandidateConstructionStep, ...] = ()

    def with_step(
        self,
        *,
        kind: CandidateConstructionStageKind,
        moved_atom_indices: tuple[AtomIndex, ...],
    ) -> "RefinementCandidateLineage":
        """Return lineage extended by one construction stage when it moved atoms."""

        if not moved_atom_indices:
            return self

        return type(self)(
            steps=(
                *self.steps,
                CandidateConstructionStep(
                    kind=kind,
                    moved_atom_indices=moved_atom_indices,
                ),
            )
        )

    def moved_atom_indices(self) -> tuple[AtomIndex, ...]:
        """Return all pre-backend atom movement in canonical deduplicated order."""

        ordered_atom_indices: list[AtomIndex] = []
        seen_atom_indices: set[AtomIndex] = set()
        for step in self.steps:
            for atom_index in step.moved_atom_indices:
                if atom_index in seen_atom_indices:
                    continue

                ordered_atom_indices.append(atom_index)
                seen_atom_indices.add(atom_index)

        return tuple(ordered_atom_indices)

    def has_step_kind(self, kind: CandidateConstructionStageKind) -> bool:
        """Return whether this lineage contains one construction stage kind."""

        return any(step.kind is kind for step in self.steps)


@dataclass(frozen=True, slots=True)
class PreparedRefinementCandidateBase:
    """Prepared local-refinement base before optional seed expansion."""

    context: ProteinTransformationContext
    lineage: RefinementCandidateLineage

    def build_seed_execution_candidates(
        self,
        *,
        component_library: ComponentLibrary,
        fallback_structure: ProteinStructure,
    ) -> tuple[RefinementExecutionCandidate, ...]:
        """Return seeded candidates generated from this prepared baseline."""

        seed_transformer = BranchedSidechainSeedTransformer(component_library)
        if not seed_transformer.is_applicable(self.context):
            return ()

        seed_candidates: list[RefinementExecutionCandidate] = []
        for seeded_candidate in seed_transformer.transform(self.context):
            moved_atom_indices = self.context.source_snapshot.moved_atom_indices_to(
                seeded_candidate.payload,
                self.context.atom_input.atom_indices,
            )
            seed_candidates.append(
                RefinementExecutionCandidate(
                    context=ProteinTransformationContext(
                        source_snapshot=seeded_candidate.payload,
                        atom_input=self.context.atom_input,
                        supporting_structures=self.context.supporting_structures,
                    ),
                    lineage=self.lineage.with_step(
                        kind=CandidateConstructionStageKind.DISCRETE_SEEDING,
                        moved_atom_indices=moved_atom_indices,
                    ),
                    fallback_structure=fallback_structure,
                )
            )

        return tuple(seed_candidates)


def build_refinement_execution_batch(
    request: "LocalRefinementRequest",
) -> RefinementExecutionBatch:
    """Construct baseline and seeded execution candidates for one request."""

    prepared_base = prepare_refinement_candidate_base(
        request.context,
        component_library=request.component_library,
        clash_basis=request.clash_basis,
    )
    candidates: list[RefinementExecutionCandidate] = []
    if prepared_base.lineage.moved_atom_indices():
        candidates.append(
            RefinementExecutionCandidate(
                context=prepared_base.context,
                lineage=prepared_base.lineage,
                fallback_structure=request.context.source_snapshot.structure,
                execution_mode=RefinementExecutionMode.DISCRETE_ONLY,
            )
        )

    baseline_candidate = RefinementExecutionCandidate(
        context=prepared_base.context,
        lineage=prepared_base.lineage,
        fallback_structure=request.context.source_snapshot.structure,
    )
    candidates.append(baseline_candidate)
    candidates.extend(
        prepared_base.build_seed_execution_candidates(
            component_library=request.component_library,
            fallback_structure=request.context.source_snapshot.structure,
        )
    )
    return RefinementExecutionBatch(candidates=tuple(candidates))


def prepare_refinement_candidate_base(
    context: ProteinTransformationContext,
    *,
    component_library: ComponentLibrary,
    clash_basis: ClashDetectionBasis,
) -> PreparedRefinementCandidateBase:
    """Return one prepared baseline before optional seed candidate expansion."""

    active_context = context
    lineage = RefinementCandidateLineage()
    discrete_transformer = DiscretePreRefinementCorrectionTransformer(
        component_library,
        clash_basis=clash_basis,
    )
    if discrete_transformer.is_applicable(active_context):
        corrected_snapshot = discrete_transformer.transform(active_context)
        corrected_moved_atom_indices = (
            active_context.source_snapshot.moved_atom_indices_to(
                corrected_snapshot,
                active_context.atom_input.atom_indices,
            )
        )
        active_context = ProteinTransformationContext(
            source_snapshot=corrected_snapshot,
            atom_input=active_context.atom_input,
            supporting_structures=active_context.supporting_structures,
        )
        lineage = lineage.with_step(
            kind=CandidateConstructionStageKind.DISCRETE_PREPARATION,
            moved_atom_indices=corrected_moved_atom_indices,
        )

    pre_untangle_transformer = ParserWitnessPreUntangleTransformer(
        component_library,
        clash_basis=clash_basis,
    )
    pre_untangled_snapshot = pre_untangle_transformer.transform(active_context)
    pre_untangled_moved_atom_indices = (
        active_context.source_snapshot.moved_atom_indices_to(
            pre_untangled_snapshot,
            active_context.atom_input.atom_indices,
        )
    )
    if pre_untangled_moved_atom_indices:
        active_context = ProteinTransformationContext(
            source_snapshot=pre_untangled_snapshot,
            atom_input=active_context.atom_input,
            supporting_structures=active_context.supporting_structures,
        )
        lineage = lineage.with_step(
            kind=CandidateConstructionStageKind.PARSER_WITNESS_PRE_UNTANGLE,
            moved_atom_indices=pre_untangled_moved_atom_indices,
        )

    return PreparedRefinementCandidateBase(
        context=active_context,
        lineage=lineage,
    )
