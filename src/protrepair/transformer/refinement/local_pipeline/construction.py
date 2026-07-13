"""Candidate construction for local refinement."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.clashes import ClashDetectionBasis
from protrepair.io.pdb_projection import RDKitNoConectPDBBlockProjector
from protrepair.structure.aggregate import ProteinStructure
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
from protrepair.transformer.refinement.local_pipeline.lineage import (
    CandidateConstructionStageKind,
    RefinementCandidateLineage,
)

if TYPE_CHECKING:
    from protrepair.transformer.refinement.local_pipeline.request import (
        LocalRefinementRequest,
    )

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
        pdb_block_projector=request.pdb_block_projector,
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
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
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
        pdb_block_projector=pdb_block_projector,
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
