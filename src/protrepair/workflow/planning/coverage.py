"""Workflow coverage planning over reconstruction and completion."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.state import (
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowTransformRequests,
)
from protrepair.workflow.planning.completion.transformer_candidates import (
    plan_atom_completion_transformers,
)


@dataclass(frozen=True, slots=True)
class CoveragePlanningOutcome:
    """Coverage planning split into span reconstruction and atom completion."""

    span_reconstruction_transformers: tuple[
        ExternalSpanReconstructionTransformer,
        ...,
    ] = ()
    atom_completion_transformers: tuple[HeavyAtomCompletionTransformer, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "span_reconstruction_transformers",
            tuple(self.span_reconstruction_transformers),
        )
        object.__setattr__(
            self,
            "atom_completion_transformers",
            tuple(self.atom_completion_transformers),
        )

    def current_phase_transformers(
        self,
    ) -> tuple[
        ExternalSpanReconstructionTransformer | HeavyAtomCompletionTransformer,
        ...,
    ]:
        """Return the currently active coverage phase in planner order."""

        if self.span_reconstruction_transformers:
            return self.span_reconstruction_transformers

        return self.atom_completion_transformers

    def has_pending_phase(self) -> bool:
        """Return whether any coverage phase still has pending transformers."""

        return bool(self.current_phase_transformers())


def plan_coverage_transformers(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    transform_requests: WorkflowTransformRequests,
    component_library: ComponentLibrary,
    coverage_facts: StructureCoverageFacts,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
) -> CoveragePlanningOutcome:
    """Return workflow coverage planning over concrete transformers."""

    span_reconstruction_transformers = tuple(
        ExternalSpanReconstructionTransformer.from_reconstruction_spec(
            reconstruction,
        )
        for reconstruction in transform_requests.external_span_reconstructions
    )
    atom_completion_outcome = plan_atom_completion_transformers(
        structure,
        requested_goals=requested_goals,
        component_library=component_library,
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )
    return CoveragePlanningOutcome(
        span_reconstruction_transformers=span_reconstruction_transformers,
        atom_completion_transformers=tuple(
            transformer
            for transformer in atom_completion_outcome.transformers
            if isinstance(transformer, HeavyAtomCompletionTransformer)
        ),
    )
