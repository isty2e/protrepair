"""Workflow chemistry-augmentation planning over hydrogen completion."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.state import (
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.contracts.request import RequestedGoalSet
from protrepair.workflow.planning.completion.transformer_candidates import (
    plan_hydrogen_completion_transformers,
)


@dataclass(frozen=True, slots=True)
class ChemistryAugmentationPlanningOutcome:
    """Workflow planning over hydrogen completion candidates."""

    transformers: tuple[HydrogenCompletionTransformer, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transformers", tuple(self.transformers))

    def has_pending_phase(self) -> bool:
        """Return whether chemistry augmentation still has pending transformers."""

        return bool(self.transformers)


def plan_chemistry_augmentation_transformers(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    coverage_facts: StructureCoverageFacts,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
) -> ChemistryAugmentationPlanningOutcome:
    """Return workflow-engine chemistry augmentation planning."""

    hydrogen_outcome = plan_hydrogen_completion_transformers(
        structure,
        requested_goals=requested_goals,
        component_library=component_library,
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )
    return ChemistryAugmentationPlanningOutcome(
        transformers=tuple(
            transformer
            for transformer in hydrogen_outcome.transformers
            if isinstance(transformer, HydrogenCompletionTransformer)
        ),
    )
