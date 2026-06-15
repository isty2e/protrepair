"""Workflow correction planning over refinement transformers."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.state import (
    ClashPresenceState,
    OrientationCorrectionEligibilityState,
    StereochemistryState,
    StructureChemistryReadinessFacts,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts.planning import WorkflowPlanningContext
from protrepair.workflow.contracts.request import WorkflowTransformRequests
from protrepair.workflow.planning.context_projection import (
    planning_context_is_holo_for_structure,
)
from protrepair.workflow.planning.intrinsic_geometry import (
    derive_structure_intrinsic_geometry_facts,
)


@dataclass(frozen=True, slots=True)
class CorrectionPlanningOutcome:
    """Correction planning over explicit local-refinement actions."""

    transformers: tuple[LocalRefinementTransformer, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transformers", tuple(self.transformers))

    def has_pending_phase(self) -> bool:
        """Return whether correction planning still has pending transformers."""

        return bool(self.transformers)


def plan_correction_transformers(
    structure: ProteinStructure,
    *,
    transform_requests: WorkflowTransformRequests,
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
    prior_phase_transformers_adopted: bool = False,
    intrinsic_geometry_facts: StructureIntrinsicGeometryFacts | None = None,
    parser_compatibility_facts: StructureParserCompatibilityFacts | None = None,
    interaction_facts: StructureInteractionFacts | None = None,
) -> CorrectionPlanningOutcome:
    """Return correction-phase refinement transformers when warranted."""

    repair_refinement = transform_requests.repair_refinement
    if repair_refinement is None:
        return CorrectionPlanningOutcome()

    active_intrinsic_geometry_facts = (
        derive_structure_intrinsic_geometry_facts(
            structure,
            component_library=component_library,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )
        if intrinsic_geometry_facts is None
        else intrinsic_geometry_facts
    )
    active_parser_compatibility_facts = (
        StructureParserCompatibilityFacts.from_structure(
            structure,
            component_library=component_library,
        )
        if parser_compatibility_facts is None
        else parser_compatibility_facts
    )
    requires_intrinsic_correction = _requires_intrinsic_geometry_correction(
        active_intrinsic_geometry_facts,
        parser_compatibility_facts=active_parser_compatibility_facts,
    )
    requires_interaction_correction = False
    if planning_context_is_holo_for_structure(planning_context, structure):
        active_interaction_facts = (
            StructureInteractionFacts.from_structure(
                structure,
                component_library=component_library,
                chemistry_readiness_facts=chemistry_readiness_facts,
            )
            if interaction_facts is None
            else interaction_facts
        )
        requires_interaction_correction = _requires_interaction_correction(
            active_interaction_facts
        )

    if (
        not prior_phase_transformers_adopted
        and not requires_intrinsic_correction
        and not requires_interaction_correction
    ):
        return CorrectionPlanningOutcome()

    return CorrectionPlanningOutcome(
        transformers=(
            LocalRefinementTransformer.from_repair_refinement(repair_refinement),
        )
    )


def _requires_intrinsic_geometry_correction(
    intrinsic_geometry_facts: StructureIntrinsicGeometryFacts,
    *,
    parser_compatibility_facts: StructureParserCompatibilityFacts | None = None,
) -> bool:
    """Return whether intrinsic geometry warrants correction planning."""

    return (
        intrinsic_geometry_facts.protein_self_clash_state
        is ClashPresenceState.PRESENT
        or intrinsic_geometry_facts.orientation_correction_eligibility_state
        is OrientationCorrectionEligibilityState.ELIGIBLE
        or intrinsic_geometry_facts.stereochemistry_state
        is StereochemistryState.VIOLATED
        or (
            parser_compatibility_facts is not None
            and parser_compatibility_facts.has_parser_visible_proximity_burden()
        )
    )


def _requires_interaction_correction(
    interaction_facts: StructureInteractionFacts,
) -> bool:
    """Return whether holo interaction burden warrants correction planning."""

    return interaction_facts.ligand_aware_clash_state is ClashPresenceState.PRESENT
