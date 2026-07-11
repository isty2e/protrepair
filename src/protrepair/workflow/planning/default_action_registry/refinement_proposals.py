"""Local and backbone-window refinement workflow proposal adapters."""

from protrepair.state import StereochemistryState
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.continuous.domain import ContinuousRelaxationRegion
from protrepair.transformer.refinement.backbone_window import (
    atom_input_from_backbone_window_refinement_spec,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.backbone_window_refinement import (
    BackboneWindowRefinementTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
    LOCAL_REFINEMENT_CAPABILITY,
)
from protrepair.workflow.planning.local_refinement_policy import (
    LocalRefinementProposalPolicy,
)

__all__ = [
    "backbone_window_refinement_is_admissible",
    "backbone_window_refinement_proposals",
    "local_refinement_is_admissible",
    "local_refinement_proposals",
]

DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY = LocalRefinementProposalPolicy()


def local_refinement_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether local refinement is admissible in the active domain."""

    if (
        domain.intrinsic_geometry_facts is not None
        and domain.intrinsic_geometry_facts.stereochemistry_state
        is StereochemistryState.VIOLATED
    ):
        return False

    if domain.explicit_repair.is_requested():
        return (
            domain.burden.has_intrinsic_geometry_burden()
            or domain.burden.has_parser_compatibility_burden()
            or domain.burden.has_interaction_burden()
            or domain.memory.has_prior_augmentation_adoption()
        )

    return DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY.automatic_is_admissible(domain)


def backbone_window_refinement_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether requested backbone-window operator work is admissible."""

    return bool(domain.state_deficit.backbone_window_operator)


def backbone_window_refinement_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit backbone-window refinement proposals."""

    snapshot = ProteinStructureSnapshot.from_structure(domain.structure)
    return tuple(
        WorkflowActionProposal(
            transformer=transformer,
            capability=BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
            explicitly_requested=True,
        )
        for operator_deficit in domain.state_deficit.backbone_window_operator
        for transformer in (
            BackboneWindowRefinementTransformer.from_window_spec(
                operator_deficit.window_spec
            ),
        )
        if _atom_input_has_realizable_disulfide_topology(
            domain,
            snapshot=snapshot,
            atom_input=atom_input_from_backbone_window_refinement_spec(
                snapshot,
                operator_deficit.window_spec,
            ),
            context_radius_angstrom=transformer.settings.context_radius_angstrom,
        )
    )


def local_refinement_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit or automatic local-refinement proposals."""

    repair_refinement = domain.transform_requests.repair_refinement
    if repair_refinement is None:
        repair_refinement_specs = (
            DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY.automatic_specs(domain)
        )
        return tuple(
            WorkflowActionProposal(
                transformer=LocalRefinementTransformer.from_repair_refinement(
                    repair_refinement_spec
                ),
                capability=LOCAL_REFINEMENT_CAPABILITY,
                explicitly_requested=False,
            )
            for repair_refinement_spec in repair_refinement_specs
            if _refinement_spec_has_realizable_disulfide_topology(
                domain,
                repair_refinement_spec,
            )
        )

    repair_refinement_specs = DEFAULT_LOCAL_REFINEMENT_PROPOSAL_POLICY.explicit_specs(
        domain,
        repair_refinement=repair_refinement,
    )
    return tuple(
        WorkflowActionProposal(
            transformer=LocalRefinementTransformer.from_repair_refinement(
                repair_refinement_spec
            ),
            capability=LOCAL_REFINEMENT_CAPABILITY,
            explicitly_requested=True,
        )
        for repair_refinement_spec in repair_refinement_specs
        if _refinement_spec_has_realizable_disulfide_topology(
            domain,
            repair_refinement_spec,
        )
    )


def _refinement_spec_has_realizable_disulfide_topology(
    domain: WorkflowActionDomain,
    repair_refinement_spec: RepairRefinementSpec,
) -> bool:
    """Return whether one proposed FF region excludes endpoint multiplicity."""

    contradictions = (
        domain.disulfide_topology_facts.endpoint_multiplicity_contradictions
    )
    if not contradictions:
        return True

    snapshot = ProteinStructureSnapshot.from_structure(domain.structure)
    atom_input = (
        repair_refinement_spec.resolved_execution_scope_spec().lower_to_atom_input(
            snapshot,
            component_library=domain.component_library,
        )
    )
    return _atom_input_has_realizable_disulfide_topology(
        domain,
        snapshot=snapshot,
        atom_input=atom_input,
        context_radius_angstrom=(
            repair_refinement_spec.config.context_radius_angstrom
        ),
    )


def _atom_input_has_realizable_disulfide_topology(
    domain: WorkflowActionDomain,
    *,
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    context_radius_angstrom: float,
) -> bool:
    """Return whether one concrete FF region excludes endpoint multiplicity."""

    contradictions = (
        domain.disulfide_topology_facts.endpoint_multiplicity_contradictions
    )
    if not contradictions:
        return True

    region = ContinuousRelaxationRegion.from_inputs(
        snapshot,
        atom_input,
        context_radius_angstrom=context_radius_angstrom,
    )
    included_residue_ids = frozenset(
        region.residue_site(residue_index).residue_id
        for residue_index in region.included_residue_indices
    )
    return not any(
        contradiction.is_contradictory_in_residue_projection(
            included_residue_ids
        )
        for contradiction in contradictions
    )
