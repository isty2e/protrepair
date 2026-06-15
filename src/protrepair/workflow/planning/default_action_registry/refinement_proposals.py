"""Local and backbone-window refinement workflow proposal adapters."""

from dataclasses import replace

from protrepair.diagnostics.geometry import (
    detect_heavy_geometry,
    severe_intrinsic_geometry_residues,
)
from protrepair.diagnostics.parser_readability import (
    rdkit_no_conect_parser_failing_residue_ids,
)
from protrepair.state import HydrogenCoverageState, StereochemistryState
from protrepair.structure.labels import ResidueId
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationConfig
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.parser_witness import (
    parser_witness_repair_candidates,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.backbone_window_refinement import (
    BackboneWindowRefinementTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
    LOCAL_REFINEMENT_CAPABILITY,
)
from protrepair.workflow.planning.joint_correction_scopes import (
    batch_joint_correction_scope_proposals,
    execution_scope_spec_for_joint_correction,
    propose_joint_correction_scopes,
)

__all__ = [
    "backbone_window_refinement_is_admissible",
    "backbone_window_refinement_proposals",
    "local_refinement_is_admissible",
    "local_refinement_proposals",
]


def local_refinement_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether local refinement is admissible in the active domain."""

    if (
        domain.intrinsic_geometry_facts is not None
        and domain.intrinsic_geometry_facts.stereochemistry_state
        is StereochemistryState.VIOLATED
    ):
        return False

    if domain.requests_explicit_repair_refinement():
        return (
            domain.has_intrinsic_geometry_burden()
            or domain.has_parser_compatibility_burden()
            or domain.has_interaction_burden()
            or domain.has_prior_augmentation_adoption()
        )

    return _automatic_repair_refinement_is_admissible(domain)


def backbone_window_refinement_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether requested backbone-window operator work is admissible."""

    return domain.requires_backbone_window_refinement()


def backbone_window_refinement_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit backbone-window refinement proposals."""

    return tuple(
        WorkflowActionProposal(
            transformer=BackboneWindowRefinementTransformer.from_window_spec(
                operator_deficit.window_spec
            ),
            capability=BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
            explicitly_requested=True,
        )
        for operator_deficit in domain.state_deficit.backbone_window_operator
    )


def local_refinement_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit or automatic local-refinement proposals."""

    repair_refinement = domain.transform_requests.repair_refinement
    if repair_refinement is None:
        return tuple(
            WorkflowActionProposal(
                transformer=LocalRefinementTransformer.from_repair_refinement(
                    repair_refinement_spec
                ),
                capability=LOCAL_REFINEMENT_CAPABILITY,
                explicitly_requested=False,
            )
            for repair_refinement_spec in _automatic_repair_refinement_specs(domain)
        )

    return tuple(
        WorkflowActionProposal(
            transformer=LocalRefinementTransformer.from_repair_refinement(
                repair_refinement_spec
            ),
            capability=LOCAL_REFINEMENT_CAPABILITY,
            explicitly_requested=True,
        )
        for repair_refinement_spec in _ordered_repair_refinement_specs(
            domain,
            repair_refinement=repair_refinement,
        )
    )


def _ordered_repair_refinement_specs(
    domain: WorkflowActionDomain,
    *,
    repair_refinement: RepairRefinementSpec,
) -> tuple[RepairRefinementSpec, ...]:
    """Return refinement specs ordered from expanded joint scopes to fallback."""

    if not repair_refinement.scope_spec.is_residuewise():
        return (repair_refinement,)

    requested_residue_ids = repair_refinement.scope_spec.referenced_residue_ids()
    joint_focus_residue_ids = requested_residue_ids
    if domain.has_prior_augmentation_adoption():
        prerequisite_residue_ids = (
            domain.explicit_repair_refinement_prerequisite_residue_ids()
        )
        if prerequisite_residue_ids:
            joint_focus_residue_ids = prerequisite_residue_ids
    joint_scope_proposals = propose_joint_correction_scopes(
        domain.structure,
        focus_residue_ids=joint_focus_residue_ids,
        component_library=domain.component_library,
        chemistry_readiness_facts=domain.chemistry_readiness_facts,
        include_ligands=domain.is_holo_context(),
    )
    if domain.has_prior_augmentation_adoption():
        ordered_joint_scope_proposals = joint_scope_proposals
    else:
        ordered_joint_scope_proposals = batch_joint_correction_scope_proposals(
            domain.structure,
            proposals=joint_scope_proposals,
        )
    ordered_specs: list[RepairRefinementSpec] = []
    seen_scope_specs = {
        (
            repair_refinement.scope_spec,
            repair_refinement.execution_scope_spec,
        )
    }
    for proposal in ordered_joint_scope_proposals:
        scope_spec = proposal.as_local_scope_spec()
        execution_scope_spec = execution_scope_spec_for_joint_correction(
            domain.structure,
            proposal=proposal,
        )
        scope_spec_key = (scope_spec, execution_scope_spec)
        if scope_spec_key in seen_scope_specs:
            continue
        ordered_specs.append(
            replace(
                repair_refinement,
                scope_spec=scope_spec,
                execution_scope_spec=execution_scope_spec,
            )
        )
        seen_scope_specs.add(scope_spec_key)

    ordered_specs.append(repair_refinement)
    return tuple(ordered_specs)


def _automatic_repair_refinement_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether automatic severe-cluster refinement may be proposed."""

    if not domain.has_adopted_action_family(LocalRefinementTransformer):
        return True

    return domain.has_adopted_action_family(CommittedPackingTransformer) and (
        domain.has_intrinsic_geometry_burden()
        or domain.has_parser_compatibility_burden()
        or domain.has_interaction_burden()
    )


def _automatic_repair_refinement_specs(
    domain: WorkflowActionDomain,
) -> tuple[RepairRefinementSpec, ...]:
    """Return automatic repair-refinement specs for intrinsic and joint burdens."""

    if not _automatic_repair_refinement_is_admissible(domain):
        return ()

    automatic_specs: list[RepairRefinementSpec] = []
    seen_scope_specs: set[tuple[LocalScopeSpec, LocalScopeSpec | None]] = set()

    parser_compatibility_facts = domain.parser_compatibility_facts
    parser_visible_proximity_burden = (
        parser_compatibility_facts is None
        or parser_compatibility_facts.has_parser_visible_proximity_burden()
    )
    parser_candidates = (
        parser_witness_repair_candidates(
            domain.structure,
            component_library=domain.component_library,
        )
        if parser_visible_proximity_burden
        else ()
    )
    for candidate in parser_candidates:
        scope_spec_key = (candidate.repair_refinement.scope_spec, None)
        if scope_spec_key in seen_scope_specs:
            continue

        automatic_specs.append(candidate.repair_refinement)
        seen_scope_specs.add(scope_spec_key)

    if automatic_specs:
        return tuple(automatic_specs)

    if not parser_visible_proximity_burden:
        return tuple(automatic_specs)

    if (
        domain.chemistry_readiness_facts.hydrogen_coverage_state
        is not HydrogenCoverageState.COMPLETE
    ):
        return tuple(automatic_specs)

    failing_residue_ids = _parser_failing_residue_ids_for_domain(domain)
    if not failing_residue_ids:
        return tuple(automatic_specs)

    geometry_report = detect_heavy_geometry(
        domain.structure,
        component_library=domain.component_library,
        residue_ids=failing_residue_ids,
    )
    severe_geometry_residues = tuple(
        finding
        for finding in severe_intrinsic_geometry_residues(geometry_report)
        if finding.residue_id in failing_residue_ids
    )
    for finding in severe_geometry_residues[:1]:
        scope_spec = LocalScopeSpec.from_residues((finding.residue_id,))
        scope_spec_key = (scope_spec, None)
        if scope_spec_key in seen_scope_specs:
            continue

        automatic_specs.append(
            RepairRefinementSpec(
                scope_spec=scope_spec,
                binding=RecommendedContinuousRelaxationBinding(),
                config=ContinuousRelaxationConfig(
                    context_radius_angstrom=3.0,
                ),
            )
        )
        seen_scope_specs.add(scope_spec_key)

    joint_scope_proposals = propose_joint_correction_scopes(
        domain.structure,
        focus_residue_ids=failing_residue_ids,
        component_library=domain.component_library,
        chemistry_readiness_facts=domain.chemistry_readiness_facts,
        include_ligands=False,
    )
    batched_joint_scope_proposals = batch_joint_correction_scope_proposals(
        domain.structure,
        proposals=joint_scope_proposals,
    )
    for proposal in batched_joint_scope_proposals[:1]:
        scope_spec = proposal.as_local_scope_spec()
        execution_scope_spec = execution_scope_spec_for_joint_correction(
            domain.structure,
            proposal=proposal,
        )
        scope_spec_key = (scope_spec, execution_scope_spec)
        if scope_spec_key in seen_scope_specs:
            continue

        automatic_specs.append(
            RepairRefinementSpec(
                scope_spec=scope_spec,
                execution_scope_spec=execution_scope_spec,
                binding=RecommendedContinuousRelaxationBinding(),
            )
        )
        seen_scope_specs.add(scope_spec_key)

    return tuple(automatic_specs)


def _parser_failing_residue_ids_for_domain(
    domain: WorkflowActionDomain,
) -> tuple[ResidueId, ...]:
    """Return parser-visible residue ids for automatic refinement fallback."""

    return tuple(
        sorted(
            rdkit_no_conect_parser_failing_residue_ids(
                domain.structure,
                component_library=domain.component_library,
            ),
            key=lambda residue_id: residue_id.display_token(),
        )
    )
