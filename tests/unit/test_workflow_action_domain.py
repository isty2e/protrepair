"""Unit tests for workflow action domain admissibility."""

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_benchmarks import resolve_fixture_path
from tests.support.refinement_cases import EXPLORATORY_REFINEMENT_FIXTURE_SOURCES
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.relation.blueprint import StructureBlueprintCoverageGap
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state import (
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    StructureBoundaryStateFacts,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.local.projection import (
    LocalContinuousExecutionResidueProjection,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.retained_non_polymer_hydrogen_completion import (
    RetainedNonPolymerHydrogenCompletionTransformer,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    RequestedGoalSet,
    WorkflowPlanningContext,
    WorkflowSpanDonorAvailability,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.registry import (
    action_adoption_for_registry,
    action_registry_entry_for_type,
)
from protrepair.workflow.planning.assessment.deficits import WorkflowStateDeficit
from protrepair.workflow.planning.default_action_registry import (
    WORKFLOW_ACTION_REGISTRY,
)


def test_heavy_completion_domain_is_admissible_for_required_atom_coverage() -> None:
    """Heavy completion should be admissible when required atom coverage remains."""

    domain = _workflow_action_domain(
        _sidechain_incomplete_structure(),
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=SidechainHeavyAtomCompletenessState.COMPLETE,
                ),
            )
        ),
    )

    assert _is_admissible(HeavyAtomCompletionTransformer, domain)
    assert not _is_admissible(HydrogenCompletionTransformer, domain)


def test_hydrogen_completion_domain_waits_for_atom_coverage_prerequisites() -> None:
    """Hydrogen completion should stay inadmissible until heavy coverage is clear."""

    missing_heavy_domain = _workflow_action_domain(
        _sidechain_incomplete_structure(),
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
            )
        ),
    )
    heavy_complete_domain = _workflow_action_domain(
        _heavy_complete_structure(),
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
            )
        ),
    )

    assert not _is_admissible(HydrogenCompletionTransformer, missing_heavy_domain)
    assert _is_admissible(HydrogenCompletionTransformer, heavy_complete_domain)


def test_hydrogen_completion_domain_allows_prior_atom_coverage_adoption() -> None:
    """Hydrogen completion may rely on prior atom-coverage adoption memory."""

    domain = _workflow_action_domain(
        _sidechain_incomplete_structure(),
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
            )
        ),
        adopted_transformers=(
            HeavyAtomCompletionTransformer.from_completion_scope(
                scope=_whole_structure_residue_scope()
            ),
        ),
    )

    assert _is_admissible(HydrogenCompletionTransformer, domain)


def test_retained_non_polymer_hydrogen_completion_domain_is_admissible() -> None:
    """Retained non-polymer hydrogen completion should be admissible."""

    domain = _workflow_action_domain(
        _retained_non_polymer_hydrogen_structure(),
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
            )
        ),
        component_library=build_retained_non_polymer_component_library(),
    )

    assert not _is_admissible(HydrogenCompletionTransformer, domain)
    assert _is_admissible(RetainedNonPolymerHydrogenCompletionTransformer, domain)


def test_retained_non_polymer_hydrogen_domain_accepts_partial_template_surface() -> (
    None
):
    """Linked glycans should not require full component topology to add Hs."""

    domain = _workflow_action_domain(
        read_structure(
            resolve_fixture_path(
                EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[
                    "2h6o-glycan-problematic"
                ].output_path
            )
        ),
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
            )
        ),
    )

    assert any(
        fact.requires_hydrogen_completion()
        and fact.heavy_atom_topology_availability_state.value == "absent"
        for fact in domain.chemistry_readiness_facts.retained_non_polymer_facts
    )
    assert _is_admissible(RetainedNonPolymerHydrogenCompletionTransformer, domain)


def test_external_span_domain_requires_selected_gap_and_donor_context() -> None:
    """Donor span reconstruction should require both request and donor context."""

    transform_requests = WorkflowTransformRequests(
        external_span_reconstructions=(_internal_gap_reconstruction_spec(),),
    )
    blocked_domain = _workflow_action_domain(
        _gap_structure(),
        transform_requests=transform_requests,
        planning_context=WorkflowPlanningContext(
            span_donor_availability=WorkflowSpanDonorAvailability.NONE,
        ),
    )
    admissible_domain = _workflow_action_domain(
        _gap_structure(),
        transform_requests=transform_requests,
        planning_context=WorkflowPlanningContext(
            span_donor_availability=WorkflowSpanDonorAvailability.AVAILABLE,
        ),
    )

    assert not _is_admissible(ExternalSpanReconstructionTransformer, blocked_domain)
    assert _is_admissible(ExternalSpanReconstructionTransformer, admissible_domain)


def test_workflow_action_domain_exposes_cohesive_read_views() -> None:
    """Action-domain query families should be owned by focused read views."""

    domain = _workflow_action_domain(_heavy_complete_structure())

    assert domain.completion.requires_atom_completion() is False
    assert domain.explicit_repair.prerequisite_residue_ids() == ()
    assert domain.memory.has_prior_augmentation_adoption() is False
    assert domain.burden.has_intrinsic_geometry_burden() is False
    assert domain.span_reconstruction.allows_reconstruction() is False

    assert not hasattr(domain, "requires_atom_completion")
    assert not hasattr(domain, "explicit_repair_refinement_prerequisite_residue_ids")
    assert not hasattr(domain, "has_adopted_action_family")


def test_local_refinement_domain_can_use_prior_augmentation_memory() -> None:
    """Local refinement admissibility may come from prior augmentation memory."""

    residue_id = ResidueId("A", 1)
    domain = _workflow_action_domain(
        _heavy_complete_structure(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
        adopted_transformers=(
            HeavyAtomCompletionTransformer.from_completion_scope(
                scope=_whole_structure_residue_scope()
            ),
        ),
    )

    assert _is_admissible(LocalRefinementTransformer, domain)


def test_automatic_local_refinement_stays_blocked_after_prior_refinement() -> None:
    """Prior local refinement should block generic automatic re-entry."""

    residue_id = ResidueId("A", 1)
    adopted_refinement = LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues((residue_id,)),
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationForceField.UFF
            ),
        )
    )
    structure = _heavy_complete_structure()
    domain = _workflow_action_domain(
        structure,
        adopted_transformers=(adopted_refinement,),
    )

    assert not _is_admissible(LocalRefinementTransformer, domain)


def test_explicit_repair_refinement_requires_local_atom_completion() -> None:
    """Explicit repair should make local heavy completion admissible without goals."""

    residue_id = ResidueId("A", 1)
    domain = _workflow_action_domain(
        _sidechain_incomplete_structure(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    assert domain.explicit_repair.atom_completion_residue_ids() == (
        residue_id,
    )
    assert _is_admissible(HeavyAtomCompletionTransformer, domain)
    assert not _is_admissible(HydrogenCompletionTransformer, domain)


def test_explicit_repair_refinement_requires_local_hydrogen_completion() -> None:
    """Explicit repair should make local hydrogen completion admissible."""

    residue_id = ResidueId("A", 1)
    domain = _workflow_action_domain(
        _heavy_complete_structure(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    assert domain.explicit_repair.hydrogen_missing_residue_ids() == (
        residue_id,
    )
    assert _is_admissible(HydrogenCompletionTransformer, domain)


def _workflow_action_domain(
    structure,
    *,
    requested_goals: RequestedGoalSet | None = None,
    transform_requests: WorkflowTransformRequests | None = None,
    planning_context: WorkflowPlanningContext | None = None,
    component_library=None,
    adopted_transformers=(),
) -> WorkflowActionDomain:
    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    active_requested_goals = (
        RequestedGoalSet() if requested_goals is None else requested_goals
    )
    active_transform_requests = (
        WorkflowTransformRequests()
        if transform_requests is None
        else transform_requests
    )
    active_planning_context = (
        WorkflowPlanningContext()
        if planning_context is None
        else planning_context
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=active_component_library,
        )
    )
    state_deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        requested_goals=active_requested_goals,
        planning_context=active_planning_context,
    )
    explicit_repair_refinement_execution_projection = None
    if active_transform_requests.repair_refinement is not None:
        explicit_repair_refinement_execution_projection = (
            LocalContinuousExecutionResidueProjection.from_scope_spec(
                ProteinStructureSnapshot.from_structure(structure),
                active_transform_requests.repair_refinement.resolved_execution_scope_spec(),
                context_radius_angstrom=(
                    active_transform_requests.repair_refinement.config.context_radius_angstrom
                ),
                component_library=active_component_library,
            )
        )
    return WorkflowActionDomain(
        structure=structure,
        requested_goals=active_requested_goals,
        transform_requests=active_transform_requests,
        component_library=active_component_library,
        planning_context=active_planning_context,
        state_deficit=state_deficit,
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        boundary_facts=StructureBoundaryStateFacts.from_structure(structure),
        explicit_repair_refinement_execution_projection=(
            explicit_repair_refinement_execution_projection
        ),
        adopted_actions=tuple(
            adoption
            for adopted_transformer in adopted_transformers
            if (
                adoption := action_adoption_for_registry(
                    adopted_transformer,
                    registry=WORKFLOW_ACTION_REGISTRY,
                )
            )
            is not None
        ),
    )


def _is_admissible(action_type, domain: WorkflowActionDomain) -> bool:
    entry = action_registry_entry_for_type(
        action_type,
        registry=WORKFLOW_ACTION_REGISTRY,
    )
    assert entry is not None
    return entry.is_admissible(domain)


def _sidechain_incomplete_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-action-domain-sidechain-incomplete",
    )


def _heavy_complete_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-action-domain-heavy-complete",
    )


def _retained_non_polymer_hydrogen_structure():
    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-action-domain-retained-non-polymer-hydrogen",
    )


def _gap_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 4),
                        atoms=(
                            atom_payload("N", "N", Vec3(4.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(5.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(6.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(7.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-action-domain-gap",
        polymer_blueprint=PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id="A",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="D"),
                        PolymerResidueSlot(sequence_position=3, token="E"),
                        PolymerResidueSlot(sequence_position=4, token="G"),
                    ),
                ),
            )
        ),
    )



def _internal_gap_reconstruction_spec() -> ExternalSpanReconstructionSpec:
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("X", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="GLU",
                        residue_id=ResidueId("X", 3),
                        atoms=(
                            atom_payload("N", "N", Vec3(4.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(5.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(6.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(7.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(5.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-action-domain-donor",
    )
    return ExternalSpanReconstructionSpec(
        blueprint_coverage_gap=StructureBlueprintCoverageGap(
            structure_chain_id="A",
            blueprint_chain_id="A",
            absent_sequence_positions=(2, 3),
            preceding_residue_id=ResidueId("A", 1),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 2), ResidueId("X", 3)),
    )


def _whole_structure_residue_scope() -> ResidueSetScope:
    return ResidueSetScope(residue_ids=(ResidueId("A", 1),))
