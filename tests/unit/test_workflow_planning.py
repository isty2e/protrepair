"""Unit tests for generic workflow planning outcomes."""

from functools import lru_cache
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.correction_state_fixtures import load_refinement_fixture
from tests.support.refinement_benchmarks import (
    load_case_structure,
    resolve_fixture_path,
)
from tests.support.refinement_cases import (
    EXPLORATORY_REFINEMENT_FIXTURE_SOURCES,
    REFINEMENT_BENCHMARK_CASES,
)
from tests.support.refinement_corpus import ALL_CORPUS_METADATA
from tests.support.request_builders import (
    c_terminal_oxt_requested_goals,
    whole_structure_requested_goals,
)
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)
from tests.support.whole_structure_sources import WHOLE_STRUCTURE_CORPUS_SOURCES

import protrepair.workflow.planning.planner as workflow_planning
from protrepair.chemistry import (
    ComponentLibrary,
    build_default_component_library,
)
from protrepair.diagnostics.parser_readability import RDKitProximityBondCluster
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.relation.blueprint import StructureBlueprintCoverageGap
from protrepair.scope import (
    AtomSetScope,
    ResidueBoundaryScope,
    ResidueBoundarySide,
    ResidueSetScope,
    WholeStructureScope,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    ClashPresenceState,
    HydrogenCoverageState,
    OxtPresenceState,
    ParserCompatibilityProfile,
    ParserCompatibilityState,
    SidechainHeavyAtomCompletenessState,
    StructureParserCompatibilityFacts,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.packing import PackingScope, PackingSpec
from protrepair.transformer.refinement.parser_witness import (
    ParserWitnessRepairBudget,
    ParserWitnessRepairCandidate,
)
from protrepair.transformer.refinement.spec import (
    BackboneWindowRefinementSpec,
    RefinementOperatorFamily,
    RepairRefinementSpec,
)
from protrepair.workflow.actions.backbone_window_refinement import (
    BackboneWindowRefinementTransformer,
)
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.actions.heavy_completion import (
    HeavyAtomCompletionTransformer,
)
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import (
    CommittedPackingTransformer,
)
from protrepair.workflow.actions.retained_non_polymer_hydrogen_completion import (
    RetainedNonPolymerHydrogenCompletionTransformer,
)
from protrepair.workflow.actions.stereochemistry import (
    StereochemistryCorrectionTransformer,
)
from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    LigandPolicy,
    RequestedGoalSet,
    StructureIngressOptions,
    WorkflowLigandContextMode,
    WorkflowPlanningContext,
    WorkflowSpanDonorAvailability,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.engine import normalize_source_structure
from protrepair.workflow.planning import (
    local_refinement_policy as workflow_local_refinement_policy,
)
from protrepair.workflow.planning.capability import WorkflowCapabilityDeficitFamily
from protrepair.workflow.planning.context_projection import (
    planning_context_is_holo_for_structure,
)
from protrepair.workflow.planning.planner import (
    BlockedRequestedGoal,
    WorkflowPlannerMemory,
    plan_workflow_actions,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required dependency import guard
    Chem = None

RDKIT_AVAILABLE = Chem is not None


@lru_cache(maxsize=1)
def _hydrogenated_1afc_structure() -> ProteinStructure:
    """Return the canonical hydrogenated 1AFC representative structure."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    return add_hydrogens(
        structure,
        component_library=build_default_component_library(),
        protonate_histidines=True,
    ).structure



def test_plan_workflow_actions_emits_heavy_action_before_hydrogen() -> None:
    """Hydrogen requests on heavy-incomplete input should first plan heavy repair."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-heavy-then-hydrogen",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        HeavyAtomCompletionTransformer,
    )


def test_plan_workflow_actions_emits_hydrogen_after_heavy_adoption() -> None:
    """Hydrogen completion should become the next proposal after heavy adoption."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-hydrogen-after-heavy",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(
                HeavyAtomCompletionTransformer(
                    scope=ResidueSetScope(residue_ids=(residue_id,))
                ),
            )
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        HydrogenCompletionTransformer,
    )


def test_plan_workflow_actions_keeps_parser_witness_ahead_of_global_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser-witness repair should not wait on global geometry diagnostics."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-parser-witness-before-geometry",
    )
    hydrogenated_structure = add_hydrogens(
        structure,
        component_library=build_default_component_library(),
        protonate_histidines=False,
    ).structure
    parser_candidate = ParserWitnessRepairCandidate(
        cluster=RDKitProximityBondCluster(
            residue_ids=(residue_id,),
            bonds=(),
        ),
        repair_refinement=RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues((residue_id,)),
            binding=RecommendedContinuousRelaxationBinding(),
            config=ContinuousRelaxationConfig(max_iterations=20),
        ),
        budget=ParserWitnessRepairBudget(max_iterations=20),
    )

    def fake_parser_witness_repair_candidates(
        current_structure: ProteinStructure,
        *,
        component_library,
        budget: ParserWitnessRepairBudget | None = None,
    ) -> tuple[ParserWitnessRepairCandidate, ...]:
        del current_structure
        del component_library
        del budget
        return (parser_candidate,)

    def fail_detect_heavy_geometry(*args, **kwargs):
        del args
        del kwargs
        raise AssertionError("global geometry should not run before parser witness")

    monkeypatch.setattr(
        "protrepair.workflow.planning.local_refinement_policy.parser_witness_repair_candidates",
        fake_parser_witness_repair_candidates,
    )
    monkeypatch.setattr(
        "protrepair.workflow.planning.local_refinement_policy.detect_heavy_geometry",
        fail_detect_heavy_geometry,
    )
    _patch_parser_compatibility_facts(
        monkeypatch,
        ParserCompatibilityState.INCOMPATIBLE,
        extra_proximity_bond_count=1,
        extra_heavy_proximity_bond_count=1,
    )

    outcome = plan_workflow_actions(
        hydrogenated_structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_default_component_library(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], LocalRefinementTransformer)
    assert (
        outcome.transformers[0].repair_refinement
        == parser_candidate.repair_refinement
    )


def test_plan_workflow_actions_skips_parser_witness_when_parser_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed parser compatibility should stop parser-driven repair probing."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-parser-compatible",
    )
    hydrogenated_structure = add_hydrogens(
        structure,
        component_library=build_default_component_library(),
        protonate_histidines=False,
    ).structure

    def fail_parser_witness_repair_candidates(
        current_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
        budget: ParserWitnessRepairBudget | None = None,
    ) -> tuple[ParserWitnessRepairCandidate, ...]:
        del current_structure
        del component_library
        del budget
        raise AssertionError("parser witnesses should require parser burden")

    def fail_parser_failing_residue_ids(
        current_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
    ) -> tuple[ResidueId, ...]:
        del current_structure
        del component_library
        raise AssertionError("parser fallback should require parser burden")

    _patch_parser_compatibility_facts(
        monkeypatch,
        ParserCompatibilityState.COMPATIBLE,
    )
    monkeypatch.setattr(
        workflow_local_refinement_policy,
        "parser_witness_repair_candidates",
        fail_parser_witness_repair_candidates,
    )
    monkeypatch.setattr(
        workflow_local_refinement_policy,
        "rdkit_no_conect_parser_failing_residue_ids",
        fail_parser_failing_residue_ids,
    )

    outcome = plan_workflow_actions(
        hydrogenated_structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_default_component_library(),
    )

    assert not any(
        isinstance(transformer, LocalRefinementTransformer)
        for transformer in outcome.transformers
    )


def test_plan_workflow_actions_emits_retained_non_polymer_hydrogen_completion() -> None:
    """Whole-structure hydrogen goals should propose retained non-polymer completion."""

    structure = build_structure(
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
        source_name="workflow-planning-retained-non-polymer-hydrogen",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_retained_non_polymer_component_library(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        RetainedNonPolymerHydrogenCompletionTransformer,
    )


def test_plan_workflow_actions_emits_retained_glycan_h_for_partial_template_surface(
) -> None:
    """Linked glycan hydrogen planning should not require full component topology."""

    component_library = build_default_component_library()
    structure = add_hydrogens(
        read_structure(
            resolve_fixture_path(
                EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[
                    "2h6o-glycan-problematic"
                ].output_path
            )
        ),
        component_library=component_library,
    ).structure

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=component_library,
        )
    )
    assert any(
        fact.requires_hydrogen_completion()
        and fact.heavy_atom_topology_availability_state.value == "absent"
        for fact in chemistry_readiness_facts.retained_non_polymer_facts
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(),
        component_library=component_library,
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        RetainedNonPolymerHydrogenCompletionTransformer,
    )


def test_plan_workflow_actions_prefers_stereochemistry_correction_before_refinement(
) -> None:
    """Stereo-invalid correction should be planned before local refinement."""

    structure = _inverted_threonine_workflow_structure()

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 30),)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
        component_library=build_default_component_library(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], StereochemistryCorrectionTransformer)
    assert outcome.transformers[0].scope.residue_ids == (ResidueId("A", 30),)


def test_plan_workflow_actions_explicit_repair_requires_local_heavy_then_hydrogen(
) -> None:
    """Explicit repair should stage local heavy/H prerequisites before refinement."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.5, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.9, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.9, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-explicit-local-prereqs",
    )
    repair_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        binding=ManualContinuousRelaxationBinding(
            ContinuousRelaxationForceField.UFF
        ),
    )

    initial_outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=repair_refinement,
        ),
    )

    assert len(initial_outcome.transformers) == 1
    assert isinstance(initial_outcome.transformers[0], HeavyAtomCompletionTransformer)
    assert initial_outcome.transformers[0].scope.residue_ids == (residue_id,)

    hydrogen_outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=repair_refinement,
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(initial_outcome.transformers[0],)
        ),
    )

    assert len(hydrogen_outcome.transformers) == 1
    assert isinstance(hydrogen_outcome.transformers[0], HydrogenCompletionTransformer)
    assert hydrogen_outcome.transformers[0].scope.residue_ids == (residue_id,)


def test_plan_workflow_actions_emits_override_backed_retained_non_polymer_h_completion(
) -> None:
    """Override-backed retained non-polymers should plan hydrogen completion."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-retained-non-polymer-override",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_retained_non_polymer_component_library(),
        retained_non_polymer_chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=ResidueId("L", 1),
                smiles="CO",
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        RetainedNonPolymerHydrogenCompletionTransformer,
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_plan_workflow_actions_emits_rdkit_fallback_retained_non_polymer_h_completion(
) -> None:
    """Fallback-hydrogenatable retained non-polymers should plan completion."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-retained-non-polymer-fallback",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_retained_non_polymer_component_library(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        RetainedNonPolymerHydrogenCompletionTransformer,
    )


def test_workflow_planning_context_projects_holo_from_kept_ligands() -> None:
    """Planning context should keep apo/holo projection in the boundary layer."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="FAD",
                residue_id=ResidueId("A", 1001),
                atoms=(atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-context-holo",
    )

    assert (
        planning_context_is_holo_for_structure(
            WorkflowPlanningContext(),
            structure,
        )
        is False
    )
    assert (
        planning_context_is_holo_for_structure(
            WorkflowPlanningContext(
                ligand_context_mode=(
                    WorkflowLigandContextMode.CONSIDER_IF_PRESENT
                )
            ),
            structure,
        )
        is True
    )


def test_workflow_planning_context_validates_speculative_node_limit() -> None:
    """Planning context should reject invalid speculative execution limits."""

    assert WorkflowPlanningContext(max_speculative_nodes=1).max_speculative_nodes == 1

    with pytest.raises(TypeError, match="max_speculative_nodes"):
        WorkflowPlanningContext(max_speculative_nodes=True)

    with pytest.raises(ValueError, match="positive"):
        WorkflowPlanningContext(max_speculative_nodes=0)


def test_plan_workflow_actions_rejects_external_spans_without_donor_context() -> None:
    """Explicit external spans should require donor availability in context."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                    _build_residue("GLY", "A", 4, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-missing-donor-context",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ASP", "X", 2, ("N", "CA", "C", "O", "CB")),
                    _build_residue("GLU", "X", 3, ("N", "CA", "C", "O", "CB", "CG")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-donor",
    )

    with pytest.raises(ValueError, match="span-donor availability"):
        plan_workflow_actions(
            structure,
            requested_goals=RequestedGoalSet(),
            transform_requests=WorkflowTransformRequests(
                external_span_reconstructions=(
                    ExternalSpanReconstructionSpec(
                        blueprint_coverage_gap=StructureBlueprintCoverageGap(
                            structure_chain_id="A",
                            blueprint_chain_id="A",
                            absent_sequence_positions=(2, 3),
                            preceding_residue_id=ResidueId("A", 1),
                            following_residue_id=ResidueId("A", 4),
                        ),
                        donor_structure=donor_structure,
                        donor_residue_ids=(ResidueId("X", 2), ResidueId("X", 3)),
                    ),
                ),
            ),
            planning_context=WorkflowPlanningContext(
                span_donor_availability=WorkflowSpanDonorAvailability.NONE
            ),
        )


def test_workflow_transformers_reject_invalid_family_payloads() -> None:
    """Concrete workflow transformers should validate their own family payloads."""

    residue_id = ResidueId(chain_id="A", seq_num=1)

    with pytest.raises(TypeError, match="RepairRefinementSpec"):
        LocalRefinementTransformer(
            scope=ResidueSetScope(residue_ids=(residue_id,)),
            repair_refinement="invalid-payload",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="PackingSpec"):
        CommittedPackingTransformer(
            scope=ResidueSetScope(residue_ids=(residue_id,)),
            packing_spec="invalid-payload",  # type: ignore[arg-type]
        )


def test_plan_workflow_actions_ranks_heavy_before_explicit_packing() -> None:
    """Explicit packing should not outrank required atom-coverage burden."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-committed-packing",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(
                SidechainHeavyAtomCompletenessState.COMPLETE
            )
        ),
        transform_requests=WorkflowTransformRequests(
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
            )
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], HeavyAtomCompletionTransformer)


def test_plan_workflow_actions_emits_committed_packing_after_heavy_adoption() -> None:
    """Explicit packing may surface once atom-coverage augmentation was adopted."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-committed-packing-after-heavy",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
            )
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(
                HeavyAtomCompletionTransformer(
                    scope=ResidueSetScope(residue_ids=(residue_id,))
                ),
            )
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], CommittedPackingTransformer)
    assert outcome.transformers[0].scope == ResidueSetScope(residue_ids=(residue_id,))
    assert isinstance(outcome.transformers[0].scope, ResidueSetScope)
    assert outcome.transformers[0].packing_spec.mutable_residue_ids == (residue_id,)
    assert outcome.transformers[0].packing_spec.scope is PackingScope.LOCAL


def test_plan_workflow_actions_ranks_span_reconstruction_before_atom_completion() -> (
    None
):
    """Donor-backed span repair should outrank same-turn atom completion."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),
                    _build_residue("GLU", "A", 4, ("N", "CA", "C", "O", "CB", "CG")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-span-before-heavy",
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
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ASP", "X", 2, ("N", "CA", "C", "O", "CB")),
                    _build_residue("GLU", "X", 3, ("N", "CA", "C", "O", "CB", "CG")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-span-donor",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(
                SidechainHeavyAtomCompletenessState.COMPLETE
            )
        ),
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    blueprint_coverage_gap=StructureBlueprintCoverageGap(
                        structure_chain_id="A",
                        blueprint_chain_id="A",
                        absent_sequence_positions=(2, 3),
                        preceding_residue_id=ResidueId("A", 1),
                        following_residue_id=ResidueId("A", 4),
                    ),
                    donor_structure=donor_structure,
                    donor_residue_ids=(ResidueId("X", 2), ResidueId("X", 3)),
                ),
            ),
        ),
        planning_context=WorkflowPlanningContext(
            span_donor_availability=WorkflowSpanDonorAvailability.AVAILABLE
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], ExternalSpanReconstructionTransformer)


def test_plan_workflow_actions_ranks_hydrogen_before_boundary_goal_only() -> None:
    """General chemistry readiness should outrank boundary-only augmentation."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-hydrogen-before-boundary",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            goals=(
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
                requested_process_goal(
                    scope=ResidueBoundaryScope(
                        residue_id=residue_id,
                        side=ResidueBoundarySide.C_TERMINUS,
                    ),
                    value=OxtPresenceState.PRESENT,
                ),
            )
        ),
        transform_requests=WorkflowTransformRequests(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], HydrogenCompletionTransformer)


def test_plan_workflow_actions_ranks_local_refinement_before_packing() -> None:
    """Intrinsic revision should prefer local refinement once H prerequisites clear."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-refinement-before-packing",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
            ),
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            ),
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(
                HeavyAtomCompletionTransformer(
                    scope=ResidueSetScope(residue_ids=(residue_id,))
                ),
                HydrogenCompletionTransformer(
                    scope=ResidueSetScope(residue_ids=(residue_id,))
                ),
            )
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], LocalRefinementTransformer)


def test_plan_workflow_actions_skips_committed_packing_for_hydrogen_only_gaps() -> None:
    """Committed packing should not target residues missing only hydrogens."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-committed-packing-hydrogen-only",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
            )
        ),
    )

    assert outcome.transformers == ()


def test_plan_workflow_actions_does_not_repropose_adopted_committed_packing() -> None:
    """Committed packing should not loop once the same proposal was adopted."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    adopted_packing_transformer = CommittedPackingTransformer(
        scope=ResidueSetScope(residue_ids=(residue_id,)),
        packing_spec=PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(residue_id,),
        ),
    )
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-adopted-committed-packing",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
            )
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(adopted_packing_transformer,)
        ),
    )

    assert outcome.transformers == ()


def test_plan_workflow_actions_emits_local_refinement_after_completion_adoption() -> (
    None
):
    """Intrinsic correction burden should explicitly plan repair refinement."""

    residue_id = ResidueId(chain_id="A", seq_num=182)
    structure = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])

    hydrogen_outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    assert len(hydrogen_outcome.transformers) == 1
    assert isinstance(hydrogen_outcome.transformers[0], HydrogenCompletionTransformer)

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(hydrogen_outcome.transformers[0],)
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        LocalRefinementTransformer,
    )
    assert outcome.transformers[0].scope == ResidueSetScope(residue_ids=(residue_id,))


def test_plan_workflow_actions_emits_atom_set_local_refinement_scope() -> None:
    """Atomwise refinement selections should lower into atom-set scopes."""

    residue_id = ResidueId(chain_id="A", seq_num=182)
    atom_ref = AtomRef(residue_id=residue_id, atom_name="CB")
    structure = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])

    hydrogen_outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_atoms((atom_ref,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    assert len(hydrogen_outcome.transformers) == 1
    assert isinstance(hydrogen_outcome.transformers[0], HydrogenCompletionTransformer)

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_atoms((atom_ref,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(hydrogen_outcome.transformers[0],)
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        LocalRefinementTransformer,
    )
    assert outcome.transformers[0].scope == AtomSetScope(atom_refs=(atom_ref,))
    assert isinstance(outcome.transformers[0].scope, AtomSetScope)


def test_plan_workflow_actions_does_not_emit_standalone_local_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit repair stays dormant when no burden or prerequisite remains."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HB3",
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-no-standalone-refinement",
    )
    _patch_parser_compatibility_facts(
        monkeypatch,
        ParserCompatibilityState.COMPATIBLE,
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            ),
        ),
    )

    assert outcome.transformers == ()


def test_plan_workflow_actions_emits_local_refinement_after_coverage_adoption() -> None:
    """Explicit repair refinement should stage hydrogen after heavy adoption."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-post-coverage-refinement",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            ),
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(
                HeavyAtomCompletionTransformer(
                    scope=ResidueSetScope(residue_ids=(residue_id,))
                ),
            )
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], HydrogenCompletionTransformer)


def test_plan_workflow_actions_prioritizes_chemistry_before_interaction_correction(
) -> None:
    """Hydrogen augmentation should precede holo-aware correction planning."""

    residue_id = ResidueId(chain_id="A", seq_num=92)
    structure = load_refinement_fixture("1jd0_gln92_local.pdb")

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((residue_id,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
        planning_context=WorkflowPlanningContext(
            ligand_context_mode=WorkflowLigandContextMode.CONSIDER_IF_PRESENT,
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        HydrogenCompletionTransformer,
    )




def test_plan_workflow_actions_emits_terminal_augmentation_action() -> None:
    """Authentic terminal OXT requests should plan a terminal action."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-terminal-augmentation",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(c_terminal_oxt_requested_goals(residue_id)),
        transform_requests=WorkflowTransformRequests(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(
        outcome.transformers[0],
        TerminalAugmentationTransformer,
    )
    assert outcome.transformers[0].scope == ResidueSetScope(residue_ids=(residue_id,))


def test_plan_workflow_actions_filters_attempted_but_unadopted_action() -> None:
    """Attempted no-op branches should not re-propose the same exact action."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-attempted-terminal-augmentation",
    )
    terminal_transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(residue_id,)),
    )
    planner_memory = WorkflowPlannerMemory().with_attempted_transformer(
        terminal_transformer
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(c_terminal_oxt_requested_goals(residue_id)),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=planner_memory,
    )

    assert outcome.transformers == ()
    assert planner_memory.adopted_transformers == ()
    assert planner_memory.has_attempted_transformer(terminal_transformer)


def test_plan_workflow_actions_records_unsupported_scoped_requests() -> None:
    """Unsupported requested scopes should be surfaced through planning outcome."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-unsupported-scope",
    )
    requested_state = requested_process_goal(
        scope=ResidueSetScope(residue_ids=(residue_id,)),
        value=HydrogenCoverageState.COMPLETE,
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet((requested_state,)),
        transform_requests=WorkflowTransformRequests(),
    )

    assert outcome.unsupported_requested_goals == (requested_state,)
    assert outcome.is_terminal()
    assert outcome.current_proposal_batch() == ()


def test_plan_workflow_actions_records_unsupported_axis_requests() -> None:
    """Unsupported requested axes should also be surfaced through planning."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-unsupported-axis",
    )
    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=ClashPresenceState.NONE,
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet((requested_state,)),
        transform_requests=WorkflowTransformRequests(),
    )

    assert outcome.unsupported_requested_goals == (requested_state,)
    assert outcome.is_terminal()
    assert outcome.current_proposal_batch() == ()


def test_plan_workflow_actions_records_blocked_completion_requests() -> None:
    """Whole-structure completion requests should record blockers when stopped."""

    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("UNK", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-blocked-request",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet((requested_state,)),
        transform_requests=WorkflowTransformRequests(),
    )

    assert len(outcome.blocked_requested_goals) == 1
    blocked_requested_goal = outcome.blocked_requested_goals[0]
    assert isinstance(blocked_requested_goal, BlockedRequestedGoal)
    assert blocked_requested_goal.requested_goal == requested_state
    assert len(blocked_requested_goal.blockers) == 1
    assert (
        blocked_requested_goal.blockers[0].deficit_family
        is WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS
    )
    assert blocked_requested_goal.blocking_scopes() == (
        ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
    )


def _build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
) -> CanonicalResiduePayload:
    """Return one toy residue with stable atom ordering."""

    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=tuple(
            atom_payload(
                name=atom_name,
                element=_element_for_atom_name(atom_name),
                position=Vec3(float(index), 0.0, 0.0),
            )
            for index, atom_name in enumerate(atom_names)
        ),
    )


def _element_for_atom_name(atom_name: str) -> str:
    """Return one coarse element token for toy atom construction."""

    if atom_name.startswith("O"):
        return "O"
    if atom_name.startswith("N"):
        return "N"
    if atom_name.startswith("H"):
        return "H"
    return "C"


def _patch_parser_compatibility_facts(
    monkeypatch: pytest.MonkeyPatch,
    compatibility_state: ParserCompatibilityState,
    *,
    extra_proximity_bond_count: int = 0,
    extra_heavy_proximity_bond_count: int = 0,
) -> None:
    """Patch planner-visible parser compatibility facts for toy structures."""

    def fake_parser_compatibility_facts_from_structure(
        cls: type[StructureParserCompatibilityFacts],
        current_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> StructureParserCompatibilityFacts:
        del cls
        del component_library
        return StructureParserCompatibilityFacts(
            carrier=current_structure,
            profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
            compatibility_state=compatibility_state,
            extra_proximity_bond_count=extra_proximity_bond_count,
            extra_heavy_proximity_bond_count=extra_heavy_proximity_bond_count,
        )

    monkeypatch.setattr(
        workflow_planning.StructureParserCompatibilityFacts,
        "from_structure",
        classmethod(fake_parser_compatibility_facts_from_structure),
    )


def test_plan_workflow_actions_auto_emits_joint_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrogen-complete parser-visible severe clusters can trigger joint repair."""

    monkeypatch.setattr(
        workflow_local_refinement_policy,
        "parser_witness_repair_candidates",
        lambda structure, *, component_library: (),
    )

    outcome = plan_workflow_actions(
        _hydrogenated_1afc_structure(),
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )

    joint_transformers = tuple(
        transformer
        for transformer in outcome.transformers
        if isinstance(transformer, LocalRefinementTransformer)
    )

    assert len(joint_transformers) == 1
    assert isinstance(joint_transformers[0].scope, ResidueSetScope)
    assert len(joint_transformers[0].scope.residue_ids) >= 2
    assert isinstance(
        joint_transformers[0].repair_refinement.binding,
        RecommendedContinuousRelaxationBinding,
    )


def test_plan_workflow_actions_does_not_auto_sweep_joint_refinement_without_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic joint repair should not sweep generic geometry alone."""

    monkeypatch.setattr(
        workflow_local_refinement_policy,
        "parser_witness_repair_candidates",
        lambda structure, *, component_library: (),
    )
    monkeypatch.setattr(
        workflow_local_refinement_policy,
        "rdkit_no_conect_parser_failing_residue_ids",
        lambda structure, *, component_library: (),
    )

    def fail_joint_scope_proposals(*args, **kwargs):
        raise AssertionError("joint correction should require parser-visible focus")

    monkeypatch.setattr(
        workflow_local_refinement_policy,
        "propose_joint_correction_scopes",
        fail_joint_scope_proposals,
    )

    outcome = plan_workflow_actions(
        _hydrogenated_1afc_structure(),
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )

    assert not any(
        isinstance(transformer, LocalRefinementTransformer)
        for transformer in outcome.transformers
    )


def test_plan_workflow_actions_auto_emits_intrinsic_geometry_refinement() -> None:
    """Hydrogen-complete severe intrinsic geometry should trigger local repair."""

    structure = add_hydrogens(
        build_structure(
            chains=(
                chain_payload(
                    "D",
                    (
                        residue_payload(
                            component_id="LEU",
                            residue_id=ResidueId("D", 32),
                            atoms=(
                                atom_payload("N", "N", Vec3(-4.300, 6.200, 18.600)),
                                atom_payload("CA", "C", Vec3(-3.600, 5.000, 19.100)),
                                atom_payload("C", "C", Vec3(-2.400, 5.300, 20.000)),
                                atom_payload("O", "O", Vec3(-1.400, 4.700, 19.800)),
                                atom_payload("CB", "C", Vec3(-3.474, 4.042, 20.144)),
                                atom_payload("CG", "C", Vec3(-4.009, 2.598, 20.202)),
                                atom_payload("CD1", "C", Vec3(-5.389, 2.549, 20.596)),
                                atom_payload("CD2", "C", Vec3(-4.341, 2.566, 18.974)),
                            ),
                        ),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="workflow-planning-intrinsic-geometry-refinement",
        ),
        component_library=build_default_component_library(),
    ).structure

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )

    intrinsic_transformers = tuple(
        transformer
        for transformer in outcome.transformers
        if isinstance(transformer, LocalRefinementTransformer)
        and transformer.scope == ResidueSetScope(residue_ids=(ResidueId("D", 32),))
    )

    assert len(intrinsic_transformers) == 1
    assert intrinsic_transformers[0].repair_refinement.scope_spec == (
        LocalScopeSpec.from_residues((ResidueId("D", 32),))
    )
    assert isinstance(
        intrinsic_transformers[0].repair_refinement.binding,
        RecommendedContinuousRelaxationBinding,
    )


def test_plan_workflow_actions_auto_targets_2qls_leu32_intrinsic_geometry() -> None:
    """2QLS chain D should route automatic intrinsic repair onto LEU D:32."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["2qls-whole-structure"]
    component_library = build_default_component_library()
    structure = normalize_source_structure(
        source.output_path,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=("D",),
    )
    structure = add_hydrogens(
        structure,
        component_library=component_library,
    ).structure
    structure = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
    ).structure

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        component_library=component_library,
    )

    intrinsic_transformers = tuple(
        transformer
        for transformer in outcome.transformers
        if isinstance(transformer, LocalRefinementTransformer)
        and transformer.scope == ResidueSetScope(residue_ids=(ResidueId("D", 32),))
    )

    assert len(intrinsic_transformers) == 1


def test_plan_workflow_actions_blocks_auto_joint_refinement_after_adoption() -> None:
    """Automatic joint refinement should be an opportunistic bounded batch."""

    adopted_refinement = LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residue_sidechains(
                (ResidueId("C", 11), ResidueId("C", 45))
            ),
            binding=RecommendedContinuousRelaxationBinding(),
        )
    )

    outcome = plan_workflow_actions(
        _hydrogenated_1afc_structure(),
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(adopted_refinement,)
        ),
    )

    assert not any(
        isinstance(transformer, LocalRefinementTransformer)
        for transformer in outcome.transformers
    )


def test_plan_workflow_actions_promotes_backbone_joint_refinement_to_residue_scope(
) -> None:
    """Backbone-local severe contacts should plan residue-local refinement first."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(-1.2, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(-0.2, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.2, 1.5, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("B", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(2.0, 1.5, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-backbone-joint-refinement",
    )

    hydrogen_outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    assert len(hydrogen_outcome.transformers) == 1
    assert isinstance(hydrogen_outcome.transformers[0], HydrogenCompletionTransformer)

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(hydrogen_outcome.transformers[0],)
        ),
    )

    assert outcome.transformers
    assert isinstance(outcome.transformers[0], LocalRefinementTransformer)
    assert outcome.transformers[0].repair_refinement.scope_spec.lowering is (
        LocalScopeSpec.from_residues((ResidueId("A", 1),)).lowering
    )
    assert outcome.transformers[0].scope == ResidueSetScope(
        residue_ids=(ResidueId("A", 1), ResidueId("B", 2))
    )


def test_plan_workflow_actions_keeps_backbone_joint_scope_semantic_but_widens_execution(
) -> None:
    """Backbone joint refinement should widen execution without widening scope."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(-6.8, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(-5.8, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(-4.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(-4.0, 0.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(-1.2, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(-0.2, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.2, 1.5, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 3),
                        atoms=(
                            atom_payload("N", "N", Vec3(8.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(9.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(10.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(10.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(9.0, 1.5, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 4),
                        atoms=(
                            atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(2.0, 1.5, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-backbone-joint-execution-window",
    )

    hydrogen_outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 2),)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    assert len(hydrogen_outcome.transformers) == 1
    assert isinstance(hydrogen_outcome.transformers[0], HydrogenCompletionTransformer)

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 2),)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(hydrogen_outcome.transformers[0],)
        ),
    )

    assert outcome.transformers
    assert isinstance(outcome.transformers[0], LocalRefinementTransformer)
    assert outcome.transformers[0].scope == ResidueSetScope(
        residue_ids=(ResidueId("A", 2), ResidueId("A", 4))
    )
    assert (
        outcome.transformers[0].repair_refinement.execution_scope_spec
        == LocalScopeSpec.from_residues(
            (
                ResidueId("A", 1),
                ResidueId("A", 2),
                ResidueId("A", 3),
                ResidueId("A", 4),
            )
        )
    )


def test_plan_workflow_actions_routes_backbone_window_operator_deficit() -> None:
    """Backbone-window operator deficits should not masquerade as local FF."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue(
                        "ALA",
                        "A",
                        10,
                        ("N", "CA", "C", "O", "CB"),
                    ),
                    _build_residue("GLY", "A", 11, ("N", "CA", "C", "O")),
                    _build_residue(
                        "SER",
                        "A",
                        12,
                        ("N", "CA", "C", "O", "CB", "OG"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-backbone-window-operator",
    )
    window_spec = BackboneWindowRefinementSpec(
        residue_ids=(
            ResidueId("A", 10),
            ResidueId("A", 11),
            ResidueId("A", 12),
        )
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            backbone_window_refinements=(window_spec,)
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], BackboneWindowRefinementTransformer)
    assert outcome.transformers[0].window_spec == window_spec
    assert not any(
        isinstance(transformer, LocalRefinementTransformer)
        for transformer in outcome.transformers
    )
    assert outcome.state_deficit is not None
    assert len(outcome.state_deficit.backbone_window_operator) == 1
    assert (
        outcome.state_deficit.backbone_window_operator[0].window_spec
        == window_spec
    )
    assert (
        outcome.state_deficit.backbone_window_operator[0].disposition.value
        == "required"
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "3j6b-terminal-helix-misthread",
        "3j9e-loop-backbone-error",
        "7s9d-prestin-segment",
    ),
)
def test_exploratory_backbone_window_metadata_routes_to_operator(
    case_id: str,
) -> None:
    """Corpus-declared backbone-window pressure should route to its operator."""

    source = EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[case_id]
    metadata = ALL_CORPUS_METADATA[case_id]
    component_library = build_default_component_library()
    structure = read_structure(
        resolve_fixture_path(source.output_path),
        policy=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )
    structure = add_hydrogens(
        structure,
        component_library=component_library,
    ).structure
    structure = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
    ).structure
    window_spec = BackboneWindowRefinementSpec(
        residue_ids=source.seed_residue_ids
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            backbone_window_refinements=(window_spec,)
        ),
        component_library=component_library,
    )

    assert (
        metadata.required_operator_family
        is RefinementOperatorFamily.BACKBONE_WINDOW_REFINEMENT
    )
    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], BackboneWindowRefinementTransformer)
    assert outcome.transformers[0].window_spec == window_spec


def _inverted_threonine_workflow_structure() -> ProteinStructure:
    """Return one focused THR structure with inverted side-chain chirality."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue_id = ResidueId("A", 30)
    residue_site = structure.constitution.chain("A").residue(residue_id)
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    inverted_structure = structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry.with_atom_geometries(
            (
                ("OG1", residue_geometry.atom_geometry("CG2")),
                ("CG2", residue_geometry.atom_geometry("OG1")),
            )
        ),
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    (
                        inverted_structure.constitution.chain("A").residue(residue_id),
                        inverted_structure.geometry.residue_geometry(
                            constitution=inverted_structure.constitution,
                            residue_index=inverted_structure.constitution.residue_index(
                                residue_id
                            ),
                        ),
                        inverted_structure.topology.residue_formal_charge_by_atom_name(
                            constitution=inverted_structure.constitution,
                            residue_index=inverted_structure.constitution.residue_index(
                                residue_id
                            ),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-planning-inverted-thr",
    )
