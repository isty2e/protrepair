"""Workflow-spine tests over path and canonical structure sources."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalChainPayload,
    CanonicalResiduePayload,
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.request_builders import (
    c_terminal_oxt_requested_goals,
    ingress_options,
    whole_structure_requested_goals,
)

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssue,
    ValidationIssueKind,
    detect_sidechain_stereochemistry,
)
from protrepair.diagnostics.parser_readability import (
    RDKitNoConectParserReadabilityProbe,
    RDKitNoConectSanitizeReadabilityMetrics,
    RDKitParserProblemWitness,
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.io import read_structure, write_structure_string
from protrepair.io.structure_ingress import (
    apply_structure_normalization_policy,
)
from protrepair.scope import (
    ResidueBoundaryScope,
    ResidueBoundarySide,
    ResidueSetScope,
    WholeStructureScope,
)
from protrepair.state import (
    HydrogenCoverageState,
    OxtPresenceState,
    StructureProjectionStateFacts,
)
from protrepair.structure import (
    ProteinStructure,
    StructureIngress,
    StructureProvenance,
)
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.packing import (
    PackingMode,
    PackingPlan,
    PackingResult,
    PackingScope,
    PackingSpec,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.parser_witness import (
    ParserWitnessRepairBudget,
    ParserWitnessRepairCandidate,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.refinement.speculative_planning import (
    SpeculativeAdoptedChild,
    SpeculativeAdoptionDecision,
)
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.heavy_completion import (
    HeavyAtomCompletionTransformer,
)
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
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
    DisabledHistidineProtonationRequest,
    LigandPolicy,
    OrphanFragmentPolicy,
    PrasRatioHistidineProtonationRequest,
    ProcessResult,
    RequestedGoalCompletionVerdict,
    RequestedGoalReport,
    RequestedGoalSet,
    RequestedGoalStatus,
    RetainedNonPolymerChemistryOverride,
    StructureIngressOptions,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.result import WorkflowBranchQualityScore
from protrepair.workflow.engine.finalization import (
    _final_parser_readability_issues,
)
from protrepair.workflow.engine.packing import (
    WorkflowPackingReference,
    prepare_workflow_packing_reference,
)
from protrepair.workflow.engine.reporting import evaluate_requested_goal_report
from protrepair.workflow.engine.runtime import (
    WorkflowRuntimeState,
    _workflow_children_with_regression_retention,
    _workflow_children_within_node_budget,
    execute_iterative_workflow,
)
from protrepair.workflow.planning.action.registry import WorkflowStateAction
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    WorkflowPlanningOutcome,
    plan_workflow_actions,
)
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required dependency import guard
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def _fake_parser_witness_clusters(
    residue_id: ResidueId,
    extra_bond_count: int,
) -> tuple[RDKitProximityBondCluster, ...]:
    if extra_bond_count <= 0:
        return ()

    witnesses = tuple(
        RDKitProximityBondWitness(
            atom_ref_1=AtomRef(residue_id, f"N{index}"),
            atom_ref_2=AtomRef(residue_id, f"CA{index}"),
            element_1="N",
            element_2="C",
            is_known_component_bond=False,
        )
        for index in range(extra_bond_count)
    )
    return (
        RDKitProximityBondCluster(
            residue_ids=(residue_id,),
            bonds=witnesses,
        ),
    )


def _fake_parser_probe_from_clusters(
    clusters: tuple[RDKitProximityBondCluster, ...],
) -> RDKitNoConectParserReadabilityProbe:
    if not clusters:
        return RDKitNoConectParserReadabilityProbe(
            sanitize_readable=True,
            residue_problem_witnesses=(),
            structure_messages=(),
        )

    return RDKitNoConectParserReadabilityProbe(
        sanitize_readable=False,
        residue_problem_witnesses=tuple(
            RDKitParserProblemWitness(
                residue_id=cluster.residue_ids[0],
                atom_names=("N", "CA"),
                messages=("fake parser problem",),
                proximity_bonds=cluster.bonds,
            )
            for cluster in clusters
        ),
        structure_messages=(),
    )


def test_process_structure_rejects_source_chain_selection_for_canonical_input() -> None:
    """Canonical inputs should reject raw-source chain slicing requests."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
            build_chain(
                "B",
                (build_residue("GLY", "B", 2, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(build_residue("FAD", "B", 99, ("C1", "N1", "O1"), is_hetero=True),),
        source_format=FileFormat.PDB,
        source_name="canonical-fixture",
    )

    with pytest.raises(
        ValueError,
        match="selected_source_chain_ids applies only to raw source inputs",
    ):
        process_structure(
            structure,
            ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
            selected_source_chain_ids=("B",),
        )


def test_process_structure_applies_source_chain_selection_to_raw_input(
    tmp_path: Path,
) -> None:
    """Raw source inputs should honor selected_source_chain_ids."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
            build_chain(
                "B",
                (build_residue("GLY", "B", 2, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(build_residue("FAD", "B", 99, ("C1", "N1", "O1"), is_hetero=True),),
        source_format=FileFormat.PDB,
        source_name="raw-fixture",
    )
    source_path = tmp_path / "source-selection.pdb"
    source_path.write_text(
        write_structure_string(structure, FileFormat.PDB),
        encoding="utf-8",
    )

    result = process_structure(
        source_path,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=("B",),
    )

    assert result.structure.chain_ids() == ("B",)
    assert ligand_component_ids(result.structure) == ("FAD",)


def test_process_structure_can_drop_ligands_for_canonical_input() -> None:
    """Canonical inputs should still honor ligand dropping."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(build_residue("FAD", "A", 99, ("C1", "N1", "O1"), is_hetero=True),),
        source_format=FileFormat.PDB,
        source_name="canonical-fixture",
    )

    result = process_structure(
        structure,
        ingress=ingress_options(ligand_policy=LigandPolicy.DROP),
    )

    assert result.structure.constitution.ligands == ()


def test_retained_non_polymer_hydrogen_transformer_uses_context_override() -> None:
    """Execution context should thread retained non-polymer overrides to repair."""

    structure = build_workflow_structure(
        chains=(),
        ligands=(
            build_residue(
                "UNK",
                "L",
                1,
                ("C1", "O1"),
                is_hetero=True,
                positions_by_name={
                    "C1": Vec3(4.0, 0.0, 0.0),
                    "O1": Vec3(5.4, 0.0, 0.0),
                },
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-retained-non-polymer-override",
    )
    transformer = RetainedNonPolymerHydrogenCompletionTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId("L", 1),)),
    )
    result = transformer.execute(
        ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        ),
        context=TransformerExecutionContext(
            component_library=build_default_component_library(),
            original_structure=structure,
            orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
            allow_retained_non_polymer_rdkit_fallback=False,
            retained_non_polymer_chemistry_evidence=(
                RetainedNonPolymerChemistryOverride(
                    residue_id=ResidueId("L", 1),
                    smiles="CO",
                    heavy_atom_names=("C1", "O1"),
                ).to_evidence(),
            ),
        ),
    )

    ligand = result.structure.constitution.ligands[0]
    hydrogen_atom_names = tuple(
        atom_site.name for atom_site in ligand.atom_sites if atom_site.element == "H"
    )
    assert hydrogen_atom_names == ("H001", "H002", "H003", "H004")


def test_process_structure_reports_strict_retained_non_polymer_fallback_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict public workflow should report fallback-blocked retained ligands."""

    ligand_residue_id = ResidueId("A", 99)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(
            build_residue(
                "UNK",
                "A",
                99,
                ("C1", "O1"),
                is_hetero=True,
                positions_by_name={
                    "C1": Vec3(4.0, 0.0, 0.0),
                    "O1": Vec3(5.4, 0.0, 0.0),
                },
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-strict-retained-non-polymer-fallback-blocked",
    )

    class FakePackingBackend:
        def pack(self, plan: PackingPlan) -> PackingResult:
            return PackingResult(
                packed_structure=plan.structure,
                changed_residue_ids=(),
                issues=(),
                backend_name=plan.spec.backend_name,
            )

    monkeypatch.setattr(
        "protrepair.transformer.packing.runtime.resolve_sidechain_packing_backend",
        lambda _backend_name: FakePackingBackend(),
    )

    result = process_structure(
        structure,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(),
                value=HydrogenCoverageState.COMPLETE,
            ),
        ),
        transform_requests=WorkflowTransformRequests(
            allow_retained_non_polymer_rdkit_fallback=False,
        ),
    )
    ligand = result.structure.constitution.residue_or_ligand(ligand_residue_id)
    assert ligand is not None

    assert all(atom_site.element != "H" for atom_site in ligand.atom_sites)
    assert any(
        issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_BLOCKED
        for issue in result.issues
    )
    assert not any(
        issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED
        for issue in result.issues
    )


def test_process_structure_rejects_invalid_retained_non_polymer_override_smiles() -> (
    None
):
    """Invalid public retained-ligand override SMILES should be an input error."""

    ligand_residue_id = ResidueId("A", 99)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(
            build_residue(
                "UNK",
                "A",
                99,
                ("C1", "O1"),
                is_hetero=True,
                positions_by_name={
                    "C1": Vec3(4.0, 0.0, 0.0),
                    "O1": Vec3(5.4, 0.0, 0.0),
                },
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-invalid-retained-non-polymer-override",
    )

    with pytest.raises(
        ValueError,
        match=(
            "invalid retained non-polymer chemistry override for A:99: "
            "SMILES evidence could not be parsed or projected"
        ),
    ):
        process_structure(
            structure,
            ingress=StructureIngressOptions(
                ligand_policy=LigandPolicy.KEEP,
                retained_non_polymer_chemistry_overrides=(
                    RetainedNonPolymerChemistryOverride(
                        residue_id=ligand_residue_id,
                        smiles="not_a_smiles",
                        heavy_atom_names=("C1", "O1"),
                    ),
                ),
            ),
        )


def test_process_structure_rejects_no_rdkit_retained_non_polymer_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit override validation should report stable broken-RDKit wording."""

    monkeypatch.setattr(
        "protrepair.chemistry.inference.retained_non_polymer_evidence.Chem",
        None,
    )
    ligand_residue_id = ResidueId("A", 99)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(
            build_residue(
                "UNK",
                "A",
                99,
                ("C1", "O1"),
                is_hetero=True,
                positions_by_name={
                    "C1": Vec3(4.0, 0.0, 0.0),
                    "O1": Vec3(5.4, 0.0, 0.0),
                },
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-no-rdkit-retained-non-polymer-override",
    )

    with pytest.raises(
        RdkitUnavailableError,
        match=(
            "retained non-polymer chemistry override validation requires "
            "the required RDKit backend for A:99"
        ),
    ):
        process_structure(
            structure,
            ingress=StructureIngressOptions(
                ligand_policy=LigandPolicy.KEEP,
                retained_non_polymer_chemistry_overrides=(
                    RetainedNonPolymerChemistryOverride(
                        residue_id=ligand_residue_id,
                        smiles="CO",
                        heavy_atom_names=("C1", "O1"),
                    ),
                ),
            ),
        )


def test_execute_iterative_workflow_threads_retained_non_polymer_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow runtime should carry strict retained-ligand policy to actions."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-retained-non-polymer-fallback-policy-threading",
    )
    transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
    )
    captured_policy_values: list[bool] = []
    planner_call_count = 0

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            current_structure,
            requested_goals,
            transform_requests,
            component_library,
            planner_memory,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        nonlocal planner_call_count
        planner_call_count += 1
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(structure)
            ),
            transformers=(transformer,) if planner_call_count == 1 else (),
        )

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del transformer
        captured_policy_values.append(
            execution_context.allow_retained_non_polymer_rdkit_fallback
        )
        return result

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    execute_iterative_workflow(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
                binding=RecommendedContinuousRelaxationBinding(),
            ),
            allow_retained_non_polymer_rdkit_fallback=False,
        ),
        component_library=build_default_component_library(),
        planning_context=WorkflowPlanningContext(),
        reference_structure=None,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
        histidine_protonation=DisabledHistidineProtonationRequest(),
    )

    assert captured_policy_values == [False]


def test_process_structure_threads_typed_histidine_request_to_hydrogen_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process workflow should carry typed histidine policy to hydrogen actions."""

    residue_id = ResidueId("A", 1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-typed-histidine-request-threading",
    )
    histidine_request = PrasRatioHistidineProtonationRequest(ratio=0.0)
    transformer = HydrogenCompletionTransformer(
        scope=ResidueSetScope(residue_ids=(residue_id,))
    )
    captured_requests: list[PrasRatioHistidineProtonationRequest] = []
    planner_call_count = 0

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planner_memory,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        nonlocal planner_call_count
        planner_call_count += 1
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(transformer,) if planner_call_count == 1 else (),
        )

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: PrasRatioHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert prepare_heavy_atoms is False
        assert target_residue_ids == frozenset({residue_id})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert local_refinement is None
        captured_requests.append(histidine_protonation)
        return ProcessResult(structure=structure, repairs=(), issues=(), analyses=None)

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
            include_default_heavy_completion=False,
        ),
        transform_requests=WorkflowTransformRequests(
            histidine_protonation=histidine_request,
        ),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=3),
    )

    assert result.structure is structure
    assert captured_requests == [histidine_request]


def test_local_refinement_transformer_threads_retained_non_polymer_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local refinement should pass retained-ligand fallback policy to repair stage."""

    residue_id = ResidueId("A", 1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-fallback-policy-threading",
    )
    repair_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    transformer = LocalRefinementTransformer.from_repair_refinement(repair_refinement)
    captured_policy_values: list[bool] = []

    def fake_apply_repair_stage_local_refinement(
        result: ProcessResult,
        *,
        local_refinement: RepairLocalRefinementDirective,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence=(),
    ) -> ProcessResult:
        assert component_library is not None
        assert retained_non_polymer_chemistry_evidence == ()
        assert local_refinement.scope_spec == repair_refinement.scope_spec
        captured_policy_values.append(allow_retained_non_polymer_rdkit_fallback)
        return result

    monkeypatch.setattr(
        "protrepair.workflow.actions.local_refinement.apply_repair_stage_local_refinement",
        fake_apply_repair_stage_local_refinement,
    )

    result = transformer.execute(
        ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        ),
        context=TransformerExecutionContext(
            component_library=build_default_component_library(),
            original_structure=structure,
            orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
            allow_retained_non_polymer_rdkit_fallback=False,
            retained_non_polymer_chemistry_evidence=(),
        ),
    )

    assert result.structure is structure
    assert captured_policy_values == [False]


def test_stereochemistry_correction_transformer_repairs_inverted_threonine() -> None:
    """Workflow stereo correction should repair a targeted inverted threonine."""

    structure = _focused_inverted_threonine_structure()
    transformer = StereochemistryCorrectionTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId("A", 30),)),
    )

    result = transformer.execute(
        ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        ),
        context=TransformerExecutionContext(
            component_library=build_default_component_library(),
            original_structure=structure,
            orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
        ),
    )

    report = detect_sidechain_stereochemistry(
        result.structure,
        component_library=build_default_component_library(),
    )

    assert all(
        violation.residue_id != ResidueId("A", 30) for violation in report.violations
    )
    assert any(
        repair.kind is RepairEventKind.STEREOCHEMISTRY_CORRECTED
        for repair in result.repairs
    )


def test_process_structure_routes_stereochemistry_correction_before_refinement() -> (
    None
):
    """Stereo-invalid local refinement should correct chirality before FF binding."""

    structure = _focused_inverted_threonine_structure()

    result = process_structure(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 30),)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    report = detect_sidechain_stereochemistry(
        result.structure,
        component_library=build_default_component_library(),
    )

    assert all(
        violation.residue_id != ResidueId("A", 30) for violation in report.violations
    )
    assert any(
        repair.kind is RepairEventKind.STEREOCHEMISTRY_CORRECTED
        for repair in result.repairs
    )
    corrected_residue = result.structure.constitution.residue_or_ligand(
        ResidueId("A", 30)
    )
    assert corrected_residue is not None
    assert any(atom_site.element == "H" for atom_site in corrected_residue.atom_sites)
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED for issue in result.issues
    )
    assert not any(
        "requires hydrogens to be fully realized" in issue.message
        for issue in result.issues
    )


def test_apply_structure_normalization_policy_returns_original_structure_on_noop() -> (
    None
):
    """Canonical boundary normalization should avoid no-op structure copies."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="canonical-fixture",
    )

    normalized = apply_structure_normalization_policy(
        structure,
        policy=StructureIngressOptions().structure_normalization_policy(),
    )

    assert normalized is structure


def test_process_structure_can_preserve_orphan_fragments_by_option() -> None:
    """Workflow options should thread orphan-fragment policy into heavy repair."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        ("N", "CA", "C", "O", "OG"),
                        positions_by_name={"OG": Vec3(3.00, -2.00, 0.0)},
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-preserve-orphan",
    )

    rebuilt = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(),
    )
    preserved = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(),
        transform_requests=WorkflowTransformRequests(
            orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE
        ),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    assert atom_position(rebuilt.structure, residue_id, "OG") != Vec3(
        3.0,
        -2.0,
        0.0,
    )
    assert atom_position(preserved.structure, residue_id, "OG") == Vec3(
        3.0,
        -2.0,
        0.0,
    )


def test_process_structure_preserves_c_terminal_oxt_by_default() -> None:
    """Workflow default should not augment source-boundary OXT inventory."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="preserve-c-terminal-oxt",
    )

    result = process_structure(structure)

    assert not has_atom(result.structure, ResidueId(chain_id="A", seq_num=1), "OXT")
    assert all(
        repair.kind is not RepairEventKind.C_TERMINAL_OXT_ADDED
        for repair in result.repairs
    )
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes == ()
    assert (
        result.requested_goal_completion_verdict()
        is RequestedGoalCompletionVerdict.NOT_REQUESTED
    )
    assert result.requested_goals_fully_satisfied() is None


def test_process_structure_can_augment_authentic_source_c_terminal_oxt() -> None:
    """Workflow should add OXT when the boundary policy explicitly requests it."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="augment-c-terminal-oxt",
    )

    result = process_structure(
        structure,
        requested_goals=c_terminal_oxt_requested_goals(
            ResidueId(chain_id="A", seq_num=1)
        ),
    )

    assert has_atom(result.structure, ResidueId(chain_id="A", seq_num=1), "OXT")
    assert any(
        repair.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for repair in result.repairs
    )
    assert result.requested_goal_report is not None
    requested_state = requested_process_goal(
        scope=ResidueBoundaryScope(
            residue_id=ResidueId(chain_id="A", seq_num=1),
            side=ResidueBoundarySide.C_TERMINUS,
        ),
        value=OxtPresenceState.PRESENT,
    )
    outcome = result.requested_goal_report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.SATISFIED
    assert (
        result.requested_goal_completion_verdict()
        is RequestedGoalCompletionVerdict.ACHIEVED
    )


def test_process_structure_keeps_heavy_repair_and_terminal_augmentation_separate() -> (
    None
):
    """Heavy repair should not imply OXT augmentation unless policy requests it."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="heavy-vs-terminal-policy",
    )

    preserved = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(),
    )
    augmented = process_structure(
        structure,
        requested_goals=(
            *whole_structure_requested_goals(),
            *c_terminal_oxt_requested_goals(ResidueId(chain_id="A", seq_num=1)),
        ),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    assert has_atom(preserved.structure, residue_id, "CB")
    assert not has_atom(preserved.structure, residue_id, "OXT")
    assert any(
        repair.kind is RepairEventKind.HEAVY_ATOMS_ADDED for repair in preserved.repairs
    )
    assert all(
        repair.kind is not RepairEventKind.C_TERMINAL_OXT_ADDED
        for repair in preserved.repairs
    )

    assert has_atom(augmented.structure, residue_id, "CB")
    assert has_atom(augmented.structure, residue_id, "OXT")
    assert any(
        repair.kind is RepairEventKind.C_TERMINAL_OXT_ADDED
        for repair in augmented.repairs
    )


def test_process_structure_can_augment_c_terminal_oxt_before_hydrogenation() -> None:
    """Workflow should honor explicit OXT augmentation on hydrogen-only routes."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="augment-c-terminal-before-hydrogenation",
    )

    result = process_structure(
        structure,
        requested_goals=(
            *whole_structure_requested_goals(
                HydrogenCoverageState.COMPLETE,
            ),
            *c_terminal_oxt_requested_goals(ResidueId(chain_id="A", seq_num=1)),
        ),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    assert has_atom(result.structure, residue_id, "OXT")
    assert any(
        repair.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for repair in result.repairs
    )


def test_process_structure_reports_unsupported_scoped_requested_state() -> None:
    """Unsupported scoped requested goals should be reported without raising."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    requested_state = requested_process_goal(
        scope=ResidueSetScope(residue_ids=(residue_id,)),
        value=HydrogenCoverageState.COMPLETE,
    )
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="unsupported-scoped-request",
    )

    result = process_structure(
        structure,
        requested_goals=(requested_state,),
    )

    assert result.requested_goal_report is not None
    outcome = result.requested_goal_report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.UNSUPPORTED
    assert (
        result.requested_goal_completion_verdict()
        is RequestedGoalCompletionVerdict.UNACHIEVED
    )


def test_process_structure_reports_blocked_requested_state() -> None:
    """Blocked whole-structure requests should surface blocker status."""

    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("UNK", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="blocked-requested-goal",
    )

    result = process_structure(
        structure,
        requested_goals=(requested_state,),
    )

    assert result.requested_goal_report is not None
    outcome = result.requested_goal_report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.BLOCKED
    assert outcome.blocking_scopes == (
        ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
    )
    assert (
        result.requested_goal_completion_verdict()
        is RequestedGoalCompletionVerdict.UNACHIEVED
    )


def test_requested_goal_report_skips_snapshot_for_whole_structure_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-structure requested-goal evaluation should not lower through snapshots."""

    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="requested-goal-whole-structure-fast-path",
    )

    def _fail_snapshot(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "snapshot lowering should not run for whole-structure requests"
        )

    monkeypatch.setattr(
        "protrepair.workflow.planning.assessment.goal_support.ProteinStructureSnapshot.from_structure",
        _fail_snapshot,
    )

    report = evaluate_requested_goal_report(
        structure,
        requested_goals=(requested_state,),
    )

    outcome = report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.UNMET


def test_requested_goal_report_skips_snapshot_for_boundary_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary requested-goal evaluation should use terminal facts
    without snapshots."""

    requested_state = requested_process_goal(
        scope=ResidueBoundaryScope(
            residue_id=ResidueId(chain_id="A", seq_num=1),
            side=ResidueBoundarySide.C_TERMINUS,
        ),
        value=OxtPresenceState.PRESENT,
    )
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="requested-goal-boundary-fast-path",
    )

    def _fail_snapshot(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot lowering should not run for boundary requests")

    monkeypatch.setattr(
        "protrepair.workflow.planning.assessment.goal_support.ProteinStructureSnapshot.from_structure",
        _fail_snapshot,
    )

    report = evaluate_requested_goal_report(
        structure,
        requested_goals=(requested_state,),
    )

    outcome = report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.UNMET


def test_process_structure_uses_packed_reference_for_missing_heavy_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packing backends should guide missing heavy-atom repair."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="packing-guidance",
    )

    packed_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        ("N", "CA", "C", "O", "CB"),
                        positions_by_name={"CB": Vec3(99.0, 98.0, 97.0)},
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="packing-guidance",
    )

    expected_residue_id = ResidueId(chain_id="A", seq_num=1)

    class FakePackingBackend:
        def pack(self, plan):
            assert plan.spec.backend_name == "faspr"
            return PackingResult(
                packed_structure=packed_structure,
                changed_residue_ids=(expected_residue_id,),
                issues=(),
                backend_name="faspr",
            )

    def resolve_backend(backend_name: str):
        assert backend_name == "faspr"
        return FakePackingBackend()

    monkeypatch.setattr(
        "protrepair.transformer.packing.runtime.resolve_sidechain_packing_backend",
        resolve_backend,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(),
        transform_requests=WorkflowTransformRequests(
            reference_sidechain_packing=PackingSpec(
                backend_name="faspr",
                mode=PackingMode.PACK,
                scope=PackingScope.FULL,
            ),
        ),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    assert has_atom(result.structure, residue_id, "CB")
    assert atom_position(result.structure, residue_id, "CB") == Vec3(
        99.0,
        98.0,
        97.0,
    )
    assert not result.has_errors()


def test_prepare_workflow_packing_reference_adapts_transform_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow packing references should wrap, not redefine, packing results."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-packing-reference",
    )
    packing_result = PackingResult(
        packed_structure=structure,
        changed_residue_ids=(ResidueId(chain_id="A", seq_num=1),),
        issues=(),
        backend_name="faspr",
    )

    def fake_execute_sidechain_packing(
        candidate_structure: ProteinStructure,
        spec: PackingSpec,
    ) -> PackingResult:
        assert candidate_structure is structure
        assert spec.backend_name == "faspr"
        return packing_result

    monkeypatch.setattr(
        "protrepair.workflow.engine.packing.reference.execute_sidechain_packing",
        fake_execute_sidechain_packing,
    )

    reference = prepare_workflow_packing_reference(
        structure,
        PackingSpec(
            backend_name="faspr",
            mode=PackingMode.PACK,
            scope=PackingScope.FULL,
        ),
    )

    assert isinstance(reference, WorkflowPackingReference)
    assert reference.reference_structure is structure
    assert reference.packing_result is packing_result


def test_planner_proposes_faspr_after_parser_refinement_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser burden should route through packing after one FF repair attempt."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
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
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="packing-after-refinement-stall",
    )
    prior_refinement = LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues((residue_id,)),
            binding=RecommendedContinuousRelaxationBinding(),
        )
    )

    def fake_parser_metrics(
        structure: ProteinStructure,
        **kwargs,
    ) -> RDKitNoConectSanitizeReadabilityMetrics:
        del structure, kwargs
        return RDKitNoConectSanitizeReadabilityMetrics(
            sanitize_readable=False,
            extra_proximity_bond_count=1,
            extra_heavy_proximity_bond_count=1,
        )

    def fake_extra_proximity_bond_clusters(
        structure: ProteinStructure,
        **kwargs,
    ) -> tuple[RDKitProximityBondCluster, ...]:
        del structure, kwargs
        return _fake_parser_witness_clusters(residue_id, 1)

    monkeypatch.setattr(
        "protrepair.state.structure_parser.measure_rdkit_no_conect_sanitize_readability_metrics",
        fake_parser_metrics,
    )
    monkeypatch.setattr(
        "protrepair.workflow.planning.default_action_registry.packing_proposals.rdkit_no_conect_extra_proximity_bond_clusters",
        fake_extra_proximity_bond_clusters,
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=WorkflowPlannerMemory(adopted_transformers=(prior_refinement,)),
    )

    assert len(outcome.transformers) == 1
    transformer = outcome.transformers[0]
    assert isinstance(transformer, CommittedPackingTransformer)
    assert transformer.packing_spec.backend_name == "faspr"
    assert transformer.packing_spec.scope is PackingScope.LOCAL
    assert transformer.packing_spec.mutable_residue_ids == (residue_id,)


def test_process_structure_applies_committed_packing_before_heavy_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed packing should update the workflow structure before heavy repair."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="committed-packing-order",
    )
    packed_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        ("N", "CA", "C", "O", "CB"),
                        positions_by_name={"CB": Vec3(88.0, 87.0, 86.0)},
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="committed-packing-order",
    )

    class FakePackingBackend:
        def pack(self, plan):
            assert plan.spec.backend_name == "faspr"
            return PackingResult(
                packed_structure=packed_structure,
                changed_residue_ids=(ResidueId(chain_id="A", seq_num=1),),
                issues=(),
                backend_name="faspr",
            )

    def resolve_backend(backend_name: str):
        assert backend_name == "faspr"
        return FakePackingBackend()

    def fake_repair_heavy_atoms_core(
        candidate_structure: ProteinStructure,
        *,
        component_library,
        reference_structure,
        augment_c_terminal_oxt: bool,
        target_residue_ids,
        orphan_fragment_policy: OrphanFragmentPolicy,
        local_refinement,
    ) -> ProcessResult:
        del component_library, reference_structure, augment_c_terminal_oxt
        del target_residue_ids, orphan_fragment_policy, local_refinement
        residue_id = ResidueId(chain_id="A", seq_num=1)
        assert has_atom(candidate_structure, residue_id, "CB")
        assert atom_position(candidate_structure, residue_id, "CB") == Vec3(
            88.0,
            87.0,
            86.0,
        )
        return ProcessResult(
            structure=candidate_structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    monkeypatch.setattr(
        "protrepair.transformer.packing.runtime.resolve_sidechain_packing_backend",
        resolve_backend,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.heavy_completion.repair_heavy_atoms_core",
        fake_repair_heavy_atoms_core,
    )

    result = process_structure(
        structure,
        transform_requests=WorkflowTransformRequests(
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                mode=PackingMode.PACK,
                scope=PackingScope.FULL,
            )
        ),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    assert has_atom(result.structure, residue_id, "CB")
    assert atom_position(result.structure, residue_id, "CB") == Vec3(
        88.0,
        87.0,
        86.0,
    )


def test_process_structure_applies_local_refinement_after_heavy_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow should stage hydrogenation after heavy repair before refinement."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-heavy",
    )
    heavy_completed_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-heavy-complete",
    )
    hydrogenated_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
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
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-heavy-hydrogenated",
    )
    local_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )

    calls: list[str] = []

    def fake_repair_heavy_atoms_core(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        augment_c_terminal_oxt: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not augment_c_terminal_oxt
        assert target_residue_ids == frozenset({residue_id})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert local_refinement is None
        calls.append("heavy")
        return ProcessResult(
            structure=heavy_completed_structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: DisabledHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert structure is heavy_completed_structure
        assert not prepare_heavy_atoms
        assert target_residue_ids == frozenset({residue_id})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert isinstance(histidine_protonation, DisabledHistidineProtonationRequest)
        assert local_refinement is None
        calls.append("hydrogen")
        return ProcessResult(
            structure=hydrogenated_structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_apply_repair_stage_local_refinement(
        result: ProcessResult,
        *,
        local_refinement: RepairLocalRefinementDirective,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence=(),
    ) -> ProcessResult:
        assert component_library is not None
        assert retained_non_polymer_chemistry_evidence == ()
        assert result.structure is hydrogenated_structure
        assert local_refinement == local_refinement_spec
        assert allow_retained_non_polymer_rdkit_fallback
        calls.append("refine")
        return result

    local_refinement_spec = residuewise_local_refinement_directive(local_refinement)
    monkeypatch.setattr(
        "protrepair.workflow.actions.heavy_completion.repair_heavy_atoms_core",
        fake_repair_heavy_atoms_core,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.local_refinement.apply_repair_stage_local_refinement",
        fake_apply_repair_stage_local_refinement,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=local_refinement.scope_spec,
                config=local_refinement.config,
                binding=local_refinement.binding,
            )
        ),
    )

    assert result.structure is hydrogenated_structure
    assert calls == ["heavy", "hydrogen", "refine"]


def test_process_structure_expands_multi_node_workflow_frontier_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow frontier expansion should process sibling branches in FIFO order."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-frontier-root",
    )
    residue_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    first_transformer = TerminalAugmentationTransformer(
        scope=residue_scope,
    )
    second_transformer = HeavyAtomCompletionTransformer(
        scope=residue_scope,
    )
    planner_calls: list[tuple[str, tuple[type[WorkflowStateAction], ...]]] = []
    execution_calls: list[type[WorkflowStateAction]] = []

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        planner_calls.append(
            (
                current_structure.provenance.ingress.source_name or "",
                tuple(
                    type(transformer)
                    for transformer in active_planner_memory.adopted_transformers
                ),
            )
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(first_transformer, second_transformer),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        execution_calls.append(type(transformer))
        branch_name = (
            "workflow-frontier-branch-a"
            if transformer is first_transformer
            else "workflow-frontier-branch-b"
        )
        return replace(
            result,
            structure=result.structure.with_provenance(
                StructureProvenance(
                    ingress=StructureIngress(
                        source_format=(
                            result.structure.provenance.ingress.source_format
                        ),
                        source_name=branch_name,
                    ),
                    lineage=result.structure.provenance.lineage,
                )
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(structure)

    assert execution_calls == [
        TerminalAugmentationTransformer,
        HeavyAtomCompletionTransformer,
    ]
    assert planner_calls == [
        ("workflow-frontier-root", ()),
        (
            "workflow-frontier-branch-a",
            (TerminalAugmentationTransformer,),
        ),
        (
            "workflow-frontier-branch-b",
            (HeavyAtomCompletionTransformer,),
        ),
    ]
    assert (
        result.structure.provenance.ingress.source_name == "workflow-frontier-branch-a"
    )


def test_workflow_children_retain_current_branch_when_child_regresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regressed child should not erase its still-clean current branch."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-parent",
    )
    residue_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    transformer = LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues(residue_scope.residue_ids),
            binding=RecommendedContinuousRelaxationBinding(),
        )
    )
    current_branch_state = WorkflowRuntimeState(
        result=ProcessResult(structure=structure, repairs=(), issues=()),
        planner_memory=WorkflowPlannerMemory(),
    )
    child_structure = structure.with_provenance(
        StructureProvenance(
            ingress=StructureIngress(
                source_format=FileFormat.PDB,
                source_name="workflow-child-regressed",
            ),
            lineage=structure.provenance.lineage,
        )
    )
    child_memory = current_branch_state.planner_memory.with_adopted_transformer(
        transformer
    )
    child = SpeculativeAdoptedChild(
        state=WorkflowRuntimeState(
            result=ProcessResult(structure=child_structure, repairs=(), issues=()),
            planner_memory=child_memory,
        ),
        adopted_decision=SpeculativeAdoptionDecision.reject(reason="fixture"),
    )

    def fake_evaluate_requested_goal_report(
        *args,
        **kwargs,
    ) -> RequestedGoalReport:
        del args, kwargs
        return RequestedGoalReport(())

    def fake_evaluate_workflow_branch_quality_score(
        result: ProcessResult,
        **kwargs,
    ) -> WorkflowBranchQualityScore:
        del kwargs
        return WorkflowBranchQualityScore(
            parser_incompatible=int(
                result.structure.provenance.ingress.source_name
                == "workflow-child-regressed"
            )
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.evaluate_requested_goal_report",
        fake_evaluate_requested_goal_report,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.evaluate_workflow_branch_quality_score",
        fake_evaluate_workflow_branch_quality_score,
    )

    retained_children = _workflow_children_with_regression_retention(
        current_branch_state=current_branch_state,
        attempted_transformers=(transformer,),
        transform_requests=WorkflowTransformRequests(),
        retained_children=(child,),
        requested_goals=RequestedGoalSet(),
        component_library=build_default_component_library(),
        planning_context=WorkflowPlanningContext(),
        already_satisfied_requested_goals=(),
    )

    assert len(retained_children) == 2
    retained_current_branch = retained_children[1]
    assert retained_current_branch.state.result.structure is structure
    assert retained_current_branch.state.planner_memory.adopted_transformers == ()
    assert retained_current_branch.state.planner_memory.has_attempted_transformer(
        transformer
    )
    assert retained_current_branch.adopted_decision.reason is not None
    assert _workflow_children_within_node_budget(
        children=retained_children,
        child_budget=1,
        requested_goals=RequestedGoalSet(),
        component_library=build_default_component_library(),
        planning_context=WorkflowPlanningContext(),
        already_satisfied_requested_goals=(),
    ) == (retained_current_branch,)


def test_process_structure_executes_parser_witness_repair_inside_workflow_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser-witness repair should run before planner re-entry."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="parser-witness-workflow-root",
    )
    residue_scope = ResidueSetScope(residue_ids=(residue_id,))
    primary_transformer = TerminalAugmentationTransformer(scope=residue_scope)
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
    planner_calls: list[tuple[str, tuple[type[WorkflowStateAction], ...]]] = []
    execution_calls: list[type[WorkflowStateAction]] = []
    parser_candidate_clusters: list[tuple[RDKitProximityBondCluster, ...] | None] = []

    def structure_with_source_name(
        current_structure: ProteinStructure,
        source_name: str,
    ) -> ProteinStructure:
        return current_structure.with_provenance(
            StructureProvenance(
                ingress=StructureIngress(
                    source_format=current_structure.provenance.ingress.source_format,
                    source_name=source_name,
                ),
                lineage=current_structure.provenance.lineage,
            )
        )

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        planner_calls.append(
            (
                current_structure.provenance.ingress.source_name or "",
                tuple(
                    type(transformer)
                    for transformer in active_planner_memory.adopted_transformers
                ),
            )
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(primary_transformer,),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_parser_witness_repair_candidates(
        current_structure: ProteinStructure,
        *,
        component_library,
        budget: ParserWitnessRepairBudget | None = None,
        clusters: tuple[RDKitProximityBondCluster, ...] | None = None,
    ) -> tuple[ParserWitnessRepairCandidate, ...]:
        del component_library
        del budget
        parser_candidate_clusters.append(clusters)
        if current_structure.provenance.ingress.source_name != "after-primary":
            return ()

        return (parser_candidate,)

    def fake_probe_rdkit_no_conect_parser_readability(
        current_structure: ProteinStructure,
        *,
        component_library,
        pdb_block_projector=None,
    ) -> RDKitNoConectParserReadabilityProbe:
        del component_library
        del pdb_block_projector
        if current_structure.provenance.ingress.source_name == "after-primary":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(residue_id, 1)
            )

        return _fake_parser_probe_from_clusters(())

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        execution_calls.append(type(transformer))
        if transformer is primary_transformer:
            return replace(
                result,
                structure=structure_with_source_name(
                    result.structure,
                    "after-primary",
                ),
            )

        assert isinstance(transformer, LocalRefinementTransformer)
        assert transformer.repair_refinement == parser_candidate.repair_refinement
        return replace(
            result,
            structure=structure_with_source_name(
                result.structure,
                "after-parser-witness",
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.parser_witness_repair_candidates",
        fake_parser_witness_repair_candidates,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.probe_rdkit_no_conect_parser_readability",
        fake_probe_rdkit_no_conect_parser_readability,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(structure)

    assert execution_calls == [
        TerminalAugmentationTransformer,
        LocalRefinementTransformer,
    ]
    assert planner_calls == [
        ("parser-witness-workflow-root", ()),
        (
            "after-parser-witness",
            (TerminalAugmentationTransformer, LocalRefinementTransformer),
        ),
    ]
    assert result.structure.provenance.ingress.source_name == "after-parser-witness"
    assert parser_candidate_clusters == [_fake_parser_witness_clusters(residue_id, 1)]


def test_execute_iterative_workflow_caps_runaway_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow runtime should stop planner loops with no terminal branch."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="runaway-frontier-root",
    )
    transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(residue_id,))
    )
    planner_call_count = 0

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planner_memory,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        nonlocal planner_call_count
        planner_call_count += 1
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(transformer,),
        )

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del transformer
        del execution_context
        return result

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    with pytest.raises(ValueError, match="max_speculative_nodes=2"):
        execute_iterative_workflow(
            structure,
            requested_goals=RequestedGoalSet(),
            transform_requests=WorkflowTransformRequests(),
            component_library=build_default_component_library(),
            planning_context=WorkflowPlanningContext(max_speculative_nodes=2),
            reference_structure=None,
            orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
            histidine_protonation=DisabledHistidineProtonationRequest(),
        )

    assert planner_call_count == 2


def test_execute_iterative_workflow_caps_current_proposal_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow runtime should not execute children it cannot add to trace."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                    build_residue("ALA", "A", 3, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="batch-capped-frontier-root",
    )
    transformers = tuple(
        TerminalAugmentationTransformer(
            scope=ResidueSetScope(
                residue_ids=(ResidueId(chain_id="A", seq_num=seq_num),)
            )
        )
        for seq_num in (1, 2, 3)
    )
    planner_call_count = 0
    executed_transformers: list[WorkflowStateAction] = []

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planner_memory,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        nonlocal planner_call_count
        planner_call_count += 1
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=transformers,
        )

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        executed_transformers.append(transformer)
        return result

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    with pytest.raises(ValueError, match="max_speculative_nodes=2"):
        execute_iterative_workflow(
            structure,
            requested_goals=RequestedGoalSet(),
            transform_requests=WorkflowTransformRequests(),
            component_library=build_default_component_library(),
            planning_context=WorkflowPlanningContext(max_speculative_nodes=2),
            reference_structure=None,
            orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
            histidine_protonation=DisabledHistidineProtonationRequest(),
        )

    assert planner_call_count == 2
    assert tuple(executed_transformers) == (transformers[0],)


def test_process_structure_requeries_parser_witness_repair_between_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser-witness repair should re-cluster after each improving pass."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="parser-witness-loop-root",
    )
    residue_scope = ResidueSetScope(residue_ids=(residue_id,))
    primary_transformer = TerminalAugmentationTransformer(scope=residue_scope)
    repair_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        binding=RecommendedContinuousRelaxationBinding(),
        config=ContinuousRelaxationConfig(max_iterations=20),
    )
    first_candidate = ParserWitnessRepairCandidate(
        cluster=RDKitProximityBondCluster(
            residue_ids=(residue_id,),
            bonds=(),
        ),
        repair_refinement=repair_refinement,
        budget=ParserWitnessRepairBudget(max_passes=3, max_iterations=20),
    )
    second_candidate = ParserWitnessRepairCandidate(
        cluster=RDKitProximityBondCluster(
            residue_ids=(residue_id,),
            bonds=(),
        ),
        repair_refinement=repair_refinement,
        budget=ParserWitnessRepairBudget(max_passes=3, max_iterations=20),
    )
    parser_queries: list[str] = []
    execution_calls: list[type[WorkflowStateAction]] = []

    def structure_with_source_name(
        current_structure: ProteinStructure,
        source_name: str,
    ) -> ProteinStructure:
        return current_structure.with_provenance(
            StructureProvenance(
                ingress=StructureIngress(
                    source_format=current_structure.provenance.ingress.source_format,
                    source_name=source_name,
                ),
                lineage=current_structure.provenance.lineage,
            )
        )

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(primary_transformer,),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_probe_rdkit_no_conect_parser_readability(
        current_structure: ProteinStructure,
        *,
        component_library,
        pdb_block_projector=None,
    ) -> RDKitNoConectParserReadabilityProbe:
        del component_library
        del pdb_block_projector
        source_name = current_structure.provenance.ingress.source_name
        if source_name == "after-primary":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(residue_id, 2)
            )
        if source_name == "after-parser-witness-1":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(residue_id, 1)
            )

        return _fake_parser_probe_from_clusters(())

    def fake_parser_witness_repair_candidates(
        current_structure: ProteinStructure,
        *,
        component_library,
        budget: ParserWitnessRepairBudget | None = None,
        clusters: tuple[RDKitProximityBondCluster, ...] | None = None,
    ) -> tuple[ParserWitnessRepairCandidate, ...]:
        del component_library
        del budget
        assert clusters is not None
        source_name = current_structure.provenance.ingress.source_name or ""
        parser_queries.append(source_name)
        if source_name == "after-primary":
            return (first_candidate,)
        if source_name == "after-parser-witness-1":
            return (second_candidate,)

        return ()

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        execution_calls.append(type(transformer))
        if transformer is primary_transformer:
            return replace(
                result,
                structure=structure_with_source_name(
                    result.structure,
                    "after-primary",
                ),
            )

        assert isinstance(transformer, LocalRefinementTransformer)
        if result.structure.provenance.ingress.source_name == "after-primary":
            return replace(
                result,
                structure=structure_with_source_name(
                    result.structure,
                    "after-parser-witness-1",
                ),
            )

        return replace(
            result,
            structure=structure_with_source_name(
                result.structure,
                "after-parser-witness-2",
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.probe_rdkit_no_conect_parser_readability",
        fake_probe_rdkit_no_conect_parser_readability,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.parser_witness_repair_candidates",
        fake_parser_witness_repair_candidates,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(structure)

    assert execution_calls == [
        TerminalAugmentationTransformer,
        LocalRefinementTransformer,
        LocalRefinementTransformer,
    ]
    assert parser_queries == ["after-primary", "after-parser-witness-1"]
    assert result.structure.provenance.ingress.source_name == "after-parser-witness-2"


def test_process_structure_rolls_back_parser_witness_pass_that_worsens_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser-witness repair should not retain a pass that increases witnesses."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="parser-witness-rollback-root",
    )
    residue_scope = ResidueSetScope(residue_ids=(residue_id,))
    primary_transformer = TerminalAugmentationTransformer(scope=residue_scope)
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
        budget=ParserWitnessRepairBudget(max_passes=3, max_iterations=20),
    )
    execution_calls: list[type[WorkflowStateAction]] = []

    def structure_with_source_name(
        current_structure: ProteinStructure,
        source_name: str,
    ) -> ProteinStructure:
        return current_structure.with_provenance(
            StructureProvenance(
                ingress=StructureIngress(
                    source_format=current_structure.provenance.ingress.source_format,
                    source_name=source_name,
                ),
                lineage=current_structure.provenance.lineage,
            )
        )

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(primary_transformer,),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_probe_rdkit_no_conect_parser_readability(
        current_structure: ProteinStructure,
        *,
        component_library,
        pdb_block_projector=None,
    ) -> RDKitNoConectParserReadabilityProbe:
        del component_library
        del pdb_block_projector
        source_name = current_structure.provenance.ingress.source_name
        if source_name == "after-primary":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(residue_id, 1)
            )
        if source_name == "after-parser-witness-worse":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(residue_id, 2)
            )

        return _fake_parser_probe_from_clusters(())

    def fake_parser_witness_repair_candidates(
        current_structure: ProteinStructure,
        *,
        component_library,
        budget: ParserWitnessRepairBudget | None = None,
        clusters: tuple[RDKitProximityBondCluster, ...] | None = None,
    ) -> tuple[ParserWitnessRepairCandidate, ...]:
        del component_library
        del budget
        assert clusters is not None
        if current_structure.provenance.ingress.source_name == "after-primary":
            return (parser_candidate,)

        return ()

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        execution_calls.append(type(transformer))
        if transformer is primary_transformer:
            return replace(
                result,
                structure=structure_with_source_name(
                    result.structure,
                    "after-primary",
                ),
            )

        assert isinstance(transformer, LocalRefinementTransformer)
        return replace(
            result,
            structure=structure_with_source_name(
                result.structure,
                "after-parser-witness-worse",
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.probe_rdkit_no_conect_parser_readability",
        fake_probe_rdkit_no_conect_parser_readability,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.parser_witness_repair_candidates",
        fake_parser_witness_repair_candidates,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(structure)

    assert execution_calls == [
        TerminalAugmentationTransformer,
        LocalRefinementTransformer,
    ]
    assert result.structure.provenance.ingress.source_name == "after-primary"


def test_process_structure_skips_unproductive_parser_witness_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser-witness repair should try the next cluster after a no-op trial."""

    first_residue_id = ResidueId(chain_id="A", seq_num=1)
    second_residue_id = ResidueId(chain_id="A", seq_num=2)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="parser-witness-skip-root",
    )
    residue_scope = ResidueSetScope(residue_ids=(first_residue_id,))
    primary_transformer = TerminalAugmentationTransformer(scope=residue_scope)
    first_repair_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((first_residue_id,)),
        binding=RecommendedContinuousRelaxationBinding(),
        config=ContinuousRelaxationConfig(max_iterations=20),
    )
    second_repair_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((second_residue_id,)),
        binding=RecommendedContinuousRelaxationBinding(),
        config=ContinuousRelaxationConfig(max_iterations=20),
    )
    first_candidate = ParserWitnessRepairCandidate(
        cluster=RDKitProximityBondCluster(
            residue_ids=(first_residue_id,),
            bonds=(),
        ),
        repair_refinement=first_repair_refinement,
        budget=ParserWitnessRepairBudget(max_passes=3, max_clusters_per_pass=4),
    )
    second_candidate = ParserWitnessRepairCandidate(
        cluster=RDKitProximityBondCluster(
            residue_ids=(second_residue_id,),
            bonds=(),
        ),
        repair_refinement=second_repair_refinement,
        budget=ParserWitnessRepairBudget(max_passes=3, max_clusters_per_pass=4),
    )
    execution_calls: list[
        type[WorkflowStateAction] | tuple[type[WorkflowStateAction], ResidueId]
    ] = []
    parser_queries: list[str] = []

    def structure_with_source_name(
        current_structure: ProteinStructure,
        source_name: str,
    ) -> ProteinStructure:
        return current_structure.with_provenance(
            StructureProvenance(
                ingress=StructureIngress(
                    source_format=current_structure.provenance.ingress.source_format,
                    source_name=source_name,
                ),
                lineage=current_structure.provenance.lineage,
            )
        )

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(primary_transformer,),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_probe_rdkit_no_conect_parser_readability(
        current_structure: ProteinStructure,
        *,
        component_library,
        pdb_block_projector=None,
    ) -> RDKitNoConectParserReadabilityProbe:
        del component_library
        del pdb_block_projector
        source_name = current_structure.provenance.ingress.source_name
        if source_name == "after-primary":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(first_residue_id, 2)
            )
        if source_name == "after-parser-witness-noop":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(first_residue_id, 2)
            )
        if source_name == "after-parser-witness-improved":
            return _fake_parser_probe_from_clusters(
                _fake_parser_witness_clusters(second_residue_id, 1)
            )

        return _fake_parser_probe_from_clusters(())

    def fake_parser_witness_repair_candidates(
        current_structure: ProteinStructure,
        *,
        component_library,
        budget: ParserWitnessRepairBudget | None = None,
        clusters: tuple[RDKitProximityBondCluster, ...] | None = None,
    ) -> tuple[ParserWitnessRepairCandidate, ...]:
        del component_library
        del budget
        assert clusters is not None
        source_name = current_structure.provenance.ingress.source_name or ""
        parser_queries.append(source_name)
        if source_name == "after-primary":
            return (first_candidate, second_candidate)

        return ()

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        if transformer is primary_transformer:
            execution_calls.append(type(transformer))
            return replace(
                result,
                structure=structure_with_source_name(
                    result.structure,
                    "after-primary",
                ),
            )

        assert isinstance(transformer, LocalRefinementTransformer)
        target_residue_id = (
            transformer.repair_refinement.scope_spec.referenced_residue_ids()[0]
        )
        execution_calls.append((type(transformer), target_residue_id))
        if transformer.repair_refinement == first_repair_refinement:
            return replace(
                result,
                structure=structure_with_source_name(
                    result.structure,
                    "after-parser-witness-noop",
                ),
            )

        assert transformer.repair_refinement == second_repair_refinement
        return replace(
            result,
            structure=structure_with_source_name(
                result.structure,
                "after-parser-witness-improved",
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.probe_rdkit_no_conect_parser_readability",
        fake_probe_rdkit_no_conect_parser_readability,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.parser_witness_repair.parser_witness_repair_candidates",
        fake_parser_witness_repair_candidates,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(structure)

    assert execution_calls == [
        TerminalAugmentationTransformer,
        (LocalRefinementTransformer, first_residue_id),
        (LocalRefinementTransformer, second_residue_id),
    ]
    assert parser_queries == ["after-primary", "after-parser-witness-improved"]
    assert result.structure.provenance.ingress.source_name == (
        "after-parser-witness-improved"
    )


def test_final_parser_readability_issues_include_ambiguous_disulfide_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final workflow diagnostics should include parser/topology blockers."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("CYS", "A", 1, ("N", "CA", "C", "O", "CB", "SG")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="parser-topology-final-issues",
    )
    parser_issue = ValidationIssue(
        kind=ValidationIssueKind.PARSER_READABILITY,
        severity=IssueSeverity.ERROR,
        message="parser issue",
    )
    ambiguous_issue = ValidationIssue(
        kind=ValidationIssueKind.AMBIGUOUS_DISULFIDE,
        severity=IssueSeverity.WARNING,
        message="ambiguous disulfide issue",
    )
    probe_queries: list[ProteinStructure] = []

    class _FakeParserProbe:
        def issues(self) -> tuple[ValidationIssue, ...]:
            return (parser_issue,)

        def extra_proximity_bond_clusters(
            self,
        ) -> tuple[RDKitProximityBondCluster, ...]:
            return ()

    def fake_probe_rdkit_no_conect_parser_readability(
        current_structure: ProteinStructure,
        *,
        component_library,
        pdb_block_projector=None,
    ) -> _FakeParserProbe:
        del component_library
        del pdb_block_projector
        probe_queries.append(current_structure)
        return _FakeParserProbe()

    def fake_ambiguous_disulfide_parser_witness_blocker_issues(
        current_structure: ProteinStructure,
        *,
        clusters: tuple[RDKitProximityBondCluster, ...],
    ) -> tuple[ValidationIssue, ...]:
        del current_structure
        assert clusters == ()
        return (ambiguous_issue,)

    monkeypatch.setattr(
        "protrepair.workflow.engine.finalization.probe_rdkit_no_conect_parser_readability",
        fake_probe_rdkit_no_conect_parser_readability,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.finalization.ambiguous_disulfide_parser_witness_blocker_issues",
        fake_ambiguous_disulfide_parser_witness_blocker_issues,
    )

    issues = _final_parser_readability_issues(
        structure,
        component_library=build_default_component_library(),
    )

    assert len(issues) == 2
    assert issues[0].kind is ValidationIssueKind.PARSER_READABILITY
    assert issues[0].severity is IssueSeverity.ERROR
    assert issues[0].message.startswith("parser issue;")
    assert "blocked by ambiguous disulfide topology" in issues[0].message
    assert issues[1] == ambiguous_issue
    assert probe_queries == [structure]


def test_process_structure_prefers_terminal_branch_that_satisfies_requested_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preferred terminal branch should be chosen by requested-goal satisfaction."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-terminal-root",
    )
    requested_state = requested_process_goal(
        scope=ResidueBoundaryScope(
            residue_id=ResidueId(chain_id="A", seq_num=1),
            side=ResidueBoundarySide.C_TERMINUS,
        ),
        value=OxtPresenceState.PRESENT,
    )
    first_transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
    )
    second_transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=2),)),
    )

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(first_transformer, second_transformer),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        if transformer is first_transformer:
            return replace(
                result,
                structure=result.structure.with_provenance(
                    StructureProvenance(
                        ingress=StructureIngress(
                            source_format=(
                                result.structure.provenance.ingress.source_format
                            ),
                            source_name="workflow-terminal-branch-a",
                        ),
                        lineage=result.structure.provenance.lineage,
                    )
                ),
            )

        return replace(
            result,
            structure=build_workflow_structure(
                chains=(
                    build_chain(
                        "A",
                        (
                            build_residue(
                                "GLY",
                                "A",
                                1,
                                ("N", "CA", "C", "O", "OXT"),
                            ),
                        ),
                    ),
                ),
                ligands=tuple(
                    (
                        ligand,
                        result.structure.residue_geometry(
                            result.structure.constitution.residue_index(
                                ligand.residue_id
                            )
                        ),
                        result.structure.residue_formal_charge_by_atom_name(
                            result.structure.constitution.residue_index(
                                ligand.residue_id
                            )
                        ),
                    )
                    for ligand in result.structure.constitution.ligands
                ),
                source_format=result.structure.provenance.ingress.source_format,
                source_name="workflow-terminal-branch-b",
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(
        structure,
        requested_goals=(requested_state,),
    )

    assert (
        result.structure.provenance.ingress.source_name == "workflow-terminal-branch-b"
    )
    assert result.requested_goal_report is not None
    preferred_outcome = result.requested_goal_report.outcome_for(requested_state)
    assert preferred_outcome is not None
    assert preferred_outcome.status is RequestedGoalStatus.SATISFIED
    assert result.terminal_branch_report is not None
    assert len(result.terminal_branch_report.outcomes) == 1
    assert result.terminal_branch_report.preferred_node_id.value == 1


def test_process_structure_rejects_dominated_sibling_branch_by_issue_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling branches with strictly worse issues should be pruned before replay."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-reject-root",
    )
    worse_transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
    )
    better_transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=2),)),
    )
    planner_calls: list[str] = []

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        planner_calls.append(current_structure.provenance.ingress.source_name or "")
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(worse_transformer, better_transformer),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        if transformer is worse_transformer:
            return replace(
                result,
                structure=result.structure.with_provenance(
                    StructureProvenance(
                        ingress=StructureIngress(
                            source_format=(
                                result.structure.provenance.ingress.source_format
                            ),
                            source_name="workflow-reject-worse",
                        ),
                        lineage=result.structure.provenance.lineage,
                    )
                ),
                issues=result.issues
                + (
                    ValidationIssue.for_residue(
                        kind=ValidationIssueKind.STERIC_CLASH,
                        severity=IssueSeverity.WARNING,
                        message="worse-branch-warning",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                    ),
                ),
            )

        return replace(
            result,
            structure=result.structure.with_provenance(
                StructureProvenance(
                    ingress=StructureIngress(
                        source_format=(
                            result.structure.provenance.ingress.source_format
                        ),
                        source_name="workflow-reject-better",
                    ),
                    lineage=result.structure.provenance.lineage,
                )
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(structure)

    assert planner_calls == [
        "workflow-reject-root",
        "workflow-reject-better",
    ]
    assert result.structure.provenance.ingress.source_name == "workflow-reject-better"
    assert result.terminal_branch_report is not None
    assert len(result.terminal_branch_report.outcomes) == 1


def test_process_structure_prefers_goal_progress_when_pruning_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling pruning should prefer branches that satisfy requested goals."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-goal-pruning-root",
    )
    requested_state = requested_process_goal(
        scope=ResidueBoundaryScope(
            residue_id=ResidueId(chain_id="A", seq_num=1),
            side=ResidueBoundarySide.C_TERMINUS,
        ),
        value=OxtPresenceState.PRESENT,
    )
    unmet_transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=2),)),
    )
    satisfying_transformer = TerminalAugmentationTransformer(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
    )
    planner_calls: list[str] = []

    def fake_plan_workflow_actions(
        current_structure: ProteinStructure,
        *,
        requested_goals: RequestedGoalSet,
        transform_requests: WorkflowTransformRequests,
        component_library=None,
        planner_memory: WorkflowPlannerMemory | None = None,
        planning_context=None,
        retained_non_polymer_chemistry_evidence=(),
    ) -> WorkflowPlanningOutcome:
        del (
            requested_goals,
            transform_requests,
            component_library,
            planning_context,
            retained_non_polymer_chemistry_evidence,
        )
        planner_calls.append(current_structure.provenance.ingress.source_name or "")
        active_planner_memory = (
            WorkflowPlannerMemory() if planner_memory is None else planner_memory
        )
        if not active_planner_memory.adopted_transformers:
            return WorkflowPlanningOutcome(
                structure_planning_signature=StructurePlanningSignature.from_facts(
                    StructureProjectionStateFacts.from_structure(current_structure)
                ),
                transformers=(unmet_transformer, satisfying_transformer),
            )

        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(current_structure)
            ),
            transformers=(),
        )

    def fake_execute_workflow_transformer(
        result: ProcessResult,
        *,
        transformer: WorkflowStateAction,
        execution_context,
    ) -> ProcessResult:
        del execution_context
        if transformer is unmet_transformer:
            return replace(
                result,
                structure=result.structure.with_provenance(
                    StructureProvenance(
                        ingress=StructureIngress(
                            source_format=(
                                result.structure.provenance.ingress.source_format
                            ),
                            source_name="workflow-goal-pruning-unmet",
                        ),
                        lineage=result.structure.provenance.lineage,
                    )
                ),
            )

        return replace(
            result,
            structure=build_workflow_structure(
                chains=(
                    build_chain(
                        "A",
                        (
                            build_residue(
                                "GLY",
                                "A",
                                1,
                                ("N", "CA", "C", "O", "OXT"),
                            ),
                        ),
                    ),
                ),
                ligands=tuple(
                    (
                        ligand,
                        result.structure.residue_geometry(
                            result.structure.constitution.residue_index(
                                ligand.residue_id
                            )
                        ),
                        result.structure.residue_formal_charge_by_atom_name(
                            result.structure.constitution.residue_index(
                                ligand.residue_id
                            )
                        ),
                    )
                    for ligand in result.structure.constitution.ligands
                ),
                source_format=result.structure.provenance.ingress.source_format,
                source_name="workflow-goal-pruning-satisfied",
            ),
            issues=result.issues
            + (
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.STERIC_CLASH,
                    severity=IssueSeverity.WARNING,
                    message="goal-progress-warning",
                    residue_id=ResidueId(chain_id="A", seq_num=1),
                ),
            ),
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.plan_workflow_actions",
        fake_plan_workflow_actions,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.execute_workflow_transformer",
        fake_execute_workflow_transformer,
    )

    result = process_structure(
        structure,
        requested_goals=(requested_state,),
    )

    assert planner_calls == [
        "workflow-goal-pruning-root",
        "workflow-goal-pruning-satisfied",
    ]
    assert (
        result.structure.provenance.ingress.source_name
        == "workflow-goal-pruning-satisfied"
    )
    assert result.requested_goal_report is not None
    preferred_outcome = result.requested_goal_report.outcome_for(requested_state)
    assert preferred_outcome is not None
    assert preferred_outcome.status is RequestedGoalStatus.SATISFIED
    assert result.terminal_branch_report is not None
    assert len(result.terminal_branch_report.outcomes) == 1


def test_process_structure_applies_local_refinement_after_hydrogenation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow should execute standalone refinement after hydrogenation."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-hydrogenation",
    )
    local_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )

    calls: list[str] = []

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: DisabledHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not prepare_heavy_atoms
        assert target_residue_ids == frozenset({ResidueId(chain_id="A", seq_num=1)})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert isinstance(histidine_protonation, DisabledHistidineProtonationRequest)
        assert local_refinement is None
        calls.append("hydrogen")
        return ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_apply_repair_stage_local_refinement(
        result: ProcessResult,
        *,
        local_refinement: RepairLocalRefinementDirective,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence=(),
    ) -> ProcessResult:
        assert component_library is not None
        assert retained_non_polymer_chemistry_evidence == ()
        assert local_refinement == local_refinement_spec
        assert allow_retained_non_polymer_rdkit_fallback
        calls.append("refine")
        return result

    local_refinement_spec = residuewise_local_refinement_directive(local_refinement)
    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.local_refinement.apply_repair_stage_local_refinement",
        fake_apply_repair_stage_local_refinement,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=local_refinement.scope_spec,
                config=local_refinement.config,
                binding=local_refinement.binding,
            ),
        ),
    )

    assert result.structure is structure
    assert calls == ["hydrogen", "refine"]


def test_process_structure_applies_local_prerequisites_before_explicit_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit local refinement should stage heavy and hydrogen prerequisites."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-explicit-prereqs",
    )
    heavy_completed_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-explicit-prereqs-heavy",
    )
    hydrogenated_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
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
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-local-refinement-explicit-prereqs-hydrogenated",
    )
    local_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )

    calls: list[str] = []

    def fake_repair_heavy_atoms_core(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        augment_c_terminal_oxt: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not augment_c_terminal_oxt
        assert target_residue_ids == frozenset({residue_id})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert local_refinement is None
        calls.append("heavy")
        return ProcessResult(
            structure=heavy_completed_structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: DisabledHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert structure is heavy_completed_structure
        assert not prepare_heavy_atoms
        assert target_residue_ids == frozenset({residue_id})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert isinstance(histidine_protonation, DisabledHistidineProtonationRequest)
        assert local_refinement is None
        calls.append("hydrogen")
        return ProcessResult(
            structure=hydrogenated_structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_apply_repair_stage_local_refinement(
        result: ProcessResult,
        *,
        local_refinement: RepairLocalRefinementDirective,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence=(),
    ) -> ProcessResult:
        assert component_library is not None
        assert retained_non_polymer_chemistry_evidence == ()
        assert result.structure is hydrogenated_structure
        assert local_refinement == local_refinement_spec
        assert allow_retained_non_polymer_rdkit_fallback
        calls.append("refine")
        return result

    local_refinement_spec = residuewise_local_refinement_directive(local_refinement)
    monkeypatch.setattr(
        "protrepair.workflow.actions.heavy_completion.repair_heavy_atoms_core",
        fake_repair_heavy_atoms_core,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.local_refinement.apply_repair_stage_local_refinement",
        fake_apply_repair_stage_local_refinement,
    )

    result = process_structure(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=local_refinement.scope_spec,
                config=local_refinement.config,
                binding=local_refinement.binding,
            ),
        ),
    )

    assert result.structure is hydrogenated_structure
    assert calls == ["heavy", "hydrogen", "refine"]


def test_process_structure_recommended_policy_defers_binding_until_hydrogens_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recommended policy should carry an unbound directive into hydrogenation."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-recommended-mmff",
    )
    local_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )

    calls: list[str] = []

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: DisabledHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not prepare_heavy_atoms
        assert target_residue_ids == frozenset({ResidueId(chain_id="A", seq_num=1)})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert isinstance(histidine_protonation, DisabledHistidineProtonationRequest)
        assert local_refinement is None
        calls.append("hydrogen")
        return ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_apply_repair_stage_local_refinement(
        result: ProcessResult,
        *,
        local_refinement: RepairLocalRefinementDirective,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence=(),
    ) -> ProcessResult:
        assert component_library is not None
        assert retained_non_polymer_chemistry_evidence == ()
        assert local_refinement is not local_refinement_spec
        assert allow_retained_non_polymer_rdkit_fallback
        assert isinstance(local_refinement.selected_scope, ResidueSetScope)
        assert local_refinement.selected_scope.residue_ids == (
            ResidueId(chain_id="A", seq_num=1),
        )
        assert isinstance(
            local_refinement.binding,
            RecommendedContinuousRelaxationBinding,
        )
        calls.append("refine")
        return result

    local_refinement_spec = residuewise_local_refinement_directive(local_refinement)
    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.local_refinement.apply_repair_stage_local_refinement",
        fake_apply_repair_stage_local_refinement,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=local_refinement.scope_spec,
                config=local_refinement.config,
                binding=RecommendedContinuousRelaxationBinding(),
            ),
        ),
    )

    assert result.structure is structure
    assert isinstance(local_refinement.binding, ManualContinuousRelaxationBinding)
    assert local_refinement.binding.force_field is ContinuousRelaxationForceField.UFF
    assert calls == ["hydrogen", "refine"]


def test_process_structure_uses_composite_hydrogen_workflow_for_heavy_incomplete_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heavy-incomplete hydrogenation should execute heavy then hydrogen stages."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-composite-hydrogenation",
    )

    calls: list[tuple[str, frozenset[ResidueId], bool | None]] = []

    def fake_repair_heavy_atoms_core(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        augment_c_terminal_oxt: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not augment_c_terminal_oxt
        assert target_residue_ids == frozenset({ResidueId(chain_id="A", seq_num=1)})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert local_refinement is None
        assert target_residue_ids is not None
        calls.append(("heavy", target_residue_ids, None))
        return ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: DisabledHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not prepare_heavy_atoms
        assert target_residue_ids == frozenset({ResidueId(chain_id="A", seq_num=1)})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert isinstance(histidine_protonation, DisabledHistidineProtonationRequest)
        assert local_refinement is None
        assert target_residue_ids is not None
        calls.append(("hydrogen", target_residue_ids, prepare_heavy_atoms))
        return ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    monkeypatch.setattr(
        "protrepair.workflow.actions.heavy_completion.repair_heavy_atoms_core",
        fake_repair_heavy_atoms_core,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
    )

    assert result.structure is structure
    assert calls == [
        ("heavy", frozenset({ResidueId(chain_id="A", seq_num=1)}), None),
        ("hydrogen", frozenset({ResidueId(chain_id="A", seq_num=1)}), False),
    ]


def test_process_structure_uses_partitioned_hydrogen_route_for_heterogeneous_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heterogeneous subsets should hydrogenate only the supported subset."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("UNK", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-partitioned-hydrogenation",
    )

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: DisabledHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not prepare_heavy_atoms
        assert target_residue_ids == frozenset({ResidueId(chain_id="A", seq_num=1)})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert isinstance(histidine_protonation, DisabledHistidineProtonationRequest)
        assert local_refinement is None
        return ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
    )

    assert result.structure is structure
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.MISSING_COMPONENT_DEFINITION,
    )


def test_process_structure_defers_recommended_binding_for_unsupported_domain() -> None:
    """Unsupported workflow domains should stop before any stage execution."""

    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("UNK", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-recommended-unsupported-uff",
    )
    local_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.MMFF),
    )
    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=local_refinement.scope_spec,
                config=local_refinement.config,
                binding=RecommendedContinuousRelaxationBinding(),
            ),
        ),
    )

    assert result.structure is structure
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.MISSING_COMPONENT_DEFINITION,
    )
    assert isinstance(local_refinement.binding, ManualContinuousRelaxationBinding)
    assert local_refinement.binding.force_field is ContinuousRelaxationForceField.MMFF


def test_process_structure_recommended_policy_defers_binding_for_heavy_only_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recommended workflow policy should bind only after heavy and H stages."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-recommended-uff",
    )
    heavy_completed_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-recommended-uff-heavy",
    )
    hydrogenated_structure = build_workflow_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
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
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="workflow-recommended-uff-hydrogenated",
    )
    local_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.MMFF),
    )

    calls: list[str] = []

    def fake_repair_heavy_atoms_core(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        augment_c_terminal_oxt: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert not augment_c_terminal_oxt
        assert target_residue_ids == frozenset({residue_id})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert local_refinement is None
        calls.append("heavy")
        return ProcessResult(
            structure=heavy_completed_structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_add_hydrogens(
        structure: ProteinStructure,
        component_library=None,
        reference_structure=None,
        *,
        prepare_heavy_atoms: bool,
        target_residue_ids=None,
        orphan_fragment_policy: OrphanFragmentPolicy,
        histidine_protonation: DisabledHistidineProtonationRequest,
        local_refinement: RepairLocalRefinementDirective | None = None,
    ) -> ProcessResult:
        assert component_library is not None
        assert reference_structure is None
        assert structure is heavy_completed_structure
        assert not prepare_heavy_atoms
        assert target_residue_ids == frozenset({residue_id})
        assert orphan_fragment_policy is OrphanFragmentPolicy.REBUILD
        assert isinstance(histidine_protonation, DisabledHistidineProtonationRequest)
        assert local_refinement is None
        calls.append("hydrogen")
        return ProcessResult(
            structure=hydrogenated_structure,
            repairs=(),
            issues=(),
            analyses=None,
        )

    def fake_apply_repair_stage_local_refinement(
        result: ProcessResult,
        *,
        local_refinement: RepairLocalRefinementDirective,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence=(),
    ) -> ProcessResult:
        assert component_library is not None
        assert retained_non_polymer_chemistry_evidence == ()
        assert result.structure is hydrogenated_structure
        assert local_refinement is not local_refinement_spec
        assert allow_retained_non_polymer_rdkit_fallback
        assert isinstance(local_refinement.selected_scope, ResidueSetScope)
        assert local_refinement.selected_scope.residue_ids == (residue_id,)
        assert isinstance(
            local_refinement.binding,
            RecommendedContinuousRelaxationBinding,
        )
        calls.append("refine")
        return result

    local_refinement_spec = residuewise_local_refinement_directive(local_refinement)
    monkeypatch.setattr(
        "protrepair.workflow.actions.heavy_completion.repair_heavy_atoms_core",
        fake_repair_heavy_atoms_core,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.hydrogen_completion.add_hydrogens",
        fake_add_hydrogens,
    )
    monkeypatch.setattr(
        "protrepair.workflow.actions.local_refinement.apply_repair_stage_local_refinement",
        fake_apply_repair_stage_local_refinement,
    )

    result = process_structure(
        structure,
        requested_goals=whole_structure_requested_goals(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=local_refinement.scope_spec,
                config=local_refinement.config,
                binding=RecommendedContinuousRelaxationBinding(),
            ),
        ),
    )

    assert result.structure is hydrogenated_structure
    assert isinstance(local_refinement.binding, ManualContinuousRelaxationBinding)
    assert local_refinement.binding.force_field is ContinuousRelaxationForceField.MMFF
    assert calls == ["heavy", "hydrogen", "refine"]


def build_workflow_structure(
    *,
    chains: tuple[CanonicalChainPayload, ...],
    ligands: tuple[CanonicalResiduePayload, ...] = (),
    source_format: FileFormat,
    source_name: str,
) -> ProteinStructure:
    """Build one canonical workflow test structure from facet payloads."""

    return build_canonical_structure(
        chains=chains,
        ligands=ligands,
        source_format=source_format,
        source_name=source_name,
    )


def ligand_component_ids(structure: ProteinStructure) -> tuple[str, ...]:
    """Return ligand component ids in canonical ligand order."""

    return tuple(
        ligand_site.component_id for ligand_site in structure.constitution.ligands
    )


def has_atom(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name: str,
) -> bool:
    """Return whether one canonical atom site exists on the structure."""

    return (
        structure.constitution.resolve_atom_site(
            AtomRef(residue_id=residue_id, atom_name=atom_name)
        )
        is not None
    )


def atom_position(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name: str,
) -> Vec3:
    """Return one canonical atom position or raise if it is missing."""

    atom_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(
            AtomRef(residue_id=residue_id, atom_name=atom_name)
        )
    )

    return atom_geometry.position


def _focused_inverted_threonine_structure() -> ProteinStructure:
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
    return build_canonical_structure(
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
        source_format=FileFormat.PDB,
        source_name="workflow-process-inverted-thr",
    )


def build_chain(
    chain_id: str,
    residues: tuple[CanonicalResiduePayload, ...],
) -> CanonicalChainPayload:
    """Build one canonical chain payload for workflow tests."""

    return chain_payload(chain_id, residues)


def residuewise_local_refinement_directive(
    local_refinement: RepairRefinementSpec,
) -> RepairLocalRefinementDirective:
    """Return the canonical repair-local directive for one residuewise spec."""

    return RepairLocalRefinementDirective.from_residue_ids(
        local_refinement.scope_spec.referenced_residue_ids(),
        config=local_refinement.config,
        binding=local_refinement.binding,
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
    *,
    is_hetero: bool = False,
    positions_by_name: dict[str, Vec3] | None = None,
) -> CanonicalResiduePayload:
    """Build one canonical residue payload for workflow tests."""

    atoms = tuple(
        build_atom(
            atom_name,
            atom_index,
            position=(
                positions_by_name.get(atom_name)
                if positions_by_name is not None
                else None
            ),
        )
        for atom_index, atom_name in enumerate(atom_names, start=1)
    )
    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=is_hetero,
    )


def build_atom(
    atom_name: str,
    atom_index: int,
    *,
    position: Vec3 | None = None,
) -> CanonicalAtomPayload:
    """Build one canonical atom payload with deterministic coordinates."""

    preset_positions = (
        Vec3(0.000, 0.000, 0.000),
        Vec3(1.458, 0.000, 0.000),
        Vec3(2.028, 1.417, 0.000),
        Vec3(3.235, 1.593, 0.248),
        Vec3(1.145, -0.842, 1.074),
        Vec3(2.318, -1.152, 1.556),
    )
    resolved_position = (
        position
        if position is not None
        else preset_positions[(atom_index - 1) % len(preset_positions)]
    )

    return atom_payload(
        atom_name,
        infer_element(atom_name),
        resolved_position,
        occupancy=1.0,
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple test element from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
