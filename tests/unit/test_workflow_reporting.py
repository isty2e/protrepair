"""Workflow reporting tests over narrow fact owners and phase reports."""

from unittest.mock import patch

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.correction_state_fixtures import load_refinement_fixture
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.scenario_fixture_matrix import SCENARIO_FIXTURE_MATRIX

from protrepair.api import process_structure
from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    ClashObservationMode,
    ClashPresenceState,
    HydrogenCoverageState,
    OrientationCorrectionEligibilityState,
    ParserCompatibilityProfile,
    ParserCompatibilityState,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
    StructureChemistryReadinessFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
    chemistry_observation,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.workflow.contracts import (
    LigandPolicy,
    RequestedGoalStatus,
    StructureIngressOptions,
    WorkflowLigandContextMode,
    WorkflowPhaseStatus,
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowGoalStateValue,
)
from protrepair.workflow.contracts.result import WorkflowBranchQualityScore
from protrepair.workflow.engine.reporting import (
    evaluate_requested_goal_report,
    evaluate_workflow_phase_report,
)
from protrepair.workflow.planning.assessment.blockers import WorkflowBlocker
from protrepair.workflow.planning.capability import WorkflowCapabilityDeficitFamily


def _goal_identity_structure(
    *, missing_sidechain: bool = False, retained_ligand: bool = False
) -> ProteinStructure:
    atoms = (
        atom_payload("N", "N", Vec3(-0.528, 1.365, 0.0)),
        atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
        atom_payload("C", "C", Vec3(1.526, 0.0, 0.0)),
        atom_payload("O", "O", Vec3(2.144, 1.052, 0.0)),
        atom_payload("OXT", "O", Vec3(2.070, -1.070, 0.0)),
    )
    if not missing_sidechain:
        atoms += (atom_payload("CB", "C", Vec3(-0.525, -0.773, -1.205)),)
    ligand = residue_payload(
        component_id="UNK",
        residue_id=ResidueId("L", 99),
        atoms=(
            atom_payload("C1", "C", Vec3(10.0, 0.0, 0.0)),
            atom_payload("O1", "O", Vec3(11.4, 0.0, 0.0)),
        ),
        is_hetero=True,
    )
    return build_bonded_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA", residue_id=ResidueId("A", 1), atoms=atoms
                    ),
                ),
            ),
        ),
        ligands=(ligand,) if retained_ligand else (),
        source_format=FileFormat.PDB,
        source_name="goal-identity",
    )


@pytest.mark.parametrize("missing_sidechain", (False, True))
@pytest.mark.parametrize(
    "values",
    (
        (HydrogenCoverageState.COMPLETE,),
        (
            BackboneHeavyAtomCompletenessState.COMPLETE,
            SidechainHeavyAtomCompletenessState.COMPLETE,
            HydrogenCoverageState.COMPLETE,
        ),
        (
            HydrogenCoverageState.COMPLETE,
            SidechainHeavyAtomCompletenessState.COMPLETE,
            BackboneHeavyAtomCompletenessState.COMPLETE,
        ),
    ),
)
def test_combined_goal_statuses_keep_each_axes_initial_state(
    missing_sidechain: bool, values: tuple[WorkflowGoalStateValue, ...]
) -> None:
    source = _goal_identity_structure(missing_sidechain=missing_sidechain)
    requested = tuple(
        requested_process_goal(scope=WholeStructureScope(), value=value)
        for value in values
    )
    result = process_structure(source, requested_goals=requested)
    report = result.requested_goal_report
    assert report is not None
    expected: dict[type[WorkflowGoalStateValue], RequestedGoalStatus] = {
        BackboneHeavyAtomCompletenessState: RequestedGoalStatus.ALREADY_SATISFIED,
        SidechainHeavyAtomCompletenessState: RequestedGoalStatus.SATISFIED
        if missing_sidechain
        else RequestedGoalStatus.ALREADY_SATISFIED,
        HydrogenCoverageState: RequestedGoalStatus.SATISFIED,
    }
    assert len(report.outcomes) == len(requested)
    assert result.repairs
    for goal, outcome in zip(requested, report.outcomes, strict=True):
        assert outcome.requested_goal.value is goal.value
        assert outcome.status is expected[type(goal.value)]
        assert report.outcome_for(goal) is outcome

    repeated = process_structure(result.structure, requested_goals=requested)
    assert repeated.requested_goal_report is not None
    assert all(
        outcome.status is RequestedGoalStatus.ALREADY_SATISFIED
        for outcome in repeated.requested_goal_report.outcomes
    )


def test_goal_normalization_and_set_identity_preserve_independent_axes() -> None:
    goals = tuple(
        requested_process_goal(scope=WholeStructureScope(), value=value)
        for value in (
            BackboneHeavyAtomCompletenessState.COMPLETE,
            SidechainHeavyAtomCompletenessState.COMPLETE,
            HydrogenCoverageState.COMPLETE,
        )
    )
    normalized = RequestedGoalSet((*goals, goals[-1]))

    assert len(normalized) == 3
    assert len(set(normalized)) == 3
    assert tuple(type(goal.value) for goal in normalized) == tuple(
        type(goal.value) for goal in goals
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_unsupported_goal_classification_does_not_transfer_to_other_axes(
    reverse: bool,
) -> None:
    source = _goal_identity_structure()
    goals = tuple(
        requested_process_goal(scope=WholeStructureScope(), value=value)
        for value in (
            BackboneHeavyAtomCompletenessState.COMPLETE,
            SidechainHeavyAtomCompletenessState.COMPLETE,
            HydrogenCoverageState.COMPLETE,
        )
    )
    report = evaluate_requested_goal_report(
        source,
        requested_goals=goals[::-1] if reverse else goals,
        unsupported_requested_goals=(goals[-1],),
    )
    for outcome in report.outcomes:
        expected = (
            RequestedGoalStatus.UNSUPPORTED
            if isinstance(outcome.requested_goal.value, HydrogenCoverageState)
            else RequestedGoalStatus.SATISFIED
        )
        assert outcome.status is expected
        assert report.outcome_for(outcome.requested_goal) is outcome
    assert report.satisfied_count() == 2
    assert report.unsupported_count() == 1
    assert report.completion_verdict().value == "partially_achieved"


@pytest.mark.parametrize("reverse", (False, True))
def test_retained_hydrogen_failure_stays_on_its_requested_axis(reverse: bool) -> None:
    source = _goal_identity_structure(retained_ligand=True)
    goals = tuple(
        requested_process_goal(scope=WholeStructureScope(), value=value)
        for value in (
            BackboneHeavyAtomCompletenessState.COMPLETE,
            SidechainHeavyAtomCompletenessState.COMPLETE,
            HydrogenCoverageState.COMPLETE,
        )
    )
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=goals[::-1] if reverse else goals,
        transform_requests=WorkflowTransformRequests(
            allow_retained_non_polymer_rdkit_fallback=False
        ),
    )
    report = result.requested_goal_report
    assert report is not None
    for outcome in report.outcomes:
        expected = (
            RequestedGoalStatus.UNMET
            if isinstance(outcome.requested_goal.value, HydrogenCoverageState)
            else RequestedGoalStatus.ALREADY_SATISFIED
        )
        assert outcome.status is expected
        assert report.outcome_for(outcome.requested_goal) is outcome
        if expected is RequestedGoalStatus.ALREADY_SATISFIED:
            assert not outcome.blocking_scopes
            assert not outcome.blocking_phases
    assert report.satisfied_count() == 2
    assert report.unmet_count() == 1
    assert report.completion_verdict().value == "partially_achieved"


@pytest.mark.parametrize("reverse", (False, True))
def test_explicit_blocker_projection_does_not_transfer_to_other_axes(
    reverse: bool,
) -> None:
    source = _goal_identity_structure(retained_ligand=True)
    goals = tuple(
        requested_process_goal(scope=WholeStructureScope(), value=value)
        for value in (
            BackboneHeavyAtomCompletenessState.COMPLETE,
            SidechainHeavyAtomCompletenessState.COMPLETE,
            HydrogenCoverageState.COMPLETE,
        )
    )
    blocker = WorkflowBlocker.unsupported_component(
        deficit_family=WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
        scope=ResidueSetScope((ResidueId("L", 99),)),
    )
    report = evaluate_requested_goal_report(
        source,
        requested_goals=goals[::-1] if reverse else goals,
        blocked_requested_goal_blockers=((goals[-1], (blocker,)),),
    )
    for outcome in report.outcomes:
        if isinstance(outcome.requested_goal.value, HydrogenCoverageState):
            assert outcome.status is RequestedGoalStatus.BLOCKED
            assert outcome.blocking_scopes == (blocker.scope,)
            assert outcome.blocking_phases == (blocker.phase,)
        else:
            assert outcome.status is RequestedGoalStatus.SATISFIED
            assert not outcome.blocking_scopes
            assert not outcome.blocking_phases
        assert report.outcome_for(outcome.requested_goal) is outcome
    assert report.satisfied_count() == 2
    assert report.blocked_count() == 1


def test_phase_report_derives_joint_facts_once_without_cross_call_reuse() -> None:
    source = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    library = build_default_component_library()
    with patch.object(
        chemistry_observation,
        "StructureHydrogenExpectationModel",
        wraps=chemistry_observation.StructureHydrogenExpectationModel,
    ) as derive:
        supported = evaluate_workflow_phase_report(
            source,
            component_library=library,
            planning_context=WorkflowPlanningContext(),
            blockers=(),
        )
        assert derive.call_count == 1
        unsupported = evaluate_workflow_phase_report(
            source,
            component_library=ComponentLibrary(),
            planning_context=WorkflowPlanningContext(),
            blockers=(),
        )
        assert derive.call_count == 2
        repeated = evaluate_workflow_phase_report(
            source,
            component_library=library,
            planning_context=WorkflowPlanningContext(),
            blockers=(),
        )
        assert derive.call_count == 3

    assert supported.outcomes == repeated.outcomes
    phase = WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION
    supported_chemistry = supported.outcome_for(phase)
    unsupported_chemistry = unsupported.outcome_for(phase)
    assert supported_chemistry is not None and unsupported_chemistry is not None
    assert unsupported_chemistry.status is WorkflowPhaseStatus.UNRESOLVED
    assert supported_chemistry != unsupported_chemistry


def test_whole_heavy_goal_report_does_not_build_boundary_or_atom_scope_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-structure heavy goals need no boundary or atom-scope observation."""

    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=SidechainHeavyAtomCompletenessState.COMPLETE,
    )
    structure = build_structure(
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
        source_name="workflow-reporting-coverage-only",
    )

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unrelated fact family should not be built")

    monkeypatch.setattr(
        "protrepair.workflow.planning.assessment.goal_support.StructureBoundaryStateFacts.from_structure",
        _fail,
    )
    monkeypatch.setattr(
        "protrepair.workflow.planning.assessment.goal_support.ProteinStructureSnapshot.from_structure",
        _fail,
    )

    report = evaluate_requested_goal_report(
        structure,
        requested_goals=(requested_state,),
        already_satisfied_requested_goals=(requested_state,),
    )
    outcome = report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.UNMET


def test_whole_hydrogen_goal_report_does_not_build_boundary_or_atom_scope_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-structure H goals need no boundary or atom-scope observation."""

    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
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
        source_name="workflow-reporting-chemistry-only",
    )

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unrelated fact family should not be built")

    monkeypatch.setattr(
        "protrepair.workflow.planning.assessment.goal_support.StructureBoundaryStateFacts.from_structure",
        _fail,
    )
    monkeypatch.setattr(
        "protrepair.workflow.planning.assessment.goal_support.ProteinStructureSnapshot.from_structure",
        _fail,
    )

    report = evaluate_requested_goal_report(
        structure,
        requested_goals=(requested_state,),
    )
    outcome = report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.UNMET


def test_process_structure_reports_blocking_phase_for_matrix_blocked_scenario() -> None:
    """Blocked requested goals should preserve planner phase language."""

    assert "synthetic-unsupported-component-stop" in SCENARIO_FIXTURE_MATRIX
    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNK",
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
        source_name="workflow-reporting-blocked-coverage",
    )

    result = process_structure(
        structure,
        requested_goals=(requested_state,),
    )

    assert result.requested_goal_report is not None
    outcome = result.requested_goal_report.outcome_for(requested_state)
    assert outcome is not None
    assert outcome.status is RequestedGoalStatus.BLOCKED
    assert outcome.blocking_phases == (WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,)

    assert result.branch_report is not None
    coverage_phase = result.branch_report.preferred_outcome().phase_report.outcome_for(
        WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION
    )
    assert coverage_phase is not None
    assert coverage_phase.status is WorkflowPhaseStatus.BLOCKED


def test_terminal_branch_phase_report_distinguishes_matrix_failures() -> None:
    """Terminal reports should distinguish intrinsic and holo interaction failures."""

    assert "3g8l-asn182" in SCENARIO_FIXTURE_MATRIX
    assert "1jd0-gln92" in SCENARIO_FIXTURE_MATRIX

    intrinsic_structure = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    intrinsic_result = process_structure(intrinsic_structure)
    assert intrinsic_result.branch_report is not None
    intrinsic_outcome = intrinsic_result.branch_report.preferred_outcome()
    intrinsic_phase = intrinsic_outcome.phase_report.outcome_for(
        WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
    )
    interaction_phase = intrinsic_outcome.phase_report.outcome_for(
        WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION,
    )
    assert intrinsic_phase is not None
    assert intrinsic_phase.status is WorkflowPhaseStatus.UNRESOLVED
    assert interaction_phase is not None
    assert interaction_phase.status is WorkflowPhaseStatus.NOT_APPLICABLE

    interaction_structure = load_refinement_fixture("1jd0_gln92_local.pdb")
    interaction_result = process_structure(
        interaction_structure,
        ingress=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
        ),
        planning_context=WorkflowPlanningContext(
            ligand_context_mode=WorkflowLigandContextMode.CONSIDER_IF_PRESENT,
        ),
    )
    assert interaction_result.branch_report is not None
    interaction_outcome = interaction_result.branch_report.preferred_outcome()
    interaction_only_phase = interaction_outcome.phase_report.outcome_for(
        WorkflowPlanningPhase.INTERACTION_AWARE_CORRECTION,
    )
    assert interaction_only_phase is not None
    assert interaction_only_phase.status is WorkflowPhaseStatus.UNRESOLVED


def test_process_structure_defers_intrinsic_phase_report_for_hydrogen_complete_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal reporting should defer intrinsic phase evaluation until inspected."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
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
        source_name="workflow-reporting-deferred-intrinsic-phase",
    )
    hydrogenated_structure = add_hydrogens(
        structure,
        component_library=build_default_component_library(),
    ).structure

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("intrinsic phase facts should be evaluated lazily")

    monkeypatch.setattr(
        "protrepair.workflow.engine.reporting.derive_structure_intrinsic_geometry_facts",
        _fail,
    )
    monkeypatch.setattr(
        "protrepair.workflow.planning.local_refinement_policy.propose_joint_correction_scopes",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.runtime.evaluate_workflow_branch_quality_score",
        lambda *_args, **_kwargs: WorkflowBranchQualityScore(),
    )
    _patch_workflow_planning_parser_compatibility_facts(
        monkeypatch,
        ParserCompatibilityState.COMPATIBLE,
    )

    result = process_structure(hydrogenated_structure)

    assert result.branch_report is not None
    with pytest.raises(AssertionError, match="evaluated lazily"):
        result.branch_report.preferred_outcome().phase_report.outcome_for(
            WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION
        )


@pytest.mark.parametrize("hydrogenate", (False, True))
def test_terminal_branch_phase_report_surfaces_parser_incompatibility_as_intrinsic(
    monkeypatch: pytest.MonkeyPatch,
    hydrogenate: bool,
) -> None:
    """Parser compatibility is a geometry/reporting fact, not chemistry readiness."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
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
        source_name="workflow-reporting-parser-compatibility",
    )
    hydrogenated_structure = (
        add_hydrogens(
            structure,
            component_library=build_default_component_library(),
        ).structure
        if hydrogenate
        else structure
    )

    def _parser_incompatible(
        current_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> StructureParserCompatibilityFacts:
        del component_library
        return StructureParserCompatibilityFacts(
            carrier=current_structure,
            profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
            compatibility_state=ParserCompatibilityState.INCOMPATIBLE,
            extra_proximity_bond_count=1,
            extra_heavy_proximity_bond_count=1,
        )

    def _intrinsic_geometry_clear(
        current_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
        chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None,
    ) -> StructureIntrinsicGeometryFacts:
        del component_library
        del chemistry_readiness_facts
        return StructureIntrinsicGeometryFacts(
            carrier=current_structure,
            protein_self_clash_state=ClashPresenceState.NONE,
            protein_self_clash_count=0,
            protein_self_clash_observation_mode=(
                ClashObservationMode.ALL_ATOM_COMPLETE
            ),
            observed_heavy_atom_self_clash_count=0,
            observed_hydrogen_inclusive_self_clash_count=0,
            orientation_correction_eligibility_state=(
                OrientationCorrectionEligibilityState.NOT_ELIGIBLE
            ),
            stereochemistry_state=StereochemistryState.CONSISTENT,
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.reporting.derive_structure_intrinsic_geometry_facts",
        _intrinsic_geometry_clear,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.reporting.StructureParserCompatibilityFacts.from_structure",
        staticmethod(_parser_incompatible),
    )
    monkeypatch.setattr(
        "protrepair.workflow.planning.local_refinement_policy.propose_joint_correction_scopes",
        lambda *_args, **_kwargs: (),
    )
    _patch_workflow_planning_parser_compatibility_facts(
        monkeypatch,
        ParserCompatibilityState.COMPATIBLE,
    )

    result = process_structure(hydrogenated_structure)

    assert result.branch_report is not None
    score = result.branch_report.preferred_outcome().branch_quality_score
    assert score.parser_incompatible == 1
    assert score.parser_extra_heavy_bond_count == 1
    intrinsic_phase = result.branch_report.preferred_outcome().phase_report.outcome_for(
        WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION
    )
    assert intrinsic_phase is not None
    assert intrinsic_phase.status is WorkflowPhaseStatus.UNRESOLVED
    assert intrinsic_phase.details == (
        "parser-visible proximity defects remain after terminal planning"
    )


def test_terminal_branch_phase_report_blocks_parser_incompatibility_on_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Topology-only parser blockers should not look like repairable geometry."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
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
        source_name="workflow-reporting-parser-topology-blocker",
    )
    hydrogenated_structure = add_hydrogens(
        structure,
        component_library=build_default_component_library(),
    ).structure
    blocker_scope = ResidueSetScope(residue_ids=(ResidueId("A", 1),))

    def _parser_incompatible(
        current_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> StructureParserCompatibilityFacts:
        del component_library
        return StructureParserCompatibilityFacts(
            carrier=current_structure,
            profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
            compatibility_state=ParserCompatibilityState.INCOMPATIBLE,
            extra_proximity_bond_count=1,
            extra_heavy_proximity_bond_count=1,
        )

    def _intrinsic_geometry_clear(
        current_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
        chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None,
    ) -> StructureIntrinsicGeometryFacts:
        del component_library
        del chemistry_readiness_facts
        return StructureIntrinsicGeometryFacts(
            carrier=current_structure,
            protein_self_clash_state=ClashPresenceState.NONE,
            protein_self_clash_count=0,
            protein_self_clash_observation_mode=(
                ClashObservationMode.ALL_ATOM_COMPLETE
            ),
            observed_heavy_atom_self_clash_count=0,
            observed_hydrogen_inclusive_self_clash_count=0,
            orientation_correction_eligibility_state=(
                OrientationCorrectionEligibilityState.NOT_ELIGIBLE
            ),
            stereochemistry_state=StereochemistryState.CONSISTENT,
        )

    monkeypatch.setattr(
        "protrepair.workflow.engine.reporting.derive_structure_intrinsic_geometry_facts",
        _intrinsic_geometry_clear,
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.reporting.StructureParserCompatibilityFacts.from_structure",
        staticmethod(_parser_incompatible),
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.reporting._parser_compatibility_blocking_scopes",
        lambda *_args, **_kwargs: (blocker_scope,),
    )
    monkeypatch.setattr(
        "protrepair.workflow.planning.local_refinement_policy.propose_joint_correction_scopes",
        lambda *_args, **_kwargs: (),
    )
    _patch_workflow_planning_parser_compatibility_facts(
        monkeypatch,
        ParserCompatibilityState.COMPATIBLE,
    )

    result = process_structure(hydrogenated_structure)

    assert result.branch_report is not None
    intrinsic_phase = result.branch_report.preferred_outcome().phase_report.outcome_for(
        WorkflowPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION
    )
    assert intrinsic_phase is not None
    assert intrinsic_phase.status is WorkflowPhaseStatus.BLOCKED
    assert intrinsic_phase.blocking_scopes == (blocker_scope,)
    assert intrinsic_phase.details == (
        "parser-visible proximity defects are blocked by topology ambiguity"
    )


def _patch_workflow_planning_parser_compatibility_facts(
    monkeypatch: pytest.MonkeyPatch,
    compatibility_state: ParserCompatibilityState,
) -> None:
    """Patch planner-time parser facts without replacing reporting facts."""

    class _PlanningParserCompatibilityFacts:
        @staticmethod
        def from_structure(
            current_structure: ProteinStructure,
            *,
            component_library: ComponentLibrary | None = None,
        ) -> StructureParserCompatibilityFacts:
            del component_library
            return StructureParserCompatibilityFacts(
                carrier=current_structure,
                profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
                compatibility_state=compatibility_state,
            )

    monkeypatch.setattr(
        "protrepair.workflow.planning.planner.StructureParserCompatibilityFacts",
        _PlanningParserCompatibilityFacts,
    )
