"""Known reference chemistry must not be replaced by proximity inference."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    residue_payload,
)
from tests.support.request_builders import whole_structure_requested_goals
from tests.support.retained_reference_fixtures import LIGAND, reference_structure

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.nonstandard.registry import (
    build_bundled_nonstandard_registry,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.scope import ResidueSetScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    TopologyAvailabilityState,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryFailureKind,
    RetainedNonPolymerChemistryResolution,
)
from protrepair.state.structure_projection import (
    ProteinStructureObservation,
    StructureProjectionStateFacts,
)
from protrepair.structure import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.readiness import (
    ContinuousRelaxationReadinessPolicy,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.repair_stage import (
    apply_repair_stage_local_refinement,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.planning import WorkflowPlanningContext
from protrepair.workflow.engine.hydrogen import project_hydrogen_issues
from protrepair.workflow.engine.reporting import (
    evaluate_requested_goal_report,
    evaluate_workflow_branch_quality_score,
)


@pytest.mark.parametrize("component_id", ("3CF", "NLE", "02L"))
@pytest.mark.parametrize("with_h", (False, True))
@pytest.mark.parametrize("allow_fallback", (False, True))
def test_known_unprepared_component_is_unchanged_and_not_inferred(
    component_id: str, with_h: bool, allow_fallback: bool
) -> None:
    source = reference_structure(component_id, with_h=with_h)
    result = add_retained_non_polymer_hydrogens(
        source, allow_retained_non_polymer_rdkit_fallback=allow_fallback
    )
    assert result.structure == source
    assert not result.repairs
    assert len(result.issues) == (0 if with_h else 1)
    if not with_h:
        assert result.issues[0].kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        assert "admitted preparation" in result.issues[0].message
        assert "disabled" not in result.issues[0].message

    observation = StructureChemistryObservation(
        source,
        allow_retained_non_polymer_rdkit_fallback=allow_fallback,
    )
    fact = observation.whole_structure_facts()[1].retained_non_polymer_facts[0]
    assert fact.hydrogen_applicability_state is HydrogenApplicabilityState.APPLICABLE
    if with_h:
        assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    else:
        assert fact.hydrogen_coverage_state is HydrogenCoverageState.NONE
    assert not fact.requires_hydrogen_completion()
    assert not fact.requires_topology_resolution()
    assert not fact.hydrogen_generation_supported
    assert (
        observation.residue_completion_facts(LIGAND).hydrogen_coverage_state
        is fact.hydrogen_coverage_state
    )
    assert (
        observation.residue_facts(LIGAND).hydrogen_coverage_fact.value
        is fact.hydrogen_coverage_state
    )
    assert not observation.residue_completion_facts(LIGAND).is_supported()
    projection = StructureProjectionStateFacts.from_projection(
        context_structure=source,
        residues=source.constitution.ligands,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=allow_fallback,
    )
    assert projection.hydrogen_coverage_fact.value is fact.hydrogen_coverage_state
    assert (
        ProteinStructureObservation.from_facts(projection).hydrogen_coverage_state
        is fact.hydrogen_coverage_state
    )
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.NO_ADMITTED_PREPARATION
    )
    topology = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert topology.structure == source
    assert len(topology.issues) == len(result.issues)
    if topology.issues:
        assert topology.issues[0].kind is result.issues[0].kind


@pytest.mark.parametrize("component_id", ("3CF", "NLE", "02L"))
@pytest.mark.parametrize("with_h", (False, True))
def test_public_goal_explains_unprepared_known_chemistry(
    component_id: str, with_h: bool
) -> None:
    source = reference_structure(component_id, with_h=with_h)
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure == source
    assert result.requested_goals_fully_satisfied() is with_h
    assert result.requested_goal_report is not None
    outcome = next(
        outcome
        for outcome in result.requested_goal_report.outcomes
        if isinstance(outcome.requested_goal.value, HydrogenCoverageState)
    )
    assert outcome.observed_state is (
        HydrogenCoverageState.COMPLETE if with_h else HydrogenCoverageState.NONE
    )
    assert bool(outcome.blocking_scopes) is not with_h
    if not with_h:
        assert outcome.details is not None and "admitted preparation" in outcome.details
    issues = [
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
    ]
    assert len(issues) == (0 if with_h else 1)
    if issues:
        assert issues[0].residue_id == LIGAND


def test_public_file_h_preparation_never_infers_unprepared_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = Mock(side_effect=AssertionError("known chemistry must not be inferred"))
    for path in (
        "protrepair.state.retained_non_polymer_chemistry.infer_retained_non_polymer_rdkit_fallback",
        "protrepair.transformer.completion.retained_non_polymer_hydrogen.repair.infer_retained_non_polymer_rdkit_fallback",
    ):
        monkeypatch.setattr(path, inference)
    source = read_structure_string(
        write_structure_string(reference_structure("3CF", with_h=True), FileFormat.PDB),
        FileFormat.PDB,
    )
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure == source
    assert sum(a.is_hydrogen() for a in source.constitution.ligands[0].atom_sites) == 10
    inference.assert_not_called()


@pytest.mark.parametrize(
    "charges, expected_atom",
    (
        ((("CE", 1),), "CE"),
        ((("N", 1), ("OXT", -1)), "N"),
    ),
    ids=("carbon-charge", "legacy-zwitterion-diagnostic"),
)
def test_guard_preserves_independent_source_charge_diagnostics(
    charges: tuple[tuple[str, int], ...], expected_atom: str
) -> None:
    source = reference_structure("NLE", with_h=False)
    index = source.constitution.residue_index(LIGAND)
    charge_by_name = dict(source.residue_formal_charge_by_atom_name(index))
    charge_by_name.update(charges)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=tuple(charge_by_name.items()),
    )
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
    )
    assert result.structure == source
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        and issue.residue_id == LIGAND
        and expected_atom in issue.message
        and "source atom charges disagree" in issue.message
        for issue in result.issues
    )
    assert not any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        for issue in result.issues
    )


def test_custom_hydrogen_program_keeps_authority_over_bundled_identity() -> None:
    source = reference_structure("NLE", with_h=False)
    record = build_bundled_nonstandard_registry().get("NLE")
    assert record is not None
    library = build_default_component_library()
    custom = replace(
        library.require("NLE"),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(
            record.to_idealized_component()
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        source, component_library=library.with_template(custom)
    )
    assert not result.issues
    assert len(result.repairs) == 1
    assert (
        sum(
            a.is_hydrogen() for a in result.structure.constitution.ligands[0].atom_sites
        )
        == 13
    )


def test_explicit_chemistry_is_used_by_generation_and_public_goal() -> None:
    source = reference_structure("NLE", with_h=False)
    override = RetainedNonPolymerChemistryOverride(
        residue_id=LIGAND,
        smiles="N[C@@H](CCCC)C(=O)O",
        heavy_atom_names=("N", "CA", "CB", "CG", "CD", "CE", "C", "O", "OXT"),
    )
    result = process_structure(
        source,
        ingress=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=(override,),
        ),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.requested_goals_fully_satisfied()
    assert not any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        for issue in result.issues
    )


@pytest.mark.parametrize("history_count", (0, 1, 11))
def test_guard_warning_is_current_state_not_execution_count(history_count: int) -> None:
    source = reference_structure("3CF", with_h=False)
    library = build_default_component_library()
    issue = add_retained_non_polymer_hydrogens(source).issues[0]
    unrelated = ValidationIssue(
        kind=ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_BLOCKED,
        severity=IssueSeverity.WARNING,
        message="an independent user opt-out",
    )
    result = TransformationResult(
        source, repairs=(), issues=(unrelated, unrelated) + (issue,) * history_count
    )
    report = evaluate_requested_goal_report(
        source,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
        component_library=library,
    )
    projected = project_hydrogen_issues(
        result, requested_goal_report=report, component_library=library
    )
    assert projected == (unrelated, unrelated, issue)
    score = evaluate_workflow_branch_quality_score(
        result,
        requested_goal_report=report,
        planning_context=WorkflowPlanningContext(),
        component_library=library,
    )
    assert score.warning_count == score.issue_count == 3
    empty = evaluate_requested_goal_report(
        source, requested_goals=(), component_library=library
    )
    assert project_hydrogen_issues(
        result, requested_goal_report=empty, component_library=library
    ) == (unrelated, unrelated)


def test_indeterminate_coverage_cannot_admit_topology_or_refinement() -> None:
    source = reference_structure("3CF", with_h=True)
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    fact = replace(
        fact,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        hydrogen_generation_supported=True,
        heavy_atom_topology_availability_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_availability_state=TopologyAvailabilityState.PRESENT,
        hydrogen_coverage_state=None,
        reference_hydrogens_verified=False,
    )
    assert not fact.requires_hydrogen_completion()
    assert not fact.requires_topology_resolution()
    assert (
        ContinuousRelaxationReadinessPolicy().retained_non_polymer_readiness_blocker(
            (fact,)
        )
        is not None
    )


def test_partial_scaffold_keeps_reference_need_without_generating_atoms() -> None:
    source = reference_structure("3CF", with_h=False).without_atom_refs(
        (AtomRef(LIGAND, "C1"),)
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_applicability_state is HydrogenApplicabilityState.APPLICABLE
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.NONE


@pytest.mark.parametrize("isotope", ("D", "T"))
def test_source_isotopes_are_verified_without_modification(isotope: str) -> None:
    source = reference_structure("NLE", with_h=True, isotope=isotope)
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE


def test_scoped_h_evaluator_can_observe_verified_reference_inventory() -> None:
    source = reference_structure("3CF", with_h=True)
    report = evaluate_requested_goal_report(
        source,
        requested_goals=(
            requested_process_goal(
                scope=ResidueSetScope((LIGAND,)),
                value=HydrogenCoverageState.COMPLETE,
            ),
        ),
        component_library=build_default_component_library(),
    )
    outcome = report.outcomes[0]
    assert outcome.observed_state is HydrogenCoverageState.COMPLETE
    assert outcome.is_satisfied()
    assert not outcome.blocking_scopes


def test_full_reference_inventory_survives_all_nonstandard_template_projections() -> (
    None
):
    registry = build_bundled_nonstandard_registry()
    library = registry.component_library()
    for component_id, record in registry.records.items():
        template = library.require(component_id)
        assert template.reference_component == record.to_idealized_component()
        resolution = RetainedNonPolymerChemistryResolution.from_unprepared_reference(
            template
        )
        if template.can_add_hydrogens():
            assert resolution is None
        else:
            assert resolution is not None
            assert resolution.reference_requires_hydrogens is any(
                atom.is_hydrogen() for atom in record.atoms
            )
            assert not resolution.expected_hydrogen_atom_names
            assert not resolution.heavy_bond_definitions


@pytest.mark.parametrize("with_h", (False, True))
def test_local_prerequisites_refuse_guarded_component_without_ff(
    monkeypatch: pytest.MonkeyPatch,
    with_h: bool,
) -> None:
    source = reference_structure("NLE", with_h=with_h)
    ff = Mock(
        side_effect=AssertionError("unverified chemistry must not bind a force field")
    )
    monkeypatch.setattr(
        "protrepair.transformer.continuous.rdkit.build_rdkit_force_field", ff
    )
    directive = RepairLocalRefinementDirective.from_residue_ids(
        (LIGAND,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    result = apply_repair_stage_local_refinement(
        TransformationResult(source, repairs=(), issues=()),
        local_refinement=directive,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    assert result.structure == source
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
        and "admitted preparation" in issue.message
        for issue in result.issues
    ), tuple(issue.message for issue in result.issues)
    ff.assert_not_called()

    public = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((LIGAND,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )
    assert public.structure == source
    assert sum(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        for issue in public.issues
    ) == (0 if with_h else 1)
    assert not public.relaxation_executions
    ff.assert_not_called()


def test_unavailable_context_does_not_erase_known_reference_need(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = Mock(side_effect=ValueError("unsupported donor context"))
    monkeypatch.setattr(
        "protrepair.state.retained_non_polymer_chemistry.coordinated_component_preparation",
        unavailable,
    )
    monkeypatch.setattr(
        "protrepair.transformer.completion.retained_non_polymer_hydrogen.repair.coordinated_component_preparation",
        unavailable,
    )
    source = reference_structure("3CF", with_h=True)
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_applicability_state is HydrogenApplicabilityState.APPLICABLE
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.CONTEXT_UNRESOLVED
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    assert len(result.issues) == 1
    assert result.issues[0].kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
    assert "unsupported donor context" in result.issues[0].message


def test_guard_is_item_local_and_does_not_disable_unknown_ligand_fallback() -> None:
    known = reference_structure("3CF", with_h=False)
    residue = known.constitution.ligands[0]
    unknown_id = ResidueId("L", 601)
    source = build_bonded_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id=residue.component_id,
                residue_id=LIGAND,
                atoms=tuple(
                    atom_payload(
                        atom.name,
                        atom.element,
                        known.residue_geometry(
                            known.constitution.residue_index(LIGAND)
                        ).position(atom.name),
                    )
                    for atom in residue.atom_sites
                ),
            ),
            residue_payload(
                component_id="ZZQ",
                residue_id=unknown_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(30, 0, 0)),
                    atom_payload("O1", "O", Vec3(31.43, 0, 0)),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    result = add_retained_non_polymer_hydrogens(source)
    blocked = result.structure.constitution.residue_or_ligand(LIGAND)
    unknown = result.structure.constitution.residue_or_ligand(unknown_id)
    assert blocked is not None and not any(a.is_hydrogen() for a in blocked.atom_sites)
    assert unknown is not None and sum(a.is_hydrogen() for a in unknown.atom_sites) == 4
    assert {repair.residue_id for repair in result.repairs} == {unknown_id}
    assert {issue.kind for issue in result.issues} == {
        ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN,
        ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED,
    }
    library = build_default_component_library()
    report = evaluate_requested_goal_report(
        result.structure,
        component_library=library,
        requested_goals=(
            requested_process_goal(
                scope=ResidueSetScope((unknown_id,)),
                value=HydrogenCoverageState.COMPLETE,
            ),
        ),
    )
    assert ResidueSetScope((LIGAND,)) not in report.outcomes[0].blocking_scopes
    assert not any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        for issue in project_hydrogen_issues(
            result, requested_goal_report=report, component_library=library
        )
    )
