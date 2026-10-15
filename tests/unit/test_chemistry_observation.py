"""Input binding, projection and lifetime behavior of chemistry observations."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

import protrepair.state.chemistry_observation as observation_module
from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.scope import AtomSetScope, WholeStructureScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    ResidueProjectionStateFacts,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.hydrogen_expectation import (
    StructureHydrogenExpectationModel,
)
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
)
from protrepair.state.structure_projection import StructureProjectionStateFacts
from protrepair.state.structure_readiness import (
    derive_projection_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import AtomTopology, StructureTopology, TopologyBond
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import (
    RequestedGoalStatus,
    WorkflowPlanningContext,
    requested_process_goal,
)
from protrepair.workflow.engine.reporting import (
    evaluate_requested_goal_report,
    evaluate_workflow_branch_quality_score,
)
from protrepair.workflow.planning.assessment.goal_support import (
    already_satisfied_requested_goals,
)


def _ligand() -> ProteinStructure:
    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.4, 0, 0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _polymer() -> ProteinStructure:
    atoms = (
        ("N", "N", Vec3(0, 0, 0)),
        ("CA", "C", Vec3(1.5, 0, 0)),
        ("C", "C", Vec3(2.8, 0.4, 0)),
        ("O", "O", Vec3(3.8, 0.4, 0)),
        ("CB", "C", Vec3(1.5, 1.5, 0)),
    )
    return build_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", index),
                        atoms=tuple(
                            atom_payload(name, element, position)
                            for name, element, position in atoms[:count]
                        ),
                    )
                    for index, count in ((1, 5), (2, 4))
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _evidence(smiles: str):
    return (
        RetainedNonPolymerChemistryOverride(
            residue_id=ResidueId("L", 1),
            smiles=smiles,
            heavy_atom_names=("C1", "O1"),
        ).to_evidence(),
    )


def _facts_for_supplied_residue(consumer, source, residue, observation=None):
    if consumer == "singleton":
        return ResidueProjectionStateFacts.from_residue(source, residue)
    project = (
        derive_projection_coverage_and_chemistry_readiness_facts
        if consumer == "coverage"
        else StructureProjectionStateFacts.from_projection
    )
    return project(
        context_structure=source,
        residues=() if residue.is_hetero else (residue,),
        ligands=(residue,) if residue.is_hetero else (),
        chemistry_observation=observation,
    )


@pytest.mark.parametrize("consumer", ["singleton", "coverage", "projection"])
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize(
    "change", ["inventory", "component", "element", "role", "order"]
)
def test_supplied_residue_cannot_silently_replace_conflicting_content(
    consumer, retained, change
):
    source = _ligand() if retained else _polymer()
    residue = next(source.constitution.iter_residues(include_ligands=True))
    if change == "inventory":
        conflicting = replace(residue, atom_sites=residue.atom_sites[:-1])
    elif change == "component":
        conflicting = residue.with_component_id("GLY")
    elif change == "element":
        conflicting = residue.with_atom_site(
            replace(residue.atom_sites[0], element="P")
        )
    elif change == "role":
        conflicting = replace(residue, is_hetero=not residue.is_hetero)
    else:
        conflicting = replace(residue, atom_sites=tuple(reversed(residue.atom_sites)))

    with pytest.raises(ValueError, match="residue.*context"):
        _facts_for_supplied_residue(consumer, source, conflicting)


@pytest.mark.parametrize("consumer", ["singleton", "coverage", "projection"])
@pytest.mark.parametrize("retained", [False, True])
def test_supplied_residue_accepts_equal_value_copy(consumer, retained):
    source = _ligand() if retained else _polymer()
    residue = next(source.constitution.iter_residues(include_ligands=True))
    copied = replace(
        residue, atom_sites=tuple(replace(atom) for atom in residue.atom_sites)
    )
    observation = StructureChemistryObservation(source)
    assert copied is not residue
    assert _facts_for_supplied_residue(
        consumer, source, copied, observation
    ) == _facts_for_supplied_residue(consumer, source, residue, observation)


@pytest.mark.parametrize("consumer", ["singleton", "coverage", "projection"])
@pytest.mark.parametrize("retained", [False, True])
def test_supplied_residue_keeps_missing_identity_error(consumer, retained):
    source = _ligand() if retained else _polymer()
    residue = next(source.constitution.iter_residues(include_ligands=True))
    missing = residue.with_residue_id(ResidueId("MISSING", 999))
    with pytest.raises(KeyError):
        _facts_for_supplied_residue(consumer, source, missing)


@pytest.mark.parametrize("consumer", ["coverage", "projection"])
@pytest.mark.parametrize("retained", [False, True])
def test_warm_observation_still_rejects_conflicting_residue(consumer, retained):
    source = _ligand() if retained else _polymer()
    residue = next(source.constitution.iter_residues(include_ligands=True))
    observation = StructureChemistryObservation(source)
    expected = _facts_for_supplied_residue(consumer, source, residue, observation)
    conflicting = replace(residue, atom_sites=residue.atom_sites[:-1])
    with pytest.raises(ValueError, match="residue.*context"):
        _facts_for_supplied_residue(consumer, source, conflicting, observation)
    assert (
        _facts_for_supplied_residue(consumer, source, residue, observation) == expected
    )


@pytest.mark.parametrize("consumer", ["singleton", "coverage", "projection"])
def test_conflicting_residue_is_rejected_before_chemistry_derivation(consumer):
    source = _polymer()
    residue = source.constitution.chains[0].residues[0]
    conflicting = replace(residue, atom_sites=residue.atom_sites[:-1])
    with patch.object(
        observation_module,
        "StructureHydrogenExpectationModel",
        side_effect=AssertionError("invalid residue triggered chemistry derivation"),
    ):
        with pytest.raises(ValueError, match="residue.*context"):
            _facts_for_supplied_residue(consumer, source, conflicting)


@pytest.mark.parametrize("project_first", [True, False])
def test_whole_observation_cannot_be_poisoned_by_partial_projection(project_first):
    source = _polymer()
    library = build_default_component_library()
    observation = StructureChemistryObservation(source, component_library=library)
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=SidechainHeavyAtomCompletenessState.COMPLETE
    )
    if not project_first:
        observation.whole_structure_facts()

    partial, _ = observation.projected_facts((ResidueId("A", 1),))
    whole, _ = observation.whole_structure_facts()
    assert (
        partial.sidechain_heavy_atom_completeness_state
        is SidechainHeavyAtomCompletenessState.COMPLETE
    )
    assert (
        whole.sidechain_heavy_atom_completeness_state
        is SidechainHeavyAtomCompletenessState.INCOMPLETE
    )
    assert not already_satisfied_requested_goals(
        source,
        requested_goals=(goal,),
        component_library=library,
        chemistry_observation=observation,
    )


def test_projection_resolves_identities_and_rejects_missing_or_repeated_identity():
    observation = StructureChemistryObservation(_polymer())
    with pytest.raises(KeyError):
        observation.projected_facts((ResidueId("A", 99),))
    with pytest.raises(ValueError, match="distinct"):
        observation.projected_facts((ResidueId("A", 1), ResidueId("A", 1)))
    empty, _ = observation.projected_facts(())
    assert empty.residue_facts == ()
    whole, _ = observation.whole_structure_facts()
    assert len(whole.residue_facts) == 2


def test_repeated_queries_share_derivation_but_new_observation_does_not():
    source = _polymer()
    with patch.object(
        observation_module,
        "StructureHydrogenExpectationModel",
        wraps=StructureHydrogenExpectationModel,
    ) as derive:
        observation = StructureChemistryObservation(source)
        assert derive.call_count == 0
        first = observation.whole_structure_facts()
        observation.projected_facts((ResidueId("A", 1),))
        assert observation.whole_structure_facts() == first
        assert derive.call_count == 1
        second = StructureChemistryObservation(source).whole_structure_facts()
        assert second[0] == first[0]
        assert second[1].hydrogen_coverage_state == first[1].hydrogen_coverage_state
        assert derive.call_count == 2


def test_reused_model_rebinds_changed_override_and_preserves_original_observation():
    source = _ligand()
    library = build_default_component_library()
    original = StructureChemistryObservation(
        source,
        component_library=library,
        retained_non_polymer_chemistry_evidence=_evidence("CO"),
    )
    changed = StructureChemistryObservation(
        source,
        component_library=library,
        retained_non_polymer_chemistry_evidence=_evidence("C=O"),
        hydrogen_expectation_model=original.hydrogen_expectation_model,
    )
    rid = ResidueId("L", 1)
    assert (
        len(
            changed.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue[
                rid
            ]
        )
        == 2
    )
    assert (
        len(
            original.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue[
                rid
            ]
        )
        == 4
    )


def test_reused_model_rebinds_changed_library():
    source = _polymer()
    original = StructureChemistryObservation(source)
    assert original.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue
    changed = StructureChemistryObservation(
        source,
        component_library=ComponentLibrary(),
        hydrogen_expectation_model=original.hydrogen_expectation_model,
    )
    assert (
        not changed.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue
    )
    assert not changed.whole_structure_facts()[
        1
    ].component_support_state.is_fully_supported()


@pytest.mark.parametrize("old_permission", [True, False])
def test_reused_model_rebinds_changed_fallback_permission(old_permission):
    source = _ligand()
    original = StructureChemistryObservation(
        source, allow_retained_non_polymer_rdkit_fallback=old_permission
    )
    changed = StructureChemistryObservation(
        source,
        allow_retained_non_polymer_rdkit_fallback=not old_permission,
        hydrogen_expectation_model=original.hydrogen_expectation_model,
    )
    resolution = changed.hydrogen_expectation_model.resolution_for_retained_non_polymer(
        ResidueId("L", 1)
    )
    expected = (
        RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        if old_permission
        else RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert resolution.source is expected


@pytest.mark.parametrize("rebind_observation", [False, True])
def test_geometry_dependent_ligand_expectations_are_not_reused_after_movement(
    rebind_observation,
):
    source = _ligand()
    rid = ResidueId("L", 1)
    geometry = source.residue_geometry(source.constitution.residue_index(rid))
    moved = source.with_updated_residue_geometries(
        (
            (
                rid,
                geometry.with_atom_geometry(
                    "O1", replace(geometry.atom_geometry("O1"), position=Vec3(4, 0, 0))
                ),
            ),
        )
    )
    observation = StructureChemistryObservation(source)
    original = observation.hydrogen_expectation_model
    assert original.polymer_preparation.matches_chemistry(moved)
    changed = (
        observation.for_structure(moved)
        if rebind_observation
        else StructureChemistryObservation(moved, hydrogen_expectation_model=original)
    )
    fresh = StructureChemistryObservation(moved)
    observed = changed.hydrogen_expectation_model.resolution_for_retained_non_polymer(
        rid
    )
    assert (
        observed
        == fresh.hydrogen_expectation_model.resolution_for_retained_non_polymer(rid)
    )
    assert observed != original.resolution_for_retained_non_polymer(rid)


@pytest.mark.parametrize("change", ["topology", "charge", "provenance"])
@pytest.mark.parametrize("rebind_observation", [False, True])
def test_reused_model_rederives_after_topology_or_provenance_change(
    change,
    rebind_observation,
):
    source = _ligand()
    observation = StructureChemistryObservation(source)
    original = observation.hydrogen_expectation_model
    topology = source.topology
    provenance = source.provenance
    if change == "topology":
        topology = StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(TopologyBond(AtomIndex(0), AtomIndex(1), order=2),),
        )
    elif change == "charge":
        topology = StructureTopology(
            constitution=source.constitution,
            atom_topologies=(
                source.topology.atom_topologies[0],
                AtomTopology(formal_charge=-1),
            ),
            bonds=source.topology.bonds,
        )
    else:
        provenance = replace(
            provenance,
            ingress=replace(provenance.ingress, source_name="another-observation"),
        )
    changed = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=provenance,
        polymer_blueprint=source.polymer_blueprint,
    )

    with patch.object(
        observation_module,
        "StructureHydrogenExpectationModel",
        wraps=StructureHydrogenExpectationModel,
    ) as derive:
        rebound = (
            observation.for_structure(changed)
            if rebind_observation
            else StructureChemistryObservation(
                changed, hydrogen_expectation_model=original
            )
        ).hydrogen_expectation_model
        assert derive.call_count == 1
    fresh = StructureChemistryObservation(changed).hydrogen_expectation_model
    assert (
        rebound.expected_hydrogen_atom_names_by_residue
        == fresh.expected_hydrogen_atom_names_by_residue
    )
    assert (
        rebound.retained_non_polymer_resolution_by_residue_id
        == fresh.retained_non_polymer_resolution_by_residue_id
    )


@pytest.mark.parametrize("change", ["carrier", "library", "evidence", "policy"])
def test_consumer_cannot_mix_observation_with_different_inputs(change):
    source = _ligand()
    library = build_default_component_library()
    observation = StructureChemistryObservation(source, component_library=library)
    with pytest.raises(ValueError, match="interpretation"):
        observation.require_context(
            _ligand() if change == "carrier" else source,
            component_library=ComponentLibrary() if change == "library" else library,
            retained_non_polymer_chemistry_evidence=_evidence("CO")
            if change == "evidence"
            else (),
            allow_retained_non_polymer_rdkit_fallback=change != "policy",
        )


@pytest.mark.parametrize("warm", [False, True])
def test_rebinding_same_carrier_retains_lazy_chemistry(warm):
    source = _polymer()
    with patch.object(
        observation_module,
        "StructureHydrogenExpectationModel",
        wraps=StructureHydrogenExpectationModel,
    ) as derive:
        observation = StructureChemistryObservation(source)
        if warm:
            observation.whole_structure_facts()
        rebound = observation.for_structure(source)
        assert derive.call_count == int(warm)
        assert rebound is observation
        rebound.projected_facts((ResidueId("A", 1),))
        observation.whole_structure_facts()
        assert derive.call_count == 1


@pytest.mark.parametrize("warm", [False, True])
def test_polymer_movement_reuses_valid_expectations_but_not_residue_facts(warm):
    source = _polymer()
    rid = ResidueId("A", 1)
    geometry = source.residue_geometry(source.constitution.residue_index(rid))
    moved = source.with_updated_residue_geometries(
        (
            (
                rid,
                geometry.with_atom_geometry(
                    "CB",
                    replace(geometry.atom_geometry("CB"), position=Vec3(1.5, 1.5, 1)),
                ),
            ),
        )
    )
    observation = StructureChemistryObservation(source)
    if warm:
        observation.whole_structure_facts()

    with patch.object(
        observation_module,
        "StructureHydrogenExpectationModel",
        wraps=StructureHydrogenExpectationModel,
    ) as derive:
        rebound = observation.for_structure(moved)
        assert derive.call_count == 0
        coverage, readiness = rebound.whole_structure_facts()
        assert derive.call_count == int(not warm)
    fresh_coverage, fresh_readiness = StructureChemistryObservation(
        moved
    ).whole_structure_facts()
    assert coverage == fresh_coverage
    assert readiness.hydrogen_coverage_state == fresh_readiness.hydrogen_coverage_state
    assert (
        readiness.heavy_atom_topology_availability_state
        == fresh_readiness.heavy_atom_topology_availability_state
    )
    assert rebound.residue_completion_facts(
        rid
    ) is not observation.residue_completion_facts(rid)
    assert coverage is not observation.whole_structure_facts()[0]
    if warm:
        assert (
            rebound.hydrogen_expectation_model is observation.hydrogen_expectation_model
        )


@pytest.mark.parametrize("consumer", ["projection", "coverage", "continuous"])
@pytest.mark.parametrize("change", ["carrier", "library", "evidence", "policy"])
def test_refinement_projection_consumers_enforce_exact_input_binding(consumer, change):
    source = _ligand()
    library = build_default_component_library()
    observation = StructureChemistryObservation(source, component_library=library)
    active_structure = _ligand() if change == "carrier" else source
    active_library = ComponentLibrary() if change == "library" else library
    evidence = _evidence("CO") if change == "evidence" else ()
    with pytest.raises(ValueError, match="interpretation"):
        if consumer == "continuous":
            derive_atom_scope_continuous_relaxation_facts(
                ProteinStructureSnapshot.from_structure(active_structure),
                AtomSetScope(atom_refs=(AtomRef(ResidueId("L", 1), "C1"),)),
                component_library=active_library,
                retained_non_polymer_chemistry_evidence=evidence,
                allow_retained_non_polymer_rdkit_fallback=change != "policy",
                chemistry_observation=observation,
            )
        else:
            project = (
                StructureProjectionStateFacts.from_projection
                if consumer == "projection"
                else derive_projection_coverage_and_chemistry_readiness_facts
            )
            project(
                context_structure=active_structure,
                residues=(),
                ligands=active_structure.constitution.ligands,
                component_library=active_library,
                retained_non_polymer_chemistry_evidence=evidence,
                allow_retained_non_polymer_rdkit_fallback=change != "policy",
                chemistry_observation=observation,
            )


@pytest.mark.parametrize("consumer", ["initial_goals", "goal_report", "score"])
@pytest.mark.parametrize("change", ["carrier", "library"])
def test_workflow_consumers_enforce_observation_binding(consumer, change):
    source = _polymer()
    library = build_default_component_library()
    observation = StructureChemistryObservation(source, component_library=library)
    active_structure = _polymer() if change == "carrier" else source
    active_library = ComponentLibrary() if change == "library" else library
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=SidechainHeavyAtomCompletenessState.COMPLETE
    )
    report = evaluate_requested_goal_report(
        active_structure, requested_goals=(goal,), component_library=active_library
    )
    with pytest.raises(ValueError, match="interpretation"):
        if consumer == "initial_goals":
            already_satisfied_requested_goals(
                active_structure,
                requested_goals=(goal,),
                component_library=active_library,
                chemistry_observation=observation,
            )
        elif consumer == "goal_report":
            evaluate_requested_goal_report(
                active_structure,
                requested_goals=(goal,),
                component_library=active_library,
                chemistry_observation=observation,
            )
        else:
            evaluate_workflow_branch_quality_score(
                TransformationResult(active_structure, repairs=(), issues=()),
                requested_goal_report=report,
                planning_context=WorkflowPlanningContext(),
                component_library=active_library,
                chemistry_observation=observation,
            )


def test_unsupported_goal_report_does_not_force_chemistry_derivation():
    source = _polymer()
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=SidechainHeavyAtomCompletenessState.COMPLETE
    )
    with patch.object(
        observation_module,
        "StructureHydrogenExpectationModel",
        side_effect=AssertionError("unneeded chemistry derivation"),
    ):
        report = evaluate_requested_goal_report(
            source, requested_goals=(goal,), unsupported_requested_goals=(goal,)
        )
    assert report.outcomes[0].status is RequestedGoalStatus.UNSUPPORTED


@pytest.mark.parametrize("custom_library", [False, True])
def test_goal_and_score_share_observation_without_changing_values(custom_library):
    source = _polymer()
    library = (
        ComponentLibrary() if custom_library else build_default_component_library()
    )
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=SidechainHeavyAtomCompletenessState.COMPLETE
    )
    result = TransformationResult(source, repairs=(), issues=())
    context = WorkflowPlanningContext()
    reference_report = evaluate_requested_goal_report(
        source, requested_goals=(goal,), component_library=library
    )
    reference_score = evaluate_workflow_branch_quality_score(
        result,
        requested_goal_report=reference_report,
        planning_context=context,
        component_library=library,
        search_depth=2,
    )

    with patch.object(
        observation_module,
        "StructureHydrogenExpectationModel",
        wraps=StructureHydrogenExpectationModel,
    ) as derive:
        for expected_calls in (1, 2):
            observation = StructureChemistryObservation(
                source, component_library=library
            )
            report = evaluate_requested_goal_report(
                source,
                requested_goals=(goal,),
                component_library=library,
                chemistry_observation=observation,
            )
            score = evaluate_workflow_branch_quality_score(
                result,
                requested_goal_report=report,
                planning_context=context,
                component_library=library,
                search_depth=2,
                chemistry_observation=observation,
            )
            assert derive.call_count == expected_calls
            assert report == reference_report
            assert score == reference_score
