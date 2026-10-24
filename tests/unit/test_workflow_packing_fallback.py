"""State-qualified packing alternatives after returned native minimizations."""

from pathlib import Path

import pytest

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope
from protrepair.state import (
    HydrogenCoverageState,
    ParserCompatibilityProfile,
    ParserCompatibilityState,
    StructureParserCompatibilityFacts,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.packing.spec import PackingScope, PackingSpec
from protrepair.transformer.refinement.parser_witness import (
    ParserWitnessRepairBudget,
    ParserWitnessRepairCandidate,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.contracts import RequestedGoalSet, WorkflowTransformRequests
from protrepair.workflow.planning import local_refinement_policy
from protrepair.workflow.planning.default_action_registry import (
    WORKFLOW_ACTION_REGISTRY,
    packing_proposals,
)
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    plan_workflow_actions,
)


def _cluster(left, right):
    ids = (ResidueId("A", left), ResidueId("A", right))
    return RDKitProximityBondCluster(
        residue_ids=ids,
        bonds=(
            RDKitProximityBondWitness(
                AtomRef(ids[0], "CB"), AtomRef(ids[1], "CB"), "C", "C", False
            ),
        ),
    )


def _native_return(*refs):
    return ContinuousRelaxationExecution(
        ContinuousRelaxationSettings(profile=ContinuousRelaxationProfile.RDKIT_MMFF),
        AtomSetScope(refs),
        1,
        "2026.03.2",
    )


def _packing(cluster):
    return CommittedPackingTransformer.from_planned_spec(
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=cluster.residue_ids,
        )
    )


@pytest.fixture(scope="module")
def prepared_structure():
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    prepared = add_hydrogens(
        source,
        component_library=build_default_component_library(),
        protonate_histidines=False,
    ).structure
    _, chemistry = derive_structure_coverage_and_chemistry_readiness_facts(prepared)
    assert chemistry.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    return prepared


@pytest.fixture
def clusters(monkeypatch):
    clusters = (_cluster(10, 11), _cluster(20, 21))
    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: clusters,
    )

    def facts(cls, structure, **kwargs):
        return StructureParserCompatibilityFacts(
            structure,
            ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
            ParserCompatibilityState.INCOMPATIBLE,
            2,
            2,
        )

    monkeypatch.setattr(
        StructureParserCompatibilityFacts, "from_structure", classmethod(facts)
    )
    return clusters


def _plan(structure, memory, *, requests=None):
    # Isolate packing eligibility; other families retain their own ranking tests.
    registry = tuple(
        entry
        for entry in WORKFLOW_ACTION_REGISTRY
        if entry.action_type is CommittedPackingTransformer
    )
    return plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=requests or WorkflowTransformRequests(),
        planner_memory=memory,
        action_registry=registry,
    ).transformers


def test_rejected_native_return_unlocks_matching_cluster_without_adoption(
    prepared_structure, clusters
):
    receipt = _native_return(clusters[0].bonds[0].atom_ref_1)
    memory = WorkflowPlannerMemory(relaxation_executions=(receipt,))

    assert not memory.adopted_transformers
    assert _plan(prepared_structure, memory) == (_packing(clusters[0]),)


def test_default_registry_reaches_packing_after_unadopted_ff_trials(
    prepared_structure, clusters, monkeypatch
):
    trials = tuple(
        ParserWitnessRepairCandidate(
            cluster=cluster,
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residue_sidechains(cluster.residue_ids),
                binding=RecommendedContinuousRelaxationBinding(),
            ),
            budget=ParserWitnessRepairBudget(),
        )
        for cluster in clusters
    )
    monkeypatch.setattr(
        local_refinement_policy,
        "iter_parser_witness_repair_candidates",
        lambda *args, **kwargs: trials,
    )
    initial = plan_workflow_actions(
        prepared_structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )
    assert len(initial.transformers) == 2
    assert all(isinstance(a, LocalRefinementTransformer) for a in initial.transformers)
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(clusters[0].bonds[0].atom_ref_1),),
    )
    for transformer in initial.transformers:
        memory = memory.with_attempted_transformer(transformer)
    following = plan_workflow_actions(
        prepared_structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=memory,
    )

    assert not memory.adopted_transformers
    assert following.transformers == (_packing(clusters[0]),)


def test_unqualified_first_cluster_does_not_starve_later_native_burden(
    prepared_structure, clusters
):
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(clusters[1].bonds[0].atom_ref_1),)
    )

    assert _plan(prepared_structure, memory) == (_packing(clusters[1]),)


def test_default_registry_offers_matching_packing_before_untried_other_ff_clusters(
    prepared_structure, clusters, monkeypatch
):
    trials = tuple(
        ParserWitnessRepairCandidate(
            cluster=cluster,
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residue_sidechains(cluster.residue_ids),
                binding=RecommendedContinuousRelaxationBinding(),
            ),
            budget=ParserWitnessRepairBudget(),
        )
        for cluster in clusters
    )
    monkeypatch.setattr(
        local_refinement_policy,
        "iter_parser_witness_repair_candidates",
        lambda *args, **kwargs: trials,
    )
    tried = LocalRefinementTransformer.from_repair_refinement(
        trials[0].repair_refinement
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(clusters[0].bonds[0].atom_ref_1),),
    ).with_attempted_transformer(tried, input_structure=prepared_structure)

    following = plan_workflow_actions(
        prepared_structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=memory,
    )

    assert not memory.adopted_transformers
    assert following.current_proposal_batch() == (_packing(clusters[0]),)

    after_attempt = plan_workflow_actions(
        prepared_structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=memory.with_attempted_transformer(
            _packing(clusters[0]), input_structure=prepared_structure
        ),
    )
    assert after_attempt.current_proposal_batch() == (
        LocalRefinementTransformer.from_repair_refinement(trials[1].repair_refinement),
    )


def test_distinct_clusters_remain_reachable_after_packing_adoption(
    prepared_structure, clusters
):
    receipt = _native_return(*(c.bonds[0].atom_ref_1 for c in clusters))
    first = _packing(clusters[0])
    memory = WorkflowPlannerMemory(
        adopted_transformers=(first,), relaxation_executions=(receipt,)
    )

    assert _plan(prepared_structure, memory) == (_packing(clusters[1]),)


def test_all_qualified_clusters_are_proposed_and_exact_attempts_are_suppressed(
    prepared_structure, clusters
):
    receipt = _native_return(*(c.bonds[0].atom_ref_1 for c in clusters))
    memory = WorkflowPlannerMemory(relaxation_executions=(receipt,))
    expected = tuple(_packing(c) for c in clusters)

    assert _plan(prepared_structure, memory) == expected
    memory = memory.with_attempted_transformer(expected[0])
    assert _plan(prepared_structure, memory) == expected[1:]
    memory = memory.with_attempted_transformer(expected[1])
    assert _plan(prepared_structure, memory) == ()


@pytest.mark.parametrize(
    "refs",
    ((), (AtomRef(ResidueId("A", 50), "CB"),), (AtomRef(ResidueId("A", 10), "N"),)),
    ids=("no-return", "unrelated-residue", "same-residue-not-witness-endpoint"),
)
def test_no_matching_native_evidence_does_not_unlock_packing(
    prepared_structure, clusters, refs
):
    receipts = (_native_return(*refs),) if refs else ()
    memory = WorkflowPlannerMemory(relaxation_executions=receipts)

    assert _plan(prepared_structure, memory) == ()


def test_adopted_refinement_family_without_native_return_is_not_evidence(
    prepared_structure, clusters
):
    action = LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues(clusters[0].residue_ids),
            binding=RecommendedContinuousRelaxationBinding(),
        )
    )
    memory = WorkflowPlannerMemory(adopted_transformers=(action,))
    outcome = plan_workflow_actions(
        prepared_structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=memory,
    )

    assert not any(
        isinstance(a, CommittedPackingTransformer) for a in outcome.transformers
    )


def test_incomplete_hydrogens_block_automatic_packing(prepared_structure, clusters):
    receipt = _native_return(clusters[0].bonds[0].atom_ref_1)
    memory = WorkflowPlannerMemory(relaxation_executions=(receipt,))
    missing_h = prepared_structure.without_atom_refs(
        (AtomRef(ResidueId("A", 20), "HB1"),)
    )

    assert _plan(missing_h, memory) == ()


def test_explicit_repair_does_not_gain_automatic_packing(prepared_structure, clusters):
    receipt = _native_return(clusters[0].bonds[0].atom_ref_1)
    memory = WorkflowPlannerMemory(relaxation_executions=(receipt,))
    request = WorkflowTransformRequests(
        repair_refinement=RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues(clusters[0].residue_ids),
            binding=RecommendedContinuousRelaxationBinding(),
        )
    )

    assert _plan(prepared_structure, memory, requests=request) == ()


def test_explicit_packing_remains_available_without_native_evidence(
    prepared_structure, clusters
):
    action = _packing(clusters[1])
    request = WorkflowTransformRequests(committed_sidechain_packing=action.packing_spec)

    assert _plan(prepared_structure, WorkflowPlannerMemory(), requests=request) == (
        action,
    )


def test_unsupported_polymer_anywhere_blocks_automatic_faspr(
    prepared_structure, clusters, monkeypatch
):
    source = prepared_structure
    residue_id = ResidueId("A", 20)
    site = source.constitution.residue_or_ligand(residue_id)
    assert site is not None
    index = source.constitution.residue_index(residue_id)
    unsupported: ProteinStructure = source.with_updated_residue_facets(
        site.with_component_id("UNK"),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
    )

    def forbidden_probe(*args, **kwargs):
        raise AssertionError("unsupported FASPR input must be excluded before probing")

    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        forbidden_probe,
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(clusters[0].bonds[0].atom_ref_1),)
    )

    assert _plan(unsupported, memory) == ()


def test_duplicate_clusters_do_not_duplicate_packing_candidates(
    prepared_structure, clusters, monkeypatch
):
    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: (clusters[0], clusters[0]),
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(clusters[0].bonds[0].atom_ref_1),)
    )

    assert _plan(prepared_structure, memory) == (_packing(clusters[0]),)


def test_no_native_returns_skip_expensive_cluster_resolution(
    prepared_structure, clusters, monkeypatch
):
    def forbidden_probe(*args, **kwargs):
        raise AssertionError("no native return can qualify a packing fallback")

    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        forbidden_probe,
    )

    assert _plan(prepared_structure, WorkflowPlannerMemory()) == ()


def test_empty_cluster_does_not_invent_native_coverage(
    prepared_structure, clusters, monkeypatch
):
    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: (
            RDKitProximityBondCluster(clusters[0].residue_ids, ()),
        ),
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(clusters[0].bonds[0].atom_ref_1),)
    )

    assert _plan(prepared_structure, memory) == ()


def test_ambiguous_disulfide_partners_expand_each_local_packing_domain(
    prepared_structure, clusters, monkeypatch
):
    monkeypatch.setattr(
        packing_proposals,
        "ambiguous_disulfide_partner_residue_ids_by_residue",
        lambda structure: {
            ResidueId("A", 10): frozenset((ResidueId("A", 11), ResidueId("A", 12))),
            ResidueId("A", 20): frozenset((ResidueId("A", 21), ResidueId("A", 22))),
        },
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(
            _native_return(*(c.bonds[0].atom_ref_1 for c in clusters)),
        )
    )

    actions = _plan(prepared_structure, memory)

    assert len(actions) == 2
    assert isinstance(actions[0], CommittedPackingTransformer)
    assert isinstance(actions[1], CommittedPackingTransformer)
    assert actions[0].packing_spec.mutable_residue_ids == (
        ResidueId("A", 10),
        ResidueId("A", 11),
        ResidueId("A", 12),
    )
    assert actions[1].packing_spec.mutable_residue_ids == (
        ResidueId("A", 20),
        ResidueId("A", 21),
        ResidueId("A", 22),
    )


def test_overlapping_disulfide_expansions_are_one_attemptable_packing_scope(
    prepared_structure, clusters, monkeypatch
):
    overlapping = tuple(
        RDKitProximityBondCluster(
            residue_ids=(ResidueId("A", 10), *cluster.residue_ids),
            bonds=(*cluster.bonds, _cluster(10, first).bonds[0]),
        )
        for first, cluster in ((16, _cluster(16, 22)), (22, _cluster(22, 26)))
    )
    cysteines = frozenset(ResidueId("A", n) for n in (16, 22, 26))
    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: overlapping,
    )
    monkeypatch.setattr(
        packing_proposals,
        "ambiguous_disulfide_partner_residue_ids_by_residue",
        lambda structure: {rid: cysteines - {rid} for rid in cysteines},
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(
            _native_return(*(c.bonds[0].atom_ref_2 for c in overlapping)),
        )
    )
    expected = CommittedPackingTransformer.from_planned_spec(
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(
                ResidueId("A", 10),
                ResidueId("A", 16),
                ResidueId("A", 22),
                ResidueId("A", 26),
            ),
        )
    )

    assert _plan(prepared_structure, memory) == (expected,)
    assert _plan(prepared_structure, memory.with_attempted_transformer(expected)) == ()


def test_all_protected_automatic_clusters_are_not_proposed(
    prepared_structure, clusters, monkeypatch
):
    protected = (_cluster(26, 48),)
    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: protected,
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(protected[0].bonds[0].atom_ref_1),)
    )
    assert _plan(prepared_structure, memory) == ()
    assert _plan(prepared_structure, memory) == ()


def test_mixed_automatic_scope_retains_protected_partners_and_suppresses_repetition(
    prepared_structure, clusters, monkeypatch
):
    mixed = (_cluster(2, 26),)
    monkeypatch.setattr(
        packing_proposals,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: mixed,
    )
    monkeypatch.setattr(
        packing_proposals,
        "ambiguous_disulfide_partner_residue_ids_by_residue",
        lambda structure: {ResidueId("A", 26): frozenset((ResidueId("A", 48),))},
    )
    memory = WorkflowPlannerMemory(
        relaxation_executions=(_native_return(mixed[0].bonds[0].atom_ref_1),)
    )
    actions = _plan(prepared_structure, memory)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, CommittedPackingTransformer)
    assert action.packing_spec.mutable_residue_ids == (
        ResidueId("A", 2),
        ResidueId("A", 26),
        ResidueId("A", 48),
    )
    assert (
        _plan(
            prepared_structure,
            memory.with_attempted_transformer(
                action, input_structure=prepared_structure
            ),
        )
        == ()
    )


def test_explicit_all_protected_scope_is_attempted_only_once(prepared_structure):
    spec = PackingSpec(
        backend_name="faspr",
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(ResidueId("A", 26), ResidueId("A", 48)),
    )
    requests = WorkflowTransformRequests(committed_sidechain_packing=spec)
    memory = WorkflowPlannerMemory()
    actions = _plan(prepared_structure, memory, requests=requests)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, CommittedPackingTransformer)
    assert (
        _plan(
            prepared_structure,
            memory.with_attempted_transformer(
                action, input_structure=prepared_structure
            ),
            requests=requests,
        )
        == ()
    )
