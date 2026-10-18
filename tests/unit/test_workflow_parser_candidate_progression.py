"""Automatic parser repair progresses through untried current candidates."""

from dataclasses import replace
from pathlib import Path

import pytest

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.state import (
    ParserCompatibilityProfile,
    ParserCompatibilityState,
    StructureParserCompatibilityFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.refinement import parser_witness
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts import RequestedGoalSet, WorkflowTransformRequests
from protrepair.workflow.planning.action.attempt import WorkflowActionAttempt
from protrepair.workflow.planning.action.domain import WorkflowPlanningMemoryReadView
from protrepair.workflow.planning.default_action_registry import (
    WORKFLOW_ACTION_REGISTRY,
)
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    plan_workflow_actions,
)


@pytest.fixture(scope="module")
def prepared_structure():
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    prepared = add_hydrogens(
        source,
        component_library=build_default_component_library(),
        protonate_histidines=False,
    ).structure
    # This baseline models a complete external input, not in-memory reconstruction.
    return read_structure_string(
        write_structure_string(prepared, FileFormat.PDB), FileFormat.PDB
    )


@pytest.fixture
def candidates(prepared_structure, monkeypatch):
    ids = tuple(
        residue.residue_id
        for residue in prepared_structure.constitution.iter_residues()
        if residue.has_atom_site("CB")
    )[:12]
    clusters = tuple(
        RDKitProximityBondCluster(
            residue_ids=(left, right),
            bonds=(
                RDKitProximityBondWitness(
                    AtomRef(left, "CB"), AtomRef(right, "CB"), "C", "C", False
                ),
            ),
        )
        for left, right in zip(ids[::2], ids[1::2], strict=True)
    )
    monkeypatch.setattr(
        parser_witness,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: clusters,
    )

    def facts(cls, structure, **kwargs):
        del cls, kwargs
        return StructureParserCompatibilityFacts(
            structure,
            ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
            ParserCompatibilityState.INCOMPATIBLE,
            len(clusters),
            len(clusters),
        )

    monkeypatch.setattr(
        StructureParserCompatibilityFacts, "from_structure", classmethod(facts)
    )
    return tuple(
        LocalRefinementTransformer.from_repair_refinement(
            parser_witness.parser_witness_repair_candidates(
                prepared_structure,
                component_library=build_default_component_library(),
                clusters=(cluster,),
            )[0].repair_refinement
        )
        for cluster in clusters
    )


def _plan(structure, memory):
    return plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planner_memory=memory,
        action_registry=tuple(
            entry
            for entry in WORKFLOW_ACTION_REGISTRY
            if entry.action_type is LocalRefinementTransformer
        ),
    ).transformers


@pytest.mark.parametrize("attempted", ((0, 1, 2, 3), (0, 2), (0, 1, 2, 3, 4, 5)))
def test_attempted_candidates_do_not_consume_current_proposal_slots(
    prepared_structure, candidates, attempted
):
    memory = WorkflowPlannerMemory()
    for index in attempted:
        memory = memory.with_attempted_transformer(
            candidates[index], input_structure=prepared_structure
        )

    expected = tuple(
        candidate
        for index, candidate in enumerate(candidates)
        if index not in attempted
    )[:4]
    assert _plan(prepared_structure, memory) == expected


def test_prior_local_ff_adoption_does_not_block_distinct_current_clusters(
    prepared_structure, candidates
):
    memory = WorkflowPlannerMemory().with_adopted_transformer(
        candidates[0], input_structure=prepared_structure
    )

    assert _plan(prepared_structure, memory) == candidates[1:5]


def test_untried_proposals_keep_existing_cap_and_order(prepared_structure, candidates):
    assert _plan(prepared_structure, WorkflowPlannerMemory()) == candidates[:4]


def test_all_adopted_candidates_remain_blocked(prepared_structure, candidates):
    memory = WorkflowPlannerMemory()
    for candidate in candidates:
        memory = memory.with_adopted_transformer(
            candidate, input_structure=prepared_structure
        )

    assert _plan(prepared_structure, memory) == ()


@pytest.mark.parametrize("limit", (1, 4, 10))
def test_capped_engine_adapter_is_ordered_prefix_of_eligible_stream(
    prepared_structure, candidates, limit
):
    library = build_default_component_library()
    budget = replace(
        parser_witness.DEFAULT_PARSER_WITNESS_REPAIR_BUDGET,
        max_clusters_per_pass=limit,
    )
    stream = tuple(
        parser_witness.iter_parser_witness_repair_candidates(
            prepared_structure, component_library=library, budget=budget
        )
    )
    capped = parser_witness.parser_witness_repair_candidates(
        prepared_structure, component_library=library, budget=budget
    )

    assert len(stream) == len(candidates)
    assert capped == stream[:limit]


def test_duplicate_candidate_scopes_do_not_consume_proposal_slots(
    prepared_structure, candidates, monkeypatch
):
    stream = tuple(
        parser_witness.iter_parser_witness_repair_candidates(
            prepared_structure, component_library=build_default_component_library()
        )
    )
    monkeypatch.setattr(
        parser_witness,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *args, **kwargs: (
            (stream[0].cluster,) * 5
            + tuple(candidate.cluster for candidate in stream[1:])
        ),
    )

    assert _plan(prepared_structure, WorkflowPlannerMemory()) == candidates[:4]


def test_memory_read_view_preserves_full_action_repeat_contract(
    prepared_structure, candidates
):
    action = candidates[0]
    view = WorkflowPlanningMemoryReadView(attempts=(WorkflowActionAttempt(action),))
    different_settings = LocalRefinementTransformer.from_repair_refinement(
        replace(
            action.repair_refinement,
            config=replace(action.repair_refinement.config, max_iterations=51),
        )
    )
    clone = ProteinStructure.from_payload(
        constitution=prepared_structure.constitution,
        geometry=prepared_structure.geometry,
        topology=prepared_structure.topology,
        polymer_blueprint=prepared_structure.polymer_blueprint,
        provenance=prepared_structure.provenance,
    )

    assert view.blocks_transformer_attempt(action, structure=prepared_structure)
    assert view.blocks_transformer_attempt(action, structure=clone)
    assert not view.blocks_transformer_attempt(
        candidates[1], structure=prepared_structure
    )
    assert not view.blocks_transformer_attempt(
        different_settings, structure=prepared_structure
    )
