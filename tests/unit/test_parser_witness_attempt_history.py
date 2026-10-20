"""Attached correction publishes evaluations without inventing adoption."""

from dataclasses import replace
from pathlib import Path

import pytest

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.parser_readability import (
    RDKitNoConectParserReadabilityProbe,
    RDKitParserProblemWitness,
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope
from protrepair.structure.labels import AtomRef
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.refinement.parser_witness import (
    parser_witness_repair_candidates,
)
from protrepair.transformer.refinement.result import RefinementTransformationResult
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts import OrphanFragmentPolicy
from protrepair.workflow.engine import parser_witness_repair
from protrepair.workflow.planning.planner import WorkflowPlannerMemory


@pytest.fixture(scope="module")
def correction_case():
    structure = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    library = build_default_component_library()
    ids = tuple(
        r.residue_id
        for r in structure.constitution.iter_residues()
        if r.has_atom_site("CB")
    )[:4]
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
    candidates = tuple(
        parser_witness_repair_candidates(
            structure, component_library=library, clusters=(cluster,)
        )[0]
        for cluster in clusters
    )
    context = TransformerExecutionContext(
        component_library=library,
        original_structure=structure,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
    )
    return structure, context, candidates


def _probe(clusters):
    return RDKitNoConectParserReadabilityProbe(
        sanitize_readable=not clusters,
        residue_problem_witnesses=tuple(
            RDKitParserProblemWitness(
                residue_id=cluster.residue_ids[0],
                atom_names=("CB",),
                messages=("controlled contact",),
                proximity_bonds=cluster.bonds,
            )
            for cluster in clusters
        ),
        structure_messages=(),
    )


@pytest.mark.parametrize("native_returned", (False, True))
@pytest.mark.parametrize("improves", (False, True))
def test_loop_retains_rejected_and_adopted_evaluations(
    correction_case, monkeypatch, native_returned, improves
):
    structure, context, candidates = correction_case
    candidate = candidates[0]
    transformer = LocalRefinementTransformer.from_repair_refinement(
        candidate.repair_refinement
    )
    initial = TransformationResult(structure=structure, repairs=(), issues=())
    receipt = ContinuousRelaxationExecution(
        settings=ContinuousRelaxationSettings(
            profile=ContinuousRelaxationProfile.RDKIT_MMFF
        ),
        movable_scope=AtomSetScope((candidate.cluster.bonds[0].atom_ref_1,)),
        minimizer_status=1,
        backend_version="test-native",
    )
    executions = (receipt,) if native_returned else ()
    calls = []

    def execute(result, **kwargs):
        calls.append(kwargs["transformer"])
        return RefinementTransformationResult.with_observed_executions(
            result, executions
        )

    monkeypatch.setattr(
        parser_witness_repair,
        "probe_rdkit_no_conect_parser_readability",
        lambda *args, **kwargs: _probe(
            () if calls and improves else (candidate.cluster,)
        ),
    )
    monkeypatch.setattr(
        parser_witness_repair,
        "parser_witness_repair_candidates",
        lambda *args, **kwargs: (candidate,),
    )
    outcome = parser_witness_repair.execute_bounded_parser_witness_repair_loop(
        initial, execution_context=context, execute_transformer=execute
    )

    assert calls == [transformer]
    assert outcome.attempted_transformers == tuple(calls)
    assert outcome.adopted_transformers == ((transformer,) if improves else ())
    assert outcome.result.structure is structure
    assert (
        RefinementTransformationResult.observed_executions(outcome.result) == executions
    )


@pytest.mark.parametrize("clusters_present", (False, True))
def test_loop_does_not_invent_attempts_without_a_called_candidate(
    correction_case, monkeypatch, clusters_present
):
    structure, context, candidates = correction_case
    initial = TransformationResult(structure=structure, repairs=(), issues=())
    monkeypatch.setattr(
        parser_witness_repair,
        "probe_rdkit_no_conect_parser_readability",
        lambda *args, **kwargs: _probe(
            (candidates[0].cluster,) if clusters_present else ()
        ),
    )
    monkeypatch.setattr(
        parser_witness_repair,
        "parser_witness_repair_candidates",
        lambda *args, **kwargs: (),
    )

    def execute(*args, **kwargs):
        pytest.fail("No candidate was selected for execution")

    outcome = parser_witness_repair.execute_bounded_parser_witness_repair_loop(
        initial, execution_context=context, execute_transformer=execute
    )

    assert outcome.result is initial
    assert outcome.attempted_transformers == ()
    assert outcome.adopted_transformers == ()


def test_internal_restarts_keep_ordered_evaluations_and_deduplicate_only_in_memory(
    correction_case, monkeypatch
):
    structure, context, candidates = correction_case
    candidate = candidates[0]
    transformer = LocalRefinementTransformer.from_repair_refinement(
        candidate.repair_refinement
    )
    initial = TransformationResult(structure=structure, repairs=(), issues=())
    calls = []
    inputs = []

    def execute(result, **kwargs):
        inputs.append(result.structure)
        calls.append(kwargs["transformer"])
        residue_id = candidate.cluster.residue_ids[0]
        index = result.structure.constitution.residue_index(residue_id)
        geometry = result.structure.residue_geometry(index)
        atom_geometry = geometry.atom_geometry("CB")
        position = atom_geometry.position
        geometry = geometry.with_atom_geometry(
            "CB",
            atom_geometry.with_position(
                Vec3(position.x + 0.01, position.y, position.z)
            ),
        )
        return replace(
            result,
            structure=result.structure.with_updated_residue_geometries(
                ((residue_id, geometry),)
            ),
        )

    def probe(*args, **kwargs):
        del args, kwargs
        return _probe(tuple(c.cluster for c in candidates[len(calls) :]))

    monkeypatch.setattr(
        parser_witness_repair, "probe_rdkit_no_conect_parser_readability", probe
    )
    monkeypatch.setattr(
        parser_witness_repair,
        "parser_witness_repair_candidates",
        lambda *args, **kwargs: (candidate,),
    )
    outcome = parser_witness_repair.execute_bounded_parser_witness_repair_loop(
        initial, execution_context=context, execute_transformer=execute
    )
    memory = WorkflowPlannerMemory()
    for action in outcome.attempted_transformers:
        memory = memory.with_attempted_transformer(action, input_structure=structure)

    assert calls == [transformer, transformer]
    assert inputs[0] != inputs[1]
    assert outcome.attempted_transformers == tuple(calls)
    assert outcome.adopted_transformers == tuple(calls)
    assert memory.attempted_transformers == (transformer,)
    assert memory.adopted_transformers == ()
    assert memory.blocks_transformer_attempt(transformer, structure=structure)


def test_inner_execution_exception_is_not_published_as_a_loop_outcome(
    correction_case, monkeypatch
):
    structure, context, candidates = correction_case
    initial = TransformationResult(structure=structure, repairs=(), issues=())
    calls = []
    monkeypatch.setattr(
        parser_witness_repair,
        "probe_rdkit_no_conect_parser_readability",
        lambda *args, **kwargs: _probe((candidates[0].cluster,)),
    )
    monkeypatch.setattr(
        parser_witness_repair,
        "parser_witness_repair_candidates",
        lambda *args, **kwargs: candidates,
    )

    def execute(result, **kwargs):
        calls.append(kwargs["transformer"])
        raise RuntimeError("controlled execution failure")

    with pytest.raises(RuntimeError, match="controlled execution failure"):
        parser_witness_repair.execute_bounded_parser_witness_repair_loop(
            initial, execution_context=context, execute_transformer=execute
        )

    assert len(calls) == 1
    assert RefinementTransformationResult.observed_executions(initial) == ()
    assert initial.structure is structure
