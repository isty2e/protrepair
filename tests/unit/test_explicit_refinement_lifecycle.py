"""Explicit requests survive preparation without requiring an automatic burden."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import RestraintLibrary
from protrepair.diagnostics import RepairEventKind, ValidationIssueKind
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.artifacts import (
    MovedAtomDelta,
    RegionTransformationResult,
    StructureDelta,
)
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem
from protrepair.transformer.continuous.rdkit import RdkitContinuousRelaxationBackend
from protrepair.transformer.continuous.shared import apply_position_updates
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts import (
    RequestedGoalSet,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    plan_workflow_actions,
)


@pytest.fixture
def heavy_glycines() -> ProteinStructure:
    structure = build_bonded_structure(
        chains=tuple(
            chain_payload(
                chain_id,
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id, 1),
                        atoms=tuple(
                            atom_payload(name, element, Vec3(x + shift, y, 0.0))
                            for name, element, x, y in (
                                ("N", "N", 0.0, 0.0),
                                ("CA", "C", 1.45, 0.0),
                                ("C", "C", 2.40, 1.20),
                                ("O", "O", 2.10, 2.35),
                            )
                        ),
                    ),
                ),
            )
            for chain_id, shift in (("A", 0.0), ("B", 20.0))
        ),
        source_format=FileFormat.PDB,
    )
    return augment_c_terminal_oxt(structure).structure


@pytest.fixture
def refinement_request() -> RepairRefinementSpec:
    return RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
        binding=RecommendedContinuousRelaxationBinding(),
    )


@pytest.mark.parametrize("history", ["none", "target_h", "unrelated_h"])
def test_prepared_request_is_proposed_without_automatic_burden(
    heavy_glycines: ProteinStructure,
    refinement_request: RepairRefinementSpec,
    history: str,
) -> None:
    structure = add_hydrogens(heavy_glycines).structure
    memory = WorkflowPlannerMemory()
    if history != "none":
        memory = memory.with_adopted_transformer(
            HydrogenCompletionTransformer(
                scope=ResidueSetScope(
                    (ResidueId("A" if history == "target_h" else "B", 1),)
                )
            )
        )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=refinement_request
        ),
        planner_memory=memory,
    )

    assert outcome.state_deficit is not None
    assert outcome.state_deficit.intrinsic_geometry is None
    assert outcome.state_deficit.parser_compatibility is None
    assert outcome.state_deficit.interaction is None
    assert outcome.transformers == (
        LocalRefinementTransformer.from_repair_refinement(refinement_request),
    )


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("atomwise", [False, True])
def test_public_request_reaches_native_ff_once_after_real_preparation(
    heavy_glycines: ProteinStructure,
    refinement_request: RepairRefinementSpec,
    prepared: bool,
    atomwise: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = add_hydrogens(heavy_glycines).structure if prepared else heavy_glycines
    if atomwise:
        refinement_request = replace(
            refinement_request,
            scope_spec=LocalScopeSpec.from_atoms((AtomRef(ResidueId("A", 1), "CA"),)),
        )
    calls: list[ContinuousRelaxationProblem] = []
    original = RdkitContinuousRelaxationBackend.relax

    def observe(
        backend: RdkitContinuousRelaxationBackend,
        problem: ContinuousRelaxationProblem,
        *,
        restraint_library: RestraintLibrary,
    ) -> RegionTransformationResult:
        calls.append(problem)
        native_source = problem.region.snapshot.structure
        selected = native_source.constitution.residue_or_ligand(ResidueId("A", 1))
        assert selected is not None and selected.has_atom_site("H1")
        assert problem.spec.max_iterations == refinement_request.config.max_iterations
        return original(backend, problem, restraint_library=restraint_library)

    monkeypatch.setattr(RdkitContinuousRelaxationBackend, "relax", observe)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            repair_refinement=refinement_request
        ),
    )

    assert len(calls) == 1
    assert result.branch_report is not None
    for index in source.constitution.atom_indices_for_residue_index(
        source.constitution.residue_index(ResidueId("B", 1))
    ):
        ref = source.constitution.atom_ref_at(index)
        assert source.geometry.position(index) == result.structure.geometry.position(
            result.structure.constitution.atom_index(ref)
        )
    if not prepared:
        assert any(r.kind is RepairEventKind.HYDROGENS_ADDED for r in result.repairs)


@pytest.mark.parametrize("outcome", ["noop", "failure", "worse"])
def test_attempt_is_terminal_even_without_an_adopted_coordinate_change(
    heavy_glycines: ProteinStructure,
    refinement_request: RepairRefinementSpec,
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = add_hydrogens(heavy_glycines).structure
    calls: list[ContinuousRelaxationProblem] = []

    def execute(
        backend: RdkitContinuousRelaxationBackend,
        problem: ContinuousRelaxationProblem,
        *,
        restraint_library: RestraintLibrary,
    ) -> RegionTransformationResult:
        del backend, restraint_library
        calls.append(problem)
        if outcome == "failure":
            raise RefinementError("controlled native failure")
        structure = problem.region.snapshot.structure
        moved_atoms = ()
        if outcome == "worse":
            index = structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C"))
            structure = apply_position_updates(
                structure, moved_positions={index: Vec3(9.0, 9.0, 9.0)}
            )
            moved_atoms = (MovedAtomDelta(index, index),)
        return RegionTransformationResult(
            refined_structure=structure,
            delta=StructureDelta(
                before_constitution=source.constitution,
                after_constitution=structure.constitution,
                moved_atoms=moved_atoms,
            ),
            issues=(),
            backend_name="rdkit",
        )

    monkeypatch.setattr(RdkitContinuousRelaxationBackend, "relax", execute)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            repair_refinement=refinement_request
        ),
    )

    assert len(calls) == 1
    assert result.structure == source
    assert not any(
        r.kind is RepairEventKind.LOCAL_REFINEMENT_APPLIED for r in result.repairs
    )
    if outcome != "noop":
        assert any(
            i.kind is ValidationIssueKind.REFINEMENT_REJECTED for i in result.issues
        )


@pytest.mark.parametrize("adopted", [False, True])
def test_unrelated_chemistry_does_not_rearm_an_exact_attempt(
    heavy_glycines: ProteinStructure,
    refinement_request: RepairRefinementSpec,
    adopted: bool,
) -> None:
    action = LocalRefinementTransformer.from_repair_refinement(refinement_request)
    memory = WorkflowPlannerMemory()
    memory = (
        memory.with_adopted_transformer(action)
        if adopted
        else memory.with_attempted_transformer(action)
    )
    memory = memory.with_adopted_transformer(
        HydrogenCompletionTransformer(scope=ResidueSetScope((ResidueId("B", 1),)))
    )
    outcome = plan_workflow_actions(
        add_hydrogens(heavy_glycines).structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=refinement_request
        ),
        planner_memory=memory,
    )

    assert outcome.is_terminal()
    assert not outcome.transformers


def test_preparation_without_explicit_request_does_not_run_native_ff(
    heavy_glycines: ProteinStructure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(
        backend: RdkitContinuousRelaxationBackend,
        problem: ContinuousRelaxationProblem,
        *,
        restraint_library: RestraintLibrary,
    ) -> RegionTransformationResult:
        pytest.fail("burden-free hydrogen completion must not trigger automatic FF")

    monkeypatch.setattr(RdkitContinuousRelaxationBackend, "relax", unexpected)
    result = process_structure(
        heavy_glycines,
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert result.requested_goals_fully_satisfied()


def test_attempt_of_another_scope_does_not_consume_the_request(
    heavy_glycines: ProteinStructure,
    refinement_request: RepairRefinementSpec,
) -> None:
    attempted = LocalRefinementTransformer.from_repair_refinement(
        replace(
            refinement_request,
            scope_spec=LocalScopeSpec.from_residues((ResidueId("B", 1),)),
        )
    )
    outcome = plan_workflow_actions(
        add_hydrogens(heavy_glycines).structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=refinement_request
        ),
        planner_memory=WorkflowPlannerMemory().with_attempted_transformer(attempted),
    )

    assert outcome.transformers == (
        LocalRefinementTransformer.from_repair_refinement(refinement_request),
    )
