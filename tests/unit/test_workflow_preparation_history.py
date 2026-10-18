"""Preparation remains state-based after coordinate-changing packing."""

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from protrepair.api import process_structure
from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope, ResidueSetScope, WholeStructureScope
from protrepair.state import HydrogenCoverageState, StructureChemistryReadinessFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import AtomTopology, StructureTopology
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.discrete.axis_rotation import (
    AXIS_ROTATION_CORRECTION_DEFINITIONS,
    build_rotated_axis_correction_candidate,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.packing.runtime import execute_sidechain_packing
from protrepair.transformer.packing.spec import PackingScope, PackingSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.actions.retained_non_polymer_hydrogen_completion import (
    RetainedNonPolymerHydrogenCompletionTransformer,
)
from protrepair.workflow.contracts import (
    OrphanFragmentPolicy,
    RequestedGoalSet,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.engine.runtime import execute_workflow_transformer
from protrepair.workflow.planning.action.registry import WorkflowStateAction
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    plan_workflow_actions,
)


@dataclass(frozen=True)
class HydrogenPreparationCase:
    library: ComponentLibrary
    context: TransformerExecutionContext
    goals: RequestedGoalSet
    hydrogenation: HydrogenCompletionTransformer
    input_structure: ProteinStructure
    completed_structure: ProteinStructure
    packed_structure: ProteinStructure
    packing_action: CommittedPackingTransformer


@pytest.fixture(scope="module")
def preparation_case() -> HydrogenPreparationCase:
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    library = build_default_component_library()
    goals = RequestedGoalSet(
        goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        )
    )
    prepared = process_structure(source, requested_goals=goals.goals).structure
    residue = next(
        r for r in prepared.constitution.iter_residues() if r.component_id == "GLN"
    )
    hydrogenation = HydrogenCompletionTransformer(
        scope=ResidueSetScope((residue.residue_id,))
    )
    context = TransformerExecutionContext(
        component_library=library,
        original_structure=source,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
    )
    # A preparation-stage input, not deletion used to improve a quality score.
    incomplete = prepared.without_hydrogens_in_residues((residue.residue_id,))
    restored = execute_workflow_transformer(
        TransformationResult(structure=incomplete, repairs=(), issues=()),
        transformer=hydrogenation,
        execution_context=context,
    )
    index = restored.structure.constitution.residue_index(residue.residue_id)
    flipped = build_rotated_axis_correction_candidate(
        structure=restored.structure,
        residue_site=restored.structure.constitution.residue_site_at(index),
        residue_geometry=restored.structure.residue_geometry(index),
        formal_charge_by_atom_name=restored.structure.residue_formal_charge_by_atom_name(
            index
        ),
        definition=AXIS_ROTATION_CORRECTION_DEFINITIONS["GLN"],
        component_library=library,
    )
    spec = PackingSpec(
        backend_name="faspr",
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(residue.residue_id,),
    )
    packed = execute_sidechain_packing(flipped.structure, spec)
    assert packed.changed_residue_ids == (residue.residue_id,)
    assert any(i.kind.value == "packing_invalidated_hydrogens" for i in packed.issues)

    return HydrogenPreparationCase(
        library=library,
        context=context,
        goals=goals,
        hydrogenation=hydrogenation,
        input_structure=incomplete,
        completed_structure=restored.structure,
        packed_structure=packed.packed_structure,
        packing_action=CommittedPackingTransformer.from_planned_spec(spec),
    )


@pytest.fixture(scope="module")
def retained_preparation_input() -> ProteinStructure:
    structure = read_structure(Path("tests/fixtures/corpus/pdb6nbb.ent"))
    facts = StructureChemistryReadinessFacts.from_structure(structure)
    nad = next(f for f in facts.retained_non_polymer_facts if f.component_id == "NAD")
    assert nad.requires_hydrogen_completion()
    return structure


@pytest.mark.parametrize("bound_history", [False, True])
def test_real_packing_does_not_let_historical_h_completion_block_repreparation(
    preparation_case: HydrogenPreparationCase,
    bound_history: bool,
) -> None:
    case = preparation_case
    memory = WorkflowPlannerMemory(
        adopted_transformers=(case.hydrogenation, case.packing_action)
    )
    if bound_history:
        memory = memory.with_adopted_transformer(
            case.hydrogenation, input_structure=case.input_structure
        )
    plan = plan_workflow_actions(
        case.packed_structure,
        requested_goals=case.goals,
        transform_requests=WorkflowTransformRequests(),
        component_library=case.library,
        planner_memory=memory,
    )
    assert plan.current_proposal_batch() == (case.hydrogenation,)

    completed = execute_workflow_transformer(
        TransformationResult(structure=case.packed_structure, repairs=(), issues=()),
        transformer=plan.current_proposal_batch()[0],
        execution_context=case.context,
    )
    facts = StructureChemistryReadinessFacts.from_structure(
        completed.structure, component_library=case.library
    )
    assert facts.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert not facts.hydrogen_topology_availability_state.is_unavailable()
    for number in range(len(case.packed_structure.constitution.atom_slots)):
        old_index = AtomIndex(number)
        ref = case.packed_structure.constitution.atom_ref_at(old_index)
        new_index = completed.structure.constitution.atom_index(ref)
        assert completed.structure.geometry.atom_geometry(
            new_index
        ) == case.packed_structure.geometry.atom_geometry(old_index)
        assert completed.structure.topology.formal_charge(
            new_index
        ) == case.packed_structure.topology.formal_charge(old_index)
    for bond in case.packed_structure.topology.bonds:
        mapped_endpoints = tuple(
            completed.structure.constitution.atom_index(
                case.packed_structure.constitution.atom_ref_at(index)
            )
            for index in bond.endpoint_pair()
        )
        assert completed.structure.topology.bond_between(*mapped_endpoints) == replace(
            bond, atom_index_1=mapped_endpoints[0], atom_index_2=mapped_endpoints[1]
        )
    assert memory.has_adopted_transformer(case.hydrogenation)
    assert memory.has_attempted_transformer(case.hydrogenation)


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("clone", [False, True])
def test_preparation_suppresses_same_input_including_equal_clone(
    preparation_case: HydrogenPreparationCase,
    retained_preparation_input: ProteinStructure,
    retained: bool,
    clone: bool,
) -> None:
    case = preparation_case
    source = case.input_structure
    action: WorkflowStateAction = case.hydrogenation
    if retained:
        source = retained_preparation_input
        nad = next(r for r in source.constitution.ligands if r.component_id == "NAD")
        action = RetainedNonPolymerHydrogenCompletionTransformer(
            ResidueSetScope((nad.residue_id,))
        )
    memory = WorkflowPlannerMemory().with_attempted_transformer(
        action, input_structure=source
    )
    current = source
    if clone:
        current = ProteinStructure.from_payload(
            constitution=current.constitution,
            geometry=current.geometry,
            topology=current.topology,
            polymer_blueprint=current.polymer_blueprint,
            provenance=current.provenance,
        )
        assert current is not source

    assert memory.blocks_transformer_attempt(action, structure=current)
    assert not memory.adopted_transformers


def test_retained_h_attempt_tracks_actual_completion_and_context_geometry(
    retained_preparation_input: ProteinStructure,
) -> None:
    source = retained_preparation_input
    nad = next(r for r in source.constitution.ligands if r.component_id == "NAD")
    action = RetainedNonPolymerHydrogenCompletionTransformer(
        ResidueSetScope((nad.residue_id,))
    )
    memory = WorkflowPlannerMemory().with_attempted_transformer(
        action, input_structure=source
    )
    result = execute_workflow_transformer(
        TransformationResult(structure=source, repairs=(), issues=()),
        transformer=action,
        execution_context=TransformerExecutionContext(
            component_library=build_default_component_library(),
            original_structure=source,
            orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
        ),
    )
    facts = StructureChemistryReadinessFacts.from_structure(result.structure)
    completed_nad = next(
        f for f in facts.retained_non_polymer_facts if f.residue_id == nad.residue_id
    )
    assert not completed_nad.requires_hydrogen_completion()
    assert completed_nad.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert not memory.blocks_transformer_attempt(action, structure=result.structure)

    neighbor = next(source.constitution.iter_residues())
    index = source.constitution.residue_index(neighbor.residue_id)
    geometry = source.residue_geometry(index)
    atom_name = neighbor.atom_sites[0].name
    atom = geometry.atom_geometry(atom_name)
    position = atom.position
    changed_context = source.with_updated_residue_geometries(
        (
            (
                neighbor.residue_id,
                geometry.with_atom_geometry(
                    atom_name,
                    atom.with_position(Vec3(position.x + 1e-9, position.y, position.z)),
                ),
            ),
        )
    )
    assert changed_context.residue_geometry(
        changed_context.constitution.residue_index(nad.residue_id)
    ) == source.residue_geometry(source.constitution.residue_index(nad.residue_id))
    assert not memory.blocks_transformer_attempt(action, structure=changed_context)
    assert memory.blocks_transformer_attempt(action, structure=source)


def test_preparation_recognizes_equal_defensively_copied_geometry(
    preparation_case: HydrogenPreparationCase,
) -> None:
    case = preparation_case
    source = case.input_structure
    copied_geometry = StructureGeometry(
        constitution=source.constitution,
        atom_geometries=(
            atom.with_position(Vec3(atom.position.x, atom.position.y, atom.position.z))
            for atom in source.geometry.atom_geometries
        ),
    )
    current = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=copied_geometry,
        topology=source.topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    assert current.geometry is not source.geometry
    assert current == source
    memory = WorkflowPlannerMemory().with_attempted_transformer(
        case.hydrogenation, input_structure=source
    )
    assert memory.blocks_transformer_attempt(case.hydrogenation, structure=current)


def test_one_hydrogen_scope_does_not_block_another_incomplete_scope(
    preparation_case: HydrogenPreparationCase,
) -> None:
    case = preparation_case
    lysine = next(
        r
        for r in case.input_structure.constitution.iter_residues()
        if r.component_id == "LYS"
    )
    source = case.input_structure.without_hydrogens_in_residues((lysine.residue_id,))
    other = HydrogenCompletionTransformer(ResidueSetScope((lysine.residue_id,)))
    memory = WorkflowPlannerMemory().with_attempted_transformer(
        case.hydrogenation, input_structure=source
    )
    assert not memory.blocks_transformer_attempt(other, structure=source)
    assert memory.blocks_transformer_attempt(case.hydrogenation, structure=source)


@pytest.mark.parametrize("change", ["coordinates", "charge", "graph", "hydrogens"])
def test_changed_canonical_input_does_not_inherit_preparation_suppression(
    preparation_case: HydrogenPreparationCase,
    change: str,
) -> None:
    case = preparation_case
    source = case.input_structure
    memory = WorkflowPlannerMemory().with_adopted_transformer(
        case.hydrogenation, input_structure=source
    )
    current = source
    if change == "coordinates":
        residue_id = case.hydrogenation.scope.residue_ids[0]
        index = source.constitution.residue_index(residue_id)
        geometry = source.residue_geometry(index)
        position = geometry.position("CA")
        current = source.with_updated_residue_geometries(
            (
                (
                    residue_id,
                    geometry.with_atom_geometry(
                        "CA",
                        geometry.atom_geometry("CA").with_position(
                            Vec3(position.x + 1e-9, position.y, position.z)
                        ),
                    ),
                ),
            )
        )
    elif change == "hydrogens":
        current = case.completed_structure
    else:
        atoms = source.topology.atom_topologies
        bonds = source.topology.bonds
        if change == "charge":
            lysine = next(
                r
                for r in source.constitution.iter_residues()
                if r.component_id == "LYS"
            )
            nz = source.constitution.atom_index(AtomRef(lysine.residue_id, "NZ"))
            assert source.topology.formal_charge(nz) != 0
            atoms = (
                *atoms[: nz.value],
                AtomTopology(formal_charge=0),
                *atoms[nz.value + 1 :],
            )
        else:
            bonds = bonds[1:]
        current = ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=StructureTopology(
                constitution=source.constitution, atom_topologies=atoms, bonds=bonds
            ),
            polymer_blueprint=source.polymer_blueprint,
            provenance=source.provenance,
        )

    assert not memory.blocks_transformer_attempt(case.hydrogenation, structure=current)
    assert memory.blocks_transformer_attempt(case.hydrogenation, structure=source)
    assert memory.has_adopted_transformer(case.hydrogenation)


def test_repeated_adoption_retains_all_input_attempts_and_prevents_exact_revisit(
    preparation_case: HydrogenPreparationCase,
) -> None:
    case = preparation_case
    first = WorkflowPlannerMemory().with_adopted_transformer(
        case.hydrogenation, input_structure=case.input_structure
    )
    following = first.with_adopted_transformer(
        case.hydrogenation, input_structure=case.packed_structure
    )
    assert following.blocks_transformer_attempt(
        case.hydrogenation, structure=case.input_structure
    )
    assert following.blocks_transformer_attempt(
        case.hydrogenation, structure=case.packed_structure
    )
    assert following.attempted_transformers == (case.hydrogenation,)
    assert following.adopted_transformers == (case.hydrogenation,)
    assert first.blocks_transformer_attempt(
        case.hydrogenation, structure=case.input_structure
    )
    assert not first.blocks_transformer_attempt(
        case.hydrogenation, structure=case.packed_structure
    )
    assert (
        following.with_adopted_transformer(
            case.hydrogenation, input_structure=case.packed_structure
        )
        is following
    )


@pytest.mark.parametrize("family", ["packing", "local_ff"])
def test_coordinate_actions_still_block_reapplication_on_changed_input(
    preparation_case: HydrogenPreparationCase,
    family: str,
) -> None:
    case = preparation_case
    action = (
        case.packing_action
        if family == "packing"
        else LocalRefinementTransformer.from_repair_refinement(
            RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues(
                    case.hydrogenation.scope.residue_ids
                ),
                binding=RecommendedContinuousRelaxationBinding(),
            )
        )
    )
    memory = WorkflowPlannerMemory().with_attempted_transformer(
        action, input_structure=case.input_structure
    )
    assert memory.blocks_transformer_attempt(action, structure=case.packed_structure)
    assert memory.attempts[0].input_structure is None
    assert not memory.adopted_transformers


def test_native_receipts_survive_input_bound_attempts_without_false_adoption(
    preparation_case: HydrogenPreparationCase,
) -> None:
    case = preparation_case
    receipt = ContinuousRelaxationExecution(
        settings=ContinuousRelaxationSettings(
            profile=ContinuousRelaxationProfile.RDKIT_MMFF
        ),
        movable_scope=AtomSetScope(
            (AtomRef(case.hydrogenation.scope.residue_ids[0], "CA"),)
        ),
        minimizer_status=1,
        backend_version="2026.03.2",
    )
    memory = (
        WorkflowPlannerMemory()
        .with_returned_relaxations((receipt,))
        .with_attempted_transformer(
            case.hydrogenation, input_structure=case.input_structure
        )
    )
    assert memory.relaxation_executions == (receipt,)
    assert memory.relaxation_executions[0] is receipt
    assert memory.adopted_transformers == ()
    assert memory.blocks_transformer_attempt(
        case.hydrogenation, structure=case.input_structure
    )
    assert not memory.blocks_transformer_attempt(
        case.hydrogenation, structure=case.packed_structure
    )

    adopted = memory.with_adopted_transformer(
        case.hydrogenation, input_structure=case.packed_structure
    )
    assert adopted.relaxation_executions == (receipt,)
    assert adopted.adopted_transformers == (case.hydrogenation,)
    assert adopted.with_returned_relaxations((receipt,)) is adopted


def test_engine_records_preparation_input_before_materializing_output(
    preparation_case: HydrogenPreparationCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = preparation_case
    observed: list[tuple[ProteinStructure | None, WorkflowPlannerMemory]] = []
    original = WorkflowPlannerMemory.with_adopted_transformer

    def record_adoption(
        memory: WorkflowPlannerMemory,
        transformer: WorkflowStateAction,
        *,
        input_structure: ProteinStructure | None = None,
    ) -> WorkflowPlannerMemory:
        following = original(memory, transformer, input_structure=input_structure)
        if transformer == case.hydrogenation:
            observed.append((input_structure, following))
        return following

    monkeypatch.setattr(
        WorkflowPlannerMemory, "with_adopted_transformer", record_adoption
    )
    result = process_structure(case.input_structure, requested_goals=case.goals.goals)
    assert result.requested_goals_fully_satisfied()
    assert len(observed) == 1
    actual_input, memory = observed[0]
    assert actual_input == case.input_structure
    assert memory.blocks_transformer_attempt(
        case.hydrogenation, structure=case.input_structure
    )
    assert not memory.blocks_transformer_attempt(
        case.hydrogenation, structure=result.structure
    )
