"""Bonded-geometry coverage, motion custody, and scope advancement."""

from pathlib import Path

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure, read_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import BondedGeometryState
from protrepair.state.bonded_geometry import BondedGeometryObservation
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.continuous.shared import apply_position_updates
from protrepair.workflow.actions.bonded_geometry import (
    BondedGeometryFeedback,
    BondedGeometryRepairTransformer,
)
from protrepair.workflow.actions.bonded_geometry_motion import BondedGeometryMotion
from protrepair.workflow.contracts import (
    RequestedGoalSet,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.planning.action.attempt import WorkflowActionAttempt
from protrepair.workflow.planning.default_action_registry.bonded_geometry_proposals import (
    next_bonded_geometry_transformer,
)
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    plan_workflow_actions,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pdb"


def test_unresolved_histidine_is_not_complete_bonded_geometry() -> None:
    source = (_FIXTURES / "1aho.pdb").read_text()
    structure = read_structure_string(
        "\n".join(
            line
            for line in source.splitlines()
            if line.startswith("ATOM")
            and line[17:26] == "HIS A  54"
            and line[76:78].strip() not in {"H", "D", "T"}
        )
        + "\nEND\n",
        FileFormat.PDB,
    )
    histidine = ResidueId("A", 54)
    geometry = BondedGeometryObservation.from_structure(
        structure, component_library=build_default_component_library()
    )
    assert geometry.assessed_references
    assert geometry.unassessed_residue_ids == (histidine,)
    assert geometry.observed_state() is not BondedGeometryState.WITHIN_REFERENCE


def test_retry_uses_bond_frontier_and_can_extend_past_first_shell() -> None:
    structure = read_structure(_FIXTURES / "span-reconstruction" / "1ubq.pdb")
    rid = ResidueId("A", 8)
    geometry = BondedGeometryObservation.from_structure(
        structure, component_library=build_default_component_library()
    )
    motion = BondedGeometryMotion(structure, geometry.qualified_residue_ids)
    focus = (AtomRef(rid, "C"), AtomRef(rid, "O"))
    seed = next_bonded_geometry_transformer(focus, (), motion)
    assert seed is not None
    rejected = BondedGeometryFeedback(failure="focus residual", residual_focus=True)
    first = WorkflowActionAttempt(seed, structure, rejected)
    plus_one = next_bonded_geometry_transformer(focus, (first,), motion)
    assert plus_one is not None
    assert set(plus_one.scope.atom_refs) - set(seed.scope.atom_refs) == {
        AtomRef(rid, "CA")
    }
    second = WorkflowActionAttempt(plus_one, structure, rejected)
    plus_two = next_bonded_geometry_transformer(focus, (first, second), motion)
    assert plus_two is not None
    assert AtomRef(rid, "CB") in plus_two.scope.atom_refs
    assert AtomRef(rid, "CB") not in motion.covalent_frontier(
        frozenset(seed.scope.atom_refs)
    )


def test_scope_stops_at_a_chain_break_but_can_consider_a_contact() -> None:
    structure = read_structure(_FIXTURES / "span-reconstruction" / "1ubq.pdb")
    geometry = BondedGeometryObservation.from_structure(
        structure, component_library=build_default_component_library()
    )
    motion = BondedGeometryMotion(structure, geometry.qualified_residue_ids)
    rid = ResidueId("A", 13)
    focus = (AtomRef(rid, "C"), AtomRef(rid, "O"))
    possible_contact = AtomRef(ResidueId("A", 34), "N")
    seed = next_bonded_geometry_transformer(focus, (), motion)
    assert seed is not None
    assert possible_contact not in motion.covalent_frontier(
        frozenset(seed.scope.atom_refs)
    )
    unavailable = WorkflowActionAttempt(
        seed, structure, BondedGeometryFeedback(failure="readiness unavailable")
    )
    assert next_bonded_geometry_transformer(focus, (unavailable,), motion) is None

    observed_contact = WorkflowActionAttempt(
        seed,
        structure,
        BondedGeometryFeedback("new crossing clash", (possible_contact,)),
    )
    expanded = next_bonded_geometry_transformer(focus, (observed_contact,), motion)
    assert expanded is not None
    assert possible_contact in expanded.scope.atom_refs
    partner_index = structure.constitution.atom_index(possible_contact)
    partner_position = structure.geometry.atom_geometry(partner_index).position
    other_parent = apply_position_updates(
        structure,
        moved_positions={
            partner_index: Vec3(
                partner_position.x + 0.05, partner_position.y, partner_position.z
            )
        },
    )
    assert (
        next_bonded_geometry_transformer(
            focus,
            (observed_contact,),
            BondedGeometryMotion(other_parent, geometry.qualified_residue_ids),
        )
        == seed
    )


def test_same_parent_followup_precedes_a_different_untried_defect() -> None:
    structure = read_structure(_FIXTURES / "span-reconstruction" / "1ubq.pdb")
    rid = ResidueId("A", 8)
    focus = (AtomRef(rid, "C"), AtomRef(rid, "O"))
    carbon, oxygen = (structure.constitution.atom_index(ref) for ref in focus)
    c = structure.geometry.atom_geometry(carbon).position
    o = structure.geometry.atom_geometry(oxygen).position
    shortened = Vec3.from_iterable(a + 0.8 * (b - a) for a, b in zip(c, o, strict=True))
    parent = apply_position_updates(structure, moved_positions={oxygen: shortened})
    goals = RequestedGoalSet(
        (
            requested_process_goal(
                scope=WholeStructureScope(),
                value=BondedGeometryState.WITHIN_REFERENCE,
            ),
        )
    )

    requests = WorkflowTransformRequests()
    (first,) = plan_workflow_actions(
        parent, requested_goals=goals, transform_requests=requests
    ).transformers
    assert isinstance(first, BondedGeometryRepairTransformer)
    assert first.focus == focus
    report = BondedGeometryObservation.from_structure(
        parent, component_library=build_default_component_library()
    )
    assert any(term != focus for term, _ in report.residual_excesses)
    rejected = WorkflowPlannerMemory().with_attempted_transformer(
        first,
        input_structure=parent,
        bonded_geometry_feedback=BondedGeometryFeedback(
            "boundary remained", residual_focus=True
        ),
    )
    (followup,) = plan_workflow_actions(
        parent,
        requested_goals=goals,
        transform_requests=requests,
        planner_memory=rejected,
    ).transformers
    assert isinstance(followup, BondedGeometryRepairTransformer)
    assert followup.focus == focus
    assert set(first.scope.atom_refs) < set(followup.scope.atom_refs)


def test_source_hydrogen_and_its_parent_are_never_nominated() -> None:
    structure = read_structure(_FIXTURES / "1aho.pdb")
    rid = ResidueId("A", 27)
    motion = BondedGeometryMotion(structure, frozenset((rid,)))
    assert not motion.group_for(AtomRef(rid, "C"))
    prepared = add_hydrogens(structure, build_default_component_library()).structure
    motion = BondedGeometryMotion(prepared, frozenset((rid,)))
    assert not motion.group_for(AtomRef(rid, "CA"))
    assert motion.heavy_contact_partner(AtomRef(rid, "HA")) is None
    assert motion.group_for(AtomRef(rid, "C")) == frozenset((AtomRef(rid, "C"),))
