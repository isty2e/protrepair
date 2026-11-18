"""Opt-in X-H length reporting and terminal workflow repair tests."""

from pathlib import Path

import pytest
from tests.support.hydrogen_lengths import (
    HG,
    REFERENCE_CIF,
    THIOL,
    reference,
    structure_with_thiol,
    with_position,
)

from protrepair import process_structure
from protrepair.analysis import HydrogenLengthConvention, HydrogenLengthStatus
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.diagnostics.parser_readability import (
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure, write_structure_string
from protrepair.scope import AtomSetScope, WholeStructureScope
from protrepair.state import ClashPresenceState, ClashState, HydrogenCoverageState
from protrepair.structure import AtomRef, FileFormat, ResidueId
from protrepair.workflow.contracts import (
    HydrogenLengthRepairStatus,
    HydrogenLengthRequest,
    LigandPolicy,
    ProcessResult,
    RequestedGoalStatus,
    StructureIngressOptions,
    WorkflowPlanningContext,
    requested_process_goal,
)
from protrepair.workflow.engine.hydrogen_lengths import WorkflowHydrogenLengthPlan

pytestmark = pytest.mark.workflow


def nuclear_repair(*, supplied=None, scope=None):
    return HydrogenLengthRequest(
        (reference() if supplied is None else supplied,),
        convention=HydrogenLengthConvention.NUCLEAR,
        scope=scope,
        repair=True,
    )


def repair_row(result: ProcessResult):
    report = result.hydrogen_length_report
    assert report is not None and report.repair_outcomes is not None
    (row,) = report.repair_outcomes
    return row


def test_workflow_reports_length_only_when_explicitly_requested(tmp_path):
    path = tmp_path / "short-thiol.pdb"
    path.write_text(write_structure_string(structure_with_thiol(1.0), FileFormat.PDB))
    legacy = process_structure(path)
    checked = process_structure(
        path, hydrogen_lengths=HydrogenLengthRequest((reference(),))
    )

    assert legacy.hydrogen_length_report is None
    assert checked.hydrogen_length_report is not None
    assert (
        checked.hydrogen_length_report.observations[0].status
        is HydrogenLengthStatus.TOO_SHORT
    )
    assert checked.hydrogen_length_report.repair_outcomes is None
    assert checked.structure == legacy.structure
    assert checked.repairs == legacy.repairs
    assert (
        tuple(
            i
            for i in checked.issues
            if i.kind is not ValidationIssueKind.HYDROGEN_LENGTH_OUTLIER
        )
        == legacy.issues
    )
    (issue,) = tuple(
        i
        for i in checked.issues
        if i.kind is ValidationIssueKind.HYDROGEN_LENGTH_OUTLIER
    )
    assert issue.residue_impacts[0].atom_names == ("HG", "SG")


def test_unknown_convention_is_not_a_repair_profile():
    with pytest.raises(ValueError, match="declared convention"):
        HydrogenLengthRequest((reference(),), repair=True)


@pytest.mark.parametrize(
    ("convention", "target"),
    [
        (HydrogenLengthConvention.ELECTRON_CLOUD, 1.2),
        (HydrogenLengthConvention.NUCLEAR, 1.338),
    ],
)
def test_opt_in_repair_preserves_the_ray_and_all_nonposition_fields(convention, target):
    dg = AtomRef(THIOL.residue_id, "DG")
    raw = structure_with_thiol(1.0, name="DG", element="D")
    c = raw.constitution
    source = with_position(raw, c.atom_index(dg), Vec3(0.6, 0.8, 0.0))
    supplied = reference(REFERENCE_CIF.replace("1.34 0.01", "1.338 0.01"))
    legacy = process_structure(source)
    fixed = process_structure(
        source,
        hydrogen_lengths=HydrogenLengthRequest(
            (supplied,), convention=convention, scope=AtomSetScope((dg,)), repair=True
        ),
    )
    assert repair_row(fixed).status is HydrogenLengthRepairStatus.REPAIRED
    assert fixed.hydrogen_length_report is not None
    assert (
        fixed.hydrogen_length_report.observations[0].status
        is HydrogenLengthStatus.WITHIN_REFERENCE
    )

    structure = fixed.structure
    assert structure.constitution == legacy.structure.constitution
    assert structure.topology == legacy.structure.topology
    assert structure.provenance == legacy.structure.provenance
    assert structure.polymer_blueprint == legacy.structure.polymer_blueprint
    index = structure.constitution.atom_index(dg)
    assert tuple(structure.geometry.position(index)) == pytest.approx(
        (0.6 * target, 0.8 * target, 0.0)
    )
    for i, expected in enumerate(legacy.structure.geometry.atom_geometries):
        actual = structure.geometry.atom_geometries[i]
        assert actual == (
            expected.with_position(structure.geometry.position(index))
            if i == index.value
            else expected
        )
    assert fixed.repairs[-1].kind is RepairEventKind.HYDROGEN_LENGTH_ADJUSTED
    assert fixed.repairs[-1].residue_impacts[0].atom_names == ("DG",)
    assert ValidationIssueKind.HYDROGEN_LENGTH_OUTLIER not in {
        i.kind for i in fixed.issues
    }

    probe = probe_rdkit_no_conect_parser_readability(
        structure, assigned_hydrogen_parents={dg: THIOL}
    )
    (attachment,) = probe.projected_hydrogen_attachments
    assert attachment.neighbors == (THIOL,)
    assert attachment.parent_distance_angstrom == pytest.approx(target, abs=0.001)


@pytest.mark.parametrize(
    ("distance", "state"),
    [
        (0.0, HydrogenLengthRepairStatus.UNAVAILABLE),
        (-1.30, HydrogenLengthRepairStatus.UNAVAILABLE),
        (1.33, HydrogenLengthRepairStatus.IN_WINDOW),
    ],
)
def test_unavailable_and_in_window_hydrogens_are_never_moved(distance, state):
    source = structure_with_thiol(distance)
    fixed = process_structure(source, hydrogen_lengths=nuclear_repair())
    assert repair_row(fixed).status is state
    assert fixed.structure.geometry == source.geometry
    assert RepairEventKind.HYDROGEN_LENGTH_ADJUSTED not in {
        ev.kind for ev in fixed.repairs
    }
    if distance < 0:
        assert ValidationIssueKind.HYDROGEN_PARENT_POSITION_CONFLICT in {
            i.kind for i in fixed.issues
        }


@pytest.mark.parametrize("exhausted", [False, True])
def test_repair_does_not_stage_preparation_or_consume_search_nodes(exhausted):
    source = structure_with_thiol(1.0)
    goals = (
        (
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        )
        if exhausted
        else ()
    )
    context = WorkflowPlanningContext(max_speculative_nodes=1)
    legacy = process_structure(source, requested_goals=goals, planning_context=context)
    fixed = process_structure(
        source,
        requested_goals=goals,
        planning_context=context,
        hydrogen_lengths=nuclear_repair(),
    )
    assert legacy.branch_report is not None and fixed.branch_report is not None
    assert fixed.branch_report.search_exhausted() is exhausted
    assert (
        fixed.branch_report.preferred_node_id == legacy.branch_report.preferred_node_id
    )
    assert fixed.requested_goal_report == legacy.requested_goal_report
    assert repair_row(fixed).status is HydrogenLengthRepairStatus.REPAIRED
    assert fixed.repairs[:-1] == legacy.repairs
    assert fixed.relaxation_executions == legacy.relaxation_executions == ()


@pytest.mark.parametrize(
    ("scope", "value"),
    [
        (WholeStructureScope(), ClashPresenceState.NONE),
        (AtomSetScope((HG,)), ClashState.NONE),
    ],
)
def test_clash_goal_stays_unsupported_in_the_terminal_workflow(scope, value):
    goal = requested_process_goal(scope=scope, value=value)
    result = process_structure(
        structure_with_thiol(1.0),
        requested_goals=(goal,),
        hydrogen_lengths=nuclear_repair(),
    )
    assert repair_row(result).status is HydrogenLengthRepairStatus.REPAIRED
    assert result.requested_goal_report is not None
    outcome = result.requested_goal_report.outcome_for(goal)
    assert outcome is not None and outcome.status is RequestedGoalStatus.UNSUPPORTED


@pytest.mark.parametrize("ref", [HG, THIOL])
def test_changed_source_positions_are_unavailable_even_if_currently_in_window(ref):
    source = structure_with_thiol(1.33)
    index = source.constitution.atom_index(ref)
    previous = source.geometry.position(index)
    changed = with_position(source, index, previous.with_offset(0.01, 0.0, 0.0))
    retained = WorkflowHydrogenLengthPlan(source, nuclear_repair()).complete(
        ProcessResult(structure=changed, repairs=(), issues=()),
        component_library=build_default_component_library(),
    )
    assert retained.hydrogen_length_report is not None
    assert (
        retained.hydrogen_length_report.observations[0].status
        is HydrogenLengthStatus.WITHIN_REFERENCE
    )
    assert repair_row(retained).status is HydrogenLengthRepairStatus.UNAVAILABLE
    assert repair_row(retained).detail == "input hydrogen or parent evidence changed"
    assert retained.structure is changed


def test_hydrogens_added_by_another_request_are_not_in_the_length_batch():
    source = read_structure(
        Path(__file__).parents[1] / "fixtures/pdb/span-reconstruction/1crn.pdb"
    ).without_hydrogens()
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    )
    checked = process_structure(
        source, requested_goals=(goal,), hydrogen_lengths=nuclear_repair()
    )
    assert checked.structure.geometry.atom_count() > source.geometry.atom_count()
    report = checked.hydrogen_length_report
    assert report is not None and report.observations == report.repair_outcomes == ()


def test_an_input_hydrogen_removed_during_other_work_is_reported_unavailable():
    source = structure_with_thiol(1.0)
    removed = source.without_hydrogens()
    final = WorkflowHydrogenLengthPlan(source, nuclear_repair()).complete(
        ProcessResult(structure=removed, repairs=(), issues=()),
        component_library=build_default_component_library(),
    )
    assert final.structure is removed
    assert repair_row(final).status is HydrogenLengthRepairStatus.UNAVAILABLE
    assert repair_row(final).detail == "selected input hydrogen is no longer present"


def test_shorten_a_long_bond_and_recover_its_assigned_parent_in_the_parser():
    source = structure_with_thiol(1.85)
    (before,) = probe_rdkit_no_conect_parser_readability(
        source, assigned_hydrogen_parents={HG: THIOL}
    ).projected_hydrogen_attachments
    assert before.neighbors == ()
    result = process_structure(source, hydrogen_lengths=nuclear_repair())
    assert repair_row(result).status is HydrogenLengthRepairStatus.REPAIRED
    (after,) = probe_rdkit_no_conect_parser_readability(
        result.structure, assigned_hydrogen_parents={HG: THIOL}
    ).projected_hydrogen_attachments
    assert after.neighbors == (THIOL,)


def test_rounded_input_inside_the_window_is_not_moved():
    raw = structure_with_thiol(1.30)
    source = with_position(raw, raw.constitution.atom_index(HG), Vec3(1.2999, 0, 0))
    fixed = process_structure(source, hydrogen_lengths=nuclear_repair())
    assert repair_row(fixed).status is HydrogenLengthRepairStatus.UNAVAILABLE
    assert (
        repair_row(fixed).detail == "input length grade changes at PDB output precision"
    )
    assert fixed.structure.geometry == source.geometry


@pytest.mark.parametrize(
    ("target_row", "refusal"),
    [
        ("1.3384 0.000025", "endpoint length outside"),
        ("2.0 0.02", "did not retain only the assigned parent"),
    ],
)
def test_wire_precision_or_missing_native_attachment_rejects_the_endpoint(
    target_row, refusal
):
    source = structure_with_thiol(1.0)
    supplied = reference(REFERENCE_CIF.replace("1.34 0.01", target_row))
    fixed = process_structure(
        source, hydrogen_lengths=nuclear_repair(supplied=supplied)
    )
    row = repair_row(fixed)
    assert row.status is HydrogenLengthRepairStatus.REJECTED
    assert row.detail is not None and refusal in row.detail
    assert fixed.structure.geometry == source.geometry
    assert RepairEventKind.HYDROGEN_LENGTH_ADJUSTED not in {
        ev.kind for ev in fixed.repairs
    }


@pytest.mark.parametrize(
    "obstacle",
    [
        "HETATM    8  C1  UNL B   1       3.180   0.000   0.000  1.00 20.00           C\n",
        "ATOM      8  H   GLY B   1       2.760   0.000   0.000  1.00 20.00           H\n",
    ],
)
def test_new_ligand_or_hydrogen_clash_rejects_a_length_improvement(obstacle):
    source = structure_with_thiol(1.0, suffix=obstacle)
    fixed = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        hydrogen_lengths=nuclear_repair(scope=AtomSetScope((HG,))),
    )
    row = repair_row(fixed)
    assert row.status is HydrogenLengthRepairStatus.REJECTED
    assert row.detail is not None and "new or worsened steric clash" in row.detail
    assert fixed.structure.geometry == source.geometry


def test_only_selected_atoms_move_and_a_refused_batch_is_atomic():
    single = structure_with_thiol(1.0)
    other = "".join(
        f"{line[:6]}{int(line[6:11]) + 7:5d}{line[11:21]}B{line[22:38]}"
        f"{float(line[38:46]) + 10:8.3f}{line[46:]}\n"
        for line in write_structure_string(single, FileFormat.PDB).splitlines()
        if line.startswith("ATOM  ")
    )
    source = structure_with_thiol(
        1.0,
        suffix=other
        + "HETATM   15  C1  UNL C   1       3.180  10.000   0.000  1.00 20.00           C\n",
    )
    second = AtomRef(ResidueId("B", 1), "HG")
    ingress = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    scoped = process_structure(
        source,
        ingress=ingress,
        hydrogen_lengths=nuclear_repair(scope=AtomSetScope((HG,))),
    )
    assert repair_row(scoped).status is HydrogenLengthRepairStatus.REPAIRED
    index = source.constitution.atom_index(second)
    assert scoped.structure.geometry.position(index) == source.geometry.position(index)

    batch = process_structure(
        source,
        ingress=ingress,
        hydrogen_lengths=nuclear_repair(scope=AtomSetScope((HG, second))),
    )
    report = batch.hydrogen_length_report
    assert report is not None and report.repair_outcomes is not None
    assert tuple(row.hydrogen for row in report.repair_outcomes) == (HG, second)
    assert all(
        row.status is HydrogenLengthRepairStatus.REJECTED
        for row in report.repair_outcomes
    )
    assert batch.structure.geometry == source.geometry
    assert RepairEventKind.HYDROGEN_LENGTH_ADJUSTED not in {
        ev.kind for ev in batch.repairs
    }
