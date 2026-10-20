"""Goal classifications and selection for peers left at the search limit."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)
from tests.support.workflow_proposals import workflow_proposals

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.scope import AtomSetScope, ResidueSetScope, WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.hydrogen.protonation import (
    DisabledHistidineProtonationRequest,
)
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.transformer.continuous.binding import ManualContinuousRelaxationBinding
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.result import RefinementTransformationResult
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.refinement.speculative_planning import SpeculativeStopReason
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts import (
    OrphanFragmentPolicy,
    RequestedGoalSet,
    RequestedGoalStatus,
    WorkflowBranchStatus,
    WorkflowPhaseStatus,
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
)
from protrepair.workflow.engine import runtime
from protrepair.workflow.engine.finalization import finalize_workflow_result


@pytest.fixture
def prepared_with_unknown_component():
    structure = build_bonded_structure(
        chains=tuple(
            chain_payload(
                chain_id,
                (
                    residue_payload(
                        component_id=component,
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
            for chain_id, component, shift in (("A", "GLY", 0.0), ("B", "UNK", 30.0))
        ),
        source_format=FileFormat.PDB,
    )
    return add_hydrogens(augment_c_terminal_oxt(structure).structure).structure


@pytest.mark.parametrize("goal_kind", ("unsupported", "blocked"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("terminal_peer", (False, True))
def test_exhaustion_classifies_every_unplanned_peer(
    monkeypatch, prepared_with_unknown_component, goal_kind, reverse, terminal_peer
):
    source = prepared_with_unknown_component
    library = build_default_component_library()
    goal = requested_process_goal(
        scope=ResidueSetScope((ResidueId("A", 1),))
        if goal_kind == "unsupported"
        else WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    expected_status = (
        RequestedGoalStatus.UNSUPPORTED
        if goal_kind == "unsupported"
        else RequestedGoalStatus.BLOCKED
    )
    actions = tuple(
        LocalRefinementTransformer.from_repair_refinement(
            RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_atoms(
                    (AtomRef(ResidueId("A", 1), name),)
                ),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationProfile.RDKIT_UFF
                ),
            )
        )
        for name in ("N", "CA", "C")
    )
    initial = tuple(reversed(actions[:2])) if reverse else actions[:2]
    original_plan = runtime.plan_workflow_actions
    planned_inputs = []
    outputs = []

    def plan(structure, *, planner_memory, **kwargs):
        assert not any(
            old is structure and memory == planner_memory
            for old, memory in planned_inputs
        )
        planned_inputs.append((structure, planner_memory))
        outcome = original_plan(structure, planner_memory=planner_memory, **kwargs)
        if not planner_memory.adopted_transformers:
            proposals = initial
        elif terminal_peer and structure is outputs[1].structure:
            proposals = ()
        else:
            proposals = actions[2:]
        return replace(outcome, proposals=workflow_proposals(proposals))

    def execute(result, *, transformer, execution_context):
        del execution_context
        assert len(outputs) < 2
        movable_scope = transformer.repair_refinement.scope_spec.scope
        assert isinstance(movable_scope, AtomSetScope)
        execution = ContinuousRelaxationExecution(
            settings=ContinuousRelaxationSettings(
                profile=ContinuousRelaxationProfile.RDKIT_UFF
            ),
            movable_scope=movable_scope,
            minimizer_status=1,
            backend_version="test-return",
        )
        output = RefinementTransformationResult.with_observed_executions(
            replace(
                result,
                structure=result.structure.with_provenance(
                    replace(
                        result.structure.provenance,
                        ingress=replace(
                            result.structure.provenance.ingress,
                            source_name=str(len(outputs)),
                        ),
                    )
                ),
            ),
            (execution,),
        )
        outputs.append(output)
        return output

    parser_loop = Mock(side_effect=AssertionError("the node limit executes no repair"))
    monkeypatch.setattr(runtime, "plan_workflow_actions", plan)
    monkeypatch.setattr(runtime, "execute_workflow_transformer", execute)
    monkeypatch.setattr(
        runtime, "execute_bounded_parser_witness_repair_loop", parser_loop
    )
    context = WorkflowPlanningContext(
        max_speculative_nodes=3,
        branch_preference_policy=WorkflowBranchPreferencePolicy(
            axes=(WorkflowBranchQualityAxis.REQUESTED_GOALS,),
        ),
    )
    result = runtime.execute_iterative_workflow(
        source,
        requested_goals=RequestedGoalSet((goal,)),
        transform_requests=WorkflowTransformRequests(),
        component_library=library,
        planning_context=context,
        reference_structure=None,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
        histidine_protonation=DisabledHistidineProtonationRequest(),
    )
    assert result.stop_reason is SpeculativeStopReason.ITERATION_LIMIT_REACHED
    assert len(outputs) == 2
    assert len(result.branches) == 2
    for branch in result.branches:
        outcome = branch.outcome.requested_goal_report.outcome_for(goal)
        assert outcome is not None and outcome.status is expected_status
        assert branch.outcome.branch_quality_score.satisfied_goal_count == 0
        if goal_kind == "blocked":
            assert outcome.blocking_scopes == (ResidueSetScope((ResidueId("B", 1),)),)
            phase = branch.outcome.phase_report.outcome_for(
                WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION
            )
            assert phase is not None and phase.status is WorkflowPhaseStatus.BLOCKED
    assert result.branches[1].outcome.status is (
        WorkflowBranchStatus.TERMINAL
        if terminal_peer
        else WorkflowBranchStatus.FRONTIER
    )
    finalized = finalize_workflow_result(
        branches=result.branches,
        planning_context=context,
        component_library=library,
        requested_analyses=frozenset(),
        stop_reason=result.stop_reason,
    )
    assert finalized.structure is outputs[0].structure
    assert finalized.requested_goal_report is not None
    selected_goal = finalized.requested_goal_report.outcome_for(goal)
    assert selected_goal is not None and selected_goal.status is expected_status
    assert not finalized.requested_goals_fully_satisfied()
    assert len(planned_inputs) == 3
    parser_loop.assert_not_called()
