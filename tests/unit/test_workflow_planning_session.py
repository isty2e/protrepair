"""Lookahead plan custody and invalidation within a workflow invocation."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.state import StructureProjectionStateFacts
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.labels import ResidueId
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.contracts import (
    RequestedGoalSet,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
)
from protrepair.workflow.engine import runtime
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    WorkflowPlanningOutcome,
    plan_workflow_actions,
)
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)


@pytest.fixture
def planning_case(monkeypatch):
    structure = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    state = runtime.WorkflowRuntimeState(
        result=TransformationResult(structure=structure, repairs=(), issues=()),
        planner_memory=WorkflowPlannerMemory(),
    )
    outcome = WorkflowPlanningOutcome(
        structure_planning_signature=StructurePlanningSignature.from_facts(
            StructureProjectionStateFacts.from_structure(structure)
        ),
        proposals=(),
    )
    plan = Mock(return_value=outcome)
    monkeypatch.setattr(runtime, "plan_workflow_actions", plan)
    session = runtime.WorkflowPlanningSession(
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_default_component_library(),
        planning_context=WorkflowPlanningContext(),
        retained_non_polymer_chemistry_evidence=(),
    )
    return state, session, plan


def test_exact_plan_reuse_and_frontier_eviction(planning_case):
    state, session, plan = planning_case
    assert session.known_outcome(state) is None
    outcome = session.plan(state)
    session.retain_states((state,))
    assert session.plan(state) is outcome
    plan.assert_called_once()

    session.retain_states(())
    assert session.known_outcome(state) is None
    assert session.plan(state) is outcome
    assert plan.call_count == 2


def test_execution_seed_shares_the_planning_observation_and_its_lifetime(planning_case):
    state, session, plan = planning_case
    session.plan(state)
    observation = plan.call_args.kwargs["chemistry_observation"]
    prepared = session.observed_polymer_preparation(state)
    assert prepared is observation.hydrogen_expectation_model.polymer_preparation
    assert session.observed_polymer_preparation(state) is prepared
    assert prepared.context.source is state.result.structure
    plan.assert_called_once()

    session.retain_states(())
    assert session.observed_polymer_preparation(state) is not prepared
    assert plan.call_count == 2


@pytest.mark.parametrize(
    "binding", ("exact", "source", "library", "policy", "evidence")
)
def test_planner_observation_injection_validates_its_context(planning_case, binding):
    state, session, _ = planning_case
    source = state.result.structure
    library = session.component_library
    observation = StructureChemistryObservation(
        source,
        component_library=library,
        allow_retained_non_polymer_rdkit_fallback=binding != "policy",
        retained_non_polymer_chemistry_evidence=(
            RetainedNonPolymerChemistryEvidence(ResidueId("Z", 99), "C", ("C1",)),
        )
        if binding == "evidence"
        else (),
    )
    if binding == "source":
        source = source.with_provenance(source.provenance)
    elif binding == "library":
        library = replace(library)

    def plan():
        return plan_workflow_actions(
            source,
            requested_goals=session.requested_goals,
            transform_requests=session.transform_requests,
            component_library=library,
            chemistry_observation=observation,
        )

    if binding == "exact":
        assert plan() == plan_workflow_actions(
            source,
            requested_goals=session.requested_goals,
            transform_requests=session.transform_requests,
            component_library=library,
        )
    else:
        with pytest.raises(ValueError, match="exact context"):
            plan()


@pytest.mark.parametrize("change", ("result", "attempt", "native"))
def test_changed_planning_state_is_not_reused(planning_case, change):
    state, session, plan = planning_case
    session.plan(state)
    structure = state.result.structure
    residue_id = structure.constitution.chains[0].residues[0].residue_id
    if change == "result":
        changed = replace(state, result=replace(state.result))
    elif change == "attempt":
        changed = replace(
            state,
            planner_memory=state.planner_memory.with_attempted_transformer(
                TerminalAugmentationTransformer(scope=ResidueSetScope((residue_id,)))
            ),
        )
    else:
        execution = ContinuousRelaxationExecution(
            settings=ContinuousRelaxationSettings(
                profile=ContinuousRelaxationProfile.RDKIT_MMFF
            ),
            movable_scope=AtomSetScope(
                (
                    structure.constitution.atom_ref_at(
                        next(iter(structure.geometry.iter_positions()))[0]
                    ),
                )
            ),
            minimizer_status=1,
            backend_version="test-return",
        )
        changed = replace(
            state,
            planner_memory=state.planner_memory.with_returned_relaxations((execution,)),
        )
    assert session.known_outcome(changed) is None
    session.plan(changed)
    assert plan.call_count == 2
    assert plan.call_args.kwargs["planner_memory"] is changed.planner_memory


def test_planning_contexts_do_not_share_outcomes(planning_case):
    state, session, plan = planning_case
    session.plan(state)
    second = replace(
        session, planning_context=WorkflowPlanningContext(max_speculative_nodes=1)
    )
    assert second.known_outcome(state) is None
    second.plan(state)
    assert plan.call_count == 2
    assert plan.call_args.kwargs["planning_context"] is second.planning_context


def test_planning_failure_is_not_cached(planning_case):
    state, session, plan = planning_case
    outcome = plan.return_value
    plan.side_effect = [RuntimeError("assessment failed"), outcome]
    with pytest.raises(RuntimeError, match="assessment failed"):
        session.plan(state)
    assert session.known_outcome(state) is None
    assert session.plan(state) is outcome
    assert plan.call_count == 2
