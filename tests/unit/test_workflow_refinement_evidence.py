"""Native returns survive refinement projections and workflow reporting."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.workflow_proposals import workflow_proposals

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope
from protrepair.state import StructureProjectionStateFacts
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.completion.hydrogen.protonation import (
    DisabledHistidineProtonationRequest,
)
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.continuous.binding import ManualContinuousRelaxationBinding
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.result import (
    RefinementTransformationResult,
)
from protrepair.transformer.refinement.spec import (
    RepairRefinementSpec,
)
from protrepair.workflow.actions import local_refinement as local
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.contracts.planning import WorkflowPlanningContext
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowTransformRequests,
)
from protrepair.workflow.contracts.result import ProcessResult
from protrepair.workflow.engine import runtime as workflow_runtime
from protrepair.workflow.engine.finalization import finalize_workflow_result
from protrepair.workflow.planning.planner import (
    WorkflowPlanningOutcome,
)
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)


@pytest.fixture(scope="module")
def source():
    return read_structure(Path("tests/fixtures/pdb/1aho.pdb"))


def _evidence(profile=ContinuousRelaxationProfile.RDKIT_UFF, *, status=1):
    return ContinuousRelaxationExecution(
        settings=ContinuousRelaxationSettings(profile=profile),
        movable_scope=AtomSetScope(
            (AtomRef(ResidueId("A", 10), "CA"), AtomRef(ResidueId("A", 10), "CB"))
        ),
        minimizer_status=status,
        backend_version="2026.03.2",
    )


def test_current_h_warning_filter_preserves_native_returns_and_other_issues(
    source, monkeypatch
):
    execution = _evidence()
    transformer = local.LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_atoms(execution.movable_scope.atom_refs),
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationProfile.RDKIT_MMFF
            ),
        )
    )
    state_warning = ValidationIssue(
        kind=ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN,
        severity=IssueSeverity.WARNING,
        message="current unresolved fixed H",
    )
    other_warning = ValidationIssue(
        kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
        severity=IssueSeverity.WARNING,
        message="attempt-specific chemistry failure",
    )
    returned = RefinementTransformationResult(
        source,
        repairs=(),
        issues=(state_warning, other_warning, state_warning, other_warning),
        relaxation_executions=(execution,),
    )
    monkeypatch.setattr(
        local, "apply_repair_stage_local_refinement", lambda *args, **kwargs: returned
    )
    actual = workflow_runtime.execute_workflow_transformer(
        ProcessResult(source, repairs=(), issues=()),
        transformer=transformer,
        execution_context=TransformerExecutionContext(
            component_library=build_default_component_library(),
            original_structure=source,
            orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
        ),
    )
    assert isinstance(actual, RefinementTransformationResult)
    assert actual.structure is source
    assert actual.relaxation_executions == (execution,)
    assert actual.issues == (other_warning, other_warning)


def test_workflow_records_actual_returns_in_next_planning_pass(source, monkeypatch):
    execution = _evidence()
    transformer = local.LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_atoms(execution.movable_scope.atom_refs),
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationProfile.RDKIT_MMFF
            ),
        )
    )
    planning_history = []

    def plan(structure, *, planner_memory, **kwargs):
        planning_history.append(planner_memory.relaxation_executions)
        return WorkflowPlanningOutcome(
            structure_planning_signature=StructurePlanningSignature.from_facts(
                StructureProjectionStateFacts.from_structure(structure)
            ),
            proposals=workflow_proposals(
                (transformer,) if len(planning_history) == 1 else ()
            ),
        )

    monkeypatch.setattr(workflow_runtime, "plan_workflow_actions", plan)
    monkeypatch.setattr(
        local,
        "apply_repair_stage_local_refinement",
        lambda result, **kwargs: (
            RefinementTransformationResult.with_observed_executions(
                result, (execution,)
            )
        ),
    )
    result = workflow_runtime.execute_iterative_workflow(
        source,
        requested_goals=RequestedGoalSet(()),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=transformer.repair_refinement
        ),
        component_library=build_default_component_library(),
        planning_context=WorkflowPlanningContext(),
        reference_structure=None,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
        histidine_protonation=DisabledHistidineProtonationRequest(),
    )
    assert planning_history == [(), (execution,)]
    (branch,) = result.branches
    assert branch.result.structure is source
    assert branch.outcome.relaxation_executions == (execution,)

    final = finalize_workflow_result(
        branches=result.branches,
        planning_context=WorkflowPlanningContext(),
        component_library=build_default_component_library(),
        requested_analyses=frozenset(),
    )
    assert final.relaxation_executions == (execution,)
    assert final.structure is source

    assert final.requested_goal_report is not None
    assert final.branch_report is not None
    missing_history = replace(
        final.branch_report,
        outcomes=tuple(
            replace(outcome, relaxation_executions=())
            for outcome in final.branch_report.outcomes
        ),
    )
    for report in (None, missing_history):
        with pytest.raises(ValueError, match="native execution history"):
            ProcessResult.from_transformation_result(
                branch.result, branch_report=report
            )
        with pytest.raises(ValueError, match="native execution history"):
            ProcessResult.from_transformation_result(final, branch_report=report)
    with pytest.raises(ValueError, match="native execution history"):
        final.with_workflow_reporting(
            requested_goal_report=final.requested_goal_report,
            branch_report=missing_history,
        )

    reported = final.with_workflow_reporting(
        requested_goal_report=final.requested_goal_report,
        branch_report=final.branch_report,
    )
    assert reported.relaxation_executions == (execution,)
    assert reported.with_analyses(None).relaxation_executions == (execution,)
    assert reported.with_structure(source).relaxation_executions == (execution,)
    assert reported.with_appended_issues(final.issues).relaxation_executions == (
        execution,
    )
