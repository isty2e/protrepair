"""The planner authorizes attached parser correction on the current state."""

from pathlib import Path

import pytest
from tests.support.workflow_proposals import workflow_proposals

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.topology import LikelyDisulfideBond
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope, ResidueSetScope, WholeStructureScope
from protrepair.state import (
    HydrogenCoverageState,
    StructureProjectionStateFacts,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.labels import AtomRef
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.packing.spec import PackingScope, PackingSpec
from protrepair.transformer.refinement.result import RefinementTransformationResult
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.disulfide_topology import (
    DisulfideTopologyResolutionTransformer,
)
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.actions.stereochemistry import (
    StereochemistryCorrectionTransformer,
)
from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.contracts import (
    OrphanFragmentPolicy,
    RequestedGoalSet,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.engine import runtime
from protrepair.workflow.engine.parser_witness_repair import (
    DirectParserWitnessRepairResult,
)
from protrepair.workflow.planning.planner import WorkflowPlanningOutcome
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)


@pytest.mark.parametrize(
    "source_path",
    ("tests/fixtures/pdb/1aho.pdb", "tests/fixtures/pdb/span-reconstruction/1crn.pdb"),
)
def test_requested_hydrogens_are_prepared_before_eager_parser_repair(
    monkeypatch, source_path
):
    source = read_structure(Path(source_path))
    calls = []
    original_loop = runtime.execute_bounded_parser_witness_repair_loop

    def observed_loop(result, **kwargs):
        _, chemistry = derive_structure_coverage_and_chemistry_readiness_facts(
            result.structure
        )
        calls.append(chemistry.hydrogen_coverage_state)
        assert chemistry.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
        return original_loop(result, **kwargs)

    monkeypatch.setattr(
        runtime, "execute_bounded_parser_witness_repair_loop", observed_loop
    )
    result = process_structure(
        source,
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )

    assert result.requested_goals_fully_satisfied()
    assert not result.has_errors()
    assert calls == []  # Clean terminal states need no unplanned optimizer loop.


@pytest.mark.parametrize(
    "next_family",
    ("hydrogen", "terminal", "packing", "stereo", "topology", "none", "local"),
)
@pytest.mark.parametrize("native_returned", (False, True))
def test_attached_repair_follows_current_proposal_family_and_history(
    monkeypatch, next_family, native_returned
):
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    residue_id = source.constitution.chains[0].residues[0].residue_id
    scope = ResidueSetScope((residue_id,))
    primary = TerminalAugmentationTransformer(scope=scope)
    cysteines = tuple(
        residue.residue_id
        for residue in source.constitution.chains[0].residues
        if residue.component_id == "CYS"
    )
    local = LocalRefinementTransformer.from_repair_refinement(
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues((residue_id,)),
            binding=RecommendedContinuousRelaxationBinding(),
        )
    )
    options = {
        "hydrogen": HydrogenCompletionTransformer(scope=scope),
        "terminal": TerminalAugmentationTransformer(
            scope=ResidueSetScope(
                (source.constitution.chains[0].residues[-1].residue_id,)
            )
        ),
        "packing": CommittedPackingTransformer.from_planned_spec(
            PackingSpec(
                backend_name="faspr",
                scope=PackingScope.LOCAL,
                mutable_residue_ids=(residue_id,),
            )
        ),
        "stereo": StereochemistryCorrectionTransformer(scope=scope),
        "topology": DisulfideTopologyResolutionTransformer(
            candidates=(LikelyDisulfideBond(cysteines[0], cysteines[1], 2.0),)
        ),
        "none": None,
        "local": local,
    }
    following = options[next_family]
    receipt = ContinuousRelaxationExecution(
        settings=ContinuousRelaxationSettings(
            profile=ContinuousRelaxationProfile.RDKIT_MMFF
        ),
        movable_scope=AtomSetScope((AtomRef(residue_id, "CA"),)),
        minimizer_status=1,
        backend_version="2026.03.2",
    )
    executions = (receipt,) if native_returned else ()
    signature = StructurePlanningSignature.from_facts(
        StructureProjectionStateFacts.from_structure(source)
    )
    planner_memories = []
    actions = []
    loops = []

    def plan(structure, *, planner_memory, **kwargs):
        del structure, kwargs
        planner_memories.append(planner_memory)
        if not planner_memory.has_adopted_transformer(primary):
            candidates = (primary,)
        elif following is not None and not planner_memory.has_adopted_transformer(
            following
        ):
            candidates = (following,)
        else:
            candidates = ()
        return WorkflowPlanningOutcome(
            structure_planning_signature=signature,
            proposals=workflow_proposals(candidates),
        )

    def execute(result, *, transformer, execution_context):
        del execution_context
        actions.append(transformer)
        return RefinementTransformationResult.with_observed_executions(
            result, executions
        )

    def parser_loop(result, *, execution_context, execute_transformer):
        del execution_context, execute_transformer
        loops.append(result)
        return DirectParserWitnessRepairResult(
            result=result,
            attempted_transformers=(local,),
            adopted_transformers=(local,),
        )

    monkeypatch.setattr(runtime, "plan_workflow_actions", plan)
    monkeypatch.setattr(runtime, "execute_workflow_transformer", execute)
    monkeypatch.setattr(
        runtime, "execute_bounded_parser_witness_repair_loop", parser_loop
    )
    result = runtime.execute_iterative_workflow(
        source,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        component_library=build_default_component_library(),
        planning_context=WorkflowPlanningContext(),
        reference_structure=None,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
        histidine_protonation=WorkflowTransformRequests().histidine_protonation_request(),
    )

    assert len(loops) == (1 if next_family == "local" else 0)
    expected_actions = [primary]
    if following is not None and next_family != "local":
        expected_actions.append(following)
    assert actions == expected_actions
    assert result.branches[0].outcome.relaxation_executions == executions
    assert planner_memories[1].has_adopted_transformer(primary)
    assert not planner_memories[1].has_adopted_transformer(local)
    assert planner_memories[1].relaxation_executions == executions
