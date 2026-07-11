"""Unit tests for capability- and deficit-driven workflow goal support."""

from typing import cast

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)
from typing_extensions import assert_type

from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state import (
    ClashPresenceState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.state.structure_topology import (
    StructureDisulfideHydrogenFacts,
    StructureDisulfideTopologyFacts,
)
from protrepair.structure.labels import ResidueId
from protrepair.workflow.contracts import (
    RequestedGoalSet,
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
    requested_process_goal,
)
from protrepair.workflow.planning.assessment.blockers import (
    WorkflowBlocker,
    WorkflowBlockerKind,
)
from protrepair.workflow.planning.assessment.deficits import WorkflowStateDeficit
from protrepair.workflow.planning.assessment.goal_support import (
    already_satisfied_requested_goals,
    blocked_requested_goal_projections,
    unsupported_requested_goals,
)
from protrepair.workflow.planning.capability import WorkflowCapabilityDeficitFamily
from protrepair.workflow.planning.default_action_registry import (
    REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY,
)


def test_unsupported_requested_goals_uses_capability_registry() -> None:
    """Unsupported goals should come from capability matching, not family checks."""

    residue_id = ResidueId("A", 1)
    requested_goal = requested_process_goal(
        scope=ResidueSetScope(residue_ids=(residue_id,)),
        value=HydrogenCoverageState.COMPLETE,
    )

    unsupported_goals = unsupported_requested_goals(
        RequestedGoalSet((requested_goal,)),
        capabilities=tuple(
            registry_entry.capability
            for registry_entry in REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY
        ),
    )

    assert unsupported_goals == (requested_goal,)


def test_blocked_goal_projection_marks_heavy_support_blockers() -> None:
    """Unsupported heavy-atom repair should project chemistry-support blockers."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNK",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-goal-support-heavy-blocked",
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )
    state_deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        disulfide_topology_facts=StructureDisulfideTopologyFacts.from_structure(
            coverage_facts.carrier
        ),
        disulfide_hydrogen_facts=StructureDisulfideHydrogenFacts.from_structure(
            coverage_facts.carrier
        ),
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=SidechainHeavyAtomCompletenessState.COMPLETE,
                ),
            )
        ),
        planning_context=WorkflowPlanningContext(),
    )

    blocked_goal_projections = blocked_requested_goal_projections(
        RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=SidechainHeavyAtomCompletenessState.COMPLETE,
                ),
            )
        ),
        state_deficit=state_deficit,
        planning_context=WorkflowPlanningContext(),
        capabilities=tuple(
            registry_entry.capability
            for registry_entry in REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY
        ),
    )

    assert len(blocked_goal_projections) == 1
    blocker = blocked_goal_projections[0].blockers[0]
    assert blocker.scope == ResidueSetScope(residue_ids=(residue_id,))
    assert (
        blocker.deficit_family
        is WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS
    )


def test_workflow_blocker_scope_contract_is_residue_set_only() -> None:
    """Workflow blockers should not advertise unsupported non-residue scopes."""

    residue_id = ResidueId("A", 1)
    blocker = WorkflowBlocker.unsupported_component(
        deficit_family=WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
        scope=ResidueSetScope(residue_ids=(residue_id,)),
    )

    assert_type(blocker.scope, ResidueSetScope)
    assert blocker.scope == ResidueSetScope(residue_ids=(residue_id,))
    assert blocker.residue_ids() == frozenset({residue_id})

    with pytest.raises(TypeError, match="residue-set scope"):
        WorkflowBlocker(
            phase=WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION,
            deficit_family=WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
            kind=WorkflowBlockerKind.UNSUPPORTED_COMPONENT,
            scope=cast(ResidueSetScope, WholeStructureScope()),
        )


def test_unsupported_component_blocker_requires_single_residue_scope() -> None:
    """Unsupported-component blockers should not defer cardinality errors."""

    residue_a = ResidueId("A", 1)
    residue_b = ResidueId("A", 2)

    with pytest.raises(ValueError, match="exactly one blocked residue"):
        WorkflowBlocker.unsupported_component(
            deficit_family=WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
            scope=ResidueSetScope(residue_ids=(residue_a, residue_b)),
        )


def test_blocked_goal_projection_marks_chemistry_blockers() -> None:
    """Unsupported hydrogen completion should project chemistry-readiness blockers."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNK",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-goal-support-hydrogen-blocked",
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )
    requested_goal = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    state_deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        disulfide_topology_facts=StructureDisulfideTopologyFacts.from_structure(
            coverage_facts.carrier
        ),
        disulfide_hydrogen_facts=StructureDisulfideHydrogenFacts.from_structure(
            coverage_facts.carrier
        ),
        requested_goals=RequestedGoalSet((requested_goal,)),
        planning_context=WorkflowPlanningContext(),
    )

    blocked_goal_projections = blocked_requested_goal_projections(
        RequestedGoalSet((requested_goal,)),
        state_deficit=state_deficit,
        planning_context=WorkflowPlanningContext(),
        capabilities=tuple(
            registry_entry.capability
            for registry_entry in REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY
        ),
    )

    assert len(blocked_goal_projections) == 1
    blocker = blocked_goal_projections[0].blockers[0]
    assert blocker.scope == ResidueSetScope(residue_ids=(residue_id,))
    assert (
        blocker.deficit_family
        is WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS
    )


def test_clash_goal_stays_unsupported_without_matching_capability() -> None:
    """Structure clash goals stay unsupported until a capability declares them."""

    requested_goal = requested_process_goal(
        scope=WholeStructureScope(),
        value=ClashPresenceState.NONE,
    )

    unsupported_goals = unsupported_requested_goals(
        RequestedGoalSet((requested_goal,)),
        capabilities=tuple(
            registry_entry.capability
            for registry_entry in REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY
        ),
    )

    assert unsupported_goals == (requested_goal,)


def test_whole_structure_hydrogen_goal_counts_retained_non_polymer_hydrogens() -> None:
    """Whole-structure hydrogen satisfaction should include retained non-polymers."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-goal-support-retained-non-polymer-hydrogen",
    )
    requested_goal = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )

    assert already_satisfied_requested_goals(
        structure,
        requested_goals=(requested_goal,),
        component_library=build_retained_non_polymer_component_library(),
    ) == ()
