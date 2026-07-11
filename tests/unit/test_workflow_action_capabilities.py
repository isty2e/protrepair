"""Workflow action capability contract tests."""

from protrepair.scope import (
    ResidueBoundaryScope,
    ResidueBoundarySide,
    WholeStructureScope,
)
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    HydrogenCoverageState,
    OxtPresenceState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.structure.labels import ResidueId
from protrepair.workflow.actions.backbone_window_refinement import (
    BackboneWindowRefinementTransformer,
)
from protrepair.workflow.actions.disulfide_hydrogen import (
    DisulfideHydrogenNormalizationTransformer,
)
from protrepair.workflow.actions.disulfide_topology import (
    DisulfideTopologyResolutionTransformer,
)
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.actions.retained_non_polymer_hydrogen_completion import (
    RetainedNonPolymerHydrogenCompletionTransformer,
)
from protrepair.workflow.actions.stereochemistry import (
    StereochemistryCorrectionTransformer,
)
from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.contracts.request import requested_process_goal
from protrepair.workflow.planning.action.registry import action_registry_entry_for_type
from protrepair.workflow.planning.capability import (
    WorkflowActionActivationMode,
    WorkflowActionLocality,
    WorkflowCapabilityDeficitFamily,
    WorkflowCapabilityFactFamily,
)
from protrepair.workflow.planning.default_action_registry import (
    EXPLICIT_WORKFLOW_ACTION_REGISTRY,
    REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY,
    WORKFLOW_ACTION_REGISTRY,
)


def test_request_driven_action_registry_exposes_declared_capabilities() -> None:
    """Request-driven bootstrap entries should carry declared capabilities."""

    action_types = tuple(
        registry_entry.action_type
        for registry_entry in REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY
    )

    assert action_types == (
        HeavyAtomCompletionTransformer,
        DisulfideTopologyResolutionTransformer,
        DisulfideHydrogenNormalizationTransformer,
        HydrogenCompletionTransformer,
        RetainedNonPolymerHydrogenCompletionTransformer,
        TerminalAugmentationTransformer,
    )
    assert all(
        registry_entry.capability.activation_mode
        is not WorkflowActionActivationMode.EXPLICIT_ONLY
        for registry_entry in REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY
    )


def test_heavy_completion_capability_supports_whole_structure_heavy_goals() -> None:
    """Heavy completion should advertise atom-coverage reduction and goal support."""

    capability = _registry_entry(HeavyAtomCompletionTransformer).capability

    assert capability.can_reduce_deficit_family(
        WorkflowCapabilityDeficitFamily.ATOM_COVERAGE
    )
    assert capability.can_be_blocked_by_deficit_family(
        WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS
    )
    assert capability.reads_fact_family(WorkflowCapabilityFactFamily.COVERAGE)
    assert capability.reads_fact_family(
        WorkflowCapabilityFactFamily.CHEMISTRY_READINESS
    )
    assert capability.supports_locality(WorkflowActionLocality.RESIDUE_SET)
    assert _registry_entry(HeavyAtomCompletionTransformer).supports_goal_value(
        requested_process_goal(
            scope=WholeStructureScope(),
            value=BackboneHeavyAtomCompletenessState.COMPLETE,
        )
    )
    assert _registry_entry(HeavyAtomCompletionTransformer).supports_goal_value(
        requested_process_goal(
            scope=WholeStructureScope(),
            value=SidechainHeavyAtomCompletenessState.COMPLETE,
        )
    )


def test_terminal_augmentation_capability_keeps_boundary_goal_distinct() -> None:
    """Terminal augmentation should support boundary goals.

    The capability should not pretend to reduce a canonical deficit family.
    """

    capability = _registry_entry(TerminalAugmentationTransformer).capability
    residue_id = ResidueId("A", 10)

    c_terminal_goal = requested_process_goal(
        scope=ResidueBoundaryScope(
            residue_id=residue_id,
            side=ResidueBoundarySide.C_TERMINUS,
        ),
        value=OxtPresenceState.PRESENT,
    )
    n_terminal_goal = requested_process_goal(
        scope=ResidueBoundaryScope(
            residue_id=residue_id,
            side=ResidueBoundarySide.N_TERMINUS,
        ),
        value=OxtPresenceState.PRESENT,
    )

    assert capability.reducible_deficit_families == ()
    assert capability.reads_fact_family(WorkflowCapabilityFactFamily.BOUNDARY)
    assert capability.supports_proposition(
        scope=c_terminal_goal.scope,
        value=c_terminal_goal.value,
    )
    assert not capability.supports_proposition(
        scope=n_terminal_goal.scope,
        value=n_terminal_goal.value,
    )


def test_explicit_action_registry_entries_expose_capability_metadata() -> None:
    """Explicit workflow action entries should advertise capability metadata."""

    registry_by_type = {
        registry_entry.action_type: registry_entry
        for registry_entry in EXPLICIT_WORKFLOW_ACTION_REGISTRY
    }

    assert set(registry_by_type) == {
        ExternalSpanReconstructionTransformer,
        CommittedPackingTransformer,
        StereochemistryCorrectionTransformer,
        BackboneWindowRefinementTransformer,
        LocalRefinementTransformer,
    }
    assert registry_by_type[
        ExternalSpanReconstructionTransformer
    ].capability.can_reduce_deficit_family(WorkflowCapabilityDeficitFamily.COVERAGE_GAP)
    assert registry_by_type[
        CommittedPackingTransformer
    ].capability.can_reduce_deficit_family(
        WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY
    )
    assert not registry_by_type[
        CommittedPackingTransformer
    ].capability.can_reduce_deficit_family(WorkflowCapabilityDeficitFamily.INTERACTION)
    assert not registry_by_type[
        CommittedPackingTransformer
    ].capability.reads_fact_family(WorkflowCapabilityFactFamily.INTERACTION)
    assert registry_by_type[
        LocalRefinementTransformer
    ].capability.can_reduce_deficit_family(WorkflowCapabilityDeficitFamily.INTERACTION)
    assert registry_by_type[
        LocalRefinementTransformer
    ].capability.can_reduce_deficit_family(
        WorkflowCapabilityDeficitFamily.PARSER_COMPATIBILITY
    )
    assert registry_by_type[LocalRefinementTransformer].capability.reads_fact_family(
        WorkflowCapabilityFactFamily.PARSER_COMPATIBILITY
    )
    assert (
        registry_by_type[
            ExternalSpanReconstructionTransformer
        ].capability.activation_mode
        is WorkflowActionActivationMode.EXPLICIT_ONLY
    )
    assert (
        registry_by_type[CommittedPackingTransformer].capability.activation_mode
        is WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN
    )
    assert registry_by_type[CommittedPackingTransformer].capability.supports_locality(
        WorkflowActionLocality.WHOLE_STRUCTURE
    )
    assert registry_by_type[LocalRefinementTransformer].capability.supports_locality(
        WorkflowActionLocality.LOCAL_SCOPE
    )
    assert registry_by_type[
        StereochemistryCorrectionTransformer
    ].capability.can_reduce_deficit_family(
        WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY
    )
    assert registry_by_type[
        StereochemistryCorrectionTransformer
    ].capability.reads_fact_family(WorkflowCapabilityFactFamily.INTRINSIC_GEOMETRY)
    assert registry_by_type[
        StereochemistryCorrectionTransformer
    ].capability.supports_locality(WorkflowActionLocality.RESIDUE_SET)
    assert (
        registry_by_type[BackboneWindowRefinementTransformer].capability.activation_mode
        is WorkflowActionActivationMode.EXPLICIT_ONLY
    )
    assert registry_by_type[
        BackboneWindowRefinementTransformer
    ].capability.can_reduce_deficit_family(
        WorkflowCapabilityDeficitFamily.BACKBONE_WINDOW_OPERATOR
    )
    assert registry_by_type[
        BackboneWindowRefinementTransformer
    ].capability.reads_fact_family(WorkflowCapabilityFactFamily.INTRINSIC_GEOMETRY)
    assert registry_by_type[
        BackboneWindowRefinementTransformer
    ].capability.supports_locality(WorkflowActionLocality.RESIDUE_SPAN)


def test_workflow_action_registry_covers_every_bootstrap_family() -> None:
    """Combined bootstrap registry should cover every workflow action family once."""

    assert len(WORKFLOW_ACTION_REGISTRY) == 11
    assert len({entry.action_type for entry in WORKFLOW_ACTION_REGISTRY}) == 11
    topology_capability = _registry_entry(
        DisulfideTopologyResolutionTransformer
    ).capability
    assert topology_capability.can_reduce_deficit_family(
        WorkflowCapabilityDeficitFamily.TOPOLOGY_RESOLUTION
    )
    assert topology_capability.reads_fact_family(
        WorkflowCapabilityFactFamily.TOPOLOGY_EVIDENCE
    )
    disulfide_hydrogen_capability = _registry_entry(
        DisulfideHydrogenNormalizationTransformer
    ).capability
    assert disulfide_hydrogen_capability.can_reduce_deficit_family(
        WorkflowCapabilityDeficitFamily.CHEMISTRY_CONTRADICTION
    )
    assert disulfide_hydrogen_capability.reads_fact_family(
        WorkflowCapabilityFactFamily.CHEMISTRY_CONTRADICTION
    )
    assert _registry_entry(HydrogenCompletionTransformer).supports_goal_value(
        requested_process_goal(
            scope=WholeStructureScope(),
            value=HydrogenCoverageState.COMPLETE,
        )
    )
    assert _registry_entry(
        RetainedNonPolymerHydrogenCompletionTransformer
    ).supports_goal_value(
        requested_process_goal(
            scope=WholeStructureScope(),
            value=HydrogenCoverageState.COMPLETE,
        )
    )


def _registry_entry(action_type):
    entry = action_registry_entry_for_type(
        action_type,
        registry=WORKFLOW_ACTION_REGISTRY,
    )
    assert entry is not None
    return entry
