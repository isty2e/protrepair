"""Test-support builders for canonical workflow request values."""

from protrepair.scope import (
    ResidueBoundaryScope,
    ResidueBoundarySide,
    WholeStructureScope,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    OxtPresenceState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.structure.labels import ResidueId
from protrepair.transformer.packing import PackingSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
    OrphanFragmentPolicy,
    StructureIngressOptions,
    WorkflowGoal,
    WorkflowGoalStateValue,
    WorkflowTransformRequests,
    requested_process_goal,
)


def whole_structure_requested_goals(
    *values: WorkflowGoalStateValue,
    include_default_heavy_completion: bool = True,
) -> tuple[WorkflowGoal, ...]:
    """Return canonical whole-structure requested goals for tests."""

    requested_values: list[WorkflowGoalStateValue] = []
    if include_default_heavy_completion:
        requested_values.extend(
            (
                BackboneHeavyAtomCompletenessState.COMPLETE,
                SidechainHeavyAtomCompletenessState.COMPLETE,
            )
        )

    for value in values:
        if isinstance(value, OxtPresenceState):
            raise TypeError(
                "whole_structure_requested_goals does not accept residue-boundary "
                "OXT requests"
            )
        requested_values.append(value)
    return tuple(
        requested_process_goal(scope=WholeStructureScope(), value=value)
        for value in requested_values
    )


def c_terminal_oxt_requested_goals(
    *residue_ids: ResidueId,
) -> tuple[WorkflowGoal, ...]:
    """Return canonical residue-boundary OXT goals for tests."""

    return tuple(
        requested_process_goal(
            scope=ResidueBoundaryScope(
                residue_id=residue_id,
                side=ResidueBoundarySide.C_TERMINUS,
            ),
            value=OxtPresenceState.PRESENT,
        )
        for residue_id in residue_ids
    )


def ingress_options(
    *,
    occupancy_policy: OccupancyPolicy = OccupancyPolicy.HIGHEST,
    mutation_policy: MutationPolicy = MutationPolicy.HIGHEST_OCCUPANCY,
    ligand_policy: LigandPolicy = LigandPolicy.DROP,
    retained_non_polymer_chemistry_overrides: tuple[
        RetainedNonPolymerChemistryOverride,
        ...,
    ] = (),
) -> StructureIngressOptions:
    """Return one ingress-options value for tests."""

    return StructureIngressOptions(
        occupancy_policy=occupancy_policy,
        mutation_policy=mutation_policy,
        ligand_policy=ligand_policy,
        retained_non_polymer_chemistry_overrides=(
            retained_non_polymer_chemistry_overrides
        ),
    )


def transform_requests(
    *,
    orphan_fragment_policy: OrphanFragmentPolicy | None = None,
    external_span_reconstructions: tuple[
        ExternalSpanReconstructionSpec,
        ...,
    ] = (),
    reference_sidechain_packing: PackingSpec | None = None,
    committed_sidechain_packing: PackingSpec | None = None,
    repair_refinement: RepairRefinementSpec | None = None,
    protonate_histidines: bool = False,
) -> WorkflowTransformRequests:
    """Return one transform-requests value for tests."""

    if orphan_fragment_policy is None:
        return WorkflowTransformRequests(
            external_span_reconstructions=external_span_reconstructions,
            reference_sidechain_packing=reference_sidechain_packing,
            committed_sidechain_packing=committed_sidechain_packing,
            repair_refinement=repair_refinement,
            protonate_histidines=protonate_histidines,
        )

    return WorkflowTransformRequests(
        orphan_fragment_policy=orphan_fragment_policy,
        external_span_reconstructions=external_span_reconstructions,
        reference_sidechain_packing=reference_sidechain_packing,
        committed_sidechain_packing=committed_sidechain_packing,
        repair_refinement=repair_refinement,
        protonate_histidines=protonate_histidines,
    )
