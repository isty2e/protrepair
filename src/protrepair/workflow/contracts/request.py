"""Workflow-boundary request contracts for the redesigned ProtRepair package."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypeVar, overload

from protrepair.scope import (
    ResidueBoundaryScope,
    ResidueBoundarySide,
    Scope,
    WholeStructureScope,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    ClashPresenceState,
    ClashState,
    HydrogenCoverageState,
    OxtPresenceState,
    ScopedState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.transformer.packing.spec import PackingSpec
from protrepair.transformer.refinement.spec import (
    BackboneWindowRefinementSpec,
    RepairRefinementSpec,
)
from protrepair.workflow.contracts.external_reference import (
    ExternalSpanReconstructionSpec,
)
from protrepair.workflow.contracts.policies import (
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
    OrphanFragmentPolicy,
)

if TYPE_CHECKING:
    from protrepair.io.ingress_policy import StructureNormalizationPolicy

WorkflowGoalStateValue: TypeAlias = (
    ClashState
    | ClashPresenceState
    | BackboneHeavyAtomCompletenessState
    | SidechainHeavyAtomCompletenessState
    | HydrogenCoverageState
    | OxtPresenceState
)
WorkflowGoalStateValueT = TypeVar(
    "WorkflowGoalStateValueT",
    bound=WorkflowGoalStateValue,
)
WorkflowGoal: TypeAlias = ScopedState[WorkflowGoalStateValue]


def requested_process_goal(
    *,
    scope: Scope,
    value: WorkflowGoalStateValue,
) -> WorkflowGoal:
    """Return one canonical requested goal for the workflow boundary."""

    return ScopedState(scope=scope, value=value)


@dataclass(frozen=True, slots=True)
class RequestedGoalSet(Sequence[WorkflowGoal]):
    """Normalized requested-goal set over scoped state propositions."""

    goals: tuple[WorkflowGoal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", _normalize_requested_goals(self.goals))

    def __iter__(self) -> Iterator[WorkflowGoal]:
        """Iterate over normalized requested goals."""

        return iter(self.goals)

    def __len__(self) -> int:
        """Return the number of normalized requested goals."""

        return len(self.goals)

    @overload
    def __getitem__(
        self,
        index: int,
    ) -> WorkflowGoal: ...

    @overload
    def __getitem__(
        self,
        index: slice,
    ) -> tuple[WorkflowGoal, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> (
        WorkflowGoal
        | tuple[WorkflowGoal, ...]
    ):
        """Return one requested goal or slice of requested goals."""

        return self.goals[index]

    def goal_state_value(
        self,
        state_type: type[WorkflowGoalStateValueT],
        *,
        scope: Scope,
    ) -> WorkflowGoalStateValueT | None:
        """Return the targeted state value for one exact scope/type pair."""

        for goal in self.goals:
            if goal.scope != scope:
                continue
            if isinstance(goal.value, state_type):
                return goal.value

        return None

    def is_empty(self) -> bool:
        """Return whether the normalized requested-goal set is empty."""

        return not self.goals

    def whole_structure_goal_state_value(
        self,
        state_type: type[WorkflowGoalStateValueT],
    ) -> WorkflowGoalStateValueT | None:
        """Return one whole-structure goal value for a specific state axis."""

        return self.goal_state_value(
            state_type,
            scope=WholeStructureScope(),
        )

    def requests_whole_structure_heavy_atom_completion(self) -> bool:
        """Return whether the set requests global heavy-atom completeness."""

        return (
            self.requests_whole_structure_backbone_heavy_atom_completion()
            or self.requests_whole_structure_sidechain_heavy_atom_completion()
        )

    def requests_whole_structure_backbone_heavy_atom_completion(self) -> bool:
        """Return whether the set requests global backbone completeness."""

        requested_state = self.whole_structure_goal_state_value(
            BackboneHeavyAtomCompletenessState,
        )
        return requested_state is BackboneHeavyAtomCompletenessState.COMPLETE

    def requests_whole_structure_sidechain_heavy_atom_completion(self) -> bool:
        """Return whether the set requests global side-chain completeness."""

        requested_state = self.whole_structure_goal_state_value(
            SidechainHeavyAtomCompletenessState,
        )
        return requested_state is SidechainHeavyAtomCompletenessState.COMPLETE

    def requests_whole_structure_hydrogen_population(self) -> bool:
        """Return whether the set requests global hydrogen coverage."""

        requested_state = self.whole_structure_goal_state_value(
            HydrogenCoverageState,
        )
        return requested_state is HydrogenCoverageState.COMPLETE

    def requests_whole_structure_clash_absence(self) -> bool:
        """Return whether the set requests structure-level clash absence."""

        requested_state = self.whole_structure_goal_state_value(
            ClashPresenceState,
        )
        return requested_state is ClashPresenceState.NONE

    def boundary_goals(
        self,
        state_type: type[WorkflowGoalStateValue],
        *,
        side: ResidueBoundarySide | None = None,
    ) -> tuple[WorkflowGoal, ...]:
        """Return requested boundary-scoped goals for one specific state axis."""

        return tuple(
            goal
            for goal in self.goals
            if isinstance(goal.scope, ResidueBoundaryScope)
            and (side is None or goal.scope.side is side)
            and isinstance(goal.value, state_type)
        )


@dataclass(frozen=True, slots=True)
class StructureIngressOptions:
    """Ingress-boundary normalization configuration for source interpretation."""

    occupancy_policy: OccupancyPolicy = OccupancyPolicy.HIGHEST
    mutation_policy: MutationPolicy = MutationPolicy.HIGHEST_OCCUPANCY
    ligand_policy: LigandPolicy = LigandPolicy.DROP
    retained_non_polymer_chemistry_overrides: tuple[
        RetainedNonPolymerChemistryOverride,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.occupancy_policy, OccupancyPolicy):
            raise TypeError("ingress occupancy_policy must be an OccupancyPolicy value")
        if not isinstance(self.mutation_policy, MutationPolicy):
            raise TypeError("ingress mutation_policy must be a MutationPolicy value")
        if not isinstance(self.ligand_policy, LigandPolicy):
            raise TypeError("ingress ligand_policy must be a LigandPolicy value")
        normalized_overrides: list[RetainedNonPolymerChemistryOverride] = []
        seen_override_residue_ids = set()
        for override in self.retained_non_polymer_chemistry_overrides:
            if not isinstance(override, RetainedNonPolymerChemistryOverride):
                raise TypeError(
                    "ingress retained_non_polymer_chemistry_overrides must contain "
                    "RetainedNonPolymerChemistryOverride values"
                )
            if override.residue_id in seen_override_residue_ids:
                raise ValueError(
                    "ingress retained_non_polymer_chemistry_overrides must not "
                    f"repeat {override.residue_id.display_token()}"
                )
            normalized_overrides.append(override)
            seen_override_residue_ids.add(override.residue_id)

        object.__setattr__(
            self,
            "retained_non_polymer_chemistry_overrides",
            tuple(normalized_overrides),
        )

    def structure_normalization_policy(self) -> "StructureNormalizationPolicy":
        """Project this ingress request into canonical structure normalization."""

        from protrepair.io.ingress_policy import (
            LigandHandling,
            StructureNormalizationPolicy,
        )

        ligand_handling = (
            LigandHandling.KEEP
            if self.ligand_policy is LigandPolicy.KEEP
            else LigandHandling.DROP
        )
        return StructureNormalizationPolicy(
            occupancy_policy=self.occupancy_policy,
            mutation_policy=self.mutation_policy,
            ligand_handling=ligand_handling,
        )


@dataclass(frozen=True, slots=True)
class WorkflowTransformRequests:
    """Explicit workflow transform requests not expressible as desired states."""

    orphan_fragment_policy: OrphanFragmentPolicy = OrphanFragmentPolicy.REBUILD
    external_span_reconstructions: tuple[
        ExternalSpanReconstructionSpec,
        ...,
    ] = ()
    reference_sidechain_packing: PackingSpec | None = None
    committed_sidechain_packing: PackingSpec | None = None
    backbone_window_refinements: tuple[BackboneWindowRefinementSpec, ...] = ()
    repair_refinement: RepairRefinementSpec | None = None
    protonate_histidines: bool = False
    allow_retained_non_polymer_rdkit_fallback: bool = True

    def __post_init__(self) -> None:
        external_span_reconstructions_list: list[ExternalSpanReconstructionSpec] = []
        for reconstruction in self.external_span_reconstructions:
            if reconstruction not in external_span_reconstructions_list:
                external_span_reconstructions_list.append(reconstruction)

        external_span_reconstructions = tuple(external_span_reconstructions_list)
        backbone_window_refinements: list[BackboneWindowRefinementSpec] = []
        for refinement in self.backbone_window_refinements:
            if not isinstance(refinement, BackboneWindowRefinementSpec):
                raise TypeError(
                    "backbone_window_refinements must contain "
                    "BackboneWindowRefinementSpec values"
                )
            if refinement not in backbone_window_refinements:
                backbone_window_refinements.append(refinement)

        reference_sidechain_packing = self.reference_sidechain_packing
        committed_sidechain_packing = self.committed_sidechain_packing
        repair_refinement = self.repair_refinement
        for reconstruction in external_span_reconstructions:
            if not isinstance(
                reconstruction,
                ExternalSpanReconstructionSpec,
            ):
                raise TypeError(
                    "external_span_reconstructions must contain "
                    "ExternalSpanReconstructionSpec values"
                )
        if reference_sidechain_packing is not None and not isinstance(
            reference_sidechain_packing, PackingSpec
        ):
            raise TypeError("reference_sidechain_packing must be a PackingSpec or None")
        if committed_sidechain_packing is not None and not isinstance(
            committed_sidechain_packing, PackingSpec
        ):
            raise TypeError("committed_sidechain_packing must be a PackingSpec or None")
        if repair_refinement is not None and not isinstance(
            repair_refinement,
            RepairRefinementSpec,
        ):
            raise TypeError("repair_refinement must be a RepairRefinementSpec or None")
        if not isinstance(self.orphan_fragment_policy, OrphanFragmentPolicy):
            raise TypeError(
                "orphan_fragment_policy must be an OrphanFragmentPolicy value"
            )
        if not isinstance(self.allow_retained_non_polymer_rdkit_fallback, bool):
            raise TypeError(
                "allow_retained_non_polymer_rdkit_fallback must be a bool"
            )
        if (
            reference_sidechain_packing is not None
            and committed_sidechain_packing is not None
        ):
            raise ValueError(
                "reference_sidechain_packing and committed_sidechain_packing "
                "must not both be set"
            )

        object.__setattr__(
            self,
            "external_span_reconstructions",
            external_span_reconstructions,
        )
        object.__setattr__(
            self,
            "backbone_window_refinements",
            tuple(backbone_window_refinements),
        )

    def requests_reference_sidechain_packing(self) -> bool:
        """Return whether workflow-reference side-chain packing was requested."""

        return self.reference_sidechain_packing is not None

    def requests_committed_sidechain_packing(self) -> bool:
        """Return whether committed side-chain packing was requested."""

        return self.committed_sidechain_packing is not None

    def requests_backbone_window_refinement(self) -> bool:
        """Return whether explicit backbone-window refinement was requested."""

        return bool(self.backbone_window_refinements)

    def requests_external_span_reconstruction(self) -> bool:
        """Return whether explicit absent-span reconstruction was requested."""

        return bool(self.external_span_reconstructions)


def _normalize_requested_goals(
    requested_goals: Sequence[WorkflowGoal],
) -> tuple[WorkflowGoal, ...]:
    """Normalize one requested-goal set and reject conflicting duplicates."""

    normalized_goals: list[WorkflowGoal] = []
    goal_values_by_axis: dict[
        tuple[object, type[WorkflowGoalStateValue]],
        WorkflowGoalStateValue,
    ] = {}
    for goal in requested_goals:
        if not isinstance(goal, ScopedState):
            raise TypeError("requested_goals must contain ScopedState values")
        if not _is_workflow_goal_state_value(goal.value):
            raise TypeError(
                "requested_goals contained an unsupported requested-goal value"
            )
        _validate_workflow_goal_scope(goal)

        axis_key = (goal.scope, type(goal.value))
        existing_value = goal_values_by_axis.get(axis_key)
        if existing_value is not None:
            if existing_value != goal.value:
                raise ValueError(
                    "requested_goals must not contain conflicting values for "
                    "the same scope and state axis"
                )
            continue

        normalized_goals.append(goal)
        goal_values_by_axis[axis_key] = goal.value

    return tuple(normalized_goals)


def _is_workflow_goal_state_value(
    value: object,
) -> bool:
    """Return whether one value belongs to the current workflow goal surface."""

    return isinstance(
        value,
        (
            ClashPresenceState,
            ClashState,
            BackboneHeavyAtomCompletenessState,
            HydrogenCoverageState,
            OxtPresenceState,
            SidechainHeavyAtomCompletenessState,
        ),
    )


def _validate_workflow_goal_scope(goal: WorkflowGoal) -> None:
    """Reject requested goal propositions with incompatible scope axes."""

    if isinstance(goal.value, ClashState) and isinstance(
        goal.scope,
        WholeStructureScope,
    ):
        raise ValueError(
            "whole-structure clash goals must use ClashPresenceState"
        )
    if isinstance(goal.value, ClashPresenceState) and not isinstance(
        goal.scope,
        WholeStructureScope,
    ):
        raise ValueError("ClashPresenceState goals require WholeStructureScope")
