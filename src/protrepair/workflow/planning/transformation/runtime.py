"""Planner memory and signatures for transformation workflows."""

from dataclasses import dataclass
from enum import Enum

from protrepair.scope import Scope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    StructureBoundaryStateFacts,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureProjectionStateFacts,
    TerminalBoundaryObservation,
)
from protrepair.state.domain import (
    ClashState,
    SelectedAtomScopeFacts,
    TopologyAvailabilityObservation,
)


class LocalTransformationFamily(str, Enum):
    """Closed planner identity for selected-region transformation families."""

    DISCRETE_PRE_REFINEMENT_CORRECTION = "discrete_pre_refinement_correction"
    BRANCHED_SIDECHAIN_SEED = "branched_sidechain_seed"
    CONTINUOUS_LOCAL_RELAXATION = "continuous_local_relaxation"


@dataclass(frozen=True, slots=True)
class StructurePlanningSignature:
    """Planner-relevant whole-structure planning signature."""

    component_support_state: ComponentSupportState
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState
    hydrogen_applicability_state: HydrogenApplicabilityState
    hydrogen_coverage_state: HydrogenCoverageState
    terminal_boundary_observation: TerminalBoundaryObservation

    @classmethod
    def from_facts(
        cls,
        structure_facts: StructureProjectionStateFacts,
    ) -> "StructurePlanningSignature":
        """Derive one structure planning signature from primitive structure facts."""

        return cls(
            component_support_state=structure_facts.component_support_fact.value,
            backbone_heavy_atom_completeness_state=(
                structure_facts.backbone_heavy_atom_completeness_fact.value
            ),
            sidechain_heavy_atom_completeness_state=(
                structure_facts.sidechain_heavy_atom_completeness_fact.value
            ),
            hydrogen_applicability_state=(
                structure_facts.hydrogen_applicability_fact.value
            ),
            hydrogen_coverage_state=(structure_facts.hydrogen_coverage_fact.value),
            terminal_boundary_observation=(
                structure_facts.terminal_boundary_observation
            ),
        )

    @classmethod
    def from_coverage_chemistry_and_boundary(
        cls,
        *,
        coverage_facts: StructureCoverageFacts,
        chemistry_readiness_facts: StructureChemistryReadinessFacts,
        boundary_facts: StructureBoundaryStateFacts,
    ) -> "StructurePlanningSignature":
        """Derive one structure planning signature from split planning facts."""

        return cls(
            component_support_state=(chemistry_readiness_facts.component_support_state),
            backbone_heavy_atom_completeness_state=(
                coverage_facts.backbone_heavy_atom_completeness_state
            ),
            sidechain_heavy_atom_completeness_state=(
                coverage_facts.sidechain_heavy_atom_completeness_state
            ),
            hydrogen_applicability_state=(
                chemistry_readiness_facts.hydrogen_applicability_state
            ),
            hydrogen_coverage_state=(chemistry_readiness_facts.hydrogen_coverage_state),
            terminal_boundary_observation=(
                boundary_facts.terminal_boundary_observation
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectedScopeSignature:
    """Canonical selected-scope identity relevant to planner memory."""

    scope: Scope

    @classmethod
    def from_scope(
        cls,
        scope: Scope,
    ) -> "SelectedScopeSignature":
        """Derive one canonical selected-scope signature."""

        return cls(scope=scope)


@dataclass(frozen=True, slots=True)
class SelectedScopePlanningSignature:
    """Planner-relevant local planning signature over one selected scope."""

    selected_scope: SelectedScopeSignature
    component_support_state: ComponentSupportState
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState
    hydrogen_applicability_state: HydrogenApplicabilityState
    hydrogen_coverage_state: HydrogenCoverageState
    clash_state: ClashState
    topology_availability: TopologyAvailabilityObservation
    terminal_boundary_observation: TerminalBoundaryObservation

    @classmethod
    def from_scope_facts(
        cls,
        selected_scope: Scope,
        selected_scope_facts: SelectedAtomScopeFacts,
    ) -> "SelectedScopePlanningSignature":
        """Derive one local signature from primitive selected-scope facts."""

        return cls(
            selected_scope=SelectedScopeSignature.from_scope(selected_scope),
            component_support_state=(
                selected_scope_facts.structure_facts.component_support_fact.value
            ),
            backbone_heavy_atom_completeness_state=(
                selected_scope_facts.structure_facts.backbone_heavy_atom_completeness_fact.value
            ),
            sidechain_heavy_atom_completeness_state=(
                selected_scope_facts.structure_facts.sidechain_heavy_atom_completeness_fact.value
            ),
            hydrogen_applicability_state=(
                selected_scope_facts.structure_facts.hydrogen_applicability_fact.value
            ),
            hydrogen_coverage_state=(
                selected_scope_facts.structure_facts.hydrogen_coverage_fact.value
            ),
            clash_state=selected_scope_facts.clash_fact.value,
            topology_availability=TopologyAvailabilityObservation.from_facts(
                selected_scope_facts.topology_availability_facts
            ),
            terminal_boundary_observation=(
                selected_scope_facts.structure_facts.terminal_boundary_observation
            ),
        )


@dataclass(frozen=True, slots=True)
class TransformationPlanningSignature:
    """Canonical planner signature for repetition and cycle detection."""

    structure: StructurePlanningSignature
    selected_scope: SelectedScopePlanningSignature

    @classmethod
    def from_state_facts(
        cls,
        *,
        structure_facts: StructureProjectionStateFacts,
        selected_scope: Scope,
        selected_scope_facts: SelectedAtomScopeFacts,
    ) -> "TransformationPlanningSignature":
        """Derive one planner signature from primitive scoped facts."""

        return cls(
            structure=StructurePlanningSignature.from_facts(structure_facts),
            selected_scope=SelectedScopePlanningSignature.from_scope_facts(
                selected_scope,
                selected_scope_facts,
            ),
        )


@dataclass(frozen=True, slots=True)
class TransformationFamilyAttemptSignature:
    """Canonical attempted-family signature under one planner signature."""

    family: LocalTransformationFamily
    planning_signature: TransformationPlanningSignature

    @classmethod
    def from_state_facts(
        cls,
        *,
        family: LocalTransformationFamily,
        structure_facts: StructureProjectionStateFacts,
        selected_scope: Scope,
        selected_scope_facts: SelectedAtomScopeFacts,
    ) -> "TransformationFamilyAttemptSignature":
        """Derive one attempted-family signature from primitive scoped facts."""

        return cls(
            family=family,
            planning_signature=TransformationPlanningSignature.from_state_facts(
                structure_facts=structure_facts,
                selected_scope=selected_scope,
                selected_scope_facts=selected_scope_facts,
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservedPlanningSignatureEntry:
    """Observed-signature count recorded in planner history."""

    signature: TransformationPlanningSignature
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(
                "planning-signature observation counts must be recorded from 1 upward"
            )

    def incremented(self) -> "ObservedPlanningSignatureEntry":
        """Return one additional observation of the same planner signature."""

        return type(self)(
            signature=self.signature,
            count=self.count + 1,
        )


class TransformationProgressKind(str, Enum):
    """Closed progress kinds observed across transformation planning steps."""

    COMPLETENESS = "completeness"
    CHEMISTRY_PREPARATION = "chemistry_preparation"
    CANDIDATE_CONSTRUCTION = "candidate_construction"
    CLASH_REDUCTION = "clash_reduction"
    GEOMETRY_IMPROVEMENT = "geometry_improvement"
    NO_PROGRESS = "no_progress"


class TransformationTerminationReason(str, Enum):
    """Closed termination reasons for one transformation run."""

    STEP_LIMIT_REACHED = "step_limit_reached"
    NO_PROGRESS_LIMIT_REACHED = "no_progress_limit_reached"
    CYCLE_DETECTED = "cycle_detected"
    NO_LEGAL_TRANSFORMATIONS = "no_legal_transformations"
    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class TransformationStepBudget:
    """Step budget memory for one transformation run."""

    step_count: int = 0
    max_steps: int | None = None

    def advanced(self) -> "TransformationStepBudget":
        """Return the budget after one additional executed step."""

        return type(self)(
            step_count=self.step_count + 1,
            max_steps=self.max_steps,
        )

    def is_exhausted(self) -> bool:
        """Return whether the configured step budget is exhausted."""

        return self.max_steps is not None and self.step_count >= self.max_steps


@dataclass(frozen=True, slots=True)
class TransformationHistory:
    """Visited-signature and attempted-transformer history for one run."""

    attempted_family_signatures: tuple[
        TransformationFamilyAttemptSignature,
        ...,
    ] = ()
    observed_signature_entries: tuple[ObservedPlanningSignatureEntry, ...] = ()

    def has_attempted_family(
        self,
        family_signature: TransformationFamilyAttemptSignature,
    ) -> bool:
        """Return whether one attempted-transformer signature already exists."""

        return family_signature in self.attempted_family_signatures

    def with_attempted_family(
        self,
        family_signature: TransformationFamilyAttemptSignature,
    ) -> "TransformationHistory":
        """Return history with one attempted transformer recorded once."""

        if self.has_attempted_family(family_signature):
            return self

        return type(self)(
            attempted_family_signatures=(
                *self.attempted_family_signatures,
                family_signature,
            ),
            observed_signature_entries=self.observed_signature_entries,
        )

    @property
    def observed_planning_signatures(
        self,
    ) -> tuple[TransformationPlanningSignature, ...]:
        """Return observed planning signatures in first-seen order."""

        return tuple(
            observation.signature for observation in self.observed_signature_entries
        )

    def observation_count(
        self,
        planning_signature: TransformationPlanningSignature,
    ) -> int:
        """Return how many times one planning signature has been observed."""

        for observation in self.observed_signature_entries:
            if observation.signature == planning_signature:
                return observation.count

        return 0

    def has_observed_signature(
        self,
        planning_signature: TransformationPlanningSignature,
    ) -> bool:
        """Return whether a planning signature has already been observed."""

        return self.observation_count(planning_signature) > 0

    def with_observed_signature(
        self,
        planning_signature: TransformationPlanningSignature,
    ) -> "TransformationHistory":
        """Return history with one additional observation of a planning signature."""

        updated_observations: list[ObservedPlanningSignatureEntry] = []
        matched_observation = False
        for observation in self.observed_signature_entries:
            if observation.signature == planning_signature:
                updated_observations.append(observation.incremented())
                matched_observation = True
                continue

            updated_observations.append(observation)

        if not matched_observation:
            updated_observations.append(
                ObservedPlanningSignatureEntry(signature=planning_signature)
            )

        return type(self)(
            attempted_family_signatures=self.attempted_family_signatures,
            observed_signature_entries=tuple(updated_observations),
        )


@dataclass(frozen=True, slots=True)
class TransformationProgressMemory:
    """Progress memory summary for one transformation run."""

    no_progress_count: int = 0
    last_progress_kind: TransformationProgressKind | None = None
    best_quality_so_far: float | None = None

    def with_progress(
        self,
        progress_kind: TransformationProgressKind,
        *,
        quality_score: float | None = None,
    ) -> "TransformationProgressMemory":
        """Return progress memory updated for one completed step."""

        next_best_quality = self.best_quality_so_far
        if quality_score is not None:
            if next_best_quality is None or quality_score < next_best_quality:
                next_best_quality = quality_score

        return type(self)(
            no_progress_count=(
                self.no_progress_count + 1
                if progress_kind is TransformationProgressKind.NO_PROGRESS
                else 0
            ),
            last_progress_kind=progress_kind,
            best_quality_so_far=next_best_quality,
        )


@dataclass(frozen=True, slots=True)
class TransformationTerminationMemory:
    """Termination memory for one transformation run."""

    reason: TransformationTerminationReason | None = None

    def is_terminal(self) -> bool:
        """Return whether the run is already terminal."""

        return self.reason is not None

    def terminated(
        self,
        reason: TransformationTerminationReason,
    ) -> "TransformationTerminationMemory":
        """Return termination memory with one terminal reason recorded."""

        return type(self)(reason=reason)


@dataclass(frozen=True, slots=True)
class TransformationPlanningMemory:
    """Planner memory aggregate for one transformation run."""

    budget: TransformationStepBudget = TransformationStepBudget()
    history: TransformationHistory = TransformationHistory()
    progress: TransformationProgressMemory = TransformationProgressMemory()
    termination: TransformationTerminationMemory = TransformationTerminationMemory()

    @classmethod
    def initial(
        cls,
        *,
        max_steps: int | None = None,
    ) -> "TransformationPlanningMemory":
        """Return one canonical initial planning-memory value."""

        return cls(
            budget=TransformationStepBudget(max_steps=max_steps),
        )

    def advanced(self) -> "TransformationPlanningMemory":
        """Return planning memory after one executed planning step."""

        return type(self)(
            budget=self.budget.advanced(),
            history=self.history,
            progress=self.progress,
            termination=self.termination,
        )

    def with_attempted_family(
        self,
        family_signature: TransformationFamilyAttemptSignature,
    ) -> "TransformationPlanningMemory":
        """Return planning memory with one attempted transformer recorded."""

        return type(self)(
            budget=self.budget,
            history=self.history.with_attempted_family(family_signature),
            progress=self.progress,
            termination=self.termination,
        )

    def with_observed_signature(
        self,
        planning_signature: TransformationPlanningSignature,
    ) -> "TransformationPlanningMemory":
        """Return planning memory with one observed planning signature recorded."""

        return type(self)(
            budget=self.budget,
            history=self.history.with_observed_signature(planning_signature),
            progress=self.progress,
            termination=self.termination,
        )

    def with_progress(
        self,
        progress_kind: TransformationProgressKind,
        *,
        quality_score: float | None = None,
    ) -> "TransformationPlanningMemory":
        """Return planning memory with updated progress memory."""

        return type(self)(
            budget=self.budget,
            history=self.history,
            progress=self.progress.with_progress(
                progress_kind,
                quality_score=quality_score,
            ),
            termination=self.termination,
        )

    def terminated(
        self,
        reason: TransformationTerminationReason,
    ) -> "TransformationPlanningMemory":
        """Return planning memory marked terminal with one reason."""

        return type(self)(
            budget=self.budget,
            history=self.history,
            progress=self.progress,
            termination=self.termination.terminated(reason),
        )
