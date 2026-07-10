"""Planner-readable transformer domain admissibility inputs."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.state import (
    ResidueChemistryReadinessFacts,
    ResidueCoverageFacts,
    StereochemistryState,
    StructureBoundaryStateFacts,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
)
from protrepair.state.structure_topology import StructureDisulfideTopologyFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.local.projection import (
    LocalContinuousExecutionResidueProjection,
)
from protrepair.workflow.contracts.planning import WorkflowPlanningContext
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowTransformRequests,
)
from protrepair.workflow.planning.action.registry import WorkflowActionAdoption
from protrepair.workflow.planning.assessment.deficits import (
    WorkflowDeficitDisposition,
    WorkflowStateDeficit,
)
from protrepair.workflow.planning.capability import (
    WorkflowActionEffectClass,
    WorkflowCapabilityDeficitFamily,
)
from protrepair.workflow.planning.context_projection import (
    planning_context_is_holo_for_structure,
)


@dataclass(frozen=True, slots=True)
class WorkflowPlanningMemoryReadView:
    """Planner-memory read view over adopted workflow action families."""

    adopted_actions: tuple[WorkflowActionAdoption, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adopted_actions",
            tuple(self.adopted_actions),
        )

    def has_prior_augmentation_adoption(self) -> bool:
        """Return whether one prior augment-absence action already ran."""

        for adopted_action in self.adopted_actions:
            if (
                adopted_action.capability.effect_class
                is WorkflowActionEffectClass.AUGMENTS_ABSENCE
            ):
                return True

        return False

    def has_reducer_for_deficit_family(
        self,
        deficit_family: WorkflowCapabilityDeficitFamily,
    ) -> bool:
        """Return whether one adopted action can reduce a specific deficit family."""

        for adopted_action in self.adopted_actions:
            if adopted_action.capability.can_reduce_deficit_family(deficit_family):
                return True

        return False

    def has_action_family(
        self,
        action_family: type,
    ) -> bool:
        """Return whether one action family already appears in planner memory."""

        return any(
            adopted_action.action_family is action_family
            for adopted_action in self.adopted_actions
        )


@dataclass(frozen=True, slots=True)
class WorkflowExplicitRepairReadView:
    """Explicit local-refinement repair read view over request projections."""

    transform_requests: WorkflowTransformRequests
    coverage_facts: StructureCoverageFacts
    chemistry_readiness_facts: StructureChemistryReadinessFacts
    intrinsic_geometry_facts: StructureIntrinsicGeometryFacts | None = None
    execution_projection: LocalContinuousExecutionResidueProjection | None = None

    def __post_init__(self) -> None:
        if self.execution_projection is not None and not isinstance(
            self.execution_projection,
            LocalContinuousExecutionResidueProjection,
        ):
            raise TypeError(
                "explicit repair views require execution projection to be a "
                "LocalContinuousExecutionResidueProjection or None"
            )

    def is_requested(self) -> bool:
        """Return whether the workflow carries one explicit refinement request."""

        return self.transform_requests.repair_refinement is not None

    def execution_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return execution-scope residue ids requested by explicit repair."""

        repair_refinement = self.transform_requests.repair_refinement
        if repair_refinement is None:
            return ()

        return (
            repair_refinement.resolved_execution_scope_spec()
            .referenced_residue_ids()
        )

    def prerequisite_residue_ids(
        self,
    ) -> tuple[ResidueId, ...]:
        """Return polymer residues in the actual local FF included region."""

        projection = self.execution_projection
        if projection is None:
            return ()

        return projection.included_polymer_residue_ids

    def atom_completion_residue_ids(
        self,
    ) -> tuple[ResidueId, ...]:
        """Return explicit-repair residues that still need heavy completion."""

        return tuple(
            residue_facts.residue_id
            for residue_facts in self._completion_facts()
            if _coverage_requires_heavy_completion(residue_facts.coverage)
            and residue_facts.chemistry.is_supported()
        )

    def hydrogen_missing_residue_ids(
        self,
    ) -> tuple[ResidueId, ...]:
        """Return explicit-repair residues ready for direct hydrogen completion."""

        return tuple(
            residue_facts.residue_id
            for residue_facts in self._completion_facts()
            if (
                not _coverage_requires_heavy_completion(residue_facts.coverage)
                and residue_facts.chemistry.is_supported()
                and residue_facts.chemistry.needs_hydrogenation()
            )
        )

    def hydrogen_prerequisite_residue_ids(
        self,
    ) -> tuple[ResidueId, ...]:
        """Return explicit-repair residues that need heavy completion first."""

        return tuple(
            residue_facts.residue_id
            for residue_facts in self._completion_facts()
            if (
                _coverage_requires_heavy_completion(residue_facts.coverage)
                and residue_facts.chemistry.is_supported()
                and residue_facts.chemistry.needs_hydrogenation()
            )
        )

    def _completion_facts(
        self,
    ) -> tuple["_ExplicitRepairRefinementCompletionFacts", ...]:
        """Return completion facts projected onto the explicit repair scope."""

        prerequisite_residue_ids = self.prerequisite_residue_ids()
        if not prerequisite_residue_ids:
            return ()
        if (
            self.intrinsic_geometry_facts is not None
            and self.intrinsic_geometry_facts.stereochemistry_state
            is StereochemistryState.VIOLATED
        ):
            return ()

        coverage_by_residue_id = {
            residue_fact.residue_id: residue_fact
            for residue_fact in self.coverage_facts.residue_facts
        }
        chemistry_by_residue_id = {
            residue_fact.residue_id: residue_fact
            for residue_fact in self.chemistry_readiness_facts.residue_facts
        }

        completion_facts: list[_ExplicitRepairRefinementCompletionFacts] = []
        for residue_id in prerequisite_residue_ids:
            coverage_facts = coverage_by_residue_id.get(residue_id)
            chemistry_facts = chemistry_by_residue_id.get(residue_id)
            if coverage_facts is None or chemistry_facts is None:
                continue

            completion_facts.append(
                _ExplicitRepairRefinementCompletionFacts(
                    residue_id=residue_id,
                    coverage=coverage_facts,
                    chemistry=chemistry_facts,
                )
            )

        return tuple(completion_facts)


@dataclass(frozen=True, slots=True)
class WorkflowCompletionReadView:
    """Completion-family read view over coverage and chemistry deficits."""

    requested_goals: RequestedGoalSet
    state_deficit: WorkflowStateDeficit
    chemistry_readiness_facts: StructureChemistryReadinessFacts
    explicit_repair: WorkflowExplicitRepairReadView
    memory: WorkflowPlanningMemoryReadView

    def requires_atom_completion(self) -> bool:
        """Return whether unblocked required atom coverage remains."""

        return any(
            atom_deficit.disposition is WorkflowDeficitDisposition.REQUIRED
            and not atom_deficit.blocked_by_component_support
            for atom_deficit in self.state_deficit.coverage.atom_deficits
        ) or bool(self.explicit_repair.atom_completion_residue_ids())

    def requires_hydrogen_completion(self) -> bool:
        """Return whether hydrogen augmentation is currently admissible."""

        chemistry_deficit = self.state_deficit.chemistry_readiness
        if (
            chemistry_deficit.disposition is WorkflowDeficitDisposition.REQUIRED
            and not chemistry_deficit.hydrogen_blocked_residue_ids
            and (
                chemistry_deficit.hydrogen_missing_residue_ids
                or (
                    chemistry_deficit.hydrogen_prerequisite_residue_ids
                    and self.memory.has_reducer_for_deficit_family(
                        WorkflowCapabilityDeficitFamily.ATOM_COVERAGE
                    )
                )
            )
        ):
            return True

        if self.explicit_repair.hydrogen_missing_residue_ids():
            return True

        if self.explicit_repair.hydrogen_prerequisite_residue_ids():
            return self.memory.has_reducer_for_deficit_family(
                WorkflowCapabilityDeficitFamily.ATOM_COVERAGE
            )

        return False

    def requires_retained_non_polymer_hydrogen_completion(self) -> bool:
        """Return whether retained non-polymer hydrogen completion is admissible."""

        if not self.requested_goals.requests_whole_structure_hydrogen_population():
            return False

        return any(
            retained_fact.requires_hydrogen_completion()
            for retained_fact in (
                self.chemistry_readiness_facts.retained_non_polymer_facts
            )
        )


@dataclass(frozen=True, slots=True)
class WorkflowBurdenReadView:
    """Read view over active intrinsic/parser/interaction planning burdens."""

    structure: ProteinStructure
    planning_context: WorkflowPlanningContext
    state_deficit: WorkflowStateDeficit

    def has_intrinsic_geometry_burden(self) -> bool:
        """Return whether intrinsic geometry burden is present."""

        intrinsic_geometry = self.state_deficit.intrinsic_geometry
        return intrinsic_geometry is not None and intrinsic_geometry.has_burden()

    def has_parser_compatibility_burden(self) -> bool:
        """Return whether parser-profile compatibility burden is present."""

        parser_compatibility = self.state_deficit.parser_compatibility
        return (
            parser_compatibility is not None
            and parser_compatibility.has_burden()
        )

    def has_interaction_burden(self) -> bool:
        """Return whether ligand-aware interaction burden is present."""

        interaction = self.state_deficit.interaction
        return interaction is not None and interaction.has_burden()

    def is_holo_context(self) -> bool:
        """Return whether the active planning context is holo for this structure."""

        return planning_context_is_holo_for_structure(
            self.planning_context,
            self.structure,
        )


@dataclass(frozen=True, slots=True)
class WorkflowSpanReconstructionReadView:
    """Read view over explicit span reconstruction request availability."""

    planning_context: WorkflowPlanningContext
    transform_requests: WorkflowTransformRequests

    def allows_reconstruction(self) -> bool:
        """Return whether donor-backed span reconstruction is admissible."""

        return (
            self.planning_context.allows_span_reconstruction()
            and bool(self.transform_requests.external_span_reconstructions)
        )


@dataclass(frozen=True, slots=True)
class WorkflowActionDomain:
    """Planner-readable domain input for one workflow action family."""

    structure: ProteinStructure
    requested_goals: RequestedGoalSet
    transform_requests: WorkflowTransformRequests
    component_library: ComponentLibrary
    planning_context: WorkflowPlanningContext
    state_deficit: WorkflowStateDeficit
    coverage_facts: StructureCoverageFacts
    chemistry_readiness_facts: StructureChemistryReadinessFacts
    boundary_facts: StructureBoundaryStateFacts
    disulfide_topology_facts: StructureDisulfideTopologyFacts
    intrinsic_geometry_facts: StructureIntrinsicGeometryFacts | None = None
    parser_compatibility_facts: StructureParserCompatibilityFacts | None = None
    interaction_facts: StructureInteractionFacts | None = None
    explicit_repair_refinement_execution_projection: (
        LocalContinuousExecutionResidueProjection | None
    ) = None
    adopted_actions: tuple[WorkflowActionAdoption, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.explicit_repair_refinement_execution_projection is not None
            and not isinstance(
                self.explicit_repair_refinement_execution_projection,
                LocalContinuousExecutionResidueProjection,
            )
        ):
            raise TypeError(
                "workflow action domains require explicit repair refinement "
                "execution projection to be a "
                "LocalContinuousExecutionResidueProjection or None"
            )
        if self.disulfide_topology_facts.carrier is not self.structure:
            raise ValueError(
                "workflow action domains require disulfide topology facts for "
                "the active structure"
            )
        object.__setattr__(
            self,
            "adopted_actions",
            tuple(self.adopted_actions),
        )

    @property
    def memory(self) -> WorkflowPlanningMemoryReadView:
        """Return the planner-memory read view."""

        return WorkflowPlanningMemoryReadView(adopted_actions=self.adopted_actions)

    @property
    def explicit_repair(self) -> WorkflowExplicitRepairReadView:
        """Return the explicit repair read view."""

        return WorkflowExplicitRepairReadView(
            transform_requests=self.transform_requests,
            coverage_facts=self.coverage_facts,
            chemistry_readiness_facts=self.chemistry_readiness_facts,
            intrinsic_geometry_facts=self.intrinsic_geometry_facts,
            execution_projection=self.explicit_repair_refinement_execution_projection,
        )

    @property
    def completion(self) -> WorkflowCompletionReadView:
        """Return the completion-family read view."""

        return WorkflowCompletionReadView(
            requested_goals=self.requested_goals,
            state_deficit=self.state_deficit,
            chemistry_readiness_facts=self.chemistry_readiness_facts,
            explicit_repair=self.explicit_repair,
            memory=self.memory,
        )

    @property
    def burden(self) -> WorkflowBurdenReadView:
        """Return the active burden/context read view."""

        return WorkflowBurdenReadView(
            structure=self.structure,
            planning_context=self.planning_context,
            state_deficit=self.state_deficit,
        )

    @property
    def span_reconstruction(self) -> WorkflowSpanReconstructionReadView:
        """Return the donor-backed span reconstruction read view."""

        return WorkflowSpanReconstructionReadView(
            planning_context=self.planning_context,
            transform_requests=self.transform_requests,
        )


@dataclass(frozen=True, slots=True)
class _ExplicitRepairRefinementCompletionFacts:
    """Coverage/chemistry facts over one explicit-repair target residue."""

    residue_id: ResidueId
    coverage: ResidueCoverageFacts
    chemistry: ResidueChemistryReadinessFacts


def _coverage_requires_heavy_completion(
    coverage_facts: ResidueCoverageFacts,
) -> bool:
    """Return whether one residue still requires heavy-atom completion."""

    return (
        coverage_facts.requires_backbone_completion()
        or coverage_facts.requires_sidechain_completion()
    )
