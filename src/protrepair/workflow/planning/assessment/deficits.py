"""Planner-facing deficit IR derived from facts, goals, and context."""

from dataclasses import dataclass
from enum import Enum

from protrepair.relation.blueprint import StructureBlueprintCoverageGap
from protrepair.state import (
    ClashObservationMode,
    HydrogenApplicabilityState,
    OrientationCorrectionEligibilityState,
    ParserCompatibilityProfile,
    ParserCompatibilityState,
    StereochemistryState,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
    TopologyAvailabilityState,
)
from protrepair.state.structure_topology import (
    StructureDisulfideHydrogenFacts,
    StructureDisulfideTopologyFacts,
)
from protrepair.structure.labels import ResidueId
from protrepair.transformer.refinement.spec import BackboneWindowRefinementSpec
from protrepair.workflow.contracts.planning import (
    WorkflowPlanningContext,
    WorkflowTargetIntent,
)
from protrepair.workflow.contracts.request import RequestedGoalSet


class WorkflowDeficitDisposition(str, Enum):
    """Planner-facing urgency for one unresolved workflow burden."""

    OPTIONAL = "optional"
    REQUIRED = "required"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class CoverageGapDeficit:
    """One unresolved missing-residue span over blueprint coverage."""

    gap: StructureBlueprintCoverageGap
    disposition: WorkflowDeficitDisposition


@dataclass(frozen=True, slots=True)
class AtomCoverageDeficit:
    """One residue-local heavy-atom absence burden."""

    residue_id: ResidueId
    missing_backbone_atoms: bool = False
    missing_sidechain_atoms: bool = False
    blocked_by_component_support: bool = False
    disposition: WorkflowDeficitDisposition = WorkflowDeficitDisposition.OPTIONAL

    def __post_init__(self) -> None:
        if not self.missing_backbone_atoms and not self.missing_sidechain_atoms:
            raise ValueError(
                "atom coverage deficits require missing backbone or side-chain atoms"
            )


@dataclass(frozen=True, slots=True)
class StructureCoverageDeficit:
    """Coverage burdens separated into span and atom-completion deficits."""

    gap_deficits: tuple[CoverageGapDeficit, ...] = ()
    atom_deficits: tuple[AtomCoverageDeficit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gap_deficits", tuple(self.gap_deficits))
        object.__setattr__(self, "atom_deficits", tuple(self.atom_deficits))

    def has_burden(self) -> bool:
        """Return whether any coverage burden remains unresolved."""

        return bool(self.gap_deficits or self.atom_deficits)


@dataclass(frozen=True, slots=True)
class StructureChemistryReadinessDeficit:
    """Chemistry-readiness burdens over support, topology, and hydrogens.

    Topology ``absent`` fields mean expected topology is known but missing;
    ``unsupported`` fields mean the expected topology cannot be resolved from
    the currently supported chemistry basis.
    """

    unsupported_component_residue_ids: tuple[ResidueId, ...] = ()
    heavy_atom_topology_absent_residue_ids: tuple[ResidueId, ...] = ()
    heavy_atom_topology_unsupported_residue_ids: tuple[ResidueId, ...] = ()
    hydrogen_topology_absent_residue_ids: tuple[ResidueId, ...] = ()
    hydrogen_topology_unsupported_residue_ids: tuple[ResidueId, ...] = ()
    hydrogen_missing_residue_ids: tuple[ResidueId, ...] = ()
    hydrogen_prerequisite_residue_ids: tuple[ResidueId, ...] = ()
    hydrogen_blocked_residue_ids: tuple[ResidueId, ...] = ()
    disposition: WorkflowDeficitDisposition = WorkflowDeficitDisposition.OPTIONAL

    def __post_init__(self) -> None:
        for field_name in (
            "unsupported_component_residue_ids",
            "heavy_atom_topology_absent_residue_ids",
            "heavy_atom_topology_unsupported_residue_ids",
            "hydrogen_topology_absent_residue_ids",
            "hydrogen_topology_unsupported_residue_ids",
            "hydrogen_missing_residue_ids",
            "hydrogen_prerequisite_residue_ids",
            "hydrogen_blocked_residue_ids",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    def has_burden(self) -> bool:
        """Return whether any chemistry-readiness burden remains unresolved."""

        return any(
            (
                self.unsupported_component_residue_ids,
                self.heavy_atom_topology_absent_residue_ids,
                self.heavy_atom_topology_unsupported_residue_ids,
                self.hydrogen_topology_absent_residue_ids,
                self.hydrogen_topology_unsupported_residue_ids,
                self.hydrogen_missing_residue_ids,
                self.hydrogen_prerequisite_residue_ids,
                self.hydrogen_blocked_residue_ids,
            )
        )


@dataclass(frozen=True, slots=True)
class StructureTopologyResolutionDeficit:
    """Canonical topology evidence that still requires explicit resolution."""

    promotable_disulfide_count: int
    disposition: WorkflowDeficitDisposition

    def __post_init__(self) -> None:
        if self.promotable_disulfide_count <= 0:
            raise ValueError(
                "topology resolution deficits require promotable evidence"
            )


@dataclass(frozen=True, slots=True)
class StructureDisulfideHydrogenDeficit:
    """Forbidden thiol hydrogens that contradict canonical disulfide topology."""

    forbidden_hydrogen_count: int
    affected_residue_count: int
    disposition: WorkflowDeficitDisposition

    def __post_init__(self) -> None:
        if self.forbidden_hydrogen_count <= 0:
            raise ValueError(
                "disulfide hydrogen deficits require forbidden hydrogen atoms"
            )
        if self.affected_residue_count <= 0:
            raise ValueError(
                "disulfide hydrogen deficits require affected residues"
            )
        if self.affected_residue_count > self.forbidden_hydrogen_count:
            raise ValueError(
                "affected disulfide residues cannot exceed forbidden atoms"
            )


@dataclass(frozen=True, slots=True)
class StructureIntrinsicGeometryDeficit:
    """Intrinsic geometry burdens that revise present structure geometry."""

    protein_self_clash_count: int
    protein_self_clash_observation_mode: ClashObservationMode
    orientation_correction_eligible: bool
    stereochemistry_violated: bool
    disposition: WorkflowDeficitDisposition

    def __post_init__(self) -> None:
        if self.protein_self_clash_count < 0:
            raise ValueError("protein self-clash count must be non-negative")

    def has_burden(self) -> bool:
        """Return whether intrinsic geometry still carries unresolved burden."""

        return (
            self.protein_self_clash_count > 0
            or self.orientation_correction_eligible
            or self.stereochemistry_violated
        )


@dataclass(frozen=True, slots=True)
class StructureParserCompatibilityDeficit:
    """Parser-profile compatibility burdens over downstream tool projection."""

    profile: ParserCompatibilityProfile
    compatibility_state: ParserCompatibilityState
    extra_proximity_bond_count: int
    extra_heavy_proximity_bond_count: int
    disposition: WorkflowDeficitDisposition

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ParserCompatibilityProfile):
            raise TypeError(
                "parser compatibility deficits require a parser profile value"
            )
        if not isinstance(self.compatibility_state, ParserCompatibilityState):
            raise TypeError(
                "parser compatibility deficits require a compatibility state value"
            )
        if self.extra_proximity_bond_count < 0:
            raise ValueError("parser extra proximity-bond count must be non-negative")
        if self.extra_heavy_proximity_bond_count < 0:
            raise ValueError(
                "parser extra heavy proximity-bond count must be non-negative"
            )

    def has_burden(self) -> bool:
        """Return whether parser compatibility still carries unresolved burden."""

        return self.has_parser_visible_proximity_burden()

    def has_parser_visible_proximity_burden(self) -> bool:
        """Return whether parser compatibility failed on geometry-like proximity."""

        return (
            self.compatibility_state.is_incompatible()
            and self.extra_proximity_bond_count > 0
        )


@dataclass(frozen=True, slots=True)
class StructureInteractionDeficit:
    """Ligand-aware interaction burdens over one holo planning context."""

    ligand_aware_clash_count: int
    ligand_aware_worst_overlap_angstrom: float
    ligand_aware_total_overlap_angstrom: float
    ligand_aware_clash_observation_mode: ClashObservationMode
    disposition: WorkflowDeficitDisposition

    def has_burden(self) -> bool:
        """Return whether ligand-aware interaction burden is present."""

        return self.ligand_aware_clash_count > 0


@dataclass(frozen=True, slots=True)
class BackboneWindowOperatorDeficit:
    """One requested geometry burden that requires a backbone-window operator."""

    window_spec: BackboneWindowRefinementSpec
    disposition: WorkflowDeficitDisposition

    def __post_init__(self) -> None:
        if not isinstance(self.window_spec, BackboneWindowRefinementSpec):
            raise TypeError(
                "backbone-window operator deficits require a "
                "BackboneWindowRefinementSpec"
            )


@dataclass(frozen=True, slots=True)
class WorkflowStateDeficit:
    """Planner-facing unresolved burden derived from facts, goals, and context."""

    coverage: StructureCoverageDeficit
    topology_resolution: StructureTopologyResolutionDeficit | None
    disulfide_hydrogen: StructureDisulfideHydrogenDeficit | None
    chemistry_readiness: StructureChemistryReadinessDeficit
    backbone_window_operator: tuple[BackboneWindowOperatorDeficit, ...] = ()
    intrinsic_geometry: StructureIntrinsicGeometryDeficit | None = None
    parser_compatibility: StructureParserCompatibilityDeficit | None = None
    interaction: StructureInteractionDeficit | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backbone_window_operator",
            tuple(self.backbone_window_operator),
        )

    @classmethod
    def from_facts(
        cls,
        *,
        coverage_facts: StructureCoverageFacts,
        chemistry_readiness_facts: StructureChemistryReadinessFacts,
        disulfide_topology_facts: StructureDisulfideTopologyFacts,
        disulfide_hydrogen_facts: StructureDisulfideHydrogenFacts,
        requested_goals: RequestedGoalSet,
        planning_context: WorkflowPlanningContext,
        intrinsic_geometry_facts: StructureIntrinsicGeometryFacts | None = None,
        parser_compatibility_facts: StructureParserCompatibilityFacts | None = None,
        interaction_facts: StructureInteractionFacts | None = None,
        backbone_window_refinement_specs: tuple[
            BackboneWindowRefinementSpec,
            ...,
        ] = (),
    ) -> "WorkflowStateDeficit":
        """Derive planner-facing burden from canonical facts and goals."""

        if coverage_facts.carrier is not chemistry_readiness_facts.carrier:
            raise ValueError(
                "workflow deficits require coverage and chemistry facts for the "
                "same structure"
            )
        if disulfide_topology_facts.carrier is not coverage_facts.carrier:
            raise ValueError(
                "workflow deficits require disulfide topology facts for the "
                "same structure"
            )
        if disulfide_hydrogen_facts.carrier is not coverage_facts.carrier:
            raise ValueError(
                "workflow deficits require disulfide hydrogen facts for the "
                "same structure"
            )
        if intrinsic_geometry_facts is not None and (
            intrinsic_geometry_facts.carrier is not coverage_facts.carrier
        ):
            raise ValueError(
                "workflow deficits require intrinsic geometry facts for the "
                "same structure"
            )
        if parser_compatibility_facts is not None and (
            parser_compatibility_facts.carrier is not coverage_facts.carrier
        ):
            raise ValueError(
                "workflow deficits require parser compatibility facts for the "
                "same structure"
            )
        if interaction_facts is not None and (
            interaction_facts.carrier is not coverage_facts.carrier
        ):
            raise ValueError(
                "workflow deficits require interaction facts for the same structure"
            )

        coverage = _coverage_deficit(
            coverage_facts=coverage_facts,
            chemistry_readiness_facts=chemistry_readiness_facts,
            requested_goals=requested_goals,
            planning_context=planning_context,
        )
        chemistry_readiness = _chemistry_readiness_deficit(
            coverage_facts=coverage_facts,
            chemistry_readiness_facts=chemistry_readiness_facts,
            requested_goals=requested_goals,
            planning_context=planning_context,
        )
        topology_resolution = _topology_resolution_deficit(
            disulfide_topology_facts
        )
        disulfide_hydrogen = _disulfide_hydrogen_deficit(
            disulfide_hydrogen_facts
        )
        intrinsic_geometry = _intrinsic_geometry_deficit(
            intrinsic_geometry_facts=intrinsic_geometry_facts,
            requested_goals=requested_goals,
            planning_context=planning_context,
        )
        parser_compatibility = _parser_compatibility_deficit(
            parser_compatibility_facts=parser_compatibility_facts,
            requested_goals=requested_goals,
            planning_context=planning_context,
        )
        interaction = _interaction_deficit(
            interaction_facts=interaction_facts,
            requested_goals=requested_goals,
            planning_context=planning_context,
        )
        backbone_window_operator = _backbone_window_operator_deficits(
            backbone_window_refinement_specs=backbone_window_refinement_specs,
        )
        return cls(
            coverage=coverage,
            topology_resolution=topology_resolution,
            disulfide_hydrogen=disulfide_hydrogen,
            chemistry_readiness=chemistry_readiness,
            backbone_window_operator=backbone_window_operator,
            intrinsic_geometry=intrinsic_geometry,
            parser_compatibility=parser_compatibility,
            interaction=interaction,
        )


def _topology_resolution_deficit(
    facts: StructureDisulfideTopologyFacts,
) -> StructureTopologyResolutionDeficit | None:
    """Return a required deficit for promotable disulfide evidence."""

    if not facts.promotable_candidates:
        return None

    return StructureTopologyResolutionDeficit(
        promotable_disulfide_count=len(facts.promotable_candidates),
        disposition=WorkflowDeficitDisposition.REQUIRED,
    )


def _disulfide_hydrogen_deficit(
    facts: StructureDisulfideHydrogenFacts,
) -> StructureDisulfideHydrogenDeficit | None:
    """Return required burden for disulfide-incompatible hydrogens."""

    forbidden_atom_refs = facts.forbidden_hydrogen_atom_refs()
    if not forbidden_atom_refs:
        return None

    return StructureDisulfideHydrogenDeficit(
        forbidden_hydrogen_count=len(forbidden_atom_refs),
        affected_residue_count=len(
            {atom_ref.residue_id for atom_ref in forbidden_atom_refs}
        ),
        disposition=WorkflowDeficitDisposition.REQUIRED,
    )


def _coverage_deficit(
    *,
    coverage_facts: StructureCoverageFacts,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> StructureCoverageDeficit:
    """Project unresolved coverage burden from canonical structure facts."""

    gap_disposition = _gap_disposition(planning_context)
    gap_deficits = tuple(
        CoverageGapDeficit(
            gap=gap,
            disposition=(
                gap_disposition
                if planning_context.external_span_gap_selection_policy.selects_gap(gap)
                else WorkflowDeficitDisposition.OPTIONAL
            ),
        )
        for gap in coverage_facts.coverage_gaps
    )
    required_atom_coverage = _requires_atom_coverage(
        requested_goals=requested_goals,
        planning_context=planning_context,
    )
    chemistry_by_residue_id = {
        residue_fact.residue_id: residue_fact
        for residue_fact in chemistry_readiness_facts.residue_facts
    }
    atom_deficits: list[AtomCoverageDeficit] = []
    for residue_fact in coverage_facts.residue_facts:
        if (
            not residue_fact.requires_backbone_completion()
            and not residue_fact.requires_sidechain_completion()
        ):
            continue
        chemistry_fact = chemistry_by_residue_id[residue_fact.residue_id]
        blocked_by_component_support = not chemistry_fact.is_supported()
        disposition = WorkflowDeficitDisposition.OPTIONAL
        if required_atom_coverage:
            disposition = (
                WorkflowDeficitDisposition.BLOCKED
                if blocked_by_component_support
                else WorkflowDeficitDisposition.REQUIRED
            )
        atom_deficits.append(
            AtomCoverageDeficit(
                residue_id=residue_fact.residue_id,
                missing_backbone_atoms=residue_fact.requires_backbone_completion(),
                missing_sidechain_atoms=residue_fact.requires_sidechain_completion(),
                blocked_by_component_support=blocked_by_component_support,
                disposition=disposition,
            )
        )

    return StructureCoverageDeficit(
        gap_deficits=gap_deficits,
        atom_deficits=tuple(atom_deficits),
    )


def _chemistry_readiness_deficit(
    *,
    coverage_facts: StructureCoverageFacts,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> StructureChemistryReadinessDeficit:
    """Project unresolved chemistry-readiness burden from canonical facts."""

    coverage_by_residue_id = {
        residue_fact.residue_id: residue_fact
        for residue_fact in coverage_facts.residue_facts
    }
    unsupported_component_residue_ids: list[ResidueId] = []
    heavy_atom_topology_absent_residue_ids: list[ResidueId] = []
    heavy_atom_topology_unsupported_residue_ids: list[ResidueId] = []
    hydrogen_topology_absent_residue_ids: list[ResidueId] = []
    hydrogen_topology_unsupported_residue_ids: list[ResidueId] = []
    hydrogen_missing_residue_ids: list[ResidueId] = []
    hydrogen_prerequisite_residue_ids: list[ResidueId] = []
    hydrogen_blocked_residue_ids: list[ResidueId] = []

    requires_hydrogen_readiness = _requires_hydrogen_readiness(
        requested_goals=requested_goals,
        planning_context=planning_context,
    )
    for residue_fact in chemistry_readiness_facts.residue_facts:
        if not residue_fact.is_supported():
            unsupported_component_residue_ids.append(residue_fact.residue_id)
            if requires_hydrogen_readiness:
                hydrogen_blocked_residue_ids.append(residue_fact.residue_id)
        if (
            residue_fact.heavy_atom_topology_availability_state
            is TopologyAvailabilityState.ABSENT
        ):
            heavy_atom_topology_absent_residue_ids.append(residue_fact.residue_id)
        if (
            residue_fact.heavy_atom_topology_availability_state
            is TopologyAvailabilityState.UNSUPPORTED
        ):
            heavy_atom_topology_unsupported_residue_ids.append(
                residue_fact.residue_id
            )
        if (
            residue_fact.hydrogen_topology_availability_state
            is TopologyAvailabilityState.ABSENT
        ):
            hydrogen_topology_absent_residue_ids.append(residue_fact.residue_id)
        if (
            residue_fact.hydrogen_topology_availability_state
            is TopologyAvailabilityState.UNSUPPORTED
        ):
            hydrogen_topology_unsupported_residue_ids.append(residue_fact.residue_id)

        if (
            residue_fact.hydrogen_applicability_state
            is HydrogenApplicabilityState.NOT_APPLICABLE
            or not residue_fact.needs_hydrogenation()
            or not residue_fact.is_supported()
        ):
            continue

        coverage_fact = coverage_by_residue_id[residue_fact.residue_id]
        if (
            coverage_fact.requires_backbone_completion()
            or coverage_fact.requires_sidechain_completion()
        ):
            hydrogen_prerequisite_residue_ids.append(residue_fact.residue_id)
            continue
        hydrogen_missing_residue_ids.append(residue_fact.residue_id)

    disposition = WorkflowDeficitDisposition.OPTIONAL
    if requires_hydrogen_readiness:
        disposition = (
            WorkflowDeficitDisposition.BLOCKED
            if hydrogen_blocked_residue_ids
            else WorkflowDeficitDisposition.REQUIRED
        )

    return StructureChemistryReadinessDeficit(
        unsupported_component_residue_ids=tuple(unsupported_component_residue_ids),
        heavy_atom_topology_absent_residue_ids=tuple(
            heavy_atom_topology_absent_residue_ids
        ),
        heavy_atom_topology_unsupported_residue_ids=tuple(
            heavy_atom_topology_unsupported_residue_ids
        ),
        hydrogen_topology_absent_residue_ids=tuple(
            hydrogen_topology_absent_residue_ids
        ),
        hydrogen_topology_unsupported_residue_ids=tuple(
            hydrogen_topology_unsupported_residue_ids
        ),
        hydrogen_missing_residue_ids=tuple(hydrogen_missing_residue_ids),
        hydrogen_prerequisite_residue_ids=tuple(
            hydrogen_prerequisite_residue_ids
        ),
        hydrogen_blocked_residue_ids=tuple(hydrogen_blocked_residue_ids),
        disposition=disposition,
    )



def _intrinsic_geometry_deficit(
    *,
    intrinsic_geometry_facts: StructureIntrinsicGeometryFacts | None,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> StructureIntrinsicGeometryDeficit | None:
    """Project intrinsic geometry burden from optional canonical facts."""

    if intrinsic_geometry_facts is None:
        return None

    deficit = StructureIntrinsicGeometryDeficit(
        protein_self_clash_count=intrinsic_geometry_facts.protein_self_clash_count,
        protein_self_clash_observation_mode=(
            intrinsic_geometry_facts.protein_self_clash_observation_mode
        ),
        orientation_correction_eligible=(
            intrinsic_geometry_facts.orientation_correction_eligibility_state
            is OrientationCorrectionEligibilityState.ELIGIBLE
        ),
        stereochemistry_violated=(
            intrinsic_geometry_facts.stereochemistry_state
            is StereochemistryState.VIOLATED
        ),
        disposition=(
            WorkflowDeficitDisposition.REQUIRED
            if _requires_geometry_correction(
                requested_goals=requested_goals,
                planning_context=planning_context,
            )
            else WorkflowDeficitDisposition.OPTIONAL
        ),
    )
    return deficit if deficit.has_burden() else None


def _parser_compatibility_deficit(
    *,
    parser_compatibility_facts: StructureParserCompatibilityFacts | None,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> StructureParserCompatibilityDeficit | None:
    """Project parser compatibility burden from optional parser-profile facts."""

    if parser_compatibility_facts is None:
        return None
    if not parser_compatibility_facts.has_parser_visible_proximity_burden():
        return None

    deficit = StructureParserCompatibilityDeficit(
        profile=parser_compatibility_facts.profile,
        compatibility_state=parser_compatibility_facts.compatibility_state,
        extra_proximity_bond_count=(
            parser_compatibility_facts.extra_proximity_bond_count
        ),
        extra_heavy_proximity_bond_count=(
            parser_compatibility_facts.extra_heavy_proximity_bond_count
        ),
        disposition=(
            WorkflowDeficitDisposition.REQUIRED
            if _requires_parser_compatibility_correction(
                requested_goals=requested_goals,
                planning_context=planning_context,
            )
            else WorkflowDeficitDisposition.OPTIONAL
        ),
    )
    return deficit if deficit.has_burden() else None


def _interaction_deficit(
    *,
    interaction_facts: StructureInteractionFacts | None,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> StructureInteractionDeficit | None:
    """Project ligand-aware interaction burden from optional canonical facts."""

    if interaction_facts is None:
        return None

    deficit = StructureInteractionDeficit(
        ligand_aware_clash_count=interaction_facts.ligand_aware_clash_count,
        ligand_aware_worst_overlap_angstrom=(
            interaction_facts.ligand_aware_worst_overlap_angstrom
        ),
        ligand_aware_total_overlap_angstrom=(
            interaction_facts.ligand_aware_total_overlap_angstrom
        ),
        ligand_aware_clash_observation_mode=(
            interaction_facts.ligand_aware_clash_observation_mode
        ),
        disposition=(
            WorkflowDeficitDisposition.REQUIRED
            if _requires_interaction_correction(
                requested_goals=requested_goals,
                planning_context=planning_context,
            )
            else WorkflowDeficitDisposition.OPTIONAL
        ),
    )
    return deficit if deficit.has_burden() else None


def _backbone_window_operator_deficits(
    *,
    backbone_window_refinement_specs: tuple[BackboneWindowRefinementSpec, ...],
) -> tuple[BackboneWindowOperatorDeficit, ...]:
    """Project explicit backbone-window needs into planner operator deficits."""

    return tuple(
        BackboneWindowOperatorDeficit(
            window_spec=window_spec,
            disposition=WorkflowDeficitDisposition.REQUIRED,
        )
        for window_spec in backbone_window_refinement_specs
    )


def _gap_disposition(
    planning_context: WorkflowPlanningContext,
) -> WorkflowDeficitDisposition:
    """Return the disposition for context-selected missing-residue spans."""

    if planning_context.allows_span_reconstruction():
        return WorkflowDeficitDisposition.REQUIRED

    return WorkflowDeficitDisposition.OPTIONAL


def _requires_atom_coverage(
    *,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> bool:
    """Return whether atom-coverage burden is required under current policy."""

    return (
        requested_goals.requests_whole_structure_heavy_atom_completion()
        or requested_goals.requests_whole_structure_hydrogen_population()
        or planning_context.target_intent
        in (WorkflowTargetIntent.DOCKING, WorkflowTargetIntent.MD_READY)
    )


def _requires_hydrogen_readiness(
    *,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> bool:
    """Return whether hydrogen readiness is required under current policy."""

    return (
        requested_goals.requests_whole_structure_hydrogen_population()
        or planning_context.target_intent is WorkflowTargetIntent.MD_READY
    )


def _requires_geometry_correction(
    *,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> bool:
    """Return whether intrinsic geometry burden is goal- or intent-relevant."""

    return (
        requested_goals.requests_whole_structure_clash_absence()
        or planning_context.target_intent
        in (WorkflowTargetIntent.DOCKING, WorkflowTargetIntent.MD_READY)
    )


def _requires_parser_compatibility_correction(
    *,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> bool:
    """Return whether parser compatibility burden is goal- or intent-relevant."""

    return _requires_geometry_correction(
        requested_goals=requested_goals,
        planning_context=planning_context,
    )


def _requires_interaction_correction(
    *,
    requested_goals: RequestedGoalSet,
    planning_context: WorkflowPlanningContext,
) -> bool:
    """Return whether holo interaction burden is goal- or intent-relevant."""

    if planning_context.target_intent in (
        WorkflowTargetIntent.DOCKING,
        WorkflowTargetIntent.MD_READY,
    ):
        return True

    return requested_goals.requests_whole_structure_clash_absence()
