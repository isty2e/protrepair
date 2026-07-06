"""Observed summaries derived over one selected atom scope."""

from dataclasses import dataclass
from enum import Enum

from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.state.scoped import CarrierScopedState
from protrepair.state.structure_axes import (
    BackboneHeavyAtomCompletenessState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.state.structure_boundary import (
    TerminalBoundaryObservation,
)
from protrepair.state.structure_projection import (
    ProteinStructureObservation,
    StructureProjectionStateFacts,
)
from protrepair.state.structure_readiness import (
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
)
from protrepair.state.topology import (
    HydrogenAttachmentResolutionFacts,
    HydrogenAttachmentResolutionObservation,
    TopologyAvailabilityFacts,
    TopologyAvailabilityObservation,
)
from protrepair.state.topology import (
    HydrogenAttachmentResolutionState as HydrogenAttachmentResolutionState,
)
from protrepair.state.topology import (
    ResidueHydrogenAttachmentResolutionFact as ResidueHydrogenAttachmentResolutionFact,
)
from protrepair.state.topology import (
    ResidueTopologyAvailabilityFact as ResidueTopologyAvailabilityFact,
)
from protrepair.state.topology import (
    TopologyAvailabilityAspect as TopologyAvailabilityAspect,
)
from protrepair.state.topology import (
    TopologyAvailabilityState as TopologyAvailabilityState,
)
from protrepair.structure.aggregate import ProteinStructure


class ClashState(str, Enum):
    """Observed steric clash state over one selected local atom scope.

    This is the local/scope-specific clash axis used by transformation
    legality. Whole-structure clash goals use `ClashPresenceState` instead.
    """

    NONE = "none"
    PRESENT = "present"

    def has_clashes(self) -> bool:
        """Return whether at least one steric clash is present."""

        return self is ClashState.PRESENT


class ContinuousBondRealizabilityState(str, Enum):
    """Selected-scope bond-graph realizability under continuous semantics."""

    REALIZABLE = "realizable"
    UNREALIZABLE = "unrealizable"


@dataclass(frozen=True, slots=True)
class ContinuousBondRealizabilityFacts:
    """Planner-readable bond-graph realizability over one selected local scope."""

    carrier: ProteinStructure
    scope: ResidueSetScope | AtomSetScope
    state: ContinuousBondRealizabilityState
    blocker: str | None = None

    def __post_init__(self) -> None:
        if self.state is ContinuousBondRealizabilityState.REALIZABLE:
            if self.blocker is not None:
                raise ValueError(
                    "realizable continuous bond facts must not carry a blocker"
                )
            return

        if self.blocker is None:
            raise ValueError(
                "unrealizable continuous bond facts require a blocker explanation"
            )

    def is_realizable(self) -> bool:
        """Return whether the selected scope admits continuous bond planning."""

        return self.state is ContinuousBondRealizabilityState.REALIZABLE


@dataclass(frozen=True, slots=True)
class ContinuousBondRealizabilityObservation:
    """History-free observation over selected-scope bond-graph realizability."""

    state: ContinuousBondRealizabilityState
    blocker: str | None = None

    def __post_init__(self) -> None:
        if self.state is ContinuousBondRealizabilityState.REALIZABLE:
            if self.blocker is not None:
                raise ValueError(
                    "realizable continuous bond observation must not carry a blocker"
                )
            return

        if self.blocker is None:
            raise ValueError(
                "unrealizable continuous bond observation requires a blocker"
            )

    @classmethod
    def from_facts(
        cls,
        facts: ContinuousBondRealizabilityFacts,
    ) -> "ContinuousBondRealizabilityObservation":
        """Rebuild bond-graph realizability observation from primitive facts."""

        return cls(state=facts.state, blocker=facts.blocker)

    def is_realizable(self) -> bool:
        """Return whether the selected scope admits continuous bond planning."""

        return self.state is ContinuousBondRealizabilityState.REALIZABLE


@dataclass(frozen=True, slots=True)
class SelectedAtomScopeFacts:
    """Primitive facts observed directly over one selected atom scope."""

    atom_count: int
    residue_count: int
    hydrogen_atom_count: int
    structure_facts: StructureProjectionStateFacts
    clash_fact: CarrierScopedState[ProteinStructure, ClashState]
    topology_availability_facts: TopologyAvailabilityFacts
    hydrogen_attachment_resolution_facts: HydrogenAttachmentResolutionFacts
    continuous_bond_realizability_facts: ContinuousBondRealizabilityFacts

    def __post_init__(self) -> None:
        if self.structure_facts.carrier != self.clash_fact.carrier:
            raise ValueError(
                "atom-scope facts require one shared carrier across "
                "structure and clash facts"
            )
        if self.topology_availability_facts.carrier != self.clash_fact.carrier:
            raise ValueError(
                "atom-scope facts require one shared carrier across "
                "topology and clash facts"
            )
        if self.hydrogen_attachment_resolution_facts.carrier != self.clash_fact.carrier:
            raise ValueError(
                "atom-scope facts require one shared carrier across "
                "hydrogen attachment resolution and clash facts"
            )
        if self.continuous_bond_realizability_facts.carrier != self.clash_fact.carrier:
            raise ValueError(
                "atom-scope facts require one shared carrier across continuous "
                "bond realizability and clash facts"
            )

    @property
    def carrier(self) -> ProteinStructure:
        """Return the shared structure carrier for this selected scope."""

        return self.clash_fact.carrier


@dataclass(frozen=True, slots=True)
class ContinuousRegionReadinessFacts:
    """Readiness facts over one continuous-relaxation included local region."""

    coverage_facts: StructureCoverageFacts
    chemistry_readiness_facts: StructureChemistryReadinessFacts

    def __post_init__(self) -> None:
        if self.coverage_facts.carrier != self.chemistry_readiness_facts.carrier:
            raise ValueError(
                "continuous-region readiness facts require one shared carrier across "
                "coverage and chemistry readiness facts"
            )

    @property
    def carrier(self) -> ProteinStructure:
        """Return the shared structure carrier for this included local region."""

        return self.coverage_facts.carrier


@dataclass(frozen=True, slots=True)
class AtomScopeStateFacts:
    """Coordination shell over selected-scope and continuous-region fact owners."""

    selected_scope_facts: SelectedAtomScopeFacts
    continuous_region_readiness_facts: ContinuousRegionReadinessFacts

    def __post_init__(self) -> None:
        if (
            self.selected_scope_facts.carrier
            != self.continuous_region_readiness_facts.carrier
        ):
            raise ValueError(
                "atom-scope facts require one shared carrier across selected-scope "
                "and continuous-region fact owners"
            )

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in the selected scope."""

        return self.selected_scope_facts.atom_count

    @property
    def residue_count(self) -> int:
        """Return the number of residues referenced by the selected scope."""

        return self.selected_scope_facts.residue_count

    @property
    def hydrogen_atom_count(self) -> int:
        """Return the selected-scope hydrogen atom count."""

        return self.selected_scope_facts.hydrogen_atom_count

    @property
    def structure_facts(self) -> StructureProjectionStateFacts:
        """Return projection facts observed directly over the selected scope."""

        return self.selected_scope_facts.structure_facts

    @property
    def clash_fact(self) -> CarrierScopedState[ProteinStructure, ClashState]:
        """Return clash truth observed directly over the selected scope."""

        return self.selected_scope_facts.clash_fact

    @property
    def topology_availability_facts(self) -> TopologyAvailabilityFacts:
        """Return topology-availability facts over the selected scope."""

        return self.selected_scope_facts.topology_availability_facts

    @property
    def hydrogen_attachment_resolution_facts(self) -> HydrogenAttachmentResolutionFacts:
        """Return hydrogen-attachment resolution facts over the selected scope."""

        return self.selected_scope_facts.hydrogen_attachment_resolution_facts

    @property
    def continuous_bond_realizability_facts(self) -> ContinuousBondRealizabilityFacts:
        """Return continuous bond realizability facts over the selected scope."""

        return self.selected_scope_facts.continuous_bond_realizability_facts

    @property
    def continuous_region_coverage_facts(self) -> StructureCoverageFacts:
        """Return readiness coverage facts over the included local region."""

        return self.continuous_region_readiness_facts.coverage_facts

    @property
    def continuous_region_chemistry_readiness_facts(
        self,
    ) -> StructureChemistryReadinessFacts:
        """Return readiness chemistry facts over the included local region."""

        return self.continuous_region_readiness_facts.chemistry_readiness_facts


@dataclass(frozen=True, slots=True)
class AtomScopeObservation:
    """History-free observed summary over one selected atom scope."""

    atom_count: int
    residue_count: int
    hydrogen_atom_count: int
    component_support_state: ComponentSupportState
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState
    hydrogen_applicability_state: HydrogenApplicabilityState
    hydrogen_coverage_state: HydrogenCoverageState
    clash_state: ClashState
    topology_availability: TopologyAvailabilityObservation
    hydrogen_attachment_resolution: HydrogenAttachmentResolutionObservation
    continuous_bond_realizability: ContinuousBondRealizabilityObservation
    terminal_boundary_observation: TerminalBoundaryObservation = (
        TerminalBoundaryObservation.empty()
    )

    def requires_heavy_atom_completion(self) -> bool:
        """Return whether any heavy-atom completeness axis remains incomplete."""

        return (
            self.backbone_heavy_atom_completeness_state.requires_completion()
            or self.sidechain_heavy_atom_completeness_state.requires_completion()
        )

    @classmethod
    def from_facts(
        cls,
        facts: AtomScopeStateFacts,
    ) -> "AtomScopeObservation":
        """Rebuild one canonical local observation from primitive facts."""

        selected_scope_facts = facts.selected_scope_facts
        structure_observation = ProteinStructureObservation.from_facts(
            selected_scope_facts.structure_facts
        )
        return cls(
            atom_count=selected_scope_facts.atom_count,
            residue_count=selected_scope_facts.residue_count,
            hydrogen_atom_count=selected_scope_facts.hydrogen_atom_count,
            component_support_state=structure_observation.component_support_state,
            backbone_heavy_atom_completeness_state=(
                structure_observation.backbone_heavy_atom_completeness_state
            ),
            sidechain_heavy_atom_completeness_state=(
                structure_observation.sidechain_heavy_atom_completeness_state
            ),
            hydrogen_applicability_state=(
                structure_observation.hydrogen_applicability_state
            ),
            hydrogen_coverage_state=structure_observation.hydrogen_coverage_state,
            clash_state=selected_scope_facts.clash_fact.value,
            topology_availability=TopologyAvailabilityObservation.from_facts(
                selected_scope_facts.topology_availability_facts
            ),
            hydrogen_attachment_resolution=(
                HydrogenAttachmentResolutionObservation.from_facts(
                    selected_scope_facts.hydrogen_attachment_resolution_facts
                )
            ),
            continuous_bond_realizability=(
                ContinuousBondRealizabilityObservation.from_facts(
                    selected_scope_facts.continuous_bond_realizability_facts
                )
            ),
            terminal_boundary_observation=(
                structure_observation.terminal_boundary_observation
            ),
        )
