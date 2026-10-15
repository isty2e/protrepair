"""Residue-local coverage, chemistry readiness and projected state facts."""

from dataclasses import dataclass
from typing import cast

from protrepair.chemistry import (
    ComponentLibrary,
)
from protrepair.chemistry.microstate.preparation import (
    PolymerSitePreparation,
)
from protrepair.scope import ResidueSetScope
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
)
from protrepair.state.scoped import CarrierScopedState
from protrepair.state.structure_axes import (
    BackboneHeavyAtomCompletenessState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
)
from protrepair.state.topology import (
    TopologyAvailabilityState,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId


@dataclass(frozen=True, slots=True)
class ResidueProjectionStateFacts:
    """Primitive state facts over one singleton residue projection."""

    component_support_fact: CarrierScopedState[
        ProteinStructure,
        ComponentSupportState,
    ]
    backbone_heavy_atom_completeness_fact: CarrierScopedState[
        ProteinStructure,
        BackboneHeavyAtomCompletenessState,
    ]
    sidechain_heavy_atom_completeness_fact: CarrierScopedState[
        ProteinStructure,
        SidechainHeavyAtomCompletenessState,
    ]
    hydrogen_applicability_fact: CarrierScopedState[
        ProteinStructure,
        HydrogenApplicabilityState,
    ]
    hydrogen_coverage_fact: CarrierScopedState[
        ProteinStructure,
        HydrogenCoverageState,
    ]
    stereochemistry_fact: CarrierScopedState[
        ProteinStructure,
        StereochemistryState,
    ]

    def __post_init__(self) -> None:
        shared_facts = (
            self.component_support_fact,
            self.backbone_heavy_atom_completeness_fact,
            self.sidechain_heavy_atom_completeness_fact,
            self.hydrogen_applicability_fact,
            self.hydrogen_coverage_fact,
            self.stereochemistry_fact,
        )
        expected_carrier = shared_facts[0].carrier
        expected_scope = shared_facts[0].scope
        if any(fact.carrier != expected_carrier for fact in shared_facts[1:]):
            raise ValueError("residue projection facts require one shared carrier")
        if any(fact.scope != expected_scope for fact in shared_facts[1:]):
            raise ValueError(
                "residue projection facts require one shared projection scope"
            )
        if not isinstance(expected_scope, ResidueSetScope):
            raise TypeError("residue projection facts require a residue-set scope")
        if len(expected_scope.residue_ids) != 1:
            raise ValueError(
                "residue projection facts require exactly one residue in scope"
            )

    @property
    def carrier(self) -> ProteinStructure:
        """Return the carrier shared by all singleton residue facts."""

        return self.component_support_fact.carrier

    @property
    def scope(self) -> ResidueSetScope:
        """Return the singleton residue scope for these primitive facts."""

        return cast(ResidueSetScope, self.component_support_fact.scope)

    @property
    def residue_id(self) -> ResidueId:
        """Return the projected residue identifier."""

        return self.scope.residue_ids[0]

    @classmethod
    def from_residue(
        cls,
        context_structure: ProteinStructure,
        residue: ResidueSite,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> "ResidueProjectionStateFacts":
        """Derive singleton facts for a residue belonging to its context.

        Parameters
        ----------
        context_structure : ProteinStructure
            Complete source of geometry, topology and chemical context.
        residue : ResidueSite
            A context residue or an equal-value copy of its scaffold.
        component_library : ComponentLibrary or None
            Active component definitions, or defaults.

        Returns
        -------
        ResidueProjectionStateFacts
            Primitive facts for the supplied context residue.

        Raises
        ------
        KeyError
            The residue identity is absent from the context.
        ValueError
            The supplied scaffold differs from the context residue.
        """

        from protrepair.state.chemistry_observation import StructureChemistryObservation

        observation = StructureChemistryObservation(
            context_structure,
            component_library=component_library,
        )
        observation.require_residue(residue)
        return observation.residue_facts(residue.residue_id)


@dataclass(frozen=True, slots=True)
class ResidueCoverageFacts:
    """Coverage-relevant residue-local truth for one canonical residue."""

    residue_id: ResidueId
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState

    def requires_backbone_completion(self) -> bool:
        """Return whether backbone heavy-atom completion is required."""

        return self.backbone_heavy_atom_completeness_state.requires_completion()

    def requires_sidechain_completion(self) -> bool:
        """Return whether side-chain heavy-atom completion is required."""

        return self.sidechain_heavy_atom_completeness_state.requires_completion()


@dataclass(frozen=True, slots=True)
class ResidueChemistryReadinessFacts:
    """Chemistry-readiness residue-local truth for one canonical residue."""

    residue_id: ResidueId
    component_support_state: ComponentSupportState
    heavy_atom_topology_availability_state: TopologyAvailabilityState
    hydrogen_topology_availability_state: TopologyAvailabilityState
    hydrogen_applicability_state: HydrogenApplicabilityState
    hydrogen_coverage_state: HydrogenCoverageState

    polymer_microstate_targets: tuple[PolymerSitePreparation, ...] = ()

    def requires_microstate_realization(self) -> bool:
        """Return whether a selected graph differs from current chemistry.

        Returns
        -------
        bool
            True for a resolved but unrealized site, independently of H inventory.
        """
        return any(
            target.resolution.graph is not None and not target.is_realized()
            for target in self.polymer_microstate_targets
        )

    def has_unrealized_microstates(self) -> bool:
        """Return whether any applicable site is unresolved or unrealized.

        Returns
        -------
        bool
            A chemistry-readiness blocker, including sites with complete H counts.
        """
        return any(
            not target.is_realized() for target in self.polymer_microstate_targets
        )

    def is_supported(self) -> bool:
        """Return whether canonical chemistry support is available."""

        return self.component_support_state.is_fully_supported()

    def needs_hydrogenation(self) -> bool:
        """Return whether hydrogen completion may still add missing hydrogens."""

        return self.hydrogen_coverage_state.needs_hydrogenation()


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerChemistryReadinessFact:
    """Chemistry-readiness truth for one retained non-polymer component."""

    residue_id: ResidueId
    component_id: str
    component_support_state: ComponentSupportState
    heavy_topology_source: RetainedNonPolymerChemistryEvidenceSource
    hydrogen_expectation_source: RetainedNonPolymerChemistryEvidenceSource
    heavy_atom_topology_availability_state: TopologyAvailabilityState
    hydrogen_topology_availability_state: TopologyAvailabilityState
    hydrogen_applicability_state: HydrogenApplicabilityState
    hydrogen_coverage_state: HydrogenCoverageState
    chemistry_failure_reason: str = ""

    def is_supported(self) -> bool:
        """Return whether canonical chemistry support is available."""

        return self.component_support_state.is_fully_supported()

    def needs_hydrogenation(self) -> bool:
        """Return whether hydrogen completion may still add missing hydrogens."""

        return self.hydrogen_coverage_state.needs_hydrogenation()

    def depends_on_rdkit_fallback_support(self) -> bool:
        """Return whether this retained component uses RDKit fallback support."""

        return (
            self.heavy_topology_source
            is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
            or self.hydrogen_expectation_source
            is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
        )

    def requires_hydrogen_completion(self) -> bool:
        """Return whether retained non-polymer hydrogen completion is required."""

        return (
            self.is_supported()
            and self.hydrogen_applicability_state
            is HydrogenApplicabilityState.APPLICABLE
            and self.needs_hydrogenation()
        )

    def requires_topology_resolution(self) -> bool:
        """Select graph-only repair when supported chemistry needs no new H.

        Returns
        -------
        bool
            Whether absent graph facts remain despite a complete or inapplicable
            H inventory. Chemistry conflicts still require transformer validation.
        """
        return (
            self.is_supported()
            and not self.needs_hydrogenation()
            and (
                self.heavy_atom_topology_availability_state.is_unavailable()
                or self.hydrogen_topology_availability_state.is_unavailable()
            )
        )


@dataclass(frozen=True, slots=True)
class ResidueCompletionStateFacts:
    """Transitional aggregate over residue coverage and chemistry-readiness."""

    coverage_facts: ResidueCoverageFacts
    chemistry_readiness_facts: ResidueChemistryReadinessFacts

    @property
    def residue_id(self) -> ResidueId:
        """Return the projected residue identifier."""

        return self.coverage_facts.residue_id

    @property
    def component_support_state(self) -> ComponentSupportState:
        """Return canonical chemistry support over this residue."""

        return self.chemistry_readiness_facts.component_support_state

    @property
    def backbone_heavy_atom_completeness_state(
        self,
    ) -> BackboneHeavyAtomCompletenessState:
        """Return backbone heavy-atom completeness over this residue."""

        return self.coverage_facts.backbone_heavy_atom_completeness_state

    @property
    def sidechain_heavy_atom_completeness_state(
        self,
    ) -> SidechainHeavyAtomCompletenessState:
        """Return side-chain heavy-atom completeness over this residue."""

        return self.coverage_facts.sidechain_heavy_atom_completeness_state

    @property
    def hydrogen_applicability_state(self) -> HydrogenApplicabilityState:
        """Return hydrogen applicability over this residue."""

        return self.chemistry_readiness_facts.hydrogen_applicability_state

    @property
    def hydrogen_coverage_state(self) -> HydrogenCoverageState:
        """Return hydrogen coverage over this residue."""

        return self.chemistry_readiness_facts.hydrogen_coverage_state

    def is_supported(self) -> bool:
        """Return whether canonical chemistry support is available."""

        return self.chemistry_readiness_facts.is_supported()

    def requires_backbone_completion(self) -> bool:
        """Return whether backbone heavy-atom completion is required."""

        return self.coverage_facts.requires_backbone_completion()

    def requires_sidechain_completion(self) -> bool:
        """Return whether side-chain heavy-atom completion is required."""

        return self.coverage_facts.requires_sidechain_completion()

    def needs_hydrogenation(self) -> bool:
        """Return whether hydrogen completion may still add missing hydrogens."""

        return self.chemistry_readiness_facts.needs_hydrogenation()


__all__ = [
    "ResidueChemistryReadinessFacts",
    "ResidueCompletionStateFacts",
    "ResidueCoverageFacts",
    "ResidueProjectionStateFacts",
    "RetainedNonPolymerChemistryReadinessFact",
]
