"""Whole-structure coverage and chemistry-readiness facts."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.relation.blueprint import (
    StructureBlueprintCoverage,
    StructureBlueprintCoverageGap,
)
from protrepair.state.hydrogen_expectation import StructureHydrogenExpectationModel
from protrepair.state.structure_axes import (
    BackboneHeavyAtomCompletenessState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.state.structure_residue import (
    ResidueChemistryReadinessFacts,
    ResidueCoverageFacts,
    RetainedNonPolymerChemistryReadinessFact,
)
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite

if TYPE_CHECKING:
    from protrepair.state.chemistry_observation import StructureChemistryObservation


@dataclass(frozen=True, slots=True)
class StructureCoverageFacts:
    """Whole-structure coverage truth without chemistry-readiness axes."""

    carrier: ProteinStructure
    residue_facts: tuple[ResidueCoverageFacts, ...]
    blueprint_coverages: tuple[StructureBlueprintCoverage, ...]
    coverage_gaps: tuple[StructureBlueprintCoverageGap, ...]
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState

    def __post_init__(self) -> None:
        object.__setattr__(self, "residue_facts", tuple(self.residue_facts))
        object.__setattr__(self, "blueprint_coverages", tuple(self.blueprint_coverages))
        object.__setattr__(self, "coverage_gaps", tuple(self.coverage_gaps))

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> "StructureCoverageFacts":
        """Derive whole-structure coverage truth from one canonical structure."""

        coverage_facts, _ = derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=component_library,
        )
        return coverage_facts


@dataclass(frozen=True, slots=True)
class StructureChemistryReadinessFacts:
    """Whole-structure chemistry-readiness truth without coverage axes."""

    carrier: ProteinStructure
    residue_facts: tuple[ResidueChemistryReadinessFacts, ...]
    retained_non_polymer_facts: tuple[RetainedNonPolymerChemistryReadinessFact, ...]
    component_support_state: ComponentSupportState
    heavy_atom_topology_availability_state: TopologyAvailabilityState
    hydrogen_topology_availability_state: TopologyAvailabilityState
    hydrogen_applicability_state: HydrogenApplicabilityState
    hydrogen_coverage_state: HydrogenCoverageState

    def __post_init__(self) -> None:
        object.__setattr__(self, "residue_facts", tuple(self.residue_facts))
        object.__setattr__(
            self,
            "retained_non_polymer_facts",
            tuple(self.retained_non_polymer_facts),
        )

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> "StructureChemistryReadinessFacts":
        """Derive whole-structure chemistry-readiness from one structure."""

        _, chemistry_readiness_facts = (
            derive_structure_coverage_and_chemistry_readiness_facts(
                structure,
                component_library=component_library,
            )
        )
        return chemistry_readiness_facts


def derive_structure_coverage_and_chemistry_readiness_facts(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence, ...
    ] = (),
) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
    """Derive split workflow facts from one shared structure observation."""
    from protrepair.state.chemistry_observation import StructureChemistryObservation

    return StructureChemistryObservation(
        structure,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
    ).whole_structure_facts()


def derive_projection_coverage_and_chemistry_readiness_facts(
    *,
    context_structure: ProteinStructure,
    residues: tuple[ResidueSite, ...],
    ligands: tuple[ResidueSite, ...] = (),
    component_library: ComponentLibrary | None = None,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence, ...
    ] = (),
    hydrogen_expectation_model: StructureHydrogenExpectationModel | None = None,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    chemistry_observation: "StructureChemistryObservation | None" = None,
) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
    """Observe context-bound residue identities, using validated expectations.

    Parameters
    ----------
    context_structure : ProteinStructure
        Complete chemical context.
    residues : tuple[ResidueSite, ...]
        Polymer selections matching the context scaffolds; equal-value copies
        are accepted.
    ligands : tuple[ResidueSite, ...]
        Retained-component selections matching the context scaffolds.
    component_library : ComponentLibrary or None
        Active library, or defaults.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit chemistry evidence.
    hydrogen_expectation_model : StructureHydrogenExpectationModel or None
        Expectations reused only when all derivation inputs match.
    allow_retained_non_polymer_rdkit_fallback : bool
        Whether retained chemistry inference is permitted.
    chemistry_observation : StructureChemistryObservation or None
        Shared observation requiring the exact context and interpretation inputs.
        When supplied, it owns expectations instead of the optional model hint.

    Returns
    -------
    tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]
        Observations for the selected identities.

    Raises
    ------
    KeyError
        A selected identity is missing from the context.
    ValueError
        A selected identity is repeated, a scaffold differs from the context,
        or the observation inputs do not match.
    """
    from protrepair.state.chemistry_observation import StructureChemistryObservation

    observation = chemistry_observation
    if observation is None:
        observation = StructureChemistryObservation(
            context_structure,
            component_library=component_library,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
            hydrogen_expectation_model=hydrogen_expectation_model,
            allow_retained_non_polymer_rdkit_fallback=allow_retained_non_polymer_rdkit_fallback,
        )
    else:
        observation.require_context(
            context_structure,
            component_library=component_library,
            retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
            allow_retained_non_polymer_rdkit_fallback=allow_retained_non_polymer_rdkit_fallback,
        )

    for residue in (*residues, *ligands):
        observation.require_residue(residue)

    return observation.projected_facts(
        tuple(residue.residue_id for residue in (*residues, *ligands))
    )


def aggregate_topology_availability_state(
    states: Iterable[TopologyAvailabilityState],
) -> TopologyAvailabilityState:
    """Reduce topology availability with unsupported and incomplete precedence.

    Parameters
    ----------
    states : Iterable[TopologyAvailabilityState]
        Availability values for one projection.

    Returns
    -------
    TopologyAvailabilityState
        Aggregate availability, retaining unsupported, absent and partial evidence.
    """

    topology_states = tuple(states)
    if any(state is TopologyAvailabilityState.UNSUPPORTED for state in topology_states):
        return TopologyAvailabilityState.UNSUPPORTED
    applicable_states = tuple(
        state
        for state in topology_states
        if state is not TopologyAvailabilityState.NOT_APPLICABLE
    )
    if not applicable_states:
        return TopologyAvailabilityState.NOT_APPLICABLE
    if all(state is TopologyAvailabilityState.PRESENT for state in applicable_states):
        return TopologyAvailabilityState.PRESENT
    return TopologyAvailabilityState.ABSENT


__all__ = [
    "StructureChemistryReadinessFacts",
    "StructureCoverageFacts",
    "derive_projection_coverage_and_chemistry_readiness_facts",
    "derive_structure_coverage_and_chemistry_readiness_facts",
]
