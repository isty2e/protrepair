"""Whole-structure coverage and chemistry-readiness facts."""

from collections.abc import Iterable
from dataclasses import dataclass

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
    ResidueProjectionFactRuntime,
    RetainedNonPolymerChemistryReadinessFact,
)
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite

__all__ = [
    "StructureChemistryReadinessFacts",
    "StructureCoverageFacts",
    "derive_projection_coverage_and_chemistry_readiness_facts",
    "derive_structure_coverage_and_chemistry_readiness_facts",
]


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
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
    """Derive split workflow facts from one shared structure pass."""

    blueprint_coverages = _structure_blueprint_coverages(structure)
    return _derive_projection_coverage_and_chemistry_readiness_facts(
        context_structure=structure,
        residues=tuple(structure.constitution.iter_residues(include_ligands=False)),
        ligands=structure.constitution.ligands,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        hydrogen_expectation_model=None,
        blueprint_coverages=blueprint_coverages,
    )


def derive_projection_coverage_and_chemistry_readiness_facts(
    *,
    context_structure: ProteinStructure,
    residues: tuple[ResidueSite, ...],
    ligands: tuple[ResidueSite, ...] = (),
    component_library: ComponentLibrary | None = None,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    hydrogen_expectation_model: StructureHydrogenExpectationModel | None = None,
) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
    """Derive coverage and chemistry-readiness facts over one explicit projection."""

    return _derive_projection_coverage_and_chemistry_readiness_facts(
        context_structure=context_structure,
        residues=residues,
        ligands=ligands,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        hydrogen_expectation_model=hydrogen_expectation_model,
        blueprint_coverages=(),
    )


def _derive_projection_coverage_and_chemistry_readiness_facts(
    *,
    context_structure: ProteinStructure,
    residues: tuple[ResidueSite, ...],
    ligands: tuple[ResidueSite, ...],
    component_library: ComponentLibrary | None,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ],
    hydrogen_expectation_model: StructureHydrogenExpectationModel | None,
    blueprint_coverages: tuple[StructureBlueprintCoverage, ...],
) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
    """Derive split workflow facts over one explicit residue and ligand projection."""

    runtime = ResidueProjectionFactRuntime.from_context(
        context_structure,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        hydrogen_expectation_model=hydrogen_expectation_model,
    )
    residue_coverage_facts: list[ResidueCoverageFacts] = []
    residue_chemistry_readiness_facts: list[ResidueChemistryReadinessFacts] = []
    retained_non_polymer_chemistry_readiness_facts: list[
        RetainedNonPolymerChemistryReadinessFact
    ] = []
    for residue in residues:
        coverage_facts = runtime.derive_residue_coverage_facts(residue)
        residue_coverage_facts.append(coverage_facts)
        residue_chemistry_readiness_facts.append(
            runtime.derive_residue_chemistry_readiness_facts(
                residue,
                coverage_facts=coverage_facts,
            )
        )
    for retained_non_polymer in ligands:
        retained_non_polymer_chemistry_readiness_facts.append(
            runtime.derive_retained_non_polymer_chemistry_readiness_fact(
                retained_non_polymer
            )
        )

    coverage_residue_facts = tuple(residue_coverage_facts)
    chemistry_residue_facts = tuple(residue_chemistry_readiness_facts)
    coverage_facts = StructureCoverageFacts(
        carrier=context_structure,
        residue_facts=coverage_residue_facts,
        blueprint_coverages=blueprint_coverages,
        coverage_gaps=tuple(
            coverage_gap
            for coverage in blueprint_coverages
            for coverage_gap in coverage.coverage_gaps
        ),
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.INCOMPLETE
            if any(
                residue_fact.requires_backbone_completion()
                for residue_fact in coverage_residue_facts
            )
            else BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.INCOMPLETE
            if any(
                residue_fact.requires_sidechain_completion()
                for residue_fact in coverage_residue_facts
            )
            else SidechainHeavyAtomCompletenessState.COMPLETE
        ),
    )

    component_support = (
        ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
        if any(
            not residue_fact.is_supported()
            for residue_fact in chemistry_residue_facts
        )
        else ComponentSupportState.ALL_SUPPORTED
    )
    heavy_atom_topology_availability_state = _aggregate_topology_availability_state(
        residue_fact.heavy_atom_topology_availability_state
        for residue_fact in chemistry_residue_facts
    )
    h_applicability_state = (
        HydrogenApplicabilityState.APPLICABLE
        if any(
            residue_fact.hydrogen_applicability_state
            is HydrogenApplicabilityState.APPLICABLE
            for residue_fact in chemistry_residue_facts
        )
        else HydrogenApplicabilityState.NOT_APPLICABLE
    )
    applicable_residue_facts = tuple(
        residue_fact
        for residue_fact in chemistry_residue_facts
        if residue_fact.hydrogen_applicability_state
        is HydrogenApplicabilityState.APPLICABLE
    )
    if not applicable_residue_facts:
        h_coverage_state = HydrogenCoverageState.COMPLETE
        hydrogen_topology_availability_state = TopologyAvailabilityState.NOT_APPLICABLE
    else:
        if all(
            residue_fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
            for residue_fact in applicable_residue_facts
        ):
            h_coverage_state = HydrogenCoverageState.COMPLETE
        elif all(
            residue_fact.hydrogen_coverage_state is HydrogenCoverageState.NONE
            for residue_fact in applicable_residue_facts
        ):
            h_coverage_state = HydrogenCoverageState.NONE
        else:
            h_coverage_state = HydrogenCoverageState.PARTIAL

        hydrogen_topology_availability_state = _aggregate_topology_availability_state(
            residue_fact.hydrogen_topology_availability_state
            for residue_fact in applicable_residue_facts
        )

    chemistry_readiness_facts = StructureChemistryReadinessFacts(
        carrier=context_structure,
        residue_facts=chemistry_residue_facts,
        retained_non_polymer_facts=tuple(
            retained_non_polymer_chemistry_readiness_facts
        ),
        component_support_state=component_support,
        heavy_atom_topology_availability_state=(
            heavy_atom_topology_availability_state
        ),
        hydrogen_topology_availability_state=(
            hydrogen_topology_availability_state
        ),
        hydrogen_applicability_state=h_applicability_state,
        hydrogen_coverage_state=h_coverage_state,
    )
    return coverage_facts, chemistry_readiness_facts


def _aggregate_topology_availability_state(
    states: Iterable[TopologyAvailabilityState],
) -> TopologyAvailabilityState:
    """Aggregate residue-local topology availability without hiding unsupported."""

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


def _structure_blueprint_coverages(
    structure: ProteinStructure,
) -> tuple[StructureBlueprintCoverage, ...]:
    """Return structure-local blueprint coverages for attached blueprint chains."""

    if structure.polymer_blueprint is None:
        return ()

    coverages: list[StructureBlueprintCoverage] = []
    for chain in structure.constitution.chains:
        try:
            coverages.append(
                StructureBlueprintCoverage.from_structure(
                    structure,
                    chain.chain_id,
                )
            )
        except KeyError:
            continue

    return tuple(coverages)
