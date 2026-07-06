"""Scoped structure projection facts and observation read models."""

from dataclasses import dataclass, field
from typing import cast

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state.hydrogen_expectation import StructureHydrogenExpectationModel
from protrepair.state.scoped import CarrierScopedState
from protrepair.state.structure_axes import (
    BackboneHeavyAtomCompletenessState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
)
from protrepair.state.structure_boundary import (
    TerminalBoundaryObservation,
    terminal_boundary_observation,
)
from protrepair.state.structure_observation_rules import projection_chain_count
from protrepair.state.structure_residue import (
    ResidueProjectionFactRuntime,
    ResidueProjectionStateFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite

__all__ = [
    "ProteinStructureObservation",
    "StructureProjectionStateFacts",
    "derive_backbone_heavy_atom_completeness_state",
    "derive_component_support_state",
    "derive_hydrogen_applicability_state",
    "derive_hydrogen_coverage_state",
    "derive_sidechain_heavy_atom_completeness_state",
    "derive_stereochemistry_state",
]


def derive_component_support_state(
    residue_facts: tuple[ResidueProjectionStateFacts, ...],
) -> ComponentSupportState:
    """Derive support coverage over one coarse residue projection."""

    if any(
        residue_fact.component_support_fact.value
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
        for residue_fact in residue_facts
    ):
        return ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT

    return ComponentSupportState.ALL_SUPPORTED


def derive_backbone_heavy_atom_completeness_state(
    residue_facts: tuple[ResidueProjectionStateFacts, ...],
) -> BackboneHeavyAtomCompletenessState:
    """Derive backbone completeness over one coarse residue projection."""

    if any(
        residue_fact.backbone_heavy_atom_completeness_fact.value
        is BackboneHeavyAtomCompletenessState.INCOMPLETE
        for residue_fact in residue_facts
    ):
        return BackboneHeavyAtomCompletenessState.INCOMPLETE

    return BackboneHeavyAtomCompletenessState.COMPLETE


def derive_sidechain_heavy_atom_completeness_state(
    residue_facts: tuple[ResidueProjectionStateFacts, ...],
) -> SidechainHeavyAtomCompletenessState:
    """Derive side-chain completeness over one coarse residue projection."""

    if any(
        residue_fact.sidechain_heavy_atom_completeness_fact.value
        is SidechainHeavyAtomCompletenessState.INCOMPLETE
        for residue_fact in residue_facts
    ):
        return SidechainHeavyAtomCompletenessState.INCOMPLETE

    return SidechainHeavyAtomCompletenessState.COMPLETE


def derive_hydrogen_applicability_state(
    residue_facts: tuple[ResidueProjectionStateFacts, ...],
) -> HydrogenApplicabilityState:
    """Derive hydrogen applicability over one coarse residue projection."""

    if any(
        residue_fact.hydrogen_applicability_fact.value
        is HydrogenApplicabilityState.APPLICABLE
        for residue_fact in residue_facts
    ):
        return HydrogenApplicabilityState.APPLICABLE

    return HydrogenApplicabilityState.NOT_APPLICABLE


def derive_hydrogen_coverage_state(
    residue_facts: tuple[ResidueProjectionStateFacts, ...],
) -> HydrogenCoverageState:
    """Derive hydrogen coverage over one coarse residue projection."""

    applicable_residue_facts = tuple(
        residue_fact
        for residue_fact in residue_facts
        if residue_fact.hydrogen_applicability_fact.value
        is HydrogenApplicabilityState.APPLICABLE
    )
    if not applicable_residue_facts:
        return HydrogenCoverageState.COMPLETE

    if all(
        residue_fact.hydrogen_coverage_fact.value is HydrogenCoverageState.COMPLETE
        for residue_fact in applicable_residue_facts
    ):
        return HydrogenCoverageState.COMPLETE

    if all(
        residue_fact.hydrogen_coverage_fact.value is HydrogenCoverageState.NONE
        for residue_fact in applicable_residue_facts
    ):
        return HydrogenCoverageState.NONE

    return HydrogenCoverageState.PARTIAL


def derive_stereochemistry_state(
    residue_facts: tuple[ResidueProjectionStateFacts, ...],
) -> StereochemistryState:
    """Derive stereochemistry over one coarse residue projection."""

    if any(
        residue_fact.stereochemistry_fact.value is StereochemistryState.VIOLATED
        for residue_fact in residue_facts
    ):
        return StereochemistryState.VIOLATED

    if any(
        residue_fact.stereochemistry_fact.value is StereochemistryState.CONSISTENT
        for residue_fact in residue_facts
    ):
        return StereochemistryState.CONSISTENT

    return StereochemistryState.NOT_APPLICABLE


@dataclass(frozen=True, slots=True)
class StructureProjectionStateFacts:
    """Primitive scoped state facts over one structure or residue projection."""

    chain_count: int
    residue_count: int
    ligand_count: int
    atom_count: int
    hydrogen_atom_count: int
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
    terminal_boundary_observation: TerminalBoundaryObservation = field(
        default_factory=TerminalBoundaryObservation.empty
    )

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
            raise ValueError("structure projection facts require one shared carrier")
        if any(fact.scope != expected_scope for fact in shared_facts[1:]):
            raise ValueError(
                "structure projection facts require one shared projection scope"
            )
        if any(
            boundary_fact.carrier != expected_carrier
            for boundary_fact in self.terminal_boundary_observation.authenticity_facts
        ):
            raise ValueError(
                "terminal boundary authenticity facts must match the projection carrier"
            )
        if any(
            boundary_fact.carrier != expected_carrier
            for boundary_fact in self.terminal_boundary_observation.oxt_presence_facts
        ):
            raise ValueError(
                "terminal boundary OXT facts must match the projection carrier"
            )

    @property
    def carrier(self) -> ProteinStructure:
        """Return the canonical carrier shared by all primitive facts."""

        return self.component_support_fact.carrier

    @property
    def scope(self) -> WholeStructureScope | ResidueSetScope:
        """Return the semantic projection scope shared by all primitive facts."""

        return cast(
            WholeStructureScope | ResidueSetScope,
            self.component_support_fact.scope,
        )

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
        hydrogen_expectation_model: StructureHydrogenExpectationModel | None = None,
    ) -> "StructureProjectionStateFacts":
        """Derive primitive scoped facts over one whole structure."""

        return cls._from_projection(
            context_structure=structure,
            projection_scope=WholeStructureScope(),
            residues=tuple(structure.constitution.iter_residues(include_ligands=False)),
            ligands=structure.constitution.ligands,
            component_library=component_library,
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            hydrogen_expectation_model=hydrogen_expectation_model,
        )

    @classmethod
    def from_projection(
        cls,
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
    ) -> "StructureProjectionStateFacts":
        """Derive primitive scoped facts over one residue/ligand projection."""

        projected_residue_ids = tuple(
            residue.residue_id for residue in (*residues, *ligands)
        )
        projection_scope: WholeStructureScope | ResidueSetScope = (
            WholeStructureScope()
            if not projected_residue_ids
            else ResidueSetScope(residue_ids=projected_residue_ids)
        )
        return cls._from_projection(
            context_structure=context_structure,
            projection_scope=projection_scope,
            residues=residues,
            ligands=ligands,
            component_library=component_library,
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            hydrogen_expectation_model=hydrogen_expectation_model,
        )

    @classmethod
    def _from_projection(
        cls,
        *,
        context_structure: ProteinStructure,
        projection_scope: WholeStructureScope | ResidueSetScope,
        residues: tuple[ResidueSite, ...],
        ligands: tuple[ResidueSite, ...],
        component_library: ComponentLibrary | None,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ],
        hydrogen_expectation_model: StructureHydrogenExpectationModel | None,
    ) -> "StructureProjectionStateFacts":
        """Derive scoped facts over one explicit projection and semantic scope."""

        runtime = ResidueProjectionFactRuntime.from_context(
            context_structure,
            component_library=component_library,
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            hydrogen_expectation_model=hydrogen_expectation_model,
        )
        residue_facts = tuple(
            runtime.derive_residue_facts(residue) for residue in residues
        )
        atom_count = sum(len(residue.atom_sites) for residue in (*residues, *ligands))
        hydrogen_atom_count = sum(
            1
            for residue in (*residues, *ligands)
            for atom_site in residue.atom_sites
            if atom_site.element == "H"
        )
        terminal_observation = terminal_boundary_observation(
            context_structure,
            residues=residues,
        )
        return cls(
            chain_count=projection_chain_count(residues, ligands=ligands),
            residue_count=len(residues),
            ligand_count=len(ligands),
            atom_count=atom_count,
            hydrogen_atom_count=hydrogen_atom_count,
            component_support_fact=CarrierScopedState(
                carrier=context_structure,
                scope=projection_scope,
                value=derive_component_support_state(residue_facts),
            ),
            backbone_heavy_atom_completeness_fact=CarrierScopedState(
                carrier=context_structure,
                scope=projection_scope,
                value=derive_backbone_heavy_atom_completeness_state(residue_facts),
            ),
            sidechain_heavy_atom_completeness_fact=CarrierScopedState(
                carrier=context_structure,
                scope=projection_scope,
                value=derive_sidechain_heavy_atom_completeness_state(residue_facts),
            ),
            hydrogen_applicability_fact=CarrierScopedState(
                carrier=context_structure,
                scope=projection_scope,
                value=derive_hydrogen_applicability_state(residue_facts),
            ),
            hydrogen_coverage_fact=CarrierScopedState(
                carrier=context_structure,
                scope=projection_scope,
                value=derive_hydrogen_coverage_state(residue_facts),
            ),
            stereochemistry_fact=CarrierScopedState(
                carrier=context_structure,
                scope=projection_scope,
                value=derive_stereochemistry_state(residue_facts),
            ),
            terminal_boundary_observation=terminal_observation,
        )


@dataclass(frozen=True, slots=True)
class ProteinStructureObservation:
    """History-free observed summary about one canonical structure."""

    chain_count: int
    residue_count: int
    ligand_count: int
    atom_count: int
    hydrogen_atom_count: int
    component_support_state: ComponentSupportState
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState
    hydrogen_applicability_state: HydrogenApplicabilityState
    hydrogen_coverage_state: HydrogenCoverageState
    stereochemistry_state: StereochemistryState
    terminal_boundary_observation: TerminalBoundaryObservation = field(
        default_factory=TerminalBoundaryObservation.empty
    )

    def requires_heavy_atom_completion(self) -> bool:
        """Return whether any heavy-atom completeness axis remains incomplete."""

        return (
            self.backbone_heavy_atom_completeness_state.requires_completion()
            or self.sidechain_heavy_atom_completeness_state.requires_completion()
        )

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> "ProteinStructureObservation":
        """Derive one canonical structure observation from one structure."""

        return cls.from_facts(
            StructureProjectionStateFacts.from_structure(
                structure,
                component_library=component_library,
            )
        )

    @classmethod
    def from_projection(
        cls,
        *,
        context_structure: ProteinStructure,
        residues: tuple[ResidueSite, ...],
        ligands: tuple[ResidueSite, ...] = (),
        component_library: ComponentLibrary | None = None,
    ) -> "ProteinStructureObservation":
        """Derive one structure observation for one residue/ligand projection."""

        return cls.from_facts(
            StructureProjectionStateFacts.from_projection(
                context_structure=context_structure,
                residues=residues,
                ligands=ligands,
                component_library=component_library,
            )
        )

    @classmethod
    def from_facts(
        cls,
        facts: StructureProjectionStateFacts,
    ) -> "ProteinStructureObservation":
        """Rebuild one structure observation from primitive scoped facts."""

        return cls(
            chain_count=facts.chain_count,
            residue_count=facts.residue_count,
            ligand_count=facts.ligand_count,
            atom_count=facts.atom_count,
            hydrogen_atom_count=facts.hydrogen_atom_count,
            component_support_state=facts.component_support_fact.value,
            backbone_heavy_atom_completeness_state=(
                facts.backbone_heavy_atom_completeness_fact.value
            ),
            sidechain_heavy_atom_completeness_state=(
                facts.sidechain_heavy_atom_completeness_fact.value
            ),
            hydrogen_applicability_state=facts.hydrogen_applicability_fact.value,
            hydrogen_coverage_state=facts.hydrogen_coverage_fact.value,
            stereochemistry_state=facts.stereochemistry_fact.value,
            terminal_boundary_observation=facts.terminal_boundary_observation,
        )
