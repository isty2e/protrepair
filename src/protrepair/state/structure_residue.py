"""Residue-local structure state facts and derivation runtime."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import cast

from protrepair.chemistry import (
    ComponentLibrary,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.topology import (
    polymer_context_hydrogen_anchor_definitions,
    polymer_template_hydrogen_bond_definitions_for_names,
    template_heavy_bond_definitions_for_present_atoms,
    template_hydrogen_bond_definitions_for_names,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_heavy_atom_elements,
    retained_non_polymer_evidence_heavy_bond_definitions,
    retained_non_polymer_evidence_hydrogen_bond_definitions,
)
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    retained_non_polymer_rdkit_fallback_heavy_bond_definitions,
    retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions_for_names,
    retained_non_polymer_rdkit_fallback_hydrogenated_molecule,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
    evidence_by_residue_id,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.scope import ResidueSetScope
from protrepair.state.hydrogen_expectation import (
    RetainedNonPolymerChemistryEvidenceSource,
    StructureHydrogenExpectationModel,
    derive_structure_hydrogen_expectation_model,
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
from protrepair.state.structure_observation_rules import (
    backbone_heavy_atom_completeness_state,
    component_support_state,
    hydrogen_applicability_state,
    hydrogen_coverage_state,
    sidechain_heavy_atom_completeness_state,
    stereochemistry_state,
)
from protrepair.state.topology import (
    TopologyAvailabilityState,
    residue_bond_topology_availability_state,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex

__all__ = [
    "ResidueChemistryReadinessFacts",
    "ResidueCompletionStateFacts",
    "ResidueCoverageFacts",
    "ResidueProjectionFactRuntime",
    "ResidueProjectionStateFacts",
    "RetainedNonPolymerChemistryReadinessFact",
]


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
        """Derive primitive singleton residue facts from one observed residue."""

        runtime = ResidueProjectionFactRuntime.from_context(
            context_structure,
            component_library=component_library,
        )
        return runtime.derive_residue_facts(residue)


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

    def is_supported(self) -> bool:
        """Return whether canonical chemistry support is available."""

        return self.component_support_state.is_fully_supported()

    def needs_hydrogenation(self) -> bool:
        """Return whether hydrogen completion may still add missing hydrogens."""

        return self.hydrogen_coverage_state.needs_hydrogenation()

    def requires_hydrogen_completion(self) -> bool:
        """Return whether retained non-polymer hydrogen completion is required."""

        return (
            self.is_supported()
            and self.hydrogen_applicability_state
            is HydrogenApplicabilityState.APPLICABLE
            and self.needs_hydrogenation()
        )


def _polymer_heavy_topology_availability_state(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    component_library: ComponentLibrary,
    support_state: ComponentSupportState,
    coverage_facts: ResidueCoverageFacts,
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]],
) -> TopologyAvailabilityState:
    """Return polymer heavy topology availability from expected template bonds."""

    if not support_state.is_fully_supported():
        return TopologyAvailabilityState.UNSUPPORTED

    if (
        coverage_facts.requires_backbone_completion()
        or coverage_facts.requires_sidechain_completion()
    ):
        return TopologyAvailabilityState.ABSENT

    template = component_library.get(residue.component_id)
    if template is None:
        return TopologyAvailabilityState.UNSUPPORTED

    return residue_bond_topology_availability_state(
        structure,
        residue,
        expected_bond_definitions=template_heavy_bond_definitions_for_present_atoms(
            residue,
            template=template,
        ),
        empty_state=TopologyAvailabilityState.PRESENT,
        covalent_like_endpoint_pairs=covalent_like_endpoint_pairs,
    )


def _polymer_hydrogen_topology_availability_state(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    component_library: ComponentLibrary,
    expected_hydrogen_atom_names: tuple[str, ...],
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]],
) -> TopologyAvailabilityState:
    """Return polymer hydrogen topology availability from expected H anchors."""

    template = component_library.get(residue.component_id)
    if template is None:
        return TopologyAvailabilityState.UNSUPPORTED

    if not expected_hydrogen_atom_names:
        return TopologyAvailabilityState.ABSENT

    return residue_bond_topology_availability_state(
        structure,
        residue,
        expected_bond_definitions=(
            *polymer_template_hydrogen_bond_definitions_for_names(
                template,
                hydrogen_atom_names=expected_hydrogen_atom_names,
            ),
            *(
                anchor.bond_definition
                for anchor in polymer_context_hydrogen_anchor_definitions(
                    component_id=residue.component_id,
                    hydrogen_atom_names=expected_hydrogen_atom_names,
                )
            ),
        ),
        empty_state=TopologyAvailabilityState.ABSENT,
        covalent_like_endpoint_pairs=covalent_like_endpoint_pairs,
    )


def _retained_non_polymer_hydrogen_topology_availability_state(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    hydrogen_expectation_source: RetainedNonPolymerChemistryEvidenceSource,
    evidence: RetainedNonPolymerChemistryEvidence | None,
    template: ResidueTemplate | None,
    expected_hydrogen_atom_names: tuple[str, ...],
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]],
) -> TopologyAvailabilityState:
    """Return retained ligand H topology availability from expected anchors."""

    bond_definitions = _retained_non_polymer_hydrogen_bond_definitions(
        structure,
        residue,
        hydrogen_expectation_source=hydrogen_expectation_source,
        evidence=evidence,
        template=template,
        expected_hydrogen_atom_names=expected_hydrogen_atom_names,
    )
    return residue_bond_topology_availability_state(
        structure,
        residue,
        expected_bond_definitions=bond_definitions,
        empty_state=TopologyAvailabilityState.ABSENT,
        covalent_like_endpoint_pairs=covalent_like_endpoint_pairs,
    )


def _retained_non_polymer_hydrogen_bond_definitions(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    hydrogen_expectation_source: RetainedNonPolymerChemistryEvidenceSource,
    evidence: RetainedNonPolymerChemistryEvidence | None,
    template: ResidueTemplate | None,
    expected_hydrogen_atom_names: tuple[str, ...],
) -> tuple[BondDefinition, ...]:
    """Return retained ligand expected H-anchor bond definitions."""

    if (
        hydrogen_expectation_source
        is RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
        and evidence is not None
    ):
        return retained_non_polymer_evidence_hydrogen_bond_definitions(evidence)

    if (
        hydrogen_expectation_source
        is RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
        and template is not None
    ):
        return template_hydrogen_bond_definitions_for_names(
            template,
            hydrogen_atom_names=expected_hydrogen_atom_names,
        )

    if (
        hydrogen_expectation_source
        is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    ):
        try:
            residue_index = structure.constitution.residue_index(residue.residue_id)
            hydrogenated_molecule = (
                retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
                    residue,
                    structure.residue_geometry(residue_index),
                    formal_charge_by_atom_name=dict(
                        structure.residue_formal_charge_by_atom_name(residue_index)
                    ),
                )
            )
        except (RdkitUnavailableError, RuntimeError, ValueError):
            return ()

        return (
            retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions_for_names(
                hydrogenated_molecule,
                hydrogen_atom_names=expected_hydrogen_atom_names,
            )
        )

    return ()


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


@dataclass(frozen=True, slots=True)
class ResidueProjectionFactRuntime:
    """Derivation runtime for residue-projection state facts."""

    context_structure: ProteinStructure
    component_library: ComponentLibrary
    hydrogen_expectation_model: StructureHydrogenExpectationModel
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]]
    retained_non_polymer_evidence_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryEvidence,
    ]

    @classmethod
    def from_context(
        cls,
        context_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
    ) -> "ResidueProjectionFactRuntime":
        """Build one derivation runtime for a shared context structure."""

        library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        evidence_map = evidence_by_residue_id(retained_non_polymer_chemistry_evidence)
        hydrogen_expectation_model = derive_structure_hydrogen_expectation_model(
            context_structure,
            component_library=library,
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
        )

        return cls(
            context_structure=context_structure,
            component_library=library,
            hydrogen_expectation_model=hydrogen_expectation_model,
            covalent_like_endpoint_pairs=(
                context_structure.topology.covalent_like_endpoint_pairs()
            ),
            retained_non_polymer_evidence_by_residue_id=evidence_map,
        )

    def derive_residue_coverage_facts(
        self,
        residue: ResidueSite,
    ) -> ResidueCoverageFacts:
        """Derive coverage-relevant singleton residue facts from one residue."""

        return ResidueCoverageFacts(
            residue_id=residue.residue_id,
            backbone_heavy_atom_completeness_state=(
                backbone_heavy_atom_completeness_state(
                    (residue,),
                    component_library=self.component_library,
                )
            ),
            sidechain_heavy_atom_completeness_state=(
                sidechain_heavy_atom_completeness_state(
                    (residue,),
                    component_library=self.component_library,
                )
            ),
        )

    def derive_residue_chemistry_readiness_facts(
        self,
        residue: ResidueSite,
        *,
        coverage_facts: ResidueCoverageFacts,
    ) -> ResidueChemistryReadinessFacts:
        """Derive chemistry-readiness singleton residue facts from one residue."""

        support_state = component_support_state(
            (residue,),
            component_library=self.component_library,
        )
        h_applicability_state = hydrogen_applicability_state(
            (residue,),
            context_structure=self.context_structure,
            component_library=self.component_library,
            expected_hydrogen_atom_names_by_residue=(
                self.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue
            ),
        )
        h_coverage_state = hydrogen_coverage_state(
            self.context_structure,
            residues=(residue,),
            component_library=self.component_library,
            expected_hydrogen_atom_names_by_residue=(
                self.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue
            ),
        )
        heavy_atom_topology_availability_state = (
            _polymer_heavy_topology_availability_state(
                self.context_structure,
                residue,
                component_library=self.component_library,
                support_state=support_state,
                coverage_facts=coverage_facts,
                covalent_like_endpoint_pairs=self.covalent_like_endpoint_pairs,
            )
        )
        if heavy_atom_topology_availability_state.is_unavailable():
            hydrogen_topology_availability_state = (
                heavy_atom_topology_availability_state
            )
        elif h_applicability_state is HydrogenApplicabilityState.NOT_APPLICABLE:
            hydrogen_topology_availability_state = (
                TopologyAvailabilityState.NOT_APPLICABLE
            )
        elif h_coverage_state is HydrogenCoverageState.COMPLETE:
            hydrogen_topology_availability_state = (
                _polymer_hydrogen_topology_availability_state(
                    self.context_structure,
                    residue,
                    component_library=self.component_library,
                    expected_hydrogen_atom_names=(
                        self.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue.get(
                            residue.residue_id,
                            (),
                        )
                    ),
                    covalent_like_endpoint_pairs=self.covalent_like_endpoint_pairs,
                )
            )
        else:
            hydrogen_topology_availability_state = TopologyAvailabilityState.ABSENT

        return ResidueChemistryReadinessFacts(
            residue_id=residue.residue_id,
            component_support_state=support_state,
            heavy_atom_topology_availability_state=(
                heavy_atom_topology_availability_state
            ),
            hydrogen_topology_availability_state=(
                hydrogen_topology_availability_state
            ),
            hydrogen_applicability_state=h_applicability_state,
            hydrogen_coverage_state=h_coverage_state,
        )

    def derive_retained_non_polymer_chemistry_readiness_fact(
        self,
        residue: ResidueSite,
    ) -> RetainedNonPolymerChemistryReadinessFact:
        """Derive chemistry-readiness truth for one retained non-polymer component."""

        evidence = self.retained_non_polymer_evidence_by_residue_id.get(
            residue.residue_id
        )
        hydrogen_expectation_resolution = (
            self.hydrogen_expectation_model.resolution_for_retained_non_polymer(
                residue.residue_id
            )
        )
        template = self.component_library.get(residue.component_id)
        hydrogen_expectation_source = hydrogen_expectation_resolution.source
        selected_expected_hydrogen_atom_names = (
            hydrogen_expectation_resolution.expected_hydrogen_atom_names
        )
        expected_hydrogen_atom_names_by_residue = (
            {}
            if not selected_expected_hydrogen_atom_names
            else {residue.residue_id: selected_expected_hydrogen_atom_names}
        )
        h_applicability_state = hydrogen_applicability_state(
            (residue,),
            context_structure=self.context_structure,
            component_library=self.component_library,
            expected_hydrogen_atom_names_by_residue=(
                expected_hydrogen_atom_names_by_residue
            ),
        )
        h_coverage_state = hydrogen_coverage_state(
            self.context_structure,
            residues=(residue,),
            component_library=self.component_library,
            expected_hydrogen_atom_names_by_residue=(
                expected_hydrogen_atom_names_by_residue
            ),
        )
        hydrogen_topology_availability_state = TopologyAvailabilityState.ABSENT
        if evidence is not None:
            heavy_topology_source = (
                RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
            )
            present_heavy_atom_names = {
                atom_site.name
                for atom_site in residue.atom_sites
                if atom_site.element != "H"
            }
            expected_heavy_atom_elements = (
                retained_non_polymer_evidence_heavy_atom_elements(evidence)
            )
            heavy_atom_names_present = set(evidence.heavy_atom_names).issubset(
                present_heavy_atom_names
            )
            heavy_atom_elements_match = all(
                residue.atom_site(atom_name).element == expected_element
                for atom_name, expected_element in zip(
                    evidence.heavy_atom_names,
                    expected_heavy_atom_elements,
                    strict=True,
                )
                if residue.has_atom_site(atom_name)
            )
            heavy_atom_topology_availability_state = (
                residue_bond_topology_availability_state(
                    self.context_structure,
                    residue,
                    expected_bond_definitions=(
                        retained_non_polymer_evidence_heavy_bond_definitions(evidence)
                    ),
                    empty_state=TopologyAvailabilityState.PRESENT,
                    covalent_like_endpoint_pairs=self.covalent_like_endpoint_pairs,
                )
                if heavy_atom_names_present and heavy_atom_elements_match
                else TopologyAvailabilityState.ABSENT
            )
        elif template is not None:
            heavy_topology_source = RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
            present_heavy_atom_names = {
                atom_site.name
                for atom_site in residue.atom_sites
                if atom_site.element != "H"
            }
            expected_heavy_atom_names = set(template.expected_heavy_atom_names())
            heavy_atom_topology_availability_state = (
                residue_bond_topology_availability_state(
                    self.context_structure,
                    residue,
                    expected_bond_definitions=(
                        template_heavy_bond_definitions_for_present_atoms(
                            residue,
                            template=template,
                        )
                    ),
                    empty_state=TopologyAvailabilityState.PRESENT,
                    covalent_like_endpoint_pairs=self.covalent_like_endpoint_pairs,
                )
                if expected_heavy_atom_names.issubset(present_heavy_atom_names)
                else TopologyAvailabilityState.ABSENT
            )
        elif (
            hydrogen_expectation_resolution.source
            is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
        ):
            heavy_topology_source = (
                RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
            )
            residue_index = self.context_structure.constitution.residue_index(
                residue.residue_id
            )
            try:
                fallback_heavy_bonds = (
                    retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
                        residue,
                        self.context_structure.residue_geometry(residue_index),
                        formal_charge_by_atom_name=dict(
                            self.context_structure.residue_formal_charge_by_atom_name(
                                residue_index
                            )
                        ),
                    )
                )
            except (RdkitUnavailableError, RuntimeError, ValueError):
                heavy_atom_topology_availability_state = (
                    TopologyAvailabilityState.ABSENT
                )
            else:
                heavy_atom_topology_availability_state = (
                    residue_bond_topology_availability_state(
                        self.context_structure,
                        residue,
                        expected_bond_definitions=fallback_heavy_bonds,
                        empty_state=TopologyAvailabilityState.PRESENT,
                        covalent_like_endpoint_pairs=(
                            self.covalent_like_endpoint_pairs
                        ),
                    )
                )
        else:
            heavy_topology_source = (
                RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
            )
            heavy_atom_topology_availability_state = TopologyAvailabilityState.ABSENT
            hydrogen_topology_availability_state = TopologyAvailabilityState.ABSENT

        h_source_resolved = hydrogen_expectation_source.is_resolved()
        heavy_source_resolved = heavy_topology_source.is_resolved()
        support_state = (
            ComponentSupportState.ALL_SUPPORTED
            if heavy_source_resolved and h_source_resolved
            else ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
        )
        if heavy_source_resolved and h_source_resolved:
            if (
                heavy_atom_topology_availability_state
                is TopologyAvailabilityState.ABSENT
            ):
                hydrogen_topology_availability_state = (
                    TopologyAvailabilityState.ABSENT
                )
            elif h_applicability_state is HydrogenApplicabilityState.NOT_APPLICABLE:
                hydrogen_topology_availability_state = (
                    TopologyAvailabilityState.NOT_APPLICABLE
                )
            elif h_coverage_state is HydrogenCoverageState.COMPLETE:
                hydrogen_topology_availability_state = (
                    _retained_non_polymer_hydrogen_topology_availability_state(
                        self.context_structure,
                        residue,
                        hydrogen_expectation_source=hydrogen_expectation_source,
                        evidence=evidence,
                        template=template,
                        expected_hydrogen_atom_names=(
                            selected_expected_hydrogen_atom_names
                        ),
                        covalent_like_endpoint_pairs=(
                            self.covalent_like_endpoint_pairs
                        ),
                    )
                )
            else:
                hydrogen_topology_availability_state = (
                    TopologyAvailabilityState.ABSENT
                )

        return RetainedNonPolymerChemistryReadinessFact(
            residue_id=residue.residue_id,
            component_id=residue.component_id,
            component_support_state=support_state,
            heavy_topology_source=heavy_topology_source,
            hydrogen_expectation_source=hydrogen_expectation_source,
            heavy_atom_topology_availability_state=(
                heavy_atom_topology_availability_state
            ),
            hydrogen_topology_availability_state=(
                hydrogen_topology_availability_state
            ),
            hydrogen_applicability_state=h_applicability_state,
            hydrogen_coverage_state=h_coverage_state,
        )

    def derive_residue_completion_facts(
        self,
        residue: ResidueSite,
    ) -> ResidueCompletionStateFacts:
        """Derive transitional residue completion facts from one residue."""

        coverage_facts = self.derive_residue_coverage_facts(residue)
        chemistry_readiness_facts = self.derive_residue_chemistry_readiness_facts(
            residue,
            coverage_facts=coverage_facts,
        )
        return ResidueCompletionStateFacts(
            coverage_facts=coverage_facts,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )

    def derive_residue_facts(
        self,
        residue: ResidueSite,
    ) -> ResidueProjectionStateFacts:
        """Derive primitive singleton residue facts from one observed residue."""

        completion_facts = self.derive_residue_completion_facts(residue)
        residue_scope = ResidueSetScope(residue_ids=(residue.residue_id,))
        stereo_state = stereochemistry_state(
            self.context_structure,
            residues=(residue,),
            component_library=self.component_library,
        )
        return ResidueProjectionStateFacts(
            component_support_fact=CarrierScopedState(
                carrier=self.context_structure,
                scope=residue_scope,
                value=completion_facts.component_support_state,
            ),
            backbone_heavy_atom_completeness_fact=CarrierScopedState(
                carrier=self.context_structure,
                scope=residue_scope,
                value=completion_facts.backbone_heavy_atom_completeness_state,
            ),
            sidechain_heavy_atom_completeness_fact=CarrierScopedState(
                carrier=self.context_structure,
                scope=residue_scope,
                value=completion_facts.sidechain_heavy_atom_completeness_state,
            ),
            hydrogen_applicability_fact=CarrierScopedState(
                carrier=self.context_structure,
                scope=residue_scope,
                value=completion_facts.hydrogen_applicability_state,
            ),
            hydrogen_coverage_fact=CarrierScopedState(
                carrier=self.context_structure,
                scope=residue_scope,
                value=completion_facts.hydrogen_coverage_state,
            ),
            stereochemistry_fact=CarrierScopedState(
                carrier=self.context_structure,
                scope=residue_scope,
                value=stereo_state,
            ),
        )
