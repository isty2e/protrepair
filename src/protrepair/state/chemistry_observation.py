"""Snapshot-bound chemistry observation and projection queries."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from protrepair.chemistry import (
    ComponentLibrary,
    build_default_component_library,
)
from protrepair.chemistry.component.topology import (
    template_heavy_bond_definitions_for_present_atoms,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import (
    PolymerMicrostatePreparation,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
    evidence_by_residue_id,
)
from protrepair.relation.blueprint import StructureBlueprintCoverage
from protrepair.scope import ResidueSetScope
from protrepair.state.hydrogen_expectation import (
    StructureHydrogenExpectationModel,
)
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    RetainedNonPolymerChemistryResolution,
)
from protrepair.state.scoped import CarrierScopedState
from protrepair.state.structure_axes import (
    BackboneHeavyAtomCompletenessState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.state.structure_observation_rules import (
    backbone_heavy_atom_completeness_state,
    component_support_state,
    hydrogen_applicability_state,
    hydrogen_coverage_state,
    sidechain_heavy_atom_completeness_state,
    stereochemistry_state,
)
from protrepair.state.structure_readiness import (
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    aggregate_topology_availability_state,
)
from protrepair.state.structure_residue import (
    ResidueChemistryReadinessFacts,
    ResidueCompletionStateFacts,
    ResidueCoverageFacts,
    ResidueProjectionStateFacts,
    RetainedNonPolymerChemistryReadinessFact,
)
from protrepair.state.topology import (
    TopologyAvailabilityState,
    residue_bond_topology_availability_state,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex


@dataclass(frozen=True, slots=True, init=False, eq=False)
class StructureChemistryObservation:
    """Chemistry observations under one immutable structure and input binding.

    Parameters
    ----------
    context_structure : ProteinStructure
        Complete context; projecting residues never creates chemical termini.
    component_library : ComponentLibrary or None
        Active component definitions, or the default library.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit retained-component evidence.
    allow_retained_non_polymer_rdkit_fallback : bool
        Whether template-less retained chemistry may use RDKit inference.
    hydrogen_expectation_model : StructureHydrogenExpectationModel or None
        Existing expectations, reused only when their derivation inputs match.

    Notes
    -----
    Queries own their projection and derive from the bound context, not injected
    fact tuples. Private memoization lasts only as long as this observation.
    Workflow history, preparation requests and score policy are not state facts.
    """

    context_structure: ProteinStructure
    component_library: ComponentLibrary
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence, ...
    ]
    allow_retained_non_polymer_rdkit_fallback: bool
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]]
    retained_non_polymer_evidence_by_residue_id: Mapping[
        ResidueId, RetainedNonPolymerChemistryEvidence
    ]
    _hydrogen_expectation_model: StructureHydrogenExpectationModel | None
    _whole_facts: tuple[StructureCoverageFacts, StructureChemistryReadinessFacts] | None
    _residue_facts: dict[ResidueId, ResidueCompletionStateFacts]
    _retained_facts: dict[ResidueId, RetainedNonPolymerChemistryReadinessFact]

    def __init__(
        self,
        context_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence, ...
        ] = (),
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        hydrogen_expectation_model: StructureHydrogenExpectationModel | None = None,
    ) -> None:
        library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        evidence = evidence_by_residue_id(retained_non_polymer_chemistry_evidence)
        canonical_evidence = tuple(evidence.values())
        if (
            hydrogen_expectation_model is not None
            and not hydrogen_expectation_model.matches_inputs(
                context_structure,
                component_library=library,
                retained_non_polymer_chemistry_evidence=canonical_evidence,
                allow_retained_non_polymer_rdkit_fallback=allow_retained_non_polymer_rdkit_fallback,
            )
        ):
            hydrogen_expectation_model = None

        object.__setattr__(self, "context_structure", context_structure)
        object.__setattr__(self, "component_library", library)
        object.__setattr__(
            self, "retained_non_polymer_chemistry_evidence", canonical_evidence
        )
        object.__setattr__(
            self,
            "allow_retained_non_polymer_rdkit_fallback",
            allow_retained_non_polymer_rdkit_fallback,
        )
        object.__setattr__(
            self, "retained_non_polymer_evidence_by_residue_id", evidence
        )
        object.__setattr__(
            self,
            "covalent_like_endpoint_pairs",
            context_structure.topology.covalent_like_endpoint_pairs(),
        )
        object.__setattr__(
            self, "_hydrogen_expectation_model", hydrogen_expectation_model
        )
        object.__setattr__(self, "_whole_facts", None)
        object.__setattr__(self, "_residue_facts", {})
        object.__setattr__(self, "_retained_facts", {})

    def for_structure(
        self, structure: ProteinStructure
    ) -> "StructureChemistryObservation":
        """Observe a carrier under the same library, evidence and policy.

        Parameters
        ----------
        structure : ProteinStructure
            Exact carrier whose facts will be queried.

        Returns
        -------
        StructureChemistryObservation
            This observation for the same carrier; otherwise a new observation.
            Residue facts are never transferred. Existing H expectations are only
            reused after their complete derivation inputs have been checked.
        """
        if structure is self.context_structure:
            return self

        return StructureChemistryObservation(
            structure,
            component_library=self.component_library,
            retained_non_polymer_chemistry_evidence=self.retained_non_polymer_chemistry_evidence,
            allow_retained_non_polymer_rdkit_fallback=self.allow_retained_non_polymer_rdkit_fallback,
            hydrogen_expectation_model=self._hydrogen_expectation_model,
        )

    def require_context(
        self,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence, ...
        ] = (),
        allow_retained_non_polymer_rdkit_fallback: bool = True,
    ) -> None:
        """Reject consumption under another snapshot or interpretation.

        Parameters
        ----------
        structure : ProteinStructure
            Exact carrier being consumed.
        component_library : ComponentLibrary or None
            Active library, or the default library.
        retained_non_polymer_chemistry_evidence : tuple
            Evidence for the consuming operation.
        allow_retained_non_polymer_rdkit_fallback : bool
            Fallback permission for that operation.

        Raises
        ------
        ValueError
            Any bound input differs. Construct a new observation instead.
        """
        library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        if (
            self.context_structure is not structure
            or self.component_library is not library
            or self.retained_non_polymer_chemistry_evidence
            != retained_non_polymer_chemistry_evidence
            or self.allow_retained_non_polymer_rdkit_fallback
            != allow_retained_non_polymer_rdkit_fallback
        ):
            raise ValueError(
                "chemistry observation requires its exact context "
                "and interpretation inputs"
            )

    @property
    def hydrogen_expectation_model(self) -> StructureHydrogenExpectationModel:
        """Return expectations, deriving them on the first chemistry query."""
        model = self._hydrogen_expectation_model
        if model is None:
            model = StructureHydrogenExpectationModel(
                self.context_structure,
                component_library=self.component_library,
                retained_non_polymer_chemistry_evidence=self.retained_non_polymer_chemistry_evidence,
                allow_retained_non_polymer_rdkit_fallback=self.allow_retained_non_polymer_rdkit_fallback,
            )
            object.__setattr__(self, "_hydrogen_expectation_model", model)
        return model

    def _residue(self, residue_id: ResidueId) -> ResidueSite:
        residue = self.context_structure.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise KeyError(residue_id)
        return residue

    def whole_structure_facts(
        self,
    ) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
        """Observe all residues and blueprint gaps without dropping scope.

        Returns
        -------
        tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]
            Whole-context observations, reused within this owner.
        """
        facts = self._whole_facts
        if facts is None:
            facts = self._projection_facts(
                residues=tuple(
                    self.context_structure.constitution.iter_residues(
                        include_ligands=False
                    )
                ),
                ligands=self.context_structure.constitution.ligands,
                blueprint_coverages=_structure_blueprint_coverages(
                    self.context_structure
                ),
            )
            object.__setattr__(self, "_whole_facts", facts)
        return facts

    def projected_facts(
        self, residue_ids: tuple[ResidueId, ...]
    ) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
        """Observe selected identities within the complete chemical context.

        Parameters
        ----------
        residue_ids : tuple[ResidueId, ...]
            Distinct context identities. An empty tuple selects nothing.

        Returns
        -------
        tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]
            Selected-residue facts, without whole-structure blueprint gaps.

        Raises
        ------
        KeyError
            A selected residue is not present in the context.
        ValueError
            A residue identity is repeated.
        """
        if len(set(residue_ids)) != len(residue_ids):
            raise ValueError(
                "chemistry projection requires distinct residue identities"
            )
        selected = tuple(self._residue(residue_id) for residue_id in residue_ids)
        return self._projection_facts(
            residues=tuple(residue for residue in selected if not residue.is_hetero),
            ligands=tuple(residue for residue in selected if residue.is_hetero),
            blueprint_coverages=(),
        )

    def _retained_readiness(
        self, residue: ResidueSite
    ) -> RetainedNonPolymerChemistryReadinessFact:
        cached = self._retained_facts.get(residue.residue_id)
        if cached is None:
            cached = self._derive_retained_non_polymer_chemistry_readiness_fact(residue)
            self._retained_facts[residue.residue_id] = cached
        return cached

    def _derive_residue_coverage_facts(
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

    def _derive_residue_chemistry_readiness_facts(
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
            attachment_state = heavy_atom_topology_availability_state
        elif h_applicability_state is HydrogenApplicabilityState.NOT_APPLICABLE:
            attachment_state = TopologyAvailabilityState.NOT_APPLICABLE
        elif h_coverage_state is HydrogenCoverageState.COMPLETE:
            try:
                attachment_state = _polymer_hydrogen_topology_availability_state(
                    residue,
                    component_library=self.component_library,
                    expected_hydrogen_atom_names=(
                        self.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue.get(
                            residue.residue_id,
                            (),
                        )
                    ),
                    preparation=self.hydrogen_expectation_model.polymer_preparation,
                )
            except ValueError:
                attachment_state = TopologyAvailabilityState.ABSENT
        else:
            attachment_state = TopologyAvailabilityState.ABSENT

        return ResidueChemistryReadinessFacts(
            residue_id=residue.residue_id,
            component_support_state=support_state,
            heavy_atom_topology_availability_state=(
                heavy_atom_topology_availability_state
            ),
            hydrogen_topology_availability_state=attachment_state,
            hydrogen_applicability_state=h_applicability_state,
            hydrogen_coverage_state=h_coverage_state,
            polymer_microstate_targets=(
                self.hydrogen_expectation_model.polymer_preparation.targets_for(
                    residue.residue_id
                )
            ),
        )

    def _derive_retained_non_polymer_chemistry_readiness_fact(
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
        if (
            evidence is not None
            and hydrogen_expectation_resolution.source
            is RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
        ):
            heavy_topology_source = (
                RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
            )
            present_heavy_atom_names = {
                atom_site.name
                for atom_site in residue.atom_sites
                if not atom_site.is_hydrogen()
            }
            expected_heavy_atom_elements = (
                hydrogen_expectation_resolution.heavy_atom_elements
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
                        hydrogen_expectation_resolution.heavy_bond_definitions
                    ),
                    empty_state=TopologyAvailabilityState.PRESENT,
                    covalent_like_endpoint_pairs=self.covalent_like_endpoint_pairs,
                )
                if heavy_atom_names_present and heavy_atom_elements_match
                else TopologyAvailabilityState.ABSENT
            )
        elif evidence is not None:
            heavy_topology_source = RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
            heavy_atom_topology_availability_state = TopologyAvailabilityState.ABSENT
            hydrogen_topology_availability_state = TopologyAvailabilityState.ABSENT
        elif template is not None:
            heavy_topology_source = RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
            present_heavy_atom_names = {
                atom_site.name
                for atom_site in residue.atom_sites
                if not atom_site.is_hydrogen()
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
            heavy_atom_topology_availability_state = (
                residue_bond_topology_availability_state(
                    self.context_structure,
                    residue,
                    expected_bond_definitions=(
                        hydrogen_expectation_resolution.heavy_bond_definitions
                    ),
                    empty_state=TopologyAvailabilityState.PRESENT,
                    covalent_like_endpoint_pairs=(self.covalent_like_endpoint_pairs),
                )
            )
        else:
            heavy_topology_source = RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
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
                hydrogen_topology_availability_state = TopologyAvailabilityState.ABSENT
            elif h_applicability_state is HydrogenApplicabilityState.NOT_APPLICABLE:
                hydrogen_topology_availability_state = (
                    TopologyAvailabilityState.NOT_APPLICABLE
                )
            elif h_coverage_state is HydrogenCoverageState.COMPLETE:
                hydrogen_topology_availability_state = (
                    _retained_non_polymer_hydrogen_topology_availability_state(
                        self.context_structure,
                        residue,
                        chemistry_resolution=hydrogen_expectation_resolution,
                        covalent_like_endpoint_pairs=(
                            self.covalent_like_endpoint_pairs
                        ),
                    )
                )
            else:
                hydrogen_topology_availability_state = TopologyAvailabilityState.ABSENT

        return RetainedNonPolymerChemistryReadinessFact(
            residue_id=residue.residue_id,
            component_id=residue.component_id,
            component_support_state=support_state,
            heavy_topology_source=heavy_topology_source,
            hydrogen_expectation_source=hydrogen_expectation_source,
            heavy_atom_topology_availability_state=(
                heavy_atom_topology_availability_state
            ),
            hydrogen_topology_availability_state=(hydrogen_topology_availability_state),
            hydrogen_applicability_state=h_applicability_state,
            hydrogen_coverage_state=h_coverage_state,
        )

    def residue_completion_facts(
        self,
        residue_id: ResidueId,
    ) -> ResidueCompletionStateFacts:
        """Observe coverage and chemistry for one residue in this context.

        Parameters
        ----------
        residue_id : ResidueId
            Identity resolved against the bound structure.

        Returns
        -------
        ResidueCompletionStateFacts
            Joint observation, reused for repeated queries of this residue.

        Raises
        ------
        KeyError
            The residue is not present in the context.
        """
        cached = self._residue_facts.get(residue_id)
        if cached is not None:
            return cached
        residue = self._residue(residue_id)
        coverage_facts = self._derive_residue_coverage_facts(residue)
        chemistry_readiness_facts = self._derive_residue_chemistry_readiness_facts(
            residue,
            coverage_facts=coverage_facts,
        )
        facts = ResidueCompletionStateFacts(
            coverage_facts=coverage_facts,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )
        self._residue_facts[residue_id] = facts
        return facts

    def residue_facts(
        self,
        residue_id: ResidueId,
    ) -> ResidueProjectionStateFacts:
        """Observe primitive state for one context-resolved residue.

        Parameters
        ----------
        residue_id : ResidueId
            Residue identity in the bound context.

        Returns
        -------
        ResidueProjectionStateFacts
            Singleton facts including stereochemistry.

        Raises
        ------
        KeyError
            The residue is not present in the context.
        """
        residue = self._residue(residue_id)
        completion_facts = self.residue_completion_facts(residue_id)
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

    def _projection_facts(
        self,
        *,
        residues: tuple[ResidueSite, ...],
        ligands: tuple[ResidueSite, ...],
        blueprint_coverages: tuple[StructureBlueprintCoverage, ...],
    ) -> tuple[StructureCoverageFacts, StructureChemistryReadinessFacts]:
        # Retain the joint derivation contract even for an empty projection.
        _ = self.hydrogen_expectation_model
        residue_coverage_facts: list[ResidueCoverageFacts] = []
        residue_chemistry_readiness_facts: list[ResidueChemistryReadinessFacts] = []
        retained_non_polymer_chemistry_readiness_facts: list[
            RetainedNonPolymerChemistryReadinessFact
        ] = []
        for residue in residues:
            facts = self.residue_completion_facts(residue.residue_id)
            residue_coverage_facts.append(facts.coverage_facts)
            residue_chemistry_readiness_facts.append(facts.chemistry_readiness_facts)
        for retained_non_polymer in ligands:
            retained_non_polymer_chemistry_readiness_facts.append(
                self._retained_readiness(retained_non_polymer)
            )

        coverage_residue_facts = tuple(residue_coverage_facts)
        chemistry_residue_facts = tuple(residue_chemistry_readiness_facts)
        coverage_facts = StructureCoverageFacts(
            carrier=self.context_structure,
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
        heavy_atom_topology_availability_state = aggregate_topology_availability_state(
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
            hydrogen_topology_availability_state = (
                TopologyAvailabilityState.NOT_APPLICABLE
            )
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

            hydrogen_topology_availability_state = (
                aggregate_topology_availability_state(
                    residue_fact.hydrogen_topology_availability_state
                    for residue_fact in applicable_residue_facts
                )
            )

        chemistry_readiness_facts = StructureChemistryReadinessFacts(
            carrier=self.context_structure,
            residue_facts=chemistry_residue_facts,
            retained_non_polymer_facts=tuple(
                retained_non_polymer_chemistry_readiness_facts
            ),
            component_support_state=component_support,
            heavy_atom_topology_availability_state=(
                heavy_atom_topology_availability_state
            ),
            hydrogen_topology_availability_state=(hydrogen_topology_availability_state),
            hydrogen_applicability_state=h_applicability_state,
            hydrogen_coverage_state=h_coverage_state,
        )
        return coverage_facts, chemistry_readiness_facts


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
    residue: ResidueSite,
    *,
    component_library: ComponentLibrary,
    expected_hydrogen_atom_names: tuple[str, ...],
    preparation: PolymerMicrostatePreparation,
) -> TopologyAvailabilityState:
    """Check H attachment topology independently of heavy-site conformance."""

    template = component_library.get(residue.component_id)
    if template is None:
        return TopologyAvailabilityState.UNSUPPORTED
    if not expected_hydrogen_atom_names:
        return TopologyAvailabilityState.ABSENT
    present = {atom.name for atom in residue.atom_sites if atom.is_hydrogen()}
    if not present.issuperset(expected_hydrogen_atom_names):
        return TopologyAvailabilityState.ABSENT

    try:
        if preparation.context.hydrogen_topology_updates(
            residue.residue_id,
            PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN),
        ):
            return TopologyAvailabilityState.ABSENT
    except ValueError:
        return TopologyAvailabilityState.ABSENT
    return TopologyAvailabilityState.PRESENT


def _retained_non_polymer_hydrogen_topology_availability_state(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    chemistry_resolution: RetainedNonPolymerChemistryResolution,
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]],
) -> TopologyAvailabilityState:
    """Return retained ligand H topology availability from expected anchors."""

    return residue_bond_topology_availability_state(
        structure,
        residue,
        expected_bond_definitions=chemistry_resolution.hydrogen_bond_definitions,
        empty_state=TopologyAvailabilityState.ABSENT,
        covalent_like_endpoint_pairs=covalent_like_endpoint_pairs,
    )


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
