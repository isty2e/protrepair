"""Continuous-relaxation readiness derivation and execution-admissibility policy.

This module deliberately straddles two semantic layers:
- fact derivation projects canonical structure state into a local execution region;
- readiness assessment interprets those facts as continuous-relaxation blockers.

The execution-region dependency on :mod:`protrepair.transformer.continuous.domain` is
intentional: readiness must ask the backend problem model whether the included
bond graph can be represented before a force-field backend is bound.
"""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics import ClashPolicy, has_clashes_in_residue_projection
from protrepair.errors import RefinementError
from protrepair.scope import AtomSetScope
from protrepair.state import (
    RetainedNonPolymerChemistryReadinessFact,
    StereochemistryState,
    StructureProjectionStateFacts,
    derive_projection_coverage_and_chemistry_readiness_facts,
)
from protrepair.state.domain import (
    AtomScopeObservation,
    AtomScopeStateFacts,
    ClashState,
    ContinuousBondRealizabilityFacts,
    ContinuousBondRealizabilityState,
    ContinuousRegionReadinessFacts,
    HydrogenAttachmentResolutionFacts,
    SelectedAtomScopeFacts,
    TopologyAvailabilityFacts,
    TopologyAvailabilityState,
)
from protrepair.state.hydrogen_expectation import (
    StructureHydrogenExpectationModel,
    derive_structure_hydrogen_expectation_model,
)
from protrepair.state.scoped import CarrierScopedState
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.continuous.domain import ContinuousRelaxationRegion
from protrepair.transformer.continuous.realizability import (
    continuous_region_bond_realizability_error,
)

CONTINUOUS_RELAXATION_CLASH_POLICY = ClashPolicy(
    include_hydrogens=True,
    include_ligands=True,
)


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationReadinessAssessment:
    """Readiness-policy result for one continuous-relaxation execution domain."""

    blocker_message: str | None = None

    def is_ready(self) -> bool:
        """Return whether the assessed domain can execute continuous relaxation."""

        return self.blocker_message is None

    def require_execution(self) -> None:
        """Raise when the assessed domain cannot execute continuous relaxation."""

        if self.blocker_message is None:
            return

        raise RefinementError(
            "selected domain is not ready for continuous relaxation: "
            f"{self.blocker_message}"
        )


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationReadinessPolicy:
    """Policy that interprets primitive facts as continuous-execution blockers."""

    def assess_structure_facts(
        self,
        facts: StructureProjectionStateFacts,
    ) -> ContinuousRelaxationReadinessAssessment:
        """Assess whole-structure facts for continuous-relaxation readiness."""

        if not facts.component_support_fact.value.is_fully_supported():
            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires fully supported components before "
                "any force field can be bound"
            )

        if (
            facts.backbone_heavy_atom_completeness_fact.value.requires_completion()
            or facts.sidechain_heavy_atom_completeness_fact.value.requires_completion()
        ):
            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires complete heavy-atom coverage before "
                "any force field can be bound"
            )

        if facts.hydrogen_coverage_fact.value.needs_hydrogenation():
            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires hydrogens to be fully realized "
                "before any force field can be bound"
            )

        return ContinuousRelaxationReadinessAssessment()

    def assess_atom_scope_facts(
        self,
        facts: AtomScopeStateFacts,
    ) -> ContinuousRelaxationReadinessAssessment:
        """Assess selected-scope and included-region facts for execution."""

        selected_scope_facts = facts.selected_scope_facts
        continuous_region_facts = facts.continuous_region_readiness_facts
        chemistry_facts = continuous_region_facts.chemistry_readiness_facts
        coverage_facts = continuous_region_facts.coverage_facts

        if not chemistry_facts.component_support_state.is_fully_supported():
            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires fully supported components before "
                "any force field can be bound"
            )

        if (
            coverage_facts.backbone_heavy_atom_completeness_state.requires_completion()
            or (
                coverage_facts.sidechain_heavy_atom_completeness_state
                .requires_completion()
            )
        ):
            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires complete heavy-atom coverage before "
                "any force field can be bound"
            )

        if chemistry_facts.hydrogen_coverage_state.needs_hydrogenation():
            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires hydrogens to be fully realized "
                "before any force field can be bound"
            )

        if (
            selected_scope_facts.structure_facts.stereochemistry_fact.value
            is StereochemistryState.VIOLATED
        ):
            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires selected-scope side-chain "
                "stereochemistry to be consistent before any force field can be bound"
            )

        retained_non_polymer_blocker = (
            self.retained_non_polymer_hydrogen_blocker(
                chemistry_facts.retained_non_polymer_facts
            )
        )
        if retained_non_polymer_blocker is not None:
            return ContinuousRelaxationReadinessAssessment(
                retained_non_polymer_blocker
            )

        bond_realizability = selected_scope_facts.continuous_bond_realizability_facts
        if not self.bond_realizability_supports_execution(bond_realizability):
            if bond_realizability.blocker is not None:
                return ContinuousRelaxationReadinessAssessment(
                    bond_realizability.blocker
                )

            return ContinuousRelaxationReadinessAssessment(
                "continuous relaxation requires a realizable selected-scope bond "
                "graph before any force field can be bound"
            )

        return ContinuousRelaxationReadinessAssessment()

    def topology_availability_supports_execution(
        self,
        state: TopologyAvailabilityState,
    ) -> bool:
        """Return whether a topology-availability state is execution-admissible."""

        return state in {
            TopologyAvailabilityState.NOT_APPLICABLE,
            TopologyAvailabilityState.PRESENT,
        }

    def topology_facts_support_execution(
        self,
        facts: TopologyAvailabilityFacts,
    ) -> bool:
        """Return whether selected topology facts are execution-admissible."""

        return all(
            self.topology_availability_supports_execution(residue_fact.state)
            for residue_fact in facts.residue_facts
        )

    def bond_realizability_supports_execution(
        self,
        facts: ContinuousBondRealizabilityFacts,
    ) -> bool:
        """Return whether selected-scope bonds are execution-realizable."""

        return facts.state is ContinuousBondRealizabilityState.REALIZABLE

    def retained_non_polymer_hydrogen_blocker(
        self,
        retained_non_polymer_facts: tuple[
            RetainedNonPolymerChemistryReadinessFact,
            ...,
        ],
    ) -> str | None:
        """Return the retained non-polymer hydrogen blocker, if present."""

        if any(
            retained_fact.requires_hydrogen_completion()
            for retained_fact in retained_non_polymer_facts
        ):
            return (
                "continuous relaxation requires retained non-polymer hydrogens to be "
                "fully realized in the included local region before any force field "
                "can be bound"
            )

        if any(
            retained_fact.is_supported()
            and retained_fact.hydrogen_applicability_state.is_applicable()
            and not self.topology_availability_supports_execution(
                retained_fact.hydrogen_topology_availability_state
            )
            for retained_fact in retained_non_polymer_facts
        ):
            return (
                "continuous relaxation requires retained non-polymer hydrogen "
                "topology to be present in the included local region before any "
                "force field can be bound"
            )

        return None


DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY = (
    ContinuousRelaxationReadinessPolicy()
)


def derive_atom_scope_continuous_relaxation_facts(
    snapshot: ProteinStructureSnapshot,
    atom_scope: AtomSetScope,
    *,
    component_library: ComponentLibrary | None = None,
    context_radius_angstrom: float = 6.0,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    hydrogen_expectation_model: StructureHydrogenExpectationModel | None = None,
) -> AtomScopeStateFacts:
    """Derive atom-scope facts needed for continuous-relaxation readiness."""

    referenced_residue_ids = tuple(
        dict.fromkeys(atom_ref.residue_id for atom_ref in atom_scope.atom_refs)
    )
    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    active_hydrogen_expectation_model = (
        derive_structure_hydrogen_expectation_model(
            snapshot.structure,
            component_library=active_component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
        )
        if hydrogen_expectation_model is None
        else hydrogen_expectation_model
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in atom_scope.atom_refs
        ),
        basis=AtomInputBasis.ATOMWISE,
        selected_scope=atom_scope,
    )
    projected_residues = tuple(
        residue
        for residue_id in referenced_residue_ids
        if (residue := snapshot.structure.constitution.residue_or_ligand(residue_id))
        is not None
        and not residue.is_hetero
    )
    projected_ligands: list[ResidueSite] = []
    for residue_id in referenced_residue_ids:
        residue = snapshot.structure.constitution.residue_or_ligand(residue_id)
        if residue is not None and residue.is_hetero:
            projected_ligands.append(residue)
    structure_facts = StructureProjectionStateFacts.from_projection(
        context_structure=snapshot.structure,
        residues=projected_residues,
        ligands=tuple(projected_ligands),
        component_library=active_component_library,
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        hydrogen_expectation_model=active_hydrogen_expectation_model,
    )
    (
        continuous_region_readiness_facts,
        continuous_bond_realizability_facts,
    ) = _derive_execution_region_readiness_facts(
        snapshot,
        atom_input,
        atom_scope=atom_scope,
        component_library=active_component_library,
        context_radius_angstrom=context_radius_angstrom,
        allow_retained_non_polymer_rdkit_fallback=(
            allow_retained_non_polymer_rdkit_fallback
        ),
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        hydrogen_expectation_model=active_hydrogen_expectation_model,
    )
    hydrogen_atom_count = sum(
        1
        for atom_ref in atom_scope.atom_refs
        if (atom_site := snapshot.structure.constitution.resolve_atom_site(atom_ref))
        is not None
        and atom_site.is_hydrogen()
    )
    return AtomScopeStateFacts(
        selected_scope_facts=SelectedAtomScopeFacts(
            atom_count=len(atom_scope.atom_refs),
            residue_count=len(referenced_residue_ids),
            hydrogen_atom_count=hydrogen_atom_count,
            structure_facts=structure_facts,
            clash_fact=CarrierScopedState(
                carrier=snapshot.structure,
                scope=atom_scope,
                value=_atom_scope_clash_state(
                    snapshot,
                    residue_ids=referenced_residue_ids,
                    component_library=active_component_library,
                ),
            ),
            topology_availability_facts=TopologyAvailabilityFacts.from_projection(
                snapshot,
                residue_ids=referenced_residue_ids,
                component_library=active_component_library,
            ),
            hydrogen_attachment_resolution_facts=(
                HydrogenAttachmentResolutionFacts.from_projection(
                    snapshot,
                    residue_ids=referenced_residue_ids,
                    component_library=active_component_library,
                )
            ),
            continuous_bond_realizability_facts=continuous_bond_realizability_facts,
        ),
        continuous_region_readiness_facts=continuous_region_readiness_facts,
    )


def derive_atom_scope_continuous_relaxation_observation(
    snapshot: ProteinStructureSnapshot,
    atom_scope: AtomSetScope,
    *,
    component_library: ComponentLibrary | None = None,
    context_radius_angstrom: float = 6.0,
) -> AtomScopeObservation:
    """Derive the observation read model for continuous-relaxation readiness."""

    return AtomScopeObservation.from_facts(
        derive_atom_scope_continuous_relaxation_facts(
            snapshot,
            atom_scope,
            component_library=component_library,
            context_radius_angstrom=context_radius_angstrom,
        )
    )


def _derive_execution_region_readiness_facts(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    *,
    atom_scope: AtomSetScope,
    component_library: ComponentLibrary,
    context_radius_angstrom: float = 6.0,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    hydrogen_expectation_model: StructureHydrogenExpectationModel | None = None,
) -> tuple[ContinuousRegionReadinessFacts, ContinuousBondRealizabilityFacts]:
    """Derive facts by projecting through the continuous execution-region model."""

    continuous_region = ContinuousRelaxationRegion.from_inputs(
        snapshot,
        atom_input,
        context_radius_angstrom=context_radius_angstrom,
    )
    continuous_region_residues = tuple(
        residue_site
        for residue_index in continuous_region.included_residue_indices
        if not (residue_site := continuous_region.residue_site(residue_index)).is_hetero
    )
    continuous_region_ligands = tuple(
        residue_site
        for residue_index in continuous_region.included_residue_indices
        if (residue_site := continuous_region.residue_site(residue_index)).is_hetero
    )
    (
        continuous_region_coverage_facts,
        continuous_region_chemistry_readiness_facts,
    ) = derive_projection_coverage_and_chemistry_readiness_facts(
        context_structure=snapshot.structure,
        residues=continuous_region_residues,
        ligands=continuous_region_ligands,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
        hydrogen_expectation_model=hydrogen_expectation_model,
    )
    continuous_bond_realizability_blocker = (
        continuous_region_bond_realizability_error(
            continuous_region,
            component_library=component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            retained_non_polymer_chemistry_resolution_by_residue_id=(
                None
                if hydrogen_expectation_model is None
                else (
                    hydrogen_expectation_model
                    .retained_non_polymer_resolution_by_residue_id
                )
            ),
        )
    )
    return (
        ContinuousRegionReadinessFacts(
            coverage_facts=continuous_region_coverage_facts,
            chemistry_readiness_facts=continuous_region_chemistry_readiness_facts,
        ),
        ContinuousBondRealizabilityFacts(
            carrier=snapshot.structure,
            scope=atom_scope,
            state=(
                ContinuousBondRealizabilityState.REALIZABLE
                if continuous_bond_realizability_blocker is None
                else ContinuousBondRealizabilityState.UNREALIZABLE
            ),
            blocker=continuous_bond_realizability_blocker,
        ),
    )


def _atom_scope_clash_state(
    snapshot: ProteinStructureSnapshot,
    *,
    residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
) -> ClashState:
    """Return steric clash truth over one selected residue projection."""

    if has_clashes_in_residue_projection(
        snapshot.structure,
        residue_ids=residue_ids,
        component_library=component_library,
        policy=CONTINUOUS_RELAXATION_CLASH_POLICY,
    ):
        return ClashState.PRESENT

    return ClashState.NONE


def topology_availability_state_supports_continuous_relaxation(
    state: TopologyAvailabilityState,
) -> bool:
    """Return whether one topology-availability state admits relaxation use."""

    return (
        DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY
        .topology_availability_supports_execution(state)
    )


def topology_availability_facts_supports_continuous_relaxation(
    facts: TopologyAvailabilityFacts,
) -> bool:
    """Return whether every selected topology fact admits relaxation use."""

    return (
        DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY
        .topology_facts_support_execution(facts)
    )


def continuous_bond_realizability_state_supports_continuous_relaxation(
    state: ContinuousBondRealizabilityState,
) -> bool:
    """Return whether one bond-realizability state admits relaxation use."""

    return state is ContinuousBondRealizabilityState.REALIZABLE


def continuous_bond_realizability_facts_support_continuous_relaxation(
    facts: ContinuousBondRealizabilityFacts,
) -> bool:
    """Return whether the selected-scope bond graph admits continuous planning."""

    return (
        DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY
        .bond_realizability_supports_execution(facts)
    )


def structure_facts_continuous_relaxation_error(
    facts: StructureProjectionStateFacts,
) -> str | None:
    """Return the current whole-structure relaxation blocker, if any."""

    return (
        DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY
        .assess_structure_facts(facts)
        .blocker_message
    )


def structure_facts_supports_continuous_relaxation(
    facts: StructureProjectionStateFacts,
) -> bool:
    """Return whether one whole-structure fact bundle admits relaxation."""

    return structure_facts_continuous_relaxation_error(facts) is None


def atom_scope_facts_continuous_relaxation_error(
    facts: AtomScopeStateFacts,
) -> str | None:
    """Return the current selected-scope relaxation blocker, if any."""

    return (
        DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY
        .assess_atom_scope_facts(facts)
        .blocker_message
    )


def retained_non_polymer_hydrogen_readiness_error(
    retained_non_polymer_facts: tuple[RetainedNonPolymerChemistryReadinessFact, ...],
) -> str | None:
    """Return the retained non-polymer hydrogen blocker for one local region."""

    return (
        DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY
        .retained_non_polymer_hydrogen_blocker(retained_non_polymer_facts)
    )


def atom_scope_facts_supports_continuous_relaxation(
    facts: AtomScopeStateFacts,
) -> bool:
    """Return whether one selected-scope fact bundle admits relaxation."""

    return atom_scope_facts_continuous_relaxation_error(facts) is None


def require_atom_scope_continuous_relaxation_execution(
    facts: AtomScopeStateFacts,
) -> None:
    """Raise when current local truth cannot legally execute any force field."""

    DEFAULT_CONTINUOUS_RELAXATION_READINESS_POLICY.assess_atom_scope_facts(
        facts
    ).require_execution()
