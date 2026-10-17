"""Normalized local refinement request."""

from dataclasses import dataclass

import protrepair.transformer.refinement.local_pipeline.backend as backend_resolution
from protrepair.chemistry import (
    ComponentLibrary,
    RestraintLibrary,
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.clashes import (
    ClashDetectionBasis,
    prepare_clash_detection_basis,
)
from protrepair.io.pdb_projection import (
    RDKitNoConectPDBBlockProjector,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.backend import ContinuousRelaxationBackend
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationSettings
from protrepair.transformer.refinement.acceptance import DEFAULT_REFINEMENT_CLASH_POLICY


@dataclass(frozen=True, slots=True)
class LocalRefinementRequest:
    """Input-bound request for the full refinement pipeline.

    Parameters
    ----------
    context : ProteinTransformationContext
        Source carrier and selected atom domain.
    spec : ContinuousRelaxationSettings
        Force field, execution region and search settings.
    component_library : ComponentLibrary
        Active chemistry definitions.
    restraint_library : RestraintLibrary
        Active geometric restraints.
    backend : ContinuousRelaxationBackend
        Executor for continuous candidates.
    clash_basis : ClashDetectionBasis
        Source-bound clash preparation.
    chemistry_observation : StructureChemistryObservation
        Chemistry bound to this source, library, evidence and fallback policy.
    allow_retained_non_polymer_rdkit_fallback : bool
        Whether retained-component inference is permitted.
    retained_non_polymer_chemistry_evidence : tuple
        Explicit retained-component evidence.
    pdb_block_projector : RDKitNoConectPDBBlockProjector or None
        Source-bound coordinate projection for parser assessment.

    Raises
    ------
    ValueError
        The chemistry observation has different derivation inputs.
    """

    context: ProteinTransformationContext
    spec: ContinuousRelaxationSettings
    component_library: ComponentLibrary
    restraint_library: RestraintLibrary
    backend: ContinuousRelaxationBackend
    clash_basis: ClashDetectionBasis
    chemistry_observation: StructureChemistryObservation
    allow_retained_non_polymer_rdkit_fallback: bool = True
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = ()
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None

    def __post_init__(self) -> None:
        self.chemistry_observation.require_context(
            self.context.source_snapshot.structure,
            component_library=self.component_library,
            allow_retained_non_polymer_rdkit_fallback=self.allow_retained_non_polymer_rdkit_fallback,
            retained_non_polymer_chemistry_evidence=self.retained_non_polymer_chemistry_evidence,
        )

    @classmethod
    def from_context(
        cls,
        context: ProteinTransformationContext,
        *,
        spec: ContinuousRelaxationSettings,
        component_library: ComponentLibrary | None,
        restraint_library: RestraintLibrary | None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
        chemistry_observation: StructureChemistryObservation | None = None,
    ) -> "LocalRefinementRequest":
        """Normalize boundary inputs into one canonical refinement request.

        Parameters
        ----------
        context : ProteinTransformationContext
            Current carrier and selected atom domain.
        spec : ContinuousRelaxationSettings
            Backend, force field, region radius and iteration settings.
        component_library : ComponentLibrary or None
            Active chemistry definitions, or defaults.
        restraint_library : RestraintLibrary or None
            Active geometric restraints, or defaults.
        allow_retained_non_polymer_rdkit_fallback : bool
            Whether retained-component inference is permitted.
        retained_non_polymer_chemistry_evidence : tuple
            Explicit retained-component evidence.
        chemistry_observation : StructureChemistryObservation or None
            Shared observations requiring these exact derivation inputs.

        Returns
        -------
        LocalRefinementRequest
            Input-bound request after checking current execution readiness.

        Raises
        ------
        ValueError
            A supplied observation has different derivation inputs.
        RefinementError
            Current readiness does not permit execution.
        """

        active_component_library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        active_restraint_library = (
            build_default_restraint_library()
            if restraint_library is None
            else restraint_library
        )
        atom_scope = context.atom_input.observed_atom_scope(context.source_snapshot)
        observation = chemistry_observation
        if observation is None:
            observation = StructureChemistryObservation(
                context.source_snapshot.structure,
                component_library=active_component_library,
                allow_retained_non_polymer_rdkit_fallback=allow_retained_non_polymer_rdkit_fallback,
                retained_non_polymer_chemistry_evidence=retained_non_polymer_chemistry_evidence,
            )
        require_atom_scope_continuous_relaxation_execution(
            derive_atom_scope_continuous_relaxation_facts(
                context.source_snapshot,
                atom_scope,
                component_library=active_component_library,
                context_radius_angstrom=spec.context_radius_angstrom,
                allow_retained_non_polymer_rdkit_fallback=(
                    allow_retained_non_polymer_rdkit_fallback
                ),
                retained_non_polymer_chemistry_evidence=(
                    retained_non_polymer_chemistry_evidence
                ),
                chemistry_observation=observation,
            )
        )

        return cls(
            context=context,
            spec=spec,
            component_library=active_component_library,
            restraint_library=active_restraint_library,
            backend=backend_resolution.resolve_continuous_relaxation_backend(
                spec.backend_name
            ),
            clash_basis=prepare_clash_detection_basis(
                context.source_snapshot.structure,
                component_library=active_component_library,
                policy=DEFAULT_REFINEMENT_CLASH_POLICY,
                polar_chemistry=observation.polar_contact_chemistry,
            ),
            pdb_block_projector=prepare_rdkit_no_conect_pdb_block_projector(
                context.source_snapshot.structure
            ),
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            chemistry_observation=observation,
        )
