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
from protrepair.state.hydrogen_expectation import (
    StructureHydrogenExpectationModel,
    derive_structure_hydrogen_expectation_model,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.backend import ContinuousRelaxationBackend
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationSettings


@dataclass(frozen=True, slots=True)
class LocalRefinementRequest:
    """One normalized internal request for the full refinement pipeline."""

    context: ProteinTransformationContext
    spec: ContinuousRelaxationSettings
    component_library: ComponentLibrary
    restraint_library: RestraintLibrary
    backend: ContinuousRelaxationBackend
    clash_basis: ClashDetectionBasis
    allow_retained_non_polymer_rdkit_fallback: bool = True
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = ()
    hydrogen_expectation_model: StructureHydrogenExpectationModel | None = None

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
    ) -> "LocalRefinementRequest":
        """Normalize boundary inputs into one canonical refinement request."""

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
        hydrogen_expectation_model = derive_structure_hydrogen_expectation_model(
            context.source_snapshot.structure,
            component_library=active_component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
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
                hydrogen_expectation_model=hydrogen_expectation_model,
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
            ),
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            hydrogen_expectation_model=hydrogen_expectation_model,
        )

def normalize_local_refinement_request(
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
) -> LocalRefinementRequest:
    """Normalize boundary inputs into one canonical internal refinement request."""

    return LocalRefinementRequest.from_context(
        context=context,
        spec=spec,
        component_library=component_library,
        restraint_library=restraint_library,
        allow_retained_non_polymer_rdkit_fallback=(
            allow_retained_non_polymer_rdkit_fallback
        ),
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
    )
