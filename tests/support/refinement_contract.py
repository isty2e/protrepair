"""Test helpers for the canonical snapshot-domain refinement contract."""

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationForceField,
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.local import (
    LocalScopeSpec,
    atom_input_from_local_scope_spec,
)


def build_refinement_snapshot(
    structure: ProteinStructure,
) -> ProteinStructureSnapshot:
    """Return the canonical snapshot over one test structure."""

    return ProteinStructureSnapshot.from_structure(structure)


def build_atom_domain(
    structure: ProteinStructure,
    scope_spec: LocalScopeSpec,
    *,
    component_library: ComponentLibrary | None = None,
) -> AtomInput:
    """Return the canonical atom domain for one test structure and scope spec."""

    snapshot = build_refinement_snapshot(structure)
    return atom_input_from_local_scope_spec(
        snapshot,
        scope_spec,
        component_library=component_library,
    )


def build_refinement_inputs(
    structure: ProteinStructure,
    scope_spec: LocalScopeSpec,
    *,
    component_library: ComponentLibrary | None = None,
    profile: ContinuousRelaxationProfile | None = None,
    backend_name: str = "rdkit",
    force_field: ContinuousRelaxationForceField = ContinuousRelaxationForceField.UFF,
    context_radius_angstrom: float = 6.0,
    max_iterations: int = 200,
) -> tuple[ProteinStructureSnapshot, AtomInput, ContinuousRelaxationSettings]:
    """Return canonical refinement inputs for one test structure."""

    snapshot = build_refinement_snapshot(structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        scope_spec,
        component_library=component_library,
    )
    if profile is not None:
        spec = ContinuousRelaxationSettings(
            profile=profile,
            context_radius_angstrom=context_radius_angstrom,
            max_iterations=max_iterations,
        )
    else:
        spec = ContinuousRelaxationSettings(
            backend_name=backend_name,
            force_field=force_field,
            context_radius_angstrom=context_radius_angstrom,
            max_iterations=max_iterations,
        )

    return snapshot, atom_input, spec


def build_continuous_relaxation_problem(
    structure: ProteinStructure,
    scope_spec: LocalScopeSpec,
    *,
    component_library: ComponentLibrary,
    profile: ContinuousRelaxationProfile | None = None,
    backend_name: str = "rdkit",
    force_field: ContinuousRelaxationForceField = ContinuousRelaxationForceField.UFF,
    context_radius_angstrom: float = 6.0,
    max_iterations: int = 200,
) -> ContinuousRelaxationProblem:
    """Return one continuous-relaxation problem from canonical inputs."""

    snapshot, atom_input, spec = build_refinement_inputs(
        structure,
        scope_spec,
        component_library=component_library,
        profile=profile,
        backend_name=backend_name,
        force_field=force_field,
        context_radius_angstrom=context_radius_angstrom,
        max_iterations=max_iterations,
    )
    return ContinuousRelaxationProblem.from_inputs(
        snapshot,
        atom_input,
        spec=spec,
        component_library=component_library,
    )
