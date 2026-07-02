"""Public boundary entrypoints for selected-region transformations."""

from dataclasses import dataclass
from time import perf_counter

from protrepair.chemistry import (
    ComponentLibrary,
    RestraintLibrary,
    build_default_component_library,
)
from protrepair.errors import RefinementError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.local.models import DirectRegionTransformationSpec
from protrepair.transformer.refinement.local_pipeline.runtime import (
    LocalRefinementRuntimeProfile,
    execute_local_transformation,
    execute_local_transformation_profiled,
)


@dataclass(frozen=True, slots=True)
class LocalRegionTransformationRuntimeProfile:
    """Boundary-to-pipeline timing breakdown for one local refinement call."""

    snapshot_runtime_ms: float
    atom_input_lowering_runtime_ms: float
    readiness_runtime_ms: float
    settings_bind_runtime_ms: float
    refinement_pipeline: LocalRefinementRuntimeProfile


def transform_local_region(
    structure: ProteinStructure,
    spec: DirectRegionTransformationSpec,
    *,
    component_library: ComponentLibrary | None = None,
    restraint_library: RestraintLibrary | None = None,
) -> RegionTransformationResult:
    """Transform one selected local region against a fixed surrounding context.

    This API requires the current selected domain to be hydrogen-complete and
    topology-valid before any force field can be bound. The caller must provide
    one explicit force-field choice on the boundary request. Supporting
    structures may be represented on the boundary spec, but direct execution
    currently accepts only source-only transformation contexts.
    """

    if not isinstance(spec, DirectRegionTransformationSpec):
        raise TypeError(
            "direct region transformation requires a "
            "DirectRegionTransformationSpec value"
        )
    if not spec.context.is_source_only():
        raise NotImplementedError(
            "direct region transformation currently supports only source-only "
            "transformation contexts"
        )

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = spec.scope_spec.lower_to_atom_input(
        snapshot,
        component_library=active_component_library,
    )
    atom_scope = atom_input.observed_atom_scope(snapshot)
    require_atom_scope_continuous_relaxation_execution(
        derive_atom_scope_continuous_relaxation_facts(
            snapshot,
            atom_scope,
            component_library=active_component_library,
            context_radius_angstrom=spec.config.context_radius_angstrom,
        )
    )

    try:
        settings = spec.config.bind(spec.force_field)
    except ValueError as error:
        raise RefinementError(
            f"continuous relaxation backend {spec.config.backend_name!r} is not "
            "implemented"
        ) from error

    return execute_local_transformation(
        ProteinTransformationContext.from_snapshot_atom_input(snapshot, atom_input),
        spec=settings,
        component_library=active_component_library,
        restraint_library=restraint_library,
    )


def profile_transform_local_region(
    structure: ProteinStructure,
    spec: DirectRegionTransformationSpec,
    *,
    component_library: ComponentLibrary | None = None,
    restraint_library: RestraintLibrary | None = None,
) -> tuple[RegionTransformationResult, LocalRegionTransformationRuntimeProfile]:
    """Transform one selected local region and return boundary stage timings."""

    if not isinstance(spec, DirectRegionTransformationSpec):
        raise TypeError(
            "direct region transformation requires a "
            "DirectRegionTransformationSpec value"
        )
    if not spec.context.is_source_only():
        raise NotImplementedError(
            "direct region transformation currently supports only source-only "
            "transformation contexts"
        )

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )

    start = perf_counter()
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    snapshot_runtime_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    atom_input = spec.scope_spec.lower_to_atom_input(
        snapshot,
        component_library=active_component_library,
    )
    atom_input_lowering_runtime_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    atom_scope = atom_input.observed_atom_scope(snapshot)
    require_atom_scope_continuous_relaxation_execution(
        derive_atom_scope_continuous_relaxation_facts(
            snapshot,
            atom_scope,
            component_library=active_component_library,
            context_radius_angstrom=spec.config.context_radius_angstrom,
        )
    )
    readiness_runtime_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    try:
        settings = spec.config.bind(spec.force_field)
    except ValueError as error:
        raise RefinementError(
            f"continuous relaxation backend {spec.config.backend_name!r} is not "
            "implemented"
        ) from error
    settings_bind_runtime_ms = (perf_counter() - start) * 1000.0

    result, refinement_pipeline = execute_local_transformation_profiled(
        ProteinTransformationContext.from_snapshot_atom_input(snapshot, atom_input),
        spec=settings,
        component_library=active_component_library,
        restraint_library=restraint_library,
    )
    return (
        result,
        LocalRegionTransformationRuntimeProfile(
            snapshot_runtime_ms=snapshot_runtime_ms,
            atom_input_lowering_runtime_ms=atom_input_lowering_runtime_ms,
            readiness_runtime_ms=readiness_runtime_ms,
            settings_bind_runtime_ms=settings_bind_runtime_ms,
            refinement_pipeline=refinement_pipeline,
        ),
    )
