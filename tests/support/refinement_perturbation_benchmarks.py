"""Perturb-and-recover benchmarks over reference-backed local refinement cases."""

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import cos, radians, sin
from statistics import median
from time import perf_counter

import numpy as np
from numpy.random import RandomState

from protrepair.chemistry import (
    ComponentLibrary,
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationForceField,
)
from protrepair.transformer.continuous.shared import apply_position_updates
from protrepair.transformer.local import (
    LocalRegionTransformationRuntimeProfile,
    profile_transform_local_region,
)
from protrepair.transformer.local.lowering import atom_input_from_local_scope_spec
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.local_pipeline.backend import (
    resolve_continuous_relaxation_backend,
)
from tests.support.refinement_benchmarks import (
    REFINEMENT_BENCHMARK_PROFILES,
    REFINEMENT_BENCHMARK_TRACKS,
    REFINEMENT_STRATEGIES,
    RefinementBenchmarkExecutionProfile,
    RefinementBenchmarkQualityMetrics,
    RefinementBenchmarkRuntimeBreakdown,
    RefinementBenchmarkTrack,
    RefinementStrategy,
    compute_refinement_metrics,
    focus_atom_displacement_metrics,
    load_case_reference_structure,
    max_fixed_context_displacement_angstrom,
    runtime_breakdown_from_profiles,
)
from tests.support.refinement_cases import (
    REFINEMENT_BENCHMARK_CASES,
    RefinementBenchmarkCase,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem


class RefinementPerturbationMode(str, Enum):
    """Closed perturbation families exposed by the perturb benchmark harness."""

    RANDOM_RIGID = "random_rigid"
    DIRECTED_SHOVE = "directed_shove"


class RefinementPerturbationExpectedOutcome(str, Enum):
    """Closed semantic expectation for one perturb-and-recover scenario."""

    POSITIVE_RECOVERY = "positive_recovery"
    RUNNABLE_SUCCESS = "runnable_success"
    EXPECTED_NO_OP = "expected_no_op"
    EXPECTED_ERROR = "expected_error"

    def expects_reference_rmsd_recovery(self) -> bool:
        """Return whether the scenario should move closer to reference geometry."""

        return self is RefinementPerturbationExpectedOutcome.POSITIVE_RECOVERY

    def matches_status(self, status: str) -> bool:
        """Return whether one actual perturbation status satisfies this outcome."""

        if self is RefinementPerturbationExpectedOutcome.EXPECTED_ERROR:
            return status == "error"
        if self is RefinementPerturbationExpectedOutcome.EXPECTED_NO_OP:
            return status == "no_op"

        return status == "success"


@dataclass(frozen=True, slots=True)
class RefinementPerturbationSpec:
    """One deterministic perturbation recipe over a realized movable atom set."""

    name: str
    mode: RefinementPerturbationMode
    random_seed: int
    rotation_degrees: float
    translation_angstrom: float


@dataclass(frozen=True, slots=True)
class RefinementPerturbationBenchmarkResult:
    """One perturb-and-recover benchmark result for one case/profile pair."""

    case_id: str
    perturbation_name: str
    track: str
    execution_profile: str
    backend_name: str
    method: str
    strategy_name: str
    status: str
    perturbation_runtime_ms: float
    backend_runtime_median_ms: float
    backend_runtime_samples_ms: tuple[float, ...]
    refinement_runtime_median_ms: float
    refinement_runtime_samples_ms: tuple[float, ...]
    runtime_breakdown: RefinementBenchmarkRuntimeBreakdown | None
    metrics_before: RefinementBenchmarkQualityMetrics
    metrics_after: RefinementBenchmarkQualityMetrics | None
    moved_atom_count: int | None
    focus_atom_rms_displacement_angstrom: float | None
    focus_atom_max_displacement_angstrom: float | None
    fixed_context_max_displacement_angstrom: float | None
    error_type: str | None
    error_message: str | None

    def as_serializable_dict(self) -> dict[str, object]:
        """Return one JSON-serializable projection."""

        return {
            "case_id": self.case_id,
            "perturbation_name": self.perturbation_name,
            "track": self.track,
            "execution_profile": self.execution_profile,
            "backend_name": self.backend_name,
            "method": self.method,
            "strategy_name": self.strategy_name,
            "status": self.status,
            "perturbation_runtime_ms": self.perturbation_runtime_ms,
            "backend_runtime_median_ms": self.backend_runtime_median_ms,
            "backend_runtime_samples_ms": list(self.backend_runtime_samples_ms),
            "refinement_runtime_median_ms": self.refinement_runtime_median_ms,
            "refinement_runtime_samples_ms": list(self.refinement_runtime_samples_ms),
            "runtime_breakdown": (
                None
                if self.runtime_breakdown is None
                else asdict(self.runtime_breakdown)
            ),
            "metrics_before": asdict(self.metrics_before),
            "metrics_after": (
                None if self.metrics_after is None else asdict(self.metrics_after)
            ),
            "moved_atom_count": self.moved_atom_count,
            "focus_atom_rms_displacement_angstrom": (
                self.focus_atom_rms_displacement_angstrom
            ),
            "focus_atom_max_displacement_angstrom": (
                self.focus_atom_max_displacement_angstrom
            ),
            "fixed_context_max_displacement_angstrom": (
                self.fixed_context_max_displacement_angstrom
            ),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class RefinementPerturbationBenchmarkScenario:
    """One validated perturb-and-recover benchmark scenario."""

    scenario_id: str
    case: RefinementBenchmarkCase
    strategy: RefinementStrategy
    perturbation: RefinementPerturbationSpec
    track: RefinementBenchmarkTrack
    execution_profile: RefinementBenchmarkExecutionProfile
    expected_outcome: RefinementPerturbationExpectedOutcome = (
        RefinementPerturbationExpectedOutcome.POSITIVE_RECOVERY
    )


REFERENCE_REFINEMENT_BENCHMARK_CASES: dict[str, RefinementBenchmarkCase] = {
    case_id: case
    for case_id, case in REFINEMENT_BENCHMARK_CASES.items()
    if case.reference_fixture_path is not None
}


REFINEMENT_PERTURBATION_SPECS: dict[str, RefinementPerturbationSpec] = {
    "rigid_small": RefinementPerturbationSpec(
        name="rigid_small",
        mode=RefinementPerturbationMode.RANDOM_RIGID,
        random_seed=17,
        rotation_degrees=20.0,
        translation_angstrom=0.60,
    ),
    "rigid_medium": RefinementPerturbationSpec(
        name="rigid_medium",
        mode=RefinementPerturbationMode.RANDOM_RIGID,
        random_seed=23,
        rotation_degrees=45.0,
        translation_angstrom=1.00,
    ),
    "clash_shove": RefinementPerturbationSpec(
        name="clash_shove",
        mode=RefinementPerturbationMode.DIRECTED_SHOVE,
        random_seed=31,
        rotation_degrees=12.0,
        translation_angstrom=1.10,
    ),
}


def _self_reference_perturbation_case(case_id: str) -> RefinementBenchmarkCase:
    """Return a perturbation case that uses its own fixture as reference truth."""

    case = REFINEMENT_BENCHMARK_CASES[case_id]
    return replace(
        case,
        case_id=f"{case_id}-self-reference",
        title=f"{case.title} self-reference perturbation",
        description=f"Self-reference perturb-and-recover case: {case.description}",
        reference_fixture_path=case.fixture_path,
    )


SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES: dict[str, RefinementBenchmarkCase] = {
    case.case_id: case
    for case in (
        _self_reference_perturbation_case("1jd0-gln92"),
        _self_reference_perturbation_case("4jfx-ptr8"),
    )
}


PERTURBATION_REFINEMENT_BENCHMARK_CASES: dict[str, RefinementBenchmarkCase] = {
    **REFERENCE_REFINEMENT_BENCHMARK_CASES,
    **SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES,
}


PERTURBATION_BENCHMARK_SCENARIOS: dict[
    str,
    RefinementPerturbationBenchmarkScenario,
] = {
    "1xgo-leu253-rigid-small-external-reference": (
        RefinementPerturbationBenchmarkScenario(
            scenario_id="1xgo-leu253-rigid-small-external-reference",
            case=REFERENCE_REFINEMENT_BENCHMARK_CASES["1xgo-leu253"],
            strategy=REFINEMENT_STRATEGIES["sidechain-local"],
            perturbation=REFINEMENT_PERTURBATION_SPECS["rigid_small"],
            track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
            execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
        )
    ),
    "1jd0-gln92-rigid-medium-self-reference": (
        RefinementPerturbationBenchmarkScenario(
            scenario_id="1jd0-gln92-rigid-medium-self-reference",
            case=SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES[
                "1jd0-gln92-self-reference"
            ],
            strategy=REFINEMENT_STRATEGIES["sidechain-local"],
            perturbation=REFINEMENT_PERTURBATION_SPECS["rigid_medium"],
            track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
            execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
        )
    ),
    "4jfx-ptr8-rigid-small-self-reference": (
        RefinementPerturbationBenchmarkScenario(
            scenario_id="4jfx-ptr8-rigid-small-self-reference",
            case=SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES[
                "4jfx-ptr8-self-reference"
            ],
            strategy=REFINEMENT_STRATEGIES["sidechain-local"],
            perturbation=REFINEMENT_PERTURBATION_SPECS["rigid_small"],
            track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
            execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
        )
    ),
    "4jfx-ptr8-rigid-medium-self-reference": (
        RefinementPerturbationBenchmarkScenario(
            scenario_id="4jfx-ptr8-rigid-medium-self-reference",
            case=SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES[
                "4jfx-ptr8-self-reference"
            ],
            strategy=REFINEMENT_STRATEGIES["sidechain-local"],
            perturbation=REFINEMENT_PERTURBATION_SPECS["rigid_medium"],
            track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
            execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
            expected_outcome=RefinementPerturbationExpectedOutcome.RUNNABLE_SUCCESS,
        )
    ),
}


def run_refinement_perturbation_benchmark_case(
    case: RefinementBenchmarkCase,
    strategy: RefinementStrategy,
    *,
    perturbation: RefinementPerturbationSpec,
    track: RefinementBenchmarkTrack,
    execution_profile: RefinementBenchmarkExecutionProfile | None = None,
    force_field: ContinuousRelaxationForceField | None = None,
    repeats: int,
    component_library: ComponentLibrary | None = None,
) -> RefinementPerturbationBenchmarkResult:
    """Run one perturb-and-recover benchmark from a checked-in reference structure."""

    if case.reference_fixture_path is None:
        raise ValueError(
            "perturbation benchmarks require a reference-backed benchmark case"
        )
    if repeats <= 0:
        raise ValueError("perturbation benchmark repeats must be positive")
    if execution_profile is None and force_field is None:
        raise TypeError(
            "perturbation benchmarks require one execution_profile or legacy "
            "force_field"
        )
    if execution_profile is not None and force_field is not None:
        raise TypeError(
            "perturbation benchmarks must not mix execution_profile with force_field"
        )

    if execution_profile is not None:
        normalized_execution_profile = execution_profile
    else:
        assert force_field is not None
        normalized_execution_profile = REFINEMENT_BENCHMARK_PROFILES[
            f"rdkit_{force_field.value}"
        ]

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    active_restraint_library = build_default_restraint_library()
    reference_structure = load_case_reference_structure(
        case,
        track=track,
        component_library=active_component_library,
    )
    if reference_structure is None:
        raise ValueError(
            "perturbation benchmarks require a loadable reference structure"
        )

    spec = strategy.build_spec(case, execution_profile=normalized_execution_profile)
    try:
        perturbed_structure, perturbation_runtime_ms = perturb_reference_structure(
            reference_structure,
            scope_spec=spec.scope_spec,
            perturbation=perturbation,
            component_library=active_component_library,
        )
    except Exception as error:
        raise RefinementError(
            f"failed to build perturbed benchmark input: {error}"
        ) from error

    backend = resolve_continuous_relaxation_backend(spec.config.backend_name)
    backend_runtime_samples_ms: list[float] = []
    runtime_samples_ms: list[float] = []
    runtime_profiles: list[LocalRegionTransformationRuntimeProfile] = []
    final_refined_structure: ProteinStructure | None = None
    final_issues: tuple[ValidationIssue, ...] = ()
    moved_atom_count: int | None = None
    focus_atom_rms_displacement_angstrom: float | None = None
    focus_atom_max_displacement_angstrom: float | None = None
    fixed_context_max_displacement_angstrom: float | None = None

    try:
        build_start = perf_counter()
        plan = build_continuous_relaxation_problem(
            perturbed_structure,
            scope_spec=spec.scope_spec,
            profile=normalized_execution_profile.profile,
            context_radius_angstrom=spec.config.context_radius_angstrom,
            max_iterations=spec.config.max_iterations,
            component_library=active_component_library,
        )
        problem_build_runtime_ms = (perf_counter() - build_start) * 1000.0
    except Exception as error:
        problem_build_runtime_ms = 0.0
        metrics_before = compute_refinement_metrics(
            perturbed_structure,
            case=case,
            component_library=active_component_library,
            restraint_library=active_restraint_library,
            reference_structure=reference_structure,
            reference_alignment_atom_refs=(),
        )
        return RefinementPerturbationBenchmarkResult(
            case_id=case.case_id,
            perturbation_name=perturbation.name,
            track=track.value,
            execution_profile=normalized_execution_profile.name,
            backend_name=normalized_execution_profile.backend_name,
            method=normalized_execution_profile.method.value,
            strategy_name=strategy.name,
            status="error",
            perturbation_runtime_ms=perturbation_runtime_ms,
            backend_runtime_median_ms=0.0,
            backend_runtime_samples_ms=(),
            refinement_runtime_median_ms=0.0,
            refinement_runtime_samples_ms=(),
            runtime_breakdown=None,
            metrics_before=metrics_before,
            metrics_after=None,
            moved_atom_count=None,
            focus_atom_rms_displacement_angstrom=None,
            focus_atom_max_displacement_angstrom=None,
            fixed_context_max_displacement_angstrom=None,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    reference_alignment_atom_refs = tuple(
        plan.region.snapshot.structure.constitution.atom_ref_at(atom_index)
        for atom_index in plan.region.fixed_context_atom_indices
    )
    metrics_before = compute_refinement_metrics(
        perturbed_structure,
        case=case,
        component_library=active_component_library,
        restraint_library=active_restraint_library,
        reference_structure=reference_structure,
        reference_alignment_atom_refs=reference_alignment_atom_refs,
    )

    for _ in range(repeats):
        start = perf_counter()
        try:
            backend.relax(
                plan,
                restraint_library=active_restraint_library,
            )
        except Exception as error:
            backend_runtime_samples_ms.append((perf_counter() - start) * 1000.0)
            return RefinementPerturbationBenchmarkResult(
                case_id=case.case_id,
                perturbation_name=perturbation.name,
                track=track.value,
                execution_profile=normalized_execution_profile.name,
                backend_name=normalized_execution_profile.backend_name,
                method=normalized_execution_profile.method.value,
                strategy_name=strategy.name,
                status="error",
                perturbation_runtime_ms=perturbation_runtime_ms,
                backend_runtime_median_ms=median(backend_runtime_samples_ms),
                backend_runtime_samples_ms=tuple(backend_runtime_samples_ms),
                refinement_runtime_median_ms=0.0,
                refinement_runtime_samples_ms=(),
                runtime_breakdown=None,
                metrics_before=metrics_before,
                metrics_after=None,
                moved_atom_count=None,
                focus_atom_rms_displacement_angstrom=None,
                focus_atom_max_displacement_angstrom=None,
                fixed_context_max_displacement_angstrom=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        backend_runtime_samples_ms.append((perf_counter() - start) * 1000.0)

    for _ in range(repeats):
        start = perf_counter()
        try:
            refinement_result, runtime_profile = profile_transform_local_region(
                perturbed_structure,
                spec,
                component_library=active_component_library,
            )
        except Exception as error:
            runtime_samples_ms.append((perf_counter() - start) * 1000.0)
            return RefinementPerturbationBenchmarkResult(
                case_id=case.case_id,
                perturbation_name=perturbation.name,
                track=track.value,
                execution_profile=normalized_execution_profile.name,
                backend_name=normalized_execution_profile.backend_name,
                method=normalized_execution_profile.method.value,
                strategy_name=strategy.name,
                status="error",
                perturbation_runtime_ms=perturbation_runtime_ms,
                backend_runtime_median_ms=median(backend_runtime_samples_ms),
                backend_runtime_samples_ms=tuple(backend_runtime_samples_ms),
                refinement_runtime_median_ms=median(runtime_samples_ms),
                refinement_runtime_samples_ms=tuple(runtime_samples_ms),
                runtime_breakdown=None,
                metrics_before=metrics_before,
                metrics_after=None,
                moved_atom_count=None,
                focus_atom_rms_displacement_angstrom=None,
                focus_atom_max_displacement_angstrom=None,
                fixed_context_max_displacement_angstrom=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        runtime_samples_ms.append((perf_counter() - start) * 1000.0)
        runtime_profiles.append(runtime_profile)
        final_refined_structure = refinement_result.refined_structure
        final_issues = refinement_result.issues
        moved_atom_count = refinement_result.moved_atom_count()

    assert final_refined_structure is not None
    metrics_after = compute_refinement_metrics(
        final_refined_structure,
        case=case,
        component_library=active_component_library,
        restraint_library=active_restraint_library,
        reference_structure=reference_structure,
        reference_alignment_atom_refs=reference_alignment_atom_refs,
    )
    (
        focus_atom_rms_displacement_angstrom,
        focus_atom_max_displacement_angstrom,
    ) = focus_atom_displacement_metrics(
        perturbed_structure,
        final_refined_structure,
        focus_atom_refs=case.focus_atom_refs,
    )
    fixed_context_max_displacement_angstrom = max_fixed_context_displacement_angstrom(
        perturbed_structure,
        final_refined_structure,
        fixed_atom_refs=reference_alignment_atom_refs,
    )
    status = (
        "success" if moved_atom_count is not None and moved_atom_count > 0 else "no_op"
    )
    error_type = None if status == "success" else "NoOpRefinement"
    error_message = (
        None
        if status == "success"
        else _no_op_refinement_error_message(final_issues)
    )
    return RefinementPerturbationBenchmarkResult(
        case_id=case.case_id,
        perturbation_name=perturbation.name,
        track=track.value,
        execution_profile=normalized_execution_profile.name,
        backend_name=normalized_execution_profile.backend_name,
        method=normalized_execution_profile.method.value,
        strategy_name=strategy.name,
        status=status,
        perturbation_runtime_ms=perturbation_runtime_ms,
        backend_runtime_median_ms=median(backend_runtime_samples_ms),
        backend_runtime_samples_ms=tuple(backend_runtime_samples_ms),
        refinement_runtime_median_ms=median(runtime_samples_ms),
        refinement_runtime_samples_ms=tuple(runtime_samples_ms),
        runtime_breakdown=runtime_breakdown_from_profiles(
            runtime_profiles,
            problem_build_runtime_ms=problem_build_runtime_ms,
        ),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        moved_atom_count=moved_atom_count,
        focus_atom_rms_displacement_angstrom=focus_atom_rms_displacement_angstrom,
        focus_atom_max_displacement_angstrom=focus_atom_max_displacement_angstrom,
        fixed_context_max_displacement_angstrom=(
            fixed_context_max_displacement_angstrom
        ),
        error_type=error_type,
        error_message=error_message,
    )


def _no_op_refinement_error_message(
    issues: tuple[ValidationIssue, ...],
) -> str:
    """Return the most actionable no-op explanation available."""

    if not issues:
        return "local refinement returned without moving any atom slots"

    first_issue = issues[0]
    return (
        "local refinement returned without moving any atom slots: "
        f"{first_issue.kind.value}: {first_issue.message}"
    )


def perturb_reference_structure(
    reference_structure: ProteinStructure,
    *,
    scope_spec: LocalScopeSpec,
    perturbation: RefinementPerturbationSpec,
    component_library: ComponentLibrary,
) -> tuple[ProteinStructure, float]:
    """Return one deterministically perturbed reference structure."""

    start = perf_counter()
    snapshot = ProteinStructureSnapshot.from_structure(reference_structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        scope_spec,
        component_library=component_library,
    )
    movable_atom_refs = atom_input.observed_atom_scope(snapshot).atom_refs
    if not movable_atom_refs:
        raise ValueError("perturbation benchmark requires at least one movable atom")

    moved_positions = perturb_atom_positions(
        reference_structure,
        movable_atom_refs=movable_atom_refs,
        perturbation=perturbation,
    )
    return (
        apply_position_updates(reference_structure, moved_positions=moved_positions),
        (perf_counter() - start) * 1000.0,
    )


def perturb_atom_positions(
    structure: ProteinStructure,
    *,
    movable_atom_refs: tuple[AtomRef, ...],
    perturbation: RefinementPerturbationSpec,
) -> dict[AtomIndex, Vec3]:
    """Return perturbed positions for one realized movable atom set."""

    random_state = RandomState(perturbation.random_seed)
    movable_atom_indices = tuple(
        structure.constitution.atom_index(atom_ref)
        for atom_ref in movable_atom_refs
    )
    movable_positions = np.asarray(
        [
            structure.geometry.atom_geometry(atom_index).position.to_array()
            for atom_index in movable_atom_indices
        ],
        dtype=np.float64,
    )
    centroid = movable_positions.mean(axis=0)
    rotation_matrix = random_rotation_matrix(
        random_state,
        angle_degrees=perturbation.rotation_degrees,
    )
    if perturbation.mode is RefinementPerturbationMode.RANDOM_RIGID:
        translation_direction = random_unit_vector(random_state)
    elif perturbation.mode is RefinementPerturbationMode.DIRECTED_SHOVE:
        translation_direction = nearest_external_translation_direction(
            structure,
            movable_atom_indices=movable_atom_indices,
            centroid=centroid,
        )
    else:
        raise TypeError("perturbation requires a valid RefinementPerturbationMode")

    translation = translation_direction * perturbation.translation_angstrom
    rotated_positions = (
        centroid
        + (movable_positions - centroid) @ rotation_matrix.T
        + translation
    )
    return {
        atom_index: Vec3(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        )
        for atom_index, position in zip(
            movable_atom_indices,
            rotated_positions,
            strict=True,
        )
    }


def random_rotation_matrix(
    random_state: RandomState,
    *,
    angle_degrees: float,
) -> np.ndarray:
    """Return one deterministic 3D rotation matrix."""

    if angle_degrees == 0.0:
        return np.eye(3, dtype=np.float64)

    axis = random_unit_vector(random_state)
    angle_radians = radians(angle_degrees)
    x, y, z = axis.tolist()
    c = cos(angle_radians)
    s = sin(angle_radians)
    one_minus_c = 1.0 - c
    return np.asarray(
        (
            (
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ),
            (
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ),
            (
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ),
        ),
        dtype=np.float64,
    )


def random_unit_vector(random_state: RandomState) -> np.ndarray:
    """Return one deterministic unit vector from the provided random state."""

    while True:
        vector = random_state.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            return vector / norm


def nearest_external_translation_direction(
    structure: ProteinStructure,
    *,
    movable_atom_indices: tuple[AtomIndex, ...],
    centroid: np.ndarray,
) -> np.ndarray:
    """Return one translation direction pointing toward the nearest external atom."""

    movable_atom_index_set = set(movable_atom_indices)
    best_distance = float("inf")
    best_target: np.ndarray | None = None
    for raw_atom_index in range(structure.geometry.atom_count()):
        atom_index = AtomIndex(raw_atom_index)
        if atom_index in movable_atom_index_set:
            continue

        position = structure.geometry.atom_geometry(atom_index).position.to_array()
        distance = float(np.linalg.norm(position - centroid))
        if distance < best_distance:
            best_distance = distance
            best_target = position

    if best_target is None:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

    direction = best_target - centroid
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

    return direction / norm


__all__ = [
    "PERTURBATION_BENCHMARK_SCENARIOS",
    "PERTURBATION_REFINEMENT_BENCHMARK_CASES",
    "REFERENCE_REFINEMENT_BENCHMARK_CASES",
    "REFINEMENT_PERTURBATION_SPECS",
    "SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES",
    "RefinementPerturbationBenchmarkResult",
    "RefinementPerturbationBenchmarkScenario",
    "RefinementPerturbationExpectedOutcome",
    "RefinementPerturbationMode",
    "RefinementPerturbationSpec",
    "perturb_reference_structure",
    "run_refinement_perturbation_benchmark_case",
]
