"""Benchmark helpers for extracted local refinement fixtures."""

from dataclasses import asdict, dataclass
from enum import Enum
from math import sqrt
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from protrepair.chemistry import (
    ComponentLibrary,
    RestraintLibrary,
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.geometry import RigidTransform
from protrepair.io import read_structure
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
    ContinuousRelaxationProfile,
    continuous_relaxation_profile_spec,
)
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalRegionTransformationRuntimeProfile,
    LocalScopeSpec,
    profile_transform_local_region,
    transform_local_region,
)
from protrepair.transformer.refinement.acceptance import (
    RefinementAcceptanceMetrics,
    measure_refinement_acceptance_metrics,
)
from protrepair.transformer.refinement.local_pipeline import (
    resolve_continuous_relaxation_backend,
)
from protrepair.workflow.contracts import LigandPolicy
from tests.support.refinement_cases import (
    ALL_REFINEMENT_BENCHMARK_CASES,
    REFINEMENT_BENCHMARK_CASES,
    RefinementBenchmarkCase,
    RefinementBenchmarkExpectedOutcome,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem
from tests.support.request_builders import ingress_options


class RefinementStrategySelectionMode(str, Enum):
    """Lowering modes exposed by benchmark strategy presets."""

    ATOMS = "atoms"
    RESIDUE_ATOMS = "residue_atoms"
    RESIDUE_SIDECHAINS = "residue_sidechains"


class RefinementBenchmarkTrack(str, Enum):
    """Input-track variants for refinement benchmark execution."""

    HEAVY_ONLY = "heavy-only"
    HYDROGENATED = "hydrogenated"


@dataclass(frozen=True, slots=True)
class RefinementBenchmarkExecutionProfile:
    """One validated execution profile exposed by the benchmark harness."""

    name: str
    profile: ContinuousRelaxationProfile

    @property
    def backend_name(self) -> str:
        """Return the canonical runtime token for this benchmark profile."""

        return continuous_relaxation_profile_spec(self.profile).backend_name

    @property
    def method(self) -> ContinuousRelaxationForceField:
        """Return the canonical local-relaxation method for this profile."""

        return continuous_relaxation_profile_spec(self.profile).method


@dataclass(frozen=True, slots=True)
class RefinementStrategy:
    """One named local-refinement benchmark strategy preset."""

    name: str
    selection_mode: RefinementStrategySelectionMode
    context_radius_angstrom: float
    max_iterations: int

    def build_spec(
        self,
        case: RefinementBenchmarkCase,
        *,
        execution_profile: RefinementBenchmarkExecutionProfile,
    ) -> DirectRegionTransformationSpec:
        """Return one direct-API refinement spec for one benchmark case."""

        if self.selection_mode is RefinementStrategySelectionMode.ATOMS:
            scope_spec = LocalScopeSpec.from_atoms(case.focus_atom_refs)
        elif self.selection_mode is RefinementStrategySelectionMode.RESIDUE_ATOMS:
            scope_spec = LocalScopeSpec.from_residues(case.focus_residue_ids)
        else:
            scope_spec = LocalScopeSpec.from_residue_sidechains(case.focus_residue_ids)

        return DirectRegionTransformationSpec(
            scope_spec=scope_spec,
            force_field=execution_profile.method,
            config=ContinuousRelaxationConfig(
                backend_name=execution_profile.backend_name,
                context_radius_angstrom=self.context_radius_angstrom,
                max_iterations=self.max_iterations,
            ),
        )


@dataclass(frozen=True, slots=True)
class RefinementBenchmarkQualityMetrics:
    """Profile-comparable local quality metrics for one benchmark structure."""

    focus_clash_count: int
    focus_geometry_outlier_count: int
    focus_restraint_backed_geometry_outlier_count: int
    focus_fallback_geometry_outlier_count: int
    focus_severe_restraint_backed_bond_length_outlier_count: int
    focus_clash_overlap_sum_angstrom: float
    focus_near_covalent_contact_count: int
    focus_worst_near_covalent_overlap_angstrom: float
    focus_total_near_covalent_overlap_angstrom: float
    whole_structure_near_covalent_contact_count: int
    whole_structure_worst_near_covalent_overlap_angstrom: float
    whole_structure_total_near_covalent_overlap_angstrom: float
    whole_structure_rdkit_sanitize_readable: bool | None
    reference_aligned_focus_atom_rmsd_angstrom: float | None


@dataclass(frozen=True, slots=True)
class RefinementBenchmarkRuntimeBreakdown:
    """Structured timing breakdown for one benchmarked refinement profile."""

    problem_build_runtime_ms: float
    snapshot_runtime_median_ms: float
    atom_input_lowering_runtime_median_ms: float
    readiness_runtime_median_ms: float
    settings_bind_runtime_median_ms: float
    request_normalization_runtime_median_ms: float
    candidate_construction_runtime_median_ms: float
    execution_candidate_count_median: float
    continuous_backend_candidate_count_median: float
    discrete_only_candidate_count_median: float
    backend_execution_stage_runtime_median_ms: float
    assessment_runtime_median_ms: float
    selection_runtime_median_ms: float
    materialization_runtime_median_ms: float


@dataclass(frozen=True, slots=True)
class RefinementBenchmarkResult:
    """One completed benchmark result for one case and strategy."""

    case_id: str
    expected_outcome: RefinementBenchmarkExpectedOutcome
    expected_success: bool
    expectation_matched: bool
    track: str
    execution_profile: str
    backend_name: str
    method: str
    strategy_name: str
    status: str
    preparation_runtime_ms: float
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

    @property
    def force_field(self) -> str:
        """Return the legacy method token for transitional benchmark callers."""

        return self.method

    def as_serializable_dict(self) -> dict[str, object]:
        """Return one JSON-serializable dictionary representation."""

        return {
            "case_id": self.case_id,
            "expected_outcome": self.expected_outcome.value,
            "expected_success": self.expected_success,
            "expectation_matched": self.expectation_matched,
            "track": self.track,
            "execution_profile": self.execution_profile,
            "backend_name": self.backend_name,
            "method": self.method,
            "strategy_name": self.strategy_name,
            "status": self.status,
            "preparation_runtime_ms": self.preparation_runtime_ms,
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


REFINEMENT_STRATEGIES: dict[str, RefinementStrategy] = {
    "atom-tight": RefinementStrategy(
        name="atom-tight",
        selection_mode=RefinementStrategySelectionMode.ATOMS,
        context_radius_angstrom=3.0,
        max_iterations=200,
    ),
    "sidechain-local": RefinementStrategy(
        name="sidechain-local",
        selection_mode=RefinementStrategySelectionMode.RESIDUE_SIDECHAINS,
        context_radius_angstrom=4.0,
        max_iterations=200,
    ),
    "residue-local": RefinementStrategy(
        name="residue-local",
        selection_mode=RefinementStrategySelectionMode.RESIDUE_ATOMS,
        context_radius_angstrom=4.0,
        max_iterations=200,
    ),
    "residue-wide": RefinementStrategy(
        name="residue-wide",
        selection_mode=RefinementStrategySelectionMode.RESIDUE_ATOMS,
        context_radius_angstrom=6.0,
        max_iterations=300,
    ),
}

REFINEMENT_BENCHMARK_TRACKS: dict[str, RefinementBenchmarkTrack] = {
    track.value: track for track in RefinementBenchmarkTrack
}
REFINEMENT_BENCHMARK_PROFILES: dict[str, RefinementBenchmarkExecutionProfile] = {
    "rdkit_uff": RefinementBenchmarkExecutionProfile(
        name="rdkit_uff",
        profile=ContinuousRelaxationProfile.RDKIT_UFF,
    ),
    "rdkit_mmff": RefinementBenchmarkExecutionProfile(
        name="rdkit_mmff",
        profile=ContinuousRelaxationProfile.RDKIT_MMFF,
    ),
}


def repository_root() -> Path:
    """Return the repository root for benchmark fixtures."""

    return Path(__file__).resolve().parents[2]


def resolve_fixture_path(path: Path) -> Path:
    """Return one repository-relative fixture path as an absolute path."""

    return repository_root() / path


def load_case_structure(case: RefinementBenchmarkCase) -> ProteinStructure:
    """Load one extracted benchmark fixture structure."""

    return read_structure(
        resolve_fixture_path(case.fixture_path),
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )


def prepare_case_structure(
    case: RefinementBenchmarkCase,
    *,
    track: RefinementBenchmarkTrack,
    component_library: ComponentLibrary | None,
) -> tuple[ProteinStructure, float]:
    """Return one benchmark input structure plus preparation runtime."""

    start = perf_counter()
    structure = load_case_structure(case)
    if track is RefinementBenchmarkTrack.HYDROGENATED:
        active_component_library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        structure = _fully_hydrogenate_benchmark_structure(
            structure,
            component_library=active_component_library,
        )

    return structure, (perf_counter() - start) * 1000.0


def load_case_reference_structure(
    case: RefinementBenchmarkCase,
    *,
    track: RefinementBenchmarkTrack,
    component_library: ComponentLibrary | None,
) -> ProteinStructure | None:
    """Load the optional reference structure for one benchmark case."""

    if case.reference_fixture_path is None:
        return None

    reference_structure = read_structure(
        resolve_fixture_path(case.reference_fixture_path),
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )
    if track is RefinementBenchmarkTrack.HYDROGENATED:
        active_component_library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        return _fully_hydrogenate_benchmark_structure(
            reference_structure,
            component_library=active_component_library,
        )

    return reference_structure


def _fully_hydrogenate_benchmark_structure(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
) -> ProteinStructure:
    """Return one benchmark structure with both polymer and retained hydrogens."""

    hydrogenated_structure = add_hydrogens(
        structure,
        component_library=component_library,
        local_refinement=None,
    ).structure
    return add_retained_non_polymer_hydrogens(
        hydrogenated_structure,
        component_library=component_library,
    ).structure


def run_refinement_benchmark_case(
    case: RefinementBenchmarkCase,
    strategy: RefinementStrategy,
    *,
    track: RefinementBenchmarkTrack,
    execution_profile: RefinementBenchmarkExecutionProfile | None = None,
    force_field: ContinuousRelaxationForceField | None = None,
    repeats: int,
    component_library: ComponentLibrary | None = None,
) -> RefinementBenchmarkResult:
    """Run one refinement benchmark case under one named strategy."""

    if repeats <= 0:
        raise ValueError("benchmark repeats must be positive")
    if execution_profile is None and force_field is None:
        raise TypeError(
            "refinement benchmarks require one execution_profile or legacy force_field"
        )
    if execution_profile is not None and force_field is not None:
        raise TypeError(
            "refinement benchmarks must not mix execution_profile with force_field"
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
    original_structure, preparation_runtime_ms = prepare_case_structure(
        case,
        track=track,
        component_library=active_component_library,
    )
    reference_structure = load_case_reference_structure(
        case,
        track=track,
        component_library=active_component_library,
    )
    spec = strategy.build_spec(case, execution_profile=normalized_execution_profile)
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
            original_structure,
            scope_spec=spec.scope_spec,
            backend_name=spec.config.backend_name,
            force_field=spec.force_field,
            context_radius_angstrom=spec.config.context_radius_angstrom,
            max_iterations=spec.config.max_iterations,
            component_library=active_component_library,
        )
        problem_build_runtime_ms = (perf_counter() - build_start) * 1000.0
    except Exception as error:
        problem_build_runtime_ms = 0.0
        metrics_before = compute_refinement_metrics(
            original_structure,
            case=case,
            component_library=active_component_library,
            restraint_library=active_restraint_library,
            reference_structure=reference_structure,
            reference_alignment_atom_refs=(),
        )
        start = perf_counter()
        try:
            transform_local_region(
                original_structure,
                spec,
                component_library=active_component_library,
            )
        except Exception:
            runtime_samples_ms.append((perf_counter() - start) * 1000.0)
            return RefinementBenchmarkResult(
                case_id=case.case_id,
                expected_outcome=case.expected_outcome,
                expected_success=case.expected_success,
                expectation_matched=case.expected_outcome.matches_status("error"),
                track=track.value,
                execution_profile=normalized_execution_profile.name,
                backend_name=normalized_execution_profile.backend_name,
                method=normalized_execution_profile.method.value,
                strategy_name=strategy.name,
                status="error",
                preparation_runtime_ms=preparation_runtime_ms,
                backend_runtime_median_ms=0.0,
                backend_runtime_samples_ms=(),
                refinement_runtime_median_ms=runtime_samples_ms[0],
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

        raise AssertionError(
            "benchmark plan construction failed, but refinement ran"
        ) from error

    metrics_before = compute_refinement_metrics(
        original_structure,
        case=case,
        component_library=active_component_library,
        restraint_library=active_restraint_library,
        reference_structure=reference_structure,
        reference_alignment_atom_refs=tuple(
            plan.region.snapshot.structure.constitution.atom_ref_at(atom_index)
            for atom_index in plan.region.fixed_context_atom_indices
        ),
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
            return RefinementBenchmarkResult(
                case_id=case.case_id,
                expected_outcome=case.expected_outcome,
                expected_success=case.expected_success,
                expectation_matched=case.expected_outcome.matches_status("error"),
                track=track.value,
                execution_profile=normalized_execution_profile.name,
                backend_name=normalized_execution_profile.backend_name,
                method=normalized_execution_profile.method.value,
                strategy_name=strategy.name,
                status="error",
                preparation_runtime_ms=preparation_runtime_ms,
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
                original_structure,
                spec,
                component_library=active_component_library,
            )
        except Exception as error:
            runtime_samples_ms.append((perf_counter() - start) * 1000.0)
            return RefinementBenchmarkResult(
                case_id=case.case_id,
                expected_outcome=case.expected_outcome,
                expected_success=case.expected_success,
                expectation_matched=case.expected_outcome.matches_status("error"),
                track=track.value,
                execution_profile=normalized_execution_profile.name,
                backend_name=normalized_execution_profile.backend_name,
                method=normalized_execution_profile.method.value,
                strategy_name=strategy.name,
                status="error",
                preparation_runtime_ms=preparation_runtime_ms,
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
        reference_alignment_atom_refs=tuple(
            plan.region.snapshot.structure.constitution.atom_ref_at(atom_index)
            for atom_index in plan.region.fixed_context_atom_indices
        ),
    )
    (
        focus_atom_rms_displacement_angstrom,
        focus_atom_max_displacement_angstrom,
    ) = focus_atom_displacement_metrics(
        original_structure,
        final_refined_structure,
        focus_atom_refs=case.focus_atom_refs,
    )
    fixed_context_max_displacement_angstrom = max_fixed_context_displacement_angstrom(
        original_structure,
        final_refined_structure,
        fixed_atom_refs=tuple(
            plan.region.snapshot.structure.constitution.atom_ref_at(atom_index)
            for atom_index in plan.region.fixed_context_atom_indices
        ),
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
    return RefinementBenchmarkResult(
        case_id=case.case_id,
        expected_outcome=case.expected_outcome,
        expected_success=case.expected_success,
        expectation_matched=case.expected_outcome.matches_status(status),
        track=track.value,
        execution_profile=normalized_execution_profile.name,
        backend_name=normalized_execution_profile.backend_name,
        method=normalized_execution_profile.method.value,
        strategy_name=strategy.name,
        status=status,
        preparation_runtime_ms=preparation_runtime_ms,
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


def runtime_breakdown_from_profiles(
    runtime_profiles: list[LocalRegionTransformationRuntimeProfile],
    *,
    problem_build_runtime_ms: float,
) -> RefinementBenchmarkRuntimeBreakdown:
    """Reduce repeated local refinement runtime profiles into one median summary."""

    if not runtime_profiles:
        return RefinementBenchmarkRuntimeBreakdown(
            problem_build_runtime_ms=problem_build_runtime_ms,
            snapshot_runtime_median_ms=0.0,
            atom_input_lowering_runtime_median_ms=0.0,
            readiness_runtime_median_ms=0.0,
            settings_bind_runtime_median_ms=0.0,
            request_normalization_runtime_median_ms=0.0,
            candidate_construction_runtime_median_ms=0.0,
            execution_candidate_count_median=0.0,
            continuous_backend_candidate_count_median=0.0,
            discrete_only_candidate_count_median=0.0,
            backend_execution_stage_runtime_median_ms=0.0,
            assessment_runtime_median_ms=0.0,
            selection_runtime_median_ms=0.0,
            materialization_runtime_median_ms=0.0,
        )

    return RefinementBenchmarkRuntimeBreakdown(
        problem_build_runtime_ms=problem_build_runtime_ms,
        snapshot_runtime_median_ms=median(
            profile.snapshot_runtime_ms for profile in runtime_profiles
        ),
        atom_input_lowering_runtime_median_ms=median(
            profile.atom_input_lowering_runtime_ms for profile in runtime_profiles
        ),
        readiness_runtime_median_ms=median(
            profile.readiness_runtime_ms for profile in runtime_profiles
        ),
        settings_bind_runtime_median_ms=median(
            profile.settings_bind_runtime_ms for profile in runtime_profiles
        ),
        request_normalization_runtime_median_ms=median(
            profile.refinement_pipeline.request_normalization_runtime_ms
            for profile in runtime_profiles
        ),
        candidate_construction_runtime_median_ms=median(
            profile.refinement_pipeline.candidate_construction_runtime_ms
            for profile in runtime_profiles
        ),
        execution_candidate_count_median=float(
            median(
                profile.refinement_pipeline.execution_candidate_count
                for profile in runtime_profiles
            )
        ),
        continuous_backend_candidate_count_median=float(
            median(
                profile.refinement_pipeline.continuous_backend_candidate_count
                for profile in runtime_profiles
            )
        ),
        discrete_only_candidate_count_median=float(
            median(
                profile.refinement_pipeline.discrete_only_candidate_count
                for profile in runtime_profiles
            )
        ),
        backend_execution_stage_runtime_median_ms=median(
            profile.refinement_pipeline.backend_execution_runtime_ms
            for profile in runtime_profiles
        ),
        assessment_runtime_median_ms=median(
            profile.refinement_pipeline.assessment_runtime_ms
            for profile in runtime_profiles
        ),
        selection_runtime_median_ms=median(
            profile.refinement_pipeline.selection_runtime_ms
            for profile in runtime_profiles
        ),
        materialization_runtime_median_ms=median(
            profile.refinement_pipeline.materialization_runtime_ms
            for profile in runtime_profiles
        ),
    )


def compute_refinement_metrics(
    structure: ProteinStructure,
    *,
    case: RefinementBenchmarkCase,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    reference_structure: ProteinStructure | None,
    reference_alignment_atom_refs: tuple[AtomRef, ...] = (),
) -> RefinementBenchmarkQualityMetrics:
    """Return focus-region diagnostics for one benchmark structure."""

    acceptance_metrics = measure_refinement_acceptance_metrics(
        structure,
        component_library=component_library,
        restraint_library=restraint_library,
        focus_residue_ids=frozenset(case.focus_residue_ids),
    )
    return benchmark_quality_metrics_from_acceptance(
        acceptance_metrics,
        reference_aligned_focus_atom_rmsd_angstrom=focus_atom_rmsd_angstrom(
            structure,
            reference_structure=reference_structure,
            focus_atom_refs=case.focus_atom_refs,
            alignment_atom_refs=reference_alignment_atom_refs,
        ),
    )


def benchmark_quality_metrics_from_acceptance(
    acceptance_metrics: RefinementAcceptanceMetrics,
    *,
    reference_aligned_focus_atom_rmsd_angstrom: float | None,
) -> RefinementBenchmarkQualityMetrics:
    """Project acceptance metrics onto the benchmark quality contract."""

    return RefinementBenchmarkQualityMetrics(
        focus_clash_count=acceptance_metrics.focus_clash_count,
        focus_geometry_outlier_count=acceptance_metrics.focus_geometry_outlier_count,
        focus_restraint_backed_geometry_outlier_count=(
            acceptance_metrics.focus_restraint_backed_geometry_outlier_count
        ),
        focus_fallback_geometry_outlier_count=(
            acceptance_metrics.focus_fallback_geometry_outlier_count
        ),
        focus_severe_restraint_backed_bond_length_outlier_count=(
            acceptance_metrics.focus_severe_restraint_backed_bond_length_outlier_count
        ),
        focus_clash_overlap_sum_angstrom=(
            acceptance_metrics.focus_clash_overlap_sum_angstrom
        ),
        focus_near_covalent_contact_count=(
            acceptance_metrics.focus_near_covalent_contact_count
        ),
        focus_worst_near_covalent_overlap_angstrom=(
            acceptance_metrics.focus_worst_near_covalent_overlap_angstrom
        ),
        focus_total_near_covalent_overlap_angstrom=(
            acceptance_metrics.focus_total_near_covalent_overlap_angstrom
        ),
        whole_structure_near_covalent_contact_count=(
            acceptance_metrics.whole_structure_near_covalent_contact_count
        ),
        whole_structure_worst_near_covalent_overlap_angstrom=(
            acceptance_metrics.whole_structure_worst_near_covalent_overlap_angstrom
        ),
        whole_structure_total_near_covalent_overlap_angstrom=(
            acceptance_metrics.whole_structure_total_near_covalent_overlap_angstrom
        ),
        whole_structure_rdkit_sanitize_readable=(
            acceptance_metrics.whole_structure_rdkit_sanitize_readable
        ),
        reference_aligned_focus_atom_rmsd_angstrom=(
            reference_aligned_focus_atom_rmsd_angstrom
        ),
    )


def focus_atom_displacement_metrics(
    original_structure: ProteinStructure,
    refined_structure: ProteinStructure,
    *,
    focus_atom_refs: tuple[AtomRef, ...],
) -> tuple[float | None, float | None]:
    """Return RMS and max displacement over focus atoms present in both structures."""

    squared_distances: list[float] = []
    max_distance = 0.0
    for atom_ref in focus_atom_refs:
        original_atom = resolve_structure_atom(original_structure, atom_ref)
        refined_atom = resolve_structure_atom(refined_structure, atom_ref)
        if original_atom is None or refined_atom is None:
            continue

        distance = float(
            np.linalg.norm(
                original_atom.position.to_array() - refined_atom.position.to_array()
            )
        )
        squared_distances.append(distance * distance)
        if distance > max_distance:
            max_distance = distance

    if not squared_distances:
        return None, None

    return sqrt(sum(squared_distances) / len(squared_distances)), max_distance


def focus_atom_rmsd_angstrom(
    structure: ProteinStructure,
    *,
    reference_structure: ProteinStructure | None,
    focus_atom_refs: tuple[AtomRef, ...],
    alignment_atom_refs: tuple[AtomRef, ...],
) -> float | None:
    """Return fixed-context-aligned RMSD to a reference over focus atoms."""

    if reference_structure is None:
        return None

    alignment_pairs = paired_atom_arrays(
        structure,
        reference_structure=reference_structure,
        atom_refs=alignment_atom_refs,
    )
    if alignment_pairs is None:
        return None

    moving_alignment_points, reference_alignment_points = alignment_pairs
    transform = RigidTransform.fit(
        moving_alignment_points,
        reference_alignment_points,
    )
    if transform is None:
        return None

    squared_distances: list[float] = []
    for atom_ref in focus_atom_refs:
        structure_atom = resolve_structure_atom(structure, atom_ref)
        reference_atom = resolve_structure_atom(reference_structure, atom_ref)
        if structure_atom is None or reference_atom is None:
            continue

        aligned_position = transform.apply(structure_atom.position.to_array())
        distance = float(
            np.linalg.norm(aligned_position - reference_atom.position.to_array())
        )
        squared_distances.append(distance * distance)

    if not squared_distances:
        return None

    return sqrt(sum(squared_distances) / len(squared_distances))


def paired_atom_arrays(
    structure: ProteinStructure,
    *,
    reference_structure: ProteinStructure,
    atom_refs: tuple[AtomRef, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return paired coordinate arrays for atom refs present in both structures."""

    moving_points: list[np.ndarray] = []
    reference_points: list[np.ndarray] = []
    for atom_ref in atom_refs:
        structure_atom = resolve_structure_atom(structure, atom_ref)
        reference_atom = resolve_structure_atom(reference_structure, atom_ref)
        if structure_atom is None or reference_atom is None:
            continue

        moving_points.append(structure_atom.position.to_array())
        reference_points.append(reference_atom.position.to_array())

    if len(moving_points) < 3:
        return None

    return np.asarray(moving_points), np.asarray(reference_points)


def max_fixed_context_displacement_angstrom(
    original_structure: ProteinStructure,
    refined_structure: ProteinStructure,
    *,
    fixed_atom_refs: tuple[AtomRef, ...],
) -> float:
    """Return the maximum fixed-context atom displacement after refinement."""

    if not fixed_atom_refs:
        return 0.0

    displacement_values: list[float] = []
    for atom_ref in fixed_atom_refs:
        original_atom = resolve_structure_atom(original_structure, atom_ref)
        refined_atom = resolve_structure_atom(refined_structure, atom_ref)
        if original_atom is None or refined_atom is None:
            continue

        displacement_values.append(original_atom.distance_to(refined_atom))

    return 0.0 if not displacement_values else max(displacement_values)


def resolve_structure_atom(
    structure: ProteinStructure,
    atom_ref: AtomRef,
):
    """Return one atom by atom reference from chains or ligands."""

    return structure.geometry.atom_geometry(structure.constitution.atom_index(atom_ref))


__all__ = [
    "ALL_REFINEMENT_BENCHMARK_CASES",
    "REFINEMENT_BENCHMARK_CASES",
    "REFINEMENT_BENCHMARK_PROFILES",
    "REFINEMENT_BENCHMARK_TRACKS",
    "REFINEMENT_STRATEGIES",
    "RefinementBenchmarkCase",
    "RefinementBenchmarkExpectedOutcome",
    "RefinementBenchmarkExecutionProfile",
    "RefinementBenchmarkTrack",
    "RefinementBenchmarkResult",
    "RefinementBenchmarkQualityMetrics",
    "RefinementStrategy",
    "RefinementStrategySelectionMode",
    "benchmark_quality_metrics_from_acceptance",
    "compute_refinement_metrics",
    "focus_atom_displacement_metrics",
    "load_case_structure",
    "prepare_case_structure",
    "run_refinement_benchmark_case",
]
