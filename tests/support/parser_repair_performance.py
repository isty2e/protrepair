"""Opt-in parser-witness repair performance probes for whole-structure cases."""

import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import ParamSpec, TypeVar

from protrepair.chemistry import (
    ComponentLibrary,
    RestraintLibrary,
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.diagnostics.clashes import detect_clashes_involving_residues
from protrepair.diagnostics.geometry import detect_heavy_geometry
from protrepair.diagnostics.parser_readability import (
    RDKitParserProblemWitness,
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationMethod,
)
from protrepair.transformer.discrete import (
    parser_witness_pre_untangle as parser_pre_untangle,
)
from protrepair.transformer.local import (
    LocalScopeSpec,
    atom_input_from_local_scope_spec,
)
from protrepair.transformer.refinement.acceptance import RefinementAcceptanceMetrics
from protrepair.transformer.refinement.local_pipeline.assessment import (
    assess_refinement_candidate_batch,
    execute_and_assess_refinement_candidate_batch,
    execute_refinement_candidate_batch,
    materialize_selected_refinement_candidate,
    select_refinement_candidate,
)
from protrepair.transformer.refinement.local_pipeline.candidates import (
    AssessedRefinementBatch,
    RefinementExecutionBatch,
)
from protrepair.transformer.refinement.local_pipeline.construction import (
    build_refinement_execution_batch,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
    normalize_local_refinement_request,
)
from protrepair.transformer.source_microstate_adjudication import (
    adjudicate_source_microstate_contradictions,
)
from protrepair.workflow.contracts.policies import LigandPolicy
from protrepair.workflow.contracts.request import StructureIngressOptions
from protrepair.workflow.engine import normalize_source_structure
from tests.support.refinement_corpus import (
    ALL_CORPUS_METADATA,
    CorpusExecutionPolicy,
)
from tests.support.whole_structure_sources import WHOLE_STRUCTURE_CORPUS_SOURCES

P = ParamSpec("P")
T = TypeVar("T")
HYDROGENATED_CACHE_SCHEMA_VERSION = 2
PARSER_REPAIR_PERFORMANCE_SCHEMA_VERSION = 3


class ParserRepairProbeMode(str, Enum):
    """Execution modes for parser-repair performance probes."""

    PRODUCTION_LIKE = "production_like"
    DIAGNOSTIC_ALL_CANDIDATES = "diagnostic_all_candidates"

    @classmethod
    def from_value(
        cls,
        value: "ParserRepairProbeMode | str",
    ) -> "ParserRepairProbeMode":
        """Return the parser-repair probe mode for one CLI/runtime value."""

        if isinstance(value, cls):
            return value

        try:
            return cls(value)
        except ValueError as error:
            valid_modes = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"unknown parser-repair probe mode {value!r}; expected one of "
                f"{valid_modes}"
            ) from error


class ParserBurdenSignal(str, Enum):
    """Whether a parser-repair probe actually observed parser burden."""

    ABSENT = "absent"
    PRESENT = "present"


class ParserRepairQualitySignal(str, Enum):
    """Parser-repair outcome semantics independent of runtime tracking."""

    NO_PARSER_BURDEN = "no_parser_burden"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"


class RuntimeTrackingSignal(str, Enum):
    """Runtime interpretation policy for one whole-structure probe case."""

    GATING_THRESHOLD_ELIGIBLE = "gating_threshold_eligible"
    TRACKING_ONLY = "tracking_only"


@dataclass(frozen=True, slots=True)
class ParserRepairFocusQuality:
    """Minimal local quality metrics needed for parser-repair budgeting."""

    rdkit_sanitize_readable: bool | None
    focus_clash_count: int
    focus_geometry_outlier_count: int


@dataclass(frozen=True, slots=True)
class ParserRepairAcceptanceMetrics:
    """Serializable subset of local-refinement acceptance metrics."""

    focus_clash_count: int
    focus_geometry_outlier_count: int
    focus_restraint_backed_geometry_outlier_count: int
    focus_fallback_geometry_outlier_count: int
    focus_severe_restraint_backed_bond_length_outlier_count: int
    focus_clash_overlap_sum_angstrom: float
    focus_near_covalent_contact_count: int
    focus_worst_near_covalent_overlap_angstrom: float
    focus_total_near_covalent_overlap_angstrom: float
    focus_stereochemistry_violation_count: int
    whole_structure_near_covalent_contact_count: int
    whole_structure_worst_near_covalent_overlap_angstrom: float
    whole_structure_total_near_covalent_overlap_angstrom: float
    whole_structure_rdkit_sanitize_readable: bool | None
    whole_structure_parser_extra_proximity_bond_count: int
    whole_structure_parser_extra_heavy_proximity_bond_count: int


@dataclass(frozen=True, slots=True)
class ParserRepairCandidateSummary:
    """One assessed refinement candidate in the performance probe."""

    index: int
    verdict: str
    before: ParserRepairAcceptanceMetrics
    after: ParserRepairAcceptanceMetrics


@dataclass(frozen=True, slots=True)
class ParserRepairHydrogenatedCache:
    """Versioned parser-repair hydrogenated structure cache payload."""

    schema_version: int
    structure: ProteinStructure

    def is_current(self) -> bool:
        """Return whether this cache payload matches current preparation semantics."""

        return self.schema_version == HYDROGENATED_CACHE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ParserRepairClusterSummary:
    """The parser-witness cluster selected for one probe run."""

    residue_ids: tuple[str, ...]
    extra_bond_count: int
    extra_bonds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParserRepairProbeSignal:
    """Orthogonal signal summary for parser-repair performance probes."""

    parser_burden: ParserBurdenSignal
    repair_quality: ParserRepairQualitySignal
    runtime_tracking: RuntimeTrackingSignal

    def as_serializable_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation of the probe signals."""

        return {
            "parser_burden": self.parser_burden.value,
            "repair_quality": self.repair_quality.value,
            "runtime_tracking": self.runtime_tracking.value,
        }


@dataclass(frozen=True, slots=True)
class ParserRepairPerformanceResult:
    """Stage timings and quality deltas from one parser-repair probe."""

    schema_version: int
    case_id: str
    mode: str
    hydrogen_cache_used: bool
    timings_sec: dict[str, float]
    cluster: ParserRepairClusterSummary
    candidate_count: int
    executed_count: int
    execution_error_count: int
    candidate_summaries: tuple[ParserRepairCandidateSummary, ...]
    before_witness_count: int
    before_heavy_witness_count: int
    after_witness_count: int
    after_heavy_witness_count: int
    before_focus: ParserRepairFocusQuality
    after_focus: ParserRepairFocusQuality
    backend_name: str
    issue_count: int

    def probe_signal(self) -> ParserRepairProbeSignal:
        """Return orthogonal parser-burden, repair, and runtime signals."""

        return classify_parser_repair_performance(self)

    def as_serializable_dict(self) -> dict[
        str,
        str
        | bool
        | int
        | float
        | None
        | dict[str, float]
        | dict[str, str]
        | dict[str, str | int | list[str]]
        | dict[str, int | bool | None]
        | list[dict[str, str | int | dict[str, int | float | bool | None]]],
    ]:
        """Return a JSON-compatible representation of this probe result."""

        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "mode": self.mode,
            "hydrogen_cache_used": self.hydrogen_cache_used,
            "timings_sec": self.timings_sec,
            "cluster": {
                "residue_ids": list(self.cluster.residue_ids),
                "extra_bond_count": self.cluster.extra_bond_count,
                "extra_bonds": list(self.cluster.extra_bonds),
            },
            "candidate_count": self.candidate_count,
            "executed_count": self.executed_count,
            "execution_error_count": self.execution_error_count,
            "candidate_summaries": [
                {
                    "index": candidate.index,
                    "verdict": candidate.verdict,
                    "before": asdict(candidate.before),
                    "after": asdict(candidate.after),
                }
                for candidate in self.candidate_summaries
            ],
            "before_witness_count": self.before_witness_count,
            "before_heavy_witness_count": self.before_heavy_witness_count,
            "after_witness_count": self.after_witness_count,
            "after_heavy_witness_count": self.after_heavy_witness_count,
            "before_focus": asdict(self.before_focus),
            "after_focus": asdict(self.after_focus),
            "backend_name": self.backend_name,
            "issue_count": self.issue_count,
            "signal": self.probe_signal().as_serializable_dict(),
        }


def classify_parser_repair_performance(
    result: ParserRepairPerformanceResult,
) -> ParserRepairProbeSignal:
    """Classify parser-repair probe semantics without conflating timing axes."""

    parser_burden = _parser_burden_signal(result)
    return ParserRepairProbeSignal(
        parser_burden=parser_burden,
        repair_quality=_parser_repair_quality_signal(
            result,
            parser_burden=parser_burden,
        ),
        runtime_tracking=_runtime_tracking_signal(result.case_id),
    )


def run_first_parser_cluster_repair_probe(
    *,
    case_id: str = "3ja8-whole-structure",
    mode: ParserRepairProbeMode | str = ParserRepairProbeMode.PRODUCTION_LIKE,
    use_hydrogen_cache: bool = False,
    hydrogen_cache_path: Path | None = None,
    context_radius_angstrom: float = 3.0,
    max_iterations: int = 20,
) -> ParserRepairPerformanceResult:
    """Run a sequential first-cluster parser-witness local repair probe."""

    if case_id not in WHOLE_STRUCTURE_CORPUS_SOURCES:
        raise ValueError(f"unknown whole-structure corpus case id: {case_id}")

    if context_radius_angstrom <= 0.0:
        raise ValueError("context_radius_angstrom must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    probe_mode = ParserRepairProbeMode.from_value(mode)
    _disable_rdkit_logs()

    timings: dict[str, float] = {}
    component_library = build_default_component_library()
    restraint_library = build_default_restraint_library()
    structure, cache_used = _load_or_prepare_hydrogenated_structure(
        case_id=case_id,
        component_library=component_library,
        timings=timings,
        use_hydrogen_cache=use_hydrogen_cache,
        hydrogen_cache_path=hydrogen_cache_path,
    )
    before_parser_probe = _timed(
        "before_parser_probe",
        lambda: probe_rdkit_no_conect_parser_readability(
            structure,
            component_library=component_library,
        ),
        timings,
    )
    before_parser_metrics = _timed(
        "before_parser_metrics",
        before_parser_probe.metrics,
        timings,
    )
    clusters = _timed(
        "parser_clusters",
        before_parser_probe.extra_proximity_bond_clusters,
        timings,
    )
    if not clusters:
        raise RuntimeError(f"{case_id} produced no parser-witness clusters")

    cluster = clusters[0]
    residue_ids = frozenset(cluster.residue_ids)
    before_witnesses = _timed(
        "before_witnesses",
        before_parser_probe.parser_problem_witnesses,
        timings,
    )
    before_focus = _timed(
        "before_focus_quality",
        lambda: _focus_quality_counts(
            structure,
            residue_ids=residue_ids,
            component_library=component_library,
            restraint_library=restraint_library,
            rdkit_sanitize_readable=before_parser_metrics.sanitize_readable,
        ),
        timings,
    )
    snapshot = _timed(
        "snapshot",
        lambda: ProteinStructureSnapshot.from_structure(structure),
        timings,
    )
    scope_spec = LocalScopeSpec.from_residues(cluster.residue_ids)
    atom_input = _timed(
        "scope_lowering",
        lambda: atom_input_from_local_scope_spec(
            snapshot,
            scope_spec,
            component_library=component_library,
        ),
        timings,
    )
    context = ProteinTransformationContext.from_snapshot_atom_input(
        snapshot,
        atom_input,
    )
    settings = ContinuousRelaxationConfig(
        backend_name="rdkit",
        context_radius_angstrom=context_radius_angstrom,
        max_iterations=max_iterations,
    ).bind(ContinuousRelaxationMethod.UFF)
    request = _timed(
        "request_normalization",
        lambda: normalize_local_refinement_request(
            context,
            spec=settings,
            component_library=component_library,
            restraint_library=restraint_library,
        ),
        timings,
    )
    batch = _timed(
        "candidate_construction",
        lambda: build_refinement_execution_batch_profiled(
            request,
            timings=timings,
        ),
        timings,
    )
    if probe_mode is ParserRepairProbeMode.PRODUCTION_LIKE:
        assessed_batch = _execute_and_assess_production_like(
            batch,
            request=request,
            timings=timings,
        )
    else:
        assessed_batch = _execute_and_assess_all_candidates(
            batch,
            request=request,
            timings=timings,
        )
    selected_candidate = _timed(
        "selection",
        lambda: select_refinement_candidate(assessed_batch),
        timings,
    )
    result = _timed(
        "materialization",
        lambda: materialize_selected_refinement_candidate(
            selected_candidate,
            request=request,
        ),
        timings,
    )
    after_parser_probe = _timed(
        "after_parser_probe",
        lambda: probe_rdkit_no_conect_parser_readability(
            result.refined_structure,
            component_library=component_library,
        ),
        timings,
    )
    after_parser_metrics = _timed(
        "after_parser_metrics",
        after_parser_probe.metrics,
        timings,
    )
    after_witnesses = _timed(
        "after_witnesses",
        after_parser_probe.parser_problem_witnesses,
        timings,
    )
    after_focus = _timed(
        "after_focus_quality",
        lambda: _focus_quality_counts(
            result.refined_structure,
            residue_ids=residue_ids,
            component_library=component_library,
            restraint_library=restraint_library,
            rdkit_sanitize_readable=after_parser_metrics.sanitize_readable,
        ),
        timings,
    )

    return ParserRepairPerformanceResult(
        schema_version=PARSER_REPAIR_PERFORMANCE_SCHEMA_VERSION,
        case_id=case_id,
        mode=probe_mode.value,
        hydrogen_cache_used=cache_used,
        timings_sec=timings,
        cluster=ParserRepairClusterSummary(
            residue_ids=tuple(
                residue_id.display_token() for residue_id in cluster.residue_ids
            ),
            extra_bond_count=len(cluster.bonds),
            extra_bonds=tuple(bond.display_token() for bond in cluster.bonds),
        ),
        candidate_count=len(batch.candidates),
        executed_count=len(assessed_batch.evaluated_proposals),
        execution_error_count=len(assessed_batch.execution_errors),
        candidate_summaries=tuple(
            ParserRepairCandidateSummary(
                index=index,
                verdict=candidate.evaluation.verdict.value,
                before=_acceptance_metrics_snapshot(
                    candidate.evaluation.before_metrics
                ),
                after=_acceptance_metrics_snapshot(candidate.evaluation.after_metrics),
            )
            for index, candidate in enumerate(assessed_batch.evaluated_proposals)
        ),
        before_witness_count=_extra_proximity_bond_count(before_witnesses),
        before_heavy_witness_count=_extra_heavy_proximity_bond_count(
            before_witnesses
        ),
        after_witness_count=_extra_proximity_bond_count(after_witnesses),
        after_heavy_witness_count=_extra_heavy_proximity_bond_count(after_witnesses),
        before_focus=before_focus,
        after_focus=after_focus,
        backend_name=result.backend_name,
        issue_count=len(result.issues),
    )


def _execute_and_assess_production_like(
    batch: RefinementExecutionBatch,
    *,
    request: LocalRefinementRequest,
    timings: dict[str, float],
) -> AssessedRefinementBatch:
    """Execute and assess one batch through the production short-circuit path."""

    assessed_batch, backend_runtime_ms, assessment_runtime_ms = (
        execute_and_assess_refinement_candidate_batch(
            batch,
            request=request,
        )
    )
    timings["backend_execution"] = backend_runtime_ms / 1000.0
    timings["assessment"] = assessment_runtime_ms / 1000.0
    return assessed_batch


def build_refinement_execution_batch_profiled(
    request: LocalRefinementRequest,
    *,
    timings: dict[str, float],
) -> RefinementExecutionBatch:
    """Build candidates while exposing parser pre-untangle substage timings."""

    original_cluster_probe = (
        parser_pre_untangle.rdkit_no_conect_extra_proximity_bond_clusters
    )
    original_build_candidate = (
        parser_pre_untangle._build_ranked_parser_witness_pre_untangle_candidate
    )
    original_rank_candidates = (
        parser_pre_untangle._ranked_parser_witness_pre_untangle_candidates
    )
    original_payload_builder = parser_pre_untangle._build_rotated_sidechain_payload
    original_materializer = (
        parser_pre_untangle._materialize_rotated_sidechain_candidate
    )
    original_parser_count = parser_pre_untangle._parser_extra_heavy_proximity_bond_count
    original_full_rank = (
        parser_pre_untangle._parser_witness_pre_untangle_candidate_rank
    )

    parser_pre_untangle.rdkit_no_conect_extra_proximity_bond_clusters = (
        _profiled_callable(
            "candidate_construction.parser_pre_untangle.cluster_probe",
            original_cluster_probe,
            timings,
        )
    )
    parser_pre_untangle._build_ranked_parser_witness_pre_untangle_candidate = (
        _profiled_callable(
            "candidate_construction.parser_pre_untangle.build_candidate",
            original_build_candidate,
            timings,
        )
    )
    parser_pre_untangle._ranked_parser_witness_pre_untangle_candidates = (
        _profiled_callable(
            "candidate_construction.parser_pre_untangle.rank_candidates",
            original_rank_candidates,
            timings,
        )
    )
    parser_pre_untangle._build_rotated_sidechain_payload = _profiled_callable(
        "candidate_construction.parser_pre_untangle.payload",
        original_payload_builder,
        timings,
    )
    parser_pre_untangle._materialize_rotated_sidechain_candidate = _profiled_callable(
        "candidate_construction.parser_pre_untangle.materialize",
        original_materializer,
        timings,
    )
    parser_pre_untangle._parser_extra_heavy_proximity_bond_count = _profiled_callable(
        "candidate_construction.parser_pre_untangle.parser_count",
        original_parser_count,
        timings,
    )
    parser_pre_untangle._parser_witness_pre_untangle_candidate_rank = (
        _profiled_callable(
            "candidate_construction.parser_pre_untangle.full_rank",
            original_full_rank,
            timings,
        )
    )
    try:
        return build_refinement_execution_batch(request)
    finally:
        parser_pre_untangle.rdkit_no_conect_extra_proximity_bond_clusters = (
            original_cluster_probe
        )
        parser_pre_untangle._build_ranked_parser_witness_pre_untangle_candidate = (
            original_build_candidate
        )
        parser_pre_untangle._ranked_parser_witness_pre_untangle_candidates = (
            original_rank_candidates
        )
        parser_pre_untangle._build_rotated_sidechain_payload = (
            original_payload_builder
        )
        parser_pre_untangle._materialize_rotated_sidechain_candidate = (
            original_materializer
        )
        parser_pre_untangle._parser_extra_heavy_proximity_bond_count = (
            original_parser_count
        )
        parser_pre_untangle._parser_witness_pre_untangle_candidate_rank = (
            original_full_rank
        )


def _execute_and_assess_all_candidates(
    batch: RefinementExecutionBatch,
    *,
    request: LocalRefinementRequest,
    timings: dict[str, float],
) -> AssessedRefinementBatch:
    """Execute and assess every built candidate for diagnostic comparison."""

    executed_batch = _timed(
        "backend_execution",
        lambda: execute_refinement_candidate_batch(batch, request=request),
        timings,
    )
    return _timed(
        "assessment",
        lambda: assess_refinement_candidate_batch(executed_batch, request=request),
        timings,
    )


def _load_or_prepare_hydrogenated_structure(
    *,
    case_id: str,
    component_library: ComponentLibrary,
    timings: dict[str, float],
    use_hydrogen_cache: bool,
    hydrogen_cache_path: Path | None,
) -> tuple[ProteinStructure, bool]:
    """Return a fully hydrogenated whole-structure corpus input."""

    cache_path = hydrogen_cache_path or Path(
        f"/tmp/protrepair_{case_id}_hydrogenated.pkl"
    )
    if use_hydrogen_cache and cache_path.exists():
        cached_structure = _timed(
            "load_hydrogenated_cache",
            lambda: _current_hydrogenated_cache_structure(cache_path),
            timings,
        )
        if cached_structure is not None:
            return cached_structure, True

    case = WHOLE_STRUCTURE_CORPUS_SOURCES[case_id]
    normalized = _timed(
        "normalize",
        lambda: normalize_source_structure(
            case.output_path,
            ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        ),
        timings,
    )
    adjudicated_structure = _timed(
        "microstate_adjudication",
        lambda: adjudicate_source_microstate_contradictions(
            normalized,
            component_library=component_library,
        )[0],
        timings,
    )
    polymer_result = _timed(
        "hydrogenate_polymer",
        lambda: add_hydrogens(
            adjudicated_structure,
            component_library=component_library,
            local_refinement=None,
        ),
        timings,
    )
    retained_result = _timed(
        "hydrogenate_retained",
        lambda: add_retained_non_polymer_hydrogens(
            polymer_result.structure,
            component_library=component_library,
        ),
        timings,
    )
    structure = retained_result.structure
    if use_hydrogen_cache:
        _timed(
            "write_hydrogenated_cache",
            lambda: cache_path.write_bytes(
                pickle.dumps(
                    ParserRepairHydrogenatedCache(
                        schema_version=HYDROGENATED_CACHE_SCHEMA_VERSION,
                        structure=structure,
                    )
                )
            ),
            timings,
        )

    return structure, False


def _current_hydrogenated_cache_structure(
    cache_path: Path,
) -> ProteinStructure | None:
    """Return a current cache structure or None when the cache is stale."""

    payload = pickle.loads(cache_path.read_bytes())
    if not isinstance(payload, ParserRepairHydrogenatedCache):
        return None
    if not payload.is_current():
        return None

    return payload.structure


def _timed(name: str, action: Callable[[], T], timings: dict[str, float]) -> T:
    """Run an action and record its wall-clock runtime in seconds."""

    start = perf_counter()
    value = action()
    timings[name] = perf_counter() - start
    return value


def _profiled_callable(
    name: str,
    action: Callable[P, T],
    timings: dict[str, float],
) -> Callable[P, T]:
    """Return an action wrapper that accumulates runtime in seconds."""

    def profiled_action(*args: P.args, **kwargs: P.kwargs) -> T:
        start = perf_counter()
        try:
            return action(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + perf_counter() - start

    return profiled_action


def _disable_rdkit_logs() -> None:
    """Suppress noisy RDKit parser warnings during opt-in probe runs."""

    try:
        from rdkit import RDLogger
    except ImportError:
        return

    disable_log = getattr(RDLogger, "DisableLog", None)
    if callable(disable_log):
        disable_log("rdApp.*")


def _parser_burden_signal(
    result: ParserRepairPerformanceResult,
) -> ParserBurdenSignal:
    """Return whether this probe actually had parser burden to repair."""

    if (
        result.before_witness_count > 0
        or result.before_heavy_witness_count > 0
        or result.cluster.extra_bond_count > 0
    ):
        return ParserBurdenSignal.PRESENT

    return ParserBurdenSignal.ABSENT


def _parser_repair_quality_signal(
    result: ParserRepairPerformanceResult,
    *,
    parser_burden: ParserBurdenSignal,
) -> ParserRepairQualitySignal:
    """Return parser-repair outcome semantics for one measured result."""

    if parser_burden is ParserBurdenSignal.ABSENT:
        return ParserRepairQualitySignal.NO_PARSER_BURDEN
    if result.after_heavy_witness_count < result.before_heavy_witness_count:
        return ParserRepairQualitySignal.IMPROVED
    if result.after_witness_count < result.before_witness_count:
        return ParserRepairQualitySignal.IMPROVED
    if result.after_heavy_witness_count > result.before_heavy_witness_count:
        return ParserRepairQualitySignal.REGRESSED
    if result.after_witness_count > result.before_witness_count:
        return ParserRepairQualitySignal.REGRESSED

    return ParserRepairQualitySignal.UNCHANGED


def _runtime_tracking_signal(case_id: str) -> RuntimeTrackingSignal:
    """Return the runtime interpretation policy for one corpus case."""

    metadata = ALL_CORPUS_METADATA.get(case_id)
    if metadata is None:
        return RuntimeTrackingSignal.TRACKING_ONLY
    if metadata.execution_policy is CorpusExecutionPolicy.TRACKING_ONLY:
        return RuntimeTrackingSignal.TRACKING_ONLY

    return RuntimeTrackingSignal.GATING_THRESHOLD_ELIGIBLE


def _focus_quality_counts(
    structure: ProteinStructure,
    *,
    residue_ids: frozenset[ResidueId],
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    rdkit_sanitize_readable: bool | None,
) -> ParserRepairFocusQuality:
    """Return focused parser-repair quality metrics for one structure."""

    clash_report = detect_clashes_involving_residues(
        structure,
        residue_ids=residue_ids,
        component_library=component_library,
    )
    geometry_report = detect_heavy_geometry(
        structure,
        component_library=component_library,
        restraint_library=restraint_library,
        residue_ids=residue_ids,
    )
    return ParserRepairFocusQuality(
        rdkit_sanitize_readable=rdkit_sanitize_readable,
        focus_clash_count=len(clash_report.clashes),
        focus_geometry_outlier_count=(
            len(geometry_report.bond_length_outliers)
            + len(geometry_report.bond_angle_outliers)
        ),
    )


def _acceptance_metrics_snapshot(
    metrics: RefinementAcceptanceMetrics,
) -> ParserRepairAcceptanceMetrics:
    """Return the serializable acceptance-metric subset used by this probe."""

    return ParserRepairAcceptanceMetrics(
        focus_clash_count=metrics.focus_clash_count,
        focus_geometry_outlier_count=metrics.focus_geometry_outlier_count,
        focus_restraint_backed_geometry_outlier_count=(
            metrics.focus_restraint_backed_geometry_outlier_count
        ),
        focus_fallback_geometry_outlier_count=(
            metrics.focus_fallback_geometry_outlier_count
        ),
        focus_severe_restraint_backed_bond_length_outlier_count=(
            metrics.focus_severe_restraint_backed_bond_length_outlier_count
        ),
        focus_clash_overlap_sum_angstrom=metrics.focus_clash_overlap_sum_angstrom,
        focus_near_covalent_contact_count=metrics.focus_near_covalent_contact_count,
        focus_worst_near_covalent_overlap_angstrom=(
            metrics.focus_worst_near_covalent_overlap_angstrom
        ),
        focus_total_near_covalent_overlap_angstrom=(
            metrics.focus_total_near_covalent_overlap_angstrom
        ),
        focus_stereochemistry_violation_count=(
            metrics.focus_stereochemistry_violation_count
        ),
        whole_structure_near_covalent_contact_count=(
            metrics.whole_structure_near_covalent_contact_count
        ),
        whole_structure_worst_near_covalent_overlap_angstrom=(
            metrics.whole_structure_worst_near_covalent_overlap_angstrom
        ),
        whole_structure_total_near_covalent_overlap_angstrom=(
            metrics.whole_structure_total_near_covalent_overlap_angstrom
        ),
        whole_structure_rdkit_sanitize_readable=(
            metrics.whole_structure_rdkit_sanitize_readable
        ),
        whole_structure_parser_extra_proximity_bond_count=(
            metrics.whole_structure_parser_extra_proximity_bond_count
        ),
        whole_structure_parser_extra_heavy_proximity_bond_count=(
            metrics.whole_structure_parser_extra_heavy_proximity_bond_count
        ),
    )


def _extra_proximity_bond_count(
    witnesses: tuple[RDKitParserProblemWitness, ...],
) -> int:
    """Return the total extra proximity-bond witness count."""

    return sum(len(witness.extra_proximity_bonds()) for witness in witnesses)


def _extra_heavy_proximity_bond_count(
    witnesses: tuple[RDKitParserProblemWitness, ...],
) -> int:
    """Return the total extra heavy-heavy proximity-bond witness count."""

    return sum(len(witness.extra_heavy_proximity_bonds()) for witness in witnesses)
