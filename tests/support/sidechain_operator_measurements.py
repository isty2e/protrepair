"""Benchmark-only side-chain operator measurements for refinement gap cases."""

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import product
from math import pi
from time import perf_counter

from protrepair.chemistry import (
    ComponentLibrary,
    RestraintLibrary,
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.discrete.models import BranchedSidechainRotationStep
from protrepair.transformer.discrete.orientation import (
    attached_hydrogen_atom_names,
    replace_residue,
    rotate_residue_atoms_about_axis,
)
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    transform_local_region,
)
from tests.support.refinement_benchmarks import (
    RefinementBenchmarkQualityMetrics,
    RefinementBenchmarkTrack,
    compute_refinement_metrics,
    load_case_reference_structure,
    prepare_case_structure,
)
from tests.support.refinement_cases import RefinementBenchmarkCase


class SidechainOperatorSignal(str, Enum):
    """Planning interpretation for one side-chain operator measurement."""

    PRODUCTION_PLANNING_SIGNAL = "production_planning_signal"
    TRACKING_ONLY_DIAGNOSTIC_SIGNAL = "tracking_only_diagnostic_signal"
    UNSUITABLE = "unsuitable"


@dataclass(frozen=True, slots=True)
class SidechainRotamerGridDefinition:
    """Benchmark-only torsion grid for one side-chain component."""

    component_id: str
    rotation_steps: tuple[BranchedSidechainRotationStep, ...]


@dataclass(frozen=True, slots=True)
class SidechainOperatorCandidateMeasurement:
    """Quality metrics for one side-chain operator candidate."""

    label: str
    metrics_after: RefinementBenchmarkQualityMetrics
    moved_atom_count: int
    runtime_ms: float

    def as_serializable_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "label": self.label,
            "metrics_after": asdict(self.metrics_after),
            "moved_atom_count": self.moved_atom_count,
            "runtime_ms": self.runtime_ms,
        }


@dataclass(frozen=True, slots=True)
class SidechainOperatorRefinementMeasurement:
    """Local-refinement acceptance result for one side-chain operator candidate."""

    status: str
    metrics_after: RefinementBenchmarkQualityMetrics | None
    moved_atom_count: int | None
    runtime_ms: float
    error_type: str | None
    error_message: str | None

    def as_serializable_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "status": self.status,
            "metrics_after": (
                None if self.metrics_after is None else asdict(self.metrics_after)
            ),
            "moved_atom_count": self.moved_atom_count,
            "runtime_ms": self.runtime_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class SidechainOperatorMeasurementResult:
    """One benchmark-only side-chain operator measurement result."""

    case_id: str
    track: str
    component_id: str
    residue_id: ResidueId
    candidate_count: int
    runtime_ms: float
    metrics_before: RefinementBenchmarkQualityMetrics
    best_candidate: SidechainOperatorCandidateMeasurement | None
    best_candidate_refinement: SidechainOperatorRefinementMeasurement | None
    signal: SidechainOperatorSignal
    notes: str

    def as_serializable_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "case_id": self.case_id,
            "track": self.track,
            "component_id": self.component_id,
            "residue_id": self.residue_id.display_token(),
            "candidate_count": self.candidate_count,
            "runtime_ms": self.runtime_ms,
            "metrics_before": asdict(self.metrics_before),
            "best_candidate": (
                None
                if self.best_candidate is None
                else self.best_candidate.as_serializable_dict()
            ),
            "best_candidate_refinement": (
                None
                if self.best_candidate_refinement is None
                else self.best_candidate_refinement.as_serializable_dict()
            ),
            "signal": self.signal.value,
            "notes": self.notes,
        }


SIDECHAIN_ROTAMER_GRID_DEFINITIONS: dict[str, SidechainRotamerGridDefinition] = {
    "GLN": SidechainRotamerGridDefinition(
        component_id="GLN",
        rotation_steps=(
            BranchedSidechainRotationStep(
                axis_atom_names=("CA", "CB"),
                rotating_atom_names=("CG", "CD", "OE1", "NE2"),
                hydrogen_anchor_atom_names=("CG", "CD", "OE1", "NE2"),
                angle_degrees_options=(-120, 120, 180),
            ),
            BranchedSidechainRotationStep(
                axis_atom_names=("CB", "CG"),
                rotating_atom_names=("CD", "OE1", "NE2"),
                hydrogen_anchor_atom_names=("CD", "OE1", "NE2"),
                angle_degrees_options=(-120, 120, 180),
            ),
            BranchedSidechainRotationStep(
                axis_atom_names=("CG", "CD"),
                rotating_atom_names=("OE1", "NE2"),
                hydrogen_anchor_atom_names=("OE1", "NE2"),
                angle_degrees_options=(180,),
            ),
        ),
    ),
    "TYR": SidechainRotamerGridDefinition(
        component_id="TYR",
        rotation_steps=(
            BranchedSidechainRotationStep(
                axis_atom_names=("CA", "CB"),
                rotating_atom_names=(
                    "CG",
                    "CD1",
                    "CD2",
                    "CE1",
                    "CE2",
                    "CZ",
                    "OH",
                ),
                hydrogen_anchor_atom_names=(
                    "CG",
                    "CD1",
                    "CD2",
                    "CE1",
                    "CE2",
                    "CZ",
                    "OH",
                ),
                angle_degrees_options=(-120, 120, 180),
            ),
            BranchedSidechainRotationStep(
                axis_atom_names=("CB", "CG"),
                rotating_atom_names=("CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
                hydrogen_anchor_atom_names=(
                    "CD1",
                    "CD2",
                    "CE1",
                    "CE2",
                    "CZ",
                    "OH",
                ),
                angle_degrees_options=(-120, 120, 180),
            ),
        ),
    ),
}


def run_sidechain_operator_measurement(
    case: RefinementBenchmarkCase,
    *,
    track: RefinementBenchmarkTrack,
    component_library: ComponentLibrary | None = None,
    restraint_library: RestraintLibrary | None = None,
    refine_best_candidate: bool = False,
) -> SidechainOperatorMeasurementResult:
    """Measure a benchmark-only rotamer grid against one refinement gap case."""

    if len(case.focus_residue_ids) != 1:
        raise ValueError("side-chain operator measurement requires one focus residue")

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
    start = perf_counter()
    original_structure, _ = prepare_case_structure(
        case,
        track=track,
        component_library=active_component_library,
    )
    reference_structure = load_case_reference_structure(
        case,
        track=track,
        component_library=active_component_library,
    )
    metrics_before = compute_refinement_metrics(
        original_structure,
        case=case,
        component_library=active_component_library,
        restraint_library=active_restraint_library,
        reference_structure=reference_structure,
    )
    residue_id = case.focus_residue_ids[0]
    residue_site = original_structure.constitution.residue_or_ligand(residue_id)
    if residue_site is None:
        raise ValueError(f"structure has no residue {residue_id.display_token()}")

    component_id = active_component_library.normalize_component_id(
        residue_site.component_id
    )
    definition = SIDECHAIN_ROTAMER_GRID_DEFINITIONS.get(component_id)
    if definition is None:
        return SidechainOperatorMeasurementResult(
            case_id=case.case_id,
            track=track.value,
            component_id=component_id,
            residue_id=residue_id,
            candidate_count=0,
            runtime_ms=(perf_counter() - start) * 1000.0,
            metrics_before=metrics_before,
            best_candidate=None,
            best_candidate_refinement=None,
            signal=SidechainOperatorSignal.UNSUITABLE,
            notes=f"no benchmark rotamer grid is defined for {component_id}",
        )

    candidates = tuple(
        measure_sidechain_operator_candidate(
            original_structure,
            case=case,
            residue_site=residue_site,
            definition=definition,
            angle_degrees_by_step=angle_degrees_by_step,
            component_library=active_component_library,
            restraint_library=active_restraint_library,
            reference_structure=reference_structure,
        )
        for angle_degrees_by_step in sidechain_rotamer_grid_angles(definition)
    )
    best_candidate = min(candidates, key=_candidate_quality_key, default=None)
    best_candidate_refinement = (
        None
        if best_candidate is None or not refine_best_candidate
        else measure_best_sidechain_candidate_refinement(
            original_structure,
            case=case,
            residue_site=residue_site,
            definition=definition,
            angle_degrees_by_step=tuple(
                int(token) for token in best_candidate.label.split("/")
            ),
            component_library=active_component_library,
            restraint_library=active_restraint_library,
            reference_structure=reference_structure,
        )
    )
    return SidechainOperatorMeasurementResult(
        case_id=case.case_id,
        track=track.value,
        component_id=component_id,
        residue_id=residue_id,
        candidate_count=len(candidates),
        runtime_ms=(perf_counter() - start) * 1000.0,
        metrics_before=metrics_before,
        best_candidate=best_candidate,
        best_candidate_refinement=best_candidate_refinement,
        signal=classify_sidechain_operator_signal(
            metrics_before=metrics_before,
            best_candidate=best_candidate,
            best_candidate_refinement=best_candidate_refinement,
        ),
        notes=sidechain_operator_measurement_notes(
            metrics_before=metrics_before,
            best_candidate=best_candidate,
            best_candidate_refinement=best_candidate_refinement,
        ),
    )


def sidechain_rotamer_grid_angles(
    definition: SidechainRotamerGridDefinition,
) -> tuple[tuple[int, ...], ...]:
    """Return non-identity angle combinations for one benchmark rotamer grid."""

    return tuple(
        angle_degrees_by_step
        for angle_degrees_by_step in product(
            *(
                (0, *rotation_step.angle_degrees_options)
                for rotation_step in definition.rotation_steps
            )
        )
        if any(angle_degrees != 0 for angle_degrees in angle_degrees_by_step)
    )


def measure_sidechain_operator_candidate(
    structure: ProteinStructure,
    *,
    case: RefinementBenchmarkCase,
    residue_site: ResidueSite,
    definition: SidechainRotamerGridDefinition,
    angle_degrees_by_step: tuple[int, ...],
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    reference_structure: ProteinStructure | None,
) -> SidechainOperatorCandidateMeasurement:
    """Return metrics for one benchmark rotamer-grid candidate."""

    start = perf_counter()
    candidate_structure, moved_atom_names = build_sidechain_rotamer_grid_candidate(
        structure,
        residue_site=residue_site,
        definition=definition,
        angle_degrees_by_step=angle_degrees_by_step,
    )
    metrics_after = compute_refinement_metrics(
        candidate_structure,
        case=case,
        component_library=component_library,
        restraint_library=restraint_library,
        reference_structure=reference_structure,
    )
    return SidechainOperatorCandidateMeasurement(
        label="/".join(str(angle_degrees) for angle_degrees in angle_degrees_by_step),
        metrics_after=metrics_after,
        moved_atom_count=len(moved_atom_names),
        runtime_ms=(perf_counter() - start) * 1000.0,
    )


def build_sidechain_rotamer_grid_candidate(
    structure: ProteinStructure,
    *,
    residue_site: ResidueSite,
    definition: SidechainRotamerGridDefinition,
    angle_degrees_by_step: tuple[int, ...],
) -> tuple[ProteinStructure, tuple[str, ...]]:
    """Return one structure after applying a benchmark-only side-chain grid move."""

    residue_index = structure.constitution.residue_index(residue_site.residue_id)
    working_residue_site = residue_site
    working_residue_geometry = structure.residue_geometry(residue_index)
    working_formal_charge_by_atom_name = structure.residue_formal_charge_by_atom_name(
        residue_index
    )
    moved_atom_names: list[str] = []
    for rotation_step, angle_degrees in zip(
        definition.rotation_steps, angle_degrees_by_step, strict=True
    ):
        if angle_degrees == 0:
            continue
        if not residue_supports_rotation_step(
            working_residue_site,
            rotation_step,
        ):
            continue

        (
            working_residue_site,
            working_residue_geometry,
            working_formal_charge_by_atom_name,
            rotated_atom_names,
        ) = rotate_residue_atoms_about_axis(
            residue_site=working_residue_site,
            residue_geometry=working_residue_geometry,
            formal_charge_by_atom_name=working_formal_charge_by_atom_name,
            axis_atom_names=rotation_step.axis_atom_names,
            rotated_atom_names={
                *rotation_step.rotating_atom_names,
                *attached_hydrogen_atom_names(
                    residue_site=working_residue_site,
                    residue_geometry=working_residue_geometry,
                    anchor_atom_names=rotation_step.hydrogen_anchor_atom_names,
                ),
            },
            theta_radians=angle_degrees * (pi / 180.0),
        )
        moved_atom_names.extend(rotated_atom_names)

    return (
        replace_residue(
            structure=structure,
            residue_site=working_residue_site,
            residue_geometry=working_residue_geometry,
            formal_charge_by_atom_name=working_formal_charge_by_atom_name,
        ),
        tuple(dict.fromkeys(moved_atom_names)),
    )


def measure_best_sidechain_candidate_refinement(
    structure: ProteinStructure,
    *,
    case: RefinementBenchmarkCase,
    residue_site: ResidueSite,
    definition: SidechainRotamerGridDefinition,
    angle_degrees_by_step: tuple[int, ...],
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    reference_structure: ProteinStructure | None,
) -> SidechainOperatorRefinementMeasurement:
    """Return RDKit-UFF local-refinement acceptance for one best candidate."""

    candidate_structure, _ = build_sidechain_rotamer_grid_candidate(
        structure,
        residue_site=residue_site,
        definition=definition,
        angle_degrees_by_step=angle_degrees_by_step,
    )
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residue_sidechains(case.focus_residue_ids),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(
            backend_name="rdkit",
            context_radius_angstrom=4.0,
            max_iterations=200,
        ),
    )
    start = perf_counter()
    try:
        refinement_result = transform_local_region(
            candidate_structure,
            spec,
            component_library=component_library,
        )
    except Exception as error:
        return SidechainOperatorRefinementMeasurement(
            status="error",
            metrics_after=None,
            moved_atom_count=None,
            runtime_ms=(perf_counter() - start) * 1000.0,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    metrics_after = compute_refinement_metrics(
        refinement_result.refined_structure,
        case=case,
        component_library=component_library,
        restraint_library=restraint_library,
        reference_structure=reference_structure,
    )
    moved_atom_count = refinement_result.moved_atom_count()
    return SidechainOperatorRefinementMeasurement(
        status="success" if moved_atom_count > 0 else "no_op",
        metrics_after=metrics_after,
        moved_atom_count=moved_atom_count,
        runtime_ms=(perf_counter() - start) * 1000.0,
        error_type=None,
        error_message=None,
    )


def residue_supports_rotation_step(
    residue_site: ResidueSite,
    rotation_step: BranchedSidechainRotationStep,
) -> bool:
    """Return whether one residue has the atoms needed for one grid step."""

    return all(
        residue_site.has_atom_site(atom_name)
        for atom_name in (
            *rotation_step.axis_atom_names,
            *rotation_step.rotating_atom_names,
        )
    )


def classify_sidechain_operator_signal(
    *,
    metrics_before: RefinementBenchmarkQualityMetrics,
    best_candidate: SidechainOperatorCandidateMeasurement | None,
    best_candidate_refinement: SidechainOperatorRefinementMeasurement | None = None,
) -> SidechainOperatorSignal:
    """Classify whether one measurement should affect production planning."""

    if best_candidate is None:
        return SidechainOperatorSignal.UNSUITABLE

    before_key = _metrics_quality_key(metrics_before)
    after_key = _metrics_quality_key(best_candidate.metrics_after)
    refined_key = None
    if best_candidate_refinement is not None:
        refined_metrics = best_candidate_refinement.metrics_after
        if refined_metrics is not None:
            refined_key = _metrics_quality_key(refined_metrics)

    if after_key < before_key or (refined_key is not None and refined_key < before_key):
        return SidechainOperatorSignal.PRODUCTION_PLANNING_SIGNAL

    if best_candidate.metrics_after.whole_structure_rdkit_sanitize_readable is False:
        return SidechainOperatorSignal.UNSUITABLE

    return SidechainOperatorSignal.TRACKING_ONLY_DIAGNOSTIC_SIGNAL


def sidechain_operator_measurement_notes(
    *,
    metrics_before: RefinementBenchmarkQualityMetrics,
    best_candidate: SidechainOperatorCandidateMeasurement | None,
    best_candidate_refinement: SidechainOperatorRefinementMeasurement | None = None,
) -> str:
    """Return one concise interpretation note for a measurement result."""

    if best_candidate is None:
        return "no side-chain operator candidate was generated"

    before_key = _metrics_quality_key(metrics_before)
    after_key = _metrics_quality_key(best_candidate.metrics_after)
    if (
        best_candidate_refinement is not None
        and best_candidate_refinement.metrics_after is not None
    ):
        refined_key = _metrics_quality_key(best_candidate_refinement.metrics_after)
        if refined_key < before_key:
            return "best candidate plus local FF improves the acceptance quality tuple"
        if refined_key > after_key:
            return "local FF after best candidate regresses the candidate quality tuple"

    if after_key < before_key:
        return "best candidate improves the acceptance quality tuple"
    if after_key == before_key:
        return "best candidate is quality-equivalent to the input"

    return "best candidate regresses the acceptance quality tuple"


def _candidate_quality_key(
    candidate: SidechainOperatorCandidateMeasurement,
) -> tuple[int, int, int, float, float]:
    """Return the sortable quality key for one measured candidate."""

    return _metrics_quality_key(candidate.metrics_after)


def _metrics_quality_key(
    metrics: RefinementBenchmarkQualityMetrics,
) -> tuple[int, int, int, float, float]:
    """Return the planning-style quality key for benchmark measurements."""

    reference_rmsd = metrics.reference_aligned_focus_atom_rmsd_angstrom
    return (
        metrics.focus_clash_count,
        metrics.focus_near_covalent_contact_count,
        metrics.focus_geometry_outlier_count,
        metrics.focus_clash_overlap_sum_angstrom,
        float("inf") if reference_rmsd is None else reference_rmsd,
    )
