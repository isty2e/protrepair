"""Acceptance policy helpers for local refinement execution results."""

from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.diagnostics.clashes import (
    ClashDetectionBasis,
    ClashReport,
    StericClash,
    detect_clashes_from_context,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.geometry import BondLengthOutlier, detect_heavy_geometry
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    ValidationIssueKind,
)
from protrepair.diagnostics.near_covalent import detect_near_covalent_contacts
from protrepair.diagnostics.parser_readability import (
    measure_rdkit_no_conect_sanitize_readability_metrics,
)
from protrepair.diagnostics.stereochemistry import detect_sidechain_stereochemistry
from protrepair.scope import (
    AtomSetScope,
    ResidueSetScope,
    Scope,
    coarsen_atom_scope_to_residue_scope,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import RegionTransformationResult

SEVERE_RESTRAINT_BACKED_BOND_LENGTH_DEVIATION_ANGSTROM = 0.20


@dataclass(frozen=True, slots=True)
class FocusRefinementQualityMetrics:
    """Selected-region quality metrics used for local acceptance decisions."""

    clash_count: int
    geometry_outlier_count: int
    restraint_backed_geometry_outlier_count: int = 0
    fallback_geometry_outlier_count: int = 0
    severe_restraint_backed_bond_length_outlier_count: int = 0
    clash_overlap_sum_angstrom: float = 0.0
    near_covalent_contact_count: int = 0
    worst_near_covalent_overlap_angstrom: float = 0.0
    total_near_covalent_overlap_angstrom: float = 0.0
    stereochemistry_violation_count: int = 0


@dataclass(frozen=True, slots=True)
class WholeStructureProximityBurdenMetrics:
    """Whole-structure proximity burden used when parser failure remains global."""

    near_covalent_contact_count: int = 0
    worst_near_covalent_overlap_angstrom: float = 0.0
    total_near_covalent_overlap_angstrom: float = 0.0


@dataclass(frozen=True, slots=True)
class WholeStructureParserCompatibilityMetrics:
    """Whole-structure parser-profile compatibility metrics."""

    rdkit_sanitize_readable: bool | None = None
    extra_proximity_bond_count: int = 0
    extra_heavy_proximity_bond_count: int = 0


@dataclass(frozen=True, slots=True, init=False)
class RefinementAcceptanceMetrics:
    """Aggregate local-refinement acceptance metrics over orthogonal axes."""

    focus_quality: FocusRefinementQualityMetrics
    whole_structure_proximity: WholeStructureProximityBurdenMetrics
    parser_compatibility: WholeStructureParserCompatibilityMetrics

    def __init__(
        self,
        *,
        focus_quality: FocusRefinementQualityMetrics | None = None,
        whole_structure_proximity: WholeStructureProximityBurdenMetrics | None = None,
        parser_compatibility: WholeStructureParserCompatibilityMetrics | None = None,
        focus_clash_count: int | None = None,
        focus_geometry_outlier_count: int | None = None,
        focus_restraint_backed_geometry_outlier_count: int = 0,
        focus_fallback_geometry_outlier_count: int = 0,
        focus_severe_restraint_backed_bond_length_outlier_count: int = 0,
        focus_clash_overlap_sum_angstrom: float = 0.0,
        focus_near_covalent_contact_count: int = 0,
        focus_worst_near_covalent_overlap_angstrom: float = 0.0,
        focus_total_near_covalent_overlap_angstrom: float = 0.0,
        focus_stereochemistry_violation_count: int = 0,
        whole_structure_near_covalent_contact_count: int = 0,
        whole_structure_worst_near_covalent_overlap_angstrom: float = 0.0,
        whole_structure_total_near_covalent_overlap_angstrom: float = 0.0,
        whole_structure_rdkit_sanitize_readable: bool | None = None,
        whole_structure_parser_extra_proximity_bond_count: int = 0,
        whole_structure_parser_extra_heavy_proximity_bond_count: int = 0,
    ) -> None:
        """Initialize from orthogonal records or legacy flat metric fields."""

        if focus_quality is None:
            if focus_clash_count is None or focus_geometry_outlier_count is None:
                raise ValueError(
                    "refinement acceptance metrics require focus quality metrics"
                )
            focus_quality = FocusRefinementQualityMetrics(
                clash_count=focus_clash_count,
                geometry_outlier_count=focus_geometry_outlier_count,
                restraint_backed_geometry_outlier_count=(
                    focus_restraint_backed_geometry_outlier_count
                ),
                fallback_geometry_outlier_count=(
                    focus_fallback_geometry_outlier_count
                ),
                severe_restraint_backed_bond_length_outlier_count=(
                    focus_severe_restraint_backed_bond_length_outlier_count
                ),
                clash_overlap_sum_angstrom=focus_clash_overlap_sum_angstrom,
                near_covalent_contact_count=focus_near_covalent_contact_count,
                worst_near_covalent_overlap_angstrom=(
                    focus_worst_near_covalent_overlap_angstrom
                ),
                total_near_covalent_overlap_angstrom=(
                    focus_total_near_covalent_overlap_angstrom
                ),
                stereochemistry_violation_count=(
                    focus_stereochemistry_violation_count
                ),
            )
        if whole_structure_proximity is None:
            whole_structure_proximity = WholeStructureProximityBurdenMetrics(
                near_covalent_contact_count=(
                    whole_structure_near_covalent_contact_count
                ),
                worst_near_covalent_overlap_angstrom=(
                    whole_structure_worst_near_covalent_overlap_angstrom
                ),
                total_near_covalent_overlap_angstrom=(
                    whole_structure_total_near_covalent_overlap_angstrom
                ),
            )
        if parser_compatibility is None:
            parser_compatibility = WholeStructureParserCompatibilityMetrics(
                rdkit_sanitize_readable=whole_structure_rdkit_sanitize_readable,
                extra_proximity_bond_count=(
                    whole_structure_parser_extra_proximity_bond_count
                ),
                extra_heavy_proximity_bond_count=(
                    whole_structure_parser_extra_heavy_proximity_bond_count
                ),
            )

        object.__setattr__(self, "focus_quality", focus_quality)
        object.__setattr__(
            self,
            "whole_structure_proximity",
            whole_structure_proximity,
        )
        object.__setattr__(
            self,
            "parser_compatibility",
            parser_compatibility,
        )

    @property
    def focus_clash_count(self) -> int:
        """Return selected-region clash count."""

        return self.focus_quality.clash_count

    @property
    def focus_geometry_outlier_count(self) -> int:
        """Return selected-region geometry outlier count."""

        return self.focus_quality.geometry_outlier_count

    @property
    def focus_restraint_backed_geometry_outlier_count(self) -> int:
        """Return selected-region restraint-backed geometry outlier count."""

        return self.focus_quality.restraint_backed_geometry_outlier_count

    @property
    def focus_fallback_geometry_outlier_count(self) -> int:
        """Return selected-region fallback geometry outlier count."""

        return self.focus_quality.fallback_geometry_outlier_count

    @property
    def focus_severe_restraint_backed_bond_length_outlier_count(self) -> int:
        """Return selected-region severe backed bond-length outlier count."""

        return self.focus_quality.severe_restraint_backed_bond_length_outlier_count

    @property
    def focus_clash_overlap_sum_angstrom(self) -> float:
        """Return selected-region clash overlap sum."""

        return self.focus_quality.clash_overlap_sum_angstrom

    @property
    def focus_near_covalent_contact_count(self) -> int:
        """Return selected-region near-covalent contact count."""

        return self.focus_quality.near_covalent_contact_count

    @property
    def focus_worst_near_covalent_overlap_angstrom(self) -> float:
        """Return selected-region worst near-covalent overlap."""

        return self.focus_quality.worst_near_covalent_overlap_angstrom

    @property
    def focus_total_near_covalent_overlap_angstrom(self) -> float:
        """Return selected-region total near-covalent overlap."""

        return self.focus_quality.total_near_covalent_overlap_angstrom

    @property
    def focus_stereochemistry_violation_count(self) -> int:
        """Return selected-region stereochemistry violation count."""

        return self.focus_quality.stereochemistry_violation_count

    @property
    def whole_structure_near_covalent_contact_count(self) -> int:
        """Return whole-structure near-covalent contact count."""

        return self.whole_structure_proximity.near_covalent_contact_count

    @property
    def whole_structure_worst_near_covalent_overlap_angstrom(self) -> float:
        """Return whole-structure worst near-covalent overlap."""

        return self.whole_structure_proximity.worst_near_covalent_overlap_angstrom

    @property
    def whole_structure_total_near_covalent_overlap_angstrom(self) -> float:
        """Return whole-structure total near-covalent overlap."""

        return self.whole_structure_proximity.total_near_covalent_overlap_angstrom

    @property
    def whole_structure_rdkit_sanitize_readable(self) -> bool | None:
        """Return whole-structure no-CONECT RDKit sanitize readability."""

        return self.parser_compatibility.rdkit_sanitize_readable

    @property
    def whole_structure_parser_extra_proximity_bond_count(self) -> int:
        """Return whole-structure parser-inferred extra proximity bond count."""

        return self.parser_compatibility.extra_proximity_bond_count

    @property
    def whole_structure_parser_extra_heavy_proximity_bond_count(self) -> int:
        """Return whole-structure parser-inferred extra heavy bond count."""

        return self.parser_compatibility.extra_heavy_proximity_bond_count


class RefinementAcceptanceVerdict(str, Enum):
    """Closed acceptance verdicts for one executed refinement candidate."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AssessedRefinementResult:
    """Executed refinement result plus assessment metrics and verdict."""

    executed_result: RegionTransformationResult
    before_metrics: RefinementAcceptanceMetrics
    after_metrics: RefinementAcceptanceMetrics
    verdict: RefinementAcceptanceVerdict
    rejection_issue: ValidationIssue | None = None

    def accepted_metrics(self) -> RefinementAcceptanceMetrics:
        """Return the metrics of the final accepted materialized outcome."""

        if self.verdict is RefinementAcceptanceVerdict.ACCEPTED:
            return self.after_metrics

        return self.before_metrics

    def is_accepted(self) -> bool:
        """Return whether the executed result passed refinement acceptance."""

        return self.verdict is RefinementAcceptanceVerdict.ACCEPTED


def assess_refinement_result(
    snapshot: ProteinStructureSnapshot,
    selected_scope: Scope,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    result: RegionTransformationResult,
    *,
    clash_basis: ClashDetectionBasis | None = None,
) -> AssessedRefinementResult:
    """Return assessed execution output without materializing fallback structure."""

    before_metrics = measure_refinement_acceptance_metrics_for_scope(
        snapshot.structure,
        selected_scope=selected_scope,
        component_library=component_library,
        restraint_library=restraint_library,
        clash_basis=clash_basis,
    )
    return assess_refinement_result_with_before_metrics(
        selected_scope,
        component_library,
        restraint_library,
        result,
        before_metrics=before_metrics,
        clash_basis=clash_basis,
    )


def assess_refinement_result_with_before_metrics(
    selected_scope: Scope,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    result: RegionTransformationResult,
    *,
    before_metrics: RefinementAcceptanceMetrics,
    clash_basis: ClashDetectionBasis | None = None,
) -> AssessedRefinementResult:
    """Return assessed execution output using precomputed fallback metrics."""

    focus_residue_ids = frozenset(_focus_residue_ids(selected_scope))
    after_metrics = measure_refinement_acceptance_metrics(
        result.refined_structure,
        focus_residue_ids=focus_residue_ids,
        component_library=component_library,
        restraint_library=restraint_library,
        clash_basis=clash_basis,
    )
    if not refinement_metrics_rejected(before_metrics, after_metrics):
        return AssessedRefinementResult(
            executed_result=result,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            verdict=RefinementAcceptanceVerdict.ACCEPTED,
        )

    return AssessedRefinementResult(
        executed_result=result,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        verdict=RefinementAcceptanceVerdict.REJECTED,
        rejection_issue=refinement_rejected_issue(
            selected_scope,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
        ),
    )


def measure_refinement_acceptance_metrics(
    structure: ProteinStructure,
    *,
    focus_residue_ids: frozenset[ResidueId],
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    clash_basis: ClashDetectionBasis | None = None,
) -> RefinementAcceptanceMetrics:
    """Return clash and geometry metrics over one selected residue region."""

    clash_context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
        basis=clash_basis,
    )
    focus_clashes = detect_clashes_from_context(
        clash_context,
        focus_residue_ids=focus_residue_ids,
    ).clashes
    geometry_report = detect_heavy_geometry(
        structure,
        component_library=component_library,
        restraint_library=restraint_library,
        residue_ids=focus_residue_ids,
    )
    near_covalent_contacts = detect_near_covalent_contacts(
        structure,
        clashes=focus_clashes,
    )
    stereochemistry_report = detect_sidechain_stereochemistry(
        structure,
        component_library=component_library,
    )
    rdkit_readability_metrics = (
        measure_rdkit_no_conect_sanitize_readability_metrics(
            structure,
            component_library=component_library,
        )
    )
    return RefinementAcceptanceMetrics(
        focus_quality=FocusRefinementQualityMetrics(
            clash_count=len(focus_clashes),
            geometry_outlier_count=(
                len(geometry_report.bond_length_outliers)
                + len(geometry_report.bond_angle_outliers)
            ),
            restraint_backed_geometry_outlier_count=(
                geometry_report.restraint_backed_outlier_count()
            ),
            fallback_geometry_outlier_count=(
                geometry_report.fallback_outlier_count()
            ),
            severe_restraint_backed_bond_length_outlier_count=(
                _severe_restraint_backed_bond_length_outlier_count(
                    geometry_report.bond_length_outliers
                )
            ),
            clash_overlap_sum_angstrom=sum(
                clash.overlap_angstrom for clash in focus_clashes
            ),
            near_covalent_contact_count=len(near_covalent_contacts),
            worst_near_covalent_overlap_angstrom=(
                _worst_clash_overlap_angstrom(near_covalent_contacts)
            ),
            total_near_covalent_overlap_angstrom=(
                _total_clash_overlap_angstrom(near_covalent_contacts)
            ),
            stereochemistry_violation_count=sum(
                1
                for violation in stereochemistry_report.violations
                if violation.residue_id in focus_residue_ids
            ),
        ),
        parser_compatibility=WholeStructureParserCompatibilityMetrics(
            rdkit_sanitize_readable=rdkit_readability_metrics.sanitize_readable,
            extra_proximity_bond_count=(
                rdkit_readability_metrics.extra_proximity_bond_count
            ),
            extra_heavy_proximity_bond_count=(
                rdkit_readability_metrics.extra_heavy_proximity_bond_count
            ),
        ),
    )


def count_focus_clashes(
    *,
    clash_report: ClashReport,
    focus_residue_ids: frozenset[ResidueId],
) -> int:
    """Return clash-pair count involving at least one selected residue."""

    return sum(
        1
        for clash in clash_report.clashes
        if clash.left_residue_id in focus_residue_ids
        or clash.right_residue_id in focus_residue_ids
    )


def count_focus_clash_overlap_sum_angstrom(
    *,
    clash_report: ClashReport,
    focus_residue_ids: frozenset[ResidueId],
) -> float:
    """Return total clash overlap involving at least one selected residue."""

    return sum(
        clash.overlap_angstrom
        for clash in clash_report.clashes
        if clash.left_residue_id in focus_residue_ids
        or clash.right_residue_id in focus_residue_ids
    )


def _worst_clash_overlap_angstrom(clashes: tuple[StericClash, ...]) -> float:
    """Return the worst overlap across one clash collection."""

    worst_overlap = 0.0
    for clash in clashes:
        overlap_angstrom = clash.overlap_angstrom
        if overlap_angstrom > worst_overlap:
            worst_overlap = overlap_angstrom

    return worst_overlap


def _total_clash_overlap_angstrom(clashes: tuple[StericClash, ...]) -> float:
    """Return the total overlap across one clash collection."""

    return sum(clash.overlap_angstrom for clash in clashes)


def refinement_metrics_regressed(
    before_metrics: RefinementAcceptanceMetrics,
    after_metrics: RefinementAcceptanceMetrics,
) -> bool:
    """Return whether one refinement result regressed selected-region quality."""

    if _has_unresolved_parser_visibility_failure(
        before_metrics,
        after_metrics,
    ):
        return (
            _whole_structure_unreadable_order_key(after_metrics)
            >= _whole_structure_unreadable_order_key(before_metrics)
            or _focus_metrics_order_key(after_metrics)
            > _focus_metrics_order_key(before_metrics)
        )

    return refinement_metrics_order_key(after_metrics) > refinement_metrics_order_key(
        before_metrics
    )


def refinement_metrics_rejected(
    before_metrics: RefinementAcceptanceMetrics,
    after_metrics: RefinementAcceptanceMetrics,
) -> bool:
    """Return whether one refinement result should be rejected."""

    return refinement_metrics_regressed(
        before_metrics,
        after_metrics,
    ) or refinement_has_new_parser_visibility_failure(
        before_metrics,
        after_metrics,
    ) or refinement_has_new_stereochemistry_failure(
        before_metrics,
        after_metrics,
    ) or refinement_has_new_severe_restraint_backed_bond_length_failure(
        before_metrics,
        after_metrics,
    )


def refinement_metrics_order_key(
    metrics: RefinementAcceptanceMetrics,
) -> tuple[int, int, float, float, int, float, int, int]:
    """Return one stable ordering key for accepted refinement quality."""

    return (
        0 if metrics.whole_structure_rdkit_sanitize_readable is not False else 1,
        *_focus_metrics_order_key(metrics),
    )


def refinement_has_new_parser_visibility_failure(
    before_metrics: RefinementAcceptanceMetrics,
    after_metrics: RefinementAcceptanceMetrics,
) -> bool:
    """Return whether refinement introduced one new parser-visible failure."""

    return (
        before_metrics.whole_structure_rdkit_sanitize_readable is not False
        and after_metrics.whole_structure_rdkit_sanitize_readable is False
    )


def refinement_has_new_stereochemistry_failure(
    before_metrics: RefinementAcceptanceMetrics,
    after_metrics: RefinementAcceptanceMetrics,
) -> bool:
    """Return whether refinement introduced new focus stereochemistry burden."""

    return (
        after_metrics.focus_stereochemistry_violation_count
        > before_metrics.focus_stereochemistry_violation_count
    )


def refinement_has_new_severe_restraint_backed_bond_length_failure(
    before_metrics: RefinementAcceptanceMetrics,
    after_metrics: RefinementAcceptanceMetrics,
) -> bool:
    """Return whether refinement introduced a high-confidence bonded failure."""

    return (
        after_metrics.focus_severe_restraint_backed_bond_length_outlier_count
        > before_metrics.focus_severe_restraint_backed_bond_length_outlier_count
    )


def _has_unresolved_parser_visibility_failure(
    before_metrics: RefinementAcceptanceMetrics,
    after_metrics: RefinementAcceptanceMetrics,
) -> bool:
    """Return whether parser-visible unreadability remained unresolved."""

    return (
        before_metrics.whole_structure_rdkit_sanitize_readable is False
        and after_metrics.whole_structure_rdkit_sanitize_readable is False
    )


def _clashes_involving_residues(
    clash_report: ClashReport,
    *,
    focus_residue_ids: frozenset[ResidueId],
) -> tuple[StericClash, ...]:
    """Return clash pairs involving at least one focus residue."""

    return tuple(
        clash
        for clash in clash_report.clashes
        if clash.left_residue_id in focus_residue_ids
        or clash.right_residue_id in focus_residue_ids
    )


def _focus_metrics_order_key(
    metrics: RefinementAcceptanceMetrics,
) -> tuple[int, float, float, int, float, int, int]:
    """Return one stable selected-region quality ordering key.

    Clash burden is primary. Stereochemistry and geometry only break ties after
    near-covalent and steric clash burden are unchanged.
    """

    return (
        metrics.focus_near_covalent_contact_count,
        metrics.focus_worst_near_covalent_overlap_angstrom,
        metrics.focus_total_near_covalent_overlap_angstrom,
        metrics.focus_clash_count,
        metrics.focus_clash_overlap_sum_angstrom,
        metrics.focus_stereochemistry_violation_count,
        metrics.focus_geometry_outlier_count,
    )


def _severe_restraint_backed_bond_length_outlier_count(
    bond_length_outliers: tuple[BondLengthOutlier, ...],
) -> int:
    """Return high-confidence bond-length violations for hard veto use."""

    return sum(
        1
        for outlier in bond_length_outliers
        if outlier.restraint_backed
        and outlier.deviation_angstrom()
        >= SEVERE_RESTRAINT_BACKED_BOND_LENGTH_DEVIATION_ANGSTROM
    )


def _whole_structure_unreadable_order_key(
    metrics: RefinementAcceptanceMetrics,
) -> tuple[int, int, int, float, float]:
    """Return one global parser-visible burden key for unreadable structures."""

    return (
        metrics.whole_structure_parser_extra_heavy_proximity_bond_count,
        metrics.whole_structure_parser_extra_proximity_bond_count,
        metrics.whole_structure_near_covalent_contact_count,
        metrics.whole_structure_worst_near_covalent_overlap_angstrom,
        metrics.whole_structure_total_near_covalent_overlap_angstrom,
    )


def refinement_rejected_issue(
    selected_scope: Scope,
    *,
    before_metrics: RefinementAcceptanceMetrics,
    after_metrics: RefinementAcceptanceMetrics,
) -> ValidationIssue:
    """Return one structured issue describing a rejected refinement result."""

    referenced_residue_ids = _focus_residue_ids(selected_scope)
    residue_id = referenced_residue_ids[0] if len(referenced_residue_ids) == 1 else None
    message = (
        "Local refinement result was rejected because selected-region quality "
        "regressed: clashes "
        f"{before_metrics.focus_clash_count}->{after_metrics.focus_clash_count}, "
        "near-covalent "
        f"{before_metrics.focus_near_covalent_contact_count}"
        f"->{after_metrics.focus_near_covalent_contact_count}, "
        "near-covalent overlap "
        f"{before_metrics.focus_total_near_covalent_overlap_angstrom:.2f}"
        f"->{after_metrics.focus_total_near_covalent_overlap_angstrom:.2f}, "
        "stereo "
        f"{before_metrics.focus_stereochemistry_violation_count}"
        f"->{after_metrics.focus_stereochemistry_violation_count}, "
        "geometry "
        f"{before_metrics.focus_geometry_outlier_count}"
        f"->{after_metrics.focus_geometry_outlier_count}, "
        "severe bond geometry "
        f"{before_metrics.focus_severe_restraint_backed_bond_length_outlier_count}"
        f"->{after_metrics.focus_severe_restraint_backed_bond_length_outlier_count}, "
        "overlap "
        f"{before_metrics.focus_clash_overlap_sum_angstrom:.2f}"
        f"->{after_metrics.focus_clash_overlap_sum_angstrom:.2f}"
    )
    if (
        before_metrics.whole_structure_rdkit_sanitize_readable is not None
        or after_metrics.whole_structure_rdkit_sanitize_readable is not None
    ):
        message += (
            ", rdkit no-CONECT sanitize "
            f"{before_metrics.whole_structure_rdkit_sanitize_readable}"
            f"->{after_metrics.whole_structure_rdkit_sanitize_readable}"
        )
    if (
        before_metrics.whole_structure_parser_extra_proximity_bond_count
        or after_metrics.whole_structure_parser_extra_proximity_bond_count
    ):
        message += (
            ", parser extra proximity bonds "
            f"{before_metrics.whole_structure_parser_extra_proximity_bond_count}"
            f"->{after_metrics.whole_structure_parser_extra_proximity_bond_count}"
            " heavy "
            f"{before_metrics.whole_structure_parser_extra_heavy_proximity_bond_count}"
            f"->{after_metrics.whole_structure_parser_extra_heavy_proximity_bond_count}"
        )
    if (
        before_metrics.whole_structure_near_covalent_contact_count
        or after_metrics.whole_structure_near_covalent_contact_count
    ):
        message += (
            ", global near-covalent "
            f"{before_metrics.whole_structure_near_covalent_contact_count}"
            f"->{after_metrics.whole_structure_near_covalent_contact_count}"
            " overlap "
            f"{before_metrics.whole_structure_total_near_covalent_overlap_angstrom:.2f}"
            f"->{after_metrics.whole_structure_total_near_covalent_overlap_angstrom:.2f}"
        )
    if residue_id is None:
        return ValidationIssue(
            kind=ValidationIssueKind.REFINEMENT_REJECTED,
            severity=IssueSeverity.INFO,
            message=message,
        )

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.REFINEMENT_REJECTED,
        severity=IssueSeverity.INFO,
        residue_id=residue_id,
        message=message,
    )


def measure_refinement_acceptance_metrics_for_scope(
    structure: ProteinStructure,
    *,
    selected_scope: Scope,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    clash_basis: ClashDetectionBasis | None = None,
) -> RefinementAcceptanceMetrics:
    """Return acceptance metrics over one semantic refinement scope."""

    return measure_refinement_acceptance_metrics(
        structure,
        focus_residue_ids=frozenset(_focus_residue_ids(selected_scope)),
        component_library=component_library,
        restraint_library=restraint_library,
        clash_basis=clash_basis,
    )


def _focus_residue_ids(selected_scope: Scope) -> tuple[ResidueId, ...]:
    """Return focus residues implied by one local refinement selection scope."""

    if isinstance(selected_scope, ResidueSetScope):
        return selected_scope.residue_ids

    if isinstance(selected_scope, AtomSetScope):
        return coarsen_atom_scope_to_residue_scope(selected_scope).residue_ids

    raise TypeError(
        "refinement acceptance requires one residue- or atom-set scope, "
        f"got {selected_scope.kind.value}"
    )
