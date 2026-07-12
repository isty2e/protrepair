"""Heavy-atom geometry diagnostics over canonical structures."""

from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from itertools import combinations
from math import acos, degrees, sqrt

from protrepair.chemistry import (
    BondDefinition,
    ComponentLibrary,
    ElementRadiusLookup,
    RadiusKind,
    ResidueTemplate,
    RestraintLibrary,
    build_default_restraint_library,
    prepare_radius_lookup,
)
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId

# Observation-only tuple math returns None for undefined angles; it does not
# share the exception-bearing NumPy placement contract.
GEOMETRY_DEGENERATE_NORM_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class HeavyGeometryPolicy:
    """Policy controlling heavy-atom plausibility thresholds."""

    bond_length_tolerance_angstrom: float = 0.25
    minimum_bond_angle_degrees: float = 85.0
    maximum_bond_angle_degrees: float = 145.0
    restraint_bond_length_esd_multiplier: float = 4.0
    restraint_bond_length_minimum_tolerance_angstrom: float = 0.04
    restraint_bond_angle_esd_multiplier: float = 4.0
    restraint_bond_angle_minimum_tolerance_degrees: float = 4.0

    def __post_init__(self) -> None:
        if self.bond_length_tolerance_angstrom <= 0.0:
            raise ValueError("bond length tolerance must be positive")

        if self.minimum_bond_angle_degrees <= 0.0:
            raise ValueError("minimum bond angle must be positive")

        if self.maximum_bond_angle_degrees >= 180.0:
            raise ValueError("maximum bond angle must be less than 180 degrees")

        if self.minimum_bond_angle_degrees >= self.maximum_bond_angle_degrees:
            raise ValueError("bond angle range must be ordered")

        if self.restraint_bond_length_esd_multiplier <= 0.0:
            raise ValueError("restraint bond-length esd multiplier must be positive")

        if self.restraint_bond_length_minimum_tolerance_angstrom <= 0.0:
            raise ValueError("restraint bond-length minimum tolerance must be positive")

        if self.restraint_bond_angle_esd_multiplier <= 0.0:
            raise ValueError("restraint bond-angle esd multiplier must be positive")

        if self.restraint_bond_angle_minimum_tolerance_degrees <= 0.0:
            raise ValueError("restraint bond-angle minimum tolerance must be positive")


@dataclass(frozen=True, slots=True)
class BondLengthOutlier:
    """One bonded heavy-atom pair outside the plausibility range."""

    residue_id: ResidueId
    component_id: str
    atom_name_1: str
    atom_name_2: str
    observed_distance_angstrom: float
    expected_distance_angstrom: float
    tolerance_angstrom: float
    restraint_backed: bool = False

    def deviation_angstrom(self) -> float:
        """Return the absolute distance deviation from the expected length."""

        return abs(self.observed_distance_angstrom - self.expected_distance_angstrom)


@dataclass(frozen=True, slots=True)
class BondAngleOutlier:
    """One heavy-atom bond angle outside the plausibility range."""

    residue_id: ResidueId
    component_id: str
    atom_name_1: str
    center_atom_name: str
    atom_name_2: str
    observed_angle_degrees: float
    minimum_angle_degrees: float
    maximum_angle_degrees: float
    restraint_backed: bool = False

    def deviation_degrees(self) -> float:
        """Return the distance outside the allowed bond-angle window."""

        if self.observed_angle_degrees < self.minimum_angle_degrees:
            return self.minimum_angle_degrees - self.observed_angle_degrees

        return self.observed_angle_degrees - self.maximum_angle_degrees


@dataclass(frozen=True, slots=True)
class HeavyGeometryReport:
    """Structured heavy-atom geometry findings for one canonical structure."""

    bond_length_outliers: tuple[BondLengthOutlier, ...]
    bond_angle_outliers: tuple[BondAngleOutlier, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bond_length_outliers",
            tuple(self.bond_length_outliers),
        )
        object.__setattr__(
            self,
            "bond_angle_outliers",
            tuple(self.bond_angle_outliers),
        )

    def is_empty(self) -> bool:
        """Return whether the report contains no outliers."""

        return not self.bond_length_outliers and not self.bond_angle_outliers

    def restraint_backed_outlier_count(self) -> int:
        """Return the count of outliers supported by explicit restraint targets."""

        return sum(
            1
            for outlier in (*self.bond_length_outliers, *self.bond_angle_outliers)
            if outlier.restraint_backed
        )

    def fallback_outlier_count(self) -> int:
        """Return the count of outliers that used fallback geometry windows."""

        return sum(
            1
            for outlier in (*self.bond_length_outliers, *self.bond_angle_outliers)
            if not outlier.restraint_backed
        )

    def to_issues(self) -> tuple[ValidationIssue, ...]:
        """Project geometry outliers into residue-level validation issues."""

        bond_length_by_residue: dict[ResidueId, list[BondLengthOutlier]] = defaultdict(
            list
        )
        for outlier in self.bond_length_outliers:
            bond_length_by_residue[outlier.residue_id].append(outlier)

        bond_angle_by_residue: dict[ResidueId, list[BondAngleOutlier]] = defaultdict(
            list
        )
        for outlier in self.bond_angle_outliers:
            bond_angle_by_residue[outlier.residue_id].append(outlier)

        residue_ids = sorted(
            {*bond_length_by_residue.keys(), *bond_angle_by_residue.keys()},
            key=residue_sort_key,
        )
        issues: list[ValidationIssue] = []
        for residue_id in residue_ids:
            bond_length_outliers = bond_length_by_residue.get(residue_id, [])
            bond_angle_outliers = bond_angle_by_residue.get(residue_id, [])
            parts: list[str] = []
            if bond_length_outliers:
                worst_length = max(
                    bond_length_outliers,
                    key=lambda outlier: outlier.deviation_angstrom(),
                )
                parts.append(
                    f"{len(bond_length_outliers)} bond-length outlier(s); worst is "
                    f"{worst_length.atom_name_1}-{worst_length.atom_name_2} "
                    f"({worst_length.observed_distance_angstrom:.2f} A)"
                )

            if bond_angle_outliers:
                worst_angle = max(
                    bond_angle_outliers,
                    key=lambda outlier: outlier.deviation_degrees(),
                )
                parts.append(
                    f"{len(bond_angle_outliers)} bond-angle outlier(s); worst is "
                    f"{worst_angle.atom_name_1}-{worst_angle.center_atom_name}-"
                    f"{worst_angle.atom_name_2} "
                    f"({worst_angle.observed_angle_degrees:.1f} deg)"
                )

            issues.append(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.INVALID_GEOMETRY,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{residue_id.display_token()} has implausible heavy-atom "
                        f"geometry: {'; '.join(parts)}"
                    ),
                    residue_id=residue_id,
                )
            )

        return tuple(issues)


@dataclass(frozen=True, slots=True)
class SevereIntrinsicGeometryResidue:
    """One residue whose intrinsic geometry is severe enough to auto-repair."""

    residue_id: ResidueId
    total_outlier_count: int
    restraint_backed_outlier_count: int
    worst_bond_length_deviation_angstrom: float
    worst_bond_angle_deviation_degrees: float


def detect_heavy_geometry(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary | None = None,
    residue_ids: Collection[ResidueId] | None = None,
    policy: HeavyGeometryPolicy | None = None,
) -> HeavyGeometryReport:
    """Return heavy-atom geometry outliers for supported template residues."""

    normalized_policy = HeavyGeometryPolicy() if policy is None else policy
    selected_residue_ids = None if residue_ids is None else frozenset(residue_ids)
    active_restraint_library = (
        build_default_restraint_library()
        if restraint_library is None
        else restraint_library
    )
    bond_length_outliers: list[BondLengthOutlier] = []
    bond_angle_outliers: list[BondAngleOutlier] = []
    for residue in structure.constitution.iter_residues():
        if (
            selected_residue_ids is not None
            and residue.residue_id not in selected_residue_ids
        ):
            continue

        template = component_library.get(residue.component_id)
        if template is None:
            continue

        restraint_template = active_restraint_library.get(residue.component_id)
        residue_geometry = structure.residue_geometry(
            structure.constitution.residue_index(residue.residue_id)
        )

        bond_length_outliers.extend(
            detect_residue_bond_length_outliers(
                residue,
                residue_geometry=residue_geometry,
                template=template,
                restraint_template=restraint_template,
                policy=normalized_policy,
            )
        )
        bond_angle_outliers.extend(
            detect_residue_bond_angle_outliers(
                residue,
                residue_geometry=residue_geometry,
                template=template,
                restraint_template=restraint_template,
                policy=normalized_policy,
            )
        )

    return HeavyGeometryReport(
        bond_length_outliers=tuple(bond_length_outliers),
        bond_angle_outliers=tuple(bond_angle_outliers),
    )


def severe_intrinsic_geometry_residues(
    report: HeavyGeometryReport,
    *,
    min_total_outlier_count: int = 3,
    min_restraint_backed_outlier_count: int = 3,
    min_worst_bond_length_deviation_angstrom: float = 0.12,
    min_worst_bond_angle_deviation_degrees: float = 10.0,
    min_total_outlier_count_without_large_deviation: int = 5,
) -> tuple[SevereIntrinsicGeometryResidue, ...]:
    """Return residue-local intrinsic geometry burdens severe enough to auto-repair."""

    bond_length_by_residue: dict[ResidueId, list[BondLengthOutlier]] = defaultdict(
        list
    )
    for outlier in report.bond_length_outliers:
        bond_length_by_residue[outlier.residue_id].append(outlier)

    bond_angle_by_residue: dict[ResidueId, list[BondAngleOutlier]] = defaultdict(list)
    for outlier in report.bond_angle_outliers:
        bond_angle_by_residue[outlier.residue_id].append(outlier)

    findings: list[SevereIntrinsicGeometryResidue] = []
    for residue_id in sorted(
        {*bond_length_by_residue.keys(), *bond_angle_by_residue.keys()},
        key=residue_sort_key,
    ):
        bond_length_outliers = bond_length_by_residue.get(residue_id, [])
        bond_angle_outliers = bond_angle_by_residue.get(residue_id, [])
        total_outlier_count = len(bond_length_outliers) + len(bond_angle_outliers)
        restraint_backed_outlier_count = sum(
            1
            for outlier in (*bond_length_outliers, *bond_angle_outliers)
            if outlier.restraint_backed
        )
        if total_outlier_count < min_total_outlier_count:
            continue
        if restraint_backed_outlier_count < min_restraint_backed_outlier_count:
            continue

        worst_bond_length_deviation_angstrom = max(
            (
                outlier.deviation_angstrom()
                for outlier in bond_length_outliers
            ),
            default=0.0,
        )
        worst_bond_angle_deviation_degrees = max(
            (
                outlier.deviation_degrees()
                for outlier in bond_angle_outliers
            ),
            default=0.0,
        )
        has_large_deviation = (
            worst_bond_length_deviation_angstrom
            >= min_worst_bond_length_deviation_angstrom
            or worst_bond_angle_deviation_degrees
            >= min_worst_bond_angle_deviation_degrees
        )
        if (
            not has_large_deviation
            and total_outlier_count < min_total_outlier_count_without_large_deviation
        ):
            continue

        findings.append(
            SevereIntrinsicGeometryResidue(
                residue_id=residue_id,
                total_outlier_count=total_outlier_count,
                restraint_backed_outlier_count=restraint_backed_outlier_count,
                worst_bond_length_deviation_angstrom=(
                    worst_bond_length_deviation_angstrom
                ),
                worst_bond_angle_deviation_degrees=(
                    worst_bond_angle_deviation_degrees
                ),
            )
        )

    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                -(
                    finding.worst_bond_length_deviation_angstrom * 10.0
                    + finding.worst_bond_angle_deviation_degrees
                ),
                -finding.worst_bond_angle_deviation_degrees,
                -finding.worst_bond_length_deviation_angstrom,
                -finding.restraint_backed_outlier_count,
                -finding.total_outlier_count,
                residue_sort_key(finding.residue_id),
            ),
        )
    )


def detect_residue_bond_length_outliers(
    residue: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    template: ResidueTemplate,
    restraint_template: ResidueRestraintTemplate | None,
    policy: HeavyGeometryPolicy,
) -> tuple[BondLengthOutlier, ...]:
    """Return bonded heavy-atom length outliers for one residue."""

    covalent_radius_lookup = _unrestrained_bond_covalent_radius_lookup(
        residue,
        template=template,
        restraint_template=restraint_template,
    )
    outliers: list[BondLengthOutlier] = []
    for bond in template.definition.bonds:
        if not residue.has_atom_site(bond.atom_name_1) or not residue.has_atom_site(
            bond.atom_name_2
        ):
            continue

        left_atom_site = residue.atom_site(bond.atom_name_1)
        right_atom_site = residue.atom_site(bond.atom_name_2)
        observed_distance = residue_geometry.atom_geometry(
            bond.atom_name_1
        ).distance_to(residue_geometry.atom_geometry(bond.atom_name_2))
        restraint_target = (
            None
            if restraint_template is None
            else restraint_template.bond_target(bond.atom_name_1, bond.atom_name_2)
        )
        if restraint_target is None:
            expected_distance = expected_bond_length_angstrom(
                left_atom_site.element,
                right_atom_site.element,
                bond=bond,
                radius_lookup=covalent_radius_lookup,
            )
            tolerance = policy.bond_length_tolerance_angstrom
        else:
            expected_distance = restraint_target.target_distance_angstrom
            tolerance = bond_length_tolerance_angstrom(
                policy=policy,
                target=restraint_target,
            )

        if abs(observed_distance - expected_distance) <= tolerance:
            continue

        outliers.append(
            BondLengthOutlier(
                residue_id=residue.residue_id,
                component_id=residue.component_id,
                atom_name_1=bond.atom_name_1,
                atom_name_2=bond.atom_name_2,
                observed_distance_angstrom=observed_distance,
                expected_distance_angstrom=expected_distance,
                tolerance_angstrom=tolerance,
                restraint_backed=restraint_target is not None,
            )
        )

    return tuple(outliers)


def detect_residue_bond_angle_outliers(
    residue: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    template: ResidueTemplate,
    restraint_template: ResidueRestraintTemplate | None,
    policy: HeavyGeometryPolicy,
) -> tuple[BondAngleOutlier, ...]:
    """Return heavy-atom bond-angle outliers for one residue."""

    if restraint_template is not None:
        return detect_restraint_bond_angle_outliers(
            residue,
            residue_geometry=residue_geometry,
            restraint_template=restraint_template,
            policy=policy,
        )

    return detect_fallback_bond_angle_outliers(
        residue,
        residue_geometry=residue_geometry,
        template=template,
        policy=policy,
    )


def detect_fallback_bond_angle_outliers(
    residue: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    template: ResidueTemplate,
    policy: HeavyGeometryPolicy,
) -> tuple[BondAngleOutlier, ...]:
    """Return broad fallback angle outliers when no restraint target exists."""

    outliers: list[BondAngleOutlier] = []
    for center_atom_name in template.expected_atom_names():
        bonded_atom_names = tuple(
            atom_name
            for atom_name in template.definition.bonded_atom_names(center_atom_name)
            if residue.has_atom_site(atom_name)
        )
        if len(bonded_atom_names) < 2 or not residue.has_atom_site(center_atom_name):
            continue

        for atom_name_1, atom_name_2 in combinations(
            sorted(bonded_atom_names),
            2,
        ):
            observed_angle = bond_angle_degrees(
                residue_geometry,
                atom_name_1=atom_name_1,
                center_atom_name=center_atom_name,
                atom_name_2=atom_name_2,
            )
            if observed_angle is None:
                continue

            if (
                policy.minimum_bond_angle_degrees
                <= observed_angle
                <= policy.maximum_bond_angle_degrees
            ):
                continue

            outliers.append(
                BondAngleOutlier(
                    residue_id=residue.residue_id,
                    component_id=residue.component_id,
                    atom_name_1=atom_name_1,
                    center_atom_name=center_atom_name,
                    atom_name_2=atom_name_2,
                    observed_angle_degrees=observed_angle,
                    minimum_angle_degrees=policy.minimum_bond_angle_degrees,
                    maximum_angle_degrees=policy.maximum_bond_angle_degrees,
                    restraint_backed=False,
                )
            )

    return tuple(outliers)


def detect_restraint_bond_angle_outliers(
    residue: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    restraint_template: ResidueRestraintTemplate,
    policy: HeavyGeometryPolicy,
) -> tuple[BondAngleOutlier, ...]:
    """Return angle outliers against explicit residue restraint targets."""

    outliers: list[BondAngleOutlier] = []
    for target in restraint_template.angle_targets:
        if not residue.has_atom_site(target.atom_name_1):
            continue

        if not residue.has_atom_site(target.center_atom_name):
            continue

        if not residue.has_atom_site(target.atom_name_2):
            continue

        observed_angle = bond_angle_degrees(
            residue_geometry,
            atom_name_1=target.atom_name_1,
            center_atom_name=target.center_atom_name,
            atom_name_2=target.atom_name_2,
        )
        if observed_angle is None:
            continue

        tolerance = bond_angle_tolerance_degrees(
            policy=policy,
            target=target,
        )
        minimum_angle_degrees = target.target_angle_degrees - tolerance
        maximum_angle_degrees = target.target_angle_degrees + tolerance
        if minimum_angle_degrees <= observed_angle <= maximum_angle_degrees:
            continue

        outliers.append(
            BondAngleOutlier(
                residue_id=residue.residue_id,
                component_id=residue.component_id,
                atom_name_1=target.atom_name_1,
                center_atom_name=target.center_atom_name,
                atom_name_2=target.atom_name_2,
                observed_angle_degrees=observed_angle,
                minimum_angle_degrees=minimum_angle_degrees,
                maximum_angle_degrees=maximum_angle_degrees,
                restraint_backed=True,
            )
        )

    return tuple(outliers)


def expected_bond_length_angstrom(
    element_1: str,
    element_2: str,
    *,
    bond: BondDefinition,
    radius_lookup: ElementRadiusLookup | None = None,
) -> float:
    """Return a broad expected bond length for one heavy-atom pair."""

    del bond
    active_radius_lookup = (
        prepare_radius_lookup((element_1, element_2), RadiusKind.COVALENT)
        if radius_lookup is None
        else radius_lookup
    )
    active_radius_lookup.require_kind(
        RadiusKind.COVALENT,
        "bond length expectation",
    )
    active_radius_lookup.require_complete("bond length expectation")
    return active_radius_lookup.radius_angstrom(
        element_1
    ) + active_radius_lookup.radius_angstrom(element_2)


def _unrestrained_bond_covalent_radius_lookup(
    residue: ResidueSite,
    *,
    template: ResidueTemplate,
    restraint_template: ResidueRestraintTemplate | None,
) -> ElementRadiusLookup:
    """Return prepared covalent radii needed by fallback bond-length checks."""

    elements: list[str] = []
    for bond in template.definition.bonds:
        if not residue.has_atom_site(bond.atom_name_1) or not residue.has_atom_site(
            bond.atom_name_2
        ):
            continue

        restraint_target = (
            None
            if restraint_template is None
            else restraint_template.bond_target(bond.atom_name_1, bond.atom_name_2)
        )
        if restraint_target is not None:
            continue

        elements.append(residue.atom_site(bond.atom_name_1).element)
        elements.append(residue.atom_site(bond.atom_name_2).element)

    radius_lookup = prepare_radius_lookup(elements, RadiusKind.COVALENT)
    radius_lookup.require_complete(
        f"heavy geometry fallback bonds for {residue.residue_id.display_token()}"
    )
    return radius_lookup


def bond_length_tolerance_angstrom(
    *,
    policy: HeavyGeometryPolicy,
    target: BondRestraintTarget,
) -> float:
    """Return the allowed bond-length deviation for one restraint target."""

    if target.esd_angstrom is None:
        return policy.restraint_bond_length_minimum_tolerance_angstrom

    return max(
        policy.restraint_bond_length_minimum_tolerance_angstrom,
        target.esd_angstrom * policy.restraint_bond_length_esd_multiplier,
    )


def bond_angle_tolerance_degrees(
    *,
    policy: HeavyGeometryPolicy,
    target: AngleRestraintTarget,
) -> float:
    """Return the allowed bond-angle deviation for one restraint target."""

    if target.esd_degrees is None:
        return policy.restraint_bond_angle_minimum_tolerance_degrees

    return max(
        policy.restraint_bond_angle_minimum_tolerance_degrees,
        target.esd_degrees * policy.restraint_bond_angle_esd_multiplier,
    )


def bond_angle_degrees(
    residue_geometry: ResidueGeometry,
    *,
    atom_name_1: str,
    center_atom_name: str,
    atom_name_2: str,
) -> float | None:
    """Return the angle in degrees, or None when a triplet is undefined."""

    atom_1 = residue_geometry.position(atom_name_1)
    center = residue_geometry.position(center_atom_name)
    atom_2 = residue_geometry.position(atom_name_2)
    vector_1 = (
        atom_1.x - center.x,
        atom_1.y - center.y,
        atom_1.z - center.z,
    )
    vector_2 = (
        atom_2.x - center.x,
        atom_2.y - center.y,
        atom_2.z - center.z,
    )
    dot_product = (
        vector_1[0] * vector_2[0]
        + vector_1[1] * vector_2[1]
        + vector_1[2] * vector_2[2]
    )
    norm_1 = sqrt(
        vector_1[0] * vector_1[0]
        + vector_1[1] * vector_1[1]
        + vector_1[2] * vector_1[2]
    )
    norm_2 = sqrt(
        vector_2[0] * vector_2[0]
        + vector_2[1] * vector_2[1]
        + vector_2[2] * vector_2[2]
    )
    if (
        norm_1 <= GEOMETRY_DEGENERATE_NORM_EPSILON
        or norm_2 <= GEOMETRY_DEGENERATE_NORM_EPSILON
    ):
        return None

    cosine = max(-1.0, min(1.0, dot_product / (norm_1 * norm_2)))
    return degrees(acos(cosine))


def residue_sort_key(residue_id: ResidueId) -> tuple[str, int, str]:
    """Return a stable sort key for one residue id."""

    insertion_code = residue_id.insertion_code or ""
    return (residue_id.chain_id, residue_id.seq_num, insertion_code)
