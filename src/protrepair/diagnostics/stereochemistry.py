"""Side-chain stereochemistry diagnostics over canonical structures."""

from dataclasses import dataclass

from protrepair.chemistry import (
    ComponentLibrary,
    ResidueTemplate,
    TetrahedralCenterSemantics,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId

STEREOCHEMISTRY_DEGENERACY_EPSILON = 1.0e-6


@dataclass(frozen=True, slots=True)
class SidechainStereochemistryViolation:
    """One violated tetrahedral side-chain stereochemistry expectation."""

    residue_id: ResidueId
    component_id: str
    center_atom_name: str
    ordered_neighbor_atom_names: tuple[str, str, str]
    expected_orientation_sign: int
    observed_signed_volume: float

    def orientation_label(self) -> str:
        """Return a human-readable orientation label for the observed volume."""

        if abs(self.observed_signed_volume) <= STEREOCHEMISTRY_DEGENERACY_EPSILON:
            return "degenerate"

        if self.observed_signed_volume > 0:
            return "positive"

        return "negative"


@dataclass(frozen=True, slots=True)
class StereochemistryReport:
    """Structured stereochemistry findings for a canonical structure."""

    violations: tuple[SidechainStereochemistryViolation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "violations", tuple(self.violations))

    def is_empty(self) -> bool:
        """Return whether the report contains no stereochemical problems."""

        return not self.violations

    def to_issues(self) -> tuple[ValidationIssue, ...]:
        """Project stereochemistry violations into validation issues."""

        return tuple(
            ValidationIssue.for_residue(
                kind=ValidationIssueKind.INVALID_STEREOCHEMISTRY,
                severity=IssueSeverity.WARNING,
                message=(
                    f"{violation.residue_id.display_token()} has inverted or "
                    f"degenerate side-chain stereochemistry at "
                    f"{violation.center_atom_name} "
                    f"({violation.orientation_label()} orientation observed)"
                ),
                residue_id=violation.residue_id,
            )
            for violation in self.violations
        )


def detect_sidechain_stereochemistry(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
) -> StereochemistryReport:
    """Return side-chain stereochemistry violations for supported residues."""

    violations: list[SidechainStereochemistryViolation] = []
    for residue in structure.constitution.iter_residues():
        template = component_library.get(residue.component_id)
        if template is None or not template.has_tetrahedral_stereochemistry():
            continue

        residue_geometry = structure.residue_geometry(
            structure.constitution.residue_index(residue.residue_id)
        )
        violations.extend(
            detect_residue_stereochemistry_violations(
                residue,
                residue_geometry=residue_geometry,
                template=template,
            )
        )

    return StereochemistryReport(violations=tuple(violations))


def detect_residue_stereochemistry_violations(
    residue: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    template: ResidueTemplate,
) -> tuple[SidechainStereochemistryViolation, ...]:
    """Return tetrahedral side-chain violations for one supported residue."""

    violations: list[SidechainStereochemistryViolation] = []
    for center_semantics in template.tetrahedral_centers:
        if not residue_has_tetrahedral_center(residue, center_semantics):
            continue

        signed_volume = tetrahedral_signed_volume(
            residue_geometry,
            center_semantics=center_semantics,
        )
        if signed_volume * center_semantics.expected_orientation_sign > (
            STEREOCHEMISTRY_DEGENERACY_EPSILON
        ):
            continue

        violations.append(
            SidechainStereochemistryViolation(
                residue_id=residue.residue_id,
                component_id=residue.component_id,
                center_atom_name=center_semantics.center_atom_name,
                ordered_neighbor_atom_names=center_semantics.ordered_neighbor_atom_names,
                expected_orientation_sign=center_semantics.expected_orientation_sign,
                observed_signed_volume=signed_volume,
            )
        )

    return tuple(violations)


def residue_has_tetrahedral_center(
    residue: ResidueSite,
    center_semantics: TetrahedralCenterSemantics,
) -> bool:
    """Return whether one residue contains all atoms needed for one center."""

    required_atom_names = (
        center_semantics.center_atom_name,
        *center_semantics.ordered_neighbor_atom_names,
    )
    return all(residue.has_atom_site(atom_name) for atom_name in required_atom_names)


def tetrahedral_signed_volume(
    residue_geometry: ResidueGeometry,
    *,
    center_semantics: TetrahedralCenterSemantics,
) -> float:
    """Return the signed tetrahedral volume for one residue-local center."""

    center = residue_geometry.position(center_semantics.center_atom_name)
    first, second, third = (
        residue_geometry.position(atom_name)
        for atom_name in center_semantics.ordered_neighbor_atom_names
    )
    vector_1 = (
        first.x - center.x,
        first.y - center.y,
        first.z - center.z,
    )
    vector_2 = (
        second.x - center.x,
        second.y - center.y,
        second.z - center.z,
    )
    vector_3 = (
        third.x - center.x,
        third.y - center.y,
        third.z - center.z,
    )
    return (
        vector_1[0] * ((vector_2[1] * vector_3[2]) - (vector_2[2] * vector_3[1]))
        - vector_1[1] * ((vector_2[0] * vector_3[2]) - (vector_2[2] * vector_3[0]))
        + vector_1[2] * ((vector_2[0] * vector_3[1]) - (vector_2[1] * vector_3[0]))
    )
