"""Structured analysis result contracts."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeGuard

from protrepair.analysis.kinds import AnalysisKind, normalize_analysis_kinds
from protrepair.structure.labels import ResidueId


class RamachandranCategory(str, Enum):
    """Closed coarse Ramachandran category labels."""

    HELIX = "helix"
    BETA = "beta"
    LEFT_HANDED = "left_handed"
    OTHER = "other"

    def secondary_structure_label(self) -> str:
        """Project this category to one coarse secondary-structure label."""

        if self is RamachandranCategory.HELIX:
            return "H"
        if self is RamachandranCategory.BETA:
            return "E"
        return "C"


@dataclass(frozen=True, slots=True)
class SecondaryStructureAssignment:
    """Secondary-structure assignment for a single residue."""

    residue_id: ResidueId
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError(
                "secondary-structure assignments require a ResidueId"
            )
        if not isinstance(self.label, str):
            raise TypeError("secondary-structure labels must be strings")

        label = self.label.strip()
        if not label:
            raise ValueError("secondary-structure labels must not be blank")
        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class SecondaryStructureAnalysis:
    """Structured secondary-structure analysis output."""

    assignments: tuple[SecondaryStructureAssignment, ...]

    def __post_init__(self) -> None:
        assignments = tuple(self.assignments)
        for assignment in assignments:
            if not isinstance(assignment, SecondaryStructureAssignment):
                raise TypeError(
                    "secondary-structure analyses require "
                    "SecondaryStructureAssignment values"
                )

        object.__setattr__(self, "assignments", assignments)

    def label_for(self, residue_id: ResidueId) -> str | None:
        """Return the assignment label for a residue if available."""

        for assignment in self.assignments:
            if assignment.residue_id == residue_id:
                return assignment.label

        return None


@dataclass(frozen=True, slots=True)
class RamachandranPoint:
    """Ramachandran result for a single residue."""

    residue_id: ResidueId
    phi_degrees: float | None
    psi_degrees: float | None
    category: RamachandranCategory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError("Ramachandran points require a ResidueId")
        object.__setattr__(
            self,
            "phi_degrees",
            _normalize_optional_angle(
                self.phi_degrees,
                "Ramachandran point phi_degrees",
            ),
        )
        object.__setattr__(
            self,
            "psi_degrees",
            _normalize_optional_angle(
                self.psi_degrees,
                "Ramachandran point psi_degrees",
            ),
        )
        if self.category is not None and not isinstance(
            self.category,
            RamachandranCategory,
        ):
            raise TypeError(
                "Ramachandran point category must be a RamachandranCategory or None"
            )


@dataclass(frozen=True, slots=True)
class RamachandranAnalysis:
    """Structured Ramachandran analysis output."""

    points: tuple[RamachandranPoint, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        for point in points:
            if not isinstance(point, RamachandranPoint):
                raise TypeError(
                    "Ramachandran analyses require RamachandranPoint values"
                )

        object.__setattr__(self, "points", points)

    def point_for(self, residue_id: ResidueId) -> RamachandranPoint | None:
        """Return the point for a residue if available."""

        for point in self.points:
            if point.residue_id == residue_id:
                return point

        return None


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Structured collection of enabled analysis outputs."""

    secondary_structure: SecondaryStructureAnalysis | None = None
    ramachandran: RamachandranAnalysis | None = None

    def __post_init__(self) -> None:
        if self.secondary_structure is not None and not isinstance(
            self.secondary_structure,
            SecondaryStructureAnalysis,
        ):
            raise TypeError(
                "analysis bundles require SecondaryStructureAnalysis or None"
            )
        if self.ramachandran is not None and not isinstance(
            self.ramachandran,
            RamachandranAnalysis,
        ):
            raise TypeError("analysis bundles require RamachandranAnalysis or None")

    def has(self, analysis_kind: AnalysisKind) -> bool:
        """Return whether a specific analysis result is populated."""

        (normalized_analysis_kind,) = normalize_analysis_kinds((analysis_kind,))
        if normalized_analysis_kind is AnalysisKind.SECONDARY_STRUCTURE:
            return self.secondary_structure is not None

        if normalized_analysis_kind is AnalysisKind.RAMACHANDRAN:
            return self.ramachandran is not None

        return False


def _normalize_optional_angle(
    angle_degrees: float | None,
    field_name: str,
) -> float | None:
    """Return one finite optional analysis angle."""

    if angle_degrees is None:
        return None
    if not _is_real_number(angle_degrees):
        raise TypeError(f"{field_name} must be a finite number or None")
    angle = float(angle_degrees)
    if not math.isfinite(angle):
        raise ValueError(f"{field_name} must be finite")

    return angle


def _is_real_number(value: object) -> TypeGuard[int | float]:
    """Return whether a value is a non-bool real number accepted by analysis DTOs."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)
