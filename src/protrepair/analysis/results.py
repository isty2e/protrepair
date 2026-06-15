"""Structured analysis result contracts."""

from dataclasses import dataclass

from protrepair.analysis.kinds import AnalysisKind
from protrepair.structure.labels import ResidueId


@dataclass(frozen=True, slots=True)
class SecondaryStructureAssignment:
    """Secondary-structure assignment for a single residue."""

    residue_id: ResidueId
    label: str

    def __post_init__(self) -> None:
        label = self.label.strip()
        if not label:
            raise ValueError("secondary-structure labels must not be blank")
        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class SecondaryStructureAnalysis:
    """Structured secondary-structure analysis output."""

    assignments: tuple[SecondaryStructureAssignment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))

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
    category: str | None = None

    def __post_init__(self) -> None:
        category = self.category
        if category is not None:
            category = category.strip() or None
        object.__setattr__(self, "category", category)


@dataclass(frozen=True, slots=True)
class RamachandranAnalysis:
    """Structured Ramachandran analysis output."""

    points: tuple[RamachandranPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))

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

    def has(self, analysis_kind: AnalysisKind) -> bool:
        """Return whether a specific analysis result is populated."""

        if analysis_kind is AnalysisKind.SECONDARY_STRUCTURE:
            return self.secondary_structure is not None

        if analysis_kind is AnalysisKind.RAMACHANDRAN:
            return self.ramachandran is not None

        return False
