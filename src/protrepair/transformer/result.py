"""Transformer-owned accumulated structure and diagnostic result carrier."""

from dataclasses import dataclass

from typing_extensions import Self

from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity
from protrepair.structure.aggregate import ProteinStructure


@dataclass(frozen=True, slots=True)
class TransformationResult:
    """Accumulated structure plus transformer-emitted repair and issue trace."""

    structure: ProteinStructure
    repairs: tuple[RepairEvent, ...]
    issues: tuple[ValidationIssue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "repairs", tuple(self.repairs))
        object.__setattr__(self, "issues", tuple(self.issues))

    def has_errors(self) -> bool:
        """Return whether any validation issue is error-severity."""

        return any(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Return whether any validation issue is warning-severity."""

        return any(issue.severity is IssueSeverity.WARNING for issue in self.issues)

    def repair_count(self) -> int:
        """Return the number of recorded repair events."""

        return len(self.repairs)

    def error_count(self) -> int:
        """Return the number of error-severity validation issues."""

        return sum(1 for issue in self.issues if issue.severity is IssueSeverity.ERROR)

    def warning_count(self) -> int:
        """Return the number of warning-severity validation issues."""

        return sum(
            1 for issue in self.issues if issue.severity is IssueSeverity.WARNING
        )

    def issue_count(self) -> int:
        """Return the number of recorded validation issues."""

        return len(self.issues)

    def with_appended_issues(
        self,
        issues: tuple[ValidationIssue, ...],
    ) -> Self:
        """Return a copy with additional validation issues appended."""

        if not issues:
            return self

        return type(self)(
            structure=self.structure,
            repairs=self.repairs,
            issues=self.issues + tuple(issues),
        )

    def with_structure(self, structure: ProteinStructure) -> Self:
        """Return a copy with an updated structure."""

        return type(self)(
            structure=structure,
            repairs=self.repairs,
            issues=self.issues,
        )
