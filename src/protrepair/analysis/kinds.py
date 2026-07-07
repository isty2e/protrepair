"""Analysis-mode taxonomy."""

from collections.abc import Iterable
from enum import Enum


class AnalysisKind(str, Enum):
    """Analysis modes supported by the package."""

    SECONDARY_STRUCTURE = "secondary_structure"
    RAMACHANDRAN = "ramachandran"


def normalize_analysis_kinds(
    analysis_kinds: Iterable[AnalysisKind],
) -> frozenset[AnalysisKind]:
    """Normalize one public analysis-kind collection into canonical enum values."""

    normalized_analysis_kinds: set[AnalysisKind] = set()
    for analysis_kind in analysis_kinds:
        if not isinstance(analysis_kind, AnalysisKind):
            raise TypeError("analyses must contain AnalysisKind values")
        normalized_analysis_kinds.add(analysis_kind)

    return frozenset(normalized_analysis_kinds)
