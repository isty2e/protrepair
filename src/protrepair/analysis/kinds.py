"""Analysis-mode taxonomy."""

from enum import Enum


class AnalysisKind(str, Enum):
    """Analysis modes supported by the package."""

    SECONDARY_STRUCTURE = "secondary_structure"
    RAMACHANDRAN = "ramachandran"
