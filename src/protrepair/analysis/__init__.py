"""Analysis modules for the redesigned ProtRepair package."""

from protrepair.analysis.kinds import AnalysisKind
from protrepair.analysis.results import (
    AnalysisBundle,
    RamachandranAnalysis,
    RamachandranCategory,
    RamachandranPoint,
    SecondaryStructureAnalysis,
    SecondaryStructureAssignment,
)

__all__ = [
    "AnalysisBundle",
    "AnalysisKind",
    "RamachandranAnalysis",
    "RamachandranCategory",
    "RamachandranPoint",
    "SecondaryStructureAnalysis",
    "SecondaryStructureAssignment",
]
