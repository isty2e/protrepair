"""Analysis modules for the redesigned ProtRepair package."""

from protrepair.analysis.hydrogen_length_reference import (
    HydrogenLengthComparison,
    HydrogenLengthConvention,
    HydrogenLengthReference,
    HydrogenLengthStatus,
)
from protrepair.analysis.hydrogen_lengths import (
    HydrogenLengthObservation,
    HydrogenParentPositionConflict,
    analyze_hydrogen_lengths,
)
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
    "HydrogenLengthComparison",
    "HydrogenLengthConvention",
    "HydrogenLengthObservation",
    "HydrogenLengthReference",
    "HydrogenLengthStatus",
    "HydrogenParentPositionConflict",
    "RamachandranAnalysis",
    "RamachandranCategory",
    "RamachandranPoint",
    "SecondaryStructureAnalysis",
    "SecondaryStructureAssignment",
    "analyze_hydrogen_lengths",
]
