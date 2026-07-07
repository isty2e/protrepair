"""Runtime analysis assembly for workflow results."""

from collections.abc import Iterable

from protrepair.analysis.kinds import AnalysisKind, normalize_analysis_kinds
from protrepair.analysis.ramachandran import build_ramachandran_analysis
from protrepair.analysis.results import AnalysisBundle
from protrepair.analysis.secondary_structure import build_secondary_structure_analysis
from protrepair.structure.aggregate import ProteinStructure


def build_analysis_bundle(
    structure: ProteinStructure,
    *,
    requested_analyses: Iterable[AnalysisKind],
) -> AnalysisBundle:
    """Return the structured analysis bundle requested for one structure."""

    normalized_analyses = normalize_analysis_kinds(requested_analyses)
    secondary_structure = (
        build_secondary_structure_analysis(structure)
        if AnalysisKind.SECONDARY_STRUCTURE in normalized_analyses
        else None
    )
    ramachandran = (
        build_ramachandran_analysis(structure)
        if AnalysisKind.RAMACHANDRAN in normalized_analyses
        else None
    )
    return AnalysisBundle(
        secondary_structure=secondary_structure,
        ramachandran=ramachandran,
    )
