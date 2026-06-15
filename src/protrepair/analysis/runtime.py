"""Runtime analysis assembly for workflow results."""

from protrepair.analysis.kinds import AnalysisKind
from protrepair.analysis.ramachandran import build_ramachandran_analysis
from protrepair.analysis.results import AnalysisBundle
from protrepair.analysis.secondary_structure import build_secondary_structure_analysis
from protrepair.structure.aggregate import ProteinStructure


def build_analysis_bundle(
    structure: ProteinStructure,
    *,
    requested_analyses: frozenset[AnalysisKind],
) -> AnalysisBundle:
    """Return the structured analysis bundle requested for one structure."""

    secondary_structure = (
        build_secondary_structure_analysis(structure)
        if AnalysisKind.SECONDARY_STRUCTURE in requested_analyses
        else None
    )
    ramachandran = (
        build_ramachandran_analysis(structure)
        if AnalysisKind.RAMACHANDRAN in requested_analyses
        else None
    )
    return AnalysisBundle(
        secondary_structure=secondary_structure,
        ramachandran=ramachandran,
    )
