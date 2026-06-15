"""Coarse secondary-structure analysis over canonical protein structures."""

from protrepair.analysis.ramachandran import build_ramachandran_analysis
from protrepair.analysis.results import (
    SecondaryStructureAnalysis,
    SecondaryStructureAssignment,
)
from protrepair.structure.aggregate import ProteinStructure


def build_secondary_structure_analysis(
    structure: ProteinStructure,
) -> SecondaryStructureAnalysis:
    """Return coarse secondary-structure assignments for one structure."""

    ramachandran = build_ramachandran_analysis(structure)
    assignments = tuple(
        SecondaryStructureAssignment(
            residue_id=point.residue_id,
            label=_secondary_structure_label(point.category),
        )
        for point in ramachandran.points
    )
    return SecondaryStructureAnalysis(assignments=assignments)


def _secondary_structure_label(category: str | None) -> str:
    """Project one coarse Ramachandran category into an assignment label."""

    if category == "helix":
        return "H"
    if category == "beta":
        return "E"
    return "C"
