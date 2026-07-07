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
            label=(
                "C"
                if point.category is None
                else point.category.secondary_structure_label()
            ),
        )
        for point in ramachandran.points
    )
    return SecondaryStructureAnalysis(assignments=assignments)
