"""Ramachandran analysis over canonical protein structures."""

from protrepair.analysis.results import (
    RamachandranAnalysis,
    RamachandranCategory,
    RamachandranPoint,
)
from protrepair.geometry.internal_coordinates import InternalCoordinateFrame
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.peptide import are_peptide_adjacent


def build_ramachandran_analysis(
    structure: ProteinStructure,
) -> RamachandranAnalysis:
    """Return Ramachandran analysis for one canonical structure."""

    points: list[RamachandranPoint] = []
    for chain in structure.constitution.chains:
        residues = tuple(
            residue
            for residue in chain.residues
            if not residue.is_hetero
        )
        for index, residue in enumerate(residues):
            previous_residue = residues[index - 1] if index > 0 else None
            next_residue = (
                residues[index + 1]
                if index + 1 < len(residues)
                else None
            )
            phi_degrees = _phi_degrees(
                previous_residue,
                residue,
                structure=structure,
            )
            psi_degrees = _psi_degrees(
                residue,
                next_residue,
                structure=structure,
            )
            points.append(
                RamachandranPoint(
                    residue_id=residue.residue_id,
                    phi_degrees=phi_degrees,
                    psi_degrees=psi_degrees,
                    category=_ramachandran_category(
                        phi_degrees=phi_degrees,
                        psi_degrees=psi_degrees,
                    ),
                )
            )

    return RamachandranAnalysis(points=tuple(points))


def _phi_degrees(
    previous_residue: ResidueSite | None,
    residue: ResidueSite,
    *,
    structure: ProteinStructure,
) -> float | None:
    """Return phi torsion in degrees when the required atoms are present."""

    if previous_residue is None:
        return None
    if not are_peptide_adjacent(previous_residue, residue, structure=structure):
        return None
    if not previous_residue.has_atom_site("C"):
        return None
    if not residue.has_atom_site("N"):
        return None
    if not residue.has_atom_site("CA"):
        return None
    if not residue.has_atom_site("C"):
        return None

    previous_geometry = structure.residue_geometry(
        structure.constitution.residue_index(previous_residue.residue_id)
    )
    residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue.residue_id)
    )
    return InternalCoordinateFrame.torsion(
        previous_geometry.position("C"),
        residue_geometry.position("N"),
        residue_geometry.position("CA"),
        residue_geometry.position("C"),
    )


def _psi_degrees(
    residue: ResidueSite,
    next_residue: ResidueSite | None,
    *,
    structure: ProteinStructure,
) -> float | None:
    """Return psi torsion in degrees when the required atoms are present."""

    if next_residue is None:
        return None
    if not are_peptide_adjacent(residue, next_residue, structure=structure):
        return None
    if not residue.has_atom_site("N"):
        return None
    if not residue.has_atom_site("CA"):
        return None
    if not residue.has_atom_site("C"):
        return None
    if not next_residue.has_atom_site("N"):
        return None

    residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue.residue_id)
    )
    next_geometry = structure.residue_geometry(
        structure.constitution.residue_index(next_residue.residue_id)
    )
    return InternalCoordinateFrame.torsion(
        residue_geometry.position("N"),
        residue_geometry.position("CA"),
        residue_geometry.position("C"),
        next_geometry.position("N"),
    )


def _ramachandran_category(
    *,
    phi_degrees: float | None,
    psi_degrees: float | None,
) -> RamachandranCategory | None:
    """Return one coarse Ramachandran region label."""

    if phi_degrees is None or psi_degrees is None:
        return None

    if -160.0 <= phi_degrees <= -20.0 and -90.0 <= psi_degrees <= 45.0:
        return RamachandranCategory.HELIX
    if -180.0 <= phi_degrees <= -40.0 and (
        psi_degrees >= 90.0 or psi_degrees <= -120.0
    ):
        return RamachandranCategory.BETA
    if 20.0 <= phi_degrees <= 120.0 and -20.0 <= psi_degrees <= 120.0:
        return RamachandranCategory.LEFT_HANDED

    return RamachandranCategory.OTHER
