"""Peptide-adjacency predicates over canonical structures."""

from protrepair.errors import ResidueNotFoundError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.topology import is_covalent_like_relationship

PEPTIDE_CN_DISTANCE_MAX_ANGSTROM = 1.8


def are_peptide_adjacent(
    left_residue: ResidueSite,
    right_residue: ResidueSite,
    *,
    structure: ProteinStructure,
) -> bool:
    """Return whether two residues can support one peptide-bond torsion."""

    if not left_residue.has_atom_site("C") or not right_residue.has_atom_site("N"):
        return False

    peptide_cn_distance = _peptide_cn_distance(
        left_residue,
        right_residue,
        structure=structure,
    )
    if (
        peptide_cn_distance is None
        or peptide_cn_distance > PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
    ):
        return False

    return _are_consecutive_polymer_residue_slots(
        left_residue,
        right_residue,
        structure=structure,
    ) or _has_topology_peptide_bond(
        left_residue,
        right_residue,
        structure=structure,
    )


def _peptide_cn_distance(
    left_residue: ResidueSite,
    right_residue: ResidueSite,
    *,
    structure: ProteinStructure,
) -> float | None:
    """Return the C-N distance for a candidate peptide neighbor pair."""

    left_carbon = structure.constitution.resolve_atom_index(
        AtomRef(left_residue.residue_id, "C")
    )
    right_nitrogen = structure.constitution.resolve_atom_index(
        AtomRef(right_residue.residue_id, "N")
    )
    if left_carbon is None or right_nitrogen is None:
        return None

    return structure.geometry.position(left_carbon).distance_to(
        structure.geometry.position(right_nitrogen)
    )


def _has_topology_peptide_bond(
    left_residue: ResidueSite,
    right_residue: ResidueSite,
    *,
    structure: ProteinStructure,
) -> bool:
    """Return whether topology explicitly carries a covalent C-N peptide edge."""

    left_carbon = structure.constitution.resolve_atom_index(
        AtomRef(left_residue.residue_id, "C")
    )
    right_nitrogen = structure.constitution.resolve_atom_index(
        AtomRef(right_residue.residue_id, "N")
    )
    if left_carbon is None or right_nitrogen is None:
        return False

    bond = structure.topology.bond_between(left_carbon, right_nitrogen)
    return bond is not None and is_covalent_like_relationship(bond)


def _are_consecutive_polymer_residue_slots(
    left_residue: ResidueSite,
    right_residue: ResidueSite,
    *,
    structure: ProteinStructure,
) -> bool:
    """Return whether residues are consecutive in their canonical chain order."""

    left_id = left_residue.residue_id
    right_id = right_residue.residue_id
    if left_id.chain_id != right_id.chain_id:
        return False
    if not structure.constitution.has_chain(left_id.chain_id):
        return False

    try:
        left_position = structure.constitution.residue_index(left_id).value
        right_position = structure.constitution.residue_index(right_id).value
    except ResidueNotFoundError:
        return False

    return right_position == left_position + 1
