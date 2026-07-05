"""Peptide-adjacency predicates over canonical structures."""

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
    if _peptide_cn_distance(left_residue, right_residue, structure=structure) > (
        PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
    ):
        return False

    return _is_immediate_sequence_neighbor(
        left_residue,
        right_residue,
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
) -> float:
    """Return the C-N distance for a candidate peptide neighbor pair."""

    left_geometry = structure.residue_geometry(
        structure.constitution.residue_index(left_residue.residue_id)
    )
    right_geometry = structure.residue_geometry(
        structure.constitution.residue_index(right_residue.residue_id)
    )
    return left_geometry.position("C").distance_to(right_geometry.position("N"))


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


def _is_immediate_sequence_neighbor(
    left_residue: ResidueSite,
    right_residue: ResidueSite,
) -> bool:
    """Return whether residue ids describe an unambiguous next sequence slot."""

    left_id = left_residue.residue_id
    right_id = right_residue.residue_id
    return (
        left_id.chain_id == right_id.chain_id
        and left_id.insertion_code is None
        and right_id.insertion_code is None
        and right_id.seq_num == left_id.seq_num + 1
    )
