"""Disulfide semantics projected from canonical structure topology."""

from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import TopologyBond, is_covalent_like_relationship


def disulfide_atom_ref_pairs(
    structure: ProteinStructure,
) -> frozenset[tuple[AtomRef, AtomRef]]:
    """Return canonical CYS SG-SG pairs supported by covalent-like topology."""

    pairs: set[tuple[AtomRef, AtomRef]] = set()
    for bond in structure.topology.bonds:
        pair = _disulfide_atom_ref_pair(structure, bond)
        if pair is not None:
            pairs.add(pair)

    return frozenset(pairs)


def disulfide_bonded_cysteine_residue_ids(
    structure: ProteinStructure,
) -> frozenset[ResidueId]:
    """Return CYS residue identities participating in canonical disulfides."""

    return frozenset(
        atom_ref.residue_id
        for atom_ref_pair in disulfide_atom_ref_pairs(structure)
        for atom_ref in atom_ref_pair
    )


def _disulfide_atom_ref_pair(
    structure: ProteinStructure,
    bond: TopologyBond,
) -> tuple[AtomRef, AtomRef] | None:
    """Project one covalent-like topology bond when both endpoints are CYS SG."""

    if not is_covalent_like_relationship(bond):
        return None

    atom_ref_1 = structure.constitution.atom_ref_at(bond.atom_index_1)
    atom_ref_2 = structure.constitution.atom_ref_at(bond.atom_index_2)
    if atom_ref_1.atom_name != "SG" or atom_ref_2.atom_name != "SG":
        return None

    residue_1 = structure.constitution.residue_site_at(
        structure.constitution.residue_index_for_atom_index(bond.atom_index_1)
    )
    residue_2 = structure.constitution.residue_site_at(
        structure.constitution.residue_index_for_atom_index(bond.atom_index_2)
    )
    if residue_1.component_id != "CYS" or residue_2.component_id != "CYS":
        return None

    if atom_ref_2 < atom_ref_1:
        return (atom_ref_2, atom_ref_1)

    return (atom_ref_1, atom_ref_2)
