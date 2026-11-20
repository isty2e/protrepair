"""Peptide-adjacency predicates over canonical structures."""

from protrepair.errors import ResidueNotFoundError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.topology import (
    BondRelationshipType,
    is_covalent_like_relationship,
)

PEPTIDE_CN_DISTANCE_MAX_ANGSTROM = 1.8
PEPTIDE_CAP_COMPONENT_IDS = frozenset(("ACE", "NME"))


def peptide_ports_allow_link(carbonyl_component: str, amide_component: str) -> bool:
    """Exclude the closed side of a known cap from a directed C--N link."""

    return carbonyl_component != "NME" and amide_component != "ACE"


def are_peptide_adjacent(
    left_residue: ResidueSite,
    right_residue: ResidueSite,
    *,
    structure: ProteinStructure,
) -> bool:
    """Return whether two residues can support one peptide-bond torsion."""

    if (
        not peptide_ports_allow_link(
            left_residue.component_id, right_residue.component_id
        )
        or not left_residue.has_atom_site("C")
        or not right_residue.has_atom_site("N")
    ):
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

    if PEPTIDE_CAP_COMPONENT_IDS.intersection(
        (left_residue.component_id, right_residue.component_id)
    ):
        return _has_topology_peptide_bond(
            left_residue, right_residue, structure=structure, require_single=True
        )

    if _are_consecutive_polymer_residue_slots(
        left_residue, right_residue, structure=structure
    ):
        omissions = structure.provenance.ingress.zero_occupancy_residue_omissions
        if not omissions or not any(
            record.preceding_residue_id == left_residue.residue_id
            and record.following_residue_id == right_residue.residue_id
            and structure.constitution.residue_or_ligand(record.residue_id) is None
            for record in omissions
        ):
            return True

    return _has_topology_peptide_bond(left_residue, right_residue, structure=structure)


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
    require_single: bool = False,
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
    if bond is None:
        return False
    if not require_single:
        return is_covalent_like_relationship(bond)
    return (
        bond.relationship_type is BondRelationshipType.COVALENT
        and bond.order == 1
        and not bond.aromatic
        and structure.constitution.atom_site_at(left_carbon).element == "C"
        and structure.constitution.atom_site_at(right_nitrogen).element == "N"
    )


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
