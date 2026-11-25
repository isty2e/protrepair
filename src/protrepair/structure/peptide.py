"""Peptide-adjacency predicates over canonical structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protrepair.errors import ResidueNotFoundError
from protrepair.structure.backbone_availability import unresolved_backbone
from protrepair.structure.bound_peptide_ports import active_peptide_ports
from protrepair.structure.cap_attachment import PEPTIDE_CAP_COMPONENT_IDS
from protrepair.structure.constitution import ResidueRole, ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    is_covalent_like_relationship,
)

if TYPE_CHECKING:
    from protrepair.structure.aggregate import ProteinStructure

PEPTIDE_CN_DISTANCE_MAX_ANGSTROM = 1.8


def peptide_ports_allow_link(carbonyl_component: str, amide_component: str) -> bool:
    """Exclude the closed side of a known cap from a directed C--N link."""

    return carbonyl_component != "NME" and amide_component != "ACE"


def has_verified_backbone_link(
    structure: ProteinStructure, before: ResidueSite, after: ResidueSite
) -> bool:
    """Check source continuity and C/N chemistry without reading coordinates.

    Reconstruction and regional packing both need this stronger test: adjacent
    slots can flank a TER, and a placeholder can give any C--N distance.
    """
    if not _are_consecutive_polymer_residue_slots(before, after, structure=structure):
        return False
    if not peptide_ports_allow_link(before.component_id, after.component_id) or any(
        before.has_atom_site(name) for name in ("OXT", "OT2")
    ):
        return False
    c = structure.constitution
    left, right = before.residue_id, after.residue_id
    if any(
        c.role_at(c.residue_index(rid)) is not ResidueRole.POLYMER
        for rid in (left, right)
    ):
        return False
    carbon, nitrogen = AtomRef(left, "C"), AtomRef(right, "N")
    i, j = c.resolve_atom_index(carbon), c.resolve_atom_index(nitrogen)
    if (
        i is None
        or j is None
        or c.atom_site_at(i).element != "C"
        or c.atom_site_at(j).element != "N"
    ):
        return False

    ingress = structure.provenance.ingress
    obs = ingress.observation
    ports = {carbon, nitrogen}
    if ports.intersection(ingress.backbone_link_vetoes) or ports.intersection(
        ingress.uncertain_peptide_link_ports or ()
    ):
        return False
    if obs is not None and (
        ports.intersection(obs.unresolved_polymer_endpoints)
        or ports.intersection(obs.unresolved_conect_ports)
        or any(
            ep.atom_ref in ports
            for edge in obs.excluded_coordination_connections
            for ep in edge.endpoint_pair()
        )
    ):
        return False
    edge = structure.topology.bond_between(i, j)
    if edge is not None and (
        edge.relationship_type is not BondRelationshipType.COVALENT
        or edge.aromatic
        or edge.order not in (None, 1)
    ):
        return False
    from_path = (left, right) in ingress.backbone_neighbors
    if from_path and obs is not None:
        from_path = all(
            (original := obs.constitution.residue_or_ligand(site.residue_id))
            is not None
            and original.component_id == site.component_id
            and obs.constitution.role_at(
                obs.constitution.residue_index(site.residue_id)
            )
            is ResidueRole.POLYMER
            for site in (before, after)
        )
    if not from_path and not (
        edge is not None and edge.provenance is BondProvenance.SOURCE_EXPLICIT
    ):
        return False

    for index, partner, rid in ((i, j, left), (j, i, right)):
        for bond in structure.topology.incident_bonds_by_atom().get(index, ()):
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            a, b = bond.endpoint_pair()
            other = b if a == index else a
            if bond.relationship_type is not BondRelationshipType.COVALENT or (
                other != partner and c.atom_ref_at(other).residue_id != rid
            ):
                return False
    return True


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
        or unresolved_backbone(structure, left_residue.residue_id)
        or unresolved_backbone(structure, right_residue.residue_id)
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

    ports = active_peptide_ports()
    if ports is not None and (
        ports.reserves(structure, AtomRef(left_residue.residue_id, "C"))
        or ports.reserves(structure, AtomRef(right_residue.residue_id, "N"))
    ):
        return _has_topology_peptide_bond(
            left_residue, right_residue, structure=structure, require_single=True
        )

    if _are_consecutive_polymer_residue_slots(
        left_residue, right_residue, structure=structure
    ):
        omissions = structure.provenance.ingress.recorded_residue_gaps()
        if not omissions or not any(
            record.preceding_residue_id == left_residue.residue_id
            and record.following_residue_id == right_residue.residue_id
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
