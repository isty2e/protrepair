"""Bind the stock peptide caps without enrolling them as amino-acid families."""

from collections.abc import Mapping
from dataclasses import dataclass

from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.peptide import PEPTIDE_CAP_COMPONENT_IDS
from protrepair.structure.topology import (
    BondRelationshipType,
    SourceBondRecordType,
    TopologyBond,
)

_METHYL_H = ("H1", "H2", "H3")
_AMIDE_H = frozenset(("H", "HN", "HN1", "HN2"))
_TYPED_LINKS = frozenset(
    (SourceBondRecordType.PDB_LINK, SourceBondRecordType.MMCIF_STRUCT_CONN)
)
_Snapshot = ProteinStructure | StructureObservation


@dataclass(frozen=True, slots=True)
class LinkedPeptideCap:
    """An occupied cap's link and H requirements in one snapshot."""

    residue_id: ResidueId
    component_id: str
    partner: AtomRef | None = None
    heavy_bond_updates: tuple[TopologyBond, ...] = ()
    hydrogen_bond_updates: tuple[TopologyBond, ...] = ()
    missing_hydrogen_parents: tuple[tuple[str, str], ...] = ()
    existing_hydrogen_parents: tuple[tuple[str, str], ...] = ()
    refusal: str | None = None

    def link_is_realized(self) -> bool:
        """Check current graph readiness independently of H placement."""
        return self.refusal is None and not self.heavy_bond_updates

    def hydrogens_are_complete(self) -> bool:
        """Require compatible identity, canonical bonds and a full inventory."""
        return (
            self.link_is_realized()
            and not self.missing_hydrogen_parents
            and not self.hydrogen_bond_updates
        )

    def expected_hydrogen_names(self) -> tuple[str, ...]:
        """Name existing and missing H without renaming an isotope."""
        return tuple(
            name
            for name, _ in (
                *self.existing_hydrogen_parents,
                *self.missing_hydrogen_parents,
            )
        )


def observe_linked_peptide_cap(
    structure: ProteinStructure, residue: ResidueSite
) -> LinkedPeptideCap | None:
    """Bind an occupied cap, even when its source linkage cannot be realized."""
    if residue.component_id not in PEPTIDE_CAP_COMPONENT_IDS:
        return None
    residue_id = residue.residue_id
    observation = structure.provenance.ingress.observation
    original = (
        None
        if observation is None
        else observation.constitution.residue_or_ligand(residue_id)
    )
    external = _external_bonds(structure, residue_id)
    source_external = (
        ()
        if observation is None or original is None
        else _external_bonds(observation, residue_id)
    )
    excluded = (
        () if observation is None else observation.excluded_coordination_connections
    )
    excluded_cap_link = any(
        endpoint.atom_ref.residue_id == residue_id
        for connection in excluded
        for endpoint in connection.endpoint_pair()
    )
    if not (external or source_external or excluded_cap_link):
        return None

    try:
        if len(external) != 1 or excluded_cap_link:
            raise ValueError("the cap requires one surviving covalent partner")
        (link,) = external
        local, partner = _external_refs(structure, residue_id, link)
        is_nme = residue.component_id == "NME"
        if (local.atom_name, partner.atom_name) != (
            ("N", "C") if is_nme else ("C", "N")
        ):
            raise ValueError("the attachment does not occupy the cap's peptide port")
        neighbor = structure.constitution.residue_or_ligand(partner.residue_id)
        if (
            neighbor is None
            or neighbor.is_hetero
            or neighbor.component_id in PEPTIDE_CAP_COMPONENT_IDS
        ):
            raise ValueError("the cap has no ordinary polymer partner")
        evidence = structure.provenance.ingress.pdb_adjacent_cap_link_for(residue_id)
        qualified_pdb_link = (
            evidence is not None
            and evidence.bond_in(
                structure.constitution, structure.geometry, structure.topology
            )
            == link
        )
        realized = _checked_link(link, qualified_pdb_link=qualified_pdb_link)
        _check_partner(structure, partner, neighbor, link)

        require_source_cap_atoms(structure, residue, partner)
        if observation is not None:
            for source_link in source_external:
                if _external_refs(observation, residue_id, source_link) != (
                    local,
                    partner,
                ):
                    raise ValueError("the cap's original covalent partner changed")
                _check_order(source_link, 1)

        if any(
            endpoint.atom_ref == partner
            for connection in excluded
            for endpoint in connection.endpoint_pair()
        ):
            raise ValueError("the cap partner has excluded coordination evidence")

        existing, local_heavy, local_hydrogen = _cap_inventory(structure, residue)
        if original is not None and observation is not None:
            _cap_inventory(observation, original)
        heavy_updates = (*local_heavy, *((realized,) if realized != link else ()))
        return LinkedPeptideCap(
            residue_id,
            residue.component_id,
            partner=partner,
            heavy_bond_updates=heavy_updates,
            hydrogen_bond_updates=local_hydrogen,
            missing_hydrogen_parents=_missing_hydrogen_parents(residue, dict(existing)),
            existing_hydrogen_parents=existing,
        )
    except ValueError as error:
        return LinkedPeptideCap(residue_id, residue.component_id, refusal=str(error))


def require_source_cap_atoms(
    structure: ProteinStructure, residue: ResidueSite, partner: AtomRef
) -> None:
    """Require original cap identity and charge custody before preparation."""
    observation = structure.provenance.ingress.observation
    if observation is None:
        return
    original = observation.constitution.residue_or_ligand(residue.residue_id)
    if original is None:
        return
    if original.component_id != residue.component_id:
        raise ValueError("the current cap does not match its original component")
    _check_source_atoms(structure, observation, original, partner)


def _external_bonds(
    snapshot: _Snapshot, residue_id: ResidueId
) -> tuple[TopologyBond, ...]:
    return tuple(
        bond
        for bond in snapshot.topology.inter_residue_bonds_for_residue(
            constitution=snapshot.constitution,
            residue_index=snapshot.constitution.residue_index(residue_id),
        )
        if bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
    )


def _external_refs(
    snapshot: _Snapshot, residue_id: ResidueId, bond: TopologyBond
) -> tuple[AtomRef, AtomRef]:
    first = snapshot.constitution.atom_ref_at(bond.atom_index_1)
    second = snapshot.constitution.atom_ref_at(bond.atom_index_2)
    return (first, second) if first.residue_id == residue_id else (second, first)


def _checked_link(bond: TopologyBond, *, qualified_pdb_link: bool) -> TopologyBond:
    metadata = bond.source_metadata
    if qualified_pdb_link and bond.relationship_type is BondRelationshipType.UNKNOWN:
        _check_order(bond, 1)
        return bond.with_resolved_chemistry(TopologyBond(*bond.endpoint_pair()))
    if bond.relationship_type is not BondRelationshipType.COVALENT:
        raise ValueError("the cap's external edge is not a typed covalent link")
    _check_order(bond, 1)
    if bond.order == 1:
        return bond
    if (
        metadata is None
        or metadata.record_type not in _TYPED_LINKS
        or metadata.reported_relationship_type is not BondRelationshipType.COVALENT
    ):
        raise ValueError("the cap's orderless edge has no typed source declaration")
    return bond.with_resolved_chemistry(TopologyBond(*bond.endpoint_pair()))


def _check_order(bond: TopologyBond, order: int) -> None:
    metadata = bond.source_metadata
    if (
        bond.relationship_type
        not in (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
        or bond.order not in (None, order)
        or bond.aromatic
        or (
            metadata is not None
            and (
                metadata.reported_order not in (None, order)
                or metadata.reported_relationship_type
                not in (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
            )
        )
    ):
        raise ValueError("a cap bond conflicts with the expected order or relationship")


def _check_partner(
    structure: ProteinStructure,
    partner: AtomRef,
    residue: ResidueSite,
    link: TopologyBond,
) -> None:
    constitution = structure.constitution
    topology = structure.topology
    index = constitution.atom_index(partner)
    if constitution.atom_site_at(
        index
    ).element != partner.atom_name or topology.formal_charge(index) not in (None, 0):
        raise ValueError("the cap's polymer port has a conflicting element or charge")
    incident = topology.incident_bonds_by_atom().get(index, ())
    if any(
        bond != link
        and any(
            constitution.atom_ref_at(other).residue_id != partner.residue_id
            for other in bond.endpoint_pair()
        )
        and bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
        for bond in incident
    ):
        raise ValueError("the cap's polymer port has another partner")
    if partner.atom_name == "C":
        neighbors = {}
        for bond in incident:
            if (
                bond == link
                or bond.relationship_type is BondRelationshipType.HYDROGEN_BOND
            ):
                continue
            other = (
                bond.atom_index_2 if index == bond.atom_index_1 else bond.atom_index_1
            )
            atom = constitution.atom_site_at(other)
            if (
                bond.relationship_type is not BondRelationshipType.COVALENT
                or bond.aromatic
                or topology.formal_charge(other) not in (None, 0)
            ):
                raise ValueError(
                    "the polymer carbonyl has an incompatible bond or charge"
                )
            neighbors[atom.name] = (atom.element, bond.order)
        if residue.has_atom_site("OXT") or neighbors != {"CA": ("C", 1), "O": ("O", 2)}:
            raise ValueError(
                "NME requires the closed polymer C=O scaffold, without OXT"
            )


def _check_source_atoms(
    structure: ProteinStructure,
    observation: StructureObservation,
    original: ResidueSite,
    partner: AtomRef,
) -> None:
    original_partner = observation.constitution.residue_or_ligand(partner.residue_id)
    current_partner = structure.constitution.residue_or_ligand(partner.residue_id)
    if (
        original_partner is not None
        and current_partner is not None
        and original_partner.component_id != current_partner.component_id
    ):
        raise ValueError("the cap's original polymer partner changed component")
    refs = [AtomRef(original.residue_id, atom.name) for atom in original.atom_sites]
    if original_partner is not None:
        refs.extend(
            AtomRef(partner.residue_id, name)
            for name in (("C", "O", "CA") if partner.atom_name == "C" else ("N",))
            if original_partner.has_atom_site(name)
        )
        for bond in observation.bonds_for_residue(partner.residue_id):
            ends = tuple(
                observation.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
            )
            if (
                partner in ends
                and bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
            ):
                other = next(ref for ref in ends if ref != partner)
                if other.residue_id == original.residue_id:
                    continue
                if other.residue_id != partner.residue_id:
                    raise ValueError("the polymer port has another original partner")
                if partner.atom_name == "C":
                    if other.atom_name not in ("O", "CA"):
                        raise ValueError(
                            "the original carbonyl has an incompatible partner"
                        )
                    _check_order(bond, 2 if other.atom_name == "O" else 1)
        if partner.atom_name == "C" and original_partner.has_atom_site("OXT"):
            raise ValueError("the attached carbonyl originally carried OXT")

    for ref in refs:
        original_index = observation.constitution.atom_index(ref)
        current_index = structure.constitution.resolve_atom_index(ref)
        charge = observation.topology.formal_charge(original_index)
        if (
            current_index is None
            or observation.constitution.atom_site_at(original_index)
            != structure.constitution.atom_site_at(current_index)
            or observation.geometry.atom_geometry(original_index).altloc
            != structure.geometry.atom_geometry(current_index).altloc
        ):
            raise ValueError(
                "the cap or its polymer port lost an original atom identity"
            )
        if charge not in (None, 0):
            raise ValueError(
                "the cap or its polymer port has an incompatible source charge"
            )


def _cap_inventory(
    snapshot: _Snapshot, residue: ResidueSite
) -> tuple[
    tuple[tuple[str, str], ...], tuple[TopologyBond, ...], tuple[TopologyBond, ...]
]:
    is_nme = residue.component_id == "NME"
    methyl = "C" if is_nme else "CH3"
    scaffold = {"C": "C", "N": "N"} if is_nme else {"C": "C", "CH3": "C", "O": "O"}
    if {
        atom.name: atom.element for atom in residue.atom_sites if not atom.is_hydrogen()
    } != scaffold:
        raise ValueError("the linked cap lacks its stock heavy scaffold")

    constitution = snapshot.constitution
    topology = snapshot.topology
    expected: dict[tuple[str, str], int] = (
        {("C", "N"): 1} if is_nme else {("C", "CH3"): 1, ("C", "O"): 2}
    )
    hydrogens: dict[str, str] = {}
    used_methyl_slots: set[str] = set()
    for atom in residue.atom_sites:
        index = constitution.atom_index(AtomRef(residue.residue_id, atom.name))
        if topology.formal_charge(index) not in (None, 0):
            raise ValueError("the linked cap must be neutral")
        if not atom.is_hydrogen():
            continue
        slot = PolymerMicrostateSite.hydrogen_slot_name(atom)
        if slot in _METHYL_H and slot not in used_methyl_slots:
            parent = methyl
            used_methyl_slots.add(slot)
        elif is_nme and slot in _AMIDE_H and "N" not in hydrogens.values():
            parent = "N"
        else:
            raise ValueError("the linked cap carries excess or ambiguously named H/D/T")
        hydrogens[atom.name] = parent
        first, second = sorted((atom.name, parent))
        expected[first, second] = 1

    for index in constitution.atom_indices_for_residue(residue.residue_id):
        for bond in topology.incident_bonds_by_atom().get(index, ()):
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            refs = tuple(
                constitution.atom_ref_at(item) for item in bond.endpoint_pair()
            )
            if any(ref.residue_id != residue.residue_id for ref in refs):
                continue
            first, second = sorted(ref.atom_name for ref in refs)
            if (first, second) not in expected:
                raise ValueError("the linked cap carries an incompatible local bond")
            _check_order(bond, expected[first, second])

    heavy_updates: list[TopologyBond] = []
    hydrogen_updates: list[TopologyBond] = []
    for (first, second), order in expected.items():
        indices = (
            constitution.atom_index(AtomRef(residue.residue_id, first)),
            constitution.atom_index(AtomRef(residue.residue_id, second)),
        )
        wanted = TopologyBond(*indices, order=order)
        bond = topology.bond_between(*indices)
        if bond is not None:
            wanted = bond.with_resolved_chemistry(wanted)
        if wanted != bond:
            updates = (
                hydrogen_updates
                if first in hydrogens or second in hydrogens
                else heavy_updates
            )
            updates.append(wanted)
    return tuple(hydrogens.items()), tuple(heavy_updates), tuple(hydrogen_updates)


def _missing_hydrogen_parents(
    residue: ResidueSite, hydrogens: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    is_nme = residue.component_id == "NME"
    methyl = "C" if is_nme else "CH3"
    taken = {
        PolymerMicrostateSite.hydrogen_slot_name(atom)
        for atom in residue.atom_sites
        if hydrogens.get(atom.name) == methyl
    }
    missing = tuple((name, methyl) for name in _METHYL_H if name not in taken)
    if is_nme and "N" not in hydrogens.values():
        return (*missing, ("HN2", "N"))
    return missing
