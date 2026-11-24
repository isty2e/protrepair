"""Peptide attachment atoms shared by cap placement and donor-span closure."""

from collections.abc import Iterator

from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.observation import StructureObservation
from protrepair.structure.topology import BondRelationshipType as Rel
from protrepair.structure.topology import TopologyBond


def neighbors(
    snapshot: ProteinStructure | StructureObservation, ref: AtomRef
) -> Iterator[tuple[AtomRef, TopologyBond]]:
    """Yield every non-hydrogen-bond neighbor in the original address space."""
    c, t = snapshot.constitution, snapshot.topology
    idx = c.resolve_atom_index(ref)
    if idx is not None:
        for bond in t.incident_bonds_by_atom().get(idx, ()):
            if bond.relationship_type is not Rel.HYDROGEN_BOND:
                a, b = bond.endpoint_pair()
                yield c.atom_ref_at(b if a == idx else a), bond


def amide_hydrogens(
    snapshot: ProteinStructure | StructureObservation,
    residue: ResidueSite,
    template: ResidueTemplate,
) -> tuple[str, ...]:
    """Corroborate local N-H/D/T names against connectivity and bond length."""
    rid = residue.residue_id
    c = snapshot.constitution
    aliases = template.polymer_hydrogen_parent_aliases()
    n_idx = c.resolve_atom_index(AtomRef(rid, "N"))
    names = []
    for atom in residue.atom_sites:
        if not atom.is_hydrogen():
            continue
        ref = AtomRef(rid, atom.name)
        declared = tuple(neighbors(snapshot, ref))
        named = aliases.get(PolymerMicrostateSite.hydrogen_slot_name(atom))
        needs_n = named == "N" or any(a == AtomRef(rid, "N") for a, _ in declared)
        pos = snapshot.geometry.position(c.atom_index(ref))
        near = n_idx is not None and source_hydrogen_bond_length_is_compatible(
            pos.distance_to(snapshot.geometry.position(n_idx)), 1.01
        )
        if needs_n and not declared:
            for other in residue.atom_sites:
                if other.is_hydrogen() or other.name == "N":
                    continue
                typical = {"C": 1.09, "N": 1.01, "O": 0.98, "S": 1.34}.get(
                    other.element
                )
                if typical is not None and source_hydrogen_bond_length_is_compatible(
                    pos.distance_to(
                        snapshot.geometry.position(
                            c.atom_index(AtomRef(rid, other.name))
                        )
                    ),
                    typical,
                ):
                    raise ValueError("ambiguous_n_h_parent")
        if not needs_n:
            if not declared and named is None and near:
                raise ValueError("unknown_h_parent")
            continue
        if (
            snapshot.topology.formal_charge(c.atom_index(ref)) not in (None, 0)
            or any(
                a != AtomRef(rid, "N")
                or b.relationship_type not in (Rel.COVALENT, Rel.UNKNOWN)
                or b.order not in (None, 1)
                or b.aromatic
                for a, b in declared
            )
            or (named is not None and named != "N")
            or not near
        ):
            raise ValueError("incompatible_n_h")
        names.append(atom.name)
    return tuple(names)


def check_unbonded_carbonyl_h(
    snapshot: ProteinStructure | StructureObservation, residue: ResidueSite
) -> None:
    """Refuse an unassigned isotope within bonding distance of carbonyl C or O."""
    c = snapshot.constitution
    rid = residue.residue_id
    for atom in residue.atom_sites:
        if not atom.is_hydrogen():
            continue
        ref = AtomRef(rid, atom.name)
        if any(neighbors(snapshot, ref)):
            continue
        point = snapshot.geometry.position(c.atom_index(ref))
        for name, length in (("C", 1.09), ("O", 0.98)):
            idx = c.resolve_atom_index(AtomRef(rid, name))
            if idx is not None and source_hydrogen_bond_length_is_compatible(
                point.distance_to(snapshot.geometry.position(idx)), length
            ):
                raise ValueError("possible_carbonyl_h")


def require_preserved_amide_hydrogens(
    parent: ProteinStructure,
    obs: StructureObservation,
    port: AtomRef,
    original_h: tuple[str, ...],
    hs: tuple[str, ...],
) -> None:
    """Require each original amide isotope and altloc on the current nitrogen."""
    c = parent.constitution
    for name in original_h:
        ref = AtomRef(port.residue_id, name)
        oi = obs.constitution.atom_index(ref)
        i = c.resolve_atom_index(ref)
        if (
            name not in hs
            or i is None
            or c.atom_site_at(i) != obs.constitution.atom_site_at(oi)
            or parent.geometry.atom_geometry(i).altloc
            != obs.geometry.atom_geometry(oi).altloc
        ):
            raise ValueError("lost_source_h")


def check_graft_port(
    snapshot: ProteinStructure | StructureObservation,
    port: AtomRef,
    template: ResidueTemplate,
) -> tuple[str, ...]:
    """Check local valence before adding one peptide bond, without changing the flank."""
    c, t = snapshot.constitution, snapshot.topology
    site = c.residue_or_ligand(port.residue_id)
    if site is None:
        raise ValueError("missing_flank")
    hs: tuple[str, ...] = ()
    heavy = set(template.expected_polymer_heavy_atom_names())
    if port.atom_name == "C":
        if site.has_atom_site("OXT") or site.has_atom_site("OT2"):
            raise ValueError("existing_terminal_oxygen")
        check_unbonded_carbonyl_h(snapshot, site)
        oxygen = AtomRef(port.residue_id, "O")
        if any(
            c.atom_site_at(c.atom_index(ref)).is_hydrogen()
            for ref, _ in neighbors(snapshot, oxygen)
        ):
            raise ValueError("protonated_carbonyl")
        if (oi := c.resolve_atom_index(oxygen)) is None or t.formal_charge(oi) not in (
            None,
            0,
        ):
            raise ValueError("incompatible_carbonyl")
        allowed = {"CA": 1, "O": 2}
    else:
        ns = template.definition.bonded_atom_names("N") & heavy
        if "CA" not in ns or len(ns) not in (1, 2):
            raise ValueError("unqualified_amide")
        hs = amide_hydrogens(snapshot, site, template)
        if len(hs) + len(ns) > 2:
            raise ValueError("surplus_terminal_h")
        allowed = dict.fromkeys((*ns, *hs), 1)

    idx = c.resolve_atom_index(port)
    if (
        idx is None
        or c.atom_site_at(idx).element != port.atom_name
        or t.formal_charge(idx) not in (None, 0)
    ):
        raise ValueError("incompatible_flank_element_or_charge")
    for ref, edge in neighbors(snapshot, port):
        if (
            ref.residue_id != port.residue_id
            or ref.atom_name not in allowed
            or edge.relationship_type not in (Rel.COVALENT, Rel.UNKNOWN)
            or edge.order not in (None, allowed[ref.atom_name])
            or edge.aromatic
        ):
            raise ValueError("occupied_or_incompatible_peptide_port")
    return hs
