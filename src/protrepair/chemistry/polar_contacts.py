"""Conservative polar-contact eligibility over observed and template chemistry."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.topology import (
    polymer_context_hydrogen_anchor_definitions,
)
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondRelationshipType, StructureTopology


@dataclass(frozen=True, slots=True)
class _Bond:
    order: int
    aromatic: bool = False


class PolarContactChemistry:
    """Interpret polar contacts without changing bonds or selecting protonation.

    Parameters
    ----------
    constitution : StructureConstitution
        Atom identities, including all existing H isotopes.
    topology : StructureTopology
        Authoritative relationships and formal charges for those identities.
    component_library : ComponentLibrary
        Supported bonds and named H attachments filling missing information,
        including exact template matches to untyped CONECT records.

    Raises
    ------
    ValueError
        If topology does not address the supplied constitution.

    Notes
    -----
    This coordinate-independent observation supports conventional N/O/S donors
    and conservative acceptor motifs, not universal H-bond detection. Unknown
    chemistry supplies no exemption. Known source relationships and bond orders
    take precedence; template projections never enter topology.
    Reuse is bound to the exact constitution, topology and component library;
    coordinate-only changes do not invalidate this interpretation.
    """

    def __init__(
        self,
        constitution: StructureConstitution,
        topology: StructureTopology,
        component_library: ComponentLibrary,
    ) -> None:
        if not topology.is_aligned_to(constitution):
            raise ValueError("polar-contact topology must align with constitution")

        self._constitution = constitution
        self._topology = topology
        self._component_library = component_library

        refs = tuple(
            constitution.atom_ref_at(AtomIndex(i))
            for i in range(len(constitution.atom_slots))
        )
        self._atoms = dict(zip(refs, constitution.atom_slots, strict=True))
        self._charges = {
            ref: topology.formal_charge(AtomIndex(i)) for i, ref in enumerate(refs)
        }
        refs_by_residue: dict[ResidueId, dict[str, AtomRef]] = {
            residue.residue_id: {} for residue in constitution.residue_slots
        }
        for ref in refs:
            refs_by_residue[ref.residue_id][ref.atom_name] = ref
        neighbors: dict[AtomRef, dict[AtomRef, _Bond]] = {ref: {} for ref in refs}
        explicit_pairs: set[frozenset[AtomRef]] = set()
        unresolved_pairs: dict[frozenset[AtomRef], int | None] = {}
        explicit_hydrogens: set[AtomRef] = set()
        unavailable: set[AtomRef] = set()

        def connect(left: AtomRef, right: AtomRef, bond: _Bond) -> None:
            neighbors[left][right] = bond
            neighbors[right][left] = bond

        for bond in topology.bonds:
            left, right = (refs[i.value] for i in bond.endpoint_pair())
            explicit_pairs.add(frozenset((left, right)))
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            explicit_hydrogens.update(
                ref for ref in (left, right) if self._atoms[ref].is_hydrogen()
            )
            if bond.relationship_type is BondRelationshipType.COVALENT:
                connect(left, right, _Bond(bond.order or 1, bond.aromatic))
                if bond.order is None:
                    unresolved_pairs[frozenset((left, right))] = None
            elif bond.relationship_type is BondRelationshipType.UNKNOWN:
                unresolved_pairs[frozenset((left, right))] = bond.order
            else:
                unavailable.update((left, right))

        def project(left: AtomRef, right: AtomRef, bond: _Bond) -> None:
            if left not in neighbors or right not in neighbors:
                return
            pair = frozenset((left, right))
            if pair in explicit_pairs:
                if pair not in unresolved_pairs:
                    return
                order = unresolved_pairs[pair]
                if order is not None and order != bond.order:
                    return
                del unresolved_pairs[pair]
            elif left in explicit_hydrogens or right in explicit_hydrogens:
                return
            connect(left, right, bond)

        for residue in constitution.residue_slots:
            template = component_library.get(residue.component_id)
            if template is None:
                continue
            named_refs = refs_by_residue[residue.residue_id]
            names = residue.atom_site_names()
            for name, charge in template.definition.formal_charges.items():
                ref = named_refs.get(name)
                if ref is not None and self._charges[ref] is None:
                    self._charges[ref] = charge
            for bond in template.definition.bonds:
                left = named_refs.get(bond.atom_name_1)
                right = named_refs.get(bond.atom_name_2)
                if left is not None and right is not None:
                    project(left, right, _Bond(bond.order, bond.aromatic))

            anchors = dict(template.template_hydrogen_anchor_by_name(names))
            if not residue.is_hetero:
                # PDB methylene names may be HB2/HB3 rather than HB1/HB2.
                # Only an unambiguous family in the supported template supplies
                # a parent; branch-ambiguous families such as THR HG do not.
                families: dict[str, set[str]] = {}
                aliases: dict[str, set[str]] = {}
                for name, parent in template.template_hydrogen_anchor_by_name(
                    template.expected_hydrogen_atom_names()
                ).items():
                    families.setdefault(name.strip("0123456789"), set()).add(parent)
                    aliases.setdefault(name, set()).add(parent)
                    if name[0].isdigit():
                        aliases.setdefault(name[1:] + name[0], set()).add(parent)
                for site in residue.atom_sites:
                    if not site.is_hydrogen() or site.name in anchors:
                        continue
                    name = site.name
                    if site.element in {"D", "T"}:
                        name = name.replace(site.element, "H", 1)
                    parents = aliases.get(name)
                    if parents is None:
                        parents = families.get(name.strip("0123456789"), set())
                    if len(parents) == 1:
                        anchors[site.name] = next(iter(parents))
                for anchor in polymer_context_hydrogen_anchor_definitions(
                    component_id=residue.component_id, hydrogen_atom_names=names
                ):
                    bond = anchor.bond_definition
                    anchors[bond.atom_name_2] = bond.atom_name_1
            for name, parent in anchors.items():
                hydrogen = named_refs.get(name)
                heavy = named_refs.get(parent)
                if (
                    hydrogen is not None
                    and heavy is not None
                    and self._atoms[hydrogen].is_hydrogen()
                ):
                    project(hydrogen, heavy, _Bond(1))

        for unresolved in unresolved_pairs:
            unavailable.update(unresolved)

        self._neighbors = neighbors
        self._unavailable = frozenset(unavailable)
        self._anchors: dict[ResidueId, Mapping[str, str]] = {}
        self._acceptors: dict[ResidueId, frozenset[str]] = {}
        self._hop_distances: dict[tuple[AtomRef, AtomRef], int | None] = {}
        for residue in constitution.residue_slots:
            named_refs = refs_by_residue[residue.residue_id]
            anchors = {}
            for site in residue.atom_sites:
                ref = named_refs[site.name]
                adjacent = neighbors[ref]
                if site.is_hydrogen() and ref not in unavailable and len(adjacent) == 1:
                    parent, bond = next(iter(adjacent.items()))
                    if (
                        bond.order == 1
                        and not bond.aromatic
                        and not self._atoms[parent].is_hydrogen()
                        and parent.residue_id == ref.residue_id
                    ):
                        anchors[site.name] = parent.atom_name
            self._anchors[residue.residue_id] = MappingProxyType(anchors)
            self._acceptors[residue.residue_id] = frozenset(
                site.name
                for site in residue.atom_sites
                if self._is_acceptor(named_refs[site.name])
            )

    def hydrogen_anchors(self, residue_id: ResidueId) -> Mapping[str, str]:
        """Return supported residue-local H parents, independent of coordinates.

        Parameters
        ----------
        residue_id : ResidueId
            Residue in this observation.

        Returns
        -------
        Mapping[str, str]
            H name to heavy parent; ambiguous and cross-residue H are omitted.
        """
        return self._anchors[residue_id]

    def acceptor_names(self, residue_id: ResidueId) -> frozenset[str]:
        """Return atoms supported as acceptors by the current chemistry.

        Parameters
        ----------
        residue_id : ResidueId
            Residue in this observation.

        Returns
        -------
        frozenset[str]
            Supported names, not a claim that omitted atoms cannot accept H bonds.
        """
        return self._acceptors[residue_id]

    def supports_donor(self, atom_ref: AtomRef) -> bool:
        """Check a heavy donor, including when its H is still being placed.

        Parameters
        ----------
        atom_ref : AtomRef
            Heavy atom in this observation.

        Returns
        -------
        bool
            Whether a conventional N/O/S donor is supported. The caller must
            establish attachment to H separately.
        """
        element = self._atoms[atom_ref].element
        charge = self._charges[atom_ref]
        if atom_ref in self._unavailable or element not in {"N", "O", "S"}:
            return False
        if charge not in {None, -1, 0, 1}:
            return False
        maximum_valence = 4 if element == "N" else 3 if charge == 1 else 2
        return (
            sum(b.order for b in self._neighbors[atom_ref].values()) <= maximum_valence
        )

    def matches_inputs(
        self,
        constitution: StructureConstitution,
        topology: StructureTopology,
        component_library: ComponentLibrary,
    ) -> bool:
        """Check the immutable inputs defining this chemical interpretation.

        Parameters
        ----------
        constitution : StructureConstitution
            Candidate atom identities.
        topology : StructureTopology
            Candidate relationships and formal charges.
        component_library : ComponentLibrary
            Candidate template definitions.

        Returns
        -------
        bool
            Whether all three inputs are the exact bound objects. Coordinates
            are deliberately absent; geometric contact tests must run separately.
        """
        return (
            constitution is self._constitution
            and topology is self._topology
            and component_library is self._component_library
        )

    def bond_hop_distance(self, left: AtomRef, right: AtomRef) -> int | None:
        """Return a residue-local covalent path length without crossing chains.

        Parameters
        ----------
        left, right : AtomRef
            Endpoints in the same residue of this observation.

        Returns
        -------
        int or None
            Shortest supported path, or None when no residue-local path exists.
        """
        if left.residue_id != right.residue_id:
            return None
        pair = (left, right)
        if pair in self._hop_distances:
            return self._hop_distances[pair]
        queue = deque([(left, 0)])
        visited = {left}
        while queue:
            current, distance = queue.popleft()
            if current == right:
                self._hop_distances[pair] = distance
                self._hop_distances[(right, left)] = distance
                return distance
            for other in self._neighbors[current]:
                if other not in visited and other.residue_id == left.residue_id:
                    visited.add(other)
                    queue.append((other, distance + 1))
        self._hop_distances[pair] = None
        self._hop_distances[(right, left)] = None
        return None

    def _is_acceptor(self, ref: AtomRef) -> bool:
        element = self._atoms[ref].element
        charge = self._charges[ref]
        if ref in self._unavailable or element not in {"N", "O", "S"}:
            return False
        if charge not in {None, -1, 0}:
            return False
        adjacent = self._neighbors[ref]
        valence = sum(bond.order for bond in adjacent.values())
        hydrogens = [other for other in adjacent if self._atoms[other].is_hydrogen()]
        heavy = [other for other in adjacent if not self._atoms[other].is_hydrogen()]
        if not heavy:
            # Explicit water is supported; a bare untyped O/N/S is not.
            return element == "O" and len(hydrogens) == valence == 2
        if element in {"O", "S"}:
            if charge == -1:
                return len(heavy) == 1 and not hydrogens and valence == 1
            if valence == 2:
                if element == "S" and (
                    hydrogens or any(self._atoms[n].element == "S" for n in heavy)
                ):
                    return False
                if hydrogens and self._resonance_center(heavy[0], ref):
                    return False
                return True
            # A neutral resonance oxygen can be encoded with a single bond when
            # the other carboxylate oxygen carries the observed negative charge.
            return (
                element == "O"
                and not hydrogens
                and valence == 1
                and any(
                    self._atoms[parent].element == "C"
                    and any(
                        peer != ref
                        and self._atoms[peer].element == "O"
                        and self._charges[peer] == -1
                        and peer not in self._unavailable
                        and len(self._neighbors[peer]) == 1
                        and self._neighbors[peer][parent].order == 1
                        and not self._neighbors[peer][parent].aromatic
                        for peer in self._neighbors[parent]
                    )
                    for parent in heavy
                )
            )
        if any(self._resonance_center(parent, ref) for parent in heavy):
            return False
        if any(bond.aromatic for bond in adjacent.values()):
            return len(adjacent) == 2 and not hydrogens
        if valence != 3:
            return False
        if any(bond.order > 1 for bond in adjacent.values()):
            return not hydrogens
        # Fully single-bonded N needs saturated neighbors: an incomplete pyrrole
        # or all-single imidazole template must not become an aliphatic amine.
        return all(
            self._atoms[parent].element == "C"
            and sum(b.order for b in self._neighbors[parent].values()) == 4
            and all(
                b.order == 1 and not b.aromatic
                for b in self._neighbors[parent].values()
            )
            for parent in heavy
        )

    def _resonance_center(self, center: AtomRef, excluded: AtomRef) -> bool:
        if self._atoms[center].element not in {"C", "S", "P"}:
            return False
        return any(
            peer != excluded
            and self._atoms[peer].element in {"N", "O", "S"}
            and bond.order > 1
            and not bond.aromatic
            for peer, bond in self._neighbors[center].items()
        )
