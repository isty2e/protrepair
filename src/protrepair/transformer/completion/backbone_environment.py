"""Connectivity and steric context that do not consult a backbone placeholder."""

from collections import defaultdict
from collections.abc import Iterator, Mapping
from itertools import chain, pairwise, product
from math import cos, floor, isfinite, radians
from types import MappingProxyType

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.polar_contacts import PolarContactChemistry
from protrepair.chemistry.radii import (
    covalent_radius_angstrom,
    van_der_waals_radius_angstrom,
)
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.peptide import has_verified_backbone_link
from protrepair.structure.topology import BondRelationshipType

_XH_LENGTH = MappingProxyType({"C": 1.09, "N": 1.01, "O": 0.98, "S": 1.34})


def _fits_xh_length(element: str, distance: float) -> bool:
    target = _XH_LENGTH.get(element)
    return target is not None and target - 0.25 <= distance <= target + 0.12


class BackboneEnvironment:
    """Verified peptide neighbors and retained atoms used for contact checks."""

    def __init__(self, structure: ProteinStructure, library: ComponentLibrary) -> None:
        self.structure = structure
        c = structure.constitution
        self.left: dict[ResidueId, ResidueSite] = {}
        self.right: dict[ResidueId, ResidueSite] = {}
        self.cells: dict[tuple[int, ...], list[tuple[AtomRef, Vec3]]] = defaultdict(
            list
        )
        self.radii: dict[AtomRef, float] = {}
        self.hydrogens: set[AtomRef] = set()
        self._positive: dict[AtomRef, Vec3] = {}
        self._elements: dict[AtomRef, str] = {}
        self._explicit_h: set[AtomRef] = set()
        self._heavy_neighbors: dict[AtomRef, set[AtomRef]] = defaultdict(set)
        self._junctions: dict[AtomRef, AtomRef] = {}
        self.protected: set[AtomRef] = set()
        ingress = structure.provenance.ingress
        obs = ingress.observation
        if obs is not None:
            self.protected.update(obs.unresolved_polymer_endpoints)
            self.protected.update(obs.unresolved_conect_ports)
            self.protected.update(
                ep.atom_ref
                for edge in obs.excluded_coordination_connections
                for ep in edge.endpoint_pair()
            )
        self.protected.update(ingress.uncertain_peptide_link_ports or ())
        self.protected.update(ingress.backbone_link_vetoes)
        for edge in structure.topology.bonds:
            a, b = (c.atom_ref_at(i) for i in edge.endpoint_pair())
            if edge.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            if edge.relationship_type is not BondRelationshipType.COVALENT or (
                a.residue_id != b.residue_id
                and {a.atom_name, b.atom_name} != {"C", "N"}
            ):
                self.protected.update((a, b))
                continue
            if a.residue_id == b.residue_id:
                hs = [
                    ref
                    for ref in (a, b)
                    if c.atom_site_at(c.atom_index(ref)).is_hydrogen()
                ]
                if hs:
                    self._explicit_h.update(hs)
                else:
                    self._heavy_neighbors[a].add(b)
                    self._heavy_neighbors[b].add(a)
        for polymer in c.chains:
            for before, after in pairwise(polymer.residues):
                carbon, nitrogen = (
                    AtomRef(before.residue_id, "C"),
                    AtomRef(after.residue_id, "N"),
                )
                if not has_verified_backbone_link(structure, before, after):
                    continue
                self._heavy_neighbors[carbon].add(nitrogen)
                self._heavy_neighbors[nitrogen].add(carbon)
                self._junctions[carbon] = nitrogen
                self._junctions[nitrogen] = carbon
                self.left[after.residue_id] = before
                self.right[before.residue_id] = after
        for site in c.residue_slots:
            for atom in site.atom_sites:
                ref = AtomRef(site.residue_id, atom.name)
                g = structure.geometry.atom_geometry(c.atom_index(ref))
                self._elements[ref] = atom.element
                self.radii[ref] = van_der_waals_radius_angstrom(
                    "H" if atom.is_hydrogen() else atom.element
                )
                if atom.is_hydrogen():
                    self.hydrogens.add(ref)
                if (
                    g.occupancy > 0
                    and isfinite(g.occupancy)
                    and all(isfinite(x) for x in g.position)
                ):
                    self.cells[self.cell(g.position)].append((ref, g.position))
                    self._positive[ref] = g.position

        self._chemistry = (
            PolarContactChemistry(c, structure.topology, library)
            if self.hydrogens
            else None
        )
        self._library = library

    @staticmethod
    def cell(point: Vec3) -> tuple[int, ...]:
        return tuple(floor(x / 6.0) for x in point)

    def hydrogen_parent(self, ref: AtomRef) -> AtomRef | None:
        """Return the unique local parent consistent with the selected H identity."""
        if self._chemistry is None:
            return None
        parent = self._chemistry.hydrogen_anchors(ref.residue_id).get(ref.atom_name)
        if parent is None:
            return None
        site = self.structure.constitution.residue_site_at(
            self.structure.constitution.residue_index(ref.residue_id)
        )
        template = self._library.get(site.component_id)
        if template is not None and not site.is_hetero:
            slot = PolymerMicrostateSite.hydrogen_slot_name(
                site.atom_site(ref.atom_name)
            )
            named = template.polymer_hydrogen_parent_aliases().get(slot)
            if named is not None and named != parent:
                return None
        return AtomRef(ref.residue_id, parent)

    def _nearby(self, point: Vec3) -> Iterator[tuple[AtomRef, Vec3]]:
        cell = self.cell(point)
        for delta in product((-1, 0, 1), repeat=3):
            key = tuple(a + b for a, b in zip(cell, delta, strict=True))
            yield from self.cells.get(key, ())

    def _local_heavy_atoms(self, ref: AtomRef) -> set[AtomRef]:
        seen = {ref}
        frontier = {ref}
        for n in range(3):
            frontier = {
                k for j in frontier for k in self._heavy_neighbors.get(j, ())
            } - seen
            seen.update(k for k in frontier if n < 2 or k.residue_id != ref.residue_id)
        return seen

    def fits_hydrogen_parent(
        self,
        h: AtomRef,
        point: Vec3,
        parent: AtomRef,
        center: Vec3,
        proposed: Mapping[AtomRef, Vec3],
    ) -> bool:
        """Qualify a known parent on the posed coordinates; never assign one by distance."""
        element = self._elements[parent]
        if self._chemistry is None or h in self.protected or parent in self.protected:
            return False
        if not _fits_xh_length(element, point.distance_to(center)):
            return False

        s = self.structure
        hi, xi = (s.constitution.atom_index(k) for k in (h, parent))
        ha, xa = (s.geometry.atom_geometry(i).altloc for i in (hi, xi))
        if (ha and xa and ha != xa) or s.topology.formal_charge(hi) not in (None, 0):
            return False

        charge = s.topology.formal_charge(xi) or 0
        maximum = {
            "C": 4 - abs(charge),
            "N": 3 + charge,
            "O": 2 + charge,
            "S": 2 + charge,
        }[element]
        valence = self._chemistry.bond_order_sum(parent)
        if abs(charge) > 1 or valence is None:
            return False
        junction = self._junctions.get(parent)
        if (
            junction is not None
            and s.topology.bond_between(xi, s.constitution.atom_index(junction)) is None
        ):
            valence += 1
        if valence > maximum:
            return False
        if parent.atom_name == "N" and junction is not None:
            site = s.constitution.residue_site_at(
                s.constitution.residue_index(parent.residue_id)
            )
            parents = self._chemistry.hydrogen_anchors(parent.residue_id)
            if (
                site.component_id == "PRO"
                or charge
                or tuple(parents.values()).count("N") != 1
            ):
                return False

        if h in self._explicit_h:
            return True
        # A second plausible heavy parent vetoes a projection. Closeness alone
        # never creates an attachment, even for an otherwise unassigned H.
        return not any(
            atom != parent
            and atom not in self.hydrogens
            and _fits_xh_length(self._elements[atom], point.distance_to(pos))
            for atom, pos in chain(self._nearby(point), proposed.items())
        )

    def _hydrogen_exemption(
        self,
        h: AtomRef,
        point: Vec3,
        other: AtomRef,
        q: Vec3,
        proposed: Mapping[AtomRef, Vec3],
    ) -> bool:
        parent = self.hydrogen_parent(h)
        if parent is None or self._chemistry is None:
            return False
        if other == parent:
            return self.fits_hydrogen_parent(h, point, parent, q, proposed)
        other_is_h = other in self.hydrogens
        if other_is_h:
            if self.hydrogen_parent(other) != parent:
                return False
        elif (
            self._junctions.get(parent) != other
            and self._chemistry.bond_hop_distance(parent, other) != 1
        ):
            return False

        center = proposed.get(parent, self._positive.get(parent))
        if center is None or not self.fits_hydrogen_parent(
            h, point, parent, center, proposed
        ):
            return False
        a = center.distance_to(q)
        if other_is_h:
            if not self.fits_hydrogen_parent(other, q, parent, center, proposed):
                return False
        else:
            if self._elements[other] not in _XH_LENGTH:
                return False
            expected = covalent_radius_angstrom(
                self._elements[other]
            ) + covalent_radius_angstrom(self._elements[parent])
            if abs(a - expected) > 0.25:
                return False

        b = center.distance_to(point)
        cosine = (a * a + b * b - point.distance_to(q) ** 2) / (2 * a * b)
        angle = (
            120.0
            if parent.atom_name == "N" and parent in self._junctions
            else self._chemistry.hydrogen_angle_reference(parent)
        )
        return angle is not None and cos(radians(angle + 15)) <= cosine <= cos(
            radians(angle - 15)
        )

    def clash(
        self, a: AtomRef, p: Vec3, b: AtomRef, q: Vec3, proposed: Mapping[AtomRef, Vec3]
    ) -> bool:
        """Test a penetration using the candidate coordinates for bonded geometry."""
        has_h = a in self.hydrogens or b in self.hydrogens
        if a == b or p.distance_to(q) >= self.radii[a] + self.radii[b] - (
            0.90 if has_h else 0.60
        ):
            return False
        if not has_h:
            return b not in self._local_heavy_atoms(a)
        if a in self.hydrogens:
            return not self._hydrogen_exemption(a, p, b, q, proposed)
        return not self._hydrogen_exemption(b, q, a, p, proposed)

    def check_contacts(
        self, ref: AtomRef, point: Vec3, proposed: Mapping[AtomRef, Vec3]
    ) -> None:
        for other, q in self._nearby(point):
            if self.clash(ref, point, other, q, proposed):
                raise GeometryPlacementError(
                    "steric clash with " + other.display_token()
                )
