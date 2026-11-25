"""Connectivity and steric context that do not consult a backbone placeholder."""

from collections import defaultdict
from itertools import pairwise, product
from math import floor, isfinite

from protrepair.chemistry.radii import van_der_waals_radius_angstrom
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.peptide import has_verified_backbone_link
from protrepair.structure.topology import BondRelationshipType


class BackboneEnvironment:
    """Verified peptide neighbors and retained atoms used for contact checks."""

    def __init__(self, structure: ProteinStructure) -> None:
        self.structure = structure
        c = structure.constitution
        self.neighbors: dict[AtomRef, set[AtomRef]] = defaultdict(set)
        self.left: dict[ResidueId, ResidueSite] = {}
        self.right: dict[ResidueId, ResidueSite] = {}
        self.cells: dict[tuple[int, ...], list[tuple[AtomRef, Vec3]]] = defaultdict(
            list
        )
        self.radii: dict[AtomRef, float] = {}
        self.hydrogens: set[AtomRef] = set()
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
                self.connect(a, b)
        for chain in c.chains:
            for before, after in pairwise(chain.residues):
                carbon, nitrogen = (
                    AtomRef(before.residue_id, "C"),
                    AtomRef(after.residue_id, "N"),
                )
                if not has_verified_backbone_link(structure, before, after):
                    continue
                self.connect(carbon, nitrogen)
                self.left[after.residue_id] = before
                self.right[before.residue_id] = after
        for site in c.residue_slots:
            for atom in site.atom_sites:
                ref = AtomRef(site.residue_id, atom.name)
                g = structure.geometry.atom_geometry(c.atom_index(ref))
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

    @staticmethod
    def cell(point: Vec3) -> tuple[int, ...]:
        return tuple(floor(x / 6.0) for x in point)

    def connect(self, a: AtomRef, b: AtomRef) -> None:
        self.neighbors[a].add(b)
        self.neighbors[b].add(a)

    def local_atoms(self, ref: AtomRef) -> set[AtomRef]:
        # 1-2 and 1-3 within a residue; up to three hops across a peptide junction.
        seen = {ref}
        frontier = {ref}
        for n in range(3):
            frontier = {k for j in frontier for k in self.neighbors.get(j, ())} - seen
            seen.update(k for k in frontier if n < 2 or k.residue_id != ref.residue_id)
        return seen

    def clash(self, a: AtomRef, p: Vec3, b: AtomRef, q: Vec3) -> bool:
        tolerance = 0.90 if a in self.hydrogens or b in self.hydrogens else 0.60
        return p.distance_to(q) < self.radii[a] + self.radii[b] - tolerance

    def check_contacts(self, ref: AtomRef, point: Vec3) -> None:
        local = self.local_atoms(ref)
        cell = self.cell(point)
        for delta in product((-1, 0, 1), repeat=3):
            key = tuple(a + b for a, b in zip(cell, delta, strict=True))
            for other, q in self.cells.get(key, ()):
                if other not in local and self.clash(ref, point, other, q):
                    raise GeometryPlacementError(
                        "steric clash with " + other.display_token()
                    )
