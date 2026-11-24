"""Local pair enumeration in a full parent environment; water has its own guard."""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from itertools import combinations, product
from math import floor

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.peptide_caps import observe_linked_peptide_cap
from protrepair.diagnostics.clashes import (
    AtomSite,
    ClashPolicy,
    cap_junction_bond_hops,
    prepare_clash_detection_context,
    probable_hydrogen_bond,
    should_ignore_pair,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType as Rel
from protrepair.transformer.completion.terminal.cap_product import neighbors

POLICY = ClashPolicy(include_hydrogen_hydrogen=True, include_ligands=True)


class PeptideCapContacts:
    """Full parent environment with a spatial index for fixed contact partners."""

    def __init__(
        self,
        structure: ProteinStructure,
        port: AtomRef,
        cap_id: ResidueId,
        component_library: ComponentLibrary,
    ) -> None:
        self.port, self.cap_id = port, cap_id
        self.ctx = prepare_clash_detection_context(
            structure, component_library=component_library, policy=POLICY
        )
        self.focus = {
            s.atom_ref(): s
            for s in self.ctx.atom_sites
            if s.residue_id in (port.residue_id, cap_id)
        }
        self.fixed = tuple(
            s
            for s in self.ctx.atom_sites
            if s.residue_id not in (port.residue_id, cap_id)
        )
        self.reach = max(self.ctx.allowed_distance_by_element_pair.values())
        self.cells: dict[tuple[int, ...], list[AtomSite]] = defaultdict(list)
        for site in self.fixed:
            self.cells[self._cell(site)].append(site)
        self.peptides = set()
        self.occupied: dict[ResidueId, tuple[AtomRef, AtomRef]] = {}
        # Qualify actual C-N covalent junctions on this flank, including a previously placed cap.
        for name, other_name, compound in (("C", "N", "NME"), ("N", "C", "ACE")):
            end = AtomRef(port.residue_id, name)
            for other, b in neighbors(structure, end):
                if (
                    b.relationship_type is Rel.COVALENT
                    and b.order == 1
                    and other.atom_name == other_name
                    and other.residue_id != port.residue_id
                ):
                    site = structure.constitution.residue_or_ligand(other.residue_id)
                    if (
                        site is not None
                        and structure.constitution.role_at(
                            structure.constitution.residue_index(site.residue_id)
                        )
                        is ResidueRole.POLYMER
                    ):
                        self.peptides.add(
                            frozenset((port.residue_id, other.residue_id))
                        )
                    if site is not None and site.component_id == compound:
                        linked = observe_linked_peptide_cap(structure, site)
                        if (
                            linked is not None
                            and linked.partner == end
                            and linked.link_is_realized()
                        ):
                            self.occupied[other.residue_id] = (other, end)

    def _cell(self, site: AtomSite) -> tuple[int, ...]:
        return tuple(floor(coord / self.reach) for coord in site.geometry.position)

    def _junction_exempt(self, a: AtomSite, b: AtomSite) -> bool:
        if a.residue_id not in self.occupied:
            a, b = b, a
        entry = self.occupied.get(a.residue_id)
        if entry is None or b.residue_id != entry[1].residue_id:
            return False
        hops = cap_junction_bond_hops(a, b, entry)
        return hops is not None and hops <= POLICY.ignore_adjacent_polymer_bond_hops

    def clashes(
        self, structure: ProteinStructure
    ) -> Iterator[tuple[tuple[AtomRef, AtomRef], float, bool]]:
        """Yield local penetrating pairs, excluding qualified peptide-bond hops."""
        geoms = {
            r: structure.residue_geometry(structure.constitution.residue_index(r))
            for r in (self.port.residue_id, self.cap_id)
            if structure.constitution.residue_or_ligand(r) is not None
        }
        sites = [
            replace(
                s,
                geometry=geoms[ref.residue_id].atom_geometry(ref.atom_name),
                context=replace(s.context, residue_geometry=geoms[ref.residue_id]),
            )
            for ref, s in self.focus.items()
        ]

        def pairs() -> Iterator[tuple[AtomSite, AtomSite]]:
            yield from combinations(sites, 2)
            for a in sites:
                x, y, z = self._cell(a)
                for dx, dy, dz in product((-1, 0, 1), repeat=3):
                    for b in self.cells.get((x + dx, y + dy, z + dz), ()):
                        yield a, b

        for a, b in pairs():
            allowed = self.ctx.allowed_distance_by_element_pair[a.element, b.element]
            distance = a.geometry.distance_to(b.geometry)
            ordinary_bond = (
                a.residue_id == b.residue_id
                or frozenset((a.residue_id, b.residue_id)) in self.peptides
            )
            if (
                distance >= allowed
                or (ordinary_bond and should_ignore_pair(a, b, policy=POLICY))
                or self._junction_exempt(a, b)
                or probable_hydrogen_bond(a, b, distance)
            ):
                continue
            uncertain = any(
                s.is_hydrogen() and s.atom_name not in s.context.hydrogen_anchor_by_name
                for s in (a, b)
            )
            pair = tuple(sorted((a.atom_ref(), b.atom_ref())))
            yield (pair[0], pair[1]), allowed - distance, uncertain
