"""Local polar-contact preferences for neutral histidine, not a pKa model."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, floor, radians

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolution,
)
from protrepair.chemistry.radii import (
    RadiusKind,
    prepare_radius_lookup,
)
from protrepair.geometry import Vec3
from protrepair.geometry.placement_vector import normalized_vector_or_none
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType

# OpenMM 8.4 Modeller uses 3.5 A and a donor-H-acceptor angle above 130 degrees.
# The lower H-A bound excludes compressed contacts, not plausible H bonds.
_DONOR_ACCEPTOR_LIMIT = 3.5
_HYDROGEN_ACCEPTOR_MINIMUM = 1.6
_HBOND_COSINE_LIMIT = cos(radians(130.0))
_GRID_SIZE = 5.0
_NH_LENGTH = 1.01


@dataclass(frozen=True, slots=True)
class _ContactAtom:
    ref: AtomRef
    position: Vec3
    acceptor: bool
    hydrogens: tuple[Vec3, ...]
    short_h_distance: float

    def donates_to(self, acceptor: Vec3, outward: Vec3) -> bool:
        direction = normalized_vector_or_none(outward.to_array() - acceptor.to_array())
        if direction is None:
            return False
        return any(
            _hydrogen_bond(self.position, h, acceptor)
            and (
                approach := normalized_vector_or_none(
                    h.to_array() - acceptor.to_array()
                )
            )
            is not None
            and float(direction @ approach) > -_HBOND_COSINE_LIMIT
            for h in self.hydrogens
        )


class HistidineEnvironment:
    """Index one immutable environment for conservative neutral-tautomer ranking.

    Parameters
    ----------
    context : PolymerMicrostateContext
        Current canonical bonds, coordinates, source evidence and saved choices.
    library : ComponentLibrary
        Active chemistry used to identify supported chemical sites.
    sidechain_resolutions : Mapping[ResidueId, MicrostateResolution]
        Simultaneous source/request/default resolutions before environmental
        ranking. Ambiguous sites cannot choose each other's tautomer.

    Notes
    -----
    Recognizes resolved standard-site acceptors, carbonyl/negative oxygen,
    water, and unprotonated imine/nitrile N.
    Unresolved neighboring HIS and untyped chemistry supply no positive support.
    Only existing, canonically attached N/O-H donors are used. No missing donor
    orientation, coupled network, solvent exposure, pKa or ring flip is inferred.
    """

    def __init__(
        self,
        context: PolymerMicrostateContext,
        library: ComponentLibrary,
        sidechain_resolutions: Mapping[ResidueId, MicrostateResolution],
    ) -> None:
        self._context = context
        self._cells: dict[tuple[int, int, int], list[_ContactAtom]] = defaultdict(list)
        source = context.source
        constitution = source.constitution
        unavailable = frozenset(
            constitution.atom_ref_at(index)
            for bond in source.topology.bonds
            if bond.relationship_type
            not in {BondRelationshipType.COVALENT, BondRelationshipType.HYDROGEN_BOND}
            for index in bond.endpoint_pair()
        )
        site_acceptors: dict[AtomRef, bool] = {}
        site_donors: set[AtomRef] = set()
        for residues in (
            *(chain.residues for chain in constitution.chains),
            constitution.ligands,
        ):
            for residue in residues:
                template = library.get(residue.component_id)
                resolution = sidechain_resolutions.get(residue.residue_id)
                if template is None or resolution is None:
                    continue
                site = PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN)
                catalog = site.candidates()
                if catalog is None:
                    continue
                graph = resolution.graph
                if graph is not None:
                    try:
                        context.hydrogen_attachments(
                            residue.residue_id, site, resolution
                        )
                    except ValueError:
                        graph = None
                for atom in catalog.candidates[0].atoms:
                    resolved = None if graph is None else graph.atom(atom.name)
                    site_acceptors[AtomRef(residue.residue_id, atom.name)] = (
                        resolved is not None
                        and resolved.element in {"N", "O"}
                        and resolved.hydrogens == 0
                        and resolved.charge <= 0
                    )
                    if resolved is not None and resolved.hydrogens > 0:
                        site_donors.add(AtomRef(residue.residue_id, atom.name))

        elements = {atom.element for atom in constitution.atom_slots}
        radii = prepare_radius_lookup(elements | {"H"}, RadiusKind.COVALENT)
        hydrogen_radius = radii.radius_angstrom("H")
        # Same near-covalent margin as parser-contact diagnostics. This
        # screens gross H crowding, not an FF energy or clash-free pose.
        short_distance_by_element = {
            element: hydrogen_radius + radius + 0.45
            for element, radius in radii.radius_by_element_symbol.items()
        }

        for index, position in source.geometry.iter_positions():
            ref = constitution.atom_ref_at(index)
            atom = constitution.atom_site_at(index)
            neighbors = context.covalent_neighbors(ref)
            donor_hydrogens = tuple(
                source.geometry.position(constitution.atom_index(other))
                for other in neighbors
                if atom.element in {"N", "O"}
                and ref not in unavailable
                and (ref not in site_acceptors or ref in site_donors)
                and constitution.atom_site_at(
                    constitution.atom_index(other)
                ).is_hydrogen()
            )
            contact = _ContactAtom(
                ref,
                position,
                ref not in unavailable and self._is_acceptor(ref, site_acceptors),
                donor_hydrogens,
                short_distance_by_element.get(atom.element, 0.0),
            )
            self._cells[_cell(position)].append(contact)

    def preference(self, residue_id: ResidueId) -> MicrostateConstraints | None:
        """Prefer a less crowded, better H-bonded neutral tautomer, or abstain.

        Parameters
        ----------
        residue_id : ResidueId
            A HIS with both neutral graphs admitted by the caller's resolver.

        Returns
        -------
        MicrostateConstraints or None
            Ring H counts only; no authority to override source chemistry.
            Tied scores or degenerate ring frames return None.

        Raises
        ------
        ResidueNotFoundError
            The requested residue is absent from this snapshot.
        """
        source = self._context.source
        geometry = source.residue_geometry(
            source.constitution.residue_index(residue_id)
        )
        positions = []
        for name, neighbors in (("ND1", ("CG", "CE1")), ("NE2", ("CE1", "CD2"))):
            if not all(geometry.has_atom(n) for n in (name, *neighbors)):
                return None
            nitrogen = geometry.position(name)
            vectors = tuple(
                normalized_vector_or_none(
                    nitrogen.to_array() - geometry.position(n).to_array()
                )
                for n in neighbors
            )
            if any(vector is None for vector in vectors):
                return None
            left, right = vectors
            assert left is not None and right is not None
            outward = normalized_vector_or_none(left + right)
            if outward is None:
                return None
            positions.append(
                (
                    nitrogen,
                    Vec3.from_iterable(nitrogen.to_array() + _NH_LENGTH * outward),
                )
            )

        scores = []
        for donor_index in (0, 1):
            donor, hydrogen = positions[donor_index]
            acceptor, outward = positions[1 - donor_index]
            donor_neighbors = self._neighbors(donor, residue_id)
            short_contacts = sum(
                hydrogen.distance_to(atom.position) < atom.short_h_distance
                for atom in donor_neighbors
            )
            donating = any(
                atom.acceptor and _hydrogen_bond(donor, hydrogen, atom.position)
                for atom in donor_neighbors
            )
            accepting = any(
                atom.donates_to(acceptor, outward)
                for atom in self._neighbors(acceptor, residue_id)
            )
            scores.append((short_contacts, -int(donating) - int(accepting)))
        if scores[0] == scores[1]:
            return None
        delta = scores[0] < scores[1]
        return MicrostateConstraints(
            hydrogens=(("ND1", int(delta)), ("NE2", int(not delta)))
        )

    def _neighbors(
        self, position: Vec3, residue_id: ResidueId
    ) -> tuple[_ContactAtom, ...]:
        x, y, z = _cell(position)
        return tuple(
            atom
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            for atom in self._cells.get((x + dx, y + dy, z + dz), ())
            if atom.ref.residue_id != residue_id
            and position.distance_to(atom.position) <= _GRID_SIZE
        )

    def _is_acceptor(self, ref: AtomRef, site_acceptors: dict[AtomRef, bool]) -> bool:
        source = self._context.source
        constitution = source.constitution
        index = constitution.atom_index(ref)
        atom = constitution.atom_site_at(index)
        if ref in site_acceptors:
            return site_acceptors[ref]
        charge = source.topology.formal_charge(index)
        if atom.element not in {"N", "O"} or (charge is not None and charge > 0):
            return False
        neighbors = self._context.covalent_neighbors(ref)
        if atom.element == "O":
            residue = constitution.residue_or_ligand(ref.residue_id)
            assert residue is not None
            if residue.component_id in {"HOH", "WAT", "DOD"} and charge in (None, 0):
                return True
            return (
                charge == -1
                and len(neighbors) == 1
                and len(self._context.covalent_neighbors(ref, order=1)) == 1
            ) or (
                len(neighbors) == 1
                and any(
                    constitution.atom_site_at(constitution.atom_index(other)).element
                    == "C"
                    for other in self._context.covalent_neighbors(ref, order=2)
                )
            )
        if any(
            constitution.atom_site_at(constitution.atom_index(other)).is_hydrogen()
            for other in neighbors
        ):
            return False
        multiple = (
            *self._context.covalent_neighbors(ref, order=2),
            *self._context.covalent_neighbors(ref, order=3),
        )
        valence = sum(
            order * len(self._context.covalent_neighbors(ref, order=order))
            for order in (1, 2, 3)
        )
        if not multiple or valence != 3:
            return False
        # Amide, amidine and guanidine N are not made acceptors by a resonance
        # encoding that happens to put a double bond on this nitrogen.
        for other in neighbors:
            for peer in self._context.covalent_neighbors(other):
                if peer != ref and constitution.atom_site_at(
                    constitution.atom_index(peer)
                ).element in {"N", "O", "S"}:
                    return False
        return charge in (None, 0)


def _cell(position: Vec3) -> tuple[int, int, int]:
    return (
        floor(position.x / _GRID_SIZE),
        floor(position.y / _GRID_SIZE),
        floor(position.z / _GRID_SIZE),
    )


def _hydrogen_bond(donor: Vec3, hydrogen: Vec3, acceptor: Vec3) -> bool:
    if donor.distance_to(acceptor) > _DONOR_ACCEPTOR_LIMIT:
        return False
    if hydrogen.distance_to(acceptor) < _HYDROGEN_ACCEPTOR_MINIMUM:
        return False
    left = normalized_vector_or_none(donor.to_array() - hydrogen.to_array())
    right = normalized_vector_or_none(acceptor.to_array() - hydrogen.to_array())
    return (
        left is not None
        and right is not None
        and float(left @ right) < _HBOND_COSINE_LIMIT
    )
