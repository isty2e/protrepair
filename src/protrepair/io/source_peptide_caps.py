"""Bind first-model PDB adjacency evidence to selected retained caps."""

from collections import defaultdict
from dataclasses import dataclass

from protrepair.structure.labels import ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.peptide import (
    PEPTIDE_CAP_COMPONENT_IDS,
    PEPTIDE_CN_DISTANCE_MAX_ANGSTROM,
)
from protrepair.structure.source_connection import (
    PdbAdjacentCapLink,
    SourceAtomIdentity,
    SourceConnection,
)
from protrepair.structure.topology import BondRelationshipType


@dataclass(frozen=True, slots=True)
class PdbCapSourceContext:
    """Raw path and ambiguity facts that would be lost during selection.

    Parameters
    ----------
    adjacent_slots : frozenset of tuple of ResidueId
        Directed neighbors in unique, uninterrupted first-model PDB slots.
    first_model_pairs : frozenset of tuple of SourceAtomIdentity
        CONECT pairs declared at file level or in the first MODEL.
    ambiguous_atoms : frozenset of SourceAtomIdentity
        Atom identities with multiple source occurrences or an incident
        CONECT whose other serial could not be bound.
    """

    adjacent_slots: frozenset[tuple[ResidueId, ResidueId]]
    first_model_pairs: frozenset[tuple[SourceAtomIdentity, SourceAtomIdentity]]
    ambiguous_atoms: frozenset[SourceAtomIdentity]

    def links_for(
        self,
        observation: StructureObservation,
        source_connections: tuple[SourceConnection, ...],
    ) -> tuple[PdbAdjacentCapLink, ...]:
        """Keep source-adjacent bare attachments of selected cap atoms only.

        Parameters
        ----------
        observation : StructureObservation
            Selected source facets, preserving reported bond attributes.
        source_connections : tuple of SourceConnection
            Decoded declarations before dropping unselected endpoints.

        Returns
        -------
        tuple of PdbAdjacentCapLink
            Exact cap attachments without competing source declarations.
        """
        constitution, geometry = observation.constitution, observation.geometry
        cap_ids = {
            residue.residue_id
            for residue in constitution.ligands
            if residue.component_id in PEPTIDE_CAP_COMPONENT_IDS
        }
        if not cap_ids:
            return ()

        incident: dict[ResidueId, list[SourceConnection]] = defaultdict(list)
        for connection in source_connections:
            left, right = (
                endpoint.atom_ref.residue_id for endpoint in connection.endpoint_pair()
            )
            if (
                left != right
                and connection.source_metadata.reported_relationship_type
                is not BondRelationshipType.HYDROGEN_BOND
            ):
                incident[left].append(connection)
                incident[right].append(connection)

        links: list[PdbAdjacentCapLink] = []
        for connection in source_connections:
            carbon, nitrogen = connection.endpoint_pair()
            carbon_id = carbon.atom_ref.residue_id
            nitrogen_id = nitrogen.atom_ref.residue_id
            if (
                not (carbon_id in cap_ids or nitrogen_id in cap_ids)
                or (carbon_id, nitrogen_id) not in self.adjacent_slots
                or (carbon, nitrogen) not in self.first_model_pairs
                or not connection.source_metadata.is_bare_pdb_conect()
            ):
                continue
            try:
                link = PdbAdjacentCapLink(carbon, nitrogen)
            except ValueError:
                continue

            bond = link.bond_in(constitution, geometry, observation.topology)
            if bond is None:
                continue
            first, second = bond.endpoint_pair()
            distance = geometry.position(first).distance_to(geometry.position(second))
            if not 0 < distance <= PEPTIDE_CN_DISTANCE_MAX_ANGSTROM:
                continue

            cap_id = link.cap_residue_id
            cap = constitution.residue_or_ligand(cap_id)
            assert cap is not None
            protected = {carbon, nitrogen}
            protected.update(
                SourceAtomIdentity(
                    constitution.atom_ref_at(index),
                    cap.component_id,
                    geometry.atom_geometry(index).altloc,
                )
                for index in constitution.atom_indices_for_residue(cap_id)
            )
            if self.ambiguous_atoms.intersection(protected):
                continue
            if any(
                other.endpoint_pair() != connection.endpoint_pair()
                and protected.intersection(other.endpoint_pair())
                for residue_id in (carbon_id, nitrogen_id)
                for other in incident[residue_id]
            ):
                continue
            links.append(link)
        return tuple(links)
