"""Identity of a peptide-cap attachment across retention and file projection."""

from collections.abc import Iterator
from dataclasses import dataclass

from protrepair.structure.constitution import ResidueRole, StructureConstitution
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.source_connection import PdbAdjacentCapLink
from protrepair.structure.topology import (
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)

PEPTIDE_CAP_COMPONENT_IDS = frozenset(("ACE", "NME"))


@dataclass(frozen=True, slots=True)
class PeptideCapAttachment:
    """A selected peptide port and the non-polymer residue attached to it.

    Parameters
    ----------
    port : AtomRef
        Polymer N (acetyl) or C (N-methylamide).
    component_id : str
        Polymer component, used to detect address reuse.
    cap_id : ResidueId
        Author address of the cap that occupied the port.
    """

    port: AtomRef
    component_id: str
    cap_id: ResidueId

    def __post_init__(self) -> None:
        if (
            self.port.atom_name not in ("N", "C")
            or self.port.residue_id == self.cap_id
            or not self.component_id.strip()
        ):
            raise ValueError("invalid peptide-cap attachment")

    def identifies_port_in(self, constitution: StructureConstitution) -> bool:
        """Whether the same polymer residue and port survive in this carrier."""
        residue = constitution.residue_or_ligand(self.port.residue_id)
        return (
            residue is not None
            and residue.component_id == self.component_id
            and constitution.role_at(constitution.residue_index(residue.residue_id))
            is ResidueRole.POLYMER
            and constitution.resolve_atom_index(self.port) is not None
        )

    @classmethod
    def bonded_ports(
        cls,
        constitution: StructureConstitution,
        geometry: StructureGeometry,
        topology: StructureTopology,
        *,
        adjacent: tuple[PdbAdjacentCapLink, ...] = (),
        uncertain: frozenset[AtomRef] = frozenset(),
        observation: StructureObservation | None = None,
    ) -> Iterator[tuple["PeptideCapAttachment", TopologyBond]]:
        """Find unique covalent cap junctions or independently qualified bare PDB pairs.

        This identifies an occupied port even when other atoms on its cap are missing.
        It does not certify completed cap chemistry.
        """
        if not any(
            site.component_id in PEPTIDE_CAP_COMPONENT_IDS
            for site in constitution.ligands
        ):
            return
        unresolved_caps: set[ResidueId] = set()
        if observation is not None:
            uncertain = (
                uncertain
                | observation.unresolved_polymer_endpoints
                | observation.unresolved_conect_ports
            )
            unresolved_caps.update(
                endpoint.atom_ref.residue_id
                for link in observation.unresolved_non_polymer_links
                for endpoint in link.endpoint_pair()
                if endpoint.component_id in PEPTIDE_CAP_COMPONENT_IDS
            )
        bare = {
            b.endpoint_pair()
            for link in adjacent
            if (b := link.bond_in(constitution, geometry, topology)) is not None
        }
        for cap in constitution.ligands:
            if (
                cap.component_id not in PEPTIDE_CAP_COMPONENT_IDS
                or cap.residue_id in unresolved_caps
            ):
                continue
            cap_name, port_name = (
                ("C", "N") if cap.component_id == "ACE" else ("N", "C")
            )
            edges = tuple(
                edge
                for i in constitution.atom_indices_for_residue(cap.residue_id)
                for edge in topology.incident_bonds_by_atom().get(i, ())
                if edge.relationship_type is not BondRelationshipType.HYDROGEN_BOND
                and any(
                    constitution.atom_ref_at(j).residue_id != cap.residue_id
                    for j in edge.endpoint_pair()
                )
            )
            if len(edges) != 1:
                continue
            edge = edges[0]
            pair = edge.endpoint_pair()
            if (
                edge.source_metadata is not None
                and edge.source_metadata.is_bare_pdb_conect()
                and pair not in bare
            ):
                continue
            if (
                edge.aromatic
                or edge.order not in (None, 1)
                or (
                    edge.relationship_type is not BondRelationshipType.COVALENT
                    and pair not in bare
                )
            ):
                continue
            cap_index, port_index = pair
            if constitution.atom_ref_at(cap_index).residue_id != cap.residue_id:
                cap_index, port_index = port_index, cap_index
            port = constitution.atom_ref_at(port_index)
            if (
                port in uncertain
                or constitution.atom_ref_at(cap_index) in uncertain
                or port.atom_name != port_name
                or constitution.atom_ref_at(cap_index).atom_name != cap_name
                or constitution.atom_site_at(cap_index).element != cap_name
                or constitution.atom_site_at(port_index).element != port_name
            ):
                continue
            residue = constitution.residue_site_at(
                constitution.residue_index(port.residue_id)
            )
            match = cls(port, residue.component_id, cap.residue_id)
            if match.identifies_port_in(constitution) and all(
                other is edge
                or other.relationship_type is BondRelationshipType.HYDROGEN_BOND
                or all(
                    constitution.atom_ref_at(j).residue_id == port.residue_id
                    for j in other.endpoint_pair()
                )
                for other in topology.incident_bonds_by_atom().get(port_index, ())
            ):
                yield match, edge
