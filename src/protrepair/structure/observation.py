"""Original structure observations, independent of the current repaired graph."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from protrepair.errors import ModelInvariantError
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


@dataclass(frozen=True, slots=True)
class StructureObservation:
    """Keep selected input facts in their original atom address space.

    Parameters
    ----------
    constitution : StructureConstitution
        Original selected atom identities, including observed H/D/T.
    geometry : StructureGeometry
        Original coordinates, retained even when current parents move.
    topology : StructureTopology
        Reported charges and source-explicit bonds only. Unreported orders are
        None and unreported relationships are UNKNOWN, not template-filled
        values. Missing H is not negative evidence.

    Raises
    ------
    ModelInvariantError
        Facets are misaligned or topology contains inferred chemistry.
    """

    constitution: StructureConstitution
    geometry: StructureGeometry
    topology: StructureTopology
    _bonds_by_residue: Mapping[ResidueId, tuple[TopologyBond, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.geometry.is_aligned_to(self.constitution):
            raise ModelInvariantError(
                "observation geometry must match its constitution"
            )
        if not self.topology.is_aligned_to(self.constitution):
            raise ModelInvariantError(
                "observation topology must match its constitution"
            )
        for bond in self.topology.bonds:
            reported_order = (
                None
                if bond.source_metadata is None
                else bond.source_metadata.reported_order
            )
            reported_relationship = (
                BondRelationshipType.UNKNOWN
                if bond.source_metadata is None
                else bond.source_metadata.reported_relationship_type
            )
            if (
                bond.provenance is not BondProvenance.SOURCE_EXPLICIT
                or bond.order != reported_order
                or bond.relationship_type is not reported_relationship
                or bond.aromatic
            ):
                raise ModelInvariantError(
                    "observation bonds must contain only source declarations"
                )

        by_residue: dict[ResidueId, list[TopologyBond]] = {}
        for bond in self.topology.bonds:
            first = self.constitution.atom_ref_at(bond.atom_index_1).residue_id
            second = self.constitution.atom_ref_at(bond.atom_index_2).residue_id
            by_residue.setdefault(first, []).append(bond)
            if second != first:
                by_residue.setdefault(second, []).append(bond)
        object.__setattr__(
            self,
            "_bonds_by_residue",
            MappingProxyType(
                {residue_id: tuple(bonds) for residue_id, bonds in by_residue.items()}
            ),
        )

    @classmethod
    def from_source_facets(
        cls,
        *,
        constitution: StructureConstitution,
        geometry: StructureGeometry,
        topology: StructureTopology,
    ) -> "StructureObservation":
        """Capture source facts before any chemistry-changing transformation.

        Parameters
        ----------
        constitution : StructureConstitution
            Selected source atom address space.
        geometry : StructureGeometry
            Geometry in that address space.
        topology : StructureTopology
            Ingress topology whose atom charges still describe input records.

        Returns
        -------
        StructureObservation
            Snapshot excluding model-resolved bonds and filled bond semantics.

        Raises
        ------
        ModelInvariantError
            The supplied facets do not share an address space.
        """
        if not topology.is_aligned_to(constitution):
            raise ModelInvariantError(
                "source topology must match observation constitution"
            )
        source_topology = StructureTopology(
            constitution=constitution,
            atom_topologies=topology.atom_topologies,
            bonds=tuple(
                replace(
                    bond,
                    order=(
                        None
                        if bond.source_metadata is None
                        else bond.source_metadata.reported_order
                    ),
                    aromatic=False,
                    relationship_type=(
                        BondRelationshipType.UNKNOWN
                        if bond.source_metadata is None
                        else bond.source_metadata.reported_relationship_type
                    ),
                )
                for bond in topology.bonds
                if bond.provenance is BondProvenance.SOURCE_EXPLICIT
            ),
        )
        return cls(
            constitution=constitution, geometry=geometry, topology=source_topology
        )

    def formal_charge(self, atom_ref: AtomRef) -> int | None:
        """Return a reported charge, preserving explicit neutral zero.

        Parameters
        ----------
        atom_ref : AtomRef
            Atom identity in the original input, not the current repaired graph.

        Returns
        -------
        int or None
            Reported charge, or None for an unobserved atom or unspecified charge.
            The constitution distinguishes those two absence cases.
        """
        atom_index = self.constitution.resolve_atom_index(atom_ref)
        return None if atom_index is None else self.topology.formal_charge(atom_index)

    def hydrogen_atoms(self, residue_id: ResidueId) -> tuple[AtomRef, ...]:
        """Return observed H/D/T identities for one original residue.

        Parameters
        ----------
        residue_id : ResidueId
            Residue identity in the original input.

        Returns
        -------
        tuple[AtomRef, ...]
            Observed hydrogens in source atom order; empty for an absent residue.
        """
        residue = self.constitution.residue_or_ligand(residue_id)
        if residue is None:
            return ()
        return tuple(
            AtomRef(residue_id, atom.name)
            for atom in residue.atom_sites
            if atom.is_hydrogen()
        )

    def bonds_for_atom(self, atom_ref: AtomRef) -> tuple[TopologyBond, ...]:
        """Return explicit connection evidence without resolving unknown chemistry.

        Parameters
        ----------
        atom_ref : AtomRef
            Atom identity in the original input.

        Returns
        -------
        tuple[TopologyBond, ...]
            Source declarations in this observation's atom address space.
            Relationship types remain distinct: a connectivity-only CONECT is
            not proof of a covalent H attachment. Empty means no declaration,
            not proof of an unbound atom; geometric inference is not performed.
        """
        atom_index = self.constitution.resolve_atom_index(atom_ref)
        if atom_index is None:
            return ()
        return tuple(
            bond
            for bond in self.bonds_for_residue(atom_ref.residue_id)
            if bond.involves(atom_index)
        )

    def bonds_for_residue(self, residue_id: ResidueId) -> tuple[TopologyBond, ...]:
        """Return original bonds touching a residue, including external endpoints.

        Parameters
        ----------
        residue_id : ResidueId
            Identity in the original observation's address space.

        Returns
        -------
        tuple[TopologyBond, ...]
            Source declarations in topology order, each appearing once. Empty for
            an absent residue or no declared bonds. The immutable index is built
            once per observation, not once per chemistry decision.
        """
        return self._bonds_by_residue.get(residue_id, ())
