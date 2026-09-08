"""Boundary-normalized source connection contracts."""

from dataclasses import dataclass, replace

from protrepair.errors import ModelInvariantError
from protrepair.io.source_identity import SourceAtomIdentity
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    TopologyBond,
    is_covalent_like_relationship,
)


@dataclass(frozen=True, slots=True)
class SourceConnection:
    """One source-declared connection before canonical topology lowering.

    ``order=None`` means the record supplies connectivity without an order.
    In particular, a single CONECT neighbor occurrence is not an assertion
    that the chemical bond is single.

    Parameters
    ----------
    endpoint_1, endpoint_2 : SourceAtomIdentity
        Source endpoints, including component and alternate-location identity.
    relationship_type : BondRelationshipType
        Source-declared type, before component or sequence resolution.
    source_metadata : SourceBondMetadata
        Record type, identifier, and reported distance from the source.
    order : int or None, default=None
        Explicit positive integral order; None supplies no order evidence.

    Raises
    ------
    TypeError
        A field has a noncanonical type.
    ValueError
        Endpoints identify the same atom or order is nonpositive.
    """

    endpoint_1: SourceAtomIdentity
    endpoint_2: SourceAtomIdentity
    relationship_type: BondRelationshipType
    source_metadata: SourceBondMetadata
    order: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_1, SourceAtomIdentity) or not isinstance(
            self.endpoint_2,
            SourceAtomIdentity,
        ):
            raise TypeError(
                "source connection endpoints must be SourceAtomIdentity values"
            )
        if self.endpoint_1.atom_ref == self.endpoint_2.atom_ref:
            raise ValueError("source connections require two distinct atoms")
        if not isinstance(self.relationship_type, BondRelationshipType):
            raise TypeError(
                "source connection relationship_type must be a BondRelationshipType"
            )
        if not isinstance(self.source_metadata, SourceBondMetadata):
            raise TypeError("source connection metadata must be SourceBondMetadata")
        if self.order is not None:
            if isinstance(self.order, bool) or not isinstance(self.order, int):
                raise TypeError("source connection order must be an integer or None")
            if self.order <= 0:
                raise ValueError("source connection order must be positive")

        endpoint_1 = self.endpoint_1
        endpoint_2 = self.endpoint_2
        if endpoint_2.sort_key() < endpoint_1.sort_key():
            endpoint_1, endpoint_2 = endpoint_2, endpoint_1

        object.__setattr__(self, "endpoint_1", endpoint_1)
        object.__setattr__(self, "endpoint_2", endpoint_2)

    def endpoint_pair(self) -> tuple[SourceAtomIdentity, SourceAtomIdentity]:
        """Return the canonically ordered source endpoint pair."""

        return (self.endpoint_1, self.endpoint_2)

    def is_peptide_link_candidate(self) -> bool:
        """Return whether this connection can support peptide-chain context."""

        residue_id_1 = self.endpoint_1.atom_ref.residue_id
        residue_id_2 = self.endpoint_2.atom_ref.residue_id
        return bool(
            self.source_metadata.record_type is not SourceBondRecordType.PDB_CONECT
            and self.relationship_type
            in {
                BondRelationshipType.COVALENT,
                BondRelationshipType.UNKNOWN,
            }
            and residue_id_1 != residue_id_2
            and residue_id_1.chain_id == residue_id_2.chain_id
            and {
                self.endpoint_1.atom_ref.atom_name,
                self.endpoint_2.atom_ref.atom_name,
            }
            == {"C", "N"}
        )

    def requires_exact_altloc_match(self) -> bool:
        """Return whether lowering requires the selected source altloc."""

        return self.source_metadata.record_type is not SourceBondRecordType.PDB_SSBOND

    def is_fallback_record(self) -> bool:
        """Return whether typed records take precedence over this record's type."""

        return self.source_metadata.record_type is SourceBondRecordType.PDB_CONECT

    def merge(self, other: "SourceConnection") -> "SourceConnection":
        """Combine declarations for the same surviving atom pair.

        Parameters
        ----------
        other : SourceConnection
            A declaration whose endpoints survived canonical selection.

        Returns
        -------
        SourceConnection
            Typed metadata with any compatible supplementary order evidence.

        Raises
        ------
        ModelInvariantError
            The endpoints, explicit orders, or typed declarations conflict.
        """
        if {self.endpoint_1.atom_ref, self.endpoint_2.atom_ref} != {
            other.endpoint_1.atom_ref,
            other.endpoint_2.atom_ref,
        }:
            raise ModelInvariantError(
                "cannot merge source connections for different atoms"
            )
        if (
            self.order is not None
            and other.order is not None
            and self.order != other.order
        ):
            raise ModelInvariantError(
                "conflicting source bond orders for one atom pair"
            )

        if not self.is_fallback_record() and not other.is_fallback_record():
            if replace(self, order=None) != replace(other, order=None):
                raise ModelInvariantError(
                    "conflicting bonds in typed source connections"
                )
        preferred = other if self.is_fallback_record() else self
        return replace(
            preferred, order=self.order if self.order is not None else other.order
        )

    def to_topology_bond(
        self,
        atom_index_1: AtomIndex,
        atom_index_2: AtomIndex,
        *,
        expected_bond: TopologyBond | None,
    ) -> TopologyBond:
        """Resolve absent attributes without replacing explicit source evidence.

        Parameters
        ----------
        atom_index_1, atom_index_2 : AtomIndex
            Surviving canonical endpoints corresponding to this connection.
        expected_bond : TopologyBond or None
            Component or sequence chemistry for the same endpoint pair.

        Returns
        -------
        TopologyBond
            Source-backed connectivity with a resolved or explicitly unknown order.

        Raises
        ------
        ModelInvariantError
            The expected endpoints differ, or a disulfide carries a multiple order.
        """
        if expected_bond is not None and set(expected_bond.endpoint_pair()) != {
            atom_index_1,
            atom_index_2,
        }:
            raise ModelInvariantError(
                "expected chemistry refers to a different atom pair"
            )
        relationship = self.relationship_type
        order = self.order
        aromatic = False
        compatible = (
            expected_bond is not None
            and is_covalent_like_relationship(expected_bond)
            and relationship
            in {
                BondRelationshipType.UNKNOWN,
                BondRelationshipType.COVALENT,
                BondRelationshipType.DISULFIDE,
            }
        )
        if compatible and expected_bond is not None:
            if relationship is BondRelationshipType.UNKNOWN:
                relationship = expected_bond.relationship_type
            if order is None:
                order = expected_bond.order
            aromatic = expected_bond.aromatic and order in {1, 2}
        if relationship is BondRelationshipType.DISULFIDE:
            if self.order not in {None, 1}:
                raise ModelInvariantError(
                    "disulfide connectivity conflicts with a multiple bond order"
                )
            order = 1
        return TopologyBond(
            atom_index_1,
            atom_index_2,
            order=order,
            aromatic=aromatic,
            relationship_type=relationship,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=self.source_metadata,
        )
