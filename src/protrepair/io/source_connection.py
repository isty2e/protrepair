"""Boundary-normalized source connection contracts."""

from dataclasses import dataclass

from protrepair.io.source_identity import SourceAtomIdentity
from protrepair.structure.topology import (
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
)


@dataclass(frozen=True, slots=True)
class SourceConnection:
    """One source-declared connection before canonical topology lowering."""

    endpoint_1: SourceAtomIdentity
    endpoint_2: SourceAtomIdentity
    relationship_type: BondRelationshipType
    source_metadata: SourceBondMetadata

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
        """Return whether this record fills only an otherwise unclaimed edge."""

        return self.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
