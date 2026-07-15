"""Tests for boundary-normalized source connection facts."""

from typing import cast

import pytest

from protrepair.io.source_connection import SourceConnection
from protrepair.io.source_identity import SourceAtomIdentity
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
)


def _source_identity(
    chain_id: str,
    seq_num: int,
    atom_name: str,
    *,
    component_id: str = "ALA",
    altloc: str | None = None,
) -> SourceAtomIdentity:
    return SourceAtomIdentity(
        atom_ref=AtomRef(ResidueId(chain_id, seq_num), atom_name),
        component_id=component_id,
        altloc=altloc,
    )


def _source_connection(
    endpoint_1: SourceAtomIdentity,
    endpoint_2: SourceAtomIdentity,
    *,
    relationship_type: BondRelationshipType = BondRelationshipType.COVALENT,
    record_type: SourceBondRecordType = SourceBondRecordType.PDB_LINK,
) -> SourceConnection:
    return SourceConnection(
        endpoint_1=endpoint_1,
        endpoint_2=endpoint_2,
        relationship_type=relationship_type,
        source_metadata=SourceBondMetadata(record_type=record_type),
    )


def test_source_connection_canonicalizes_endpoint_order() -> None:
    """Unordered source endpoints should have one stable identity."""

    later = _source_identity("B", 2, "N")
    earlier = _source_identity("A", 1, "C")

    connection = _source_connection(later, earlier)

    assert connection.endpoint_pair() == (earlier, later)


def test_source_connection_rejects_one_atom_ref_as_both_endpoints() -> None:
    """Source variants of one canonical atom cannot form a topology edge."""

    endpoint_1 = _source_identity("A", 1, "CA", altloc="A")
    endpoint_2 = _source_identity("A", 1, "CA", altloc="B")

    with pytest.raises(ValueError, match="two distinct atoms"):
        _source_connection(endpoint_1, endpoint_2)


def test_source_connection_rejects_noncanonical_field_types() -> None:
    """Boundary connection facts reject values outside their canonical axes."""

    endpoint_1 = _source_identity("A", 1, "C")
    endpoint_2 = _source_identity("A", 2, "N")
    metadata = SourceBondMetadata(record_type=SourceBondRecordType.PDB_LINK)

    with pytest.raises(TypeError, match="endpoints"):
        SourceConnection(
            endpoint_1=cast(SourceAtomIdentity, object()),
            endpoint_2=endpoint_2,
            relationship_type=BondRelationshipType.COVALENT,
            source_metadata=metadata,
        )
    with pytest.raises(TypeError, match="relationship_type"):
        SourceConnection(
            endpoint_1=endpoint_1,
            endpoint_2=endpoint_2,
            relationship_type=cast(BondRelationshipType, "covalent"),
            source_metadata=metadata,
        )
    with pytest.raises(TypeError, match="metadata"):
        SourceConnection(
            endpoint_1=endpoint_1,
            endpoint_2=endpoint_2,
            relationship_type=BondRelationshipType.COVALENT,
            source_metadata=cast(SourceBondMetadata, object()),
        )


@pytest.mark.parametrize(
    ("connection", "expected"),
    (
        (
            _source_connection(
                _source_identity("A", 1, "C"),
                _source_identity("A", 2, "N"),
            ),
            True,
        ),
        (
            _source_connection(
                _source_identity("A", 1, "C"),
                _source_identity("A", 2, "N"),
                relationship_type=BondRelationshipType.UNKNOWN,
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            ),
            True,
        ),
        (
            _source_connection(
                _source_identity("A", 1, "C"),
                _source_identity("A", 2, "N"),
                relationship_type=BondRelationshipType.UNKNOWN,
                record_type=SourceBondRecordType.PDB_CONECT,
            ),
            False,
        ),
        (
            _source_connection(
                _source_identity("A", 1, "C"),
                _source_identity("A", 2, "N"),
                relationship_type=BondRelationshipType.DISULFIDE,
                record_type=SourceBondRecordType.PDB_SSBOND,
            ),
            False,
        ),
        (
            _source_connection(
                _source_identity("A", 1, "CA"),
                _source_identity("A", 2, "N"),
            ),
            False,
        ),
        (
            _source_connection(
                _source_identity("A", 1, "C"),
                _source_identity("B", 2, "N"),
            ),
            False,
        ),
        (
            _source_connection(
                _source_identity("A", 1, "C"),
                _source_identity("A", 1, "N"),
            ),
            False,
        ),
    ),
)
def test_source_connection_peptide_context_is_typed_and_inter_residue(
    connection: SourceConnection,
    expected: bool,
) -> None:
    """Only typed same-chain C-N links can establish peptide context."""

    assert connection.is_peptide_link_candidate() is expected


def test_only_pdb_ssbond_allows_residue_level_altloc_matching() -> None:
    """SSBOND names residues, while LINK-like records identify atom variants."""

    endpoint_1 = _source_identity("A", 1, "SG", component_id="CYS", altloc="A")
    endpoint_2 = _source_identity("A", 2, "SG", component_id="CYS", altloc="B")

    ssbond = _source_connection(
        endpoint_1,
        endpoint_2,
        relationship_type=BondRelationshipType.DISULFIDE,
        record_type=SourceBondRecordType.PDB_SSBOND,
    )
    link = _source_connection(endpoint_1, endpoint_2)

    assert not ssbond.requires_exact_altloc_match()
    assert link.requires_exact_altloc_match()


def test_only_pdb_conect_has_fallback_record_semantics() -> None:
    """Untyped CONECT yields to a stronger declaration for one endpoint pair."""

    endpoint_1 = _source_identity("A", 1, "C")
    endpoint_2 = _source_identity("A", 2, "N")
    conect = _source_connection(
        endpoint_1,
        endpoint_2,
        relationship_type=BondRelationshipType.UNKNOWN,
        record_type=SourceBondRecordType.PDB_CONECT,
    )
    link = _source_connection(endpoint_1, endpoint_2)

    assert conect.is_fallback_record()
    assert not link.is_fallback_record()
