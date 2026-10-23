"""Independent anchors and native differential tests for PDB integer fields."""

import gemmi
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from protrepair.io.pdb_records import (
    decode_pdb_integer,
    encode_pdb_integer,
    pdb_atom_identity,
    pdb_atom_identity_matches,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.source_connection import SourceAtomIdentity


def _native_record(serial: int, residue_number: int) -> str:
    raw = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "ALA"
    residue.seqid = gemmi.SeqId(residue_number, " ")
    atom = gemmi.Atom()
    atom.name = "CA"
    atom.element = gemmi.Element("C")
    atom.serial = serial
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    raw.add_model(model)
    options = gemmi.PdbWriteOptions()
    options.preserve_serial = True
    return next(
        line
        for line in raw.make_pdb_string(options).splitlines()
        if line.startswith("ATOM")
    )


@given(
    st.integers(min_value=1, max_value=43770015),
    st.integers(min_value=-999, max_value=1223055),
)
@settings(max_examples=80)
def test_codec_agrees_with_gemmi_in_the_admitted_domain(
    serial: int, residue_number: int
) -> None:
    line = _native_record(serial, residue_number)
    native = gemmi.read_pdb_string(line + "\nEND\n")[0][0][0][0]
    assert encode_pdb_integer(serial, width=5) == line[6:11]
    assert encode_pdb_integer(residue_number, width=4) == line[22:26]
    assert decode_pdb_integer(line[6:11], width=5) == native.serial == serial
    assert decode_pdb_integer(line[22:26], width=4) == residue_number


@pytest.mark.parametrize(
    "value, token", [(9999, "9999"), (10000, "A000"), (1223055, "ZZZZ"), (-999, "-999")]
)
def test_residue_encoding_has_independent_semantic_anchors(
    value: int, token: str
) -> None:
    assert encode_pdb_integer(value, width=4) == token
    assert decode_pdb_integer(token, width=4) == value


@pytest.mark.parametrize(
    "value, token", [(99999, "99999"), (100000, "A0000"), (43770015, "ZZZZZ")]
)
def test_serial_encoding_has_independent_semantic_anchors(
    value: int, token: str
) -> None:
    assert encode_pdb_integer(value, width=5) == token
    assert decode_pdb_integer(token, width=5) == value


@pytest.mark.parametrize("value", [-1000, 1223056])
def test_residue_encoding_rejects_gemmi_wraparound(value: int) -> None:
    with pytest.raises(ValueError):
        encode_pdb_integer(value, width=4)


def test_serial_encoding_rejects_gemmi_wraparound() -> None:
    assert _native_record(43770016, 1)[6:11] == "00000"
    with pytest.raises(ValueError):
        encode_pdb_integer(43770016, width=5)


@pytest.mark.parametrize(
    "token",
    ["a000", "A00a", " A00", "1_23", "****", "    ", "\uff11\uff10\uff10\uff10"],
)
def test_decoder_does_not_accept_other_integer_dialects(token: str) -> None:
    with pytest.raises(ValueError):
        decode_pdb_integer(token, width=4)


@given(
    number=st.integers(min_value=-999, max_value=1223055),
    chain_field=st.sampled_from((" A", "AA", "AB", "  ", " _")),
    name=st.sampled_from(("CA", "ca", "HB2")),
    component=st.sampled_from(("ALA", "ala", "GLY")),
    altloc=st.sampled_from((" ", "A", "a", "\x00")),
    insertion=st.sampled_from((" ", "B", "b")),
)
@settings(max_examples=60)
def test_record_predicate_preserves_normalized_identity_and_checks_each_field(
    number: int,
    chain_field: str,
    name: str,
    component: str,
    altloc: str,
    insertion: str,
) -> None:
    record = _native_record(1, number)
    record = (
        record[:12]
        + name.ljust(4)
        + altloc
        + component
        + chain_field
        + record[22:26]
        + insertion
        + record[27:]
    )
    expected_altloc = None if altloc in {" ", "\x00"} else altloc
    ref = AtomRef(
        ResidueId(chain_field.strip() or "_", number, insertion.strip() or None), name
    )
    expected = SourceAtomIdentity(ref, component, expected_altloc)
    assert pdb_atom_identity(record) == expected
    assert pdb_atom_identity_matches(
        record, ref, expected.component_id, expected_altloc
    )

    alternative_refs = (
        AtomRef(ResidueId("Z", number, ref.residue_id.insertion_code), name),
        AtomRef(
            ResidueId(
                ref.residue_id.chain_id, number + 1, ref.residue_id.insertion_code
            ),
            name,
        ),
        AtomRef(ResidueId(ref.residue_id.chain_id, number, "C"), name),
        AtomRef(ref.residue_id, "OTHER"),
    )
    assert all(
        not pdb_atom_identity_matches(
            record, other, expected.component_id, expected_altloc
        )
        for other in alternative_refs
    )
    assert not pdb_atom_identity_matches(record, ref, "SER", expected_altloc)
    assert not pdb_atom_identity_matches(record, ref, expected.component_id, "z")


@pytest.mark.parametrize("record", ["HEADER", "TER", "ATOM", "ATOM  " + " " * 74])
def test_empty_identity_has_no_match(record: str) -> None:
    assert pdb_atom_identity(record) is None
    assert not pdb_atom_identity_matches(
        record, AtomRef(ResidueId("A", 1), "CA"), "ALA", None
    )


def test_record_predicate_keeps_decode_errors_ahead_of_identity_mismatches() -> None:
    record = _native_record(1, 1)
    record = record[:22] + "a000" + record[26:]
    with pytest.raises(ValueError, match="unsupported 4-column PDB integer"):
        pdb_atom_identity(record)
    with pytest.raises(ValueError, match="unsupported 4-column PDB integer"):
        pdb_atom_identity_matches(record, AtomRef(ResidueId("B", 1), "CB"), "GLY", "B")
