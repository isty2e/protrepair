"""Independent anchors and native differential tests for PDB integer fields."""

import gemmi
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from protrepair.io.pdb_records import decode_pdb_integer, encode_pdb_integer


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
