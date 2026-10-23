"""Fixed-column identities and the uppercase Gemmi PDB integer dialect."""

from typing import Literal

from protrepair.io.source_identity import (
    normalize_altloc,
    normalize_chain_id,
    normalize_insertion_code,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.source_connection import SourceAtomIdentity

PDB_MAX_ATOM_SERIAL = 43770015
PDB_MAX_RESIDUE_NUMBER = 1223055
_BASE36_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def decode_pdb_integer(field: str, *, width: Literal[4, 5]) -> int:
    """Decode decimal or uppercase hybrid-36 without accepting the lowercase block.

    Parameters
    ----------
    field : str
        Complete fixed-width field, including decimal padding.
    width : Literal[4, 5]
        Residue-number or atom-serial width.

    Returns
    -------
    int
        The identifier represented by the field.

    Raises
    ------
    ValueError
        The field is not in the supported integer dialect.
    """
    if len(field) == width and field.isascii():
        token = field.strip()
        digits = token[1:] if token.startswith(("-", "+")) else token
        if digits and all("0" <= c <= "9" for c in digits):
            return int(token)
        if "A" <= field[0] <= "Z" and all(c in _BASE36_DIGITS for c in field):
            return int(field, 36) - 10 * 36 ** (width - 1) + 10**width
    raise ValueError(f"unsupported {width}-column PDB integer {field!r}")


def encode_pdb_integer(value: int, *, width: Literal[4, 5]) -> str:
    """Encode one bounded decimal or uppercase hybrid-36 identifier.

    Parameters
    ----------
    value : int
        Identifier to encode; decimal negatives must fit the field.
    width : Literal[4, 5]
        Residue-number or atom-serial width.

    Returns
    -------
    str
        A complete fixed-width field.

    Raises
    ------
    ValueError
        The identifier would overflow or wrap the supported alphabet.
    """
    if -(10 ** (width - 1) - 1) <= value < 10**width:
        return f"{value:>{width}}"
    encoded = value + 10 * 36 ** (width - 1) - 10**width
    if value < 10**width or encoded >= 36**width:
        raise ValueError(f"PDB integer {value} exceeds the {width}-column range")
    characters: list[str] = []
    for _ in range(width):
        encoded, digit = divmod(encoded, 36)
        characters.append(_BASE36_DIGITS[digit])
    return "".join(reversed(characters))


def pdb_atom_identity(line: str) -> SourceAtomIdentity | None:
    """Read atom identity independently of the serial field.

    Parameters
    ----------
    line : str
        One PDB record, including fixed-column padding.

    Returns
    -------
    SourceAtomIdentity or None
        Identity for a populated ATOM/HETATM record, or no identity otherwise.

    Raises
    ------
    ValueError
        A populated residue number has unsupported encoding.
    """
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    atom_name, component = line[12:16].strip(), line[17:20].strip()
    if not atom_name or not component or not line[22:26].strip():
        return None
    return SourceAtomIdentity(
        atom_ref=AtomRef(
            residue_id=ResidueId(
                chain_id=normalize_chain_id(line[20:22]),
                seq_num=decode_pdb_integer(line[22:26], width=4),
                insertion_code=normalize_insertion_code(line[26:27]),
            ),
            atom_name=atom_name,
        ),
        component_id=component,
        altloc=normalize_altloc(line[16:17]),
    )
