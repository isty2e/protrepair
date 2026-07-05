"""Shared normalization helpers for gemmi-backed I/O boundaries."""

from pathlib import Path

import gemmi

from protrepair.errors import UnsupportedFileFormatError
from protrepair.structure.provenance import FileFormat

PDB_SUFFIXES = {".ent", ".pdb"}
MMCIF_SUFFIXES = {".cif", ".mmcif"}


def infer_file_format(path: Path) -> FileFormat:
    """Infer the canonical structure format from a filesystem path."""

    suffix = path.suffix.lower()
    if suffix in PDB_SUFFIXES:
        return FileFormat.PDB

    if suffix in MMCIF_SUFFIXES:
        return FileFormat.MMCIF

    raise UnsupportedFileFormatError(
        f"cannot infer structure format from path suffix {path.suffix!r}"
    )


def to_gemmi_coor_format(file_format: FileFormat):
    """Map a canonical file format to the gemmi coordinate format enum."""

    if file_format is FileFormat.PDB:
        return gemmi.CoorFormat.Pdb

    if file_format is FileFormat.MMCIF:
        return gemmi.CoorFormat.Mmcif

    raise UnsupportedFileFormatError(f"unsupported file format: {file_format}")


def normalize_formal_charge(charge: int) -> int | None:
    """Normalize gemmi formal charges to the canonical nullable form."""

    if charge == 0:
        return None

    return charge
