"""gemmi-backed canonical structure ingress for coordinate formats."""

from pathlib import Path

from protrepair.io.gemmi_normalization import (
    gemmi,
    infer_file_format,
    normalize_chain_id,
    normalize_insertion_code,
    require_gemmi,
    to_gemmi_coor_format,
)
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.structure_ingress import normalize_raw_structure
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat


def read_structure(
    path: Path,
    *,
    policy: StructureNormalizationPolicy | None = None,
) -> ProteinStructure:
    """Read a coordinate file into the canonical structure model."""

    require_gemmi()
    assert gemmi is not None

    file_format = infer_file_format(path)
    raw_structure = read_raw_structure(path, file_format)
    active_policy = StructureNormalizationPolicy() if policy is None else policy
    pdb_conect_atom_ref_pairs = (
        _pdb_conect_atom_ref_pairs(path.read_text(encoding="utf-8"))
        if file_format is FileFormat.PDB
        else ()
    )
    return normalize_raw_structure(
        raw_structure,
        file_format=file_format,
        policy=active_policy,
        source_name=path.name,
        pdb_conect_atom_ref_pairs=pdb_conect_atom_ref_pairs,
    )


def read_structure_string(
    contents: str,
    file_format: FileFormat,
    *,
    policy: StructureNormalizationPolicy | None = None,
    source_name: str | None = None,
) -> ProteinStructure:
    """Read an in-memory coordinate payload into the canonical model."""

    require_gemmi()
    assert gemmi is not None

    raw_structure = read_raw_structure_string(contents, file_format)
    active_policy = StructureNormalizationPolicy() if policy is None else policy
    pdb_conect_atom_ref_pairs = (
        _pdb_conect_atom_ref_pairs(contents) if file_format is FileFormat.PDB else ()
    )
    return normalize_raw_structure(
        raw_structure,
        file_format=file_format,
        policy=active_policy,
        source_name=source_name,
        pdb_conect_atom_ref_pairs=pdb_conect_atom_ref_pairs,
    )


def read_structure_string_with_policy(
    contents: str,
    file_format: FileFormat,
    *,
    policy: StructureNormalizationPolicy,
    source_name: str | None = None,
) -> ProteinStructure:
    """Read one in-memory payload using one canonical normalization policy."""

    require_gemmi()
    assert gemmi is not None

    raw_structure = read_raw_structure_string(contents, file_format)
    pdb_conect_atom_ref_pairs = (
        _pdb_conect_atom_ref_pairs(contents) if file_format is FileFormat.PDB else ()
    )
    return normalize_raw_structure(
        raw_structure,
        file_format=file_format,
        policy=policy,
        source_name=source_name,
        pdb_conect_atom_ref_pairs=pdb_conect_atom_ref_pairs,
    )


def read_raw_structure(path: Path, file_format: FileFormat):
    """Read one coordinate file with a format-specific gemmi ingress path."""

    assert gemmi is not None

    if file_format is FileFormat.PDB:
        return gemmi.read_pdb(str(path))

    return gemmi.read_structure(
        str(path),
        format=to_gemmi_coor_format(file_format),
    )


def read_raw_structure_string(contents: str, file_format: FileFormat):
    """Read one coordinate payload with a format-specific gemmi ingress path."""

    assert gemmi is not None

    if file_format is FileFormat.PDB:
        return gemmi.read_pdb_string(contents)

    return gemmi.read_structure_string(
        contents,
        True,
        to_gemmi_coor_format(file_format),
    )


def _pdb_conect_atom_ref_pairs(
    contents: str,
) -> tuple[tuple[AtomRef, AtomRef], ...]:
    """Return all canonically-ordered atom-ref pairs declared by PDB CONECT."""

    atom_ref_by_serial: dict[int, AtomRef] = {}
    for line in contents.splitlines():
        atom_serial_and_ref = _pdb_atom_serial_and_ref(line)
        if atom_serial_and_ref is None:
            continue

        serial, atom_ref = atom_serial_and_ref
        atom_ref_by_serial[serial] = atom_ref

    pairs: list[tuple[AtomRef, AtomRef]] = []
    seen: set[tuple[AtomRef, AtomRef]] = set()
    for line in contents.splitlines():
        if not line.startswith("CONECT"):
            continue

        serials = _pdb_conect_serials(line)
        if len(serials) < 2:
            continue

        source_atom_ref = atom_ref_by_serial.get(serials[0])
        if source_atom_ref is None:
            continue

        for target_serial in serials[1:]:
            target_atom_ref = atom_ref_by_serial.get(target_serial)
            if target_atom_ref is None or target_atom_ref == source_atom_ref:
                continue

            canonical = (
                (source_atom_ref, target_atom_ref)
                if (source_atom_ref.residue_id, source_atom_ref.atom_name)
                <= (target_atom_ref.residue_id, target_atom_ref.atom_name)
                else (target_atom_ref, source_atom_ref)
            )
            if canonical in seen:
                continue

            pairs.append(canonical)
            seen.add(canonical)

    return tuple(pairs)


def _pdb_atom_serial_and_ref(line: str) -> tuple[int, AtomRef] | None:
    """Return the PDB atom serial and canonical atom reference for one atom line."""

    if not line.startswith(("ATOM  ", "HETATM")):
        return None

    serial_token = line[6:11].strip()
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    chain_id = normalize_chain_id(line[21:22])
    seq_num_token = line[22:26].strip()
    if not serial_token or not atom_name or not residue_name or not seq_num_token:
        return None

    try:
        serial = int(serial_token)
        seq_num = int(seq_num_token)
    except ValueError:
        return None

    return (
        serial,
        AtomRef(
            residue_id=ResidueId(
                chain_id=chain_id,
                seq_num=seq_num,
                insertion_code=normalize_insertion_code(line[26:27]),
            ),
            atom_name=atom_name,
        ),
    )


def _pdb_conect_serials(line: str) -> tuple[int, ...]:
    """Return serial numbers encoded in one fixed-width PDB CONECT line."""

    serials: list[int] = []
    for offset in range(6, len(line), 5):
        token = line[offset : offset + 5].strip()
        if not token:
            continue

        try:
            serials.append(int(token))
        except ValueError:
            continue

    return tuple(serials)
