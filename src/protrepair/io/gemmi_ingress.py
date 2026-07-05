"""gemmi-backed canonical structure ingress for coordinate formats."""

from os import PathLike
from pathlib import Path

from protrepair.errors import (
    ProtrepairError,
    StructureInputTooLargeError,
    StructureNormalizationError,
)
from protrepair.io.gemmi_normalization import (
    gemmi,
    infer_file_format,
    normalize_altloc,
    normalize_chain_id,
    normalize_insertion_code,
    to_gemmi_coor_format,
)
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.structure_ingress import (
    PdbConectAtomIdentity,
    normalize_raw_structure,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat

MAX_STRUCTURE_INPUT_BYTES = 256 * 1024 * 1024


def read_structure(
    path: Path | str | PathLike[str],
    *,
    policy: StructureNormalizationPolicy | None = None,
) -> ProteinStructure:
    """Read a path-like coordinate file into the canonical first-model structure."""

    source_path = Path(path)
    try:
        _assert_structure_file_size(source_path)
        file_format = infer_file_format(source_path)
        active_policy = StructureNormalizationPolicy() if policy is None else policy
        if file_format is FileFormat.PDB:
            contents = source_path.read_text(encoding="utf-8")
            raw_structure = read_raw_structure_string(contents, file_format)
            pdb_conect_atom_identity_pairs = _pdb_conect_atom_identity_pairs(contents)
        else:
            raw_structure = read_raw_structure(source_path, file_format)
            pdb_conect_atom_identity_pairs = ()

        return normalize_raw_structure(
            raw_structure,
            file_format=file_format,
            policy=active_policy,
            source_name=source_path.name,
            pdb_conect_atom_identity_pairs=pdb_conect_atom_identity_pairs,
        )
    except ProtrepairError:
        raise
    except UnicodeDecodeError as error:
        raise StructureNormalizationError(
            f"could not decode structure file {source_path.name!r} as UTF-8"
        ) from error
    except OSError as error:
        raise StructureNormalizationError(
            f"could not read structure file {source_path.name!r}: {error.strerror}"
        ) from error
    except (RuntimeError, ValueError) as error:
        raise StructureNormalizationError(
            f"could not parse structure file {source_path.name!r}: {error}"
        ) from error


def read_structure_string(
    contents: str,
    file_format: FileFormat,
    *,
    policy: StructureNormalizationPolicy | None = None,
    source_name: str | None = None,
) -> ProteinStructure:
    """Read an in-memory coordinate payload into the canonical first-model structure."""

    active_policy = StructureNormalizationPolicy() if policy is None else policy
    return read_structure_string_with_policy(
        contents,
        file_format,
        policy=active_policy,
        source_name=source_name,
    )


def read_structure_string_with_policy(
    contents: str,
    file_format: FileFormat,
    *,
    policy: StructureNormalizationPolicy,
    source_name: str | None = None,
) -> ProteinStructure:
    """Read one first-model payload using one canonical normalization policy."""

    try:
        _assert_structure_text_size(contents, source_name=source_name)
        raw_structure = read_raw_structure_string(contents, file_format)
        pdb_conect_atom_identity_pairs = (
            _pdb_conect_atom_identity_pairs(contents)
            if file_format is FileFormat.PDB
            else ()
        )
        return normalize_raw_structure(
            raw_structure,
            file_format=file_format,
            policy=policy,
            source_name=source_name,
            pdb_conect_atom_identity_pairs=pdb_conect_atom_identity_pairs,
        )
    except ProtrepairError:
        raise
    except (RuntimeError, ValueError) as error:
        source = "" if source_name is None else f" {source_name!r}"
        raise StructureNormalizationError(
            f"could not parse structure text{source}: {error}"
        ) from error


def read_raw_structure(path: Path, file_format: FileFormat):
    """Read one coordinate file with a format-specific gemmi ingress path.

    The size guard is repeated here so direct raw-parser callers cannot bypass
    the public ingress limit enforced by read_structure().
    """

    _assert_structure_file_size(path)
    if file_format is FileFormat.PDB:
        return gemmi.read_pdb(str(path))

    return gemmi.read_structure(
        str(path),
        format=to_gemmi_coor_format(file_format),
    )


def read_raw_structure_string(contents: str, file_format: FileFormat):
    """Read one coordinate payload with a format-specific gemmi ingress path."""

    _assert_structure_text_size(contents, source_name=None)
    if file_format is FileFormat.PDB:
        return gemmi.read_pdb_string(contents)

    return gemmi.read_structure_string(
        contents,
        True,
        to_gemmi_coor_format(file_format),
    )


def _assert_structure_file_size(path: Path) -> None:
    """Reject oversized local coordinate files before parser ingress."""

    file_size = path.stat().st_size
    if file_size > MAX_STRUCTURE_INPUT_BYTES:
        raise StructureInputTooLargeError(
            "structure input file exceeds "
            f"{MAX_STRUCTURE_INPUT_BYTES} bytes: {path.name}"
        )


def _assert_structure_text_size(
    contents: str,
    *,
    source_name: str | None,
) -> None:
    """Reject oversized in-memory coordinate payloads before parser ingress."""

    if len(contents) > MAX_STRUCTURE_INPUT_BYTES:
        source = "" if source_name is None else f": {source_name}"
        raise StructureInputTooLargeError(
            "structure input text exceeds "
            f"{MAX_STRUCTURE_INPUT_BYTES} characters{source}"
        )


def _pdb_conect_atom_identity_pairs(
    contents: str,
) -> tuple[tuple[PdbConectAtomIdentity, PdbConectAtomIdentity], ...]:
    """Return all selected source-identity pairs declared by PDB CONECT."""

    atom_identity_by_serial = _first_model_unambiguous_pdb_atom_identities(contents)

    pairs: list[tuple[PdbConectAtomIdentity, PdbConectAtomIdentity]] = []
    seen: set[tuple[PdbConectAtomIdentity, PdbConectAtomIdentity]] = set()
    for line in contents.splitlines():
        if not line.startswith("CONECT"):
            continue

        serials = _pdb_conect_serials(line)
        if len(serials) < 2:
            continue

        source_identity = atom_identity_by_serial.get(serials[0])
        if source_identity is None:
            continue

        for target_serial in serials[1:]:
            target_identity = atom_identity_by_serial.get(target_serial)
            if (
                target_identity is None
                or target_identity.atom_ref == source_identity.atom_ref
            ):
                continue

            canonical = _canonical_pdb_conect_identity_pair(
                source_identity,
                target_identity,
            )
            if canonical in seen:
                continue

            pairs.append(canonical)
            seen.add(canonical)

    return tuple(pairs)


def _first_model_unambiguous_pdb_atom_identities(
    contents: str,
) -> dict[int, PdbConectAtomIdentity]:
    """Return source atom identities that unambiguously belong to model one."""

    records_by_serial: dict[int, list[tuple[int, PdbConectAtomIdentity]]] = {}
    current_model_index = 1
    explicit_model_count = 0
    for line in contents.splitlines():
        if line.startswith("MODEL"):
            explicit_model_count += 1
            current_model_index = explicit_model_count
            continue
        if line.startswith("ENDMDL"):
            current_model_index = explicit_model_count + 1
            continue

        atom_serial_and_identity = _pdb_atom_serial_and_identity(line)
        if atom_serial_and_identity is None:
            continue

        serial, identity = atom_serial_and_identity
        records_by_serial.setdefault(serial, []).append((current_model_index, identity))

    return {
        serial: records[0][1]
        for serial, records in records_by_serial.items()
        if len(records) == 1 and records[0][0] == 1
    }


def _canonical_pdb_conect_identity_pair(
    identity_1: PdbConectAtomIdentity,
    identity_2: PdbConectAtomIdentity,
) -> tuple[PdbConectAtomIdentity, PdbConectAtomIdentity]:
    """Return a deterministic order for one PDB CONECT identity pair."""

    key_1 = (
        identity_1.atom_ref.residue_id,
        identity_1.atom_ref.atom_name,
        identity_1.component_id,
        identity_1.altloc or "",
    )
    key_2 = (
        identity_2.atom_ref.residue_id,
        identity_2.atom_ref.atom_name,
        identity_2.component_id,
        identity_2.altloc or "",
    )
    return (identity_1, identity_2) if key_1 <= key_2 else (identity_2, identity_1)


def _pdb_atom_serial_and_identity(
    line: str,
) -> tuple[int, PdbConectAtomIdentity] | None:
    """Return the PDB atom serial and source identity for one atom line."""

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
        PdbConectAtomIdentity(
            atom_ref=AtomRef(
                residue_id=ResidueId(
                    chain_id=chain_id,
                    seq_num=seq_num,
                    insertion_code=normalize_insertion_code(line[26:27]),
                ),
                atom_name=atom_name,
            ),
            component_id=residue_name,
            altloc=normalize_altloc(line[16:17]),
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
