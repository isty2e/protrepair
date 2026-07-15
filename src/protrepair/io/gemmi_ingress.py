"""gemmi-backed canonical structure ingress for coordinate formats."""

from collections.abc import Iterator
from math import isfinite
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
    to_gemmi_coor_format,
)
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.source_connection import SourceConnection
from protrepair.io.source_identity import (
    SourceAtomIdentity,
    normalize_altloc,
    normalize_chain_id,
    normalize_insertion_code,
)
from protrepair.io.structure_ingress import normalize_raw_structure
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
)

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
        contents = source_path.read_text(encoding="utf-8")
        return _normalize_structure_text(
            contents,
            file_format=file_format,
            policy=active_policy,
            source_name=source_path.name,
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
    """Read one first-model payload using one canonical normalization policy.

    Parser and normalization failures are exposed as project-owned
    StructureNormalizationError values.
    """

    try:
        return _normalize_structure_text(
            contents,
            file_format=file_format,
            policy=policy,
            source_name=source_name,
        )
    except ProtrepairError:
        raise
    except (RuntimeError, ValueError) as error:
        source = "" if source_name is None else f" {source_name!r}"
        raise StructureNormalizationError(
            f"could not parse structure text{source}: {error}"
        ) from error


def _normalize_structure_text(
    contents: str,
    *,
    file_format: FileFormat,
    policy: StructureNormalizationPolicy,
    source_name: str | None,
) -> ProteinStructure:
    """Parse and normalize one coordinate payload through one boundary path."""

    _assert_structure_text_size(contents, source_name=source_name)
    if file_format is FileFormat.PDB:
        raw_structure = gemmi.read_pdb_string(contents)
        pdb_conect_source_connections = _pdb_conect_source_connections(contents)
        source_element_by_atom_identity = (
            _pdb_source_isotope_element_by_atom_identity(contents)
        )
    else:
        source_document = gemmi.cif.Document()
        raw_structure = gemmi.read_structure_string(
            contents,
            True,
            to_gemmi_coor_format(file_format),
            source_document,
        )
        pdb_conect_source_connections = ()
        source_element_by_atom_identity = (
            _mmcif_source_isotope_element_by_atom_identity(source_document)
        )

    # Typed Gemmi declarations precede untyped PDB CONECT fallback so the
    # canonical endpoint keeps the strongest surviving source fact.
    source_connections = (
        *_source_connections_from_raw_structure(
            raw_structure,
            file_format=file_format,
        ),
        *pdb_conect_source_connections,
    )
    return normalize_raw_structure(
        raw_structure,
        file_format=file_format,
        policy=policy,
        source_name=source_name,
        source_connections=source_connections,
        source_element_by_atom_identity=source_element_by_atom_identity,
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


def _source_connections_from_raw_structure(
    raw_structure: gemmi.Structure,
    *,
    file_format: FileFormat,
) -> tuple[SourceConnection, ...]:
    """Project Gemmi connection records into boundary-normalized facts."""

    source_connections: list[SourceConnection] = []
    for connection in raw_structure.connections:
        endpoint_1 = _source_atom_identity_from_connection_partner(connection.partner1)
        endpoint_2 = _source_atom_identity_from_connection_partner(connection.partner2)
        if endpoint_1 is None or endpoint_2 is None:
            continue
        if endpoint_1.atom_ref.residue_id == endpoint_2.atom_ref.residue_id:
            continue

        relationship_type = _relationship_type_from_connection(connection.type)
        source_connections.append(
            SourceConnection(
                endpoint_1=endpoint_1,
                endpoint_2=endpoint_2,
                relationship_type=relationship_type,
                source_metadata=SourceBondMetadata(
                    record_type=_source_connection_record_type(
                        file_format=file_format,
                        relationship_type=relationship_type,
                    ),
                    source_id=connection.link_id or connection.name,
                    reported_distance_angstrom=_normalize_reported_connection_distance(
                        connection.reported_distance
                    ),
                ),
            )
        )

    return tuple(dict.fromkeys(source_connections))


def _source_connection_record_type(
    *,
    file_format: FileFormat,
    relationship_type: BondRelationshipType,
) -> SourceBondRecordType:
    """Return the source record contract for one Gemmi connection."""

    if file_format is FileFormat.MMCIF:
        return SourceBondRecordType.MMCIF_STRUCT_CONN
    if relationship_type is BondRelationshipType.DISULFIDE:
        return SourceBondRecordType.PDB_SSBOND
    return SourceBondRecordType.PDB_LINK


def _source_atom_identity_from_connection_partner(
    partner: gemmi.AtomAddress,
) -> SourceAtomIdentity | None:
    """Return a normalized endpoint for one Gemmi connection partner."""

    atom_name = partner.atom_name.strip()
    component_id = partner.res_id.name.strip()
    seq_num = partner.res_id.seqid.num
    if not atom_name or not component_id or seq_num is None:
        return None

    return SourceAtomIdentity(
        atom_ref=AtomRef(
            residue_id=ResidueId(
                chain_id=normalize_chain_id(partner.chain_name),
                seq_num=int(seq_num),
                insertion_code=normalize_insertion_code(partner.res_id.seqid.icode),
            ),
            atom_name=atom_name,
        ),
        component_id=component_id,
        altloc=normalize_altloc(partner.altloc),
    )


def _relationship_type_from_connection(
    connection_type: gemmi.ConnectionType,
) -> BondRelationshipType:
    """Map one Gemmi connection type into canonical relationship semantics."""

    if connection_type is gemmi.ConnectionType.Covale:
        return BondRelationshipType.COVALENT
    if connection_type is gemmi.ConnectionType.Disulf:
        return BondRelationshipType.DISULFIDE
    if connection_type is gemmi.ConnectionType.Hydrog:
        return BondRelationshipType.HYDROGEN_BOND
    if connection_type is gemmi.ConnectionType.MetalC:
        return BondRelationshipType.METAL_COORDINATION
    return BondRelationshipType.UNKNOWN


def _normalize_reported_connection_distance(
    reported_distance: float,
) -> float | None:
    """Return a positive finite source distance when one was reported."""

    try:
        normalized_distance = float(reported_distance)
    except (TypeError, ValueError):
        return None
    if not isfinite(normalized_distance) or normalized_distance <= 0.0:
        return None
    return normalized_distance


def _pdb_conect_source_connections(
    contents: str,
) -> tuple[SourceConnection, ...]:
    """Return boundary-normalized connections declared by PDB CONECT."""

    atom_identity_by_serial = _first_model_unambiguous_pdb_atom_identities(contents)

    source_connections: list[SourceConnection] = []
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

            source_connections.append(
                SourceConnection(
                    endpoint_1=source_identity,
                    endpoint_2=target_identity,
                    relationship_type=BondRelationshipType.UNKNOWN,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                        source_id="CONECT",
                    ),
                )
            )

    return tuple(dict.fromkeys(source_connections))


def _first_model_unambiguous_pdb_atom_identities(
    contents: str,
) -> dict[int, SourceAtomIdentity]:
    """Return source atom identities that unambiguously belong to model one.

    CONECT is lowered onto the canonical first-model structure, so serial reuse
    in later MODEL sections does not make a first-model serial ambiguous. After
    first-model ENDMDL, atom records are treated as outside model one even when
    malformed trailing atoms appear without another MODEL record.
    """

    records_by_serial: dict[int, list[SourceAtomIdentity]] = {}
    for line in _first_model_pdb_atom_lines(contents):
        atom_serial_and_identity = _pdb_atom_serial_and_identity(line)
        if atom_serial_and_identity is None:
            continue

        serial, identity = atom_serial_and_identity
        records_by_serial.setdefault(serial, []).append(identity)

    return {
        serial: records[0]
        for serial, records in records_by_serial.items()
        if len(records) == 1
    }


def _pdb_source_isotope_element_by_atom_identity(
    contents: str,
) -> dict[SourceAtomIdentity, str]:
    """Return first-model PDB isotope symbols before Gemmi element projection."""

    source_elements: dict[SourceAtomIdentity, str] = {}
    for line in _first_model_pdb_atom_lines(contents):
        atom_serial_and_identity = _pdb_atom_serial_and_identity(line)
        if atom_serial_and_identity is None:
            continue

        source_symbol = line[76:78].strip().upper()
        if source_symbol not in {"D", "T"}:
            continue
        _record_source_isotope_element(
            source_elements,
            atom_serial_and_identity[1],
            source_symbol,
        )

    return source_elements


def _mmcif_source_isotope_element_by_atom_identity(
    document: gemmi.cif.Document,
) -> dict[SourceAtomIdentity, str]:
    """Return first-model mmCIF isotope symbols before Gemmi projection."""

    if len(document) == 0:
        return {}

    block = document[0]
    type_symbols = tuple(block.find_values("_atom_site.type_symbol"))
    if not type_symbols:
        return {}

    row_count = len(type_symbols)
    atom_names = _coalesced_mmcif_column(
        block,
        ("_atom_site.auth_atom_id", "_atom_site.label_atom_id"),
        row_count=row_count,
    )
    component_ids = _coalesced_mmcif_column(
        block,
        ("_atom_site.auth_comp_id", "_atom_site.label_comp_id"),
        row_count=row_count,
    )
    chain_ids = _coalesced_mmcif_column(
        block,
        ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"),
        row_count=row_count,
    )
    sequence_numbers = _coalesced_mmcif_column(
        block,
        ("_atom_site.auth_seq_id", "_atom_site.label_seq_id"),
        row_count=row_count,
    )
    insertion_codes = _coalesced_mmcif_column(
        block,
        ("_atom_site.pdbx_PDB_ins_code",),
        row_count=row_count,
        required=False,
    )
    altlocs = _coalesced_mmcif_column(
        block,
        ("_atom_site.label_alt_id",),
        row_count=row_count,
        required=False,
    )
    model_numbers = _coalesced_mmcif_column(
        block,
        ("_atom_site.pdbx_PDB_model_num",),
        row_count=row_count,
        required=False,
    )
    first_model_number = next(
        (
            _non_null_cif_value(model_number)
            for model_number in model_numbers
            if _non_null_cif_value(model_number)
        ),
        "",
    )

    source_elements: dict[SourceAtomIdentity, str] = {}
    for row_index, source_symbol in enumerate(type_symbols):
        normalized_source_symbol = _non_null_cif_value(source_symbol).upper()
        if normalized_source_symbol not in {"D", "T"}:
            continue
        if (
            first_model_number
            and _non_null_cif_value(model_numbers[row_index])
            != first_model_number
        ):
            continue

        atom_name = _non_null_cif_value(atom_names[row_index])
        component_id = _non_null_cif_value(component_ids[row_index])
        chain_id = _non_null_cif_value(chain_ids[row_index])
        sequence_number = _non_null_cif_value(sequence_numbers[row_index])
        if not atom_name or not component_id or not sequence_number:
            continue
        try:
            seq_num = int(sequence_number)
        except ValueError:
            continue

        identity = SourceAtomIdentity(
            atom_ref=AtomRef(
                residue_id=ResidueId(
                    chain_id=normalize_chain_id(chain_id),
                    seq_num=seq_num,
                    insertion_code=normalize_insertion_code(
                        _non_null_cif_value(insertion_codes[row_index])
                    ),
                ),
                atom_name=atom_name,
            ),
            component_id=component_id,
            altloc=normalize_altloc(_non_null_cif_value(altlocs[row_index])),
        )
        _record_source_isotope_element(
            source_elements,
            identity,
            normalized_source_symbol,
        )

    return source_elements


def _coalesced_mmcif_column(
    block: gemmi.cif.Block,
    tags: tuple[str, ...],
    *,
    row_count: int,
    required: bool = True,
) -> tuple[str, ...]:
    """Coalesce atom-site columns by preference with validated row counts."""

    columns: list[tuple[str, ...]] = []
    for tag in tags:
        values = tuple(block.find_values(tag))
        if not values:
            continue
        if len(values) != row_count:
            raise StructureNormalizationError(
                f"mmCIF atom-site column {tag} has inconsistent row count"
            )
        columns.append(values)

    if not columns and required:
        raise StructureNormalizationError(
            f"mmCIF atom-site data requires one of {', '.join(tags)}"
        )
    if not columns:
        return ("",) * row_count

    return tuple(
        next(
            (
                value
                for column in columns
                if (value := _non_null_cif_value(column[row_index]))
            ),
            "",
        )
        for row_index in range(row_count)
    )


def _record_source_isotope_element(
    source_elements: dict[SourceAtomIdentity, str],
    identity: SourceAtomIdentity,
    source_symbol: str,
) -> None:
    """Record one isotope symbol or reject contradictory duplicate identity."""

    previous_symbol = source_elements.get(identity)
    if previous_symbol is not None and previous_symbol != source_symbol:
        raise StructureNormalizationError(
            "source atom identity carries conflicting isotope symbols: "
            f"{identity.atom_ref.display_token()}"
        )
    source_elements[identity] = source_symbol


def _non_null_cif_value(value: str) -> str:
    """Normalize CIF null markers into an absent source token."""

    normalized_value = value.strip()
    return "" if normalized_value in {".", "?"} else normalized_value


def _first_model_pdb_atom_lines(contents: str) -> Iterator[str]:
    """Yield coordinate records belonging unambiguously to the first PDB model."""

    current_model_index = 1
    explicit_model_count = 0
    first_model_closed = False
    for line in contents.splitlines():
        if line.startswith("MODEL"):
            explicit_model_count += 1
            current_model_index = explicit_model_count
            if current_model_index > 1:
                first_model_closed = True
            continue
        if line.startswith("ENDMDL"):
            if current_model_index == 1:
                first_model_closed = True
            current_model_index = explicit_model_count + 1
            continue
        if first_model_closed or current_model_index != 1:
            continue
        if line.startswith(("ATOM  ", "HETATM")):
            yield line


def _pdb_atom_serial_and_identity(
    line: str,
) -> tuple[int, SourceAtomIdentity] | None:
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
        SourceAtomIdentity(
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
