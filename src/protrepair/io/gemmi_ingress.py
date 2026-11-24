"""gemmi-backed canonical structure ingress for coordinate formats."""

import itertools
import re
from collections import Counter
from collections.abc import Iterator
from math import isfinite
from os import PathLike
from pathlib import Path

from protrepair.errors import (
    ProtrepairError,
    StructureInputTooLargeError,
    StructureNormalizationError,
)
from protrepair.io.deposited_peptide import (
    MmcifSequenceSlot,
    PdbSequenceSlot,
    mmcif_peptide_paths,
    pdb_peptide_paths,
    pdb_seqres_sequences,
)
from protrepair.io.gemmi_normalization import (
    gemmi,
    infer_file_format,
    to_gemmi_coor_format,
)
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.pdb_records import decode_pdb_integer, pdb_atom_identity
from protrepair.io.peptide_link_evidence import (
    cif_uncertain_peptide_ports,
    pdb_uncertain_peptide_ports,
)
from protrepair.io.source_identity import (
    normalize_altloc,
    normalize_chain_id,
    normalize_insertion_code,
)
from protrepair.io.source_peptide import SourcePolymerLabels
from protrepair.io.source_peptide_caps import PdbCapSourceContext
from protrepair.io.structure_ingress import (
    SourceAtomAnnotation,
    normalize_raw_structure,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.peptide import PEPTIDE_CAP_COMPONENT_IDS
from protrepair.structure.provenance import FileFormat
from protrepair.structure.sequence_boundary import (
    DepositedPeptideBoundary,
    DepositedPeptideSequence,
)
from protrepair.structure.source_connection import SourceAtomIdentity, SourceConnection
from protrepair.structure.topology import (
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
)

MAX_STRUCTURE_INPUT_BYTES = 256 * 1024 * 1024

_OCCUPANCY_COEFFICIENT = r"[+-]?(?P<coefficient>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_OCCUPANCY_EXPONENT = r"[eE][+-]?[0-9]+"
_PDB_OCCUPANCY_TOKEN = re.compile(
    rf"{_OCCUPANCY_COEFFICIENT}(?:{_OCCUPANCY_EXPONENT})?"
)
_CIF_OCCUPANCY_TOKEN = re.compile(
    rf"{_OCCUPANCY_COEFFICIENT}(?:{_OCCUPANCY_EXPONENT})?(?:\([0-9]+\))?"
)
_UNANNOTATED_ATOM = SourceAtomAnnotation()


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
    omit_consecutive_residues = policy.omit_consecutive_zero_occupancy_polymer_residues
    prove_whole_residue_omissions = (
        policy.omit_zero_occupancy_polymer_residues or omit_consecutive_residues
    )
    prove_zero_occupancy = (
        policy.omit_zero_occupancy_sidechain_heavy_atoms
        or prove_whole_residue_omissions
    )
    source_orders_by_id: dict[str, int | None] = {}
    source_labels_by_residue: dict[ResidueId, frozenset[SourcePolymerLabels]] = {}
    pdb_cap_source = None
    unresolved_conect_ports: frozenset[AtomRef] = frozenset()
    if file_format is FileFormat.PDB:
        raw_structure = gemmi.read_pdb_string(contents)
        sequences = pdb_seqres_sequences(contents)
        uncertain_links = (
            pdb_uncertain_peptide_ports(
                contents, include_sidechains=omit_consecutive_residues
            )
            if "LINK" in contents
            else frozenset()
        )
        observe_cap_paths = len(raw_structure) != 0 and any(
            residue.het_flag == "H" and residue.name in PEPTIDE_CAP_COMPONENT_IDS
            for chain in raw_structure[0]
            for residue in chain
        )
        # Validate serials even without CONECT, independently of atom attributes.
        atom_identity_by_serial, duplicate_ports = _first_model_pdb_atom_identities(
            contents
        )
        (
            pdb_conect_source_connections,
            first_model_conect_pairs,
            unbound_conect_atoms,
            unresolved_conect_ports,
        ) = _pdb_conect_source_connections(
            contents,
            atom_identity_by_serial=atom_identity_by_serial,
            observe_cap_paths=observe_cap_paths,
            observe_sequence_ports=bool(sequences)
            or observe_cap_paths
            or omit_consecutive_residues,
            include_sidechains=omit_consecutive_residues,
            duplicate_ports=duplicate_ports,
        )
        uncertain_attachment_atoms = (
            (uncertain_links | unresolved_conect_ports)
            if omit_consecutive_residues
            else frozenset()
        )
        unresolved_conect_ports = frozenset(
            ref
            for ref in unresolved_conect_ports
            if (sequences or observe_cap_paths) and ref.atom_name in ("N", "C")
        )
        uncertain_links |= unresolved_conect_ports
        (
            source_element_by_atom_identity,
            source_annotations,
            peptide_break_after,
            adjacent_slots,
            sequence_slots,
        ) = _pdb_source_atom_attributes(
            contents,
            prove_zero_occupancy=prove_zero_occupancy,
            prove_positive_occupancy=prove_whole_residue_omissions,
            observe_cap_paths=observe_cap_paths,
            read_sequence_slots=bool(sequences),
        )
        sequence_boundaries, deposited_sequences = pdb_peptide_paths(
            sequences,
            sequence_slots,
            peptide_break_after,
            remarks=raw_structure.raw_remarks,
        )
        if observe_cap_paths:
            pdb_cap_source = PdbCapSourceContext(
                adjacent_slots,
                first_model_conect_pairs,
                unbound_conect_atoms.union(
                    identity
                    for identity, annotations in source_annotations.items()
                    if len(annotations) != 1
                ),
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
        (
            source_element_by_atom_identity,
            source_annotations,
            source_labels_by_residue,
            (sequence_boundaries, deposited_sequences),
        ) = _mmcif_source_atom_attributes(
            source_document,
            selected_model=raw_structure[0].num if len(raw_structure) else None,
            prove_zero_occupancy=prove_zero_occupancy,
            prove_whole_residue_omissions=prove_whole_residue_omissions,
        )
        source_orders_by_id = _mmcif_connection_orders(source_document)
        uncertain_links = (
            cif_uncertain_peptide_ports(
                source_document.sole_block(),
                source_labels_by_residue,
                include_sidechains=omit_consecutive_residues,
            )
            if len(source_document)
            else frozenset()
        )
        uncertain_attachment_atoms = (
            uncertain_links if omit_consecutive_residues else frozenset()
        )
        peptide_break_after = frozenset()

    # Typed Gemmi declarations precede untyped PDB CONECT fallback so the
    # canonical endpoint keeps the strongest surviving source fact.
    intra_residue_links: list[SourceConnection] = []
    source_connections = (
        *_source_connections_from_raw_structure(
            raw_structure,
            file_format=file_format,
            orders_by_id=source_orders_by_id,
            local_pdb_links=intra_residue_links if omit_consecutive_residues else None,
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
        source_annotations_by_atom_identity=source_annotations,
        source_peptide_break_after=peptide_break_after,
        source_labels_by_residue=source_labels_by_residue,
        deposited_peptide_boundaries=sequence_boundaries,
        deposited_peptide_sequences=deposited_sequences,
        unresolved_conect_ports=unresolved_conect_ports,
        uncertain_peptide_link_ports=frozenset(
            ref for ref in uncertain_links if ref.atom_name in ("N", "C")
        ),
        uncertain_attachment_atoms=uncertain_attachment_atoms,
        intra_residue_pdb_links=tuple(intra_residue_links),
        pdb_cap_source=pdb_cap_source,
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
    orders_by_id: dict[str, int | None] | None = None,
    local_pdb_links: list[SourceConnection] | None = None,
) -> tuple[SourceConnection, ...]:
    """Project Gemmi connection records into boundary-normalized facts."""

    source_connections: list[SourceConnection] = []
    for connection in raw_structure.connections:
        endpoint_1 = _source_atom_identity_from_connection_partner(connection.partner1)
        endpoint_2 = _source_atom_identity_from_connection_partner(connection.partner2)
        if endpoint_1 is None or endpoint_2 is None:
            continue
        if endpoint_1.atom_ref == endpoint_2.atom_ref:
            continue
        destination = source_connections
        if (
            file_format is FileFormat.PDB
            and endpoint_1.atom_ref.residue_id == endpoint_2.atom_ref.residue_id
        ):
            if local_pdb_links is None:
                continue
            destination = local_pdb_links

        relationship_type = _relationship_type_from_connection(connection.type)
        destination.append(
            SourceConnection(
                endpoint_1=endpoint_1,
                endpoint_2=endpoint_2,
                source_metadata=SourceBondMetadata(
                    reported_relationship_type=relationship_type,
                    record_type=_source_connection_record_type(
                        file_format=file_format,
                        relationship_type=relationship_type,
                    ),
                    source_id=connection.link_id or connection.name,
                    reported_distance_angstrom=_normalize_reported_connection_distance(
                        connection.reported_distance
                    ),
                    reported_order=None
                    if orders_by_id is None
                    else orders_by_id.get(connection.name),
                ),
            )
        )

    return tuple(dict.fromkeys(source_connections))


def _mmcif_connection_orders(document: gemmi.cif.Document) -> dict[str, int | None]:
    """Read order evidence that Gemmi's Connection object does not expose."""
    orders: dict[str, int | None] = {}
    if not document:
        return orders
    tokens = {"sing": 1, "doub": 2, "trip": 3, "quad": 4}
    for row in document[0].find("_struct_conn.", ["id", "?pdbx_value_order"]):
        source_id = gemmi.cif.as_string(row[0])
        raw_token = row[1] if row.has(1) else "?"
        if raw_token in {"?", "."}:
            order = None
        else:
            token = gemmi.cif.as_string(raw_token).lower()
            order = tokens.get(token)
            if order is None:
                raise StructureNormalizationError(
                    f"unsupported struct_conn bond order: {token!r}"
                )
        if source_id in orders and orders[source_id] != order:
            raise StructureNormalizationError(
                f"conflicting struct_conn orders for id {source_id!r}"
            )
        orders[source_id] = order
    return orders


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
    *,
    atom_identity_by_serial: dict[int, SourceAtomIdentity],
    observe_cap_paths: bool,
    observe_sequence_ports: bool,
    duplicate_ports: dict[int, tuple[AtomRef, ...]],
    include_sidechains: bool = False,
) -> tuple[
    tuple[SourceConnection, ...],
    frozenset[tuple[SourceAtomIdentity, SourceAtomIdentity]],
    frozenset[SourceAtomIdentity],
    frozenset[AtomRef],
]:
    """Decode CONECT and retain model and serial facts for cap qualification."""

    connections_by_pair: dict[
        tuple[SourceAtomIdentity, SourceAtomIdentity], SourceConnection
    ] = {}
    directed_counts: Counter[tuple[int, int]] = Counter()
    seen_rows: set[tuple[int, tuple[int, ...]]] = set()
    unbound_atoms: set[SourceAtomIdentity] = set()
    first_model_serial_pairs: set[tuple[int, int]] = set()
    unresolved_ports: set[AtomRef] = set()
    explicit_model_count = 0
    in_later_model = False
    for line in contents.splitlines():
        if observe_cap_paths or observe_sequence_ports:
            if line.startswith("MODEL"):
                explicit_model_count += 1
                in_later_model = explicit_model_count > 1
            elif line.startswith("ENDMDL"):
                in_later_model = False
        if not line.startswith("CONECT"):
            continue

        source, targets = _pdb_conect_serials(
            line, observe_blank_source=observe_cap_paths or observe_sequence_ports
        )
        unbound_serial = (observe_cap_paths or observe_sequence_ports) and (
            source is None
            or source not in atom_identity_by_serial
            or any(target not in atom_identity_by_serial for target in targets)
        )
        if unbound_serial or (
            observe_sequence_ports and (not targets or source in targets)
        ):
            identifiable = tuple(
                atom_identity_by_serial[serial]
                for serial in (source, *targets)
                if serial is not None and serial in atom_identity_by_serial
            )
            if observe_cap_paths and targets and unbound_serial:
                unbound_atoms.update(identifiable)
            if observe_sequence_ports and not in_later_model:
                unresolved_ports.update(
                    atom.atom_ref
                    for atom in identifiable
                    if include_sidechains or atom.atom_ref.atom_name in {"C", "N"}
                )
                unresolved_ports.update(
                    ref
                    for serial in (source, *targets)
                    if serial is not None
                    for ref in duplicate_ports.get(serial, ())
                    if include_sidechains or ref.atom_name in {"C", "N"}
                )
        if not targets or source is None or source not in atom_identity_by_serial:
            continue
        if observe_cap_paths and not in_later_model:
            first_model_serial_pairs.update((source, target) for target in targets)
        row = (source, tuple(sorted(targets)))
        if row in seen_rows:
            continue
        seen_rows.add(row)
        directed_counts.update((source, target) for target in targets)

    first_model_pairs: set[tuple[SourceAtomIdentity, SourceAtomIdentity]] = set()
    for (source_serial, target_serial), multiplicity in directed_counts.items():
        source_identity = atom_identity_by_serial[source_serial]
        target_identity = atom_identity_by_serial.get(target_serial)
        if (
            target_identity is None
            or target_identity.atom_ref == source_identity.atom_ref
        ):
            continue
        if multiplicity > 4:
            raise StructureNormalizationError(
                "CONECT bond order exceeds quadruple multiplicity"
            )
        connection = SourceConnection(
            endpoint_1=source_identity,
            endpoint_2=target_identity,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
                source_id="CONECT",
                reported_order=multiplicity if multiplicity > 1 else None,
            ),
        )
        pair = connection.endpoint_pair()
        if (source_serial, target_serial) in first_model_serial_pairs:
            first_model_pairs.add(pair)
        existing = connections_by_pair.get(pair)
        connections_by_pair[pair] = (
            connection if existing is None else existing.merge(connection)
        )

    return (
        tuple(connections_by_pair.values()),
        frozenset(first_model_pairs),
        frozenset(unbound_atoms),
        frozenset(unresolved_ports),
    )


def _first_model_pdb_atom_identities(
    contents: str,
) -> tuple[dict[int, SourceAtomIdentity], dict[int, tuple[AtomRef, ...]]]:
    """Separate usable first-model serials from possible ports on duplicate serials.

    CONECT is lowered onto the canonical first-model structure, so serial reuse
    in later MODEL sections does not make a first-model serial ambiguous. After
    first-model ENDMDL, atom records are treated as outside model one even when
    malformed trailing atoms appear without another MODEL record.
    """

    records_by_serial: dict[int, list[SourceAtomIdentity]] = {}
    for line in _first_model_pdb_coordinate_records(contents):
        atom_serial_and_identity = _pdb_atom_serial_and_identity(line)
        if atom_serial_and_identity is None:
            continue

        serial, identity = atom_serial_and_identity
        records_by_serial.setdefault(serial, []).append(identity)

    unique = {
        serial: records[0]
        for serial, records in records_by_serial.items()
        if len(records) == 1
    }
    return unique, {
        serial: tuple(r.atom_ref for r in records)
        for serial, records in records_by_serial.items()
        if len(records) > 1
    }


def _pdb_source_atom_attributes(
    contents: str,
    *,
    prove_zero_occupancy: bool,
    prove_positive_occupancy: bool,
    observe_cap_paths: bool,
    read_sequence_slots: bool,
) -> tuple[
    dict[SourceAtomIdentity, str],
    dict[SourceAtomIdentity, list[SourceAtomAnnotation]],
    frozenset[ResidueId],
    frozenset[tuple[ResidueId, ResidueId]],
    list[PdbSequenceSlot],
]:
    """Read source atom annotations and TER boundaries before projection."""

    source_elements: dict[SourceAtomIdentity, str] = {}
    source_annotations: dict[SourceAtomIdentity, list[SourceAtomAnnotation]] = {}
    peptide_break_after: set[ResidueId] = set()
    previous_residue: ResidueId | None = None
    source_slots: list[tuple[ResidueId, str, str]] = []
    previous_slot: tuple[ResidueId, str, str] | None = None
    sequence_slots: list[PdbSequenceSlot] = []
    sequence_key: tuple[ResidueId, str, str, bool] | None = None
    unaddressable_chains: set[str] = set()
    for line in _first_model_pdb_coordinate_records(contents):
        if line[:6].strip() == "TER":
            if previous_residue is not None:
                peptide_break_after.add(previous_residue)
            previous_slot = None
            sequence_key = None
            continue
        identity = pdb_atom_identity(line)
        if identity is None:
            if read_sequence_slots:
                unaddressable_chains.add(normalize_chain_id(line[20:22]))
            continue
        previous_residue = identity.atom_ref.residue_id
        if read_sequence_slots:
            key = (
                previous_residue,
                identity.component_id,
                line[72:76].strip(),
                line.startswith("HETATM"),
            )
            if key != sequence_key:
                sequence_slots.append(PdbSequenceSlot(*key))
                sequence_key = key
            if sequence_slots[-1].hetero and identity.atom_ref.atom_name in {
                "N",
                "CA",
                "C",
            }:
                sequence_slots[-1].backbone_names.add(identity.atom_ref.atom_name)
        if observe_cap_paths:
            slot = (previous_residue, identity.component_id, line[72:76].strip())
            if slot != previous_slot:
                source_slots.append(slot)
                previous_slot = slot
        token = line[78:80].strip()
        if not token:
            charge = None
        elif len(token) == 2 and token[0] in "0123456789" and token[1] in "+-":
            charge = int(token[0]) * (-1 if token[1] == "-" else 1)
        else:
            raise StructureNormalizationError(f"invalid PDB formal charge {token!r}")
        source_symbol = line[76:78].strip().upper()
        if prove_zero_occupancy:
            occupancy = line[54:60].strip() if len(line) >= 60 else ""
            annotation = _occupancy_annotation(
                occupancy,
                _PDB_OCCUPANCY_TOKEN,
                identity,
                formal_charge=charge,
                declared_element=source_symbol,
                prove_positive=prove_positive_occupancy,
            )
        else:
            annotation = (
                _UNANNOTATED_ATOM
                if charge is None
                else SourceAtomAnnotation(formal_charge=charge)
            )
        source_annotations.setdefault(identity, []).append(annotation)

        if source_symbol not in {"D", "T"}:
            continue
        _record_source_isotope_element(
            source_elements,
            identity,
            source_symbol,
        )

    slot_counts = Counter(residue_id for residue_id, _, _ in source_slots)
    adjacent_slots = frozenset(
        (first, second)
        for (first, _, first_segment), (
            second,
            _,
            second_segment,
        ) in itertools.pairwise(source_slots)
        if first.immediately_precedes(second)
        and first not in peptide_break_after
        and first_segment == second_segment
        and slot_counts[first] == slot_counts[second] == 1
    )
    return (
        source_elements,
        source_annotations,
        frozenset(peptide_break_after),
        adjacent_slots,
        [
            slot
            for slot in sequence_slots
            if slot.residue_id.chain_id not in unaddressable_chains
        ],
    )


def _mmcif_source_atom_attributes(
    document: gemmi.cif.Document,
    *,
    selected_model: int | None,
    prove_zero_occupancy: bool,
    prove_whole_residue_omissions: bool,
) -> tuple[
    dict[SourceAtomIdentity, str],
    dict[SourceAtomIdentity, list[SourceAtomAnnotation]],
    dict[ResidueId, frozenset[SourcePolymerLabels]],
    tuple[tuple[DepositedPeptideBoundary, ...], tuple[DepositedPeptideSequence, ...]],
]:
    """Read first-model atom annotations and uncollapsed polymer labels."""

    if len(document) == 0:
        return {}, {}, {}, ((), ())

    block = document[0]
    type_symbols = tuple(block.find_values("_atom_site.type_symbol"))
    if not type_symbols:
        return {}, {}, {}, ((), ())

    row_count = len(type_symbols)
    charge_tokens = _coalesced_mmcif_column(
        block,
        ("_atom_site.pdbx_formal_charge",),
        row_count=row_count,
        required=False,
    )
    occupancy_tokens = (
        _coalesced_mmcif_column(
            block, ("_atom_site.occupancy",), row_count=row_count, required=False
        )
        if prove_zero_occupancy
        else None
    )
    source_label_columns = (
        tuple(
            _coalesced_mmcif_column(block, (tag,), row_count=row_count, required=False)
            for tag in (
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
            )
        )
        if prove_whole_residue_omissions
        or bool(block.find_values("_atom_site.label_seq_id"))
        or bool(block.find_values("_entity_poly_seq.mon_id"))
        else None
    )
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
    label_components = (
        block.find_values("_atom_site.label_comp_id")
        if selected_model is not None and block.find_values("_entity_poly_seq.mon_id")
        else None
    )
    if label_components and block.find_values("_atom_site.pdbx_PDB_model_num"):
        try:
            if int(first_model_number) != selected_model:
                label_components = None
        except ValueError:
            label_components = None
    sequence_slots: list[MmcifSequenceSlot] = []
    unaddressable_chains: set[str] = set()

    source_elements: dict[SourceAtomIdentity, str] = {}
    source_annotations: dict[SourceAtomIdentity, list[SourceAtomAnnotation]] = {}
    source_labels: dict[ResidueId, set[SourcePolymerLabels]] = {}
    for row_index, source_symbol in enumerate(type_symbols):
        normalized_source_symbol = _non_null_cif_value(source_symbol).upper()
        model = _non_null_cif_value(model_numbers[row_index])
        if first_model_number and model != first_model_number:
            if label_components:
                try:
                    ambiguous_model = int(model) == selected_model
                except ValueError:
                    ambiguous_model = True
                if ambiguous_model:
                    unaddressable_chains.add(
                        normalize_chain_id(_non_null_cif_value(chain_ids[row_index]))
                    )
            continue

        atom_name = _non_null_cif_value(atom_names[row_index])
        component_id = _non_null_cif_value(component_ids[row_index])
        chain_id = _non_null_cif_value(chain_ids[row_index])
        sequence_number = _non_null_cif_value(sequence_numbers[row_index])
        if not atom_name or not component_id or not sequence_number:
            if label_components:
                unaddressable_chains.add(normalize_chain_id(chain_id))
            continue
        try:
            seq_num = int(sequence_number)
        except ValueError:
            if label_components:
                unaddressable_chains.add(normalize_chain_id(chain_id))
            continue

        residue_id = ResidueId(
            chain_id=normalize_chain_id(chain_id),
            seq_num=seq_num,
            insertion_code=normalize_insertion_code(
                _non_null_cif_value(insertion_codes[row_index])
            ),
        )
        identity = SourceAtomIdentity(
            atom_ref=AtomRef(residue_id, atom_name),
            component_id=component_id,
            altloc=normalize_altloc(_non_null_cif_value(altlocs[row_index])),
        )
        if source_label_columns is not None:
            subchain_id, entity_id, position_token = (
                _non_null_cif_value(column[row_index])
                for column in source_label_columns
            )
            labels = SourcePolymerLabels(subchain_id, entity_id, position_token)
            if subchain_id or entity_id or position_token:
                source_labels.setdefault(residue_id, set()).add(labels)
            if label_components:
                row = MmcifSequenceSlot(
                    residue_id,
                    component_id,
                    _non_null_cif_value(label_components[row_index]),
                    labels,
                )
                if not sequence_slots or sequence_slots[-1] != row:
                    sequence_slots.append(row)

        token = _non_null_cif_value(charge_tokens[row_index])
        try:
            charge = None if not token else int(token)
        except ValueError as error:
            raise StructureNormalizationError(
                f"invalid mmCIF formal charge {token!r}"
            ) from error
        if occupancy_tokens is not None:
            occupancy = _non_null_cif_value(occupancy_tokens[row_index])
            occupancy = "" if occupancy in {"?", "."} else occupancy
            annotation = _occupancy_annotation(
                occupancy,
                _CIF_OCCUPANCY_TOKEN,
                identity,
                formal_charge=charge,
                declared_element=normalized_source_symbol,
                prove_positive=prove_whole_residue_omissions,
            )
        else:
            annotation = (
                _UNANNOTATED_ATOM
                if charge is None
                else SourceAtomAnnotation(formal_charge=charge)
            )
        source_annotations.setdefault(identity, []).append(annotation)
        if normalized_source_symbol in {"D", "T"}:
            _record_source_isotope_element(
                source_elements, identity, normalized_source_symbol
            )

    return (
        source_elements,
        source_annotations,
        {residue_id: frozenset(labels) for residue_id, labels in source_labels.items()},
        mmcif_peptide_paths(
            block, sequence_slots, unaddressable_chains=frozenset(unaddressable_chains)
        )
        if sequence_slots
        else ((), ()),
    )


def _occupancy_annotation(
    token: str,
    token_pattern: re.Pattern[str],
    identity: SourceAtomIdentity,
    *,
    formal_charge: int | None,
    declared_element: str,
    prove_positive: bool,
) -> SourceAtomAnnotation:
    """Test the source coefficient without rounding the value to native precision."""

    zero = positive = False
    if token:
        match = token_pattern.fullmatch(token)
        if match is None:
            raise StructureNormalizationError(
                f"invalid source occupancy {token!r} at {identity.atom_ref.display_token()}"
            )
        zero = all(digit in "0." for digit in match["coefficient"])
        positive = prove_positive and not zero and not token.startswith("-")
    return SourceAtomAnnotation(
        formal_charge=formal_charge,
        explicitly_zero_occupancy=zero,
        declared_element=declared_element,
        explicitly_positive_occupancy=positive,
    )


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
                if _non_null_cif_value(value := column[row_index])
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

    return gemmi.cif.as_string(value).strip()


def _first_model_pdb_coordinate_records(contents: str) -> Iterator[str]:
    """Yield atom and TER records belonging to the first PDB model."""

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
        if line.startswith(("ATOM  ", "HETATM")) or line[:6].strip() == "TER":
            yield line


def _pdb_atom_serial_and_identity(
    line: str,
) -> tuple[int, SourceAtomIdentity] | None:
    """Return the PDB atom serial and source identity for one atom line."""

    if not line.startswith(("ATOM  ", "HETATM")):
        return None

    try:
        serial = decode_pdb_integer(line[6:11], width=5)
    except ValueError as error:
        raise StructureNormalizationError(
            f"invalid PDB atom serial {line[6:11]!r}: {error}"
        ) from error
    try:
        identity = pdb_atom_identity(line)
    except ValueError as error:
        raise StructureNormalizationError(
            f"invalid PDB residue number {line[22:26]!r}: {error}"
        ) from error
    return None if identity is None else (serial, identity)


def _pdb_conect_serials(
    line: str, *, observe_blank_source: bool
) -> tuple[int | None, tuple[int, ...]]:
    """Decode CONECT, optionally retaining targets of a blank source field."""

    source = None
    targets: list[int] = []
    # Accept RDKit's extended fifth/sixth neighbor fields through column 41.
    for offset in range(6, min(len(line), 41), 5):
        field = line[offset : offset + 5]
        if not field.strip():
            if offset == 6 and not observe_blank_source:
                return None, ()
            continue
        try:
            serial = decode_pdb_integer(field.ljust(5), width=5)
        except ValueError as error:
            # A blank source row was ignored by the decoder. Its parseable
            # targets are useful as a veto, but it still cannot declare a bond.
            if offset != 6 and source is None:
                continue
            raise StructureNormalizationError(
                f"invalid PDB CONECT serial {field!r}: {error}"
            ) from error
        if offset == 6:
            source = serial
        else:
            targets.append(serial)

    return source, tuple(targets)
