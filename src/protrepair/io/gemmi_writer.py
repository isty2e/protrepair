"""gemmi-backed writers that project canonical structures to coordinate text."""

import os
import secrets
from os import PathLike
from pathlib import Path

from protrepair.chemistry.nonstandard.registry import (
    build_bundled_nonstandard_registry,
)
from protrepair.chemistry.standard.components import (
    build_standard_component_library,
)
from protrepair.errors import ModelInvariantError, UnsupportedFileFormatError
from protrepair.io.gemmi_normalization import (
    gemmi,
    infer_file_format,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondRecordType,
    TopologyBond,
    is_covalent_like_relationship,
    is_model_resolved_provenance,
)


def write_structure(
    structure: ProteinStructure,
    output_path: Path | str | PathLike[str],
    *,
    file_format: FileFormat | None = None,
) -> None:
    """Serialize a canonical structure to a path-like coordinate file."""

    target_path = Path(output_path)
    resolved_format = (
        infer_file_format(target_path) if file_format is None else file_format
    )
    serialized_structure = write_structure_string(structure, resolved_format)
    _atomic_write_text(target_path, serialized_structure)


def _atomic_write_text(output_path: Path, text: str) -> None:
    """Write text through a same-directory temporary file and atomic replace."""

    temp_path: Path | None = None
    file_descriptor: int | None = None
    try:
        temp_path, file_descriptor = _open_same_directory_temp_file(output_path)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = None
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        _preserve_existing_output_mode(temp_path, output_path)
        temp_path.replace(output_path)
    except BaseException:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temp_path is not None:
            _remove_temporary_output(temp_path)
        raise


def _open_same_directory_temp_file(output_path: Path) -> tuple[Path, int]:
    """Create one exclusive temporary file next to the output path."""

    parent = output_path.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"output directory does not exist for {output_path}: {parent}"
        )
    if not parent.is_dir():
        raise NotADirectoryError(
            f"output parent is not a directory for {output_path}: {parent}"
        )

    for _ in range(100):
        temp_path = parent / f".{output_path.name}.{secrets.token_hex(8)}.tmp"
        try:
            file_descriptor = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
        except FileExistsError:
            continue

        return temp_path, file_descriptor

    raise FileExistsError(
        f"could not allocate an unused temporary output path for {output_path}"
    )


def _remove_temporary_output(temp_path: Path) -> None:
    """Best-effort cleanup for temporary output files."""

    try:
        temp_path.unlink()
    except OSError:
        pass


def _preserve_existing_output_mode(temp_path: Path, output_path: Path) -> None:
    """Keep existing target permissions across atomic replacement."""

    try:
        output_mode = output_path.stat().st_mode & 0o777
    except FileNotFoundError:
        return

    temp_path.chmod(output_mode)


def write_structure_string(structure: ProteinStructure, file_format: FileFormat) -> str:
    """Serialize a canonical structure to a coordinate string."""

    raw_structure = build_gemmi_structure(structure)
    if file_format is FileFormat.PDB:
        add_topology_connections_to_gemmi_structure(
            raw_structure,
            structure,
            bonds=pdb_typed_connection_topology_bonds_for_egress(structure),
        )
        pdb_text = _restore_pdb_isotope_element_symbols(
            raw_structure.make_pdb_string(),
            structure,
        )
        return append_pdb_conect_records_from_topology(
            pdb_text,
            structure,
        )

    if file_format is FileFormat.MMCIF:
        add_topology_connections_to_gemmi_structure(
            raw_structure,
            structure,
            bonds=gemmi_connection_topology_bonds_for_egress(
                structure,
                include_model_resolved=True,
            ),
        )
        mmcif_document = raw_structure.make_mmcif_document()
        _restore_mmcif_isotope_element_symbols(mmcif_document, structure)
        return mmcif_document.as_string()

    raise UnsupportedFileFormatError(f"unsupported file format: {file_format}")


def write_pdb_structure_string_without_conect(structure: ProteinStructure) -> str:
    """Serialize a canonical structure to PDB text without CONECT records."""

    raw_structure = build_gemmi_structure(structure)
    add_topology_connections_to_gemmi_structure(
        raw_structure,
        structure,
        bonds=pdb_typed_connection_topology_bonds_for_egress(structure),
    )
    return _restore_pdb_isotope_element_symbols(
        raw_structure.make_pdb_string(),
        structure,
    )


def _restore_pdb_isotope_element_symbols(
    pdb_text: str,
    structure: ProteinStructure,
) -> str:
    """Restore isotope aliases that Gemmi cannot represent as elements."""

    if not any(
        atom_site.element_identity.is_isotope_alias()
        for atom_site in structure.constitution.atom_slots
    ):
        return pdb_text

    lines = pdb_text.splitlines(keepends=True)
    coordinate_line_indices = tuple(
        index
        for index, line in enumerate(lines)
        if line.startswith(("ATOM  ", "HETATM"))
    )
    atom_index_values = atom_index_values_in_coordinate_write_order(structure)
    if len(coordinate_line_indices) != len(atom_index_values):
        raise ModelInvariantError(
            "PDB isotope restoration requires one coordinate line per atom slot"
        )

    for line_index, atom_index_value in zip(
        coordinate_line_indices,
        atom_index_values,
        strict=True,
    ):
        atom_site = structure.constitution.atom_site_at(AtomIndex(atom_index_value))
        if not atom_site.element_identity.is_isotope_alias():
            continue

        line = lines[line_index]
        line_ending = line[len(line.rstrip("\r\n")) :]
        record = line.removesuffix(line_ending).ljust(78)
        lines[line_index] = (
            record[:76]
            + f"{atom_site.element_identity.source_symbol:>2}"
            + record[78:]
            + line_ending
        )

    return "".join(lines)


def _restore_mmcif_isotope_element_symbols(
    document: gemmi.cif.Document,
    structure: ProteinStructure,
) -> None:
    """Restore source isotope aliases in one generated mmCIF document."""

    if not any(
        atom_site.element_identity.is_isotope_alias()
        for atom_site in structure.constitution.atom_slots
    ):
        return

    atom_index_values = atom_index_values_in_coordinate_write_order(structure)
    if not atom_index_values:
        return

    block = document.sole_block()
    atom_site_type_symbols = block.find_loop("_atom_site.type_symbol")
    if len(atom_site_type_symbols) != len(atom_index_values):
        raise ModelInvariantError(
            "mmCIF isotope restoration requires one type symbol per atom slot"
        )

    for row_index, atom_index_value in enumerate(atom_index_values):
        atom_site = structure.constitution.atom_site_at(AtomIndex(atom_index_value))
        if atom_site.element_identity.is_isotope_alias():
            atom_site_type_symbols[row_index] = (
                atom_site.element_identity.source_symbol
            )

    atom_type_symbols = block.find_loop("_atom_type.symbol")
    if not atom_type_symbols:
        raise ModelInvariantError(
            "mmCIF isotope restoration requires an atom-type symbol loop"
        )
    atom_type_symbols.get_loop().set_all_values(
        (tuple(dict.fromkeys(atom_site_type_symbols)),)
    )


def build_gemmi_structure(
    structure: ProteinStructure,
) -> gemmi.Structure:
    """Project the canonical structure model into a gemmi structure."""

    raw_structure = gemmi.Structure()
    raw_structure.name = structure.provenance.ingress.source_name or "protrepair"
    model = gemmi.Model(1)
    for chain_id, residues in residues_by_chain_id(structure):
        raw_chain = gemmi.Chain(chain_id)
        for residue_site in residues:
            residue_index = structure.constitution.residue_index(
                residue_site.residue_id
            )
            residue_geometry = structure.residue_geometry(residue_index)
            raw_chain.add_residue(
                build_gemmi_residue(
                    residue_site,
                    residue_geometry=residue_geometry,
                    formal_charge_by_atom_name=(
                        structure.residue_formal_charge_by_atom_name(residue_index)
                    ),
                )
            )

        model.add_chain(raw_chain)

    raw_structure.add_model(model)
    raw_structure.setup_entities()
    raw_structure.assign_label_seq_id()
    raw_structure.assign_subchains()
    raw_structure.assign_serial_numbers()
    return raw_structure


def add_topology_connections_to_gemmi_structure(
    raw_structure: gemmi.Structure,
    structure: ProteinStructure,
    *,
    bonds: tuple[TopologyBond, ...],
) -> None:
    """Add topology bonds as gemmi connection records."""

    for bond in bonds:
        raw_connection = gemmi.Connection()
        raw_connection.name = source_connection_name(bond)
        raw_connection.link_id = source_connection_link_id(bond)
        raw_connection.type = gemmi_connection_type(bond.relationship_type)
        if (
            bond.source_metadata is not None
            and bond.source_metadata.reported_distance_angstrom is not None
        ):
            raw_connection.reported_distance = (
                bond.source_metadata.reported_distance_angstrom
            )

        populate_gemmi_connection_partner(
            raw_connection.partner1,
            structure,
            bond.atom_index_1,
        )
        populate_gemmi_connection_partner(
            raw_connection.partner2,
            structure,
            bond.atom_index_2,
        )
        raw_structure.connections.append(raw_connection)


def source_explicit_topology_bonds_for_egress(
    structure: ProteinStructure,
) -> tuple[TopologyBond, ...]:
    """Return all source-explicit topology bonds.

    This is a source-preservation projection, not a concrete writer policy.
    Format-specific writers should use gemmi_connection_topology_bonds_for_egress
    or pdb_conect_topology_bonds_for_egress because PDB and mmCIF preserve
    source-explicit relationships through different boundary records.
    """

    return tuple(
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )


def gemmi_connection_topology_bonds_for_egress(
    structure: ProteinStructure,
    *,
    include_model_resolved: bool,
    include_pdb_conect_origin: bool = True,
) -> tuple[TopologyBond, ...]:
    """Return topology bonds emitted through gemmi connection records.

    Gemmi connection records preserve typed relationships such as LINK,
    struct_conn, disulfide, hydrogen, and metal coordination. Callers that
    append PDB CONECT separately exclude bonds originating from that untyped
    connectivity table.
    """

    return tuple(
        bond
        for bond in structure.topology.bonds
        if _topology_bond_is_gemmi_connection_emittable(
            bond,
            include_model_resolved=include_model_resolved,
        )
        and (
            include_pdb_conect_origin
            or bond.source_metadata is None
            or bond.source_metadata.record_type is not SourceBondRecordType.PDB_CONECT
        )
    )


def pdb_typed_connection_topology_bonds_for_egress(
    structure: ProteinStructure,
) -> tuple[TopologyBond, ...]:
    """Return source bonds represented by typed PDB connection records."""

    return tuple(
        bond
        for bond in gemmi_connection_topology_bonds_for_egress(
            structure,
            include_model_resolved=False,
            include_pdb_conect_origin=False,
        )
        if bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
    )


def _topology_bond_is_gemmi_connection_emittable(
    bond: TopologyBond,
    *,
    include_model_resolved: bool,
) -> bool:
    """Return whether one topology bond may become a gemmi connection."""

    if bond.provenance is BondProvenance.SOURCE_EXPLICIT:
        return True

    return include_model_resolved and _topology_bond_is_model_resolved_emittable(bond)


def source_connection_name(bond: TopologyBond) -> str:
    """Return a stable connection identifier for one source topology bond."""

    source_metadata = bond.source_metadata
    if (
        source_metadata is not None
        and source_metadata.source_id is not None
        and _source_metadata_id_is_specific(bond)
    ):
        return source_metadata.source_id

    return f"protrepair_{bond.atom_index_1.value}_{bond.atom_index_2.value}"


def source_connection_link_id(bond: TopologyBond) -> str:
    """Return the best boundary link identifier for one source topology bond."""

    source_metadata = bond.source_metadata
    if (
        source_metadata is not None
        and source_metadata.source_id is not None
        and _source_metadata_id_is_specific(bond)
    ):
        return source_metadata.source_id

    return source_connection_name(bond)


def populate_gemmi_connection_partner(
    raw_partner: gemmi.AtomAddress,
    structure: ProteinStructure,
    atom_index: AtomIndex,
) -> None:
    """Populate one gemmi connection partner from a canonical atom slot."""

    atom_ref = structure.constitution.atom_ref_at(atom_index)
    residue_site = structure.constitution.residue_site_at(
        structure.constitution.residue_index(atom_ref.residue_id)
    )
    raw_partner.chain_name = atom_ref.residue_id.chain_id
    raw_partner.res_id = gemmi.ResidueId()
    raw_partner.res_id.name = residue_site.component_id
    raw_partner.res_id.seqid = gemmi.SeqId(
        atom_ref.residue_id.seq_num,
        atom_ref.residue_id.insertion_code or " ",
    )
    raw_partner.atom_name = atom_ref.atom_name
    atom_geometry = structure.geometry.atom_geometry(atom_index)
    raw_partner.altloc = "\0" if atom_geometry.altloc is None else atom_geometry.altloc


def gemmi_connection_type(
    relationship_type: BondRelationshipType,
) -> gemmi.ConnectionType:
    """Return the gemmi connection type for one canonical relationship."""

    if relationship_type is BondRelationshipType.COVALENT:
        return gemmi.ConnectionType.Covale
    if relationship_type is BondRelationshipType.DISULFIDE:
        return gemmi.ConnectionType.Disulf
    if relationship_type is BondRelationshipType.HYDROGEN_BOND:
        return gemmi.ConnectionType.Hydrog
    if relationship_type is BondRelationshipType.METAL_COORDINATION:
        return gemmi.ConnectionType.MetalC

    return gemmi.ConnectionType.Unknown


def append_pdb_conect_records_from_topology(
    pdb_text: str,
    structure: ProteinStructure,
) -> str:
    """Append PDB CONECT records projected from canonical topology."""

    serial_by_atom_ref = pdb_atom_serial_by_atom_ref(pdb_text)
    neighbor_serials_by_source: dict[int, set[int]] = {}
    for bond in pdb_conect_topology_bonds_for_egress(structure):
        atom_ref_1 = structure.constitution.atom_ref_at(bond.atom_index_1)
        atom_ref_2 = structure.constitution.atom_ref_at(bond.atom_index_2)
        serial_1 = serial_by_atom_ref.get(atom_ref_1)
        serial_2 = serial_by_atom_ref.get(atom_ref_2)
        if serial_1 is None or serial_2 is None:
            continue

        neighbor_serials_by_source.setdefault(serial_1, set()).add(serial_2)
        neighbor_serials_by_source.setdefault(serial_2, set()).add(serial_1)

    conect_lines = tuple(
        line
        for source_serial, neighbor_serials in sorted(
            neighbor_serials_by_source.items()
        )
        for line in format_pdb_conect_lines(
            source_serial,
            tuple(sorted(neighbor_serials)),
        )
    )
    if not conect_lines:
        return pdb_text

    conect_block = "\n".join(conect_lines)
    if "\nEND" in pdb_text:
        return pdb_text.replace("\nEND", f"\n{conect_block}\nEND", 1)

    return pdb_text.rstrip("\n") + "\n" + conect_block + "\n"


def pdb_conect_topology_bonds_for_egress(
    structure: ProteinStructure,
) -> tuple[TopologyBond, ...]:
    """Return topology bonds that should be represented as PDB CONECT.

    PDB CONECT cannot encode relationship type, so this projection preserves
    source PDB CONECT records and emits covalent-like repaired/model-resolved
    bonds. Typed source relationships such as metal coordination stay in gemmi
    connection records unless they came from PDB CONECT.
    """

    return tuple(
        bond
        for bond in structure.topology.bonds
        if _topology_bond_is_pdb_conect_emittable(bond)
    )


def _topology_bond_is_pdb_conect_emittable(bond: TopologyBond) -> bool:
    """Return whether one topology bond may be serialized as PDB CONECT."""

    if bond.provenance is BondProvenance.SOURCE_EXPLICIT:
        if (
            bond.source_metadata is not None
            and bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
        ):
            return True

        return is_covalent_like_relationship(bond)

    return _topology_bond_is_model_resolved_emittable(bond)


def _topology_bond_is_model_resolved_emittable(bond: TopologyBond) -> bool:
    """Return whether one model-resolved topology bond may be serialized."""

    return is_model_resolved_provenance(bond) and is_covalent_like_relationship(bond)


def _source_metadata_id_is_specific(bond: TopologyBond) -> bool:
    """Return whether source metadata carries a boundary-specific identifier."""

    if bond.source_metadata is None or bond.source_metadata.source_id is None:
        return False
    return not (
        bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
        and bond.source_metadata.source_id == "CONECT"
    )


def pdb_atom_serial_by_atom_ref(pdb_text: str) -> dict[AtomRef, int]:
    """Return assigned PDB atom serials keyed by canonical atom reference."""

    serial_by_atom_ref: dict[AtomRef, int] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue

        serial_token = line[6:11].strip()
        atom_name = line[12:16].strip()
        chain_id = line[21:22].strip()
        seq_num_token = line[22:26].strip()
        if not serial_token or not atom_name or not chain_id or not seq_num_token:
            continue

        try:
            serial = int(serial_token)
            seq_num = int(seq_num_token)
        except ValueError:
            continue

        insertion_code = line[26:27].strip() or None
        serial_by_atom_ref[
            AtomRef(
                residue_id=ResidueId(
                    chain_id=chain_id,
                    seq_num=seq_num,
                    insertion_code=insertion_code,
                ),
                atom_name=atom_name,
            )
        ] = serial

    return serial_by_atom_ref


def atom_index_values_in_coordinate_write_order(
    structure: ProteinStructure,
) -> tuple[int, ...]:
    """Return canonical atom indices in coordinate serialization order."""

    atom_index_values: list[int] = []
    for _chain_id, residues in residues_by_chain_id(structure):
        for residue_site in residues:
            residue_index = structure.constitution.residue_index(
                residue_site.residue_id
            )
            atom_index_values.extend(
                structure.constitution.atom_index_in_residue(
                    residue_index,
                    atom_site.name,
                ).value
                for atom_site in residue_site.atom_sites
            )

    return tuple(atom_index_values)


def format_pdb_conect_lines(
    source_serial: int,
    neighbor_serials: tuple[int, ...],
) -> tuple[str, ...]:
    """Return one or more fixed-width PDB CONECT records for one source atom."""

    return tuple(
        f"CONECT{source_serial:>5}"
        + "".join(f"{neighbor_serial:>5}" for neighbor_serial in neighbor_chunk)
        for neighbor_chunk in _chunks(neighbor_serials, size=4)
    )


def _chunks(values: tuple[int, ...], *, size: int) -> tuple[tuple[int, ...], ...]:
    """Return fixed-size chunks of one tuple."""

    return tuple(values[index : index + size] for index in range(0, len(values), size))


def build_gemmi_residue(
    residue_site: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...],
) -> gemmi.Residue:
    """Project a canonical residue into a gemmi residue."""

    raw_residue = gemmi.Residue()
    raw_residue.name = residue_site.component_id
    raw_residue.seqid = gemmi.SeqId(
        residue_site.residue_id.seq_num,
        residue_site.residue_id.insertion_code or " ",
    )
    raw_residue.het_flag = _gemmi_het_flag_for_residue(residue_site)
    raw_residue.entity_type = (
        gemmi.EntityType.NonPolymer
        if residue_site.is_hetero
        else gemmi.EntityType.Polymer
    )

    residue_formal_charge_by_atom_name = dict(formal_charge_by_atom_name)
    for atom_site in residue_site.atom_sites:
        raw_residue.add_atom(
            build_gemmi_atom(
                atom_site,
                atom_geometry=residue_geometry.atom_geometry(atom_site.name),
                formal_charge=residue_formal_charge_by_atom_name.get(atom_site.name),
            )
        )

    return raw_residue


def _gemmi_het_flag_for_residue(residue_site: ResidueSite) -> str:
    """Return canonical PDB/mmCIF record spelling independently of entity role."""

    if residue_site.is_hetero:
        return "H"
    if build_standard_component_library().has(residue_site.component_id):
        return "A"
    if build_bundled_nonstandard_registry().get(residue_site.component_id) is not None:
        return "H"
    return "A"


def build_gemmi_atom(
    atom_site: AtomSite,
    *,
    atom_geometry: AtomGeometry,
    formal_charge: int | None,
) -> gemmi.Atom:
    """Project a canonical atom into a gemmi atom."""

    raw_atom = gemmi.Atom()
    raw_atom.name = atom_site.name
    raw_atom.altloc = "\x00" if atom_geometry.altloc is None else atom_geometry.altloc
    raw_atom.charge = 0 if formal_charge is None else formal_charge
    raw_atom.element = gemmi.Element(atom_site.element)
    raw_atom.pos = gemmi.Position(
        atom_geometry.position.x,
        atom_geometry.position.y,
        atom_geometry.position.z,
    )
    raw_atom.occ = atom_geometry.occupancy
    raw_atom.b_iso = 0.0 if atom_geometry.b_factor is None else atom_geometry.b_factor
    return raw_atom


def residues_by_chain_id(
    structure: ProteinStructure,
) -> tuple[tuple[str, list[ResidueSite]], ...]:
    """Return polymer and ligand residues grouped by chain in write order."""

    grouped_residues: dict[str, list[ResidueSite]] = {
        chain.chain_id: list(chain.residues) for chain in structure.constitution.chains
    }
    polymer_chain_ids = tuple(chain.chain_id for chain in structure.constitution.chains)

    for ligand in structure.constitution.ligands:
        chain_id = ligand.residue_id.chain_id
        if chain_id not in grouped_residues:
            grouped_residues[chain_id] = []

        grouped_residues[chain_id].append(ligand)

    chain_order = polymer_chain_ids + tuple(
        chain_id for chain_id in grouped_residues if chain_id not in polymer_chain_ids
    )
    return tuple((chain_id, grouped_residues[chain_id]) for chain_id in chain_order)
