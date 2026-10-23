"""PDB text projections for parser-facing structure consumers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.errors import ModelInvariantError
from protrepair.io.gemmi_writer import (
    atom_index_values_in_coordinate_write_order,
    pdb_typed_connection_topology_bonds_for_egress,
    write_pdb_structure_string_without_conect,
)
from protrepair.io.pdb_records import decode_pdb_integer, pdb_atom_identity_matches
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.element import ElementIdentity
from protrepair.structure.geometry import AtomGeometry, StructureGeometry
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import StructureTopology


@dataclass(frozen=True, slots=True)
class RDKitPDBAtomCorrespondence:
    """Immutable source identities and emitted signatures keyed by PDB serial.

    Parameters
    ----------
    records_by_serial : Mapping[int, tuple[AtomRef, str, ElementIdentity]]
        Canonical reference, emitted atom name and element for each record.
    """

    records_by_serial: Mapping[int, tuple[AtomRef, str, ElementIdentity]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "records_by_serial", MappingProxyType(dict(self.records_by_serial))
        )

    def require_atom_ref(
        self,
        serial: int,
        *,
        atom_name: str,
        element_symbol: str,
        isotope_mass_number: int,
    ) -> AtomRef:
        """Resolve native metadata only when its emitted signature agrees.

        Parameters
        ----------
        serial : int
            Serial preserved by the native reader, not its atom index.
        atom_name : str
            Native PDB atom-name metadata.
        element_symbol : str
            Native chemical element symbol.
        isotope_mass_number : int
            Native isotope mass, or zero for an unspecified isotope.

        Returns
        -------
        AtomRef
            The original canonical atom identity.

        Raises
        ------
        ModelInvariantError
            The native atom has no unique compatible source record.
        """
        record = self.records_by_serial.get(serial)
        if record is None:
            raise ModelInvariantError(
                f"PDB atom correspondence has unknown native serial {serial}"
            )
        atom_ref, expected_name, element = record
        if (
            atom_name.strip().upper() != expected_name
            or element_symbol.upper() != element.chemical_symbol
            or isotope_mass_number != (element.isotope_mass_number or 0)
        ):
            raise ModelInvariantError(
                f"PDB atom correspondence signature differs at {atom_ref.display_token()}"
            )
        return atom_ref


@dataclass(frozen=True, slots=True)
class RDKitNoConectPDBInput:
    """Rendered parser input paired with its source atom correspondence.

    Parameters
    ----------
    pdb_block : str
        The unchanged no-CONECT parser surface.
    correspondence : RDKitPDBAtomCorrespondence
        Immutable source association for those emitted records.
    """

    pdb_block: str
    correspondence: RDKitPDBAtomCorrespondence


@dataclass(frozen=True, slots=True)
class RDKitNoConectPDBBlockProjector:
    """Coordinate-only no-CONECT PDB block projection for parser probes.

    Parameters
    ----------
    base_constitution : StructureConstitution
        Source atom identities and write order.
    base_topology : StructureTopology
        Source charges and connections affecting serialization.
    base_geometry : StructureGeometry
        Coordinates and PDB metadata used for compatibility checks.
    base_input : RDKitNoConectPDBInput
        Original text and immutable source association.
    base_lines : tuple[str, ...]
        Original records without newline terminators.
    coordinate_line_patches : tuple[tuple[int, int, str, str], ...]
        Record index, atom slot and text surrounding each coordinate field.
    _connection_endpoint_atom_index_values : frozenset[int]
        Atoms whose motion requires regenerating connection records.
    """

    base_constitution: StructureConstitution
    base_topology: StructureTopology
    base_geometry: StructureGeometry
    base_input: RDKitNoConectPDBInput
    base_lines: tuple[str, ...]
    coordinate_line_patches: tuple[tuple[int, int, str, str], ...]
    _connection_endpoint_atom_index_values: frozenset[int]

    def can_render(self, structure: ProteinStructure) -> bool:
        """Return whether every unpatched PDB serialization fact is unchanged."""

        if not self._has_compatible_immutable_facets(structure):
            return False
        if structure.geometry is self.base_geometry:
            return True

        return all(
            self._atom_geometry_is_projection_compatible(
                atom_index_value,
                current,
                base,
            )
            for atom_index_value, (current, base) in enumerate(
                zip(
                    structure.geometry.atom_geometries,
                    self.base_geometry.atom_geometries,
                    strict=True,
                )
            )
        )

    def render(self, structure: ProteinStructure) -> str:
        """Return a no-CONECT PDB block with current coordinates patched in."""

        projected_pdb_block = self._render_if_compatible(structure)
        if projected_pdb_block is None:
            raise ValueError(
                "PDB block projection requires projection-compatible "
                "serialization facts"
            )

        return projected_pdb_block

    def _render_if_compatible(self, structure: ProteinStructure) -> str | None:
        """Return a projected block, or None when canonical writing is required."""

        if not self._has_compatible_immutable_facets(structure):
            return None
        if structure.geometry is self.base_geometry:
            return self.base_input.pdb_block

        atom_geometries = structure.geometry.atom_geometries
        base_atom_geometries = self.base_geometry.atom_geometries
        lines: list[str] | None = None
        for (
            line_index,
            atom_index_value,
            prefix,
            suffix,
        ) in self.coordinate_line_patches:
            atom_geometry = atom_geometries[atom_index_value]
            base_atom_geometry = base_atom_geometries[atom_index_value]
            if not self._atom_geometry_is_projection_compatible(
                atom_index_value,
                atom_geometry,
                base_atom_geometry,
            ):
                return None
            if atom_geometry.position == base_atom_geometry.position:
                continue

            if lines is None:
                lines = list(self.base_lines)
            lines[line_index] = (
                prefix
                + _format_pdb_coordinates(
                    atom_geometry.position.x,
                    atom_geometry.position.y,
                    atom_geometry.position.z,
                )
                + suffix
            )

        if lines is None:
            return self.base_input.pdb_block

        return "\n".join(lines) + "\n"

    def _has_compatible_immutable_facets(
        self,
        structure: ProteinStructure,
    ) -> bool:
        """Return whether constitution and topology serialize identically."""

        return (
            (
                structure.constitution is self.base_constitution
                or structure.constitution == self.base_constitution
            )
            and (
                structure.topology is self.base_topology
                or structure.topology == self.base_topology
            )
            and structure.geometry.atom_count() == self.base_geometry.atom_count()
        )

    @staticmethod
    def _has_matching_pdb_metadata(
        atom_geometry: AtomGeometry,
        base_atom_geometry: AtomGeometry,
    ) -> bool:
        """Return whether non-coordinate atom geometry fields match."""

        return (
            atom_geometry.occupancy == base_atom_geometry.occupancy
            and atom_geometry.b_factor == base_atom_geometry.b_factor
            and atom_geometry.altloc == base_atom_geometry.altloc
        )

    def _atom_geometry_is_projection_compatible(
        self,
        atom_index_value: int,
        atom_geometry: AtomGeometry,
        base_atom_geometry: AtomGeometry,
    ) -> bool:
        """Return whether one atom can reuse every unpatched PDB field."""

        return self._has_matching_pdb_metadata(
            atom_geometry,
            base_atom_geometry,
        ) and (
            atom_geometry.position == base_atom_geometry.position
            or atom_index_value not in self._connection_endpoint_atom_index_values
        )


def prepare_rdkit_no_conect_pdb_block_projector(
    structure: ProteinStructure,
) -> RDKitNoConectPDBBlockProjector | None:
    """Return a coordinate-only no-CONECT PDB projection when safely mappable."""

    pdb_block = pdb_without_conect(structure)
    base_lines = tuple(pdb_block.splitlines())
    atom_index_values = atom_index_values_in_coordinate_write_order(structure)
    coordinate_line_patches: list[tuple[int, int, str, str]] = []
    atom_index_value_cursor = 0
    for line_index, line in enumerate(base_lines):
        if not _is_pdb_atom_line(line):
            continue

        if atom_index_value_cursor >= len(atom_index_values):
            return None

        atom_index_value = atom_index_values[atom_index_value_cursor]
        coordinate_line_patches.append(
            (line_index, atom_index_value, line[:30], line[54:])
        )
        atom_index_value_cursor += 1

    if atom_index_value_cursor != structure.geometry.atom_count():
        return None

    return RDKitNoConectPDBBlockProjector(
        base_constitution=structure.constitution,
        base_topology=structure.topology,
        base_geometry=structure.geometry,
        base_input=_parser_input_with_correspondence(structure, pdb_block),
        base_lines=base_lines,
        coordinate_line_patches=tuple(coordinate_line_patches),
        _connection_endpoint_atom_index_values=frozenset(
            atom_index.value
            for bond in pdb_typed_connection_topology_bonds_for_egress(structure)
            for atom_index in (bond.atom_index_1, bond.atom_index_2)
        ),
    )


def pdb_without_conect(structure: ProteinStructure) -> str:
    """Return the structure serialized as PDB text without CONECT records."""

    pdb_text = write_pdb_structure_string_without_conect(structure)
    return (
        "\n".join(
            line for line in pdb_text.splitlines() if not line.startswith("CONECT")
        )
        + "\n"
    )


def pdb_without_conect_for_parser_probe(
    structure: ProteinStructure,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None,
) -> str:
    """Return a no-CONECT PDB block, using a coordinate projector when valid."""

    if pdb_block_projector is not None:
        projected_pdb_block = pdb_block_projector._render_if_compatible(structure)
        if projected_pdb_block is not None:
            return projected_pdb_block

    return pdb_without_conect(structure)


def prepare_rdkit_no_conect_pdb_input(
    structure: ProteinStructure,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None,
) -> RDKitNoConectPDBInput:
    """Render parser input with the source association for its actual records.

    Parameters
    ----------
    structure : ProteinStructure
        The current canonical source, including its selected atom cohort.
    pdb_block_projector : RDKitNoConectPDBBlockProjector or None
        Optional coordinate preparation; incompatible inputs are rendered afresh.

    Returns
    -------
    RDKitNoConectPDBInput
        Exact parser text and its matching immutable source correspondence.

    Raises
    ------
    ModelInvariantError
        Emitted atom records cannot be associated with the source write order.
    """
    if pdb_block_projector is not None:
        text = pdb_block_projector._render_if_compatible(structure)
        if text is not None:
            return RDKitNoConectPDBInput(
                text, pdb_block_projector.base_input.correspondence
            )
    return _parser_input_with_correspondence(structure, pdb_without_conect(structure))


def _parser_input_with_correspondence(
    structure: ProteinStructure, pdb_block: str
) -> RDKitNoConectPDBInput:
    indices = atom_index_values_in_coordinate_write_order(structure)
    records: dict[int, tuple[AtomRef, str, ElementIdentity]] = {}
    cursor = 0
    constitution = structure.constitution
    for line in pdb_block.splitlines():
        if not _is_pdb_atom_line(line):
            continue
        if cursor >= len(indices):
            raise ModelInvariantError(
                "PDB atom correspondence has extra emitted records"
            )
        index = AtomIndex(indices[cursor])
        ref = constitution.atom_ref_at(index)
        site = constitution.atom_site_at(index)
        residue = constitution.residue_site_at(
            constitution.residue_index_for_atom_index(index)
        )
        geometry = structure.geometry.atom_geometry(index)
        # The private transport retains Gemmi's existing four-/three-column aliases.
        emitted_ref = (
            ref
            if len(ref.atom_name) <= 4
            else AtomRef(ref.residue_id, ref.atom_name[:4])
        )
        try:
            serial = decode_pdb_integer(line[6:11], width=5)
            matches = pdb_atom_identity_matches(
                line,
                emitted_ref,
                residue.component_id[:3],
                None if geometry.altloc is None else geometry.altloc.upper(),
            )
            element = ElementIdentity(line[76:78])
        except ValueError as error:
            raise ModelInvariantError(
                "PDB atom correspondence has an invalid emitted identity"
            ) from error
        if not matches or element != site.element_identity:
            raise ModelInvariantError(
                f"PDB atom correspondence does not match {ref.display_token()}"
            )
        if serial in records:
            raise ModelInvariantError(
                f"PDB atom correspondence has duplicate emitted serial {serial}"
            )
        records[serial] = (ref, line[12:16].strip().upper(), element)
        cursor += 1
    if cursor != len(indices):
        raise ModelInvariantError("PDB atom correspondence has missing emitted records")
    return RDKitNoConectPDBInput(pdb_block, RDKitPDBAtomCorrespondence(records))


def _is_pdb_atom_line(line: str) -> bool:
    """Return whether a PDB line carries atom coordinates."""

    return line.startswith(("ATOM  ", "HETATM"))


def _format_pdb_coordinates(x: float, y: float, z: float) -> str:
    """Format one coordinate triple with Gemmi's PDB normalization."""

    normalized_x = 0.0 if -5e-4 < x < 0.0 else x + 1e-10
    normalized_y = 0.0 if -5e-4 < y < 0.0 else y + 1e-10
    normalized_z = 0.0 if -5e-4 < z < 0.0 else z + 1e-10
    normalized_fields = f"{normalized_x:8.3f}{normalized_y:8.3f}{normalized_z:8.3f}"
    if len(normalized_fields) == 24:
        return normalized_fields

    # Gemmi fixes overflowed records in-place: x keeps its normalized field,
    # while y and z are rewritten from the original values at fixed columns.
    return normalized_fields[:8] + f"{y:8.3f}"[:8] + f"{z:8.3f}"[:8]
