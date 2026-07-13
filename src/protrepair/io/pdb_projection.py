"""PDB text projections for parser-facing structure consumers."""

from dataclasses import dataclass

from protrepair.io.gemmi_writer import (
    atom_index_values_in_coordinate_write_order,
    write_pdb_structure_string_without_conect,
)
from protrepair.structure.address_space import StructureAddressSpaceKey
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import StructureGeometry


@dataclass(frozen=True, slots=True)
class RDKitNoConectPDBBlockProjector:
    """Coordinate-only no-CONECT PDB block projection for parser probes."""

    address_space_key: StructureAddressSpaceKey
    base_geometry: StructureGeometry
    base_pdb_block: str
    base_lines: tuple[str, ...]
    coordinate_line_patches: tuple[tuple[int, int, str, str], ...]
    atom_count: int

    def can_render(self, structure: ProteinStructure) -> bool:
        """Return whether the structure has the same atom address space."""

        return (
            structure.constitution.address_space_key == self.address_space_key
            and structure.geometry.atom_count() == self.atom_count
        )

    def render(self, structure: ProteinStructure) -> str:
        """Return a no-CONECT PDB block with current coordinates patched in."""

        if not self.can_render(structure):
            raise ValueError(
                "PDB block projection requires the original atom address space"
            )

        if structure.geometry is self.base_geometry:
            return self.base_pdb_block

        atom_geometries = structure.geometry.atom_geometries
        base_atom_geometries = self.base_geometry.atom_geometries
        lines: list[str] | None = None
        for line_index, atom_index_value, prefix, suffix in (
            self.coordinate_line_patches
        ):
            position = atom_geometries[atom_index_value].position
            if position == base_atom_geometries[atom_index_value].position:
                continue

            if lines is None:
                lines = list(self.base_lines)
            lines[line_index] = (
                prefix
                + _format_pdb_coordinates(position.x, position.y, position.z)
                + suffix
            )

        if lines is None:
            return self.base_pdb_block

        return "\n".join(lines) + "\n"


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
        address_space_key=structure.constitution.address_space_key,
        base_geometry=structure.geometry,
        base_pdb_block=pdb_block,
        base_lines=base_lines,
        coordinate_line_patches=tuple(coordinate_line_patches),
        atom_count=structure.geometry.atom_count(),
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

    if pdb_block_projector is not None and pdb_block_projector.can_render(structure):
        return pdb_block_projector.render(structure)

    return pdb_without_conect(structure)


def _is_pdb_atom_line(line: str) -> bool:
    """Return whether a PDB line carries atom coordinates."""

    return line.startswith(("ATOM  ", "HETATM"))


def _format_pdb_coordinates(x: float, y: float, z: float) -> str:
    """Format one coordinate triple with Gemmi's PDB normalization."""

    normalized_x = 0.0 if -5e-4 < x < 0.0 else x + 1e-10
    normalized_y = 0.0 if -5e-4 < y < 0.0 else y + 1e-10
    normalized_z = 0.0 if -5e-4 < z < 0.0 else z + 1e-10
    normalized_fields = (
        f"{normalized_x:8.3f}"
        f"{normalized_y:8.3f}"
        f"{normalized_z:8.3f}"
    )
    if len(normalized_fields) == 24:
        return normalized_fields

    # Gemmi fixes overflowed records in-place: x keeps its normalized field,
    # while y and z are rewritten from the original values at fixed columns.
    return (
        normalized_fields[:8]
        + f"{y:8.3f}"[:8]
        + f"{z:8.3f}"[:8]
    )
