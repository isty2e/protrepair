"""PDB text projections for parser-facing structure consumers."""

from dataclasses import dataclass

from protrepair.io.gemmi_writer import (
    pdb_atom_index_values_in_write_order,
    write_pdb_structure_string_without_conect,
)
from protrepair.structure.address_space import StructureAddressSpaceKey
from protrepair.structure.aggregate import ProteinStructure


@dataclass(frozen=True, slots=True)
class RDKitNoConectPDBBlockProjector:
    """Coordinate-only no-CONECT PDB block projection for parser probes."""

    address_space_key: StructureAddressSpaceKey
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

        lines = list(self.base_lines)
        atom_geometries = structure.geometry.atom_geometries
        for line_index, atom_index_value, prefix, suffix in (
            self.coordinate_line_patches
        ):
            position = atom_geometries[atom_index_value].position
            lines[line_index] = (
                prefix + f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}" + suffix
            )

        return "\n".join(lines) + "\n"


def prepare_rdkit_no_conect_pdb_block_projector(
    structure: ProteinStructure,
) -> RDKitNoConectPDBBlockProjector | None:
    """Return a coordinate-only no-CONECT PDB projection when safely mappable."""

    pdb_block = pdb_without_conect(structure)
    base_lines = tuple(pdb_block.splitlines())
    atom_index_values = pdb_atom_index_values_in_write_order(structure)
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
