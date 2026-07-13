"""PDB text projections for parser-facing structure consumers."""

from dataclasses import dataclass

from protrepair.io.gemmi_writer import (
    atom_index_values_in_coordinate_write_order,
    pdb_typed_connection_topology_bonds_for_egress,
    write_pdb_structure_string_without_conect,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.geometry import AtomGeometry, StructureGeometry
from protrepair.structure.topology import StructureTopology


@dataclass(frozen=True, slots=True)
class RDKitNoConectPDBBlockProjector:
    """Coordinate-only no-CONECT PDB block projection for parser probes."""

    base_constitution: StructureConstitution
    base_topology: StructureTopology
    base_geometry: StructureGeometry
    base_pdb_block: str
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
            return self.base_pdb_block

        atom_geometries = structure.geometry.atom_geometries
        base_atom_geometries = self.base_geometry.atom_geometries
        lines: list[str] | None = None
        for line_index, atom_index_value, prefix, suffix in (
            self.coordinate_line_patches
        ):
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
            return self.base_pdb_block

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
        base_pdb_block=pdb_block,
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
