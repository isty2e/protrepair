"""Spatial context selection for continuous-relaxation regions."""

from collections import defaultdict
from math import floor

from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot

GRID_NEIGHBOR_OFFSETS = (-1, 0, 1)
GRID_NEIGHBOR_CELL_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in GRID_NEIGHBOR_OFFSETS
    for dy in GRID_NEIGHBOR_OFFSETS
    for dz in GRID_NEIGHBOR_OFFSETS
)


def radius_context_residue_indices(
    snapshot: ProteinStructureSnapshot,
    *,
    movable_atom_index_set: set[AtomIndex],
    selected_residue_index_set: set[ResidueIndex],
    context_radius_angstrom: float,
) -> set[ResidueIndex]:
    """Return non-selected residues within the movable atom distance shell."""

    constitution = snapshot.structure.constitution
    movable_atom_geometries = tuple(
        snapshot.structure.geometry.atom_geometry(atom_index)
        for atom_index in movable_atom_index_set
    )
    if context_radius_angstrom <= 0.0:
        return _radius_context_residue_indices_by_scan(
            snapshot,
            movable_atom_index_set=movable_atom_index_set,
            selected_residue_index_set=selected_residue_index_set,
            movable_atom_geometries=movable_atom_geometries,
            context_radius_angstrom=context_radius_angstrom,
        )

    radius_squared = context_radius_angstrom * context_radius_angstrom
    movable_atom_geometries_by_cell: dict[
        tuple[int, int, int],
        list[AtomGeometry],
    ] = defaultdict(list)
    for movable_atom_geometry in movable_atom_geometries:
        movable_atom_geometries_by_cell[
            _atom_geometry_grid_cell(
                movable_atom_geometry,
                cell_size_angstrom=context_radius_angstrom,
            )
        ].append(movable_atom_geometry)

    context_residue_index_set: set[ResidueIndex] = set()
    for residue_slot_index in range(len(constitution.residue_slots)):
        residue_index = ResidueIndex(residue_slot_index)
        if residue_index in selected_residue_index_set:
            continue

        for atom_index in constitution.atom_indices_for_residue_index(residue_index):
            if atom_index in movable_atom_index_set:
                continue

            candidate_atom_geometry = snapshot.structure.geometry.atom_geometry(
                atom_index
            )
            if _is_within_radius_of_indexed_atom_geometries(
                candidate_atom_geometry,
                atom_geometries_by_cell=movable_atom_geometries_by_cell,
                radius_angstrom=context_radius_angstrom,
                radius_squared=radius_squared,
            ):
                context_residue_index_set.add(residue_index)
                break

    return context_residue_index_set


def _radius_context_residue_indices_by_scan(
    snapshot: ProteinStructureSnapshot,
    *,
    movable_atom_index_set: set[AtomIndex],
    selected_residue_index_set: set[ResidueIndex],
    movable_atom_geometries: tuple[AtomGeometry, ...],
    context_radius_angstrom: float,
) -> set[ResidueIndex]:
    """Return non-selected radius context using all-pairs distance scanning."""

    constitution = snapshot.structure.constitution
    context_residue_index_set: set[ResidueIndex] = set()
    for residue_slot_index in range(len(constitution.residue_slots)):
        residue_index = ResidueIndex(residue_slot_index)
        if residue_index in selected_residue_index_set:
            continue

        for atom_index in constitution.atom_indices_for_residue_index(residue_index):
            if atom_index in movable_atom_index_set:
                continue

            candidate_atom_geometry = snapshot.structure.geometry.atom_geometry(
                atom_index
            )
            if any(
                movable_atom_geometry.distance_to(candidate_atom_geometry)
                <= context_radius_angstrom
                for movable_atom_geometry in movable_atom_geometries
            ):
                context_residue_index_set.add(residue_index)
                break

    return context_residue_index_set


def _atom_geometry_grid_cell(
    atom_geometry: AtomGeometry,
    *,
    cell_size_angstrom: float,
) -> tuple[int, int, int]:
    """Return the spatial grid cell for one atom geometry."""

    position = atom_geometry.position
    return (
        floor(position.x / cell_size_angstrom),
        floor(position.y / cell_size_angstrom),
        floor(position.z / cell_size_angstrom),
    )


def _is_within_radius_of_indexed_atom_geometries(
    atom_geometry: AtomGeometry,
    *,
    atom_geometries_by_cell: dict[tuple[int, int, int], list[AtomGeometry]],
    radius_angstrom: float,
    radius_squared: float,
) -> bool:
    """Return whether one atom is within radius of an indexed atom geometry."""

    atom_cell = _atom_geometry_grid_cell(
        atom_geometry,
        cell_size_angstrom=radius_angstrom,
    )
    cell_x, cell_y, cell_z = atom_cell
    for offset_x, offset_y, offset_z in GRID_NEIGHBOR_CELL_OFFSETS:
        neighboring_cell = (
            cell_x + offset_x,
            cell_y + offset_y,
            cell_z + offset_z,
        )
        for indexed_atom_geometry in atom_geometries_by_cell.get(
            neighboring_cell,
            (),
        ):
            if (
                _atom_geometry_distance_squared(indexed_atom_geometry, atom_geometry)
                <= radius_squared
            ):
                return True

    return False


def _atom_geometry_distance_squared(
    left_atom_geometry: AtomGeometry,
    right_atom_geometry: AtomGeometry,
) -> float:
    """Return squared Cartesian distance between two atom geometries."""

    left_position = left_atom_geometry.position
    right_position = right_atom_geometry.position
    dx = left_position.x - right_position.x
    dy = left_position.y - right_position.y
    dz = left_position.z - right_position.z
    return dx * dx + dy * dy + dz * dz
