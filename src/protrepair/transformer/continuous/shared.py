"""Shared helpers for continuous-relaxation execution backends."""

from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.slots import AtomIndex


def apply_position_updates(
    structure: ProteinStructure,
    *,
    moved_positions: dict[AtomIndex, Vec3],
) -> ProteinStructure:
    """Return one structure with updated atom coordinates applied."""

    updated_geometry = structure.geometry
    for atom_index, position in moved_positions.items():
        updated_geometry = updated_geometry.with_atom_geometry(
            atom_index,
            updated_geometry.atom_geometry(atom_index).with_position(position),
        )

    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=updated_geometry,
        topology=structure.topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
