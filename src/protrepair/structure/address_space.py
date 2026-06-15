"""Structure facet address-space contracts."""

from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex

StructureAddressSpaceKey = tuple[tuple[ResidueId, tuple[str, ...]], ...]


def atom_count_in_address_space(
    address_space_key: StructureAddressSpaceKey,
) -> int:
    """Return the atom slot count represented by one address-space key."""

    return sum(len(atom_names) for _residue_id, atom_names in address_space_key)


def address_space_key_without_atoms(
    address_space_key: StructureAddressSpaceKey,
    *,
    removed_atom_indices: set[AtomIndex],
) -> StructureAddressSpaceKey:
    """Project one address-space key after dropping selected atom slots."""

    removed_atom_index_values = {
        atom_index.value for atom_index in removed_atom_indices
    }
    projected_address_space: list[tuple[ResidueId, tuple[str, ...]]] = []
    atom_cursor = 0
    for residue_id, atom_names in address_space_key:
        kept_atom_names = tuple(
            atom_name
            for atom_offset, atom_name in enumerate(atom_names)
            if atom_cursor + atom_offset not in removed_atom_index_values
        )
        projected_address_space.append((residue_id, kept_atom_names))
        atom_cursor += len(atom_names)

    return tuple(projected_address_space)
