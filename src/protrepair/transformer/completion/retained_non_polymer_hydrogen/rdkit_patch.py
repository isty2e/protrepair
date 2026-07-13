"""RDKit-to-patch projection for retained non-polymer hydrogens."""

from collections.abc import Sequence
from math import isfinite
from typing import TYPE_CHECKING

from protrepair.geometry import Vec3
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


def rdkit_hydrogen_append_patch(
    payload: CompletionResiduePayload,
    *,
    hydrogenated_molecule: "Mol",
    hydrogen_atom_names: Sequence[str],
) -> OrderedAtomPatch:
    """Append or update RDKit H coordinates under caller-resolved final names."""

    normalized_hydrogen_atom_names = tuple(
        atom_name.strip().upper() for atom_name in hydrogen_atom_names
    )
    if any(not atom_name for atom_name in normalized_hydrogen_atom_names):
        raise ValueError(
            "retained non-polymer hydrogen atom names must be non-empty after "
            "normalization"
        )
    if len(set(normalized_hydrogen_atom_names)) != len(
        normalized_hydrogen_atom_names
    ):
        raise ValueError(
            "retained non-polymer hydrogen atom names must be unique after "
            "normalization"
        )
    existing_atom_site_by_name = {
        atom_site.name: atom_site for atom_site in payload.atom_sites
    }
    if any(
        (existing_atom_site := existing_atom_site_by_name.get(atom_name)) is not None
        and not existing_atom_site.is_hydrogen()
        for atom_name in normalized_hydrogen_atom_names
    ):
        raise ValueError(
            "retained non-polymer hydrogen atom name must not collide with an "
            "existing non-hydrogen atom"
        )

    conformer = hydrogenated_molecule.GetConformer()
    hydrogen_positions: list[Vec3] = []
    for atom in hydrogenated_molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue

        coordinates = conformer.GetAtomPosition(atom.GetIdx())
        position = Vec3(
            float(coordinates.x),
            float(coordinates.y),
            float(coordinates.z),
        )
        if not all(isfinite(component) for component in position):
            raise ValueError(
                "retained non-polymer RDKit hydrogen coordinates must be finite"
            )
        hydrogen_positions.append(position)
    if len(normalized_hydrogen_atom_names) != len(hydrogen_positions):
        raise ValueError(
            "retained non-polymer hydrogen name count must match RDKit "
            "hydrogen coordinate count"
        )

    current_patch = OrderedAtomPatch.from_residue_payload(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
    )
    hydrogen_position_by_name = dict(
        zip(
            normalized_hydrogen_atom_names,
            hydrogen_positions,
            strict=True,
        )
    )
    appended_hydrogen_atom_names = tuple(
        atom_name
        for atom_name in normalized_hydrogen_atom_names
        if atom_name not in existing_atom_site_by_name
    )
    current_position_by_name = current_patch.position_map()
    return OrderedAtomPatch.from_atom_coordinates(
        atom_names=[*current_patch.atom_names, *appended_hydrogen_atom_names],
        atom_coordinates=[
            *(
                hydrogen_position_by_name.get(
                    atom_name,
                    current_position_by_name[atom_name],
                )
                for atom_name in current_patch.atom_names
            ),
            *(
                hydrogen_position_by_name[atom_name]
                for atom_name in appended_hydrogen_atom_names
            ),
        ],
    )
