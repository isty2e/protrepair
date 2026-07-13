"""Shared residue-local rotation and atom-reference utilities."""

import numpy as np
from numpy.typing import NDArray

from protrepair.geometry import AxisRotation
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry

FloatArray = NDArray[np.float64]
ATTACHED_HYDROGEN_DISTANCE_MAX_ANGSTROM = 1.35


def rotate_residue_atoms_about_axis(
    *,
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...] = (),
    axis_atom_names: tuple[str, str],
    rotated_atom_names: set[str],
    theta_radians: float,
) -> tuple[
    ResidueSite,
    ResidueGeometry,
    tuple[tuple[str, int | None], ...],
    tuple[str, ...],
]:
    """Return one residue after rotating a named atom set around a residue axis."""

    axis_start = residue_geometry.position(axis_atom_names[0])
    axis_rotation = AxisRotation.from_points(
        axis_start,
        residue_geometry.position(axis_atom_names[1]),
    )
    rotated_atom_sites = tuple(
        atom_site
        for atom_site in residue_site.atom_sites
        if atom_site.name in rotated_atom_names
    )
    rotated_positions = axis_rotation.rotate_points(
        (
            residue_geometry.position(atom_site.name)
            for atom_site in rotated_atom_sites
        ),
        origin=axis_start,
        theta_radians=theta_radians,
    )
    updated_atom_geometries = tuple(
        (
            atom_site.name,
            residue_geometry.atom_geometry(atom_site.name).with_position(
                rotated_position
            ),
        )
        for atom_site, rotated_position in zip(
            rotated_atom_sites,
            rotated_positions,
            strict=True,
        )
    )
    moved_atom_names = tuple(
        dict.fromkeys(atom_site.name for atom_site in rotated_atom_sites)
    )

    return (
        residue_site,
        residue_geometry.with_atom_geometries(updated_atom_geometries),
        formal_charge_by_atom_name,
        moved_atom_names,
    )


def attached_hydrogen_atom_names(
    *,
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    anchor_atom_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return present hydrogens directly bonded to one of the anchor heavy atoms."""

    anchor_atoms = tuple(
        residue_geometry.atom_geometry(atom_name)
        for atom_name in anchor_atom_names
        if residue_site.has_atom_site(atom_name)
    )
    return tuple(
        atom_site.name
        for atom_site in residue_site.atom_sites
        if atom_site.is_hydrogen()
        and any(
            residue_geometry.atom_geometry(atom_site.name).distance_to(anchor_atom)
            <= ATTACHED_HYDROGEN_DISTANCE_MAX_ANGSTROM
            for anchor_atom in anchor_atoms
        )
    )


def replace_residue(
    *,
    structure: ProteinStructure,
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...] = (),
) -> ProteinStructure:
    """Return one structure with a single residue replaced in-place by id."""

    residue_index = structure.constitution.residue_index(residue_site.residue_id)
    if residue_site == structure.constitution.residue_site_at(
        residue_index
    ) and formal_charge_by_atom_name == structure.residue_formal_charge_by_atom_name(
        residue_index
    ):
        return structure.with_updated_residue_geometries(
            ((residue_site.residue_id, residue_geometry),)
        )

    return structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )
