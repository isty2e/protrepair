"""Hydrogen-placement geometry helpers for completion transformers."""

from collections.abc import Mapping
from math import pi

from protrepair.geometry import (
    AxisRotation,
    GeometryPlacementError,
    InternalCoordinateFrame,
    PlanarCenter,
    TetrahedralCenter,
    Vec3,
)
from protrepair.geometry.placement_vector import (
    PlacementVector,
    scaled_point_from_origin_or_none,
)
from protrepair.transformer.completion.hydrogen.scoring import (
    ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS,
    ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_CUTOFF_SQ_ANGSTROM,
    CoordinateLike,
    RotatableHydrogenEnvironment,
    RotatableHydrogenLocalSite,
    RotatableHydrogenSearch,
    hydrogen_potential_energy,
    max_rotatable_hydrogen_steric_cutoff_angstrom,
    rotatable_hydrogen_steric_cutoff_angstrom,
    rotatable_hydrogen_vdw_radius_angstrom,
)


def backbone_hydrogen(
    next_alpha_carbon: CoordinateLike,
    next_nitrogen: CoordinateLike,
    carbonyl_carbon: CoordinateLike,
) -> Vec3:
    """Return the propagated backbone hydrogen placed on the next residue."""

    return PlanarCenter(
        next_alpha_carbon,
        next_nitrogen,
        carbonyl_carbon,
    ).projected(bond_length=1.01)


def hydroxyl_hydrogen(
    donor: CoordinateLike,
    atom_b: CoordinateLike,
    atom_c: CoordinateLike,
    *,
    rotation_degrees: float,
    bond_length: float,
) -> Vec3:
    """Return the pre-optimization coordinate for a rotatable hydroxyl/thiol H."""

    donor_vector = Vec3.coerce(donor).to_array()
    bond_vector = Vec3.coerce(atom_b).to_array() - donor_vector
    rotated = AxisRotation.from_points(
        atom_b,
        atom_c,
    ).rotate_vector(
        bond_vector,
        theta_radians=rotation_degrees * pi / 180.0,
    )
    candidate = rotated + donor_vector
    return scale_bond(donor_vector, candidate, bond_length)


def serine_hydroxyl(
    og: CoordinateLike,
    cb: CoordinateLike,
    ca: CoordinateLike,
) -> Vec3:
    """Return the initial SER hydroxyl hydrogen coordinate."""

    hb1, _ = TetrahedralCenter(ca, og, cb).pair()
    return hydroxyl_hydrogen(og, cb, hb1, rotation_degrees=-240.2, bond_length=0.96)


def threonine_hydroxyl(
    og1: CoordinateLike,
    cb: CoordinateLike,
    cg2: CoordinateLike,
) -> Vec3:
    """Return the initial THR hydroxyl hydrogen coordinate."""

    return hydroxyl_hydrogen(og1, cb, cg2, rotation_degrees=-243.2, bond_length=0.96)


def cysteine_thiol(
    sg: CoordinateLike,
    cb: CoordinateLike,
    ca: CoordinateLike,
) -> Vec3:
    """Return the initial CYS thiol hydrogen coordinate."""

    return hydroxyl_hydrogen(sg, cb, ca, rotation_degrees=-243.2, bond_length=1.3)


def tyrosine_hydroxyl(
    oh: CoordinateLike,
    cz: CoordinateLike,
    ce2: CoordinateLike,
) -> Vec3:
    """Return the initial TYR hydroxyl hydrogen coordinate."""

    return hydroxyl_hydrogen(oh, cz, ce2, rotation_degrees=-220.2, bond_length=0.96)


def n_terminal_hydrogens(
    residue_name: str,
    atom_coordinates: Mapping[str, CoordinateLike],
) -> tuple[Vec3, ...]:
    """Return the ordered N-terminal hydrogens for the first residue in a chain."""

    if residue_name == "PRO":
        return TetrahedralCenter(
            atom_coordinates["CA"],
            atom_coordinates["CD"],
            atom_coordinates["N"],
        ).pair(
            bond_length=1.01,
        )

    reference = (
        atom_coordinates["C"] if residue_name == "GLY" else atom_coordinates["CB"]
    )
    return (
        InternalCoordinateFrame(
            reference,
            atom_coordinates["CA"],
            atom_coordinates["N"],
        ).place(
            bond_length=1.01,
            bond_angle_degrees=109.5,
            dihedral_degrees=179.6,
        ),
        InternalCoordinateFrame(
            reference,
            atom_coordinates["CA"],
            atom_coordinates["N"],
        ).place(
            bond_length=1.01,
            bond_angle_degrees=109.6,
            dihedral_degrees=-60.4,
        ),
        InternalCoordinateFrame(
            reference,
            atom_coordinates["CA"],
            atom_coordinates["N"],
        ).place(
            bond_length=1.01,
            bond_angle_degrees=109.6,
            dihedral_degrees=60.2,
        ),
    )


def scale_bond(
    origin: PlacementVector,
    candidate: PlacementVector,
    bond_length: float,
) -> Vec3:
    """Scale a candidate point to the desired bond length from the origin."""

    scaled = scaled_point_from_origin_or_none(origin, candidate, bond_length)
    if scaled is None:
        raise GeometryPlacementError(
            "hydrogen placement produced a degenerate bond vector"
        )

    return Vec3.from_iterable(scaled)


__all__ = [
    "ROTATABLE_HYDROGEN_LOCAL_IGNORE_BOND_HOPS",
    "ROTATABLE_HYDROGEN_POTENTIAL_ENERGY_CUTOFF_SQ_ANGSTROM",
    "CoordinateLike",
    "RotatableHydrogenEnvironment",
    "RotatableHydrogenLocalSite",
    "RotatableHydrogenSearch",
    "backbone_hydrogen",
    "cysteine_thiol",
    "hydrogen_potential_energy",
    "hydroxyl_hydrogen",
    "max_rotatable_hydrogen_steric_cutoff_angstrom",
    "n_terminal_hydrogens",
    "rotatable_hydrogen_steric_cutoff_angstrom",
    "rotatable_hydrogen_vdw_radius_angstrom",
    "scale_bond",
    "serine_hydroxyl",
    "threonine_hydroxyl",
    "tyrosine_hydroxyl",
]
