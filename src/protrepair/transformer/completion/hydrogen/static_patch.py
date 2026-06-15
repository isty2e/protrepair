"""Static hydrogen-patch construction over declarative residue semantics."""

from collections.abc import Mapping
from copy import deepcopy

from protrepair.chemistry import HydrogenSemantics, RotatableHydrogenKind
from protrepair.chemistry.hydrogen_plans import DISULFIDE_CYSTEINE_PLAN
from protrepair.geometry import (
    InternalCoordinateFrame,
    PlanarCenter,
    TetrahedralCenter,
    Vec3,
)
from protrepair.transformer.completion.hydrogen.domain import HydrogenResidueSite
from protrepair.transformer.completion.hydrogen.geometry import (
    RotatableHydrogenSearch,
    cysteine_thiol,
    n_terminal_hydrogens,
    serine_hydroxyl,
    threonine_hydroxyl,
    tyrosine_hydroxyl,
)
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

CoordinateBlock = list[Vec3]


def generate_hydrogen_patch(
    *,
    site: HydrogenResidueSite,
    patch: OrderedAtomPatch,
    semantics: HydrogenSemantics,
) -> OrderedAtomPatch:
    """Return the hydrogenated patch for one residue."""

    sidechain_atom_names: list[str]
    sidechain_coordinates: CoordinateBlock

    if semantics.rotatable_kind is RotatableHydrogenKind.CYS:
        sidechain_atom_names, sidechain_coordinates = cysteine_sidechain_hydrogens(
            patch=patch,
            site=site,
        )
    elif semantics.rotatable_kind is RotatableHydrogenKind.SER:
        sidechain_atom_names, sidechain_coordinates = serine_sidechain_hydrogens(
            patch=patch,
            site=site,
        )
    elif semantics.rotatable_kind is RotatableHydrogenKind.THR:
        sidechain_atom_names, sidechain_coordinates = threonine_sidechain_hydrogens(
            patch=patch,
            site=site,
        )
    elif semantics.rotatable_kind is RotatableHydrogenKind.TYR:
        sidechain_atom_names, sidechain_coordinates = tyrosine_sidechain_hydrogens(
            patch=patch,
            site=site,
        )
    else:
        sidechain_atom_names, sidechain_coordinates = standard_sidechain_hydrogens(
            patch=patch,
            semantics=semantics,
            include_backbone_hydrogen=site.includes_backbone_hydrogen(),
        )

    atom_names = list(deepcopy(patch.atom_names))
    atom_coordinates = [list(position) for position in patch.atom_coordinates]
    atom_names.extend(sidechain_atom_names)
    atom_coordinates.extend(list(coordinates) for coordinates in sidechain_coordinates)

    return OrderedAtomPatch.from_atom_coordinates(
        atom_names=atom_names,
        atom_coordinates=atom_coordinates,
    )


def standard_sidechain_hydrogens(
    *,
    patch: OrderedAtomPatch,
    semantics: HydrogenSemantics,
    include_backbone_hydrogen: bool,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for a static residue plan."""

    plan = semantics.static_plan(include_backbone_hydrogen=include_backbone_hydrogen)
    if plan is None:
        raise ValueError("static hydrogen semantics require a plan")

    atom_coordinates = patch.position_map()
    sidechain_atom_names: list[str] = []
    sidechain_coordinates: CoordinateBlock = []
    for output_names, method_name, arguments in plan:
        coordinates = HydrogenSemantics.evaluate_operation(
            method_name,
            arguments,
            atom_coordinates=atom_coordinates,
        )
        sidechain_atom_names.extend(output_names)
        sidechain_coordinates.extend(coordinates)

    return sidechain_atom_names, sidechain_coordinates


def cysteine_sidechain_hydrogens(
    *,
    patch: OrderedAtomPatch,
    site: HydrogenResidueSite,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for a cysteine residue."""

    atom_coordinates = patch.position_map()
    if site.has_disulfide_partner(atom_coordinates["SG"]):
        return evaluate_plan(DISULFIDE_CYSTEINE_PLAN, atom_coordinates)

    hydrogen = cysteine_thiol(
        atom_coordinates["SG"],
        atom_coordinates["CB"],
        atom_coordinates["CA"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["SG"],
        hydrogen=hydrogen,
        build_bond_length=1.34,
        reproject_bond_length=0.96,
        dihedral=InternalCoordinateFrame.torsion(
            atom_coordinates["CA"],
            atom_coordinates["CB"],
            atom_coordinates["SG"],
            hydrogen,
        ),
        partial_charge=0.19,
        sigma=0.11,
        epsilon=0.07,
        donor_element="S",
    )
    optimized = site.optimize_rotatable(search)
    return ["HA", "HB1", "HB2", "HG"], [
        TetrahedralCenter(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ).remaining(),
        *TetrahedralCenter(
            atom_coordinates["CA"],
            atom_coordinates["SG"],
            atom_coordinates["CB"],
        ).pair(),
        optimized,
    ]


def serine_sidechain_hydrogens(
    *,
    patch: OrderedAtomPatch,
    site: HydrogenResidueSite,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for serine."""

    atom_coordinates = patch.position_map()
    initial_hydrogen = serine_hydroxyl(
        atom_coordinates["OG"],
        atom_coordinates["CB"],
        atom_coordinates["CA"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["OG"],
        hydrogen=initial_hydrogen,
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        dihedral=InternalCoordinateFrame.torsion(
            atom_coordinates["CA"],
            atom_coordinates["CB"],
            atom_coordinates["OG"],
            initial_hydrogen,
        ),
        partial_charge=0.41,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )
    optimized = site.optimize_rotatable(search)
    hb1, hb2 = TetrahedralCenter(
        atom_coordinates["CA"],
        atom_coordinates["OG"],
        atom_coordinates["CB"],
    ).pair()
    return ["HA", "HB1", "HB2", "HG"], [
        TetrahedralCenter(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ).remaining(),
        hb1,
        hb2,
        optimized,
    ]


def threonine_sidechain_hydrogens(
    *,
    patch: OrderedAtomPatch,
    site: HydrogenResidueSite,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for threonine."""

    atom_coordinates = patch.position_map()
    initial_hydrogen = threonine_hydroxyl(
        atom_coordinates["OG1"],
        atom_coordinates["CB"],
        atom_coordinates["CG2"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["OG1"],
        hydrogen=initial_hydrogen,
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        dihedral=InternalCoordinateFrame.torsion(
            atom_coordinates["CA"],
            atom_coordinates["CB"],
            atom_coordinates["OG1"],
            initial_hydrogen,
        ),
        partial_charge=0.41,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )
    optimized = site.optimize_rotatable(search)
    return ["HG1", "HA", "HB", "1HG2", "2HG2", "3HG2"], [
        optimized,
        TetrahedralCenter(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ).remaining(),
        TetrahedralCenter(
            atom_coordinates["CA"],
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
        ).remaining(),
        InternalCoordinateFrame(
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
            atom_coordinates["CG2"],
        ).place(
            bond_length=1.09,
            bond_angle_degrees=109.4,
            dihedral_degrees=60.5,
        ),
        InternalCoordinateFrame(
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
            atom_coordinates["CG2"],
        ).place(
            bond_length=1.09,
            bond_angle_degrees=109.5,
            dihedral_degrees=-179.5,
        ),
        InternalCoordinateFrame(
            atom_coordinates["OG1"],
            atom_coordinates["CB"],
            atom_coordinates["CG2"],
        ).place(
            bond_length=1.09,
            bond_angle_degrees=109.5,
            dihedral_degrees=-59.5,
        ),
    ]


def tyrosine_sidechain_hydrogens(
    *,
    patch: OrderedAtomPatch,
    site: HydrogenResidueSite,
) -> tuple[list[str], CoordinateBlock]:
    """Return ordered sidechain hydrogens for tyrosine."""

    atom_coordinates = patch.position_map()
    initial_hydrogen = tyrosine_hydroxyl(
        atom_coordinates["OH"],
        atom_coordinates["CZ"],
        atom_coordinates["CE2"],
    )
    search = RotatableHydrogenSearch(
        outer_anchor=atom_coordinates["CE2"],
        inner_anchor=atom_coordinates["CZ"],
        donor=atom_coordinates["OH"],
        hydrogen=initial_hydrogen,
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        dihedral=InternalCoordinateFrame.torsion(
            atom_coordinates["CE2"],
            atom_coordinates["CZ"],
            atom_coordinates["OH"],
            initial_hydrogen,
        ),
        partial_charge=0.37,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )
    optimized = site.optimize_rotatable(search)
    hb1, hb2 = TetrahedralCenter(
        atom_coordinates["CA"],
        atom_coordinates["CG"],
        atom_coordinates["CB"],
    ).pair()
    return ["HA", "HB1", "HB2", "HD1", "HD2", "HE1", "HE2", "HH"], [
        TetrahedralCenter(
            atom_coordinates["CB"],
            atom_coordinates["N"],
            atom_coordinates["CA"],
        ).remaining(),
        hb1,
        hb2,
        PlanarCenter(
            atom_coordinates["CG"],
            atom_coordinates["CD1"],
            atom_coordinates["CE1"],
        ).projected(
            bond_length=1.08,
        ),
        PlanarCenter(
            atom_coordinates["CE2"],
            atom_coordinates["CD2"],
            atom_coordinates["CG"],
        ).projected(
            bond_length=1.08,
        ),
        PlanarCenter(
            atom_coordinates["CZ"],
            atom_coordinates["CE1"],
            atom_coordinates["CD1"],
        ).projected(
            bond_length=1.08,
        ),
        PlanarCenter(
            atom_coordinates["CZ"],
            atom_coordinates["CE2"],
            atom_coordinates["CD2"],
        ).projected(
            bond_length=1.08,
        ),
        optimized,
    ]


def histidine_delta_hydrogen(patch: OrderedAtomPatch) -> Vec3:
    """Return the additional ND1 hydrogen used for protonated histidines."""

    atom_coordinates = patch.position_map()
    return PlanarCenter(
        atom_coordinates["CE1"],
        atom_coordinates["ND1"],
        atom_coordinates["CG"],
    ).projected(
        bond_length=1.01,
    )


def n_terminal_hydrogen_coordinates(
    patch: OrderedAtomPatch,
    component_id: str,
) -> tuple[Vec3, ...]:
    """Return the ordered N-terminal hydrogens for the first residue in a chain."""

    return n_terminal_hydrogens(
        component_id,
        patch.position_map(),
    )


def evaluate_plan(
    plan: tuple[tuple[tuple[str, ...], str, tuple[str | float, ...]], ...],
    atom_coordinates: Mapping[str, Vec3],
) -> tuple[list[str], CoordinateBlock]:
    """Evaluate a declarative hydrogen plan against one residue."""

    atom_names: list[str] = []
    coordinates: CoordinateBlock = []
    for output_names, method_name, arguments in plan:
        coordinate_values = HydrogenSemantics.evaluate_operation(
            method_name,
            arguments,
            atom_coordinates=atom_coordinates,
        )
        atom_names.extend(output_names)
        coordinates.extend(list(coordinate_values))

    return atom_names, coordinates
