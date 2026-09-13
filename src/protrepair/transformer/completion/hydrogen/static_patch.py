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
    cysteine_thiol,
    serine_hydroxyl,
    threonine_hydroxyl,
    tyrosine_hydroxyl,
)
from protrepair.transformer.completion.hydrogen.rotatable import (
    ROTATABLE_HYDROGEN_SPECS,
)
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

CoordinateBlock = list[Vec3]


def generate_hydrogen_patch(
    *,
    site: HydrogenResidueSite,
    patch: OrderedAtomPatch,
    semantics: HydrogenSemantics,
    selected_hydrogen_names: frozenset[str] | None = None,
) -> OrderedAtomPatch:
    """Place fixed-template H without selecting polymer microstates.

    Parameters
    ----------
    site : HydrogenResidueSite
        Residue and environment used by rotatable H placement.
    patch : OrderedAtomPatch
        Heavy coordinates to extend.
    semantics : HydrogenSemantics
        Static placement program or rotatable-group semantics.
    selected_hydrogen_names : frozenset[str] or None
        Static-program H names to evaluate; None evaluates the whole program.
        Rotatable-group placement uses its own fixed inventory.

    Returns
    -------
    OrderedAtomPatch
        Extended coordinates; chemistry and topology are applied by the caller.

    Raises
    ------
    ValueError
        Static semantics have no placement plan.
    KeyError
        A selected operation lacks a required anchor.
    GeometryPlacementError
        Required geometry is degenerate or non-finite.
    """

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
            selected_hydrogen_names=selected_hydrogen_names,
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
    selected_hydrogen_names: frozenset[str] | None = None,
) -> tuple[list[str], CoordinateBlock]:
    """Evaluate selected operations in a static H placement program.

    Parameters
    ----------
    patch : OrderedAtomPatch
        Heavy coordinates used as anchors.
    semantics : HydrogenSemantics
        Component placement program.
    include_backbone_hydrogen : bool
        Select the program variant containing backbone H.
    selected_hydrogen_names : frozenset[str] or None
        Evaluate operations producing at least one selected name. None selects all;
        a selected multi-output operation still returns all its outputs.

    Returns
    -------
    tuple[list[str], CoordinateBlock]
        Generated names and matching coordinates in program order.

    Raises
    ------
    ValueError
        No static program is available.
    KeyError
        A selected operation lacks an anchor.
    GeometryPlacementError
        Selected anchor geometry is invalid.
    """

    plan = semantics.static_plan(include_backbone_hydrogen=include_backbone_hydrogen)
    if plan is None:
        raise ValueError("static hydrogen semantics require a plan")

    atom_coordinates = patch.position_map()
    sidechain_atom_names: list[str] = []
    sidechain_coordinates: CoordinateBlock = []
    for output_names, method_name, arguments in plan:
        if (
            selected_hydrogen_names is not None
            and not selected_hydrogen_names.intersection(output_names)
        ):
            continue
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
    if site.is_disulfide_bonded():
        return evaluate_plan(DISULFIDE_CYSTEINE_PLAN, atom_coordinates)

    hydrogen = cysteine_thiol(
        atom_coordinates["SG"],
        atom_coordinates["CB"],
        atom_coordinates["CA"],
    )
    search = ROTATABLE_HYDROGEN_SPECS[RotatableHydrogenKind.CYS].search(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["SG"],
        hydrogen=hydrogen,
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
    search = ROTATABLE_HYDROGEN_SPECS[RotatableHydrogenKind.SER].search(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["OG"],
        hydrogen=initial_hydrogen,
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
    search = ROTATABLE_HYDROGEN_SPECS[RotatableHydrogenKind.THR].search(
        outer_anchor=atom_coordinates["CA"],
        inner_anchor=atom_coordinates["CB"],
        donor=atom_coordinates["OG1"],
        hydrogen=initial_hydrogen,
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
    search = ROTATABLE_HYDROGEN_SPECS[RotatableHydrogenKind.TYR].search(
        outer_anchor=atom_coordinates["CE2"],
        inner_anchor=atom_coordinates["CZ"],
        donor=atom_coordinates["OH"],
        hydrogen=initial_hydrogen,
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
