"""Native validation of explicit component graphs and stereo references."""

from collections import Counter
from functools import lru_cache
from math import isfinite
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by backend failure tests
    Chem = None

from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.chemistry.inference.rdkit_graph import (
    native_bond_projection_from_molecule,
    require_compatible_named_pose_stereochemistry,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.structure.geometry import ResidueGeometry

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


def require_compatible_component_reference(component: IdealizedComponent) -> None:
    """Validate a complete explicit graph and its declared reference stereo.

    Parameters
    ----------
    component : IdealizedComponent
        Selected precursor, including every explicit hydrogen. Declared centers
        require finite ideal coordinates for the native reference conformer.

    Raises
    ------
    ValueError
        The graph is disconnected, radical, valence-invalid, or its reference
        geometry disagrees with a declared R/S center.
    RdkitUnavailableError
        Native graph interpretation is unavailable.
    """
    if Chem is None:
        raise RdkitUnavailableError("component reference requires operational RDKit")
    try:
        _component_reference_molecule(component)
    except RuntimeError as error:
        raise ValueError(f"invalid component reference graph: {error}") from error


def component_reference_molecule(component: IdealizedComponent) -> "Mol":
    """Project a validated explicit component into a defensive native graph.

    Parameters
    ----------
    component : IdealizedComponent
        Complete named graph, H and charges, with any declared reference stereo.

    Returns
    -------
    Mol
        Separate sanitized graph in component atom order.

    Raises
    ------
    ValueError
        Graph or declared reference stereochemistry is incompatible.
    RdkitUnavailableError
        Native interpretation is unavailable.
    """
    require_compatible_component_reference(component)
    assert Chem is not None
    return Chem.Mol(_component_reference_molecule(component))


def require_compatible_component_pose(
    component: IdealizedComponent,
    geometry: ResidueGeometry,
    *,
    hydrogen_attachments: tuple[tuple[str, str], ...],
    bond_order_constraints: dict[tuple[str, str], int],
) -> None:
    """Validate a named pose and declared orders without inferring chemistry.

    Parameters
    ----------
    component : IdealizedComponent
        Full explicit reference graph.
    geometry : ResidueGeometry
        Current named coordinates.
    hydrogen_attachments : tuple[tuple[str, str], ...]
        Interpreted current H names and their reference parents.
    bond_order_constraints : dict[tuple[str, str], int]
        Known source/current heavy orders; unknown orders are omitted.

    Raises
    ------
    ValueError
        Reference, source orders or confirmed pose stereo disagree.
    RdkitUnavailableError
        Native validation is unavailable.
    """
    molecule = component_reference_molecule(component)
    assert Chem is not None
    native_bond_projection_from_molecule(
        molecule,
        atom_names_by_index={
            i: atom.atom_name for i, atom in enumerate(component.atoms)
        },
        bond_order_constraints=bond_order_constraints,
    )
    heavy = _component_heavy_pose_molecule(component, molecule)
    require_compatible_named_pose_stereochemistry(
        heavy,
        component.heavy_atom_names(),
        geometry,
        hydrogen_attachments=hydrogen_attachments,
    )


def component_reference_hydrogen_bond_lengths(
    component: IdealizedComponent,
    geometry: ResidueGeometry,
    *,
    bond_order_constraints: dict[tuple[str, str], int],
) -> tuple[tuple[str, float], ...]:
    """Prepare temporary reference H lengths on fixed current heavy coordinates.

    Parameters
    ----------
    component : IdealizedComponent
        Validated full reference graph, including its explicit H inventory.
    geometry : ResidueGeometry
        Current named heavy coordinates; source H positions are not replaced.
    bond_order_constraints : dict[tuple[str, str], int]
        Known heavy orders that must remain compatible with the reference.

    Returns
    -------
    tuple[tuple[str, float], ...]
        Heavy-parent name and length in angstroms for each temporary H. Parent
        counts match the reference; no generated coordinates are published.

    Raises
    ------
    ValueError
        Reference/orders, coordinate mapping or temporary H preparation fail.
    RdkitUnavailableError
        Native preparation is unavailable.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("reference H correspondence requires RDKit")
    molecule = component_reference_molecule(component)
    native_bond_projection_from_molecule(
        molecule,
        atom_names_by_index={
            i: atom.atom_name for i, atom in enumerate(component.atoms)
        },
        bond_order_constraints=bond_order_constraints,
    )
    heavy = _component_heavy_pose_molecule(component, molecule)
    names = component.heavy_atom_names()
    if heavy.GetNumAtoms() != len(names):
        raise ValueError("reference H removal did not preserve the heavy inventory")
    conformer = chem.Conformer(len(names))
    conformer.Set3D(True)
    for index, name in enumerate(names):
        if not geometry.has_atom(name):
            raise ValueError(f"reference correspondence has no coordinate for {name}")
        position = geometry.position(name)
        if not all(isfinite(value) for value in position):
            raise ValueError(
                "reference correspondence requires finite heavy coordinates"
            )
        conformer.SetAtomPosition(index, tuple(position))
    heavy.AddConformer(conformer)
    try:
        prepared = chem.AddHs(heavy, addCoords=True)
    except RuntimeError as error:
        raise ValueError(f"reference H preparation failed: {error}") from error
    positions = prepared.GetConformer()
    if any(
        tuple(positions.GetAtomPosition(index)) != tuple(geometry.position(name))
        for index, name in enumerate(names)
    ):
        raise ValueError("reference H preparation moved fixed heavy coordinates")
    lengths = []
    for atom in prepared.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbors = tuple(atom.GetNeighbors())
        if len(neighbors) != 1 or neighbors[0].GetIdx() >= len(names):
            raise ValueError("temporary H requires one reference heavy parent")
        parent = names[neighbors[0].GetIdx()]
        position = Vec3.from_iterable(positions.GetAtomPosition(atom.GetIdx()))
        distance = position.distance_to(geometry.position(parent))
        if not isfinite(distance) or distance <= 0:
            raise ValueError("reference H preparation produced an invalid bond length")
        lengths.append((parent, distance))
    if Counter(parent for parent, _ in lengths) != Counter(
        component.hydrogen_anchor_atom_name(atom.atom_name)
        for atom in component.hydrogen_atoms()
    ):
        raise ValueError("prepared H parents do not match the reference inventory")
    return tuple(lengths)


def _component_heavy_pose_molecule(
    component: IdealizedComponent, molecule: "Mol"
) -> "Mol":
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("component pose projection requires RDKit")
    for native, atom in zip(molecule.GetAtoms(), component.atoms, strict=True):
        if atom.stereo_config is None:
            native.SetChiralTag(chem.ChiralType.CHI_UNSPECIFIED)
    for bond in molecule.GetBonds():
        bond.SetStereo(chem.BondStereo.STEREONONE)
        bond.SetBondDir(chem.BondDir.NONE)
    molecule.RemoveAllConformers()
    removal = chem.RemoveHsParameters()
    removal.removeIsotopes = True
    removal.removeDefiningBondStereo = True
    removal.updateExplicitCount = True
    return chem.RemoveHs(molecule, removal)


@lru_cache(maxsize=128)
def _component_reference_molecule(component: IdealizedComponent) -> "Mol":
    assert Chem is not None
    molecule = Chem.RWMol()
    indices = {atom.atom_name: i for i, atom in enumerate(component.atoms)}
    for atom in component.atoms:
        native = Chem.Atom("H" if atom.is_hydrogen() else atom.element.title())
        if atom.element in {"D", "T"}:
            native.SetIsotope(2 if atom.element == "D" else 3)
        native.SetFormalCharge(atom.formal_charge)
        native.SetNoImplicit(True)
        molecule.AddAtom(native)
    orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for bond in component.bonds:
        order = orders.get(bond.order)
        if order is None:
            raise ValueError("component reference has unsupported bond order")
        if bond.aromatic:
            order = Chem.BondType.AROMATIC
        first, second = indices[bond.atom_name_1], indices[bond.atom_name_2]
        if molecule.GetBondBetweenAtoms(first, second) is not None:
            raise ValueError("component reference has duplicate bonds")
        molecule.AddBond(first, second, order)
        if bond.aromatic:
            molecule.GetBondBetweenAtoms(first, second).SetIsAromatic(True)
            molecule.GetAtomWithIdx(first).SetIsAromatic(True)
            molecule.GetAtomWithIdx(second).SetIsAromatic(True)

    native_graph = molecule.GetMol()
    Chem.SanitizeMol(native_graph)
    if len(Chem.GetMolFrags(native_graph)) != 1 or any(
        atom.GetNumRadicalElectrons() for atom in native_graph.GetAtoms()
    ):
        raise ValueError("component reference requires one closed-shell scaffold")
    declared = tuple(atom for atom in component.atoms if atom.stereo_config)
    if not declared:
        return native_graph
    conformer = Chem.Conformer(len(component.atoms))
    conformer.Set3D(True)
    for atom in component.atoms:
        if atom.ideal_position is None or not all(
            isfinite(value) for value in atom.ideal_position
        ):
            raise ValueError("component stereo requires finite reference coordinates")
        conformer.SetAtomPosition(indices[atom.atom_name], atom.ideal_position)
    native_graph.AddConformer(conformer)
    Chem.AssignStereochemistryFrom3D(native_graph)
    Chem.AssignStereochemistry(native_graph, cleanIt=True, force=True)
    for atom in declared:
        native = native_graph.GetAtomWithIdx(indices[atom.atom_name])
        if (
            not native.HasProp("_CIPCode")
            or native.GetProp("_CIPCode") != atom.stereo_config
        ):
            raise ValueError(
                f"component reference stereo disagrees at {atom.atom_name}"
            )
    return native_graph
