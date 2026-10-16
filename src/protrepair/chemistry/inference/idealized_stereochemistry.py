"""Native validation of explicit component graphs and stereo references."""

from functools import lru_cache
from math import isfinite
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by backend failure tests
    Chem = None

from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.errors import RdkitUnavailableError

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
