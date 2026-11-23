"""Validate and project explicit named component graphs without inventing hydrogens."""

from functools import lru_cache
from typing import TYPE_CHECKING

from protrepair.chemistry.retained_non_polymer.definition import (
    RetainedComponentDefinition,
)
from protrepair.errors import RdkitUnavailableError

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

try:
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
except ImportError:  # pragma: no cover
    Chem = None
    rdCIPLabeler = None


def component_heavy_molecule(definition: RetainedComponentDefinition) -> "Mol":
    """Project the named dictionary to a native heavy graph.

    Parameters
    ----------
    definition : RetainedComponentDefinition
        Explicit atom, bond, charge, hydrogen and stereo inventory.

    Returns
    -------
    Mol
        A defensive copy with the dictionary's H counts and declared stereo.

    Raises
    ------
    ValueError
        The graph cannot retain its declared chemistry.
    RdkitUnavailableError
        The required native backend is unavailable.
    """
    if Chem is None:
        raise RdkitUnavailableError("reading named component chemistry requires RDKit")
    return Chem.Mol(_validated_heavy_graph(definition))


@lru_cache(maxsize=128)
def _validated_heavy_graph(definition: RetainedComponentDefinition) -> "Mol":
    assert Chem is not None
    mol = Chem.RWMol()
    indices = {}
    for row in definition.atoms:
        atom = Chem.Atom(row.element.capitalize())
        atom.SetFormalCharge(row.charge)
        if atom.GetFormalCharge() != row.charge:
            raise ValueError(f"unrepresentable formal charge at {row.name}")
        atom.SetNoImplicit(True)
        indices[row.name] = mol.AddAtom(atom)

    kinds = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "AROM": Chem.BondType.AROMATIC,
    }
    for bond in definition.bonds:
        mol.AddBond(indices[bond.first], indices[bond.second], kinds[bond.order])
    Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("disconnected component")
    check = Chem.Mol(mol)
    for atom in check.GetAtoms():
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(False)
    check.UpdatePropertyCache(strict=True)
    for row, atom in zip(definition.atoms, mol.GetAtoms(), strict=True):
        if (
            atom.GetNumRadicalElectrons()
            or check.GetAtomWithIdx(atom.GetIdx()).GetNumImplicitHs()
        ):
            raise ValueError(
                f"incomplete H inventory or unsupported valence at {row.name}"
            )
        if row.aromatic is not None and row.aromatic != atom.GetIsAromatic():
            raise ValueError(f"cannot represent declared aromaticity at {row.name}")
        if row.element == "H" and (
            row.charge
            or atom.GetDegree() != 1
            or atom.GetNeighbors()[0].GetAtomicNum() == 1
        ):
            raise ValueError(f"invalid ordinary hydrogen at {row.name}")
    for row in definition.bonds:
        edge = mol.GetBondBetweenAtoms(indices[row.first], indices[row.second])
        if row.aromatic is not None and row.aromatic != edge.GetIsAromatic():
            raise ValueError(
                f"cannot represent declared aromaticity at {row.first}-{row.second}"
            )

    # Remove only explicit graph H; native AddHs must later restore exactly those counts.
    params = Chem.RemoveHsParameters()
    heavy = Chem.RemoveHs(mol, params, sanitize=False)
    counts = definition.hydrogen_slots()
    for name, atom in zip(definition.heavy_atom_names, heavy.GetAtoms(), strict=True):
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(len(counts.get(name, ())))
    Chem.SanitizeMol(heavy)
    _set_declared_stereo(heavy, definition)
    return heavy


def _set_declared_stereo(mol: "Mol", definition: RetainedComponentDefinition) -> None:
    assert Chem is not None and rdCIPLabeler is not None
    indices = {name: i for i, name in enumerate(definition.heavy_atom_names)}
    atoms = []
    bonds = []
    for row in definition.atoms:
        if row.stereo is None:
            continue
        if row.name not in indices:
            raise ValueError(f"invalid atom stereo at {row.name}")
        atom = mol.GetAtomWithIdx(indices[row.name])
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        atoms.append((atom, row.stereo, row.name))
    for edge in definition.bonds:
        if edge.stereo is None:
            continue
        if edge.order != "DOUB":
            raise ValueError(f"non-double bond with E/Z at {edge.first}-{edge.second}")
        bond = mol.GetBondBetweenAtoms(indices[edge.first], indices[edge.second])
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        defining = [
            next(
                (
                    a.GetIdx()
                    for a in atom.GetNeighbors()
                    if a.GetIdx() != other.GetIdx()
                ),
                None,
            )
            for atom, other in ((left, right), (right, left))
        ]
        if defining[0] is None or defining[1] is None:
            raise ValueError(f"unrepresentable E/Z at {edge.first}-{edge.second}")
        bond.SetStereoAtoms(defining[0], defining[1])
        bond.SetStereo(Chem.BondStereo.STEREOTRANS)
        bonds.append((bond, edge.stereo, f"{edge.first}-{edge.second}"))
    if not atoms and not bonds:
        return

    # A centre may only become distinguishable once stereo on its branches is set.
    # Clear native labels between passes; absence of a fresh label is significant.
    def assign_cip() -> None:
        for item, _, _ in (*atoms, *bonds):
            item.ClearProp("_CIPCode")
        assert rdCIPLabeler is not None
        rdCIPLabeler.AssignCIPLabels(mol)

    for _ in range(len(atoms) + len(bonds) + 1):
        assign_cip()
        wrong_bonds = [
            (b, want)
            for b, want, _ in bonds
            if b.HasProp("_CIPCode") and b.GetProp("_CIPCode") != want
        ]
        if wrong_bonds:
            for bond, _ in wrong_bonds:
                bond.SetStereo(
                    Chem.BondStereo.STEREOCIS
                    if bond.GetStereo() == Chem.BondStereo.STEREOTRANS
                    else Chem.BondStereo.STEREOTRANS
                )
            continue
        wrong_atoms = [
            a
            for a, want, _ in atoms
            if a.HasProp("_CIPCode") and a.GetProp("_CIPCode") != want
        ]
        if not wrong_atoms:
            break
        for atom in wrong_atoms:
            atom.InvertChirality()
    Chem.SetDoubleBondNeighborDirections(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    # CIP assignment rewrites E/Z into its relative cis/trans encoding. Verify
    # a copy so the published graph retains the pose comparator's encoding.
    checked = Chem.Mol(mol)
    for item in (*checked.GetAtoms(), *checked.GetBonds()):
        item.ClearProp("_CIPCode")
    rdCIPLabeler.AssignCIPLabels(checked)
    for atom, label, name in atoms:
        verified = checked.GetAtomWithIdx(atom.GetIdx())
        if (
            not atom.GetChiralTag()
            or not verified.HasProp("_CIPCode")
            or verified.GetProp("_CIPCode") != label
        ):
            raise ValueError(f"cannot represent declared {label} at {name}")
    for bond, label, name in bonds:
        verified = checked.GetBondWithIdx(bond.GetIdx())
        if (
            bond.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
            or not verified.HasProp("_CIPCode")
            or verified.GetProp("_CIPCode") != label
        ):
            raise ValueError(f"cannot represent declared {label} at {name}")
