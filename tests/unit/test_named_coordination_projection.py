"""Named graph projection keeps covalent valence and dative direction separate."""

from collections.abc import Mapping

import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, Mol

from protrepair.chemistry.component.graph import (
    BondDefinition,
    CoordinationBondDefinition,
)
from protrepair.chemistry.inference import rdkit_graph
from protrepair.chemistry.inference.coordination_donor import NitrogenCoordinationDonor
from protrepair.chemistry.inference.rdkit_graph import (
    NativeBondProjection,
    covalent_bond_definitions_from_molecule,
    kekulized_molecule,
    native_bond_projection_from_molecule,
    require_aromatic_nitrogen_donor,
)
from protrepair.errors import (
    RdkitUnavailableError,
    UnsupportedCoordinationChemistryError,
)


def test_kekule_copy_keeps_oriented_dative_separate_from_covalent_projection() -> None:
    molecule = _molecule("[Fe+3]<-n1ccccc1", add_hydrogens=False)
    dative = next(
        bond for bond in molecule.GetBonds() if bond.GetBondType() == BondType.DATIVE
    )
    assert dative.GetBeginAtom().GetSymbol() == "N"
    assert dative.GetEndAtom().GetSymbol() == "Fe"

    projected = kekulized_molecule(molecule)

    bond = projected.GetBondWithIdx(dative.GetIdx())
    assert bond.GetBondType() == BondType.DATIVE
    assert (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) == (
        dative.GetBeginAtomIdx(),
        dative.GetEndAtomIdx(),
    )
    assert _atom_facts(projected) == _atom_facts(molecule)
    with pytest.raises(ValueError, match="not resolved covalent"):
        covalent_bond_definitions_from_molecule(
            molecule,
            atom_names_by_index={
                index: f"A{index}" for index in range(molecule.GetNumAtoms())
            },
        )


@pytest.mark.parametrize("smiles", ("[Fe+3]<-n1ccccc1", "[Fe+2]<-[n-]1cccc1"))
@pytest.mark.parametrize("reverse", (False, True))
def test_joint_projection_preserves_mapping_direction_and_integral_valence(
    smiles, reverse
):
    original = _molecule(smiles, add_hydrogens=False)
    order = (
        tuple(reversed(range(original.GetNumAtoms())))
        if reverse
        else tuple(range(original.GetNumAtoms()))
    )
    molecule = Chem.RenumberAtoms(original, order)
    names = {index: f"X{old}" for index, old in enumerate(order)}
    before = _atom_facts(molecule)
    projection = native_bond_projection_from_molecule(
        molecule, atom_names_by_index=names
    )

    assert projection.coordination_bonds == (CoordinationBondDefinition("X1", "X0"),)
    assert len(projection.covalent_bonds) == original.GetNumAtoms() - 1
    assert all(
        bond.aromatic and bond.order in (1, 2) for bond in projection.covalent_bonds
    )
    assert _atom_facts(molecule) == before
    require_aromatic_nitrogen_donor(molecule, order.index(1))
    with pytest.raises(ValueError, match="not resolved covalent"):
        covalent_bond_definitions_from_molecule(molecule, atom_names_by_index=names)


@pytest.mark.parametrize(
    "coordination",
    (
        (CoordinationBondDefinition("N", "FE"), CoordinationBondDefinition("FE", "N")),
        (CoordinationBondDefinition("N", "FE"),),
    ),
)
def test_joint_projection_rejects_duplicate_or_competing_physical_pairs(coordination):
    with pytest.raises(ValueError, match="competing endpoint pairs"):
        NativeBondProjection((BondDefinition("N", "FE"),), coordination)


@pytest.mark.parametrize("smiles", ("[Fe+3]<-[nH+]1ccccc1", "Cn1(->[Fe+3])cccc1"))
def test_native_donor_grammar_does_not_borrow_sanitize_permission(smiles):
    molecule = _molecule(smiles, add_hydrogens=False)
    donor = next(
        bond.GetBeginAtomIdx()
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DATIVE
    )
    with pytest.raises(ValueError, match="unprotonated aromatic"):
        require_aromatic_nitrogen_donor(molecule, donor)


@pytest.mark.parametrize("smiles", ("C#N", "CC(=O)[O-]", "[Na+].[Cl-]", "[He]"))
def test_non_aromatic_orders_and_disconnected_ions_are_not_reinterpreted(
    smiles: str,
) -> None:
    molecule = _molecule(smiles, add_hydrogens=False)
    names = {index: f"A{index}" for index in range(molecule.GetNumAtoms())}

    definitions = covalent_bond_definitions_from_molecule(
        molecule, atom_names_by_index=names
    )

    assert definitions == tuple(
        BondDefinition(
            names[bond.GetBeginAtomIdx()],
            names[bond.GetEndAtomIdx()],
            order=int(bond.GetBondTypeAsDouble()),
        )
        for bond in molecule.GetBonds()
    )


@pytest.mark.parametrize("smiles", ("c1ccc2ccccc2c1", "Cc1[nH+]cc[nH]1", "Cc1ncc[n-]1"))
def test_reordered_native_slots_preserve_mapped_chemical_identity(smiles: str) -> None:
    molecule = _molecule(smiles)
    old_by_new = tuple(reversed(range(molecule.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(molecule, old_by_new)
    names = {index: f"A{old_by_new[index]}" for index in range(reordered.GetNumAtoms())}

    definitions = covalent_bond_definitions_from_molecule(
        reordered, atom_names_by_index=names
    )

    reconstructed = Chem.RWMol()
    for original_atom in molecule.GetAtoms():
        atom = Chem.Atom(original_atom)
        atom.SetIsAromatic(False)
        reconstructed.AddAtom(atom)
    old_index_by_name = {f"A{index}": index for index in range(molecule.GetNumAtoms())}
    for definition in definitions:
        reconstructed.AddBond(
            old_index_by_name[definition.atom_name_1],
            old_index_by_name[definition.atom_name_2],
            {1: BondType.SINGLE, 2: BondType.DOUBLE}[definition.order],
        )
    Chem.SanitizeMol(reconstructed)
    assert _atom_facts(reconstructed) == _atom_facts(molecule)
    assert Chem.MolToSmiles(reconstructed) == Chem.MolToSmiles(molecule)


@pytest.mark.parametrize(
    "names",
    ({0: "A"}, {0: "A", 1: "B", 2: "C"}, {0: "A", 1: " a "}, {0: "A", 1: " "}),
)
def test_projection_refuses_missing_extra_colliding_or_blank_names(
    names: Mapping[int, str],
) -> None:
    with pytest.raises(ValueError, match="atom.name"):
        covalent_bond_definitions_from_molecule(
            _molecule("CO", add_hydrogens=False), atom_names_by_index=names
        )


@pytest.mark.parametrize(
    "bond_type",
    (BondType.DATIVE, BondType.ZERO, BondType.UNSPECIFIED, BondType.ONEANDAHALF),
)
def test_projection_does_not_round_or_drop_unrepresentable_relationships(
    bond_type: BondType,
) -> None:
    molecule = Chem.RWMol()
    molecule.AddAtom(Chem.Atom("N"))
    molecule.AddAtom(Chem.Atom("Fe"))
    molecule.AddBond(0, 1, bond_type)

    with pytest.raises(ValueError, match="not resolved covalent"):
        covalent_bond_definitions_from_molecule(
            molecule, atom_names_by_index={0: "N", 1: "FE"}
        )
    assert molecule.GetBondWithIdx(0).GetBondType() == bond_type


@pytest.mark.parametrize("smarts", ("[#6]-[#8]", "C-,=C"))
def test_query_graph_is_not_published_as_resolved_chemistry(smarts: str) -> None:
    molecule = Chem.MolFromSmarts(smarts)
    assert molecule is not None
    with pytest.raises(ValueError, match="query chemistry|not resolved covalent"):
        covalent_bond_definitions_from_molecule(
            molecule, atom_names_by_index={0: "C1", 1: "O1"}
        )


def _molecule(smiles: str, *, add_hydrogens: bool = True) -> Mol:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.AddHs(molecule) if add_hydrogens else molecule


def _atom_facts(
    molecule: Mol,
) -> tuple[tuple[int, int, int, int, bool, int, int, bool], ...]:
    return tuple(
        (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            int(atom.GetChiralTag()),
            atom.GetNoImplicit(),
            atom.GetNumExplicitHs(),
            atom.GetNumImplicitHs(),
            atom.GetIsAromatic(),
        )
        for atom in molecule.GetAtoms()
    )


def test_joint_projection_reports_unavailable_native_runtime(monkeypatch):
    molecule = _molecule("CO", add_hydrogens=False)
    monkeypatch.setattr(rdkit_graph, "Chem", None)
    with pytest.raises(RdkitUnavailableError):
        native_bond_projection_from_molecule(
            molecule, atom_names_by_index={0: "C", 1: "O"}
        )


@pytest.mark.parametrize(
    "smiles", ("CS(=O)(=O)[NH-]", "CCS(=O)(=O)[NH-]", "c1ccccc1S(=O)(=O)[NH-]")
)
def test_primary_sulfonamidate_has_one_h_and_only_the_admitted_partner(smiles):
    molecule = _molecule(smiles, add_hydrogens=False)
    donor = next(
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 7
    )
    role = NitrogenCoordinationDonor.from_molecule(molecule, donor)
    assert role is NitrogenCoordinationDonor.PRIMARY_SULFONAMIDATE
    assert role.hydrogen_count == 1
    assert role.admits_metal("ZN")
    assert not role.admits_metal("FE")
    assert NitrogenCoordinationDonor.from_molecule(Chem.AddHs(molecule), donor) is role


@pytest.mark.parametrize(
    "smiles",
    ("CS(=O)(=O)N", "CS(=O)(=O)[N-]C", "CS(=O)[NH-]", "COS(=O)(=O)[NH-]", "C[NH-]"),
)
def test_similar_unadmitted_nitrogen_motifs_are_not_sulfonamidate_donors(smiles):
    molecule = _molecule(smiles, add_hydrogens=False)
    donor = next(
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 7
    )
    with pytest.raises(UnsupportedCoordinationChemistryError):
        NitrogenCoordinationDonor.from_molecule(molecule, donor)
