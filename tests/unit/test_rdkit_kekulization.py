"""Constrained aromatic projection preserves selected native chemistry."""

from collections.abc import Mapping
from itertools import product

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import BondType, Mol

from protrepair.chemistry.inference import rdkit_graph
from protrepair.chemistry.inference.rdkit_graph import kekulized_molecule
from protrepair.errors import BondOrderConstraintError, RdkitUnavailableError


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccccc1",
        "c1ccc2ccccc2c1",
        "Cc1[nH+]cc[nH]1",
        "Cc1ncc[n-]1",
        "[2H]c1ccc([C@H](F)Cl)cc1",
        "[3H]c1ccccc1",
        "F/C=C/c1ccccc1",
        "[Na+].[O-]c1ccccc1",
    ),
)
def test_kekule_copy_preserves_atoms_stereo_conformers_and_native_input(
    smiles: str,
) -> None:
    molecule = _molecule(smiles)
    molecule.SetProp("source", "original")
    for conformer_id in range(2):
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        conformer.Set3D(True)
        for index in range(molecule.GetNumAtoms()):
            conformer.SetAtomPosition(index, (index * 1.1, conformer_id, -index * 0.2))
        molecule.AddConformer(conformer, assignId=True)
    atoms_before = _atom_facts(molecule)
    bond_types_before = tuple(bond.GetBondType() for bond in molecule.GetBonds())
    aromatic_before = tuple(bond.GetIsAromatic() for bond in molecule.GetBonds())
    stereo_before = tuple(
        (bond.GetStereo(), tuple(bond.GetStereoAtoms())) for bond in molecule.GetBonds()
    )

    projected = kekulized_molecule(molecule)

    assert projected is not molecule
    assert _atom_facts(projected) == atoms_before == _atom_facts(molecule)
    assert (
        tuple(bond.GetBondType() for bond in molecule.GetBonds()) == bond_types_before
    )
    assert (
        tuple(bond.GetIsAromatic() for bond in projected.GetBonds()) == aromatic_before
    )
    assert (
        tuple(
            (bond.GetStereo(), tuple(bond.GetStereoAtoms()))
            for bond in projected.GetBonds()
        )
        == stereo_before
    )
    assert {bond.GetBondType() for bond in projected.GetBonds()} <= {
        BondType.SINGLE,
        BondType.DOUBLE,
    }
    for before, after in zip(
        molecule.GetConformers(), projected.GetConformers(), strict=True
    ):
        assert before.GetId() == after.GetId()
        assert before.Is3D() == after.Is3D()
        np.testing.assert_array_equal(before.GetPositions(), after.GetPositions())
    projected.GetAtomWithIdx(0).SetFormalCharge(7)
    projected.GetConformer().SetAtomPosition(0, (99, 99, 99))
    projected.SetProp("source", "changed")
    assert _atom_facts(molecule) == atoms_before
    assert molecule.GetProp("source") == "original"
    assert tuple(molecule.GetConformer().GetAtomPosition(0)) == (0, 0, 0)


def test_failed_kekulization_keeps_the_original_graph_unchanged() -> None:
    molecule = Chem.MolFromSmiles("c1cccc1", sanitize=False)
    assert molecule is not None
    before = Chem.MolToMolBlock(molecule, kekulize=False)

    with pytest.raises(ValueError, match="no compatible Kekule"):
        kekulized_molecule(molecule)

    assert Chem.MolToMolBlock(molecule, kekulize=False) == before


def test_native_unavailability_uses_the_existing_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molecule = _molecule("CO", add_hydrogens=False)
    monkeypatch.setattr(rdkit_graph, "Chem", None)
    with pytest.raises(RdkitUnavailableError):
        kekulized_molecule(molecule)


def test_every_partial_benzene_constraint_matches_the_two_resonance_forms() -> None:
    molecule = _molecule("c1ccccc1")
    ring = tuple(b for b in molecule.GetBonds() if b.GetIsAromatic())
    baseline = kekulized_molecule(molecule)
    orders = tuple(
        int(baseline.GetBondWithIdx(b.GetIdx()).GetBondTypeAsDouble()) for b in ring
    )
    forms = (orders, tuple(3 - order for order in orders))
    with rdBase.BlockLogs():
        for choices in product((0, 1, 2), repeat=len(ring)):
            constraints = {
                (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()): order
                for bond, order in zip(ring, choices, strict=True)
                if order
            }
            compatible = tuple(
                form
                for form in forms
                if all(
                    not requested or requested == actual
                    for requested, actual in zip(choices, form, strict=True)
                )
            )
            if not compatible:
                with pytest.raises(BondOrderConstraintError):
                    kekulized_molecule(molecule, bond_order_constraints=constraints)
                continue
            projected = kekulized_molecule(molecule, bond_order_constraints=constraints)
            observed = tuple(
                int(projected.GetBondWithIdx(b.GetIdx()).GetBondTypeAsDouble())
                for b in ring
            )
            assert observed in compatible
            assert _atom_facts(projected) == _atom_facts(molecule)
            assert all(
                b.GetIsAromatic()
                for b in projected.GetBonds()
                if b.GetIdx() in {r.GetIdx() for r in ring}
            )
    assert all(b.GetBondType() == BondType.AROMATIC for b in ring)


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccc2ccccc2c1",
        "[2H]c1ccc([C@H](F)Cl)cc1",
        "[3H]c1ccccc1",
        "F/C=C/c1ccccc1",
        "[O-][n+]1ccccc1",
        "[Fe+3]<-n1ccccc1",
    ),
)
def test_constrained_copy_preserves_resolved_chemistry_and_oriented_coordination(
    smiles: str,
) -> None:
    molecule = _molecule(smiles)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (index, -index, index * 0.25))
    molecule.AddConformer(conformer)
    baseline = kekulized_molecule(molecule)
    selected = next(b for b in baseline.GetBonds() if b.GetIsAromatic())
    requested = 3 - int(selected.GetBondTypeAsDouble())
    before = Chem.MolToMolBlock(molecule, kekulize=False)

    projected = kekulized_molecule(
        molecule,
        bond_order_constraints={
            (selected.GetEndAtomIdx(), selected.GetBeginAtomIdx()): requested,
        },
    )

    assert (
        projected.GetBondWithIdx(selected.GetIdx()).GetBondTypeAsDouble() == requested
    )
    assert _atom_facts(projected) == _atom_facts(molecule)
    np.testing.assert_array_equal(
        projected.GetConformer().GetPositions(), molecule.GetConformer().GetPositions()
    )
    assert Chem.MolToMolBlock(molecule, kekulize=False) == before
    for original, actual in zip(molecule.GetBonds(), projected.GetBonds(), strict=True):
        assert (
            actual.GetStereo(),
            tuple(actual.GetStereoAtoms()),
            actual.GetBondDir(),
        ) == (
            original.GetStereo(),
            tuple(original.GetStereoAtoms()),
            original.GetBondDir(),
        )
        if not original.GetIsAromatic():
            assert actual.GetBondType() == original.GetBondType()
            assert (actual.GetBeginAtomIdx(), actual.GetEndAtomIdx()) == (
                original.GetBeginAtomIdx(),
                original.GetEndAtomIdx(),
            )
    sanitized = Chem.Mol(projected)
    Chem.SanitizeMol(sanitized)
    assert Chem.MolToSmiles(sanitized) == Chem.MolToSmiles(molecule)


@pytest.mark.parametrize("smiles", ("c1cc[nH]c1", "Cc1[nH+]cc[nH]1", "Cc1ncc[n-]1"))
def test_constraints_do_not_move_heterocyclic_protons_or_charges(smiles: str) -> None:
    molecule = _molecule(smiles)
    baseline = kekulized_molecule(molecule)
    constraints = {
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): int(b.GetBondTypeAsDouble())
        for b in baseline.GetBonds()
        if b.GetIsAromatic()
    }
    assert _atom_facts(
        kekulized_molecule(molecule, bond_order_constraints=constraints)
    ) == _atom_facts(molecule)
    nitrogen = next(a for a in baseline.GetAtoms() if a.GetSymbol() == "N")
    edge = next(b for b in nitrogen.GetBonds() if b.GetIsAromatic())
    constraints[(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())] = 3 - int(
        edge.GetBondTypeAsDouble()
    )
    with pytest.raises(BondOrderConstraintError):
        kekulized_molecule(molecule, bond_order_constraints=constraints)
    assert _atom_facts(molecule) == _atom_facts(baseline)


@pytest.mark.parametrize(
    "constraints",
    (
        {(0, 0): 1},
        {(-1, 1): 1},
        {(0, 6): 1},
        {(False, 1): 1},
        {(0, 1): True},
        {(0, 1): 0},
        {(0, 1): 3},
        {(0, 3): 1},
        {(0, 1): 1, (1, 0): 2},
    ),
)
def test_invalid_constraints_cannot_mutate_native_input(
    constraints: Mapping[tuple[int, int], int],
) -> None:
    molecule = _molecule("c1ccccc1", add_hydrogens=False)
    before = Chem.MolToMolBlock(molecule, kekulize=False)
    with pytest.raises(BondOrderConstraintError):
        kekulized_molecule(molecule, bond_order_constraints=constraints)
    assert Chem.MolToMolBlock(molecule, kekulize=False) == before


@pytest.mark.parametrize(
    "smiles,pair,order",
    (("CO", (0, 1), 2), ("C#N", (0, 1), 1), ("[Fe+3]<-n1ccccc1", (0, 1), 1)),
)
def test_constraints_cannot_reinterpret_non_aromatic_or_dative_edges(
    smiles: str, pair: tuple[int, int], order: int
) -> None:
    with pytest.raises(BondOrderConstraintError):
        kekulized_molecule(_molecule(smiles), bond_order_constraints={pair: order})


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
