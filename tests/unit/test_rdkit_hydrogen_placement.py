"""Native-only generated H correction on independently deposited geometry."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from protrepair.chemistry.inference.rdkit_graph import (
    has_declared_stereochemistry,
    molecule_with_pose_stereochemistry,
    require_compatible_pose_stereochemistry,
)
from protrepair.chemistry.inference.rdkit_hydrogen_geometry import (
    add_hydrogens_preserving_stereochemistry,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    template_without_hydrogens,
)
from protrepair.io import read_structure

SMILES = "[P@]1(=O)([O-])OC[C@H]2O[C@@H](n3cnc4c(N)ncnc34)[C@H](O)[C@@H]2O1"
NAMES = (
    "P",
    "O1P",
    "O2P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C1'",
    "N9",
    "C8",
    "N7",
    "C5",
    "C6",
    "N6",
    "N1",
    "C2",
    "N3",
    "C4",
    "C2'",
    "O2'",
    "C3'",
    "O3'",
)


def _pose(index):
    source = read_structure(Path("tests/fixtures/corpus/pdb1hw5.ent"))
    residues = tuple(r for r in source.constitution.ligands if r.component_id == "CMP")
    residue = residues[index]
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    molecule = template_without_hydrogens(SMILES)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for i, name in enumerate(NAMES):
        conformer.SetAtomPosition(i, tuple(geometry.position(name)))
    molecule.AddConformer(conformer)
    return molecule


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("source_stereo", (False, True))
def test_generated_h_preserves_deposited_stereo_positions_and_native_lengths(
    index, source_stereo
):
    pose = _pose(index)
    if source_stereo:
        pose = molecule_with_pose_stereochemistry(pose)
    before = tuple(
        tuple(pose.GetConformer().GetAtomPosition(i)) for i in range(pose.GetNumAtoms())
    )
    raw = Chem.AddHs(pose, addCoords=True)
    raw_observation = molecule_with_pose_stereochemistry(raw)
    mismatched = {
        a.GetIdx()
        for a, b in zip(raw.GetAtoms(), raw_observation.GetAtoms(), strict=True)
        if a.GetChiralTag()
        and b.GetChiralTag()
        and a.GetChiralTag() != b.GetChiralTag()
    }
    corrected = add_hydrogens_preserving_stereochemistry(pose)
    require_compatible_pose_stereochemistry(corrected)
    assert corrected.GetNumAtoms() == raw.GetNumAtoms()
    assert (
        tuple(
            tuple(pose.GetConformer().GetAtomPosition(i))
            for i in range(pose.GetNumAtoms())
        )
        == before
    )
    for i, point in enumerate(before):
        assert tuple(corrected.GetConformer().GetAtomPosition(i)) == point
    for atom in corrected.GetAtoms():
        if atom.GetIdx() < pose.GetNumAtoms():
            continue
        parent = atom.GetNeighbors()[0].GetIdx()
        current = np.array(corrected.GetConformer().GetAtomPosition(atom.GetIdx()))
        original = np.array(raw.GetConformer().GetAtomPosition(atom.GetIdx()))
        center = np.array(corrected.GetConformer().GetAtomPosition(parent))
        assert np.linalg.norm(current - center) == pytest.approx(
            np.linalg.norm(original - center)
        )
        if parent not in mismatched:
            assert np.array_equal(current, original)
    # This geometric check does not reuse the native stereo comparator.
    for index in mismatched:
        atom = corrected.GetAtomWithIdx(index)
        anchors = [
            a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() < pose.GetNumAtoms()
        ]
        hydrogen = next(
            a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() >= pose.GetNumAtoms()
        )
        conformer = corrected.GetConformer()
        center = np.array(conformer.GetAtomPosition(index))
        original_vectors = np.array(
            [np.array(conformer.GetAtomPosition(i)) - center for i in anchors]
        )
        h_point = np.array(conformer.GetAtomPosition(hydrogen))
        tetra_vectors = np.array(
            [np.array(conformer.GetAtomPosition(i)) - h_point for i in anchors]
        )
        assert np.linalg.det(original_vectors) * np.linalg.det(tetra_vectors) > 0


@pytest.mark.parametrize("smiles", ("F/C=C/F", "F/C=C\\F"))
@pytest.mark.parametrize("opposite", (False, True))
def test_double_bond_pose_is_checked_without_changing_source(smiles, opposite):
    molecule = Chem.MolFromSmiles(smiles)
    selected_y = -1.0 if smiles == "F/C=C/F" else 1.0
    if opposite:
        selected_y *= -1.0
    points = (
        (-1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.3, 0.0, 0.0),
        (2.3, selected_y, 0.0),
    )
    conformer = Chem.Conformer(4)
    for index, point in enumerate(points):
        conformer.SetAtomPosition(index, point)
    molecule.AddConformer(conformer)
    source_smiles = Chem.MolToSmiles(molecule)
    if opposite:
        with pytest.raises(ValueError, match="bond stereochemistry"):
            add_hydrogens_preserving_stereochemistry(molecule)
    else:
        result = add_hydrogens_preserving_stereochemistry(molecule)
        require_compatible_pose_stereochemistry(result)
        assert result.GetNumAtoms() == 6
        for index, point in enumerate(points):
            assert tuple(result.GetConformer().GetAtomPosition(index)) == point
    assert Chem.MolToSmiles(molecule) == source_smiles
    for index, point in enumerate(points):
        assert tuple(molecule.GetConformer().GetAtomPosition(index)) == point


def test_undefined_stereo_does_not_create_an_opposite_pose_constraint():
    molecule = _pose(1)
    Chem.RemoveStereochemistry(molecule)
    assert not has_declared_stereochemistry(molecule)
    require_compatible_pose_stereochemistry(molecule)
    result = add_hydrogens_preserving_stereochemistry(molecule)
    raw = Chem.AddHs(molecule, addCoords=True)
    assert np.array_equal(
        result.GetConformer().GetPositions(), raw.GetConformer().GetPositions()
    )


@pytest.mark.parametrize("order", ("reverse", "rotate", "identity"))
def test_tetrahedral_placement_respects_native_atom_permutations(order):
    molecule = _pose(1)
    indices = list(range(molecule.GetNumAtoms()))
    if order == "reverse":
        indices.reverse()
    elif order == "rotate":
        indices = indices[3:] + indices[:3]
    molecule = Chem.RenumberAtoms(molecule, indices)
    before = molecule.GetConformer().GetPositions().copy()
    result = add_hydrogens_preserving_stereochemistry(molecule)
    require_compatible_pose_stereochemistry(result)
    assert np.array_equal(result.GetConformer().GetPositions()[: len(indices)], before)
    assert np.array_equal(molecule.GetConformer().GetPositions(), before)


@pytest.mark.parametrize("isotope", (0, 2, 3))
def test_repeated_hydrogenation_keeps_existing_hydrogen_isotopes_and_positions(isotope):
    source = add_hydrogens_preserving_stereochemistry(_pose(1))
    hydrogen = next(atom for atom in source.GetAtoms() if atom.GetAtomicNum() == 1)
    hydrogen.SetIsotope(isotope)
    positions = source.GetConformer().GetPositions().copy()
    result = add_hydrogens_preserving_stereochemistry(source)
    assert result.GetNumAtoms() == source.GetNumAtoms()
    assert result.GetAtomWithIdx(hydrogen.GetIdx()).GetIsotope() == isotope
    assert np.array_equal(result.GetConformer().GetPositions(), positions)


def test_confirmed_mirrored_heavy_pose_is_rejected_without_moving_it():
    molecule = _pose(1)
    conformer = molecule.GetConformer()
    for index in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (-point.x, point.y, point.z))
    before = conformer.GetPositions().copy()
    with pytest.raises(ValueError, match="atom stereochemistry"):
        add_hydrogens_preserving_stereochemistry(molecule)
    assert np.array_equal(molecule.GetConformer().GetPositions(), before)


def test_inverted_new_h_is_corrected_without_changing_source_atoms(monkeypatch):
    add_hydrogens = Chem.AddHs
    captured = {}

    def inverted_placement(molecule, *args, **kwargs):
        result = add_hydrogens(molecule, *args, **kwargs)
        observed = Chem.Mol(result)
        Chem.RemoveStereochemistry(observed)
        Chem.AssignStereochemistryFrom3D(observed, replaceExistingTags=True)
        atom = result.GetAtomWithIdx(7)
        hydrogen = next(
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
            and neighbor.GetIdx() >= molecule.GetNumAtoms()
        )
        if observed.GetAtomWithIdx(7).GetChiralTag() == atom.GetChiralTag():
            conformer = result.GetConformer()
            point = 2 * np.array(conformer.GetAtomPosition(7)) - np.array(
                conformer.GetAtomPosition(hydrogen.GetIdx())
            )
            conformer.SetAtomPosition(hydrogen.GetIdx(), tuple(point))
        captured["raw"] = Chem.Mol(result)
        captured["hydrogen"] = hydrogen.GetIdx()
        return result

    monkeypatch.setattr(Chem, "AddHs", inverted_placement)
    source = _pose(1)
    result = add_hydrogens_preserving_stereochemistry(source)
    raw = captured["raw"]
    hydrogen_index = captured["hydrogen"]
    point = np.array(result.GetConformer().GetAtomPosition(hydrogen_index))
    old_point = np.array(raw.GetConformer().GetAtomPosition(hydrogen_index))
    center = np.array(result.GetConformer().GetAtomPosition(7))
    assert not np.array_equal(point, old_point)
    assert np.linalg.norm(point - center) == pytest.approx(
        np.linalg.norm(old_point - center)
    )
    for index in range(source.GetNumAtoms()):
        assert tuple(result.GetConformer().GetAtomPosition(index)) == tuple(
            source.GetConformer().GetAtomPosition(index)
        )
    atom = result.GetAtomWithIdx(7)
    anchors = [a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() != hydrogen_index]
    vectors = np.array(
        [
            np.array(result.GetConformer().GetAtomPosition(index)) - center
            for index in anchors
        ]
    )
    translated = vectors + center - point
    assert np.linalg.det(vectors) * np.linalg.det(translated) > 0
