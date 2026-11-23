from itertools import permutations

import pytest
from rdkit import Chem

from protrepair.chemistry.inference.ccd_graph import component_heavy_molecule
from protrepair.chemistry.inference.smiles_model import (
    NamedProjection,
    strict_smiles_model,
)


def _model(s):
    return strict_smiles_model(s, max_bytes=4096, max_atoms=128)


@pytest.mark.parametrize(
    ("smiles", "literal_count", "chemistries"),
    [
        ("CC(=O)O", 2, 2),
        ("CC(=O)[O-]", 2, 1),
        ("[O-]S(=O)(=O)[O-]", 6, 1),
        ("C[N+]([O-])=O", 2, 1),
        ("[CH2-]C[CH2+]", 2, 2),
        ("NC(=[NH2+])N", 3, 3),
        ("CC(=O)C", 1, 1),
        ("c1ccccc1", 1, 1),
    ],
)
def test_whole_graph_consensus_against_exhaustive_element_and_edge_permutations(
    smiles, literal_count, chemistries
):
    m = _model(smiles)
    atoms = tuple(a.GetSymbol() for a in m.GetAtoms())
    edges = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in m.GetBonds()
    }
    indexed = tuple(f"A{i}" for i in range(len(atoms)))
    projections = set()
    for perm in permutations(range(len(atoms))):
        if (
            tuple(atoms[i] for i in perm) != atoms
            or {tuple(sorted((perm[i], perm[j]))) for i, j in edges} != edges
        ):
            continue
        projections.add(
            NamedProjection.from_mapping(m, tuple(indexed[i] for i in perm))
        )
    assert len(projections) == literal_count
    assert len({p.oxygen_class() for p in projections}) == chemistries


@pytest.mark.parametrize(
    "smiles",
    [
        "N[C@](F)(Cl)Br",
        "F/C=C/Cl",
        "C/C=C/[C@H](F)/C=C\\C",
        "F[C@H](Cl)[C@@H](Cl)F",
        "c1cc[nH]c1",
    ],
)
def test_declared_stereo_and_h_survive_named_publication_and_reordered_query(smiles):
    m = _model(smiles)
    names = tuple(f"A{i:02d}" for i in range(m.GetNumAtoms()))
    expected = NamedProjection.from_mapping(m, names)
    perm = tuple(reversed(range(len(names))))
    other = Chem.RenumberAtoms(m, perm)
    assert (
        NamedProjection.from_mapping(other, tuple(names[i] for i in perm)) == expected
    )
    published = expected.named_graph("LIG", m, names, set(names))
    assert (
        NamedProjection.from_mapping(
            component_heavy_molecule(published), published.heavy_atom_names
        )
        == expected
    )


def test_named_tetrahedral_parity_and_double_bond_sides_are_not_interchangeable():
    for smiles in ("N[C@](F)(Cl)Br", "F/C=C/Cl"):
        m = _model(smiles)
        names = tuple(f"A{i}" for i in range(m.GetNumAtoms()))
        base = NamedProjection.from_mapping(m, names)
        flipped = Chem.Mol(m)
        if smiles.startswith("N"):
            next(a for a in flipped.GetAtoms() if a.GetChiralTag()).InvertChirality()
        else:
            bond = next(b for b in flipped.GetBonds() if b.GetStereo())
            bond.SetStereo(
                Chem.BondStereo.STEREOZ
                if bond.GetStereo() == Chem.BondStereo.STEREOE
                else Chem.BondStereo.STEREOE
            )
        assert (
            NamedProjection.from_mapping(flipped, names).oxygen_class()
            != base.oxygen_class()
        )
