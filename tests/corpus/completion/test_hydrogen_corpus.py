"""Corpus regression snapshots for coupled polymer hydrogen placement."""

from pathlib import Path

import pytest
from tests.support.structure_summary import summarize_structure

from protrepair.io import read_structure
from protrepair.structure import ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.workflow.contracts import StructureIngressOptions

pytestmark = pytest.mark.corpus


@pytest.mark.representative_regression
@pytest.mark.parametrize(
    ("input_path", "expected_digest"),
    (
        pytest.param(
            Path("tests/fixtures/corpus/pdb2a1d.ent"),
            "fe69481347dab228ecc83328bc6683375b574e55790b2a76f985cb5e4b8ef0de",
            id="pdb2a1d",
        ),
        pytest.param(
            Path("tests/fixtures/corpus/pdb2xbi.ent"),
            "0e13b060c9b57441ca758f32cdeaf4f5466e1caa05491206ef354c1345ed18d4",
            id="pdb2xbi",
        ),
    ),
)
def test_add_hydrogens_matches_coupled_chemistry_snapshot(
    input_path: Path,
    expected_digest: str,
) -> None:
    """Pin current identities and coordinates, not legacy uncoupled chemistry."""

    structure = read_structure(
        input_path,
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    result = add_hydrogens(structure)

    assert summarize_structure(result.structure).semantic_digest == expected_digest
    for offset, atom in enumerate(structure.constitution.atom_slots):
        if atom.is_hydrogen():
            continue
        index = AtomIndex(offset)
        ref = structure.constitution.atom_ref_at(index)
        output_index = result.structure.constitution.atom_index(ref)
        assert result.structure.geometry.atom_geometry(
            output_index
        ) == structure.geometry.atom_geometry(index)


def test_add_hydrogens_omits_hg_from_cross_chain_source_disulfides() -> None:
    """Cross-chain SSBOND cysteines should remain oxidized after completion."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb2a1d.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    repaired = add_hydrogens(structure).structure

    for residue_id in (
        ("A", 1),
        ("B", 122),
        ("E", 1),
        ("F", 122),
    ):
        residue = repaired.constitution.residue_or_ligand(
            ResidueId(residue_id[0], residue_id[1])
        )
        assert residue is not None
        assert not residue.has_atom_site("HG")
