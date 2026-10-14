from pathlib import Path

import pytest

from protrepair.api import process_structure
from protrepair.io import read_structure
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize(
    "filename",
    [
        "pdb3lgf.ent",
        "pdb4oee.ent",
        "pdb2z62.ent",
        "pdb1tca.ent",
        "pdb1g82.ent",
        "6dmb.cif",
        "pdb1afc.ent",
        "pdb1vbo.ent",
    ],
)
def test_keep_retains_ligand_only_chains_in_deposited_structures(filename: str) -> None:
    structure = read_structure(Path("tests/fixtures/corpus") / filename)
    polymer_chain_ids = set(structure.constitution.chain_ids())
    assert any(
        ligand.residue_id.chain_id not in polymer_chain_ids
        for ligand in structure.constitution.ligands
    )

    normalized = apply_structure_normalization_policy(
        structure, policy=StructureNormalizationPolicy()
    )
    assert normalized is structure


def test_public_processing_preserves_glycan_chain() -> None:
    structure = read_structure(Path("tests/fixtures/corpus/pdb1tca.ent"))
    result = process_structure(
        structure, ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    )

    assert len(result.structure.constitution.ligands) == 2
    assert result.structure.constitution == structure.constitution
    assert result.structure.geometry == structure.geometry
    assert result.structure.topology == structure.topology
