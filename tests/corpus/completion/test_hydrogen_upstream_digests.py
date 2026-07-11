"""Upstream digest compatibility checks for hydrogen placement."""

from pathlib import Path

import pytest
from tests.support.structure_summary import summarize_structure

from protrepair.io import read_structure
from protrepair.structure import ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.workflow.contracts import StructureIngressOptions

pytestmark = pytest.mark.corpus


@pytest.mark.representative_regression
@pytest.mark.parametrize(
    ("input_path", "expected_digests"),
    (
        pytest.param(
            Path("tests/fixtures/corpus/pdb2a1d.ent"),
            frozenset(
                {
                    "331f4fc7df32cdeafe561ecf03c820cc945902be99b15e79a94f843ea3b9dee4",
                    "50e9b4880ac62f9d3e51f4f1886d9cfa70db116a3a445db8b29d8860b8c8c70f",
                    "6ac4b51edef7084208a3f975815b4535e6047ac1a4debba5f9d1cf2fd07f1d5d",
                    "68293d3f3f56f119cf5e82e27e0a3d7a12066717c89f075f2551190f8627c8da",
                    "7e592081d5e5f4b774509c57c698be2b61adb6dfbaa0cfc78ca93acd1389c20d",
                    "8c845fc012e27c3a99c6ed4e406d0ab2aee7cbb2fa28ce876f1fc2093c0a5620",
                    "4e251e1af03e2a985b07100bdb346d4b210aa93b88daeecfbf97d8932cf1ea7b",
                    "a141aee6f97e69f3e006e6acd470421056b18f102ffe4316fb744793833ec036",
                    "40c8b3a84fb3204ba8539a9d4623664ae00a2c0548ad7724a2305aa0c9e1c710",
                    "0a153accaa3d9b5e065e4d3b5d5df93a20b37f5d60492c5db005cb93a557bdde",
                }
            ),
            id="pdb2a1d",
        ),
        pytest.param(
            Path("tests/fixtures/corpus/pdb2xbi.ent"),
            frozenset(
                {
                    "8525db26aaf22cc9074c3642242f7522af2eaa11d93efe4411b92bf19c4133cd",
                    "bd1b10d9c9facfb4655a5c36d6a17cd4180eb63a5a921f0c4796abf7561f6f85",
                    "5b511846c25a13a3863fa0aafe020e00db70788d9a24163bd279227776f2ecee",
                    "3ce1628656ab9e79c56aeb9c667f69b89ec484c7476f0e29919f01e81da228c6",
                    "f844e90899ddc4e36256d5ca4dd96c606d55024bc5219ae6cb9f278302422e68",
                }
            ),
            id="pdb2xbi",
        ),
    ),
)
def test_add_hydrogens_matches_known_upstream_hydrogen_digests(
    input_path: Path,
    expected_digests: frozenset[str],
) -> None:
    """Supported repair fixtures should retain verified upstream digests."""

    structure = read_structure(
        input_path,
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    result = add_hydrogens(structure)

    assert summarize_structure(result.structure).semantic_digest in expected_digests


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
