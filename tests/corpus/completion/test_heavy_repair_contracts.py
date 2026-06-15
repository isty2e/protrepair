"""Heavy-atom repair contract tests."""

from pathlib import Path

import pytest

from protrepair.io import read_structure

pytestmark = pytest.mark.corpus


def test_strip_hydrogens_returns_original_structure_when_no_hydrogens() -> None:
    """Heavy-only structures should bypass no-op hydrogen stripping."""

    structure = read_structure(Path("tests/fixtures/corpus/pdb2xbi.ent")).select_chains(
        ("A",)
    )

    stripped = structure.without_hydrogens()

    assert stripped is structure
