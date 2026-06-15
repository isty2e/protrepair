"""Workflow analysis attachment tests."""

from pathlib import Path

import pytest

from protrepair.analysis.kinds import AnalysisKind
from protrepair.api import process_structure
from protrepair.io import read_structure

pytestmark = pytest.mark.workflow


def test_process_structure_attaches_requested_analyses() -> None:
    """Workflow should attach requested analyses to the final result."""

    structure = read_structure(Path("tests/fixtures/corpus/pdb1afc.ent"))

    result = process_structure(
        structure,
        analyses=frozenset(
            {
                AnalysisKind.SECONDARY_STRUCTURE,
                AnalysisKind.RAMACHANDRAN,
            }
        ),
    )

    assert result.analyses is not None
    assert result.analyses.secondary_structure is not None
    assert result.analyses.ramachandran is not None
    assert result.analyses.secondary_structure.assignments
    assert result.analyses.ramachandran.points
