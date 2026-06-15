"""Executable smoke coverage for README usage examples."""

from pathlib import Path

from protrepair import (
    AnalysisKind,
    StructureIngressOptions,
    process_structure,
    requested_process_goal,
)
from protrepair.io import write_structure
from protrepair.scope import WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.workflow.contracts import LigandPolicy


def test_readme_basic_repair_usage_runs(tmp_path: Path) -> None:
    """The README repair example should use supported public import paths."""

    result = process_structure(
        Path("tests/fixtures/pdb/1aho.pdb"),
        ingress=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
        ),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(),
                value=BackboneHeavyAtomCompletenessState.COMPLETE,
            ),
            requested_process_goal(
                scope=WholeStructureScope(),
                value=SidechainHeavyAtomCompletenessState.COMPLETE,
            ),
            requested_process_goal(
                scope=WholeStructureScope(),
                value=HydrogenCoverageState.COMPLETE,
            ),
        ),
    )

    assert not result.has_errors()
    assert result.structure.geometry.atom_count() > 0

    output_path = tmp_path / "output.pdb"
    write_structure(result.structure, output_path)
    assert "ATOM" in output_path.read_text()


def test_readme_analysis_usage_runs() -> None:
    """The README analysis example should use supported public import paths."""

    analysis_result = process_structure(
        Path("tests/fixtures/pdb/1aho.pdb"),
        analyses=frozenset(
            {
                AnalysisKind.SECONDARY_STRUCTURE,
                AnalysisKind.RAMACHANDRAN,
            }
        ),
    )

    assert analysis_result.analyses is not None
