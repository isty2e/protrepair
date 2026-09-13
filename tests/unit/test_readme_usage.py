"""Executable smoke coverage for README usage examples."""

import re
import shutil
from pathlib import Path

import pytest

from protrepair.io import read_structure

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README_PATH = _REPO_ROOT / "README.md"
_GETTING_STARTED_PATH = _REPO_ROOT / "docs" / "getting-started.md"
_PYTHON_BLOCK_PATTERN = re.compile(r"```python\n(?P<body>.*?)\n```", re.S)


@pytest.mark.parametrize(
    ("document_path", "output_filename"),
    (
        (_README_PATH, "output.pdb"),
        (_GETTING_STARTED_PATH, "repaired.pdb"),
    ),
)
def test_primary_documentation_python_examples_run_in_document_order(
    document_path: Path,
    output_filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary Python examples should remain executable documentation contracts."""

    fixture_path = _REPO_ROOT / "tests" / "fixtures" / "pdb" / "1aho.pdb"
    working_fixture_path = tmp_path / "input.pdb"
    shutil.copy2(fixture_path, working_fixture_path)

    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__documentation_example__"}
    for example_index, snippet in enumerate(
        _python_examples(document_path),
        start=1,
    ):
        exec(
            compile(
                snippet,
                f"{document_path.name} python example {example_index}",
                "exec",
            ),
            namespace,
        )

    output_path = tmp_path / output_filename
    written = read_structure(output_path)
    assert written.constitution.chains
    assert written.constitution.atom_slots


def _python_examples(document_path: Path) -> tuple[str, ...]:
    """Return Python fenced blocks from one documentation file."""

    document = document_path.read_text()
    return tuple(
        match.group("body") for match in _PYTHON_BLOCK_PATTERN.finditer(document)
    )
