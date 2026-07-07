"""Executable smoke coverage for README usage examples."""

import re
import shutil
from pathlib import Path

from pytest import MonkeyPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
README_PYTHON_BLOCK_PATTERN = re.compile(r"```python\n(?P<body>.*?)\n```", re.S)


def test_readme_python_examples_run_in_document_order(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """README Python snippets should be the executed documentation contract."""

    fixture_path = REPO_ROOT / "tests" / "fixtures" / "pdb" / "1aho.pdb"
    working_fixture_path = tmp_path / "tests" / "fixtures" / "pdb" / "1aho.pdb"
    working_fixture_path.parent.mkdir(parents=True)
    shutil.copy2(fixture_path, working_fixture_path)

    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__readme_example__"}
    for example_index, snippet in enumerate(_readme_python_examples(), start=1):
        exec(
            compile(snippet, f"README.md python example {example_index}", "exec"),
            namespace,
        )

    output_path = tmp_path / "output.pdb"
    assert "ATOM" in output_path.read_text()


def _readme_python_examples() -> tuple[str, ...]:
    """Return README Python fenced blocks as executable snippets."""

    readme = README_PATH.read_text()
    return tuple(
        match.group("body")
        for match in README_PYTHON_BLOCK_PATTERN.finditer(readme)
    )
