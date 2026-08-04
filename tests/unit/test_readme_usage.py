"""Executable smoke coverage for README usage examples."""

import re
import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README_PATH = _REPO_ROOT / "README.md"
_GETTING_STARTED_PATH = _REPO_ROOT / "docs" / "getting-started.md"
_PYTHON_BLOCK_PATTERN = re.compile(r"```python\n(?P<body>.*?)\n```", re.S)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")


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
    assert "ATOM" in output_path.read_text()


def test_all_documentation_python_examples_compile() -> None:
    """Every Python fence in user documentation should compile."""

    for document_path in _documentation_paths():
        for example_index, snippet in enumerate(
            _python_examples(document_path),
            start=1,
        ):
            compile(
                snippet,
                f"{document_path.name} python example {example_index}",
                "exec",
            )


def test_documentation_local_links_resolve() -> None:
    """Relative Markdown links should resolve from their owning document."""

    for document_path in _documentation_paths():
        document = document_path.read_text()
        for match in _MARKDOWN_LINK_PATTERN.finditer(document):
            target = match.group("target")
            if "://" in target or target.startswith(("#", "mailto:")):
                continue

            relative_path = target.partition("#")[0]
            assert (document_path.parent / relative_path).exists(), (
                f"broken link in {document_path}: {target}"
            )


def _python_examples(document_path: Path) -> tuple[str, ...]:
    """Return Python fenced blocks from one documentation file."""

    document = document_path.read_text()
    return tuple(
        match.group("body") for match in _PYTHON_BLOCK_PATTERN.finditer(document)
    )


def _documentation_paths() -> tuple[Path, ...]:
    """Return the Markdown files that form the user documentation surface."""

    return (_README_PATH, *sorted((_REPO_ROOT / "docs").glob("*.md")))
