"""Import-direction contracts for canonical chemistry modules."""

import ast
from pathlib import Path


def test_chemistry_does_not_import_io_boundary() -> None:
    """Canonical chemistry must not depend on external-format adapters."""

    imported_io_modules = {
        (source_path, imported_module)
        for source_path in Path("src/protrepair/chemistry").rglob("*.py")
        for imported_module in _imports_in(source_path)
        if imported_module == "protrepair.io"
        or imported_module.startswith("protrepair.io.")
    }

    assert imported_io_modules == set()


def _imports_in(path: Path) -> set[str]:
    """Return absolute modules imported by one Python source file."""

    module = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    return imported_modules
