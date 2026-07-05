"""Package-ontology tests for diagnostic ownership boundaries."""

import ast
from pathlib import Path


def test_diagnostics_package_does_not_import_runtime_or_source_owners() -> None:
    """Diagnostics should stay read-only and avoid source/workflow/transform owners."""

    forbidden_prefixes = (
        "protrepair.sources",
        "protrepair.transformer",
        "protrepair.workflow",
    )
    violations: list[str] = []
    for path in Path("src/protrepair/diagnostics").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module
                if module_name is not None and module_name.startswith(
                    forbidden_prefixes
                ):
                    violations.append(f"{path}:{node.lineno}:{module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_prefixes):
                        violations.append(f"{path}:{node.lineno}:{alias.name}")

    assert not violations


def test_diagnostics_package_does_not_own_coordinate_text_projection() -> None:
    """Diagnostics may consume parser PDB projections but must not own gemmi IO."""

    forbidden_modules = {
        "protrepair.io.gemmi_normalization",
        "protrepair.io.gemmi_writer",
    }
    violations: list[str] = []
    for path in Path("src/protrepair/diagnostics").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                violations.append(f"{path}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    imported_module = f"{node.module}.{alias.name}"
                    if imported_module in forbidden_modules:
                        violations.append(f"{path}:{node.lineno}:{imported_module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append(f"{path}:{node.lineno}:{alias.name}")

    assert not violations
