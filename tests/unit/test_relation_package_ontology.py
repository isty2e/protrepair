"""Package-ontology tests for canonical relation ownership."""

import ast
from pathlib import Path


def test_relation_package_does_not_import_runtime_or_source_owners() -> None:
    """Canonical relations should not depend on source, IO, or transformer owners."""

    forbidden_prefixes = (
        "protrepair.io",
        "protrepair.sources",
        "protrepair.transformer",
        "protrepair.workflow",
    )
    violations: list[str] = []
    for path in Path("src/protrepair/relation").glob("*.py"):
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
