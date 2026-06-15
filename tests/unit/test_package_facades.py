"""Package facade ontology tests."""

import ast
import importlib
from pathlib import Path

EXPLICIT_FACADE_PACKAGES = (
    "protrepair.io",
    "protrepair.diagnostics",
    "protrepair.sources",
)


def test_cleaned_package_facades_are_explicit_exports() -> None:
    """Package facades without concrete cycles should not use lazy exports."""

    for package_name in EXPLICIT_FACADE_PACKAGES:
        package_path = Path("src", *package_name.split("."), "__init__.py")
        tree = ast.parse(package_path.read_text())
        lazy_defs = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        ]

        assert not lazy_defs, f"{package_name} still defines lazy exports"


def test_package_facade_exports_resolve() -> None:
    """Every package facade export should resolve at package import time."""

    for package_name in (*EXPLICIT_FACADE_PACKAGES, "protrepair.relation"):
        package = importlib.import_module(package_name)
        missing_exports = [
            export_name
            for export_name in package.__all__
            if not hasattr(package, export_name)
        ]

        assert not missing_exports, (
            f"{package_name} has unresolved exports: {missing_exports}"
        )


def test_relation_lazy_facade_documents_concrete_cycle() -> None:
    """The remaining lazy relation facade should document its concrete cycle."""

    package_path = Path("src/protrepair/relation/__init__.py")
    contents = package_path.read_text()

    assert "def __getattr__" in contents
    assert "structure.provenance" in contents
    assert "relation.evidence" in contents

