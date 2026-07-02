"""Package facade ontology tests."""

import ast
import importlib
from pathlib import Path

EXPLICIT_FACADE_PACKAGES = (
    "protrepair.io",
    "protrepair.diagnostics",
    "protrepair.sources",
    "protrepair.relation",
)

INTERNAL_MARKER_PACKAGES = (
    "protrepair.transformer.continuous",
    "protrepair.transformer.refinement.local_pipeline",
)

INTERNAL_MARKER_PACKAGE_PATHS = tuple(
    Path("src", *package_name.split("."), "__init__.py")
    for package_name in INTERNAL_MARKER_PACKAGES
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

    for package_name in EXPLICIT_FACADE_PACKAGES:
        package = importlib.import_module(package_name)
        missing_exports = [
            export_name
            for export_name in package.__all__
            if not hasattr(package, export_name)
        ]

        assert not missing_exports, (
            f"{package_name} has unresolved exports: {missing_exports}"
        )


def test_internal_execution_packages_do_not_reexport_stage_symbols() -> None:
    """Internal execution packages should not act as broad import barrels."""

    for package_name in INTERNAL_MARKER_PACKAGES:
        package = importlib.import_module(package_name)

        assert package.__all__ == []


def test_internal_execution_packages_are_not_symbol_import_facades() -> None:
    """Production code should import execution symbols from owner modules."""

    offenders = []
    for source_path in Path("src/protrepair").rglob("*.py"):
        if source_path in INTERNAL_MARKER_PACKAGE_PATHS:
            continue

        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in INTERNAL_MARKER_PACKAGES:
                continue
            imported_names = tuple(alias.name for alias in node.names)
            offenders.append((source_path.as_posix(), imported_names))

    assert not offenders


def test_internal_execution_symbols_are_importable_from_owner_modules() -> None:
    """Direct owner-module imports remain the supported internal path."""

    continuous_support = importlib.import_module(
        "protrepair.transformer.continuous.support"
    )
    local_pipeline_backend = importlib.import_module(
        "protrepair.transformer.refinement.local_pipeline.backend"
    )
    local_pipeline_runtime = importlib.import_module(
        "protrepair.transformer.refinement.local_pipeline.runtime"
    )

    assert hasattr(continuous_support, "resolve_local_bond_planning_support")
    assert hasattr(local_pipeline_backend, "resolve_continuous_relaxation_backend")
    assert hasattr(local_pipeline_runtime, "execute_local_transformation")
