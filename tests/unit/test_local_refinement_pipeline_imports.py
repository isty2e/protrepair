"""Import-direction tests for the local-refinement pipeline."""

import ast
from pathlib import Path

LOCAL_PIPELINE_ROOT = Path("src/protrepair/transformer/refinement/local_pipeline")
LOCAL_PIPELINE_PREFIX = (
    "protrepair.transformer.refinement.local_pipeline."
)


def test_local_refinement_pipeline_has_one_way_import_direction() -> None:
    """Lower pipeline models should not import later orchestration stages."""

    imports_by_module = {
        module_name: _local_pipeline_imports(path)
        for module_name, path in {
            "request": LOCAL_PIPELINE_ROOT / "request.py",
            "candidates": LOCAL_PIPELINE_ROOT / "candidates.py",
            "construction": LOCAL_PIPELINE_ROOT / "construction.py",
            "assessment": LOCAL_PIPELINE_ROOT / "assessment.py",
            "runtime": LOCAL_PIPELINE_ROOT / "runtime.py",
        }.items()
    }

    assert "construction" not in imports_by_module["request"]
    assert "candidates" not in imports_by_module["request"]
    assert "construction" not in imports_by_module["candidates"]
    assert "assessment" not in imports_by_module["candidates"]
    assert "runtime" not in imports_by_module["candidates"]


def test_local_refinement_pipeline_core_modules_have_no_import_scc() -> None:
    """Core local-refinement pipeline modules should form an acyclic graph."""

    modules = {
        "request",
        "lineage",
        "candidates",
        "construction",
        "assessment",
        "runtime",
    }
    graph = {
        module_name: _local_pipeline_imports(
            LOCAL_PIPELINE_ROOT / f"{module_name}.py"
        )
        & modules
        for module_name in modules
    }

    assert _strongly_connected_components(graph) == tuple(
        (module_name,) for module_name in sorted(modules)
    )


def _local_pipeline_imports(path: Path) -> set[str]:
    """Return local_pipeline module names imported anywhere in one source file."""

    module = ast.parse(path.read_text())
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add_local_pipeline_module(alias.name, imported_modules)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            _add_local_pipeline_module(node.module, imported_modules)

    return imported_modules


def _add_local_pipeline_module(
    imported_name: str,
    imported_modules: set[str],
) -> None:
    """Add one local_pipeline child module import to the accumulator."""

    if not imported_name.startswith(LOCAL_PIPELINE_PREFIX):
        return

    remainder = imported_name.removeprefix(LOCAL_PIPELINE_PREFIX)
    imported_modules.add(remainder.split(".", maxsplit=1)[0])


def _strongly_connected_components(
    graph: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    """Return sorted strongly connected components for one directed graph."""

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for successor in sorted(graph[node]):
            if successor not in indices:
                visit(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])

        if lowlinks[node] != indices[node]:
            return

        component: list[str] = []
        while True:
            successor = stack.pop()
            on_stack.remove(successor)
            component.append(successor)
            if successor == node:
                break

        components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)

    return tuple(sorted(components))
