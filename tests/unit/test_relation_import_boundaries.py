"""Import-boundary tests for structure provenance and relation evidence."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

from protrepair.scope import AtomSetScope, WholeStructureScope
from protrepair.structure import StructureEndpoint as StructureFacadeEndpoint
from protrepair.structure.endpoint import StructureEndpoint as StructureOwnedEndpoint
from protrepair.structure.labels import AtomRef, ResidueId


def test_structure_provenance_does_not_import_relation_modules() -> None:
    """Structure provenance should own provenance without depending on relation."""

    imported_modules = _imports_in(Path("src/protrepair/structure/provenance.py"))

    assert not any(
        module_name == "protrepair.relation"
        or module_name.startswith("protrepair.relation.")
        for module_name in imported_modules
    )


def test_relation_facade_uses_eager_exports_without_lazy_getattr() -> None:
    """The relation facade should not hide cycles behind lazy package exports."""

    module = ast.parse(Path("src/protrepair/relation/__init__.py").read_text())

    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        for node in module.body
    )


def test_retired_relation_endpoint_module_is_not_imported_or_present() -> None:
    """No internal module should keep the retired relation endpoint path alive."""

    assert not Path("src/protrepair/relation/endpoint.py").exists()

    imported_modules: set[str] = set()
    for source_path in Path("src/protrepair").rglob("*.py"):
        imported_modules.update(_imports_in(source_path))

    assert "protrepair.relation.endpoint" not in imported_modules


def test_relation_endpoint_facade_exports_structure_owned_type() -> None:
    """Relation may re-export endpoints, but structure remains the type owner."""

    relation = importlib.import_module("protrepair.relation")

    assert relation.StructureEndpoint is StructureOwnedEndpoint
    assert StructureFacadeEndpoint is StructureOwnedEndpoint
    assert relation.StructureEndpoint.__module__ == "protrepair.structure.endpoint"


def test_cold_import_orders_do_not_require_lazy_relation_exports() -> None:
    """Fresh interpreters should import relation and provenance in either order."""

    pythonpath = os.pathsep.join(
        part
        for part in (
            str(Path("src").resolve()),
            os.environ.get("PYTHONPATH", ""),
        )
        if part
    )
    env = {**os.environ, "PYTHONPATH": pythonpath}
    snippets = (
        "import protrepair.structure.provenance; import protrepair.relation; "
        "assert protrepair.relation.StructureEndpoint.__module__ == "
        "'protrepair.structure.endpoint'",
        "import protrepair.relation; import protrepair.structure.provenance; "
        "assert protrepair.relation.StructureEndpoint.__module__ == "
        "'protrepair.structure.endpoint'",
        "import protrepair.relation.evidence; import protrepair.structure.provenance; "
        "import protrepair.relation; assert not hasattr(protrepair.relation, "
        "'__getattr__')",
    )

    for snippet in snippets:
        subprocess.run(
            [sys.executable, "-c", snippet],
            check=True,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )


def test_relation_and_structure_provenance_support_both_import_orders() -> None:
    """Relation evidence and structure provenance should import in either order."""

    structure_provenance = importlib.import_module("protrepair.structure.provenance")
    relation = importlib.import_module("protrepair.relation")

    source_endpoint = relation.StructureEndpoint.source(WholeStructureScope())
    supporting_endpoint = relation.StructureEndpoint.supporting(
        AtomSetScope(atom_refs=(AtomRef(ResidueId("A", 1), "CA"),))
    )
    source_origin = structure_provenance.StructureProvenanceOrigin(
        structure_endpoint=source_endpoint,
    )
    supporting_origin = structure_provenance.StructureProvenanceOrigin(
        structure_endpoint=supporting_endpoint,
    )

    assert source_origin.is_source()
    assert supporting_origin.is_supporting()

    relation = importlib.import_module("protrepair.relation")
    structure_provenance = importlib.import_module("protrepair.structure.provenance")

    assert relation.ExternalCorrespondenceEvidence(
        source_origin=source_origin,
        counterpart_origin=supporting_origin,
        evidence_label=" evidence ",
    ).evidence_label == "evidence"
    assert structure_provenance.StructureProvenanceOrigin(
        structure_endpoint=relation.StructureEndpoint.source(WholeStructureScope()),
    ).is_source()


def _imports_in(path: Path) -> set[str]:
    """Return imported module names from one Python source file."""

    module = ast.parse(path.read_text())
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    return imported_modules
