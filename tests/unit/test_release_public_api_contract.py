"""Release-facing public import contract tests."""

import importlib
import re
from pathlib import Path

RELEASE_PUBLIC_IMPORTS: dict[str, tuple[str, ...]] = {
    "protrepair": (
        "AnalysisKind",
        "ProcessResult",
        "StructureIngressOptions",
        "process_structure",
        "requested_process_goal",
    ),
    "protrepair.scope": (
        "ResidueSetScope",
        "Scope",
        "WholeStructureScope",
    ),
    "protrepair.state": (
        "BackboneHeavyAtomCompletenessState",
        "HydrogenCoverageState",
        "SidechainHeavyAtomCompletenessState",
    ),
    "protrepair.workflow.contracts": (
        "LigandPolicy",
        "ProcessResult",
        "StructureIngressOptions",
        "requested_process_goal",
    ),
    "protrepair.io": (
        "FileFormat",
        "read_structure",
        "read_structure_string",
        "write_structure",
        "write_structure_string",
    ),
    "protrepair.geometry": (
        "GeometryPlacementError",
    ),
}


def test_release_public_imports_resolve() -> None:
    """Release-facing public imports should resolve from documented modules."""

    for module_name, public_names in RELEASE_PUBLIC_IMPORTS.items():
        module = importlib.import_module(module_name)
        for public_name in public_names:
            assert hasattr(module, public_name), f"{module_name}.{public_name}"


def test_readme_uses_current_state_axis_names() -> None:
    """README should not advertise stale pre-ontology state names."""

    readme = Path("README.md").read_text()

    assert "BackboneHeavyAtomCompletenessState" in readme
    assert "SidechainHeavyAtomCompletenessState" in readme
    assert "HydrogenCoverageState" in readme
    assert (
        re.search(
            r"(?<!Backbone)(?<!Sidechain)\bHeavyAtomCompletenessState\b",
            readme,
        )
        is None
    )
    assert "HydrogenRealizationState" not in readme
