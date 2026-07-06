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
    "protrepair.structure": (
        "AtomRef",
        "ProteinStructure",
        "ResidueId",
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
        "WorkflowGoal",
        "WorkflowTransformRequests",
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
            assert public_name in module.__all__, f"{module_name}.__all__ {public_name}"


def test_inactive_workflow_policy_enums_are_not_public_contracts() -> None:
    """Public policy enums must either affect execution or stay unexported."""

    contracts = importlib.import_module("protrepair.workflow.contracts")
    policies = importlib.import_module("protrepair.workflow.contracts.policies")

    for inactive_policy_name in ("HydrogenPolicy", "CTerminalOxtPolicy"):
        assert not hasattr(contracts, inactive_policy_name)
        assert not hasattr(policies, inactive_policy_name)
        assert inactive_policy_name not in contracts.__all__
        assert inactive_policy_name not in policies.__all__


def test_workflow_policy_module_exports_only_runtime_bound_policies() -> None:
    """Policy module exports should be request-bound or execution-bound."""

    policies = importlib.import_module("protrepair.workflow.contracts.policies")

    assert set(policies.__all__) == {
        "LigandPolicy",
        "MutationPolicy",
        "OccupancyPolicy",
        "OrphanFragmentPolicy",
    }
    for policy_name in policies.__all__:
        assert hasattr(policies, policy_name), policy_name


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


def test_readme_uses_stable_subpackage_facade_imports() -> None:
    """README examples should import typed contracts from owner facades."""

    readme = Path("README.md").read_text()

    assert "from protrepair import WorkflowTransformRequests" not in readme
    assert "from protrepair import AnalysisKind" not in readme
    assert "from protrepair.workflow.contracts import (" in readme
    assert "LigandPolicy" in readme
    assert "from protrepair.analysis import AnalysisKind" in readme
