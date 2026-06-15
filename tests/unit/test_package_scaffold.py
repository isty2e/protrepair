"""Smoke tests for the current ``protrepair`` package scaffold."""

import importlib


def test_package_imports() -> None:
    package = importlib.import_module("protrepair")

    assert package.__all__ == [
        "__version__",
        "AnalysisKind",
        "ProcessResult",
        "RepairRefinementSpec",
        "RetainedNonPolymerChemistryOverride",
        "StructureIngressOptions",
        "WorkflowTransformRequests",
        "process_structure",
        "requested_process_goal",
    ]
    assert isinstance(package.__version__, str)
    assert package.__version__
    assert package.AnalysisKind is not None
    assert package.ProcessResult is not None
    assert package.StructureIngressOptions is not None
    assert package.WorkflowTransformRequests is not None
    assert package.RepairRefinementSpec is not None
    assert package.requested_process_goal is not None


def test_scaffold_modules_import() -> None:
    module_names = [
        "protrepair.api",
        "protrepair.errors",
        "protrepair.relation",
        "protrepair.state",
        "protrepair.transformer",
        "protrepair.io",
        "protrepair.chemistry",
        "protrepair.analysis",
        "protrepair.workflow",
        "protrepair.transformer.local",
        "protrepair.workflow.planning.action.domain",
        "protrepair.workflow.planning.action.proposals",
        "protrepair.workflow.planning.action.registry",
        "protrepair.workflow.planning.transformation.runtime",
        "protrepair.workflow.planning.transformation.legality",
        "protrepair.workflow.planning.transformation.selection",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name) is not None
