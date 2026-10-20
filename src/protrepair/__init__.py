"""Top-level public API for the ``protrepair`` package."""

from importlib.metadata import PackageNotFoundError, version

from protrepair.analysis.kinds import AnalysisKind
from protrepair.api import process_structure
from protrepair.errors import ProtrepairError
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import (
    ProcessResult,
    RetainedNonPolymerChemistryOverride,
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)

try:
    __version__ = version("protrepair")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AnalysisKind",
    "ProcessResult",
    "ProtrepairError",
    "RepairRefinementSpec",
    "RetainedNonPolymerChemistryOverride",
    "StructureIngressOptions",
    "WorkflowTransformRequests",
    "__version__",
    "process_structure",
    "requested_process_goal",
]
