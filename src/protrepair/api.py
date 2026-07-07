"""Public API surface for the ``protrepair`` package."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from protrepair.analysis.kinds import AnalysisKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.contracts import (
    ProcessResult,
    StructureIngressOptions,
    WorkflowGoal,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
)
from protrepair.workflow.engine import process_structure_source


def process_structure(
    source: Path | str | ProteinStructure,
    *,
    ingress: StructureIngressOptions | None = None,
    selected_source_chain_ids: tuple[str, ...] | None = None,
    requested_goals: Sequence[WorkflowGoal] = (),
    transform_requests: WorkflowTransformRequests | None = None,
    planning_context: WorkflowPlanningContext | None = None,
    analyses: Iterable[AnalysisKind] = frozenset(),
) -> ProcessResult:
    """Process one structure source through the current workflow spine.

    Args:
        source: Raw path-like structure source or canonical ``ProteinStructure``.
        ingress: Source-interpretation policy for normalization boundaries.
        selected_source_chain_ids: Optional raw-source chain subset to read
            before canonical normalization. This applies only to path-like
            sources, not canonical ``ProteinStructure`` inputs.
        requested_goals: Desired scoped state propositions over the canonical model.
        transform_requests: Explicit transform requests not reducible to desired
            states alone.
        analyses: Structured analysis outputs to attach to the final result.

    Returns:
        One structured workflow result containing the final structure, issues,
        requested-goal reporting, and any requested analyses.
    """

    return process_structure_source(
        source,
        ingress=ingress,
        selected_source_chain_ids=selected_source_chain_ids,
        requested_goals=requested_goals,
        transform_requests=transform_requests,
        planning_context=planning_context,
        analyses=analyses,
    )


__all__ = ["process_structure"]
