"""Public API surface for the ``protrepair`` package."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from protrepair.analysis.kinds import AnalysisKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.workflow.contracts import (
    HydrogenLengthRequest,
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
    hydrogen_lengths: HydrogenLengthRequest | None = None,
) -> ProcessResult:
    """Process one structure source through the current workflow spine.

    Parameters
    ----------
    source : Path, str or ProteinStructure
        Structure source or canonical input.
    ingress : StructureIngressOptions or None
        Source-interpretation policy for normalization.
    selected_source_chain_ids : tuple[str, ...] or None
        Chain selection for path inputs, not canonical inputs.
    requested_goals : Sequence[WorkflowGoal]
        Desired scoped state propositions over the canonical model.
    transform_requests : WorkflowTransformRequests or None
        Explicit transform requests not reducible to desired states alone.
    planning_context : WorkflowPlanningContext or None
        Selection policy and search limits.
    analyses : Iterable[AnalysisKind]
        Structured analyses to attach to the final result.
    hydrogen_lengths : HydrogenLengthRequest or None
        Scope or grade the observation of existing X-H lengths, or request a
        guarded correction under a declared convention. None measures input
        H/D/T on the selected result without grading or moving them.

    Returns
    -------
    ProcessResult
        Final structure, issues, goal reporting and analyses.
    """

    return process_structure_source(
        source,
        ingress=ingress,
        selected_source_chain_ids=selected_source_chain_ids,
        requested_goals=requested_goals,
        transform_requests=transform_requests,
        planning_context=planning_context,
        analyses=analyses,
        hydrogen_lengths=hydrogen_lengths,
    )


__all__ = ["process_structure"]
