"""Workflow execution seam for committed side-chain packing actions."""

from dataclasses import dataclass

from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.packing.domain import PackingResult
from protrepair.transformer.packing.runtime import execute_sidechain_packing
from protrepair.transformer.packing.spec import PackingSpec
from protrepair.transformer.refinement.speculative_planning import (
    EvaluatedSpeculativeProposal,
    SpeculativeAdoptionDecision,
    SpeculativeExecution,
)


@dataclass(frozen=True, slots=True)
class WorkflowCommittedPackingProposal:
    """One committed workflow packing proposal over a canonical structure."""

    structure: ProteinStructure
    spec: PackingSpec


def execute_committed_workflow_packing(
    structure: ProteinStructure,
    spec: PackingSpec,
) -> PackingResult:
    """Execute one committed workflow packing request through the speculative kernel."""

    proposal = WorkflowCommittedPackingProposal(
        structure=structure,
        spec=spec,
    )
    execution = SpeculativeExecution(
        proposal=proposal,
        outcome=execute_sidechain_packing(structure, spec),
    )
    evaluated_proposal = EvaluatedSpeculativeProposal(
        execution=execution,
        evaluation=execution.outcome,
    )
    decision = SpeculativeAdoptionDecision.adopt(evaluated_proposal)
    return decision.require_candidate().evaluation
