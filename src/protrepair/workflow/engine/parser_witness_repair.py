"""Engine-owned bounded parser-witness repair loop."""

from dataclasses import dataclass, replace
from typing import Protocol

from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.io.pdb_projection import prepare_rdkit_no_conect_pdb_block_projector
from protrepair.transformer.refinement.parser_witness import (
    DEFAULT_PARSER_WITNESS_REPAIR_BUDGET,
    ParserWitnessRepairCandidate,
    parser_witness_repair_candidates,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.planning.action.registry import WorkflowStateAction


@dataclass(frozen=True, slots=True)
class DirectParserWitnessRepairResult:
    """Result of bounded parser-witness repair attached to one workflow action."""

    result: TransformationResult
    adopted_transformers: tuple[WorkflowStateAction, ...]


class WorkflowTransformerExecutor(Protocol):
    """Execution callback used by parser-witness repair subloops."""

    def __call__(
        self,
        result: TransformationResult,
        *,
        transformer: WorkflowStateAction,
        execution_context: TransformerExecutionContext,
    ) -> TransformationResult:
        """Execute one workflow transformer under the active execution context."""

        ...


def execute_bounded_parser_witness_repair_loop(
    result: TransformationResult,
    *,
    execution_context: TransformerExecutionContext,
    execute_transformer: WorkflowTransformerExecutor,
) -> DirectParserWitnessRepairResult:
    """Execute deterministic parser-witness repair candidates within budget."""

    active_result = result
    adopted_transformers: list[WorkflowStateAction] = []
    budget = DEFAULT_PARSER_WITNESS_REPAIR_BUDGET
    active_pdb_block_projector = prepare_rdkit_no_conect_pdb_block_projector(
        active_result.structure
    )
    active_parser_probe = probe_rdkit_no_conect_parser_readability(
        active_result.structure,
        component_library=execution_context.component_library,
        pdb_block_projector=active_pdb_block_projector,
    )
    before_clusters = active_parser_probe.extra_proximity_bond_clusters()
    initial_extra_bond_count = _parser_witness_extra_bond_count(before_clusters)
    pass_limit = budget.pass_limit_for_initial_extra_heavy_bond_count(
        initial_extra_bond_count,
    )
    for _ in range(pass_limit):
        before_extra_bond_count = _parser_witness_extra_bond_count(before_clusters)
        if before_extra_bond_count == 0:
            break

        candidates = parser_witness_repair_candidates(
            active_result.structure,
            component_library=execution_context.component_library,
            budget=budget,
            clusters=before_clusters,
        )
        if not candidates:
            break

        accepted_improvement = False
        for candidate in candidates:
            transformer = LocalRefinementTransformer.from_repair_refinement(
                candidate.repair_refinement,
            )
            execution_outcome = execute_transformer(
                active_result,
                transformer=transformer,
                execution_context=execution_context,
            )
            candidate_result = _with_parser_witness_repair_diagnostics(
                active_result,
                execution_outcome,
                candidate=candidate,
            )
            candidate_parser_probe = probe_rdkit_no_conect_parser_readability(
                candidate_result.structure,
                component_library=execution_context.component_library,
                pdb_block_projector=active_pdb_block_projector,
            )
            after_extra_bond_count = _parser_witness_extra_bond_count(
                candidate_parser_probe.extra_proximity_bond_clusters()
            )
            if after_extra_bond_count >= before_extra_bond_count:
                continue

            active_result = candidate_result
            active_parser_probe = candidate_parser_probe
            active_pdb_block_projector = prepare_rdkit_no_conect_pdb_block_projector(
                active_result.structure
            )
            before_clusters = active_parser_probe.extra_proximity_bond_clusters()
            adopted_transformers.append(transformer)
            accepted_improvement = True
            break

        if not accepted_improvement:
            break

    return DirectParserWitnessRepairResult(
        result=active_result,
        adopted_transformers=tuple(adopted_transformers),
    )


def _parser_witness_extra_bond_count(
    clusters: tuple[RDKitProximityBondCluster, ...],
) -> int:
    """Return current whole-structure parser-witness extra heavy-bond count."""

    return sum(len(cluster.bonds) for cluster in clusters)


def _with_parser_witness_repair_diagnostics(
    before: TransformationResult,
    after: TransformationResult,
    *,
    candidate: ParserWitnessRepairCandidate,
) -> TransformationResult:
    """Annotate new local-refinement diagnostics with parser-witness budget data."""

    new_repairs = after.repairs[len(before.repairs) :]
    new_issues = after.issues[len(before.issues) :]
    if not new_repairs and not new_issues:
        return after

    budget_details = _parser_witness_repair_budget_details(candidate)
    return replace(
        after,
        repairs=before.repairs
        + tuple(
            _annotated_parser_witness_repair_event(
                repair,
                budget_details=budget_details,
            )
            for repair in new_repairs
        ),
        issues=before.issues
        + tuple(
            _annotated_parser_witness_repair_issue(
                issue,
                budget_details=budget_details,
            )
            for issue in new_issues
        ),
    )


def _annotated_parser_witness_repair_event(
    repair: RepairEvent,
    *,
    budget_details: str,
) -> RepairEvent:
    """Return one repair event with parser-witness budget details when relevant."""

    if repair.kind is not RepairEventKind.LOCAL_REFINEMENT_APPLIED:
        return repair

    return replace(
        repair,
        details=_append_diagnostic_detail(repair.details, budget_details),
    )


def _annotated_parser_witness_repair_issue(
    issue: ValidationIssue,
    *,
    budget_details: str,
) -> ValidationIssue:
    """Return one issue with parser-witness budget details when relevant."""

    if issue.kind is not ValidationIssueKind.REFINEMENT_REJECTED:
        return issue

    return replace(
        issue,
        message=_append_diagnostic_detail(issue.message, budget_details),
    )


def _append_diagnostic_detail(
    current_detail: str | None,
    additional_detail: str,
) -> str:
    """Append a semicolon-separated diagnostic detail."""

    if current_detail is None or not current_detail:
        return additional_detail

    return f"{current_detail}; {additional_detail}"


def _parser_witness_repair_budget_details(
    candidate: ParserWitnessRepairCandidate,
) -> str:
    """Return one deterministic parser-witness budget diagnostic string."""

    budget = candidate.budget
    return (
        "parser-witness budget "
        f"cluster={candidate.cluster.display_token()} "
        f"max_passes={budget.max_passes} "
        f"base_passes={budget.base_passes} "
        f"extra_heavy_bonds_per_pass={budget.extra_heavy_bonds_per_pass} "
        f"max_clusters_per_pass={budget.max_clusters_per_pass} "
        f"max_cluster_residues={budget.max_cluster_residues} "
        f"context_radius_angstrom={budget.context_radius_angstrom:g} "
        f"max_iterations={budget.max_iterations}"
    )
