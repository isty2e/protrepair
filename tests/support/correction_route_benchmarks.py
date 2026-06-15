"""Explainable workflow-route benchmarks for diagnosis-driven correction."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.state import HydrogenCoverageState, StructureProjectionStateFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.continuous.readiness import (
    structure_facts_supports_continuous_relaxation,
)
from protrepair.workflow.contracts import ProcessResult
from protrepair.workflow.engine import process_canonical_structure
from protrepair.workflow.planning.completion import (
    WorkflowCompletionPartitionKind,
    WorkflowCompletionSelectionReason,
    WorkflowCompletionStageKind,
    WorkflowExecutionStage,
    choose_workflow_completion_plan,
    workflow_legal_completion_plans,
)
from protrepair.workflow.planning.transformation.runtime import (
    StructurePlanningSignature,
)
from tests.support.correction_state_fixtures import (
    build_chain,
    build_residue,
    build_structure,
)
from tests.support.correction_state_registry import CORRECTION_STATE_CASES
from tests.support.request_builders import whole_structure_requested_goals


@dataclass(frozen=True, slots=True)
class WorkflowRouteBenchmarkStage:
    """One chosen workflow stage plus its canonical execution-scope tokens."""

    kind: str
    scope_kind: str
    scope_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowRouteQualityMetrics:
    """Route-quality metrics tied to diagnosed state and chosen route."""

    selected_plan_is_legal: bool
    route_selection_matches_expected: bool
    heavy_completion_cleared: bool | None
    hydrogen_gap_cleared: bool | None
    unsupported_stop_reported: bool | None


@dataclass(frozen=True, slots=True)
class WorkflowRouteBenchmarkExpectation:
    """Expected chosen workflow route for one benchmark case."""

    selection_reason: str
    partition_kind_values: tuple[str, ...]
    execution_plan: tuple[WorkflowRouteBenchmarkStage, ...]


CorrectionRouteStructureFactory = Callable[[ComponentLibrary], ProteinStructure]


@dataclass(frozen=True, slots=True)
class WorkflowRouteBenchmarkCase:
    """One benchmarkable whole-workflow route-planning scenario."""

    case_id: str
    description: str
    requests_hydrogen_population: bool
    structure_factory: CorrectionRouteStructureFactory
    expected: WorkflowRouteBenchmarkExpectation

    def build_structure(
        self,
        component_library: ComponentLibrary,
    ) -> ProteinStructure:
        """Materialize the canonical benchmark structure."""

        return self.structure_factory(component_library)


@dataclass(frozen=True, slots=True)
class WorkflowRouteBenchmarkResult:
    """One benchmark result with diagnosed state, chosen route, and quality."""

    case_id: str
    hydrogen_policy: str
    diagnosed_state_before: StructurePlanningSignature
    diagnosed_state_after: StructurePlanningSignature
    continuous_relaxation_ready_before: bool
    continuous_relaxation_ready_after: bool
    selection_reason: str
    partition_kind_values: tuple[str, ...]
    execution_plan: tuple[WorkflowRouteBenchmarkStage, ...]
    repair_count: int
    issue_kind_values: tuple[str, ...]
    route_quality: WorkflowRouteQualityMetrics

    def as_serializable_dict(self) -> "SerializedWorkflowRouteBenchmarkResult":
        """Return one JSON-serializable benchmark output."""

        return {
            "case_id": self.case_id,
            "hydrogen_policy": self.hydrogen_policy,
            "diagnosed_state_before": _structure_planning_signature_dict(
                self.diagnosed_state_before
            ),
            "diagnosed_state_after": _structure_planning_signature_dict(
                self.diagnosed_state_after
            ),
            "continuous_relaxation_ready_before": (
                self.continuous_relaxation_ready_before
            ),
            "continuous_relaxation_ready_after": (
                self.continuous_relaxation_ready_after
            ),
            "selection_reason": self.selection_reason,
            "partition_kind_values": list(self.partition_kind_values),
            "execution_plan": [
                {
                    "kind": stage.kind,
                    "scope_kind": stage.scope_kind,
                    "scope_tokens": list(stage.scope_tokens),
                }
                for stage in self.execution_plan
            ],
            "repair_count": self.repair_count,
            "issue_kind_values": list(self.issue_kind_values),
            "route_quality": {
                "selected_plan_is_legal": (self.route_quality.selected_plan_is_legal),
                "route_selection_matches_expected": (
                    self.route_quality.route_selection_matches_expected
                ),
                "heavy_completion_cleared": (
                    self.route_quality.heavy_completion_cleared
                ),
                "hydrogen_gap_cleared": (self.route_quality.hydrogen_gap_cleared),
                "unsupported_stop_reported": (
                    self.route_quality.unsupported_stop_reported
                ),
            },
        }


class SerializedStructurePlanningSignature(TypedDict):
    """JSON-safe representation of one whole-structure planning signature."""

    component_support_state: str
    backbone_heavy_atom_completeness_state: str
    sidechain_heavy_atom_completeness_state: str
    hydrogen_applicability_state: str
    hydrogen_coverage_state: str


class SerializedWorkflowRouteBenchmarkStage(TypedDict):
    """JSON-safe representation of one executable workflow stage."""

    kind: str
    scope_kind: str
    scope_tokens: list[str]


class SerializedWorkflowRouteQualityMetrics(TypedDict):
    """JSON-safe route-quality metrics for one workflow benchmark case."""

    selected_plan_is_legal: bool
    route_selection_matches_expected: bool
    heavy_completion_cleared: bool | None
    hydrogen_gap_cleared: bool | None
    unsupported_stop_reported: bool | None


class SerializedWorkflowRouteBenchmarkResult(TypedDict):
    """JSON-safe workflow route benchmark output."""

    case_id: str
    hydrogen_policy: str
    diagnosed_state_before: SerializedStructurePlanningSignature
    diagnosed_state_after: SerializedStructurePlanningSignature
    continuous_relaxation_ready_before: bool
    continuous_relaxation_ready_after: bool
    selection_reason: str
    partition_kind_values: list[str]
    execution_plan: list[SerializedWorkflowRouteBenchmarkStage]
    repair_count: int
    issue_kind_values: list[str]
    route_quality: SerializedWorkflowRouteQualityMetrics


WORKFLOW_ROUTE_BENCHMARK_CASES: dict[str, WorkflowRouteBenchmarkCase] = {
    "hydrogen-only-workflow-continuation": WorkflowRouteBenchmarkCase(
        case_id="hydrogen-only-workflow-continuation",
        description=(
            "Heavy-complete residues should choose hydrogen-only continuation."
        ),
        requests_hydrogen_population=True,
        structure_factory=CORRECTION_STATE_CASES[
            "hydrogen-only-workflow-continuation"
        ].build_structure,
        expected=WorkflowRouteBenchmarkExpectation(
            selection_reason=(
                WorkflowCompletionSelectionReason.REQUESTED_HYDROGEN_POPULATION_REQUIRES_HYDROGEN_CONTINUATION.value
            ),
            partition_kind_values=("hydrogen_only",),
            execution_plan=(
                WorkflowRouteBenchmarkStage(
                    kind="hydrogen_completion",
                    scope_kind="residue_set",
                    scope_tokens=("A:1",),
                ),
            ),
        ),
    ),
    "heavy-then-hydrogen-workflow": WorkflowRouteBenchmarkCase(
        case_id="heavy-then-hydrogen-workflow",
        description=(
            "Heavy-incomplete residues should choose heavy repair before hydrogenation."
        ),
        requests_hydrogen_population=True,
        structure_factory=lambda _component_library: build_structure(
            "heavy-then-hydrogen-workflow",
            (
                build_chain(
                    "A",
                    (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
                ),
            ),
        ),
        expected=WorkflowRouteBenchmarkExpectation(
            selection_reason=(
                WorkflowCompletionSelectionReason.REQUESTED_HYDROGEN_POPULATION_REQUIRES_HEAVY_AND_HYDROGEN_STAGES.value
            ),
            partition_kind_values=("heavy_then_hydrogen",),
            execution_plan=(
                WorkflowRouteBenchmarkStage(
                    kind="heavy_atom_repair",
                    scope_kind="residue_set",
                    scope_tokens=("A:1",),
                ),
                WorkflowRouteBenchmarkStage(
                    kind="hydrogen_completion",
                    scope_kind="residue_set",
                    scope_tokens=("A:1",),
                ),
            ),
        ),
    ),
    "heterogeneous-workflow-partition": WorkflowRouteBenchmarkCase(
        case_id="heterogeneous-workflow-partition",
        description=(
            "Mixed supported and unsupported residues should partition route execution."
        ),
        requests_hydrogen_population=True,
        structure_factory=CORRECTION_STATE_CASES[
            "heterogeneous-workflow-partition"
        ].build_structure,
        expected=WorkflowRouteBenchmarkExpectation(
            selection_reason=(
                WorkflowCompletionSelectionReason.HETEROGENEOUS_SUBSETS_REQUIRE_PARTITIONED_COMPLETION.value
            ),
            partition_kind_values=("hydrogen_only", "unsupported_stop"),
            execution_plan=(
                WorkflowRouteBenchmarkStage(
                    kind="hydrogen_completion",
                    scope_kind="residue_set",
                    scope_tokens=("A:1",),
                ),
            ),
        ),
    ),
}


def run_workflow_route_benchmark_case(
    case: WorkflowRouteBenchmarkCase,
    *,
    component_library: ComponentLibrary | None = None,
) -> WorkflowRouteBenchmarkResult:
    """Run one explainable workflow-route benchmark case."""

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    structure = case.build_structure(active_component_library)
    facts_before = StructureProjectionStateFacts.from_structure(
        structure,
        component_library=active_component_library,
    )
    legal_plans = workflow_legal_completion_plans(
        structure,
        component_library=active_component_library,
        requests_heavy_atom_completion=True,
        requests_hydrogen_population=case.requests_hydrogen_population,
    )
    planning_decision = choose_workflow_completion_plan(legal_plans)
    workflow_result = process_canonical_structure(
        structure,
        requested_goals=whole_structure_requested_goals(
            *(
                ()
                if not case.requests_hydrogen_population
                else (HydrogenCoverageState.COMPLETE,)
            )
        ),
    )
    facts_after = StructureProjectionStateFacts.from_structure(
        workflow_result.structure,
        component_library=active_component_library,
    )
    execution_plan = tuple(
        _workflow_route_benchmark_stage(stage)
        for stage in planning_decision.plan.execution_plan()
    )
    route_quality = WorkflowRouteQualityMetrics(
        selected_plan_is_legal=planning_decision.plan in legal_plans.plans,
        route_selection_matches_expected=(
            planning_decision.reason.value == case.expected.selection_reason
            and planning_decision.plan.partition_kinds()
            == tuple(
                _workflow_partition_kind(partition_kind)
                for partition_kind in case.expected.partition_kind_values
            )
            and execution_plan == case.expected.execution_plan
        ),
        heavy_completion_cleared=_heavy_completion_cleared(
            structure,
            workflow_result.structure,
            residue_ids=planning_decision.plan.residue_ids_for_stage(
                _workflow_stage_kind("heavy_atom_repair")
            ),
            component_library=active_component_library,
        ),
        hydrogen_gap_cleared=_hydrogen_gap_cleared(
            structure,
            workflow_result.structure,
            residue_ids=planning_decision.plan.residue_ids_for_stage(
                _workflow_stage_kind("hydrogen_completion")
            ),
            component_library=active_component_library,
        ),
        unsupported_stop_reported=_unsupported_stop_reported(
            workflow_result,
            residue_ids=planning_decision.plan.unsupported_residue_ids(),
        ),
    )
    return WorkflowRouteBenchmarkResult(
        case_id=case.case_id,
        hydrogen_policy=(
            "add_missing" if case.requests_hydrogen_population else "preserve"
        ),
        diagnosed_state_before=StructurePlanningSignature.from_facts(facts_before),
        diagnosed_state_after=StructurePlanningSignature.from_facts(facts_after),
        continuous_relaxation_ready_before=(
            structure_facts_supports_continuous_relaxation(facts_before)
        ),
        continuous_relaxation_ready_after=(
            structure_facts_supports_continuous_relaxation(facts_after)
        ),
        selection_reason=planning_decision.reason.value,
        partition_kind_values=tuple(
            partition_kind.value
            for partition_kind in planning_decision.plan.partition_kinds()
        ),
        execution_plan=execution_plan,
        repair_count=workflow_result.repair_count(),
        issue_kind_values=tuple(issue.kind.value for issue in workflow_result.issues),
        route_quality=route_quality,
    )


def _workflow_route_benchmark_stage(
    stage: WorkflowExecutionStage,
) -> WorkflowRouteBenchmarkStage:
    """Project one executable workflow stage into benchmark output."""

    return WorkflowRouteBenchmarkStage(
        kind=stage.kind.value,
        scope_kind=stage.scope.kind.value,
        scope_tokens=stage.scope.display_tokens(),
    )


def _subset_structure(
    structure: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
) -> ProteinStructure | None:
    """Return one polymer-only substructure over one residue subset."""

    if not residue_ids:
        return None

    residue_id_set = frozenset(residue_ids)
    chains = []
    for chain_site in structure.constitution.chains:
        selected_residue_payloads = []
        for residue_site in chain_site.residues:
            if residue_site.residue_id not in residue_id_set:
                continue

            residue_index = structure.constitution.residue_index(
                residue_site.residue_id
            )
            residue_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=residue_index,
            )
            selected_residue_payloads.append(
                (
                    residue_site,
                    residue_geometry,
                    structure.topology.residue_formal_charge_by_atom_name(
                        constitution=structure.constitution,
                        residue_index=residue_index,
                    ),
                )
            )

        selected_residues = tuple(selected_residue_payloads)
        if selected_residues:
            chains.append(build_chain(chain_site.chain_id, selected_residues))

    if not chains:
        return None

    return build_structure(
        structure.provenance.ingress.source_name or "subset",
        tuple(chains),
        ligands=(),
    )


def _heavy_completion_cleared(
    before_structure: ProteinStructure,
    after_structure: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
) -> bool | None:
    """Return whether heavy-completion targets no longer require completion."""

    before_subset = _subset_structure(before_structure, residue_ids=residue_ids)
    after_subset = _subset_structure(after_structure, residue_ids=residue_ids)
    if before_subset is None or after_subset is None:
        return None

    before_facts = StructureProjectionStateFacts.from_structure(
        before_subset,
        component_library=component_library,
    )
    after_facts = StructureProjectionStateFacts.from_structure(
        after_subset,
        component_library=component_library,
    )
    return (
        before_facts.backbone_heavy_atom_completeness_fact.value.requires_completion()
        or (
            before_facts.sidechain_heavy_atom_completeness_fact.value.requires_completion()
        )
    ) and not (
        after_facts.backbone_heavy_atom_completeness_fact.value.requires_completion()
        or (
            after_facts.sidechain_heavy_atom_completeness_fact.value.requires_completion()
        )
    )


def _hydrogen_gap_cleared(
    before_structure: ProteinStructure,
    after_structure: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
) -> bool | None:
    """Return whether hydrogen-completion targets no longer need hydrogens."""

    before_subset = _subset_structure(before_structure, residue_ids=residue_ids)
    after_subset = _subset_structure(after_structure, residue_ids=residue_ids)
    if before_subset is None or after_subset is None:
        return None

    before_facts = StructureProjectionStateFacts.from_structure(
        before_subset,
        component_library=component_library,
    )
    after_facts = StructureProjectionStateFacts.from_structure(
        after_subset,
        component_library=component_library,
    )
    return (
        before_facts.hydrogen_coverage_fact.value.needs_hydrogenation()
        and not after_facts.hydrogen_coverage_fact.value.needs_hydrogenation()
    )


def _unsupported_stop_reported(
    workflow_result: ProcessResult,
    *,
    residue_ids: tuple[ResidueId, ...],
) -> bool | None:
    """Return whether unsupported-stop residues surfaced as explicit issues."""

    if not residue_ids:
        return None

    return all(
        any(
            issue.kind is ValidationIssueKind.MISSING_COMPONENT_DEFINITION
            and issue.residue_id == residue_id
            for issue in workflow_result.issues
        )
        for residue_id in residue_ids
    )


def _structure_planning_signature_dict(
    signature: StructurePlanningSignature,
) -> SerializedStructurePlanningSignature:
    """Return one structure planning signature as a serializable dictionary."""

    return {
        "component_support_state": signature.component_support_state.value,
        "backbone_heavy_atom_completeness_state": (
            signature.backbone_heavy_atom_completeness_state.value
        ),
        "sidechain_heavy_atom_completeness_state": (
            signature.sidechain_heavy_atom_completeness_state.value
        ),
        "hydrogen_applicability_state": (signature.hydrogen_applicability_state.value),
        "hydrogen_coverage_state": signature.hydrogen_coverage_state.value,
    }


def _workflow_partition_kind(
    value: str,
) -> WorkflowCompletionPartitionKind:
    """Return the workflow partition kind enum for one canonical token."""

    return WorkflowCompletionPartitionKind(value)


def _workflow_stage_kind(
    value: str,
) -> WorkflowCompletionStageKind:
    """Return the workflow stage kind enum for one canonical token."""

    return WorkflowCompletionStageKind(value)


__all__ = [
    "WORKFLOW_ROUTE_BENCHMARK_CASES",
    "WorkflowRouteBenchmarkCase",
    "WorkflowRouteBenchmarkExpectation",
    "WorkflowRouteBenchmarkResult",
    "WorkflowRouteBenchmarkStage",
    "WorkflowRouteQualityMetrics",
    "run_workflow_route_benchmark_case",
]
