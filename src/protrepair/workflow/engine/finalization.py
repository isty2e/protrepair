"""Retained workflow result selection and final diagnostics."""

from dataclasses import replace

from protrepair.analysis.kinds import AnalysisKind
from protrepair.analysis.runtime import build_analysis_bundle
from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.events import EventScope, ValidationIssue
from protrepair.diagnostics.geometry import detect_heavy_geometry
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.diagnostics.parser_readability import (
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.diagnostics.parser_topology import (
    ambiguous_disulfide_parser_witness_blocker_issues,
)
from protrepair.diagnostics.source_coordination import (
    diagnose_source_coordination_omissions,
)
from protrepair.io.pdb_projection import prepare_rdkit_no_conect_pdb_block_projector
from protrepair.state.structure_topology import (
    DisulfideEndpointMultiplicityContradiction,
    DisulfideTopologyConflict,
    DisulfideTopologyConflictReason,
    StructureDisulfideTopologyFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.refinement.speculative_planning import SpeculativeStopReason
from protrepair.workflow.contracts.planning import WorkflowPlanningContext
from protrepair.workflow.contracts.result import (
    ProcessResult,
    WorkflowBranchReport,
)
from protrepair.workflow.engine.polymer_hydrogen import project_polymer_hydrogen_issues
from protrepair.workflow.engine.runtime import WorkflowBranch


def finalize_workflow_result(
    *,
    branches: tuple[WorkflowBranch, ...],
    planning_context: WorkflowPlanningContext,
    component_library: ComponentLibrary,
    requested_analyses: frozenset[AnalysisKind],
    stop_reason: SpeculativeStopReason = SpeculativeStopReason.NO_PROPOSALS_AVAILABLE,
    preliminary_issues: tuple[ValidationIssue, ...] = (),
    repair_refinement: RepairRefinementSpec | None = None,
) -> ProcessResult:
    """Select the preferred assessed branch and attach final diagnostics.

    Parameters
    ----------
    branches : tuple[WorkflowBranch, ...]
        Assessed terminal and frontier candidates.
    planning_context : WorkflowPlanningContext
        Selection policy and search limits for these candidates.
    component_library : ComponentLibrary
        Active chemistry for diagnostics and requested analyses.
    requested_analyses : frozenset[AnalysisKind]
        Analyses to compute on the selected structure.
    stop_reason : SpeculativeStopReason
        Why exploration ended.
    preliminary_issues : tuple[ValidationIssue, ...]
        Request-level issues to retain on the selected result.
    repair_refinement : RepairRefinementSpec or None
        Active local request whose included region requires H preparation,
        even without an explicit H-completion goal.

    Returns
    -------
    ProcessResult
        Selected structure, branch assessment and current final diagnostics.
    """

    branch_report = _branch_report(
        branches=branches,
        planning_context=planning_context,
        stop_reason=stop_reason,
    )
    preferred_branch = branch_report.require_preferred_value(
        {branch.node_id: branch for branch in branches}
    )
    result = ProcessResult.from_transformation_result(
        preferred_branch.result,
        requested_goal_report=(branch_report.preferred_outcome().requested_goal_report),
        branch_report=branch_report,
    )
    if branch_report.search_exhausted():
        result = result.with_appended_issues(
            (
                ValidationIssue(
                    kind=ValidationIssueKind.WORKFLOW_SEARCH_EXHAUSTED,
                    severity=IssueSeverity.WARNING,
                    message=(
                        "workflow search reached "
                        "max_speculative_nodes="
                        f"{planning_context.max_speculative_nodes}; "
                        "returning the preferred assessed retained candidate; "
                        "requested goals and unresolved defects are reported separately"
                    ),
                ),
            )
        )
    parser_readability_issues = final_parser_readability_issues(
        result.structure,
        component_library=component_library,
    )
    if parser_readability_issues:
        result = result.with_appended_issues(parser_readability_issues)
    disulfide_topology_issues = _final_disulfide_topology_issues(result.structure)
    if disulfide_topology_issues:
        result = result.with_appended_issues(disulfide_topology_issues)
    source_coordination_issues = diagnose_source_coordination_omissions(
        result.structure
    )
    if source_coordination_issues:
        result = result.with_appended_issues(source_coordination_issues)
    geometry_issues = _final_reconstructed_geometry_issues(
        result, component_library=component_library
    )
    if geometry_issues:
        result = result.with_appended_issues(geometry_issues)
    if preliminary_issues:
        result = result.with_appended_issues(preliminary_issues)
    result = replace(
        result,
        issues=project_polymer_hydrogen_issues(
            result,
            requested_goal_report=result.requested_goal_report,
            component_library=component_library,
            repair_refinement=repair_refinement,
        ),
    )
    return _attach_requested_analyses(
        result,
        requested_analyses=requested_analyses,
    )


def _final_reconstructed_geometry_issues(
    result: ProcessResult,
    *,
    component_library: ComponentLibrary,
) -> tuple[ValidationIssue, ...]:
    """Report residual span geometry after branch selection and any later repair."""

    origins_by_residue = {
        residue_id: repair.provenance_origins
        for repair in result.repairs
        if repair.kind is RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED
        for residue_id in repair.scope.residue_ids
    }
    if not origins_by_residue:
        return ()

    report = detect_heavy_geometry(
        result.structure,
        component_library=component_library,
        residue_ids=tuple(origins_by_residue),
    )
    return tuple(
        replace(
            issue,
            provenance_origins=origins_by_residue[issue.scope.residue_ids[0]],
        )
        for issue in report.to_issues()
    )


def _final_disulfide_topology_issues(
    structure: ProteinStructure,
) -> tuple[ValidationIssue, ...]:
    """Return unresolved contradictions between S-S evidence and topology."""

    facts = StructureDisulfideTopologyFacts.from_structure(structure)
    return (
        *tuple(
            _disulfide_topology_conflict_issue(conflict) for conflict in facts.conflicts
        ),
        *tuple(
            _disulfide_endpoint_multiplicity_issue(contradiction)
            for contradiction in facts.endpoint_multiplicity_contradictions
        ),
    )


def _disulfide_endpoint_multiplicity_issue(
    contradiction: DisulfideEndpointMultiplicityContradiction,
) -> ValidationIssue:
    """Project one over-assigned disulfide sulfur into a terminal error."""

    partner_tokens = ", ".join(
        partner_atom_ref.display_token()
        for partner_atom_ref in contradiction.partner_atom_refs()
    )
    return ValidationIssue(
        kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
        severity=IssueSeverity.ERROR,
        scope=EventScope.for_residue_set(contradiction.residue_ids()),
        message=(
            f"{contradiction.sulfur_atom_ref.display_token()} participates in multiple "
            f"canonical disulfide relationships "
            f"({len(contradiction.disulfide_atom_ref_pairs)}) with {partner_tokens}; "
            "canonical topology was preserved, "
            "but continuous refinement over the contradictory region was blocked "
            "rather than choosing a partner"
        ),
    )


def _disulfide_topology_conflict_issue(
    conflict: DisulfideTopologyConflict,
) -> ValidationIssue:
    """Project one typed disulfide topology conflict into a validation issue."""

    candidate = conflict.candidate
    if conflict.reason is DisulfideTopologyConflictReason.EXISTING_PAIR_RELATIONSHIP:
        reason = (
            "the candidate endpoint pair already has a noncovalent or unknown "
            "relationship"
        )
    else:
        reason = (
            "one candidate sulfur already has another inter-residue covalent partner"
        )
    return ValidationIssue(
        kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
        severity=IssueSeverity.WARNING,
        scope=EventScope.for_residue_set(conflict.residue_ids()),
        message=(
            f"{candidate.left_residue_id.display_token()}-"
            f"{candidate.right_residue_id.display_token()} has unique SG-SG "
            f"evidence at {candidate.sg_distance_angstrom:.2f} A, but {reason}; "
            "canonical topology was preserved"
        ),
    )


def _branch_report(
    *,
    branches: tuple[WorkflowBranch, ...],
    planning_context: WorkflowPlanningContext,
    stop_reason: SpeculativeStopReason,
) -> WorkflowBranchReport:
    """Build the preference report for retained terminal and frontier candidates."""

    return WorkflowBranchReport.from_outcomes(
        tuple(branch.outcome for branch in branches),
        branch_preference_policy=planning_context.branch_preference_policy,
        stop_reason=stop_reason,
    )


def _attach_requested_analyses(
    result: ProcessResult,
    *,
    requested_analyses: frozenset[AnalysisKind],
) -> ProcessResult:
    """Attach structured analyses to one completed workflow result."""

    if not requested_analyses:
        return result

    return result.with_analyses(
        build_analysis_bundle(
            result.structure,
            requested_analyses=requested_analyses,
        )
    )


def final_parser_readability_issues(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
) -> tuple[ValidationIssue, ...]:
    """Diagnose final parser readability from canonical output geometry.

    Parameters
    ----------
    structure : ProteinStructure
        Returned structure, including any partial completion.
    component_library : ComponentLibrary
        Definitions used to distinguish canonical and parser-inferred connections.

    Returns
    -------
    tuple[ValidationIssue, ...]
        Parser defects and topology blockers for final reporting.
    """

    pdb_block_projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    parser_probe = probe_rdkit_no_conect_parser_readability(
        structure,
        component_library=component_library,
        pdb_block_projector=pdb_block_projector,
    )
    parser_readability_issues = parser_probe.issues()
    if not parser_readability_issues:
        return ()

    parser_witness_clusters = parser_probe.extra_proximity_bond_clusters()
    ambiguous_disulfide_blockers = ambiguous_disulfide_parser_witness_blocker_issues(
        structure,
        clusters=parser_witness_clusters,
    )
    return (
        _linked_parser_readability_blocker_issues(
            parser_readability_issues,
            ambiguous_disulfide_blockers=ambiguous_disulfide_blockers,
        )
        + ambiguous_disulfide_blockers
    )


def _linked_parser_readability_blocker_issues(
    parser_readability_issues: tuple[ValidationIssue, ...],
    *,
    ambiguous_disulfide_blockers: tuple[ValidationIssue, ...],
) -> tuple[ValidationIssue, ...]:
    """Return parser-readability issues annotated with terminal blockers."""

    if not ambiguous_disulfide_blockers:
        return parser_readability_issues

    return tuple(
        (
            replace(
                issue,
                message=(
                    f"{issue.message}; parser-readability repair is blocked by "
                    "ambiguous disulfide topology, so ordinary parser-witness local "
                    "FF repair was not proposed"
                ),
            )
            if _issue_overlaps_any_blocker(
                issue,
                blockers=ambiguous_disulfide_blockers,
            )
            else issue
        )
        for issue in parser_readability_issues
    )


def _issue_overlaps_any_blocker(
    issue: ValidationIssue,
    *,
    blockers: tuple[ValidationIssue, ...],
) -> bool:
    """Return whether one issue shares structure/residue scope with blockers."""

    if not issue.scope.residue_ids:
        return True

    issue_residue_ids = frozenset(issue.scope.residue_ids)
    return any(
        not blocker.scope.residue_ids
        or bool(issue_residue_ids.intersection(blocker.scope.residue_ids))
        for blocker in blockers
    )
