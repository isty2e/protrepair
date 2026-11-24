"""Current H-preparation diagnostics for workflow assessment and output."""

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.microstate.bound_protonation import active_site_protonations
from protrepair.chemistry.microstate.context import input_declares_thiolate
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.chemistry.retained_non_polymer.definition import (
    NamedGraphAssignment,
)
from protrepair.chemistry.retained_non_polymer.evidence import RefusedSmiles
from protrepair.chemistry.retained_non_polymer.orthophosphate import (
    is_orthophosphate_component,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.polymer_hydrogen import (
    PendingCapLinkageIssue,
    declared_thiolate_conflict,
    linked_cap_issue,
    unresolved_fixed_hydrogen_issue,
)
from protrepair.diagnostics.retained_hydrogen import unresolved_retained_hydrogen_issue
from protrepair.scope import WholeStructureScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.bound_peptide_ports import active_peptide_ports
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.completion.terminal.cap_placement import (
    CapPortAssessment,
    PortCondition,
)
from protrepair.transformer.local.projection import (
    LocalContinuousExecutionResidueProjection,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts.result import RequestedGoalReport


def project_hydrogen_issues(
    result: TransformationResult,
    *,
    requested_goal_report: RequestedGoalReport | None,
    component_library: ComponentLibrary,
    repair_refinement: RepairRefinementSpec | None = None,
    preparation: AminoAcidPreparation | None = None,
    chemistry_observation: StructureChemistryObservation | None = None,
) -> tuple[ValidationIssue, ...]:
    """Replace H-preparation history with current requested-site warnings.

    Parameters
    ----------
    result : TransformationResult
        Current structure and accumulated diagnostics.
    requested_goal_report : RequestedGoalReport or None
        Current H goals. A complete inventory can still contain contradictory
        input chemistry.
    component_library : ComponentLibrary
        Active chemistry definitions.
    repair_refinement : RepairRefinementSpec or None
        Local request whose included region requires H preparation.
    preparation : AminoAcidPreparation or None
        Optional seed rebound to the current structure and library before use.
    chemistry_observation : StructureChemistryObservation or None
        Current observation with the active retained chemistry evidence.

    Returns
    -------
    tuple[ValidationIssue, ...]
        Unrelated issues in their original order and multiplicity, followed by
        at most one current unresolved-H warning per requested residue.
        Stale or out-of-scope unresolved-H warnings are omitted. An H-stage
        warning about a reserved cap linkage remains until that exact junction
        passes verification on this structure.

    Raises
    ------
    ValueError
        The supplied observation belongs to a different structure or library.
    """
    structure = result.structure
    ports = active_peptide_ports()
    retained = tuple(
        issue
        for issue in result.issues
        if issue.kind
        not in {
            ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN,
            ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN,
        }
        and not (
            ports is not None
            and isinstance(issue, PendingCapLinkageIssue)
            and ports.reserves(structure, issue.port)
            and CapPortAssessment.observe(ports, structure, issue.port).condition
            is PortCondition.COMPLETE
        )
    )
    selected: set[ResidueId] = set()
    assigned = (
        set()
        if chemistry_observation is None
        else {
            item.residue_id
            for item in chemistry_observation.retained_non_polymer_chemistry_evidence
            if isinstance(item, (NamedGraphAssignment, RefusedSmiles))
        }
    )
    snapshot: ProteinStructureSnapshot | None = None
    hydrogen_goals = (
        ()
        if requested_goal_report is None
        else tuple(
            outcome
            for outcome in requested_goal_report.outcomes
            if outcome.requested_goal.value is HydrogenCoverageState.COMPLETE
        )
    )
    check_when_complete = set(assigned)
    if hydrogen_goals and (binding := active_site_protonations()) is not None:
        # A complete name count can still hide an incompatible fixed H/D inventory.
        check_when_complete.update(
            request.residue_id
            for request in binding.requests.values()
            if binding.identity_failure(structure, request) is None
        )
    thiolates = (
        {
            residue.residue_id
            for residue in structure.constitution.iter_residues()
            if residue.component_id == "CYS"
            and input_declares_thiolate(structure, residue.residue_id)
        }
        if hydrogen_goals or repair_refinement is not None
        else set()
    )
    check_thiolates: set[ResidueId] = set()
    for outcome in hydrogen_goals:
        if outcome.is_satisfied() and not (check_when_complete or thiolates):
            continue
        scope = outcome.requested_goal.scope
        if isinstance(scope, WholeStructureScope):
            in_scope = {
                r.residue_id
                for r in structure.constitution.iter_residues(include_ligands=True)
            }
        elif OBSERVED_ATOM_SCOPE_LOWERING.supports(scope):
            if snapshot is None:
                snapshot = ProteinStructureSnapshot.from_structure(structure)
            lowered = OBSERVED_ATOM_SCOPE_LOWERING.lower(scope, carrier=snapshot)
            in_scope = {ref.residue_id for ref in lowered.atom_refs}
        else:
            continue
        check_thiolates.update(thiolates.intersection(in_scope))
        selected.update(
            check_when_complete.intersection(in_scope)
            if outcome.is_satisfied()
            else in_scope
        )

    if repair_refinement is not None:
        if snapshot is None:
            snapshot = ProteinStructureSnapshot.from_structure(structure)
        scope_spec = repair_refinement.resolved_execution_scope_spec()
        try:
            projection = LocalContinuousExecutionResidueProjection.from_scope_spec(
                snapshot,
                scope_spec,
                context_radius_angstrom=repair_refinement.config.context_radius_angstrom,
                component_library=component_library,
            )
            selected.update(projection.included_polymer_residue_ids)
            selected.update(projection.included_retained_residue_ids)
        except ValueError:
            # Missing prerequisites can prevent region projection before H work.
            selected.update(scope_spec.referenced_residue_ids())
    check_thiolates.update(thiolates.intersection(selected))
    if not selected and not check_thiolates:
        return retained

    observation = (
        StructureChemistryObservation(structure, component_library=component_library)
        if chemistry_observation is None
        else chemistry_observation
    )
    observation.require_context(
        structure,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=(
            observation.retained_non_polymer_chemistry_evidence
        ),
        allow_retained_non_polymer_rdkit_fallback=(
            observation.allow_retained_non_polymer_rdkit_fallback
        ),
    )
    preparation = (
        observation.hydrogen_expectation_model.polymer_preparation
        if preparation is None
        else preparation.for_inputs(structure, component_library)
    )
    polymer_issues = tuple(
        issue
        for residue in structure.constitution.iter_residues()
        if residue.residue_id in selected
        for assessment in (
            observation.hydrogen_expectation_model.polymer_reference_assessments.get(
                residue.residue_id
            ),
        )
        for issue in (
            unresolved_fixed_hydrogen_issue(
                preparation,
                residue.residue_id,
                reference_failure_detail=(
                    assessment.failure.detail
                    if assessment is not None and assessment.failure is not None
                    else None
                ),
            ),
        )
        if issue is not None
    )
    retained_issues = tuple(
        unresolved_retained_hydrogen_issue(
            fact.residue_id, fact.component_id, fact.chemistry_failure
        )
        for fact in observation.projected_facts(
            tuple(
                residue.residue_id
                for residue in structure.constitution.ligands
                if residue.residue_id in selected
            )
        )[1].retained_non_polymer_facts
        if fact.chemistry_failure is not None
        and not fact.has_complete_reference_hydrogens()
        and (
            observation.hydrogen_expectation_model.resolution_for_retained_non_polymer(
                fact.residue_id
            ).unresolved_hydrogen_requirement
            is not None
            or fact.residue_id in assigned
            or (
                fact.residue_id
                in observation.retained_non_polymer_evidence_by_residue_id
                and is_orthophosphate_component(fact.component_id)
            )
        )
    )
    refused_retained_caps = tuple(
        linked_cap_issue(cap, cap.refusal)
        for residue in structure.constitution.ligands
        if residue.residue_id in selected
        and (cap := preparation.linked_peptide_cap(residue.residue_id)) is not None
        and cap.refusal is not None
    )
    sulfur_issues = tuple(
        issue
        for residue_id in sorted(check_thiolates)
        for target in preparation.targets_for(residue_id)
        if (issue := declared_thiolate_conflict(target)) is not None
    )
    if sulfur_issues:
        current_sulfur = frozenset(sulfur_issues)
        retained = tuple(issue for issue in retained if issue not in current_sulfur)
    return (
        retained
        + polymer_issues
        + retained_issues
        + refused_retained_caps
        + sulfur_issues
    )
