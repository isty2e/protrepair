"""Current H-preparation diagnostics for workflow assessment and output."""

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.polymer_hydrogen import unresolved_fixed_hydrogen_issue
from protrepair.diagnostics.retained_hydrogen import unresolved_retained_hydrogen_issue
from protrepair.scope import WholeStructureScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
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
        Current H goals; satisfied goals require no further warning.
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
        Stale or out-of-scope unresolved-H warnings are omitted.

    Raises
    ------
    ValueError
        The supplied observation belongs to a different structure or library.
    """
    retained = tuple(
        issue
        for issue in result.issues
        if issue.kind
        not in {
            ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN,
            ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN,
        }
    )
    structure = result.structure
    selected: set[ResidueId] = set()
    snapshot: ProteinStructureSnapshot | None = None
    if requested_goal_report is not None:
        for outcome in requested_goal_report.outcomes:
            if (
                outcome.requested_goal.value is not HydrogenCoverageState.COMPLETE
                or outcome.is_satisfied()
            ):
                continue
            scope = outcome.requested_goal.scope
            if isinstance(scope, WholeStructureScope):
                selected.update(
                    r.residue_id
                    for r in structure.constitution.iter_residues(include_ligands=True)
                )
            elif OBSERVED_ATOM_SCOPE_LOWERING.supports(scope):
                if snapshot is None:
                    snapshot = ProteinStructureSnapshot.from_structure(structure)
                lowered = OBSERVED_ATOM_SCOPE_LOWERING.lower(scope, carrier=snapshot)
                selected.update(ref.residue_id for ref in lowered.atom_refs)

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
    if not selected:
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
        and observation.hydrogen_expectation_model.resolution_for_retained_non_polymer(
            fact.residue_id
        ).unresolved_hydrogen_requirement
        is not None
    )
    return retained + polymer_issues + retained_issues
