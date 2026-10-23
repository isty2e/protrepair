"""Current fixed-polymer H diagnostics for workflow assessment and output."""

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.polymer_hydrogen import unresolved_fixed_hydrogen_issue
from protrepair.scope import WholeStructureScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.local.projection import (
    LocalContinuousExecutionResidueProjection,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts.result import RequestedGoalReport


def project_polymer_hydrogen_issues(
    result: TransformationResult,
    *,
    requested_goal_report: RequestedGoalReport | None,
    component_library: ComponentLibrary,
    repair_refinement: RepairRefinementSpec | None = None,
    preparation: PolymerMicrostatePreparation | None = None,
) -> tuple[ValidationIssue, ...]:
    """Replace fixed-polymer H history with current requested-site warnings.

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
    preparation : PolymerMicrostatePreparation or None
        Optional seed rebound to the current structure and library before use.

    Returns
    -------
    tuple[ValidationIssue, ...]
        Unrelated issues in their original order and multiplicity, followed by
        at most one current unresolved-H warning per requested polymer residue.
        Stale or out-of-scope unresolved-H warnings are omitted.
    """
    retained = tuple(
        issue
        for issue in result.issues
        if issue.kind is not ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN
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
                    r.residue_id for r in structure.constitution.iter_residues()
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
        except ValueError:
            # Missing prerequisites can prevent region projection before H work.
            selected.update(scope_spec.referenced_residue_ids())
    if not selected:
        return retained

    preparation = (
        PolymerMicrostatePreparation(
            PolymerMicrostateContext(structure), component_library
        )
        if preparation is None
        else preparation.for_inputs(structure, component_library)
    )
    return retained + tuple(
        issue
        for residue in structure.constitution.iter_residues()
        if residue.residue_id in selected
        for issue in (unresolved_fixed_hydrogen_issue(preparation, residue.residue_id),)
        if issue is not None
    )
