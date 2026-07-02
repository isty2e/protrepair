"""Repair-stage local refinement orchestration."""

from dataclasses import dataclass, replace
from enum import Enum

from protrepair.chemistry import ComponentLibrary
from protrepair.errors import RefinementError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.binding import ContinuousRelaxationBindingReason
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.local_pipeline.runtime import (
    execute_local_transformation,
)
from protrepair.transformer.refinement.outcome import (
    RefinementStage,
    RefinementStageOutcome,
    merge_refinement_stage_outcome,
)
from protrepair.transformer.result import TransformationResult


@dataclass(frozen=True, slots=True)
class _LocalRefinementPrerequisiteTargets:
    """Chemical completion prerequisites for one local refinement directive."""

    heavy_residue_ids: tuple[ResidueId, ...] = ()
    hydrogen_residue_ids: tuple[ResidueId, ...] = ()
    retained_non_polymer_hydrogen_residue_ids: tuple[ResidueId, ...] = ()

    def is_empty(self) -> bool:
        """Return whether no prerequisite completion remains."""

        return (
            not self.heavy_residue_ids
            and not self.hydrogen_residue_ids
            and not self.retained_non_polymer_hydrogen_residue_ids
        )


class RepairStageRefinementPrerequisitePolicy(str, Enum):
    """How repair-stage refinement handles missing execution prerequisites."""

    STAGE_MISSING_PREREQUISITES = "stage_missing_prerequisites"
    REJECT_INCOMPLETE_DOMAIN = "reject_incomplete_domain"


def apply_repair_stage_local_refinement(
    result: TransformationResult,
    *,
    local_refinement: RepairLocalRefinementDirective | None,
    component_library: ComponentLibrary,
    prerequisite_policy: RepairStageRefinementPrerequisitePolicy = (
        RepairStageRefinementPrerequisitePolicy.STAGE_MISSING_PREREQUISITES
    ),
) -> TransformationResult:
    """Apply the optional local-refinement stage after chemical completion."""

    if local_refinement is None:
        return result

    if not isinstance(
        prerequisite_policy,
        RepairStageRefinementPrerequisitePolicy,
    ):
        raise TypeError(
            "repair-stage local-refinement prerequisite_policy must be a "
            "RepairStageRefinementPrerequisitePolicy value"
        )

    staged_result = result
    if (
        prerequisite_policy
        is RepairStageRefinementPrerequisitePolicy.STAGE_MISSING_PREREQUISITES
    ):
        staged_result = _apply_local_refinement_prerequisites(
            result,
            local_refinement=local_refinement,
            component_library=component_library,
        )
    snapshot = ProteinStructureSnapshot.from_structure(staged_result.structure)
    try:
        bound_execution = local_refinement.bind_execution(
            snapshot,
            component_library=component_library,
        )
    except (RefinementError, ValueError) as error:
        return _merge_rejected_refinement_outcome(
            staged_result,
            local_refinement=local_refinement,
            message=str(error),
        )

    if _should_retry_recommended_mmff_with_uff(bound_execution.binding_decision.reason):
        try:
            refinement_result = execute_local_transformation(
                ProteinTransformationContext.from_snapshot_atom_input(
                    snapshot,
                    bound_execution.atom_input,
                ),
                spec=bound_execution.binding_decision.settings,
                component_library=component_library,
            )
        except RefinementError as error:
            if not _is_mmff_parameterization_failure(error):
                return _merge_rejected_refinement_outcome(
                    staged_result,
                    local_refinement=local_refinement,
                    message=str(error),
                )

            try:
                refinement_result = execute_local_transformation(
                    ProteinTransformationContext.from_snapshot_atom_input(
                        snapshot,
                        bound_execution.atom_input,
                    ),
                    spec=bound_execution.binding_decision.settings.__class__(
                        backend_name=bound_execution.binding_decision.settings.backend_name,
                        force_field=ContinuousRelaxationForceField.UFF,
                        context_radius_angstrom=(
                            bound_execution.binding_decision.settings.context_radius_angstrom
                        ),
                        max_iterations=(
                            bound_execution.binding_decision.settings.max_iterations
                        ),
                    ),
                    component_library=component_library,
                )
            except RefinementError as fallback_error:
                return _merge_rejected_refinement_outcome(
                    staged_result,
                    local_refinement=local_refinement,
                    message=str(fallback_error),
                )
    else:
        try:
            refinement_result = execute_local_transformation(
                ProteinTransformationContext.from_snapshot_atom_input(
                    snapshot,
                    bound_execution.atom_input,
                ),
                spec=bound_execution.binding_decision.settings,
                component_library=component_library,
            )
        except RefinementError as error:
            return _merge_rejected_refinement_outcome(
                staged_result,
                local_refinement=local_refinement,
                message=str(error),
            )
    return merge_refinement_stage_outcome(
        staged_result,
        RefinementStageOutcome.from_result(
            refinement_result,
            stage=RefinementStage.REPAIR,
        ),
    )


def _merge_rejected_refinement_outcome(
    result: TransformationResult,
    *,
    local_refinement: RepairLocalRefinementDirective,
    message: str,
) -> TransformationResult:
    """Merge one rejected repair-stage refinement outcome into the workflow result."""

    return merge_refinement_stage_outcome(
        result,
        RefinementStageOutcome.rejected(
            result.structure,
            stage=RefinementStage.REPAIR,
            residue_id=local_refinement.single_focus_residue_id(),
            message=message,
        ),
    )


def _apply_local_refinement_prerequisites(
    result: TransformationResult,
    *,
    local_refinement: RepairLocalRefinementDirective,
    component_library: ComponentLibrary,
) -> TransformationResult:
    """Stage proposal-local completion prerequisites before one FF execution."""

    staged_result = result
    for _ in range(4):
        prerequisite_targets = _local_refinement_prerequisite_targets(
            staged_result.structure,
            local_refinement=local_refinement,
            component_library=component_library,
        )
        if prerequisite_targets.is_empty():
            return staged_result

        if prerequisite_targets.heavy_residue_ids:
            from protrepair.transformer.completion.heavy.core import (
                repair_heavy_atoms_core,
            )

            heavy_result = repair_heavy_atoms_core(
                staged_result.structure,
                component_library=component_library,
                reference_structure=None,
                augment_c_terminal_oxt=False,
                target_residue_ids=frozenset(
                    prerequisite_targets.heavy_residue_ids
                ),
                local_refinement=None,
            )
            staged_result = _merge_completion_stage_result(
                staged_result,
                stage_result=heavy_result,
            )
            continue

        if prerequisite_targets.hydrogen_residue_ids:
            from protrepair.transformer.completion.hydrogen.core import (
                materialize_hydrogens_core,
            )

            hydrogen_result = materialize_hydrogens_core(
                staged_result.structure,
                component_library=component_library,
                target_residue_ids=frozenset(
                    prerequisite_targets.hydrogen_residue_ids
                ),
            )
            staged_result = _merge_completion_stage_result(
                staged_result,
                stage_result=hydrogen_result,
            )
            continue

        if prerequisite_targets.retained_non_polymer_hydrogen_residue_ids:
            from protrepair.transformer.completion.retained_non_polymer_hydrogen import (  # noqa: E501
                repair as retained_non_polymer_hydrogen_repair,
            )

            retained_non_polymer_result = (
                retained_non_polymer_hydrogen_repair.add_retained_non_polymer_hydrogens(
                    staged_result.structure,
                    component_library=component_library,
                    target_residue_ids=frozenset(
                        prerequisite_targets.retained_non_polymer_hydrogen_residue_ids
                    ),
                )
            )
            staged_result = _merge_completion_stage_result(
                staged_result,
                stage_result=retained_non_polymer_result,
            )
            continue

    return staged_result


def _local_refinement_prerequisite_targets(
    structure: ProteinStructure,
    *,
    local_refinement: RepairLocalRefinementDirective,
    component_library: ComponentLibrary,
) -> _LocalRefinementPrerequisiteTargets:
    """Return local completion prerequisite targets for one local directive."""

    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = local_refinement.resolve_atom_input(snapshot)
    atom_scope = atom_input.observed_atom_scope(snapshot)
    atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=component_library,
        context_radius_angstrom=local_refinement.config.context_radius_angstrom,
    )
    continuous_region_readiness_facts = (
        atom_scope_facts.continuous_region_readiness_facts
    )
    chemistry_by_residue_id = {
        residue_facts.residue_id: residue_facts
        for residue_facts in (
            continuous_region_readiness_facts.chemistry_readiness_facts.residue_facts
        )
    }
    heavy_residue_ids: list[ResidueId] = []
    hydrogen_residue_ids: list[ResidueId] = []
    retained_non_polymer_hydrogen_residue_ids: list[ResidueId] = []
    for coverage_facts in (
        continuous_region_readiness_facts.coverage_facts.residue_facts
    ):
        chemistry_facts = chemistry_by_residue_id.get(coverage_facts.residue_id)
        if chemistry_facts is None or not chemistry_facts.is_supported():
            continue

        if (
            coverage_facts.requires_backbone_completion()
            or coverage_facts.requires_sidechain_completion()
        ):
            heavy_residue_ids.append(coverage_facts.residue_id)
            continue

        if chemistry_facts.needs_hydrogenation():
            hydrogen_residue_ids.append(coverage_facts.residue_id)

    for retained_fact in (
        continuous_region_readiness_facts.chemistry_readiness_facts.retained_non_polymer_facts
    ):
        if not retained_fact.requires_hydrogen_completion():
            continue
        retained_non_polymer_hydrogen_residue_ids.append(retained_fact.residue_id)

    return _LocalRefinementPrerequisiteTargets(
        heavy_residue_ids=tuple(heavy_residue_ids),
        hydrogen_residue_ids=tuple(hydrogen_residue_ids),
        retained_non_polymer_hydrogen_residue_ids=tuple(
            retained_non_polymer_hydrogen_residue_ids
        ),
    )


def _merge_completion_stage_result(
    result: TransformationResult,
    *,
    stage_result: TransformationResult,
) -> TransformationResult:
    """Return one process result with one local prerequisite stage merged in."""

    return replace(
        result,
        structure=stage_result.structure,
        repairs=result.repairs + stage_result.repairs,
        issues=result.issues + stage_result.issues,
    )


def _should_retry_recommended_mmff_with_uff(
    reason: ContinuousRelaxationBindingReason,
) -> bool:
    """Return whether one recommended binding should retry UFF after MMFF."""

    return reason is ContinuousRelaxationBindingReason.HYDROGENATED_DOMAIN_POLICY


def _is_mmff_parameterization_failure(error: RefinementError) -> bool:
    """Return whether one refinement error came from MMFF parameterization."""

    message = str(error)
    return (
        "RDKit MMFF could not parameterize" in message
        or "RDKit MMFF could not build a force field" in message
    )
