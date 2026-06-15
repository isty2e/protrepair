"""Hydrogen completion orchestration over primitive hydrogen transforms."""

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.hydrogen.cleanup import (
    cleanup_residual_hydrogen_clashes,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.repair_stage import (
    apply_repair_stage_local_refinement,
)
from protrepair.transformer.result import TransformationResult


def add_hydrogens(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    reference_structure: ProteinStructure | None = None,
    *,
    prepare_heavy_atoms: bool = True,
    target_residue_ids: frozenset[ResidueId] | None = None,
    orphan_fragment_policy: OrphanFragmentPolicy = OrphanFragmentPolicy.REBUILD,
    protonate_histidines: bool = False,
    local_refinement: RepairLocalRefinementDirective | None = None,
) -> TransformationResult:
    """Add hydrogens to supported chains while preserving input ligands."""

    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    prepared_structure = structure
    repairs: tuple[RepairEvent, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    if prepare_heavy_atoms:
        heavy_result = repair_heavy_atoms_core(
            structure,
            component_library=library,
            reference_structure=reference_structure,
            target_residue_ids=target_residue_ids,
            orphan_fragment_policy=orphan_fragment_policy,
            local_refinement=local_refinement,
        )
        prepared_structure = heavy_result.structure
        repairs = heavy_result.repairs
        issues = heavy_result.issues

    placement_result = materialize_hydrogens_core(
        prepared_structure,
        component_library=library,
        target_residue_ids=target_residue_ids,
        protonate_histidines=protonate_histidines,
    )
    staged_result = TransformationResult(
        structure=placement_result.structure,
        repairs=(*repairs, *placement_result.repairs),
        issues=(*issues, *placement_result.issues),
    )
    refinement_result = apply_repair_stage_local_refinement(
        staged_result,
        local_refinement=local_refinement,
        component_library=library,
    )
    return _apply_hydrogen_cleanup_stage(
        refinement_result,
        component_library=library,
        target_residue_ids=target_residue_ids,
    )


def _apply_hydrogen_cleanup_stage(
    result: TransformationResult,
    *,
    component_library: ComponentLibrary,
    target_residue_ids: frozenset[ResidueId] | None,
) -> TransformationResult:
    """Execute the ligand-aware post-hydrogen cleanup stage."""

    if not result.structure.constitution.ligands:
        return result

    cleaned_structure, cleanup_issues = cleanup_residual_hydrogen_clashes(
        result.structure,
        component_library=component_library,
        target_residue_ids=target_residue_ids,
    )
    return TransformationResult(
        structure=cleaned_structure,
        repairs=result.repairs,
        issues=result.issues + tuple(cleanup_issues),
    )


__all__ = ["add_hydrogens", "materialize_hydrogens_core"]
