"""Heavy-atom completion orchestration over primitive atom transforms."""

from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHeavyAtomSemantics,
    build_default_component_library,
)
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.repair_stage import (
    RepairStageRefinementPrerequisitePolicy,
    apply_repair_stage_local_refinement,
)
from protrepair.transformer.result import TransformationResult


def repair_heavy_atoms(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    reference_structure: ProteinStructure | None = None,
    *,
    augment_c_terminal_oxt: bool = True,
    target_residue_ids: frozenset[ResidueId] | None = None,
    orphan_fragment_policy: OrphanFragmentPolicy = OrphanFragmentPolicy.REBUILD,
    local_refinement: RepairLocalRefinementDirective | None = None,
) -> TransformationResult:
    """Repair missing heavy atoms for supported protein residues."""

    result = repair_heavy_atoms_core(
        structure,
        component_library=component_library,
        reference_structure=reference_structure,
        augment_c_terminal_oxt=augment_c_terminal_oxt,
        target_residue_ids=target_residue_ids,
        orphan_fragment_policy=orphan_fragment_policy,
        local_refinement=local_refinement,
    )
    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    return apply_repair_stage_local_refinement(
        result,
        local_refinement=local_refinement,
        component_library=library,
        prerequisite_policy=_heavy_repair_refinement_prerequisite_policy(
            result,
            local_refinement=local_refinement,
            component_library=library,
        ),
    )


def _heavy_repair_refinement_prerequisite_policy(
    result: TransformationResult,
    *,
    local_refinement: RepairLocalRefinementDirective | None,
    component_library: ComponentLibrary,
) -> RepairStageRefinementPrerequisitePolicy:
    """Return whether heavy repair should stage more chemistry before FF execution."""

    if local_refinement is None:
        return RepairStageRefinementPrerequisitePolicy.STAGE_MISSING_PREREQUISITES

    if _has_refinement_gated_heavy_repair(
        result,
        local_refinement=local_refinement,
        component_library=component_library,
    ):
        return RepairStageRefinementPrerequisitePolicy.REJECT_INCOMPLETE_DOMAIN

    return RepairStageRefinementPrerequisitePolicy.STAGE_MISSING_PREREQUISITES


def _has_refinement_gated_heavy_repair(
    result: TransformationResult,
    *,
    local_refinement: RepairLocalRefinementDirective,
    component_library: ComponentLibrary,
) -> bool:
    """Return whether this heavy pass used refinement only as repair authority."""

    for repair in result.repairs:
        if repair.kind is not RepairEventKind.HEAVY_ATOMS_ADDED:
            continue

        residue_id = repair.residue_id
        if residue_id is None or not local_refinement.targets_residue(residue_id):
            continue

        residue_site = result.structure.constitution.residue_or_ligand(residue_id)
        if residue_site is None:
            continue

        template = component_library.get(residue_site.component_id)
        if template is None:
            continue

        heavy_atom_semantics = template.heavy_atom_semantics
        if (
            isinstance(heavy_atom_semantics, IdealGeometryHeavyAtomSemantics)
            and heavy_atom_semantics.requires_local_refinement()
        ):
            return True

    return False


__all__ = ["repair_heavy_atoms", "repair_heavy_atoms_core"]
