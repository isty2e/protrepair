"""Dependent-hydrogen revalidation after accepted local refinement."""

from protrepair.chemistry import ComponentLibrary, RestraintLibrary
from protrepair.diagnostics.clashes import ClashDetectionBasis
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity
from protrepair.diagnostics.parser_readability import (
    rdkit_no_conect_parser_failing_residue_ids,
)
from protrepair.scope import Scope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.artifacts import (
    MovedAtomDelta,
    RegionTransformationResult,
    StructureDelta,
)
from protrepair.transformer.refinement.acceptance import (
    RefinementAcceptanceMetrics,
    measure_refinement_acceptance_metrics_for_scope,
    refinement_metrics_rejected,
)


def revalidate_dependent_hydrogens_after_refinement(
    result: RegionTransformationResult,
    *,
    selected_scope: Scope,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    current_metrics: RefinementAcceptanceMetrics | None = None,
    clash_basis: ClashDetectionBasis | None = None,
) -> RegionTransformationResult:
    """Re-materialize stale dependent hydrogens only behind validation gates."""

    active_current_metrics = current_metrics
    if active_current_metrics is None:
        active_current_metrics = measure_refinement_acceptance_metrics_for_scope(
            result.refined_structure,
            selected_scope=selected_scope,
            component_library=component_library,
            restraint_library=restraint_library,
            clash_basis=clash_basis,
        )
    if active_current_metrics.parser_compatibility.rdkit_sanitize_readable is not False:
        return result

    hydrogen_only_failure = _has_hydrogen_only_parser_visibility_failure(
        active_current_metrics
    )
    if not hydrogen_only_failure:
        return result

    target_residue_ids = _dependent_hydrogen_revalidation_residue_ids(
        result,
        component_library=component_library,
    )
    if not target_residue_ids:
        return result

    from protrepair.transformer.completion.hydrogen.core import (
        materialize_hydrogens_core,
    )

    hydrogen_result = materialize_hydrogens_core(
        result.refined_structure,
        component_library=component_library,
        target_residue_ids=frozenset(target_residue_ids),
    )
    if any(issue.severity is IssueSeverity.ERROR for issue in hydrogen_result.issues):
        return result

    candidate_structure = hydrogen_result.structure
    if not _is_coordinate_only_hydrogen_rematerialization(
        result.refined_structure,
        candidate_structure,
    ):
        return result

    moved_hydrogen_atom_indices = _hydrogen_atom_indices_with_changed_positions(
        result.refined_structure,
        candidate_structure,
    )
    if not moved_hydrogen_atom_indices:
        return result

    candidate_metrics = measure_refinement_acceptance_metrics_for_scope(
        candidate_structure,
        selected_scope=selected_scope,
        component_library=component_library,
        restraint_library=restraint_library,
        clash_basis=clash_basis,
    )
    if refinement_metrics_rejected(active_current_metrics, candidate_metrics):
        return result

    return _with_refreshed_hydrogen_coordinates(
        result,
        candidate_structure=candidate_structure,
        moved_hydrogen_atom_indices=moved_hydrogen_atom_indices,
        appended_issues=hydrogen_result.issues,
    )


def _has_hydrogen_only_parser_visibility_failure(
    metrics: RefinementAcceptanceMetrics,
) -> bool:
    """Return whether parser visibility currently fails only through H contacts."""

    parser_compatibility = metrics.parser_compatibility
    return (
        parser_compatibility.rdkit_sanitize_readable is False
        and parser_compatibility.extra_proximity_bond_count > 0
        and parser_compatibility.extra_heavy_proximity_bond_count == 0
    )


def _dependent_hydrogen_revalidation_residue_ids(
    result: RegionTransformationResult,
    *,
    component_library: ComponentLibrary,
) -> tuple[ResidueId, ...]:
    """Return residue ids eligible for dependent-hydrogen re-materialization."""

    candidate_residue_ids = (
        *_heavy_moved_polymer_residue_ids(result),
        *rdkit_no_conect_parser_failing_residue_ids(
            result.refined_structure,
            component_library=component_library,
        ),
    )
    residue_ids: list[ResidueId] = []
    seen_residue_ids: set[ResidueId] = set()
    for residue_id in candidate_residue_ids:
        if residue_id in seen_residue_ids:
            continue
        if not _supports_dependent_hydrogen_revalidation(
            result.refined_structure,
            residue_id,
            component_library=component_library,
        ):
            continue

        residue_ids.append(residue_id)
        seen_residue_ids.add(residue_id)

    return tuple(residue_ids)


def _heavy_moved_polymer_residue_ids(
    result: RegionTransformationResult,
) -> tuple[ResidueId, ...]:
    """Return polymer residue ids with moved heavy-atom coordinates."""

    constitution = result.refined_structure.constitution
    residue_ids: list[ResidueId] = []
    seen_residue_ids: set[ResidueId] = set()
    for moved_atom in result.delta.moved_atoms:
        atom_index = moved_atom.after_atom_index
        if atom_index.value >= len(constitution.atom_slots):
            continue

        atom_site = constitution.atom_site_at(atom_index)
        if atom_site.is_hydrogen():
            continue

        residue_index = constitution.residue_index_for_atom_index(atom_index)
        residue_site = constitution.residue_site_at(residue_index)
        if residue_site.is_hetero or residue_site.residue_id in seen_residue_ids:
            continue

        residue_ids.append(residue_site.residue_id)
        seen_residue_ids.add(residue_site.residue_id)

    return tuple(residue_ids)


def _supports_dependent_hydrogen_revalidation(
    structure: ProteinStructure,
    residue_id: ResidueId,
    *,
    component_library: ComponentLibrary,
) -> bool:
    """Return whether one residue has authoritative template-owned hydrogens."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    if residue_site is None or residue_site.is_hetero:
        return False

    template = component_library.get(residue_site.component_id)
    if template is None or not template.can_add_hydrogens():
        return False

    return any(site.is_hydrogen() for site in residue_site.atom_sites)


def _is_coordinate_only_hydrogen_rematerialization(
    current_structure: ProteinStructure,
    candidate_structure: ProteinStructure,
) -> bool:
    """Return whether the candidate changed only hydrogen coordinates."""

    if candidate_structure.constitution != current_structure.constitution:
        return False
    if candidate_structure.topology != current_structure.topology:
        return False

    for atom_index in (
        AtomIndex(index)
        for index in range(current_structure.geometry.atom_count())
    ):
        current_position = current_structure.geometry.atom_geometry(
            atom_index
        ).position
        candidate_position = candidate_structure.geometry.atom_geometry(
            atom_index
        ).position
        if current_position == candidate_position:
            continue

        atom_site = current_structure.constitution.atom_site_at(atom_index)
        if not atom_site.is_hydrogen():
            return False

    return True


def _hydrogen_atom_indices_with_changed_positions(
    current_structure: ProteinStructure,
    candidate_structure: ProteinStructure,
) -> tuple[AtomIndex, ...]:
    """Return hydrogen atom slots whose coordinates changed in the candidate."""

    changed_atom_indices: list[AtomIndex] = []
    for atom_index in (
        AtomIndex(index)
        for index in range(current_structure.geometry.atom_count())
    ):
        atom_site = current_structure.constitution.atom_site_at(atom_index)
        if not atom_site.is_hydrogen():
            continue

        current_position = current_structure.geometry.atom_geometry(
            atom_index
        ).position
        candidate_position = candidate_structure.geometry.atom_geometry(
            atom_index
        ).position
        if current_position != candidate_position:
            changed_atom_indices.append(atom_index)

    return tuple(changed_atom_indices)


def _with_refreshed_hydrogen_coordinates(
    result: RegionTransformationResult,
    *,
    candidate_structure: ProteinStructure,
    moved_hydrogen_atom_indices: tuple[AtomIndex, ...],
    appended_issues: tuple[ValidationIssue, ...],
) -> RegionTransformationResult:
    """Return a result carrying accepted hydrogen-coordinate materialization."""

    return RegionTransformationResult(
        refined_structure=candidate_structure,
        delta=StructureDelta(
            before_constitution=result.delta.before_constitution,
            after_constitution=candidate_structure.constitution,
            moved_atoms=(
                *result.delta.moved_atoms,
                *(
                    MovedAtomDelta(
                        before_atom_index=atom_index,
                        after_atom_index=atom_index,
                    )
                    for atom_index in moved_hydrogen_atom_indices
                ),
            ),
            created_atom_indices=result.delta.created_atom_indices,
            deleted_atom_indices=result.delta.deleted_atom_indices,
            created_residue_indices=result.delta.created_residue_indices,
            deleted_residue_indices=result.delta.deleted_residue_indices,
            residue_identity_rewrites=result.delta.residue_identity_rewrites,
            residue_topology_rewrites=result.delta.residue_topology_rewrites,
            atom_element_rewrites=result.delta.atom_element_rewrites,
            atom_formal_charge_rewrites=result.delta.atom_formal_charge_rewrites,
            bond_topology_rewrites=result.delta.bond_topology_rewrites,
        ),
        issues=(*result.issues, *appended_issues),
        backend_name=result.backend_name,
        backend_version=result.backend_version,
    )
