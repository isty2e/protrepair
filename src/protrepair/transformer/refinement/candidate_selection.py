"""Candidate policies for assessed refinement outputs."""

from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.artifacts.patch import MovedAtomDelta, StructureDelta
from protrepair.transformer.refinement.acceptance import (
    AssessedRefinementResult,
    refinement_metrics_order_key,
)


def assessed_refinement_candidate_order_key(
    assessment: AssessedRefinementResult,
) -> tuple[int, int, float, float, int, float, int, int]:
    """Return a stable ordering key for one assessed candidate."""

    return refinement_metrics_order_key(assessment.accepted_metrics())


def materialize_assessed_refinement_candidate(
    assessment: AssessedRefinementResult,
    *,
    original_structure: ProteinStructure,
    pre_backend_moved_atom_indices: tuple[AtomIndex, ...],
) -> RegionTransformationResult:
    """Return the final result after fallback policy and move merging."""

    executed_result = assessment.executed_result
    if assessment.is_accepted():
        materialized_result = executed_result
    else:
        rejection_issue = assessment.rejection_issue
        assert rejection_issue is not None
        return RegionTransformationResult(
            refined_structure=original_structure,
            delta=StructureDelta(
                before_constitution=original_structure.constitution,
                after_constitution=original_structure.constitution,
            ),
            issues=(
                *executed_result.issues,
                rejection_issue,
            ),
            backend_name=executed_result.backend_name,
            backend_version=executed_result.backend_version,
        )

    if not pre_backend_moved_atom_indices:
        return materialized_result

    pre_backend_moved_atoms = tuple(
        MovedAtomDelta(
            before_atom_index=atom_index,
            after_atom_index=atom_index,
        )
        for atom_index in pre_backend_moved_atom_indices
    )

    return RegionTransformationResult(
        refined_structure=materialized_result.refined_structure,
        delta=StructureDelta(
            before_constitution=materialized_result.delta.before_constitution,
            after_constitution=materialized_result.delta.after_constitution,
            moved_atoms=(
                *pre_backend_moved_atoms,
                *materialized_result.delta.moved_atoms,
            ),
            created_atom_indices=materialized_result.delta.created_atom_indices,
            deleted_atom_indices=materialized_result.delta.deleted_atom_indices,
            created_residue_indices=materialized_result.delta.created_residue_indices,
            deleted_residue_indices=materialized_result.delta.deleted_residue_indices,
            residue_identity_rewrites=(
                materialized_result.delta.residue_identity_rewrites
            ),
            residue_topology_rewrites=(
                materialized_result.delta.residue_topology_rewrites
            ),
            atom_element_rewrites=materialized_result.delta.atom_element_rewrites,
            atom_formal_charge_rewrites=(
                materialized_result.delta.atom_formal_charge_rewrites
            ),
            bond_topology_rewrites=materialized_result.delta.bond_topology_rewrites,
        ),
        issues=materialized_result.issues,
        backend_name=materialized_result.backend_name,
        backend_version=materialized_result.backend_version,
    )
