"""Localized side-chain stereochemistry correction over canonical residues."""

from protrepair.chemistry import (
    ComponentLibrary,
    build_default_component_library,
)
from protrepair.diagnostics import (
    StereochemistryReport,
    detect_sidechain_stereochemistry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.heavy.repair import repair_heavy_atoms
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.transformer.completion.stereochemistry.batch import (
    StereochemistryCorrectionBatch,
)
from protrepair.transformer.result import TransformationResult


def correct_sidechain_stereochemistry(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    *,
    target_residue_ids: frozenset[ResidueId] | None = None,
) -> TransformationResult:
    """Repair supported side-chain chirality inversions by local subtree rebuild."""

    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    heavy_structure = structure.without_hydrogens()
    initial_report = detect_sidechain_stereochemistry(
        heavy_structure,
        component_library=library,
    )
    if target_residue_ids is not None:
        initial_report = StereochemistryReport(
            violations=tuple(
                violation
                for violation in initial_report.violations
                if violation.residue_id in target_residue_ids
            )
        )
    if initial_report.is_empty():
        return TransformationResult(
            structure=structure,
            repairs=(),
            issues=(),
        )

    correction_batch = StereochemistryCorrectionBatch.from_violations(
        initial_report.violations
    )
    prepared_structure = correction_batch.prepared_structure(
        heavy_structure,
        component_library=library,
    )
    repair_result = repair_heavy_atoms(
        prepared_structure,
        component_library=library,
        target_residue_ids=target_residue_ids,
    )
    completed_result = correction_batch.completed_result(
        original_structure=structure,
        heavy_repair_result=repair_result,
        component_library=library,
    )
    corrected_residue_ids = (
        correction_batch.corrected_residue_ids()
        if target_residue_ids is None
        else target_residue_ids
    )
    if not _residue_subset_contains_hydrogens(
        structure,
        residue_ids=corrected_residue_ids,
    ):
        return completed_result

    hydrogen_result = add_hydrogens(
        completed_result.structure,
        component_library=library,
        prepare_heavy_atoms=False,
        target_residue_ids=corrected_residue_ids,
    )
    return TransformationResult(
        structure=hydrogen_result.structure,
        repairs=(*completed_result.repairs, *hydrogen_result.repairs),
        issues=(*completed_result.issues, *hydrogen_result.issues),
    )


def _residue_subset_contains_hydrogens(
    structure: ProteinStructure,
    *,
    residue_ids: frozenset[ResidueId],
) -> bool:
    """Return whether any targeted residue currently carries explicit hydrogens."""

    for residue_id in residue_ids:
        residue = structure.constitution.residue_or_ligand(residue_id)
        if residue is None:
            continue
        if any(atom_site.element == "H" for atom_site in residue.atom_sites):
            return True

    return False
