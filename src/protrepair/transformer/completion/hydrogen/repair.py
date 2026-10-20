"""Hydrogen completion orchestration over primitive hydrogen transforms."""

from dataclasses import replace

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.hydrogen.cleanup import (
    cleanup_residual_hydrogen_clashes,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.protonation import (
    HistidineProtonationRequest,
    normalize_histidine_protonation_request,
)
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
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
    histidine_protonation: HistidineProtonationRequest | None = None,
    protonate_histidines: bool = False,
    local_refinement: RepairLocalRefinementDirective | None = None,
    preparation: PolymerMicrostatePreparation | None = None,
) -> TransformationResult:
    """Add hydrogens to supported chains while preserving input ligands.

    Parameters
    ----------
    structure : ProteinStructure
        Current canonical input.
    component_library : ComponentLibrary or None
        Active definitions, or the default library.
    reference_structure : ProteinStructure or None
        Optional heavy-atom reconstruction donor.
    prepare_heavy_atoms : bool
        Complete the scaffold and required terminal O before H placement.
    target_residue_ids : frozenset[ResidueId] or None
        Polymer residues to prepare; None selects all polymer residues.
    orphan_fragment_policy : OrphanFragmentPolicy
        Handling of disconnected heavy fragments.
    histidine_protonation : HistidineProtonationRequest or None
        Explicit HIS selection, or no new request.
    protonate_histidines : bool
        Request the default ratio; mutually exclusive with an explicit request.
    local_refinement : RepairLocalRefinementDirective or None
        Optional post-preparation refinement.
    preparation : PolymerMicrostatePreparation or None
        Optional seed; every consuming stage rebinds after input changes.

    Returns
    -------
    TransformationResult
        Prepared structure with repairs and residue-local diagnostics.

    Raises
    ------
    TypeError
        Histidine request inputs are not supported canonical values.
    ValueError
        Conflicting histidine request forms were supplied.
    RdkitUnavailableError
        Required native H placement is unavailable.
    """

    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    histidine_protonation_request = normalize_histidine_protonation_request(
        histidine_protonation,
        protonate_histidines=protonate_histidines,
    )
    fixed_ligand_hydrogens = frozenset(
        AtomRef(residue.residue_id, atom.name)
        for residue in structure.constitution.ligands
        for atom in residue.atom_sites
        if atom.is_hydrogen()
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
            augment_c_terminal_oxt=False,
            # The core uses this only to authorize refinement-gated scaffold
            # placement. Execution stays below, after coupled H/graph realization.
            local_refinement=local_refinement,
        )
        prepared_structure = heavy_result.structure
        repairs = heavy_result.repairs
        issues = heavy_result.issues
        preparation = (
            PolymerMicrostatePreparation(
                PolymerMicrostateContext(prepared_structure), library
            )
            if preparation is None
            else preparation.for_inputs(prepared_structure, library)
        )
        terminal_targets = frozenset(
            residue.residue_id
            for chain in prepared_structure.constitution.chains
            for residue in chain.residues
            if target_residue_ids is None or residue.residue_id in target_residue_ids
            if any(
                target.requires_terminal_oxygen()
                for target in preparation.targets_for(residue.residue_id)
            )
        )
        if terminal_targets:
            terminal_result = augment_c_terminal_oxt(
                prepared_structure, library, target_residue_ids=terminal_targets
            )
            prepared_structure = terminal_result.structure
            repairs += terminal_result.repairs
            issues += terminal_result.issues

    placement_result = materialize_hydrogens_core(
        prepared_structure,
        component_library=library,
        source_isotope_structure=structure,
        target_residue_ids=target_residue_ids,
        histidine_protonation=histidine_protonation_request,
        preparation=preparation,
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
        fixed_hydrogen_atom_refs=fixed_ligand_hydrogens,
    )


def _apply_hydrogen_cleanup_stage(
    result: TransformationResult,
    *,
    component_library: ComponentLibrary,
    target_residue_ids: frozenset[ResidueId] | None,
    fixed_hydrogen_atom_refs: frozenset[AtomRef],
) -> TransformationResult:
    """Execute the ligand-aware post-hydrogen cleanup stage."""

    if not result.structure.constitution.ligands:
        return result

    cleaned_structure, cleanup_issues = cleanup_residual_hydrogen_clashes(
        result.structure,
        component_library=component_library,
        target_residue_ids=target_residue_ids,
        fixed_hydrogen_atom_refs=fixed_hydrogen_atom_refs,
    )
    return replace(
        result,
        structure=cleaned_structure,
        issues=result.issues + tuple(cleanup_issues),
    )


__all__ = ["add_hydrogens", "materialize_hydrogens_core"]
