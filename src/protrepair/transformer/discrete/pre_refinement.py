"""Deterministic discrete pre-refinement transformers."""

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.clashes import ClashDetectionBasis
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.discrete.axis_rotation import (
    select_best_discrete_correction_candidate,
    snapshot_has_applicable_axis_rotation_correction,
)


class DiscretePreRefinementCorrectionTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer for residue-local discrete flips."""

    def __init__(
        self,
        component_library: ComponentLibrary,
        *,
        clash_basis: ClashDetectionBasis | None = None,
    ) -> None:
        self._component_library = component_library
        self._clash_basis = clash_basis

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether any referenced residue admits axis-rotation correction."""

        return snapshot_has_applicable_axis_rotation_correction(
            context.source_snapshot,
            context.atom_input,
            self._component_library,
        )

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after residue-local discrete correction selection."""

        working_snapshot = context.source_snapshot
        for residue_id in context.atom_input.referenced_residue_ids():
            candidate = select_best_discrete_correction_candidate(
                snapshot=working_snapshot,
                atom_input=context.atom_input,
                structure=working_snapshot.structure,
                residue_id=residue_id,
                component_library=self._component_library,
                clash_basis=self._clash_basis,
            )
            if candidate is None:
                continue

            working_snapshot = working_snapshot.with_structure(candidate.structure)

        return working_snapshot


def apply_discrete_pre_refinement_corrections(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    *,
    component_library: ComponentLibrary,
) -> ProteinStructureSnapshot:
    """Return one snapshot after residue-local discrete correction selection."""

    return DiscretePreRefinementCorrectionTransformer(component_library).transform(
        ProteinTransformationContext.from_snapshot_atom_input(snapshot, atom_input)
    )
