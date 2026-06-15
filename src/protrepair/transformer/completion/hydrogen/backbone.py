"""Primitive backbone-hydrogen propagation transformer."""

from dataclasses import dataclass

from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.completion.hydrogen.domain import HydrogenResidueSite
from protrepair.transformer.context import ProteinTransformationContext


@dataclass(frozen=True, slots=True)
class BackboneHydrogenPropagationTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Propagate one residue's peptide hydrogen onto the next residue."""

    site: HydrogenResidueSite

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one backbone-hydrogen propagation may run."""

        if self.site.next_residue_index is None:
            return False

        selected_atom_index_set = frozenset(context.atom_input.atom_indices)
        if selected_atom_index_set.isdisjoint(
            context.source_snapshot.structure.constitution.atom_indices_for_residue_index(
                self.site.residue_index
            )
        ):
            return False
        if selected_atom_index_set.isdisjoint(
            context.source_snapshot.structure.constitution.atom_indices_for_residue_index(
                self.site.next_residue_index
            )
        ):
            return False

        return self.site.backbone_hydrogen_position(context.source_snapshot) is not None

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after backbone-hydrogen propagation."""

        if not self.is_applicable(context):
            return context.source_snapshot

        position = self.site.backbone_hydrogen_position(context.source_snapshot)
        if position is None:
            return context.source_snapshot

        return self.site.propagate_backbone_hydrogen(
            context.source_snapshot,
            position,
        )
