"""Reference-guided atom grafting for completion transforms."""

from dataclasses import dataclass

from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.completion.shared.domain import ResidueCompletionSite
from protrepair.transformer.context import ProteinTransformationContext


@dataclass(frozen=True, slots=True)
class ReferenceAtomGraftTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer for one reference-guided graft completion step."""

    site: ResidueCompletionSite

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether this residue admits one reference-guided graft patch."""

        if not self.site.is_selected_by(
            context.source_snapshot,
            context.atom_input,
        ):
            return False

        payload = self.site.payload(context.source_snapshot)
        if payload is None:
            return False

        return self.site.reference_graft_patch(context.source_snapshot) is not None

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after reference-guided atom grafting."""

        payload = self.site.payload(context.source_snapshot)
        if payload is None:
            return context.source_snapshot

        patch = self.site.reference_graft_patch(context.source_snapshot)
        if patch is None:
            return context.source_snapshot

        return self.site.apply_patch(context.source_snapshot, patch)
