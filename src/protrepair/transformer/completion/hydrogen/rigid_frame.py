"""Rigid-frame hydrogen placement transformers over residue-local hydrogen sites."""

from dataclasses import dataclass

from protrepair.chemistry import IdealGeometryHydrogenSemantics
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.completion.hydrogen.component_patch import (
    generate_component_hydrogen_patch,
)
from protrepair.transformer.completion.hydrogen.domain import HydrogenResidueSite
from protrepair.transformer.context import ProteinTransformationContext


@dataclass(frozen=True, slots=True)
class RigidFrameHydrogenPlacementTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer for rigid-frame hydrogen placement."""

    site: HydrogenResidueSite
    semantics: IdealGeometryHydrogenSemantics

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one rigid-frame hydrogen placement may run."""

        return (
            self.site.is_selected_by(
                context.source_snapshot,
                context.atom_input,
            )
            and self.site.payload(context.source_snapshot) is not None
        )

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after rigid-frame hydrogen placement."""

        if not self.is_applicable(context):
            return context.source_snapshot

        residue = self.site.payload(context.source_snapshot)
        if residue is None:
            return context.source_snapshot

        patch = self.site.patch(context.source_snapshot)
        patch = generate_component_hydrogen_patch(
            residue=residue,
            patch=patch,
            semantics=self.semantics,
        )
        if patch is None:
            return context.source_snapshot

        return self.site.apply_patch(context.source_snapshot, patch)
