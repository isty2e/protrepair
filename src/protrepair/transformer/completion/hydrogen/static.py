"""Static hydrogen-placement transformers over residue-local hydrogen sites."""

from dataclasses import dataclass

from protrepair.chemistry import HydrogenSemantics
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.completion.hydrogen.domain import HydrogenResidueSite
from protrepair.transformer.completion.hydrogen.static_patch import (
    generate_hydrogen_patch,
    histidine_delta_hydrogen,
    n_terminal_hydrogen_coordinates,
)
from protrepair.transformer.context import ProteinTransformationContext


@dataclass(frozen=True, slots=True)
class StaticHydrogenPlacementTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer for template-static hydrogen placement."""

    site: HydrogenResidueSite
    semantics: HydrogenSemantics

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one static hydrogen placement may run."""

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
        """Return one snapshot after template-static hydrogen placement."""

        if not self.is_applicable(context):
            return context.source_snapshot

        patch = self.site.patch(context.source_snapshot)
        patch = generate_hydrogen_patch(
            site=self.site,
            patch=patch,
            semantics=self.semantics,
        )
        return self.site.apply_patch(context.source_snapshot, patch)


@dataclass(frozen=True, slots=True)
class HistidineDeltaHydrogenTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer for residue-local histidine delta protonation."""

    site: HydrogenResidueSite

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one histidine delta hydrogen can be appended."""

        residue = self.site.payload(context.source_snapshot)
        return (
            self.site.is_selected_by(
                context.source_snapshot,
                context.atom_input,
            )
            and residue is not None
            and residue.component_id == "HIS"
            and not residue.has_atom("HD1")
        )

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after histidine delta hydrogen placement."""

        if not self.is_applicable(context):
            return context.source_snapshot

        patch = self.site.patch(context.source_snapshot)
        patch = patch.append_atoms(
            ("HD1",),
            (histidine_delta_hydrogen(patch),),
        )
        return self.site.apply_patch(context.source_snapshot, patch)


@dataclass(frozen=True, slots=True)
class NTerminalHydrogenPlacementTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer for N-terminal protonation."""

    site: HydrogenResidueSite

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether this site can receive N-terminal hydrogens."""

        residue = self.site.payload(context.source_snapshot)
        return (
            self.site.is_selected_by(
                context.source_snapshot,
                context.atom_input,
            )
            and residue is not None
            and self.site.backbone_family_component_id is not None
        )

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after N-terminal hydrogen placement."""

        if not self.is_applicable(context):
            return context.source_snapshot

        backbone_family_component_id = self.site.backbone_family_component_id
        assert backbone_family_component_id is not None
        patch = self.site.patch(context.source_snapshot)
        atom_coordinates = tuple(
            n_terminal_hydrogen_coordinates(
                patch,
                backbone_family_component_id,
            )
        )
        atom_names = (
            ("H1", "H2")
            if backbone_family_component_id == "PRO"
            else ("H1", "H2", "H3")
        )
        return self.site.apply_patch(
            context.source_snapshot,
            patch.append_atoms(atom_names, atom_coordinates),
        )
