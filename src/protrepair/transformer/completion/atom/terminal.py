"""Terminal-atom placement for completion transforms."""

from dataclasses import dataclass

from protrepair.geometry import GeometryPlacementError
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.completion.atom.backbone import PeptideCarbonylFrame
from protrepair.transformer.completion.atom.outcome import GeometryPlacementOutcome
from protrepair.transformer.completion.shared.domain import ResidueCompletionSite
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.context import ProteinTransformationContext


@dataclass(frozen=True, slots=True)
class TerminalAtomPlacementTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer that places one terminal completion atom."""

    site: ResidueCompletionSite

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one terminal completion atom can be placed."""

        if not self.site.is_selected_by(
            context.source_snapshot,
            context.atom_input,
        ):
            return False

        payload = self.site.payload(context.source_snapshot)
        if payload is None:
            return False

        required_atom_names = ("N", "CA", "C", "O")
        return not payload.has_atom("OXT") and all(
            payload.has_atom(atom_name) for atom_name in required_atom_names
        )

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after terminal atom placement."""

        return self.placement_outcome(context).snapshot

    def placement_outcome(
        self,
        context: ProteinTransformationContext,
    ) -> GeometryPlacementOutcome:
        """Return the snapshot and OXT skip evidence from terminal placement."""

        payload = self.site.payload(context.source_snapshot)
        if payload is None:
            return GeometryPlacementOutcome(context.source_snapshot)

        residue_geometry = payload.residue_geometry
        try:
            oxt_position = PeptideCarbonylFrame(
                nitrogen=residue_geometry.position("N"),
                alpha_carbon=residue_geometry.position("CA"),
                carbonyl_carbon=residue_geometry.position("C"),
            ).terminal_oxygen(residue_geometry.position("O"))
        except GeometryPlacementError as error:
            return GeometryPlacementOutcome(
                snapshot=context.source_snapshot,
                skipped_atom_names=("OXT",),
                failure_reason=str(error),
            )

        patch = OrderedAtomPatch.from_atom_coordinates(
            atom_names=(*payload.atom_names(), "OXT"),
            atom_coordinates=(
                *tuple(
                    residue_geometry.position(atom_site.name)
                    for atom_site in payload.atom_sites
                ),
                oxt_position,
            ),
        )
        return GeometryPlacementOutcome(
            self.site.apply_patch(context.source_snapshot, patch)
        )
