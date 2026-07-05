"""Internal-coordinate atom placement for completion transforms."""

from collections.abc import Collection
from dataclasses import dataclass

from protrepair.chemistry import HeavyAtomSemantics
from protrepair.chemistry.internal_coordinates import (
    InternalCoordinateExecutionContext,
    InternalCoordinatePlacement,
)
from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3
from protrepair.structure.peptide import are_peptide_adjacent
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.completion.atom.backbone import (
    PeptideCarbonylFrame,
    backbone_psi_degrees,
)
from protrepair.transformer.completion.atom.geometry_state import AtomGeometryState
from protrepair.transformer.completion.shared.domain import (
    CompletionResiduePayload,
    ResidueCompletionSite,
)
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.context import ProteinTransformationContext

LOCAL_BACKBONE_OXYGEN_FALLBACK_PSI_DEGREES = 0.0


@dataclass(slots=True)
class InternalCoordinateEnvironment(InternalCoordinateExecutionContext):
    """Mutable execution environment for one residue-local program."""

    residue: CompletionResiduePayload
    next_peptide_residue: CompletionResiduePayload | None
    missing_atom_names: frozenset[str]
    geometry: AtomGeometryState

    @classmethod
    def from_payloads(
        cls,
        *,
        residue: CompletionResiduePayload,
        next_peptide_residue: CompletionResiduePayload | None,
        missing_atom_names: tuple[str, ...],
    ) -> "InternalCoordinateEnvironment":
        """Build one environment from neighboring residues and ingress targets."""

        return cls(
            residue=residue,
            next_peptide_residue=next_peptide_residue,
            missing_atom_names=frozenset(
                atom_name.strip().upper() for atom_name in missing_atom_names
            ),
            geometry=AtomGeometryState.from_payload(residue),
        )

    def has_atom(self, atom_name: str) -> bool:
        """Return whether one atom is currently present in the working geometry."""

        return atom_name.strip().upper() in self.geometry

    def missing_any(self, atom_names: Collection[str]) -> bool:
        """Return whether any trigger atom was missing when execution began."""

        return any(
            atom_name.strip().upper() in self.missing_atom_names
            for atom_name in atom_names
        )

    def torsion(self, atom_names: tuple[str, str, str, str]) -> float:
        """Return the current torsion angle for four named atoms."""

        return InternalCoordinateFrame.torsion(
            *(self.geometry.position(atom_name) for atom_name in atom_names)
        )

    def ensure_backbone_oxygen(self) -> None:
        """Ensure the current residue backbone oxygen exists."""

        if self.has_atom("O"):
            return
        if not all(self.has_atom(atom_name) for atom_name in ("N", "CA", "C")):
            return

        next_nitrogen = self._next_peptide_nitrogen_position()
        psi_degrees = (
            LOCAL_BACKBONE_OXYGEN_FALLBACK_PSI_DEGREES
            if next_nitrogen is None
            else self._psi_degrees(next_nitrogen)
        )
        clash_reference = (
            self.geometry.position("N")
            if next_nitrogen is None
            else next_nitrogen
        )

        position = PeptideCarbonylFrame(
            nitrogen=self.geometry.position("N"),
            alpha_carbon=self.geometry.position("CA"),
            carbonyl_carbon=self.geometry.position("C"),
        ).backbone_oxygen(
            psi_degrees=psi_degrees,
            clash_reference=clash_reference,
        )
        self.geometry.assign("O", position)

    def ensure_beta_carbon(self, *, dihedral_degrees: float) -> None:
        """Ensure the current residue beta carbon exists."""

        if self.has_atom("CB"):
            return

        position = InternalCoordinateFrame(
            self.geometry.position("N"),
            self.geometry.position("C"),
            self.geometry.position("CA"),
        ).place(
            bond_length=1.52,
            bond_angle_degrees=109.5,
            dihedral_degrees=dihedral_degrees,
        )
        self.geometry.assign("CB", position)

    def place(self, placement: InternalCoordinatePlacement) -> None:
        """Place one atom from one canonical internal-coordinate instruction."""

        dihedral_degrees = placement.dihedral.resolve(self)
        position = InternalCoordinateFrame(
            self.geometry.position(placement.anchor_atom_names[0]),
            self.geometry.position(placement.anchor_atom_names[1]),
            self.geometry.position(placement.anchor_atom_names[2]),
        ).place(
            bond_length=placement.bond_length,
            bond_angle_degrees=placement.bond_angle,
            dihedral_degrees=dihedral_degrees,
        )
        self.geometry.assign(placement.target_atom_name, position)

    def build_patch(self, atom_order: tuple[str, ...]) -> OrderedAtomPatch:
        """Project the current working geometry back into deterministic atom order."""

        return self.geometry.to_patch(atom_order)

    def build_successful_partial_patch(
        self,
        atom_order: tuple[str, ...],
    ) -> OrderedAtomPatch | None:
        """Project original atoms plus successfully placed missing atoms."""

        original_atom_names = self.residue.atom_names()
        original_atom_name_set = frozenset(original_atom_names)
        added_atom_names = tuple(
            atom_name
            for atom_name in atom_order
            if atom_name in self.missing_atom_names
            and atom_name not in original_atom_name_set
            and self.has_atom(atom_name)
        )
        if not added_atom_names:
            return None

        return self.geometry.to_patch((*original_atom_names, *added_atom_names))

    def _next_peptide_nitrogen_position(self) -> Vec3 | None:
        """Return the next peptide nitrogen position when available."""

        if (
            self.next_peptide_residue is None
            or not self.next_peptide_residue.has_atom_site("N")
        ):
            return None

        return self.next_peptide_residue.position("N")

    def _psi_degrees(self, next_nitrogen: Vec3) -> float:
        """Return the backbone psi torsion used by oxygen placement."""

        return backbone_psi_degrees(
            (
                self.geometry.position("N"),
                self.geometry.position("CA"),
                self.geometry.position("C"),
                next_nitrogen,
            )
        )


@dataclass(frozen=True, slots=True)
class InternalCoordinatePlacementTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer that places missing atoms by internal coordinates."""

    site: ResidueCompletionSite

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one residue-local internal-coordinate patch can be built."""

        if not self.site.is_selected_by(
            context.source_snapshot,
            context.atom_input,
        ):
            return False

        semantics = self.site.template.heavy_atom_semantics
        if not isinstance(semantics, HeavyAtomSemantics):
            return False

        residue = self.site.payload(context.source_snapshot)
        missing_atom_names = self.site.missing_atom_names(context.source_snapshot)
        return (
            residue is not None
            and _has_required_backbone_oxygen_context(
                residue,
                missing_atom_names=missing_atom_names,
            )
            and bool(missing_atom_names)
            and bool(semantics.atom_order)
        )

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after internal-coordinate atom placement."""

        semantics = self.site.template.heavy_atom_semantics
        residue = self.site.payload(context.source_snapshot)
        _previous_residue, next_residue = self.site.neighbor_payloads(
            context.source_snapshot
        )
        missing_atom_names = self.site.missing_atom_names(context.source_snapshot)
        if (
            not isinstance(semantics, HeavyAtomSemantics)
            or residue is None
            or not _has_required_backbone_oxygen_context(
                residue,
                missing_atom_names=missing_atom_names,
            )
        ):
            return context.source_snapshot

        environment = InternalCoordinateEnvironment.from_payloads(
            residue=residue,
            next_peptide_residue=_next_peptide_residue(
                residue,
                next_residue,
                snapshot=context.source_snapshot,
            ),
            missing_atom_names=missing_atom_names,
        )
        try:
            semantics.program.apply(environment)
        except GeometryPlacementError:
            partial_patch = environment.build_successful_partial_patch(
                semantics.atom_order
            )
            if partial_patch is None:
                return context.source_snapshot

            return self.site.apply_patch(context.source_snapshot, partial_patch)

        patch = environment.build_patch(semantics.atom_order)
        return self.site.apply_patch(context.source_snapshot, patch)


def _has_required_backbone_oxygen_context(
    residue: CompletionResiduePayload,
    *,
    missing_atom_names: tuple[str, ...],
) -> bool:
    """Return whether missing O can be placed without inventing peptide context."""

    if "O" not in {atom_name.strip().upper() for atom_name in missing_atom_names}:
        return True

    return (
        residue.has_atom_site("N")
        and residue.has_atom_site("CA")
        and residue.has_atom_site("C")
    )


def _next_peptide_residue(
    residue: CompletionResiduePayload,
    next_residue: CompletionResiduePayload | None,
    *,
    snapshot: ProteinStructureSnapshot,
) -> CompletionResiduePayload | None:
    """Return the next residue only when canonical peptide adjacency allows it."""

    if next_residue is None:
        return None

    if are_peptide_adjacent(
        residue.residue_site,
        next_residue.residue_site,
        structure=snapshot.structure,
    ):
        return next_residue

    return None
