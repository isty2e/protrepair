"""Shared value objects for residue-local discrete correction transformers."""

from dataclasses import dataclass

from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.discrete.orientation import attached_hydrogen_atom_names


@dataclass(frozen=True, order=True, slots=True)
class DiscreteCorrectionScore:
    """Lexicographic score for one residue-local discrete correction candidate."""

    focus_heavy_clash_count: int
    focus_geometry_outlier_count: int
    focus_heavy_fractional_clash_overlap_sum: float
    focus_clash_count: int
    focus_fractional_clash_overlap_sum: float
    focus_heavy_clash_overlap_sum_angstrom: float
    focus_clash_overlap_sum_angstrom: float


@dataclass(frozen=True, slots=True)
class DiscreteCorrectionCandidate:
    """One candidate structure for a residue-local discrete correction."""

    structure: ProteinStructure
    moved_atom_indices: tuple[AtomIndex, ...]
    score: DiscreteCorrectionScore


@dataclass(frozen=True, slots=True)
class AxisRotationCorrectionDefinition:
    """Residue-local axis-rotation metadata for one correctable component."""

    component_id: str
    axis_atom_names: tuple[str, str]
    rotating_atom_names: tuple[str, ...]
    hydrogen_anchor_atom_names: tuple[str, ...]

    def supports_residue(self, residue_site: ResidueSite) -> bool:
        """Return whether a residue contains this definition's required atoms."""

        return all(
            residue_site.has_atom_site(atom_name)
            for atom_name in (*self.axis_atom_names, *self.rotating_atom_names)
        )

    def rotated_atom_names(
        self,
        *,
        residue_site: ResidueSite,
        residue_geometry: ResidueGeometry,
    ) -> frozenset[str]:
        """Return atoms moved by this correction definition."""

        return frozenset(
            (
                *self.rotating_atom_names,
                *attached_hydrogen_atom_names(
                    residue_site=residue_site,
                    residue_geometry=residue_geometry,
                    anchor_atom_names=self.hydrogen_anchor_atom_names,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class BranchedSidechainRotationStep:
    """One torsional rotation step within a branched side-chain seed."""

    axis_atom_names: tuple[str, str]
    rotating_atom_names: tuple[str, ...]
    hydrogen_anchor_atom_names: tuple[str, ...]
    angle_degrees_options: tuple[int, ...]

    def required_atom_names(self) -> frozenset[str]:
        """Return atom names required for this rotation step to be executable."""

        return frozenset((*self.axis_atom_names, *self.rotating_atom_names))

    def rotated_atom_names(
        self,
        *,
        residue_site: ResidueSite,
        residue_geometry: ResidueGeometry,
    ) -> frozenset[str]:
        """Return atoms moved by this rotation step."""

        return frozenset(
            (
                *self.rotating_atom_names,
                *attached_hydrogen_atom_names(
                    residue_site=residue_site,
                    residue_geometry=residue_geometry,
                    anchor_atom_names=self.hydrogen_anchor_atom_names,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class BranchedSidechainCorrectionDefinition:
    """One branched side-chain seed family emitted before local refinement."""

    component_id: str
    rotation_steps: tuple[BranchedSidechainRotationStep, ...]

    def supports_residue(self, residue_site: ResidueSite) -> bool:
        """Return whether a residue contains every atom required by this seed."""

        return all(
            residue_site.has_atom_site(atom_name)
            for step in self.rotation_steps
            for atom_name in step.required_atom_names()
        )

    def required_rotated_atom_names(
        self,
        *,
        residue_site: ResidueSite,
        residue_geometry: ResidueGeometry,
    ) -> frozenset[str]:
        """Return all atoms that atomwise selection must authorize."""

        return frozenset(
            atom_name
            for step in self.rotation_steps
            for atom_name in step.rotated_atom_names(
                residue_site=residue_site,
                residue_geometry=residue_geometry,
            )
        )


@dataclass(frozen=True, slots=True)
class BranchedSidechainSeedProvenance:
    """Typed provenance for one branched side-chain seed candidate."""

    residue_id: ResidueId
    component_id: str
    angle_degrees_by_step: tuple[int, ...]
