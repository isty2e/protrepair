"""Canonical structure snapshots."""

from dataclasses import dataclass

from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.slots import AtomIndex


@dataclass(frozen=True, slots=True)
class ProteinStructureSnapshot:
    """Canonical truth-bearing snapshot for one protein structure."""

    structure: ProteinStructure

    @classmethod
    def from_structure(cls, structure: ProteinStructure) -> "ProteinStructureSnapshot":
        """Build one snapshot from a canonical structure."""

        return cls(structure=structure)

    def with_structure(
        self,
        structure: ProteinStructure,
    ) -> "ProteinStructureSnapshot":
        """Return a re-derived snapshot over one updated structure."""

        return type(self).from_structure(structure)

    def with_updated_residue_facets(
        self,
        residue_site: ResidueSite,
        *,
        residue_geometry: ResidueGeometry,
        formal_charge_by_atom_name: tuple[tuple[str, int | None], ...] = (),
    ) -> "ProteinStructureSnapshot":
        """Return a re-derived snapshot with one residue payload replaced."""

        return self.with_structure(
            self.structure.with_updated_residue_facets(
                residue_site,
                residue_geometry=residue_geometry,
                formal_charge_by_atom_name=formal_charge_by_atom_name,
            )
        )

    def moved_atom_indices_to(
        self,
        after_snapshot: "ProteinStructureSnapshot",
        atom_indices: tuple[AtomIndex, ...],
    ) -> tuple[AtomIndex, ...]:
        """Return requested atom slots whose coordinates changed."""

        moved_atom_indices: list[AtomIndex] = []
        for atom_index in atom_indices:
            if (
                atom_index.value >= self.structure.geometry.atom_count()
                or atom_index.value >= after_snapshot.structure.geometry.atom_count()
            ):
                continue

            before_atom_geometry = self.structure.geometry.atom_geometry(atom_index)
            after_atom_geometry = after_snapshot.structure.geometry.atom_geometry(
                atom_index
            )
            if before_atom_geometry.position != after_atom_geometry.position:
                moved_atom_indices.append(atom_index)

        return tuple(moved_atom_indices)
