"""Mutable geometry state for atom-completion kernels."""

from collections.abc import Sequence
from dataclasses import dataclass

from protrepair.geometry import Vec3
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch


@dataclass(slots=True)
class AtomGeometryState:
    """Mutable residue-local geometry used only while placing heavy atoms."""

    positions_by_atom_name: dict[str, Vec3]

    @classmethod
    def from_patch(cls, patch: OrderedAtomPatch) -> "AtomGeometryState":
        """Build mutable geometry from one ordered atom patch."""

        return cls(positions_by_atom_name=patch.position_map())

    @classmethod
    def from_payload(
        cls,
        residue: CompletionResiduePayload,
    ) -> "AtomGeometryState":
        """Build mutable geometry from one residue payload."""

        return cls.from_patch(
            OrderedAtomPatch.from_residue_payload(
                residue.residue_site,
                residue_geometry=residue.residue_geometry,
            )
        )

    def __contains__(self, atom_name: object) -> bool:
        """Return whether one atom name is present in this geometry."""

        if not isinstance(atom_name, str):
            return False

        return atom_name.strip().upper() in self.positions_by_atom_name

    def __getitem__(self, atom_name: str) -> list[float]:
        """Return one atom coordinate as a numeric vector."""

        return list(self.position(atom_name))

    def __setitem__(self, atom_name: str, coordinates: Vec3 | Sequence[float]) -> None:
        """Assign one atom coordinate from a canonical or numeric position."""

        self.assign(atom_name, coordinates)

    def get(
        self,
        atom_name: str,
        default: list[float] | None = None,
    ) -> list[float] | None:
        """Return one optional atom coordinate as a numeric vector."""

        normalized_atom_name = atom_name.strip().upper()
        position = self.positions_by_atom_name.get(normalized_atom_name)
        if position is None:
            return default

        return list(position)

    def position(self, atom_name: str) -> Vec3:
        """Return one canonical position for the requested atom name."""

        normalized_atom_name = atom_name.strip().upper()
        return self.positions_by_atom_name[normalized_atom_name]

    def assign(self, atom_name: str, coordinates: Vec3 | Sequence[float]) -> None:
        """Assign one atom coordinate within this mutable geometry."""

        normalized_atom_name = atom_name.strip().upper()
        position = (
            coordinates
            if isinstance(coordinates, Vec3)
            else Vec3.from_iterable(coordinates)
        )
        self.positions_by_atom_name[normalized_atom_name] = position

    def to_patch(self, atom_order: Sequence[str]) -> OrderedAtomPatch:
        """Project this mutable geometry back into deterministic atom order."""

        return OrderedAtomPatch.from_atom_coordinates(
            atom_names=tuple(atom_order),
            atom_coordinates=tuple(
                self.positions_by_atom_name[atom_name] for atom_name in atom_order
            ),
        )
