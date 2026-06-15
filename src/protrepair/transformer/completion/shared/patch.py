"""Patch materialization primitives for completion transforms."""

from collections.abc import Sequence
from dataclasses import dataclass

from protrepair.geometry import Vec3
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot


@dataclass(frozen=True, slots=True)
class AtomPlacement:
    """One named atom position inside one completion patch."""

    atom_name: str
    position: Vec3
    element: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_name", self.atom_name.strip().upper())
        if self.element is not None:
            object.__setattr__(self, "element", self.element.strip().upper())


@dataclass(frozen=True, slots=True)
class OrderedAtomPatch:
    """Ordered atom placements over one residue-local completion patch."""

    placements: tuple[AtomPlacement, ...]

    @classmethod
    def from_residue_payload(
        cls,
        residue_site: ResidueSite,
        *,
        residue_geometry: ResidueGeometry,
    ) -> "OrderedAtomPatch":
        """Project one canonical residue payload into one ordered completion patch."""

        return cls(
            placements=tuple(
                AtomPlacement(
                    atom_name=atom_site.name,
                    position=residue_geometry.position(atom_site.name),
                    element=atom_site.element,
                )
                for atom_site in residue_site.atom_sites
            )
        )

    @classmethod
    def from_atom_coordinates(
        cls,
        *,
        atom_names: list[str] | tuple[str, ...],
        atom_coordinates: Sequence[Vec3 | Sequence[float]],
    ) -> "OrderedAtomPatch":
        """Build one ordered patch from atom names and raw coordinates."""

        return cls(
            placements=tuple(
                AtomPlacement(
                    atom_name=atom_name,
                    position=Vec3.from_iterable(coordinates),
                )
                for atom_name, coordinates in zip(
                    atom_names,
                    atom_coordinates,
                    strict=True,
                )
            )
        )

    @property
    def atom_names(self) -> tuple[str, ...]:
        """Return ordered atom names."""

        return tuple(placement.atom_name for placement in self.placements)

    @property
    def atom_coordinates(self) -> tuple[list[float], ...]:
        """Return ordered coordinates as plain numeric vectors for legacy kernels."""

        return tuple(
            [placement.position.x, placement.position.y, placement.position.z]
            for placement in self.placements
        )

    def position_map(self) -> dict[str, Vec3]:
        """Return one mutable mapping from atom name to canonical positions."""

        return {
            placement.atom_name: placement.position for placement in self.placements
        }

    def coordinate_map(self) -> dict[str, Vec3]:
        """Return one read-only canonical coordinate mapping for this patch."""

        return self.position_map()

    def position(self, atom_name: str) -> Vec3:
        """Return one canonical position for the requested atom name."""

        normalized_atom_name = atom_name.strip().upper()
        for placement in self.placements:
            if placement.atom_name == normalized_atom_name:
                return placement.position

        raise KeyError(normalized_atom_name)

    def append_atoms(
        self,
        atom_names: Sequence[str],
        atom_coordinates: Sequence[Vec3 | Sequence[float]],
    ) -> "OrderedAtomPatch":
        """Return one ordered patch with appended atoms in the given order."""

        return type(self).from_atom_coordinates(
            atom_names=[*self.atom_names, *atom_names],
            atom_coordinates=[*self.atom_coordinates, *atom_coordinates],
        )

    def materialize_on_payload(
        self,
        residue_site: ResidueSite,
        *,
        residue_geometry: ResidueGeometry,
        formal_charge_by_atom_name: tuple[tuple[str, int | None], ...] = (),
    ) -> tuple[ResidueSite, ResidueGeometry, tuple[tuple[str, int | None], ...]]:
        """Project this ordered patch back into residue facet payload."""

        original_atom_sites = {
            atom_site.name: atom_site for atom_site in residue_site.atom_sites
        }
        updated_residue_site = residue_site
        updated_residue_geometry = residue_geometry
        updated_formal_charge_by_name = dict(formal_charge_by_atom_name)
        default_atom_site = next(iter(original_atom_sites.values()))
        default_atom_geometry = residue_geometry.atom_geometry(default_atom_site.name)
        default_formal_charge = updated_formal_charge_by_name.get(
            default_atom_site.name
        )
        for placement in self.placements:
            existing_atom_site = original_atom_sites.get(placement.atom_name)
            if existing_atom_site is None:
                updated_residue_site = updated_residue_site.with_atom_site(
                    AtomSite(
                        name=placement.atom_name,
                        element=placement.element or self._infer_element(
                            placement.atom_name
                        ),
                    )
                )
                updated_residue_geometry = updated_residue_geometry.with_atom_geometry(
                    placement.atom_name,
                    AtomGeometry(
                        position=placement.position,
                        occupancy=default_atom_geometry.occupancy,
                        b_factor=default_atom_geometry.b_factor,
                        altloc=default_atom_geometry.altloc,
                    ),
                )
                if default_formal_charge is not None:
                    updated_formal_charge_by_name[placement.atom_name] = (
                        default_formal_charge
                    )
                continue

            updated_residue_geometry = updated_residue_geometry.with_atom_geometry(
                placement.atom_name,
                residue_geometry.atom_geometry(placement.atom_name).with_position(
                    placement.position
                ),
            )

        return (
            updated_residue_site,
            updated_residue_geometry,
            tuple(
                (atom_name, updated_formal_charge_by_name[atom_name])
                for atom_name in updated_residue_site.atom_site_names()
                if atom_name in updated_formal_charge_by_name
            ),
        )

    def apply_to_snapshot(
        self,
        snapshot: ProteinStructureSnapshot,
        residue_index: ResidueIndex,
    ) -> ProteinStructureSnapshot:
        """Materialize this ordered patch onto the given snapshot residue."""

        residue_site = snapshot.structure.constitution.residue_site_at(residue_index)
        residue_geometry = snapshot.structure.residue_geometry(residue_index)

        (
            updated_residue_site,
            updated_residue_geometry,
            updated_formal_charge_by_name,
        ) = self.materialize_on_payload(
            residue_site,
            residue_geometry=residue_geometry,
            formal_charge_by_atom_name=snapshot.structure
            .residue_formal_charge_by_atom_name(residue_index),
        )
        return snapshot.with_updated_residue_facets(
            updated_residue_site,
            residue_geometry=updated_residue_geometry,
            formal_charge_by_atom_name=updated_formal_charge_by_name,
        )

    @staticmethod
    def _infer_element(atom_name: str) -> str:
        """Infer the element symbol for a repaired atom."""

        letters = "".join(character for character in atom_name if character.isalpha())
        if not letters:
            raise ValueError(f"cannot infer element for atom name {atom_name!r}")

        return letters[0]


def _infer_element(atom_name: str) -> str:
    """Infer the element symbol for a repaired atom."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"cannot infer element for atom name {atom_name!r}")

    return letters[0]
