"""Structure geometry facets for the redesigned ProtRepair package."""

from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass

from protrepair.errors import ModelInvariantError
from protrepair.geometry import Vec3
from protrepair.structure.address_space import (
    StructureAddressSpaceKey,
    address_space_key_without_atoms,
    atom_count_in_address_space,
)
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.slots import AtomIndex, ResidueIndex


@dataclass(frozen=True, slots=True)
class AtomGeometry:
    """Geometry payload attached to one constitution-native atom site."""

    position: Vec3
    occupancy: float = 1.0
    b_factor: float | None = None
    altloc: str | None = None

    def __post_init__(self) -> None:
        altloc = self.altloc
        if altloc is not None:
            altloc = altloc.strip() or None

        object.__setattr__(self, "altloc", altloc)

    def distance_to(self, other: "AtomGeometry") -> float:
        """Return the Euclidean distance to another atom geometry payload."""

        return self.position.distance_to(other.position)

    def with_position(self, position: Vec3) -> "AtomGeometry":
        """Return a copy with updated coordinates."""

        return type(self)(
            position=Vec3.coerce(position),
            occupancy=self.occupancy,
            b_factor=self.b_factor,
            altloc=self.altloc,
        )


@dataclass(frozen=True, slots=True)
class ResidueGeometry:
    """Residue-local geometry keyed by atom name."""

    atoms_by_name: dict[str, AtomGeometry]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms_by_name", dict(self.atoms_by_name))

    def __hash__(self) -> int:
        """Return a stable hash over residue-local geometry payload."""

        return hash(
            (
                tuple(self.atoms_by_name.items()),
            )
        )

    def atom_names(self) -> tuple[str, ...]:
        """Return atom names in stored order."""

        return tuple(self.atoms_by_name)

    def has_atom(self, atom_name: str) -> bool:
        """Return whether geometry exists for one residue-local atom name."""

        return atom_name.strip().upper() in self.atoms_by_name

    def atom_geometry(self, atom_name: str) -> AtomGeometry:
        """Return geometry for one residue-local atom name."""

        return self.atoms_by_name[atom_name.strip().upper()]

    def position(self, atom_name: str) -> Vec3:
        """Return the position for one residue-local atom name."""

        return self.atom_geometry(atom_name).position

    def with_atom_geometry(
        self,
        atom_name: str,
        atom_geometry: AtomGeometry,
    ) -> "ResidueGeometry":
        """Return a copy with one atom geometry added or replaced by name."""

        updated_atoms_by_name = dict(self.atoms_by_name)
        updated_atoms_by_name[atom_name.strip().upper()] = atom_geometry
        return type(self)(
            atoms_by_name=updated_atoms_by_name,
        )

    def with_atom_geometries(
        self,
        atom_geometries: Iterable[tuple[str, AtomGeometry]],
    ) -> "ResidueGeometry":
        """Return a copy with multiple atom geometries added or replaced by name."""

        residue_geometry = self
        for atom_name, atom_geometry in atom_geometries:
            residue_geometry = residue_geometry.with_atom_geometry(
                atom_name,
                atom_geometry,
            )

        return residue_geometry

    def without_atoms(self, atom_names: Collection[str]) -> "ResidueGeometry":
        """Return a copy without the requested residue-local atom names."""

        names_to_remove = {atom_name.strip().upper() for atom_name in atom_names}
        return type(self)(
            atoms_by_name={
                atom_name: atom_geometry
                for atom_name, atom_geometry in self.atoms_by_name.items()
                if atom_name not in names_to_remove
            },
        )


@dataclass(frozen=True, slots=True, init=False)
class StructureGeometry:
    """Structure-level geometry aligned to constitution-owned atom slots."""

    atom_geometries: tuple[AtomGeometry, ...]
    _address_space_key: StructureAddressSpaceKey

    def __init__(
        self,
        *,
        constitution: StructureConstitution,
        atom_geometries: Iterable[AtomGeometry],
    ) -> None:
        structure_geometry = type(self)._from_address_space_key(
            atom_geometries=tuple(atom_geometries),
            address_space_key=constitution.address_space_key,
        )
        object.__setattr__(self, "atom_geometries", structure_geometry.atom_geometries)
        object.__setattr__(
            self,
            "_address_space_key",
            structure_geometry._address_space_key,
        )

    @classmethod
    def _from_address_space_key(
        cls,
        *,
        atom_geometries: tuple[AtomGeometry, ...],
        address_space_key: StructureAddressSpaceKey,
    ) -> "StructureGeometry":
        structure_geometry = object.__new__(cls)
        object.__setattr__(
            structure_geometry,
            "atom_geometries",
            tuple(atom_geometries),
        )
        object.__setattr__(
            structure_geometry,
            "_address_space_key",
            tuple(address_space_key),
        )
        if len(structure_geometry.atom_geometries) != atom_count_in_address_space(
            structure_geometry._address_space_key
        ):
            raise ModelInvariantError(
                "structure geometry atom slots must align with the constitution "
                "address space"
            )
        return structure_geometry

    def is_aligned_to(self, constitution: StructureConstitution) -> bool:
        """Return whether this geometry payload matches one constitution."""

        return self._address_space_key == constitution.address_space_key

    def __hash__(self) -> int:
        """Return a stable hash over structure-level geometry payload."""

        return hash((self.atom_geometries, self._address_space_key))

    def atom_count(self) -> int:
        """Return the number of stored atom geometries."""

        return len(self.atom_geometries)

    def atom_geometry(self, atom_index: AtomIndex) -> AtomGeometry:
        """Return geometry for the requested atom slot index."""

        return self.atom_geometries[atom_index.value]

    def residue_geometry(
        self,
        *,
        constitution: StructureConstitution,
        residue_index: ResidueIndex,
    ) -> ResidueGeometry:
        """Return residue-local geometry for one residue slot index."""

        if not self.is_aligned_to(constitution):
            raise ModelInvariantError(
                "structure geometry residue projection requires the matching "
                "constitution address space"
            )

        residue_site = constitution.residue_site_at(residue_index)
        return ResidueGeometry(
            atoms_by_name={
                atom_site.name: self.atom_geometry(atom_index)
                for atom_site, atom_index in zip(
                    residue_site.atom_sites,
                    constitution.atom_indices_for_residue_index(residue_index),
                    strict=True,
                )
            },
        )

    def position(self, atom_index: AtomIndex) -> Vec3:
        """Return the position for the requested atom slot index."""

        return self.atom_geometry(atom_index).position

    def iter_positions(self) -> Iterator[tuple[AtomIndex, Vec3]]:
        """Iterate over atom slot indices and positions in stored order."""

        for atom_index, atom_geometry in enumerate(self.atom_geometries):
            yield AtomIndex(atom_index), atom_geometry.position

    def with_atom_geometry(
        self,
        atom_index: AtomIndex,
        atom_geometry: AtomGeometry,
    ) -> "StructureGeometry":
        """Return a copy with one atom geometry replaced by slot index."""

        updated_atom_geometries = list(self.atom_geometries)
        updated_atom_geometries[atom_index.value] = atom_geometry
        return type(self)._from_address_space_key(
            atom_geometries=tuple(updated_atom_geometries),
            address_space_key=self._address_space_key,
        )

    def with_residue_geometries(
        self,
        *,
        constitution: StructureConstitution,
        residue_geometries_by_index: Mapping[ResidueIndex, ResidueGeometry],
    ) -> "StructureGeometry":
        """Return a copy with residue-local geometries replaced by slot index."""

        if not residue_geometries_by_index:
            return self

        if not self.is_aligned_to(constitution):
            raise ModelInvariantError(
                "structure geometry residue update requires the matching "
                "constitution address space"
            )

        updated_atom_geometries = list(self.atom_geometries)
        for residue_index, residue_geometry in residue_geometries_by_index.items():
            residue_site = constitution.residue_site_at(residue_index)
            expected_atom_names = set(residue_site.atom_site_names())
            updated_atom_names = set(residue_geometry.atom_names())
            if updated_atom_names != expected_atom_names:
                raise ModelInvariantError(
                    "residue geometry update must preserve atom names for "
                    f"{residue_site.residue_id.display_token()}"
                )

            for atom_site, atom_index in zip(
                residue_site.atom_sites,
                constitution.atom_indices_for_residue_index(residue_index),
                strict=True,
            ):
                updated_atom_geometries[atom_index.value] = (
                    residue_geometry.atom_geometry(atom_site.name)
                )

        return type(self)._from_address_space_key(
            atom_geometries=tuple(updated_atom_geometries),
            address_space_key=self._address_space_key,
        )

    def without_atoms(self, atom_indices: Collection[AtomIndex]) -> "StructureGeometry":
        """Return a copy without the requested atom slot indices."""

        removed_atom_index_values = {atom_index.value for atom_index in atom_indices}
        return type(self)._from_address_space_key(
            atom_geometries=tuple(
                atom_geometry
                for atom_index, atom_geometry in enumerate(self.atom_geometries)
                if atom_index not in removed_atom_index_values
            ),
            address_space_key=address_space_key_without_atoms(
                self._address_space_key,
                removed_atom_indices=set(atom_indices),
            ),
        )
