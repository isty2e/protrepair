"""Canonical chemistry graph entities and local atom-typing facts."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class TetrahedralCenterSemantics:
    """Expected handedness for one tetrahedral side-chain center."""

    center_atom_name: str
    ordered_neighbor_atom_names: tuple[str, str, str]
    expected_orientation_sign: int

    def __post_init__(self) -> None:
        center_atom_name = self.center_atom_name.strip().upper()
        ordered_neighbor_atom_names = tuple(
            atom_name.strip().upper() for atom_name in self.ordered_neighbor_atom_names
        )
        if not center_atom_name:
            raise ValueError("tetrahedral center atom name must not be blank")

        if len(ordered_neighbor_atom_names) != 3:
            raise ValueError("tetrahedral centers require exactly three neighbors")

        if len(set(ordered_neighbor_atom_names)) != 3:
            raise ValueError("tetrahedral center neighbors must be unique")

        if any(not atom_name for atom_name in ordered_neighbor_atom_names):
            raise ValueError("tetrahedral center neighbors must not be blank")

        if self.expected_orientation_sign not in (-1, 1):
            raise ValueError("tetrahedral center expected sign must be -1 or 1")

        object.__setattr__(self, "center_atom_name", center_atom_name)
        object.__setattr__(
            self,
            "ordered_neighbor_atom_names",
            ordered_neighbor_atom_names,
        )

    def matches(
        self,
        *,
        center_atom_name: str,
        ordered_neighbor_atom_names: tuple[str, str, str],
    ) -> bool:
        """Return whether this center matches one canonical observed tuple."""

        normalized_neighbor_atom_names = tuple(
            atom_name.strip().upper() for atom_name in ordered_neighbor_atom_names
        )
        return self.center_atom_name == center_atom_name.strip().upper() and (
            self.ordered_neighbor_atom_names == normalized_neighbor_atom_names
        )


@dataclass(frozen=True, slots=True)
class BondDefinition:
    """Bond relationship between two atoms in a component definition."""

    atom_name_1: str
    atom_name_2: str
    order: int = 1
    aromatic: bool = False

    def __post_init__(self) -> None:
        atom_name_1 = self.atom_name_1.strip().upper()
        atom_name_2 = self.atom_name_2.strip().upper()
        if not atom_name_1 or not atom_name_2:
            raise ValueError("bond atom names must not be blank")

        if self.order <= 0:
            raise ValueError("bond order must be positive")

        object.__setattr__(self, "atom_name_1", atom_name_1)
        object.__setattr__(self, "atom_name_2", atom_name_2)


@dataclass(frozen=True, slots=True)
class ForceFieldAtomParams:
    """Per-atom force-field parameters used by residue templates."""

    charge: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True, slots=True)
class ChemicalComponentDefinition:
    """Chemistry graph for one residue- or ligand-like component."""

    component_id: str
    atom_names: tuple[str, ...]
    bonds: tuple[BondDefinition, ...] = ()
    formal_charges: Mapping[str, int] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    _bonded_atom_names_by_atom_name: Mapping[str, frozenset[str]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _bond_hop_distances_by_atom_name_pair: Mapping[tuple[str, str], int] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        atom_names = tuple(atom_name.strip().upper() for atom_name in self.atom_names)
        aliases = tuple(alias.strip().upper() for alias in self.aliases)

        if not component_id:
            raise ValueError("component_id must not be blank")

        if not atom_names:
            raise ValueError("component definitions must contain at least one atom")

        if len(atom_names) != len(set(atom_names)):
            raise ValueError("component atom names must be unique")

        bonds = tuple(self.bonds)
        charges = {
            atom_name.strip().upper(): int(charge)
            for atom_name, charge in self.formal_charges.items()
        }
        atom_name_set = set(atom_names)
        for bond in bonds:
            if (
                bond.atom_name_1 not in atom_name_set
                or bond.atom_name_2 not in atom_name_set
            ):
                raise ValueError(
                    "component bond definitions must reference known atom names"
                )

        bonded_atom_names_by_atom_name = build_bonded_atom_names_by_atom_name(
            atom_names,
            bonds,
        )
        bond_hop_distances_by_atom_name_pair = (
            build_bond_hop_distances_by_atom_name_pair(
                bonded_atom_names_by_atom_name
            )
        )

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "atom_names", atom_names)
        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(
            self,
            "formal_charges",
            MappingProxyType(charges),
        )
        object.__setattr__(
            self,
            "_bonded_atom_names_by_atom_name",
            MappingProxyType(bonded_atom_names_by_atom_name),
        )
        object.__setattr__(
            self,
            "_bond_hop_distances_by_atom_name_pair",
            MappingProxyType(bond_hop_distances_by_atom_name_pair),
        )

    def expected_atom_names(self) -> tuple[str, ...]:
        """Return the canonical atom order for the component."""

        return self.atom_names

    def has_atom(self, atom_name: str) -> bool:
        """Return whether the definition contains a named atom."""

        return atom_name.strip().upper() in self.atom_names

    def bonded_atom_names(self, atom_name: str) -> frozenset[str]:
        """Return directly bonded neighbors for one atom name."""

        normalized_atom_name = atom_name.strip().upper()
        return self._bonded_atom_names_by_atom_name.get(
            normalized_atom_name,
            frozenset(),
        )

    def bond_hop_distance(
        self,
        atom_name_1: str,
        atom_name_2: str,
        *,
        max_hops: int | None = None,
    ) -> int | None:
        """Return the bond-graph hop distance between two atoms if connected."""

        normalized_atom_name_1 = atom_name_1.strip().upper()
        normalized_atom_name_2 = atom_name_2.strip().upper()
        if normalized_atom_name_1 == normalized_atom_name_2:
            return 0

        if normalized_atom_name_1 not in self.atom_names:
            return None

        if normalized_atom_name_2 not in self.atom_names:
            return None

        hop_distance = self._bond_hop_distances_by_atom_name_pair.get(
            (normalized_atom_name_1, normalized_atom_name_2)
        )
        if hop_distance is None:
            return None
        if max_hops is not None and hop_distance > max_hops:
            return None

        return hop_distance

    def all_component_ids(self) -> tuple[str, ...]:
        """Return the canonical identifier plus all aliases."""

        return (self.component_id, *self.aliases)


def build_bonded_atom_names_by_atom_name(
    atom_names: tuple[str, ...],
    bonds: tuple[BondDefinition, ...],
) -> dict[str, frozenset[str]]:
    """Return direct bond adjacency keyed by canonical atom name."""

    bonded_atom_names_by_atom_name: dict[str, set[str]] = {
        atom_name: set() for atom_name in atom_names
    }
    for bond in bonds:
        bonded_atom_names_by_atom_name[bond.atom_name_1].add(bond.atom_name_2)
        bonded_atom_names_by_atom_name[bond.atom_name_2].add(bond.atom_name_1)

    return {
        atom_name: frozenset(neighbor_atom_names)
        for atom_name, neighbor_atom_names in bonded_atom_names_by_atom_name.items()
    }


def build_bond_hop_distances_by_atom_name_pair(
    bonded_atom_names_by_atom_name: Mapping[str, frozenset[str]],
) -> dict[tuple[str, str], int]:
    """Return all-pairs shortest bond hop distances for one component graph."""

    bond_hop_distances_by_atom_name_pair: dict[tuple[str, str], int] = {}
    for origin_atom_name in bonded_atom_names_by_atom_name:
        queue: deque[tuple[str, int]] = deque([(origin_atom_name, 0)])
        visited = {origin_atom_name}
        while queue:
            current_atom_name, current_hops = queue.popleft()
            bond_hop_distances_by_atom_name_pair[
                (origin_atom_name, current_atom_name)
            ] = current_hops
            for neighbor_atom_name in bonded_atom_names_by_atom_name[
                current_atom_name
            ]:
                if neighbor_atom_name in visited:
                    continue

                visited.add(neighbor_atom_name)
                queue.append((neighbor_atom_name, current_hops + 1))

    return bond_hop_distances_by_atom_name_pair
