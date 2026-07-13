"""Spatial candidate-pair generation for clash diagnostics."""

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from typing_extensions import Protocol

from protrepair.structure.labels import ResidueId

NEIGHBOR_CELL_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)


class ContactDomain(str, Enum):
    """Closed contact-domain classifications used by clash pair policies."""

    POLYMER = "polymer"
    RETAINED_NON_POLYMER = "retained_non_polymer"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"

    @classmethod
    def normalize(cls, value: "ContactDomain | str") -> "ContactDomain":
        """Return one canonical contact domain from a diagnostic ingress value."""

        if isinstance(value, ContactDomain):
            return value
        if not isinstance(value, str):
            raise TypeError("contact domains require a ContactDomain or string value")

        normalized_value = value.strip().lower().replace("-", "_")
        if normalized_value in {"ligand", "retained_non_polymer", "non_polymer"}:
            return cls.RETAINED_NON_POLYMER
        if normalized_value == "polymer":
            return cls.POLYMER
        if normalized_value == "unknown":
            return cls.UNKNOWN
        if normalized_value in {"not_applicable", "not_applicable_chemistry", "none"}:
            return cls.NOT_APPLICABLE

        raise ValueError(f"unsupported contact domain: {value!r}")

    def excluded_when_ligands_are_disabled(self) -> bool:
        """Return whether ligand-disabled policies should skip this domain."""

        return self is ContactDomain.RETAINED_NON_POLYMER


@dataclass(frozen=True, slots=True)
class ContactPairPolicy:
    """Metric-neutral atom scope and bonded-neighbor exclusions."""

    include_hydrogens: bool = True
    include_hydrogen_hydrogen: bool = False
    include_ligands: bool = False
    ignore_same_residue_bond_hops: int = 2
    ignore_adjacent_polymer_bond_hops: int = 3

    def __post_init__(self) -> None:
        if self.ignore_same_residue_bond_hops < 0:
            raise ValueError("ignore_same_residue_bond_hops must be non-negative")
        if self.ignore_adjacent_polymer_bond_hops < 0:
            raise ValueError(
                "ignore_adjacent_polymer_bond_hops must be non-negative"
            )

    def as_contact_pair_policy(self) -> "ContactPairPolicy":
        """Return the canonical metric-neutral projection of this policy."""

        if type(self) is ContactPairPolicy:
            return self
        return ContactPairPolicy(
            include_hydrogens=self.include_hydrogens,
            include_hydrogen_hydrogen=self.include_hydrogen_hydrogen,
            include_ligands=self.include_ligands,
            ignore_same_residue_bond_hops=self.ignore_same_residue_bond_hops,
            ignore_adjacent_polymer_bond_hops=(
                self.ignore_adjacent_polymer_bond_hops
            ),
        )


class ClashPairAtomSite(Protocol):
    """Atom-site surface required by spatial pair generation."""

    @property
    def residue_id(self) -> ResidueId:
        """Return residue id."""

        ...

    @property
    def domain(self) -> ContactDomain:
        """Return contact domain."""

        ...

    @property
    def grid_cell(self) -> tuple[int, int, int]:
        """Return spatial grid cell."""

        ...

    @property
    def is_hydrogen_atom(self) -> bool:
        """Return whether the site is hydrogen."""

        ...


class SpatialPairPolicy(Protocol):
    """Policy surface required by pre-distance pair rejection."""

    @property
    def include_ligands(self) -> bool:
        """Return whether ligand pairs are included."""

        ...

    @property
    def include_hydrogen_hydrogen(self) -> bool:
        """Return whether hydrogen-hydrogen pairs are included."""

        ...


@dataclass(frozen=True, slots=True)
class PreparedAtomSitePairIndex:
    """Immutable spatial index for one atom-site tuple and optional focus."""

    atom_sites: tuple[ClashPairAtomSite, ...]
    focus_residue_ids: frozenset[ResidueId] | None = None
    _cell_to_indexed_atom_sites: Mapping[
        tuple[int, int, int],
        tuple[tuple[int, ClashPairAtomSite], ...],
    ] = field(init=False, repr=False, compare=False)
    _focused_indexed_atom_sites: tuple[tuple[int, ClashPairAtomSite], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        atom_sites = tuple(self.atom_sites)
        focus_residue_ids = (
            None
            if self.focus_residue_ids is None
            else frozenset(self.focus_residue_ids)
        )
        cell_to_indexed_atom_sites: dict[
            tuple[int, int, int],
            list[tuple[int, ClashPairAtomSite]],
        ] = defaultdict(list)
        focused_indexed_atom_sites: list[tuple[int, ClashPairAtomSite]] = []
        for atom_index, atom_site in enumerate(atom_sites):
            indexed_atom_site = (atom_index, atom_site)
            cell_to_indexed_atom_sites[atom_site.grid_cell].append(indexed_atom_site)
            if (
                focus_residue_ids is not None
                and atom_site.residue_id in focus_residue_ids
            ):
                focused_indexed_atom_sites.append(indexed_atom_site)

        object.__setattr__(self, "atom_sites", atom_sites)
        object.__setattr__(self, "focus_residue_ids", focus_residue_ids)
        object.__setattr__(
            self,
            "_cell_to_indexed_atom_sites",
            MappingProxyType(
                {
                    cell: tuple(indexed_atom_sites)
                    for cell, indexed_atom_sites in cell_to_indexed_atom_sites.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "_focused_indexed_atom_sites",
            tuple(focused_indexed_atom_sites),
        )

    def require_compatible(
        self,
        atom_sites: tuple[ClashPairAtomSite, ...],
        *,
        focus_residue_ids: frozenset[ResidueId] | None,
    ) -> None:
        """Reject reuse across another coordinate frame or focus."""

        if atom_sites is not self.atom_sites:
            raise ValueError(
                "prepared atom-site pair index requires its original atom-site frame"
            )
        normalized_focus_residue_ids = (
            None if focus_residue_ids is None else frozenset(focus_residue_ids)
        )
        if normalized_focus_residue_ids != self.focus_residue_ids:
            raise ValueError("prepared atom-site pair index requires a matching focus")

    def candidate_pairs(
        self,
        *,
        policy: SpatialPairPolicy | None,
        include_same_residue_heavy_pairs: bool = False,
    ) -> Iterator[tuple[ClashPairAtomSite, ClashPairAtomSite]]:
        """Yield spatial candidates while applying one metric's pair policy."""

        if self.focus_residue_ids is not None:
            yield from _iter_focused_candidate_atom_site_pairs_from_index(
                self._cell_to_indexed_atom_sites,
                self._focused_indexed_atom_sites,
                focus_residue_ids=self.focus_residue_ids,
                policy=policy,
                include_same_residue_heavy_pairs=include_same_residue_heavy_pairs,
            )
            return

        yield from _iter_candidate_atom_site_pairs_from_index(
            self.atom_sites,
            self._cell_to_indexed_atom_sites,
            policy=policy,
            include_same_residue_heavy_pairs=include_same_residue_heavy_pairs,
        )


def iter_candidate_atom_site_pairs(
    atom_sites: tuple[ClashPairAtomSite, ...],
    *,
    focus_residue_ids: frozenset[ResidueId] | None,
    policy: SpatialPairPolicy | None,
    include_same_residue_heavy_pairs: bool = False,
) -> Iterator[tuple[ClashPairAtomSite, ClashPairAtomSite]]:
    """Yield locality-pruned candidate pairs, optionally restricted to one focus."""

    if focus_residue_ids is not None:
        yield from _iter_focused_candidate_atom_site_pairs(
            atom_sites,
            focus_residue_ids=focus_residue_ids,
            policy=policy,
            include_same_residue_heavy_pairs=include_same_residue_heavy_pairs,
        )
        return

    cell_to_atom_sites: dict[
        tuple[int, int, int],
        list[ClashPairAtomSite],
    ] = defaultdict(list)
    for atom_site in atom_sites:
        current_cell = atom_site.grid_cell
        cell_x, cell_y, cell_z = current_cell
        for offset_x, offset_y, offset_z in NEIGHBOR_CELL_OFFSETS:
            previous_atom_sites = cell_to_atom_sites.get(
                (
                    cell_x + offset_x,
                    cell_y + offset_y,
                    cell_z + offset_z,
                )
            )
            if previous_atom_sites is None:
                continue

            for previous_atom_site in previous_atom_sites:
                if pair_can_be_rejected_before_distance(
                    previous_atom_site,
                    atom_site,
                    policy=policy,
                    include_same_residue_heavy_pairs=(
                        include_same_residue_heavy_pairs
                    ),
                ):
                    continue

                yield previous_atom_site, atom_site

        cell_to_atom_sites[current_cell].append(atom_site)


def _iter_focused_candidate_atom_site_pairs(
    atom_sites: tuple[ClashPairAtomSite, ...],
    *,
    focus_residue_ids: frozenset[ResidueId],
    policy: SpatialPairPolicy | None,
    include_same_residue_heavy_pairs: bool,
) -> Iterator[tuple[ClashPairAtomSite, ClashPairAtomSite]]:
    """Yield candidate pairs touching focus atoms without streaming all pairs."""

    cell_to_indexed_atom_sites: dict[
        tuple[int, int, int],
        list[tuple[int, ClashPairAtomSite]],
    ] = defaultdict(list)
    focused_indexed_atom_sites: list[tuple[int, ClashPairAtomSite]] = []
    for atom_index, atom_site in enumerate(atom_sites):
        cell_to_indexed_atom_sites[atom_site.grid_cell].append((atom_index, atom_site))
        if atom_site.residue_id in focus_residue_ids:
            focused_indexed_atom_sites.append((atom_index, atom_site))

    yield from _iter_focused_candidate_atom_site_pairs_from_index(
        cell_to_indexed_atom_sites,
        tuple(focused_indexed_atom_sites),
        focus_residue_ids=focus_residue_ids,
        policy=policy,
        include_same_residue_heavy_pairs=include_same_residue_heavy_pairs,
    )


def _iter_candidate_atom_site_pairs_from_index(
    atom_sites: tuple[ClashPairAtomSite, ...],
    cell_to_indexed_atom_sites: Mapping[
        tuple[int, int, int],
        Sequence[tuple[int, ClashPairAtomSite]],
    ],
    *,
    policy: SpatialPairPolicy | None,
    include_same_residue_heavy_pairs: bool,
) -> Iterator[tuple[ClashPairAtomSite, ClashPairAtomSite]]:
    """Yield all spatial candidates from one immutable cell index."""

    for atom_index, atom_site in enumerate(atom_sites):
        current_cell = atom_site.grid_cell
        cell_x, cell_y, cell_z = current_cell
        for offset_x, offset_y, offset_z in NEIGHBOR_CELL_OFFSETS:
            neighboring_atom_sites = cell_to_indexed_atom_sites.get(
                (
                    cell_x + offset_x,
                    cell_y + offset_y,
                    cell_z + offset_z,
                )
            )
            if neighboring_atom_sites is None:
                continue

            for previous_atom_index, previous_atom_site in neighboring_atom_sites:
                if previous_atom_index >= atom_index:
                    break
                if pair_can_be_rejected_before_distance(
                    previous_atom_site,
                    atom_site,
                    policy=policy,
                    include_same_residue_heavy_pairs=(
                        include_same_residue_heavy_pairs
                    ),
                ):
                    continue

                yield previous_atom_site, atom_site


def _iter_focused_candidate_atom_site_pairs_from_index(
    cell_to_indexed_atom_sites: Mapping[
        tuple[int, int, int],
        Sequence[tuple[int, ClashPairAtomSite]],
    ],
    focused_indexed_atom_sites: tuple[tuple[int, ClashPairAtomSite], ...],
    *,
    focus_residue_ids: frozenset[ResidueId],
    policy: SpatialPairPolicy | None,
    include_same_residue_heavy_pairs: bool,
) -> Iterator[tuple[ClashPairAtomSite, ClashPairAtomSite]]:
    """Yield focused spatial candidates from one immutable cell index."""

    for focus_atom_index, focus_atom_site in focused_indexed_atom_sites:
        current_cell = focus_atom_site.grid_cell
        cell_x, cell_y, cell_z = current_cell
        for offset_x, offset_y, offset_z in NEIGHBOR_CELL_OFFSETS:
            neighboring_atom_sites = cell_to_indexed_atom_sites.get(
                (
                    cell_x + offset_x,
                    cell_y + offset_y,
                    cell_z + offset_z,
                )
            )
            if neighboring_atom_sites is None:
                continue

            for other_atom_index, other_atom_site in neighboring_atom_sites:
                if other_atom_index == focus_atom_index:
                    continue
                if (
                    other_atom_site.residue_id in focus_residue_ids
                    and other_atom_index < focus_atom_index
                ):
                    continue

                if other_atom_index < focus_atom_index:
                    left_site = other_atom_site
                    right_site = focus_atom_site
                else:
                    left_site = focus_atom_site
                    right_site = other_atom_site

                if pair_can_be_rejected_before_distance(
                    left_site,
                    right_site,
                    policy=policy,
                    include_same_residue_heavy_pairs=(
                        include_same_residue_heavy_pairs
                    ),
                ):
                    continue

                yield left_site, right_site


def pair_can_be_rejected_before_distance(
    left_site: ClashPairAtomSite,
    right_site: ClashPairAtomSite,
    *,
    policy: SpatialPairPolicy | None,
    include_same_residue_heavy_pairs: bool = False,
) -> bool:
    """Return whether one pair can be rejected without distance/topology work."""

    if (
        policy is not None
        and not policy.include_ligands
        and (
            left_site.domain.excluded_when_ligands_are_disabled()
            or right_site.domain.excluded_when_ligands_are_disabled()
        )
    ):
        return True

    if (
        policy is not None
        and not policy.include_hydrogen_hydrogen
        and left_site.is_hydrogen_atom
        and right_site.is_hydrogen_atom
    ):
        return True

    return (
        policy is not None
        and not include_same_residue_heavy_pairs
        and left_site.residue_id == right_site.residue_id
        and not left_site.is_hydrogen_atom
        and not right_site.is_hydrogen_atom
    )
