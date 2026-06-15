"""Spatial candidate-pair generation for clash diagnostics."""

from collections import defaultdict
from collections.abc import Iterator

from typing_extensions import Protocol

from protrepair.structure.labels import ResidueId

NEIGHBOR_CELL_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)


class ClashPairAtomSite(Protocol):
    """Atom-site surface required by spatial pair generation."""

    @property
    def residue_id(self) -> ResidueId:
        """Return residue id."""

        ...

    @property
    def domain(self) -> object:
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


class ClashPairPolicy(Protocol):
    """Policy surface required by pre-distance pair rejection."""

    @property
    def include_ligands(self) -> bool:
        """Return whether ligand pairs are included."""

        ...

    @property
    def include_hydrogen_hydrogen(self) -> bool:
        """Return whether hydrogen-hydrogen pairs are included."""

        ...


def iter_candidate_atom_site_pairs(
    atom_sites: tuple[ClashPairAtomSite, ...],
    *,
    focus_residue_ids: frozenset[ResidueId] | None,
    policy: ClashPairPolicy | None,
) -> Iterator[tuple[ClashPairAtomSite, ClashPairAtomSite]]:
    """Yield locality-pruned candidate pairs, optionally restricted to one focus."""

    if focus_residue_ids is not None:
        yield from _iter_focused_candidate_atom_site_pairs(
            atom_sites,
            focus_residue_ids=focus_residue_ids,
            policy=policy,
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
                ):
                    continue

                yield previous_atom_site, atom_site

        cell_to_atom_sites[current_cell].append(atom_site)


def _iter_focused_candidate_atom_site_pairs(
    atom_sites: tuple[ClashPairAtomSite, ...],
    *,
    focus_residue_ids: frozenset[ResidueId],
    policy: ClashPairPolicy | None,
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
                ):
                    continue

                yield left_site, right_site


def pair_can_be_rejected_before_distance(
    left_site: ClashPairAtomSite,
    right_site: ClashPairAtomSite,
    *,
    policy: ClashPairPolicy | None,
) -> bool:
    """Return whether one pair can be rejected without distance/topology work."""

    if (
        policy is not None
        and not policy.include_ligands
        and (
            _domain_value(left_site) == "ligand"
            or _domain_value(right_site) == "ligand"
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
        and left_site.residue_id == right_site.residue_id
        and not left_site.is_hydrogen_atom
        and not right_site.is_hydrogen_atom
    )


def _domain_value(site: ClashPairAtomSite) -> str:
    domain = site.domain
    value = getattr(domain, "value", domain)
    return str(value)
