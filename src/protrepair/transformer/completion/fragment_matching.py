"""Generic fragment correspondence artifacts for partial residue repair."""

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass

from protrepair.chemistry import ResidueTemplate
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload


@dataclass(frozen=True, slots=True)
class FragmentBoundary:
    """One direct template bond crossing from matched atoms to missing atoms."""

    present_atom_name: str
    missing_atom_name: str

    def __post_init__(self) -> None:
        present_atom_name = self.present_atom_name.strip().upper()
        missing_atom_name = self.missing_atom_name.strip().upper()
        if not present_atom_name or not missing_atom_name:
            raise ValueError("fragment boundary atom names must not be blank")

        if present_atom_name == missing_atom_name:
            raise ValueError("fragment boundaries must cross two distinct atoms")

        object.__setattr__(self, "present_atom_name", present_atom_name)
        object.__setattr__(self, "missing_atom_name", missing_atom_name)


@dataclass(frozen=True, slots=True)
class ResidueFragmentMatch:
    """Canonical fragment correspondence between one residue and one template."""

    residue: CompletionResiduePayload
    template: ResidueTemplate
    excluded_atom_names: tuple[str, ...]
    matched_atom_names: tuple[str, ...]
    missing_atom_names: tuple[str, ...]
    unexpected_atom_names: tuple[str, ...]
    present_fragments: tuple[tuple[str, ...], ...]
    missing_fragments: tuple[tuple[str, ...], ...]
    boundary_bonds: tuple[FragmentBoundary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "excluded_atom_names",
            tuple(self.excluded_atom_names),
        )
        object.__setattr__(
            self,
            "matched_atom_names",
            tuple(self.matched_atom_names),
        )
        object.__setattr__(
            self,
            "missing_atom_names",
            tuple(self.missing_atom_names),
        )
        object.__setattr__(
            self,
            "unexpected_atom_names",
            tuple(self.unexpected_atom_names),
        )
        object.__setattr__(
            self,
            "present_fragments",
            tuple(tuple(fragment) for fragment in self.present_fragments),
        )
        object.__setattr__(
            self,
            "missing_fragments",
            tuple(tuple(fragment) for fragment in self.missing_fragments),
        )
        object.__setattr__(
            self,
            "boundary_bonds",
            tuple(self.boundary_bonds),
        )

    def is_complete(self) -> bool:
        """Return whether the residue fully matches the template fragment-wise."""

        return not self.missing_atom_names and not self.unexpected_atom_names

    def has_missing_atoms(self) -> bool:
        """Return whether the residue is missing expected template atoms."""

        return bool(self.missing_atom_names)

    def has_unexpected_atoms(self) -> bool:
        """Return whether the residue contains atoms outside the template."""

        return bool(self.unexpected_atom_names)

    def repair_seed_atom_names(self) -> tuple[str, ...]:
        """Return matched atom names that border missing template atoms."""

        seed_atom_names: list[str] = []
        seen_atom_names: set[str] = set()
        for boundary in self.boundary_bonds:
            if boundary.present_atom_name in seen_atom_names:
                continue

            seed_atom_names.append(boundary.present_atom_name)
            seen_atom_names.add(boundary.present_atom_name)

        return tuple(seed_atom_names)

    def repair_target_atom_names(self) -> tuple[str, ...]:
        """Return missing template atom names directly adjacent to matched atoms."""

        target_atom_names: list[str] = []
        seen_atom_names: set[str] = set()
        for boundary in self.boundary_bonds:
            if boundary.missing_atom_name in seen_atom_names:
                continue

            target_atom_names.append(boundary.missing_atom_name)
            seen_atom_names.add(boundary.missing_atom_name)

        return tuple(target_atom_names)

    def largest_present_fragment(self) -> tuple[str, ...]:
        """Return the largest connected matched fragment in template order."""

        if not self.present_fragments:
            return ()

        return max(
            self.present_fragments,
            key=lambda fragment: (
                len(fragment),
                tuple(self._atom_order(atom_name) for atom_name in fragment),
            ),
        )

    def primary_repair_fragment(
        self,
        *,
        preferred_anchor_atom_names: Collection[str],
    ) -> tuple[str, ...]:
        """Return the fragment to preserve as the seed for reconstruction."""

        if not self.present_fragments:
            return ()

        normalized_anchor_atom_names = {
            atom_name.strip().upper() for atom_name in preferred_anchor_atom_names
        }
        return max(
            self.present_fragments,
            key=lambda fragment: (
                sum(
                    atom_name in normalized_anchor_atom_names
                    for atom_name in fragment
                ),
                len(fragment),
                tuple(self._atom_order(atom_name) for atom_name in fragment),
            ),
        )

    def orphan_atom_names(
        self,
        *,
        preferred_anchor_atom_names: Collection[str],
    ) -> tuple[str, ...]:
        """Return matched template atoms that should be rebuilt."""

        primary_fragment = set(
            self.primary_repair_fragment(
                preferred_anchor_atom_names=preferred_anchor_atom_names,
            )
        )
        orphan_atom_names = [
            atom_name
            for fragment in self.present_fragments
            for atom_name in fragment
            if atom_name not in primary_fragment
        ]
        return tuple(orphan_atom_names)

    def _atom_order(self, atom_name: str) -> int:
        ordered_atom_names = self.template.ordered_atom_names()
        return ordered_atom_names.index(atom_name)


def match_residue_fragment(
    residue: CompletionResiduePayload,
    template: ResidueTemplate,
    *,
    exclude_atom_names: Collection[str] = (),
) -> ResidueFragmentMatch:
    """Return a canonical fragment match between one residue and one template."""

    excluded_atom_names = tuple(
        atom_name.strip().upper()
        for atom_name in exclude_atom_names
        if atom_name.strip()
    )
    excluded_atom_name_set = set(excluded_atom_names)
    expected_atom_names = template.ordered_atom_names()
    present_atom_name_set = set(residue.atom_names())

    matched_atom_names = tuple(
        atom_name
        for atom_name in expected_atom_names
        if atom_name not in excluded_atom_name_set
        and atom_name in present_atom_name_set
    )
    missing_atom_names = tuple(
        atom_name
        for atom_name in expected_atom_names
        if atom_name not in excluded_atom_name_set
        and atom_name not in present_atom_name_set
    )
    unexpected_atom_names = tuple(
        atom_name
        for atom_name in residue.atom_names()
        if atom_name not in excluded_atom_name_set
        and not template.definition.has_atom(atom_name)
    )

    return ResidueFragmentMatch(
        residue=residue,
        template=template,
        excluded_atom_names=excluded_atom_names,
        matched_atom_names=matched_atom_names,
        missing_atom_names=missing_atom_names,
        unexpected_atom_names=unexpected_atom_names,
        present_fragments=connected_template_fragments(
            template,
            matched_atom_names,
        ),
        missing_fragments=connected_template_fragments(
            template,
            missing_atom_names,
        ),
        boundary_bonds=boundary_bonds(
            template,
            matched_atom_names=matched_atom_names,
            missing_atom_names=missing_atom_names,
        ),
    )


def connected_template_fragments(
    template: ResidueTemplate,
    atom_names: Collection[str],
) -> tuple[tuple[str, ...], ...]:
    """Return connected template fragments for one atom-name subset."""

    ordered_atom_names = template.ordered_atom_names()
    atom_name_order = {
        atom_name: index for index, atom_name in enumerate(ordered_atom_names)
    }
    remaining_atom_names = {
        atom_name.strip().upper()
        for atom_name in atom_names
        if template.definition.has_atom(atom_name)
    }
    fragments: list[tuple[str, ...]] = []

    while remaining_atom_names:
        seed_atom_name = min(
            remaining_atom_names,
            key=lambda atom_name: atom_name_order[atom_name],
        )
        queue: deque[str] = deque([seed_atom_name])
        visited_atom_names = {seed_atom_name}
        fragment_atom_names: list[str] = []

        while queue:
            current_atom_name = queue.popleft()
            fragment_atom_names.append(current_atom_name)
            for neighbor_atom_name in template.definition.bonded_atom_names(
                current_atom_name
            ):
                if neighbor_atom_name not in remaining_atom_names:
                    continue

                if neighbor_atom_name in visited_atom_names:
                    continue

                visited_atom_names.add(neighbor_atom_name)
                queue.append(neighbor_atom_name)

        for atom_name in visited_atom_names:
            remaining_atom_names.remove(atom_name)

        fragment_atom_names.sort(key=lambda atom_name: atom_name_order[atom_name])
        fragments.append(tuple(fragment_atom_names))

    fragments.sort(
        key=lambda fragment: atom_name_order[fragment[0]],
    )
    return tuple(fragments)


def boundary_bonds(
    template: ResidueTemplate,
    *,
    matched_atom_names: Collection[str],
    missing_atom_names: Collection[str],
) -> tuple[FragmentBoundary, ...]:
    """Return direct template bonds crossing from matched atoms to missing atoms."""

    missing_atom_name_set = {
        atom_name.strip().upper() for atom_name in missing_atom_names
    }
    boundaries: list[FragmentBoundary] = []
    for present_atom_name in template.ordered_atom_names():
        if present_atom_name not in matched_atom_names:
            continue

        for missing_atom_name in template.ordered_atom_names():
            if missing_atom_name not in missing_atom_name_set:
                continue

            if (
                template.definition.bond_hop_distance(
                    present_atom_name,
                    missing_atom_name,
                    max_hops=1,
                )
                != 1
            ):
                continue

            boundaries.append(
                FragmentBoundary(
                    present_atom_name=present_atom_name,
                    missing_atom_name=missing_atom_name,
                )
            )

    return tuple(boundaries)
