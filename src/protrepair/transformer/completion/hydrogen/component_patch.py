"""Rigid-frame hydrogen component-patch construction."""

from collections import deque
from copy import deepcopy

import numpy as np

from protrepair.chemistry import (
    IdealGeometryHydrogenSemantics,
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.geometry import RigidTransform
from protrepair.transformer.completion.atom.rigid_frame import (
    build_ideal_transform,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

BACKBONE_OR_TERMINAL_HEAVY_ATOM_NAMES: frozenset[str] = frozenset({"N", "OXT"})
MINIMUM_HYDROGEN_ANCHOR_COUNT = 3
MAX_LOCAL_HYDROGEN_ANCHOR_HOPS = 3


def generate_component_hydrogen_patch(
    *,
    residue: CompletionResiduePayload,
    patch: OrderedAtomPatch,
    semantics: IdealGeometryHydrogenSemantics,
    skip_hydrogens_with_absent_anchors: bool = False,
) -> OrderedAtomPatch | None:
    """Return one residue-local hydrogen patch for one idealized component."""

    idealized_component = semantics.component
    hydrogen_atoms = supported_component_hydrogen_atoms(idealized_component)
    global_anchor_atom_names = tuple(
        atom.atom_name
        for atom in idealized_component.heavy_atoms()
        if residue.has_atom(atom.atom_name)
    )
    atom_names = list(deepcopy(patch.atom_names))
    atom_coordinates = [list(position) for position in patch.atom_coordinates]
    transform_by_anchor_atom_names: dict[tuple[str, ...], RigidTransform] = {}
    global_transform: RigidTransform | None = None
    for template_atom in hydrogen_atoms:
        anchor_atom_name = idealized_component.hydrogen_anchor_atom_name(
            template_atom.atom_name
        )
        if anchor_atom_name is None:
            return None

        if not residue.has_atom(anchor_atom_name):
            if skip_hydrogens_with_absent_anchors:
                continue

            return None

        local_anchor_atom_names = _local_hydrogen_anchor_atom_names(
            idealized_component,
            hydrogen_atom_name=template_atom.atom_name,
            placed_atom_names=residue.atom_names(),
        )
        transform = None
        if len(local_anchor_atom_names) >= MINIMUM_HYDROGEN_ANCHOR_COUNT:
            local_anchor_atom_names = local_anchor_atom_names[
                :MINIMUM_HYDROGEN_ANCHOR_COUNT
            ]
            transform = transform_by_anchor_atom_names.get(local_anchor_atom_names)
            if transform is None:
                transform = build_ideal_transform(
                    residue=residue,
                    idealized_component=idealized_component,
                    anchor_atom_names=local_anchor_atom_names,
                )
                if transform is not None:
                    transform_by_anchor_atom_names[local_anchor_atom_names] = transform

        if transform is None:
            if global_transform is None:
                global_transform = build_ideal_transform(
                    residue=residue,
                    idealized_component=idealized_component,
                    anchor_atom_names=global_anchor_atom_names,
                )
            transform = global_transform
        if transform is None:
            return None

        assert template_atom.ideal_position is not None
        placed_position = transform.apply(
            np.asarray(template_atom.ideal_position, dtype=np.float64)
        )
        atom_names.append(template_atom.atom_name)
        atom_coordinates.append(list(placed_position))

    return OrderedAtomPatch.from_atom_coordinates(
        atom_names=atom_names,
        atom_coordinates=atom_coordinates,
    )


def supported_component_hydrogen_atoms(
    idealized_component: IdealizedComponent,
) -> tuple[IdealizedComponentAtom, ...]:
    """Return non-backbone hydrogen atoms in packaged order."""

    hydrogen_atoms: list[IdealizedComponentAtom] = []
    for atom in idealized_component.atoms:
        if atom.element != "H" or atom.ideal_position is None:
            continue

        anchor_atom_name = idealized_component.hydrogen_anchor_atom_name(atom.atom_name)
        if anchor_atom_name is None:
            continue

        if anchor_atom_name in BACKBONE_OR_TERMINAL_HEAVY_ATOM_NAMES:
            continue

        hydrogen_atoms.append(atom)

    return tuple(hydrogen_atoms)


def _local_hydrogen_anchor_atom_names(
    idealized_component: IdealizedComponent,
    *,
    hydrogen_atom_name: str,
    placed_atom_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return one local heavy-atom frame for one hydrogen target."""

    normalized_hydrogen_atom_name = hydrogen_atom_name.strip().upper()
    normalized_placed_atom_names = frozenset(
        atom_name.strip().upper() for atom_name in placed_atom_names
    )
    atom_order = {
        atom.atom_name: index for index, atom in enumerate(idealized_component.atoms)
    }
    anchor_candidates: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque([(normalized_hydrogen_atom_name, 0)])
    visited = {normalized_hydrogen_atom_name}
    while queue:
        current_atom_name, hop_distance = queue.popleft()
        if hop_distance >= MAX_LOCAL_HYDROGEN_ANCHOR_HOPS:
            continue

        next_hop_distance = hop_distance + 1
        for neighbor_atom_name in idealized_component.bonded_atom_names(
            current_atom_name
        ):
            if neighbor_atom_name in visited:
                continue

            visited.add(neighbor_atom_name)
            queue.append((neighbor_atom_name, next_hop_distance))
            if neighbor_atom_name not in normalized_placed_atom_names:
                continue

            neighbor_atom = idealized_component.atom_with_ideal_position(
                neighbor_atom_name
            )
            if neighbor_atom is None or neighbor_atom.element == "H":
                continue

            anchor_candidates[neighbor_atom_name] = next_hop_distance

    return tuple(
        atom_name
        for atom_name, _ in sorted(
            anchor_candidates.items(),
            key=lambda item: (item[1], atom_order[item[0]]),
        )
    )
