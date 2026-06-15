"""Rigid-frame atom placement over canonical component placement directives."""

from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from protrepair.chemistry import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.geometry import RigidTransform, Vec3
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.completion.shared.domain import (
    CompletionResiduePayload,
    LocalFramePlacementDirective,
    MseBridgePlacementDirective,
    ResidueCompletionSite,
    ResidueFramePlacementDirective,
    RigidComponentPlacementDirective,
)
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.context import ProteinTransformationContext

FloatArray = npt.NDArray[np.float64]
MAX_PLACEMENT_ANCHOR_HOPS = 2
MINIMUM_PLACEMENT_ANCHOR_COUNT = 3
MSE_COMPONENT_ID = "MSE"
MSE_BRIDGE_ANCHOR_ATOM_NAMES: tuple[str, ...] = ("CB", "CG", "CE")
MSE_SALVAGE_ALLOWED_POLICIES: frozenset[OrphanFragmentPolicy] = frozenset(
    {OrphanFragmentPolicy.SALVAGE_WHEN_SAFE}
)
SUPPORTED_HYP_TARGET_ATOM_NAMES: frozenset[str] = frozenset({"OD1"})
SUPPORTED_HYP_RETAINED_ANCHOR_ATOM_NAMES: frozenset[str] = frozenset({"CB", "CG", "CD"})


@dataclass(frozen=True, slots=True)
class RigidFramePlacementTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministic transformer for rigid-frame component atom placement."""

    site: ResidueCompletionSite
    directive: RigidComponentPlacementDirective

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one rigid-frame placement patch can be built."""

        if not self.site.is_selected_by(
            context.source_snapshot,
            context.atom_input,
        ):
            return False

        payload = self.site.payload(context.source_snapshot)
        if payload is None:
            return False

        return self._build_patch(context.source_snapshot) is not None

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after rigid-frame component atom placement."""

        payload = self.site.payload(context.source_snapshot)
        if payload is None:
            return context.source_snapshot

        patch = self._build_patch(context.source_snapshot)
        if patch is None:
            return context.source_snapshot

        return self.site.apply_patch(context.source_snapshot, patch)

    def _build_patch(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> OrderedAtomPatch | None:
        """Return one rigid-frame patch for the current directive."""

        payload = self.site.payload(snapshot)
        if payload is None:
            return None

        directive = self.directive
        if isinstance(directive, LocalFramePlacementDirective):
            return _place_component_atoms_from_local_frames(
                residue=payload,
                idealized_component=directive.idealized_component,
                reconstruction_layers=directive.reconstruction_layers,
                target_atom_names=directive.target_atom_names,
            )

        if isinstance(directive, ResidueFramePlacementDirective):
            return _place_component_atoms_from_residue_frame(
                residue=payload,
                idealized_component=directive.idealized_component,
                retained_fragment_atom_names=directive.retained_fragment_atom_names,
                reconstruction_layers=directive.reconstruction_layers,
                target_atom_names=directive.target_atom_names,
            )

        if isinstance(directive, MseBridgePlacementDirective):
            return _place_mse_bridge_atoms(
                original_residue=directive.original_payload,
                residue=payload,
                idealized_component=directive.idealized_component,
                target_atom_names=directive.target_atom_names,
                orphan_fragment_policy=directive.orphan_fragment_policy,
            )

        raise TypeError(
            f"unsupported rigid placement directive type {type(directive)!r}"
        )


def _place_component_atoms_from_local_frames(
    *,
    residue: CompletionResiduePayload,
    idealized_component: IdealizedComponent,
    reconstruction_layers: Collection[Collection[str]],
    target_atom_names: Collection[str],
) -> OrderedAtomPatch | None:
    """Return one patch placed from local anchor frames grown layer by layer."""

    normalized_target_atom_names = normalize_atom_names(target_atom_names)
    atom_by_name = {atom.atom_name: atom for atom in idealized_component.atoms}
    working_residue = residue
    for layer_atom_names in reconstruction_layers:
        for atom_name in layer_atom_names:
            if atom_name not in normalized_target_atom_names:
                continue

            placed_atom_names = frozenset(working_residue.atom_names())
            anchor_atom_names = _placement_anchor_atom_names(
                idealized_component,
                atom_name=atom_name,
                placed_atom_names=placed_atom_names,
            )
            if len(anchor_atom_names) < MINIMUM_PLACEMENT_ANCHOR_COUNT:
                return None

            transform = build_ideal_transform(
                residue=working_residue,
                idealized_component=idealized_component,
                anchor_atom_names=anchor_atom_names,
            )
            if transform is None:
                return None

            template_atom = atom_by_name.get(atom_name)
            if template_atom is None or template_atom.ideal_position is None:
                return None

            placed_position = Vec3.from_iterable(
                transform.apply(
                    np.asarray(template_atom.ideal_position, dtype=np.float64)
                )
            )
            working_residue = working_residue.with_atom_payload(
                build_atom_site_from_template_atom(
                    residue=working_residue,
                    template_atom=template_atom,
                ),
                atom_geometry=build_atom_geometry_from_template_atom(
                    residue=working_residue,
                    position=placed_position,
                ),
                formal_charge=(
                    None
                    if template_atom.formal_charge == 0
                    else template_atom.formal_charge
                ),
            )

    return OrderedAtomPatch.from_residue_payload(
        working_residue.residue_site,
        residue_geometry=working_residue.residue_geometry,
    )


def _place_component_atoms_from_residue_frame(
    *,
    residue: CompletionResiduePayload,
    idealized_component: IdealizedComponent,
    retained_fragment_atom_names: Collection[str],
    reconstruction_layers: Collection[Collection[str]],
    target_atom_names: Collection[str],
) -> OrderedAtomPatch | None:
    """Return one patch placed from the retained residue frame."""

    residue_frame_anchor_atom_names = tuple(
        atom_name
        for atom_name in retained_fragment_atom_names
        if idealized_component.atom_with_ideal_position(atom_name) is not None
    )
    if len(residue_frame_anchor_atom_names) < MINIMUM_PLACEMENT_ANCHOR_COUNT:
        return None

    transform = build_ideal_transform(
        residue=residue,
        idealized_component=idealized_component,
        anchor_atom_names=residue_frame_anchor_atom_names,
    )
    if transform is None:
        return None

    atom_by_name = {atom.atom_name: atom for atom in idealized_component.atoms}
    working_residue = residue
    for layer_atom_names in reconstruction_layers:
        for atom_name in layer_atom_names:
            if atom_name not in target_atom_names:
                continue

            template_atom = atom_by_name.get(atom_name)
            if template_atom is None or template_atom.ideal_position is None:
                return None

            placed_position = Vec3.from_iterable(
                transform.apply(
                    np.asarray(template_atom.ideal_position, dtype=np.float64)
                )
            )
            working_residue = working_residue.with_atom_payload(
                build_atom_site_from_template_atom(
                    residue=working_residue,
                    template_atom=template_atom,
                ),
                atom_geometry=build_atom_geometry_from_template_atom(
                    residue=working_residue,
                    position=placed_position,
                ),
                formal_charge=(
                    None
                    if template_atom.formal_charge == 0
                    else template_atom.formal_charge
                ),
            )

    return OrderedAtomPatch.from_residue_payload(
        working_residue.residue_site,
        residue_geometry=working_residue.residue_geometry,
    )


def _place_mse_bridge_atoms(
    *,
    original_residue: CompletionResiduePayload,
    residue: CompletionResiduePayload,
    idealized_component: IdealizedComponent,
    target_atom_names: Collection[str],
    orphan_fragment_policy: OrphanFragmentPolicy,
) -> OrderedAtomPatch | None:
    """Return one MSE bridge patch or ``None`` when repair is unsupported."""

    if idealized_component.component_id != MSE_COMPONENT_ID:
        return None

    normalized_target_atom_names = normalize_atom_names(target_atom_names)
    if "SE" not in normalized_target_atom_names:
        return None

    if not normalized_target_atom_names.issubset(frozenset({"SE", "CE"})):
        return None

    working_residue = residue
    if not working_residue.has_atom("CE"):
        if orphan_fragment_policy not in MSE_SALVAGE_ALLOWED_POLICIES:
            return None

        if not original_residue.has_atom("CE"):
            return None

        working_residue = working_residue.with_atom_payload(
            original_residue.atom_site("CE"),
            atom_geometry=original_residue.atom_geometry("CE"),
            formal_charge=original_residue.formal_charge("CE"),
        )

    if not all(
        working_residue.has_atom(atom_name)
        for atom_name in MSE_BRIDGE_ANCHOR_ATOM_NAMES
    ):
        return None

    transform = build_ideal_transform(
        residue=working_residue,
        idealized_component=idealized_component,
        anchor_atom_names=MSE_BRIDGE_ANCHOR_ATOM_NAMES,
    )
    if transform is None:
        return None

    template_atom = idealized_component.atom_with_ideal_position("SE")
    if template_atom is None or template_atom.ideal_position is None:
        return None

    placed_position = Vec3.from_iterable(
        transform.apply(np.asarray(template_atom.ideal_position, dtype=np.float64))
    )
    updated_residue = working_residue.with_atom_payload(
        build_atom_site_from_template_atom(
            residue=working_residue,
            template_atom=template_atom,
        ),
        atom_geometry=build_atom_geometry_from_template_atom(
            residue=working_residue,
            position=placed_position,
        ),
        formal_charge=(
            None if template_atom.formal_charge == 0 else template_atom.formal_charge
        ),
    )
    return OrderedAtomPatch.from_residue_payload(
        updated_residue.residue_site,
        residue_geometry=updated_residue.residue_geometry,
    )


def _placement_anchor_atom_names(
    idealized_component: IdealizedComponent,
    *,
    atom_name: str,
    placed_atom_names: Collection[str],
) -> tuple[str, ...]:
    """Return placed atom names that define a local ideal frame for one target."""

    normalized_placed_atom_names = normalize_atom_names(placed_atom_names)
    adjacency = bond_adjacency(idealized_component)
    atom_order = {
        atom.atom_name: index for index, atom in enumerate(idealized_component.atoms)
    }
    anchor_candidates: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque([(atom_name.strip().upper(), 0)])
    visited = {atom_name.strip().upper()}
    while queue:
        current_atom_name, hop_distance = queue.popleft()
        if hop_distance >= MAX_PLACEMENT_ANCHOR_HOPS:
            continue

        next_hop_distance = hop_distance + 1
        for neighbor_atom_name in adjacency.get(current_atom_name, ()):
            if neighbor_atom_name in visited:
                continue

            visited.add(neighbor_atom_name)
            queue.append((neighbor_atom_name, next_hop_distance))
            if neighbor_atom_name not in normalized_placed_atom_names:
                continue

            anchor_candidates[neighbor_atom_name] = next_hop_distance

    return tuple(
        atom_name
        for atom_name, _ in sorted(
            anchor_candidates.items(),
            key=lambda item: (item[1], atom_order[item[0]]),
        )
        if idealized_component.atom_with_ideal_position(atom_name) is not None
    )


def build_ideal_transform(
    *,
    residue: CompletionResiduePayload,
    idealized_component: IdealizedComponent,
    anchor_atom_names: Collection[str],
) -> RigidTransform | None:
    """Return the rigid transform that aligns ideal anchors onto observed ones."""

    source_points = ideal_coordinate_matrix(idealized_component, anchor_atom_names)
    target_points = observed_coordinate_matrix(residue, anchor_atom_names)
    return RigidTransform.fit(
        source_points,
        target_points,
        minimum_point_count=MINIMUM_PLACEMENT_ANCHOR_COUNT,
        minimum_rank=2,
    )


def ideal_coordinate_matrix(
    idealized_component: IdealizedComponent,
    atom_names: Collection[str],
) -> FloatArray:
    """Return ideal coordinates for one ordered atom-name collection."""

    ideal_points: list[tuple[float, float, float]] = []
    for atom_name in atom_names:
        template_atom = idealized_component.atom_with_ideal_position(atom_name)
        if template_atom is None or template_atom.ideal_position is None:
            raise ValueError(
                f"{idealized_component.component_id} is missing ideal coordinates "
                f"for {atom_name}"
            )

        ideal_points.append(template_atom.ideal_position)

    return np.asarray(ideal_points, dtype=np.float64)


def observed_coordinate_matrix(
    residue: CompletionResiduePayload,
    atom_names: Collection[str],
) -> FloatArray:
    """Return observed coordinates for one ordered atom-name collection."""

    residue_geometry = residue.residue_geometry
    return np.asarray(
        [tuple(residue_geometry.position(atom_name)) for atom_name in atom_names],
        dtype=np.float64,
    )


def build_atom_site_from_template_atom(
    *,
    residue: CompletionResiduePayload,
    template_atom: IdealizedComponentAtom,
) -> AtomSite:
    """Return one placed atom site projected back into residue payload."""

    return AtomSite(
        name=template_atom.atom_name,
        element=template_atom.element,
    )


def build_atom_geometry_from_template_atom(
    *,
    residue: CompletionResiduePayload,
    position: Vec3,
) -> AtomGeometry:
    """Return one placed atom geometry projected back into residue payload."""

    prototype_atom_site = residue.atom_sites[0]
    prototype_atom_geometry = residue.atom_geometry(prototype_atom_site.name)
    return AtomGeometry(
        position=position,
        occupancy=prototype_atom_geometry.occupancy,
        b_factor=prototype_atom_geometry.b_factor,
        altloc=prototype_atom_geometry.altloc,
    )


def bond_adjacency(
    idealized_component: IdealizedComponent,
) -> Mapping[str, frozenset[str]]:
    """Return one immutable atom-name adjacency mapping for an idealized component."""

    adjacency: dict[str, set[str]] = {}
    for bond in idealized_component.bonds:
        adjacency.setdefault(bond.atom_name_1, set()).add(bond.atom_name_2)
        adjacency.setdefault(bond.atom_name_2, set()).add(bond.atom_name_1)

    return {
        atom_name: frozenset(neighbor_atom_names)
        for atom_name, neighbor_atom_names in adjacency.items()
    }


def normalize_atom_names(atom_names: Collection[str]) -> frozenset[str]:
    """Return one uppercase atom-name set with blanks removed."""

    return frozenset(
        atom_name.strip().upper() for atom_name in atom_names if atom_name.strip()
    )
