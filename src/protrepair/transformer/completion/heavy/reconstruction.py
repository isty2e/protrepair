"""Canonical reconstruction plans for ideal-geometry completion paths."""

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass

from protrepair.chemistry import (
    IdealGeometryHeavyAtomMode,
    IdealGeometryHeavyAtomSemantics,
    IdealizedComponent,
    ResidueTemplate,
)
from protrepair.transformer.completion.fragment_matching import (
    FragmentBoundary,
    ResidueFragmentMatch,
    match_residue_fragment,
)
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.completion.shared.domain import (
    CompletionResiduePayload,
    LocalFramePlacementDirective,
    MseBridgePlacementDirective,
    ResidueFramePlacementDirective,
    RigidComponentPlacementDirective,
)

PRIMARY_NONSTANDARD_ANCHOR_ATOM_NAMES: tuple[str, ...] = ("N", "CA", "C", "O", "CB")
STRUCTURAL_NONSTANDARD_ANCHOR_ATOM_NAMES: tuple[str, ...] = (
    "N",
    "CA",
    "C",
    "O",
    "CB",
)
MINIMUM_PLACEMENT_ANCHOR_COUNT = 3
MSE_COMPONENT_ID = "MSE"
MSE_BRIDGE_TARGET_ATOM_NAMES: frozenset[str] = frozenset({"SE", "CE"})
MSE_SALVAGE_ALLOWED_POLICIES: frozenset[OrphanFragmentPolicy] = frozenset(
    {OrphanFragmentPolicy.SALVAGE_WHEN_SAFE}
)
SUPPORTED_HYP_TARGET_ATOM_NAMES: frozenset[str] = frozenset({"OD1"})
SUPPORTED_HYP_RETAINED_ANCHOR_ATOM_NAMES: frozenset[str] = frozenset({"CB", "CG", "CD"})


@dataclass(frozen=True, slots=True)
class ParentAtomMapping:
    """Canonical correspondence between one idealized residue and its parent."""

    parent_standard_id: str
    shared_atom_names: tuple[str, ...]
    parent_only_atom_names: tuple[str, ...]
    component_only_atom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_standard_id",
            self.parent_standard_id.strip().upper(),
        )
        object.__setattr__(self, "shared_atom_names", tuple(self.shared_atom_names))
        object.__setattr__(
            self,
            "parent_only_atom_names",
            tuple(self.parent_only_atom_names),
        )
        object.__setattr__(
            self,
            "component_only_atom_names",
            tuple(self.component_only_atom_names),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionPlacementLayer:
    """One breadth-first placement layer away from the retained anchor fragment."""

    hop_distance: int
    atom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hop_distance <= 0:
            raise ValueError("reconstruction layers must have positive hop distance")

        object.__setattr__(self, "atom_names", tuple(self.atom_names))


@dataclass(frozen=True, slots=True)
class ComponentReconstructionPlan:
    """Canonical reconstruction plan for one partial idealized residue."""

    residue: CompletionResiduePayload
    idealized_component: IdealizedComponent
    template: ResidueTemplate
    fragment_match: ResidueFragmentMatch
    parent_mapping: ParentAtomMapping | None
    retained_fragment_atom_names: tuple[str, ...]
    structural_anchor_atom_names: tuple[str, ...]
    orphan_atom_names: tuple[str, ...]
    reconstruction_atom_names: tuple[str, ...]
    frontier_bonds: tuple[FragmentBoundary, ...]
    reconstruction_layers: tuple[ReconstructionPlacementLayer, ...]
    unreachable_atom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "retained_fragment_atom_names",
            tuple(self.retained_fragment_atom_names),
        )
        object.__setattr__(
            self,
            "structural_anchor_atom_names",
            tuple(self.structural_anchor_atom_names),
        )
        object.__setattr__(self, "orphan_atom_names", tuple(self.orphan_atom_names))
        object.__setattr__(
            self,
            "reconstruction_atom_names",
            tuple(self.reconstruction_atom_names),
        )
        object.__setattr__(self, "frontier_bonds", tuple(self.frontier_bonds))
        object.__setattr__(
            self,
            "reconstruction_layers",
            tuple(self.reconstruction_layers),
        )
        object.__setattr__(
            self,
            "unreachable_atom_names",
            tuple(self.unreachable_atom_names),
        )

    def has_reconstruction_work(self) -> bool:
        """Return whether this plan contains atoms to reconstruct."""

        return bool(self.reconstruction_atom_names)

    def has_structural_anchor(self) -> bool:
        """Return whether the retained fragment contains backbone-like anchors."""

        return bool(self.structural_anchor_atom_names)

    def placement_atom_names(self) -> tuple[str, ...]:
        """Return reconstruction targets flattened in placement order."""

        return tuple(
            atom_name
            for layer in self.reconstruction_layers
            for atom_name in layer.atom_names
        )

    def rigid_component_placement_directive(
        self,
        *,
        original_residue: CompletionResiduePayload,
        residue: CompletionResiduePayload,
        semantics: IdealGeometryHeavyAtomSemantics,
        target_atom_names: Collection[str],
        orphan_fragment_policy: OrphanFragmentPolicy,
    ) -> RigidComponentPlacementDirective | None:
        """Return the chosen rigid-frame placement directive for this plan."""

        normalized_target_atom_names = tuple(
            atom_name
            for atom_name in self.template.ordered_atom_names()
            if atom_name in normalized_atom_name_set(target_atom_names)
        )
        if not normalized_target_atom_names:
            return None

        if self._supports_local_frame_placement(
            semantics,
            target_atom_names=normalized_target_atom_names,
        ):
            return LocalFramePlacementDirective(
                idealized_component=semantics.component,
                reconstruction_layers=tuple(
                    layer.atom_names for layer in self.reconstruction_layers
                ),
                target_atom_names=normalized_target_atom_names,
            )

        if self._supports_residue_frame_placement(
            semantics,
            target_atom_names=normalized_target_atom_names,
        ):
            return ResidueFramePlacementDirective(
                idealized_component=semantics.component,
                retained_fragment_atom_names=self.retained_fragment_atom_names,
                reconstruction_layers=tuple(
                    layer.atom_names for layer in self.reconstruction_layers
                ),
                target_atom_names=normalized_target_atom_names,
            )

        if not self._supports_mse_bridge_salvage(
            original_residue=original_residue,
            residue=residue,
            semantics=semantics,
            target_atom_names=normalized_target_atom_names,
            orphan_fragment_policy=orphan_fragment_policy,
        ):
            return None

        return MseBridgePlacementDirective(
            original_payload=original_residue,
            idealized_component=semantics.component,
            target_atom_names=normalized_target_atom_names,
            orphan_fragment_policy=orphan_fragment_policy,
        )

    def _supports_local_frame_placement(
        self,
        semantics: IdealGeometryHeavyAtomSemantics,
        *,
        target_atom_names: Collection[str],
    ) -> bool:
        """Return whether this plan fits the narrow local-frame placement path."""

        if semantics.mode is not IdealGeometryHeavyAtomMode.LOCAL_FRAME:
            return False

        if self.unreachable_atom_names:
            return False

        if self.orphan_atom_names:
            return False

        if frozenset(target_atom_names) != SUPPORTED_HYP_TARGET_ATOM_NAMES:
            return False

        if frozenset(self.reconstruction_atom_names) != SUPPORTED_HYP_TARGET_ATOM_NAMES:
            return False

        return SUPPORTED_HYP_RETAINED_ANCHOR_ATOM_NAMES.issubset(
            frozenset(self.retained_fragment_atom_names)
        )

    def _supports_residue_frame_placement(
        self,
        semantics: IdealGeometryHeavyAtomSemantics,
        *,
        target_atom_names: Collection[str],
    ) -> bool:
        """Return whether this plan supports retained-fragment subtree seeding."""

        if semantics.mode is not IdealGeometryHeavyAtomMode.RESIDUE_FRAME:
            return False

        if self.unreachable_atom_names:
            return False

        if self.orphan_atom_names:
            return False

        if frozenset(target_atom_names) != frozenset(self.reconstruction_atom_names):
            return False

        residue_frame_anchor_atom_names = tuple(
            atom_name
            for atom_name in self.retained_fragment_atom_names
            if semantics.component.atom_with_ideal_position(atom_name) is not None
        )
        return len(residue_frame_anchor_atom_names) >= MINIMUM_PLACEMENT_ANCHOR_COUNT

    def _supports_mse_bridge_salvage(
        self,
        *,
        original_residue: CompletionResiduePayload,
        residue: CompletionResiduePayload,
        semantics: IdealGeometryHeavyAtomSemantics,
        target_atom_names: Collection[str],
        orphan_fragment_policy: OrphanFragmentPolicy,
    ) -> bool:
        """Return whether this plan can fall back to the MSE bridge salvage path."""

        if not semantics.allows_reference_or_salvage():
            return False

        if semantics.component.component_id != MSE_COMPONENT_ID:
            return False

        normalized_target_atom_names = frozenset(target_atom_names)
        if "SE" not in normalized_target_atom_names:
            return False

        if not normalized_target_atom_names.issubset(MSE_BRIDGE_TARGET_ATOM_NAMES):
            return False

        if residue.has_atom("CE"):
            return True

        return (
            orphan_fragment_policy in MSE_SALVAGE_ALLOWED_POLICIES
            and original_residue.has_atom("CE")
        )


def build_component_reconstruction_plan(
    residue: CompletionResiduePayload,
    idealized_component: IdealizedComponent,
    *,
    template: ResidueTemplate,
    parent_template: ResidueTemplate | None = None,
    orphan_fragment_policy: OrphanFragmentPolicy = OrphanFragmentPolicy.REBUILD,
    exclude_atom_names: Collection[str] = ("OXT",),
) -> ComponentReconstructionPlan:
    """Build one canonical reconstruction plan for an idealized residue."""

    fragment_match = match_residue_fragment(
        residue,
        template,
        exclude_atom_names=exclude_atom_names,
    )
    retained_fragment_atom_names = fragment_match.primary_repair_fragment(
        preferred_anchor_atom_names=PRIMARY_NONSTANDARD_ANCHOR_ATOM_NAMES,
    )
    structural_anchor_atom_names = tuple(
        atom_name
        for atom_name in retained_fragment_atom_names
        if atom_name in STRUCTURAL_NONSTANDARD_ANCHOR_ATOM_NAMES
    )
    orphan_atom_names = planned_orphan_atom_names(
        fragment_match,
        orphan_fragment_policy=orphan_fragment_policy,
    )
    planned_reconstruction_atom_names = plan_reconstruction_atom_names(
        template,
        missing_atom_names=fragment_match.missing_atom_names,
        orphan_atom_names=orphan_atom_names,
    )
    frontier_bonds = reconstruction_frontier_bonds(
        template,
        retained_fragment_atom_names=retained_fragment_atom_names,
        reconstruction_atom_names=planned_reconstruction_atom_names,
    )
    reconstruction_layers = reconstruction_placement_layers(
        template,
        reconstruction_atom_names=planned_reconstruction_atom_names,
        frontier_bonds=frontier_bonds,
    )
    reachable_atom_names = {
        atom_name for layer in reconstruction_layers for atom_name in layer.atom_names
    }
    unreachable_atom_names = tuple(
        atom_name
        for atom_name in planned_reconstruction_atom_names
        if atom_name not in reachable_atom_names
    )
    return ComponentReconstructionPlan(
        residue=residue,
        idealized_component=idealized_component,
        template=template,
        fragment_match=fragment_match,
        parent_mapping=build_parent_atom_mapping(
            idealized_component,
            parent_template=parent_template,
            exclude_atom_names=exclude_atom_names,
        ),
        retained_fragment_atom_names=retained_fragment_atom_names,
        structural_anchor_atom_names=structural_anchor_atom_names,
        orphan_atom_names=orphan_atom_names,
        reconstruction_atom_names=planned_reconstruction_atom_names,
        frontier_bonds=frontier_bonds,
        reconstruction_layers=reconstruction_layers,
        unreachable_atom_names=unreachable_atom_names,
    )


def build_parent_atom_mapping(
    idealized_component: IdealizedComponent,
    *,
    parent_template: ResidueTemplate | None,
    exclude_atom_names: Collection[str] = (),
) -> ParentAtomMapping | None:
    """Build the canonical atom-level mapping to one optional parent template."""

    if (
        idealized_component.lineage_parent_component_id is None
        or parent_template is None
    ):
        return None

    excluded_atom_names = normalized_atom_name_set(exclude_atom_names)
    component_atom_names = tuple(
        atom.atom_name
        for atom in idealized_component.heavy_atoms()
        if atom.atom_name not in excluded_atom_names
    )
    parent_atom_names = tuple(
        atom_name
        for atom_name in parent_template.ordered_atom_names()
        if atom_name not in excluded_atom_names
    )
    parent_atom_name_set = set(parent_atom_names)
    component_atom_name_set = set(component_atom_names)
    shared_atom_names = tuple(
        atom_name
        for atom_name in component_atom_names
        if atom_name in parent_atom_name_set
    )
    component_only_atom_names = tuple(
        atom_name
        for atom_name in component_atom_names
        if atom_name not in parent_atom_name_set
    )
    parent_only_atom_names = tuple(
        atom_name
        for atom_name in parent_atom_names
        if atom_name not in component_atom_name_set
    )
    return ParentAtomMapping(
        parent_standard_id=idealized_component.lineage_parent_component_id,
        shared_atom_names=shared_atom_names,
        parent_only_atom_names=parent_only_atom_names,
        component_only_atom_names=component_only_atom_names,
    )


def planned_orphan_atom_names(
    fragment_match: ResidueFragmentMatch,
    *,
    orphan_fragment_policy: OrphanFragmentPolicy,
) -> tuple[str, ...]:
    """Return orphan atoms that should join the reconstruction target set."""

    if orphan_fragment_policy is OrphanFragmentPolicy.PRESERVE:
        return ()

    return fragment_match.orphan_atom_names(
        preferred_anchor_atom_names=PRIMARY_NONSTANDARD_ANCHOR_ATOM_NAMES,
    )


def plan_reconstruction_atom_names(
    template: ResidueTemplate,
    *,
    missing_atom_names: Collection[str],
    orphan_atom_names: Collection[str],
) -> tuple[str, ...]:
    """Return template-ordered atoms that must be reconstructed."""

    reconstruction_atom_name_set = normalized_atom_name_set(missing_atom_names)
    reconstruction_atom_name_set.update(normalized_atom_name_set(orphan_atom_names))
    return tuple(
        atom_name
        for atom_name in template.ordered_atom_names()
        if atom_name in reconstruction_atom_name_set
    )


def reconstruction_frontier_bonds(
    template: ResidueTemplate,
    *,
    retained_fragment_atom_names: Collection[str],
    reconstruction_atom_names: Collection[str],
) -> tuple[FragmentBoundary, ...]:
    """Return direct bonds crossing from the retained fragment into reconstruction."""

    retained_atom_name_set = normalized_atom_name_set(retained_fragment_atom_names)
    reconstruction_atom_name_set = normalized_atom_name_set(reconstruction_atom_names)
    boundaries: list[FragmentBoundary] = []
    for present_atom_name in template.ordered_atom_names():
        if present_atom_name not in retained_atom_name_set:
            continue

        for missing_atom_name in template.ordered_atom_names():
            if missing_atom_name not in reconstruction_atom_name_set:
                continue

            if not template.definition.bonded_atom_names(present_atom_name).issuperset(
                {missing_atom_name}
            ):
                continue

            boundaries.append(
                FragmentBoundary(
                    present_atom_name=present_atom_name,
                    missing_atom_name=missing_atom_name,
                )
            )

    return tuple(boundaries)


def reconstruction_placement_layers(
    template: ResidueTemplate,
    *,
    reconstruction_atom_names: Collection[str],
    frontier_bonds: Collection[FragmentBoundary],
) -> tuple[ReconstructionPlacementLayer, ...]:
    """Return breadth-first placement layers over one reconstruction subgraph."""

    ordered_atom_names = template.ordered_atom_names()
    atom_name_order = {
        atom_name: index for index, atom_name in enumerate(ordered_atom_names)
    }
    reconstruction_atom_name_set = normalized_atom_name_set(reconstruction_atom_names)
    frontier_atom_names = tuple(
        atom_name
        for atom_name in ordered_atom_names
        if atom_name in {boundary.missing_atom_name for boundary in frontier_bonds}
    )
    if not frontier_atom_names:
        return ()

    distance_by_atom_name: dict[str, int] = {
        atom_name: 1 for atom_name in frontier_atom_names
    }
    queue: deque[str] = deque(frontier_atom_names)
    while queue:
        current_atom_name = queue.popleft()
        current_distance = distance_by_atom_name[current_atom_name]
        for neighbor_atom_name in template.definition.bonded_atom_names(
            current_atom_name
        ):
            if neighbor_atom_name not in reconstruction_atom_name_set:
                continue

            if neighbor_atom_name in distance_by_atom_name:
                continue

            distance_by_atom_name[neighbor_atom_name] = current_distance + 1
            queue.append(neighbor_atom_name)

    layer_atom_names: dict[int, list[str]] = {}
    for atom_name, hop_distance in sorted(
        distance_by_atom_name.items(),
        key=lambda item: atom_name_order[item[0]],
    ):
        layer_atom_names.setdefault(hop_distance, []).append(atom_name)

    return tuple(
        ReconstructionPlacementLayer(
            hop_distance=hop_distance,
            atom_names=tuple(atom_names),
        )
        for hop_distance, atom_names in sorted(layer_atom_names.items())
    )


def normalized_atom_name_set(atom_names: Collection[str]) -> set[str]:
    """Return a normalized uppercase atom-name set."""

    return {atom_name.strip().upper() for atom_name in atom_names if atom_name.strip()}
