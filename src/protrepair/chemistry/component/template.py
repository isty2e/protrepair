"""Residue-template aggregates over chemistry graphs and completion semantics."""

from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from protrepair.chemistry.component.graph import (
    ChemicalComponentDefinition,
    ForceFieldAtomParams,
    TetrahedralCenterSemantics,
)
from protrepair.chemistry.component.semantics import (
    HeavyAtomSemantics,
    HydrogenSemantics,
    IdealGeometryHeavyAtomSemantics,
    IdealGeometryHydrogenSemantics,
)


@dataclass(frozen=True, slots=True)
class ResidueTemplate:
    """Residue template combining chemistry graph and completion semantics."""

    definition: ChemicalComponentDefinition
    lineage_parent_component_id: str | None = None
    backbone_family_component_id: str = ""
    forcefield_parameters: Mapping[str, ForceFieldAtomParams] = field(
        default_factory=dict
    )
    preferred_atom_order: tuple[str, ...] = ()
    heavy_atom_semantics: (
        HeavyAtomSemantics | IdealGeometryHeavyAtomSemantics | None
    ) = None
    hydrogen_semantics: HydrogenSemantics | IdealGeometryHydrogenSemantics | None = None
    tetrahedral_centers: tuple[TetrahedralCenterSemantics, ...] = ()

    def __post_init__(self) -> None:
        normalized_params = {
            atom_name.strip().upper(): params
            for atom_name, params in self.forcefield_parameters.items()
        }
        preferred_atom_order = tuple(
            atom_name.strip().upper() for atom_name in self.preferred_atom_order
        )
        lineage_parent_component_id = self.lineage_parent_component_id
        if lineage_parent_component_id is not None:
            lineage_parent_component_id = (
                lineage_parent_component_id.strip().upper() or None
            )

        backbone_family_component_id = (
            self.backbone_family_component_id.strip().upper()
            if self.backbone_family_component_id.strip()
            else self.definition.component_id
        )

        object.__setattr__(
            self,
            "lineage_parent_component_id",
            lineage_parent_component_id,
        )
        object.__setattr__(
            self,
            "backbone_family_component_id",
            backbone_family_component_id,
        )
        object.__setattr__(
            self,
            "forcefield_parameters",
            MappingProxyType(normalized_params),
        )
        object.__setattr__(self, "preferred_atom_order", preferred_atom_order)
        object.__setattr__(
            self,
            "tetrahedral_centers",
            tuple(self.tetrahedral_centers),
        )

    def ordered_atom_names(self) -> tuple[str, ...]:
        """Return the preferred atom order for serialization or comparison."""

        return self.preferred_atom_order or self.definition.expected_atom_names()

    @property
    def component_id(self) -> str:
        """Return the canonical component identifier for the template."""

        return self.definition.component_id

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return all aliases accepted for this template."""

        return self.definition.aliases

    def expected_atom_names(self) -> tuple[str, ...]:
        """Return the expected canonical atom order for the component."""

        return self.definition.expected_atom_names()

    def expected_heavy_atom_names(self) -> tuple[str, ...]:
        """Return expected heavy atoms for completeness-sensitive workflows."""

        if isinstance(self.heavy_atom_semantics, HeavyAtomSemantics):
            return self.heavy_atom_semantics.atom_order

        if isinstance(
            self.heavy_atom_semantics,
            IdealGeometryHeavyAtomSemantics,
        ):
            return self.heavy_atom_semantics.component.heavy_atom_names()

        return tuple(
            atom_name for atom_name in self.expected_atom_names() if atom_name != "OXT"
        )

    def expected_hydrogen_atom_names(self) -> tuple[str, ...]:
        """Return expected hydrogen atoms for hydrogen-completeness workflows."""

        hydrogen_semantics = self.hydrogen_semantics
        if hydrogen_semantics is None:
            return ()

        return hydrogen_semantics.expected_hydrogen_atom_names()

    def missing_atom_names(
        self,
        present_atom_names: Collection[str],
        *,
        exclude_atom_names: Collection[str] = (),
    ) -> tuple[str, ...]:
        """Return missing template atoms for a given present-atom set."""

        present = {atom_name.strip().upper() for atom_name in present_atom_names}
        excluded = {atom_name.strip().upper() for atom_name in exclude_atom_names}
        return tuple(
            atom_name
            for atom_name in self.expected_atom_names()
            if atom_name not in present and atom_name not in excluded
        )

    def can_repair_heavy_atoms(self) -> bool:
        """Return whether heavy-atom repair semantics are available."""

        return self.heavy_atom_semantics is not None

    def can_add_hydrogens(self) -> bool:
        """Return whether hydrogen-placement semantics are available."""

        return self.hydrogen_semantics is not None

    def template_hydrogen_anchor_by_name(
        self,
        present_atom_names: Collection[str],
    ) -> Mapping[str, str]:
        """Return explicit template-resolved anchors for present hydrogens."""

        normalized_present_atom_names = {
            atom_name.strip().upper() for atom_name in present_atom_names
        }
        hydrogen_semantics = self.hydrogen_semantics
        if hydrogen_semantics is None:
            return MappingProxyType({})

        anchor_by_name = hydrogen_semantics.template_hydrogen_anchor_by_name()
        return MappingProxyType(
            {
                hydrogen_atom_name: anchor_atom_name
                for hydrogen_atom_name, anchor_atom_name in anchor_by_name.items()
                if hydrogen_atom_name in normalized_present_atom_names
            }
        )

    def requires_coordinate_hydrogen_topology(
        self,
        present_atom_names: Collection[str],
    ) -> bool:
        """Return whether present hydrogens need coordinate-derived attachment."""

        normalized_present_hydrogen_atom_names = {
            atom_name.strip().upper()
            for atom_name in present_atom_names
            if atom_name.strip()
        }
        if not normalized_present_hydrogen_atom_names:
            return False

        template_resolved_hydrogen_atom_names = set(
            self.template_hydrogen_anchor_by_name(
                normalized_present_hydrogen_atom_names
            ).keys()
        )
        return any(
            hydrogen_atom_name not in template_resolved_hydrogen_atom_names
            for hydrogen_atom_name in normalized_present_hydrogen_atom_names
        )

    def has_tetrahedral_stereochemistry(self) -> bool:
        """Return whether side-chain tetrahedral expectations are defined."""

        return bool(self.tetrahedral_centers)

    def tetrahedral_center(
        self,
        *,
        center_atom_name: str,
        ordered_neighbor_atom_names: tuple[str, str, str],
    ) -> TetrahedralCenterSemantics | None:
        """Return the matching tetrahedral-center semantics when available."""

        for center_semantics in self.tetrahedral_centers:
            if center_semantics.matches(
                center_atom_name=center_atom_name,
                ordered_neighbor_atom_names=ordered_neighbor_atom_names,
            ):
                return center_semantics

        return None

    def correction_subtree_atom_names(
        self,
        center_semantics: TetrahedralCenterSemantics,
    ) -> tuple[str, ...]:
        """Return subtree atoms that should be rebuilt for one center correction."""

        center_atom_name = center_semantics.center_atom_name
        branch_root_atom_names = center_semantics.ordered_neighbor_atom_names[1:]
        atom_names_to_correct: set[str] = set()
        for branch_root_atom_name in branch_root_atom_names:
            queue: deque[str] = deque([branch_root_atom_name])
            visited = {center_atom_name}
            while queue:
                atom_name = queue.popleft()
                if atom_name in visited:
                    continue

                visited.add(atom_name)
                atom_names_to_correct.add(atom_name)
                for neighbor_atom_name in self.definition.bonded_atom_names(atom_name):
                    if neighbor_atom_name not in visited:
                        queue.append(neighbor_atom_name)

        return tuple(
            atom_name
            for atom_name in self.ordered_atom_names()
            if atom_name in atom_names_to_correct
        )

    def has_forcefield_params(self, atom_name: str) -> bool:
        """Return whether force-field parameters exist for a named atom."""

        return atom_name.strip().upper() in self.forcefield_parameters

    def bond_hop_distance(
        self,
        atom_name_1: str,
        atom_name_2: str,
        *,
        max_hops: int | None = None,
    ) -> int | None:
        """Return the heavy-atom bond hop distance between two named atoms."""

        return self.definition.bond_hop_distance(
            atom_name_1,
            atom_name_2,
            max_hops=max_hops,
        )
