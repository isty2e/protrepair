"""Run-local geometry usable after reconstructing selected backbone sites."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from protrepair.geometry import RigidTransform, Vec3
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef, ResidueId

if TYPE_CHECKING:
    from protrepair.structure.aggregate import ProteinStructure


@dataclass(frozen=True, slots=True)
class BackbonePlacement:
    """Backbone atoms assessed together, and the positive points supporting the fit."""

    component: str
    targets: frozenset[str]
    positions: Mapping[AtomRef, Vec3]
    reason: str | None

    def unresolved(
        self, structure: "ProteinStructure", origin: "ProteinStructure"
    ) -> bool:
        if self.reason is not None:
            return True
        observed = []
        modeled = []
        moved = False
        for ref, point in self.positions.items():
            index = structure.constitution.resolve_atom_index(ref)
            original_index = origin.constitution.resolve_atom_index(ref)
            if index is None or original_index is None:
                return True
            geometry = structure.geometry.atom_geometry(index)
            original = origin.geometry.atom_geometry(original_index)
            if (original.occupancy > 0) != (
                geometry.occupancy > 0
            ) or original.altloc != geometry.altloc:
                return True
            (observed if original.occupancy > 0 else modeled).append(
                (point, geometry.position)
            )
            moved |= point.distance_to(geometry.position) > 0.002
        if not moved:
            return False
        # Ignore a rigid change of reference frame, but do not reuse a fit on
        # a different arrangement of its measured anchors.
        if len(observed) < 3:
            return True
        transform = RigidTransform.fit(
            np.array([p.to_array() for p, _ in observed]),
            np.array([q.to_array() for _, q in observed]),
        )
        return transform is None or any(
            Vec3.coerce(transform.apply(p)).distance_to(q) > 0.002
            for p, q in (*observed, *modeled)
        )


@dataclass(frozen=True, slots=True)
class BackboneAvailability:
    """Branch-local view of coordinate usability, keyed by immutable atom identities."""

    source: "ProteinStructure"
    placements: Mapping[ResidueId, BackbonePlacement]

    def placement_for(
        self, structure: "ProteinStructure", rid: ResidueId
    ) -> BackbonePlacement | None:
        placement = self.placements.get(rid)
        if placement is None:
            return None
        ingress = structure.provenance.ingress
        original = self.source.provenance.ingress
        if ingress is not original and (
            original.observation is None
            or ingress.observation is not original.observation
        ):
            return None
        constitution = structure.constitution
        site = constitution.residue_or_ligand(rid)
        return (
            placement
            if site is not None
            and site.component_id == placement.component
            and constitution.role_at(constitution.residue_index(rid))
            is ResidueRole.POLYMER
            else None
        )

    def unresolved(self, structure: "ProteinStructure", rid: ResidueId) -> bool:
        placement = self.placement_for(structure, rid)
        if placement is None:
            return False
        if structure is self.source:
            return placement.reason is not None
        if placement.unresolved(structure, self.source):
            return True
        for ref in placement.positions:
            origin = self.source.constitution.residue_or_ligand(ref.residue_id)
            current = structure.constitution.residue_or_ligand(ref.residue_id)
            if (
                origin is None
                or current is None
                or origin.component_id != current.component_id
                or structure.constitution.role_at(
                    structure.constitution.residue_index(ref.residue_id)
                )
                is not ResidueRole.POLYMER
            ):
                return True
        return False


_ACTIVE: ContextVar[BackboneAvailability | None] = ContextVar(
    "backbone_availability", default=None
)


def active_backbone_availability() -> BackboneAvailability | None:
    return _ACTIVE.get()


def unresolved_backbone(structure: "ProteinStructure", rid: ResidueId) -> bool:
    binding = _ACTIVE.get()
    return binding is not None and binding.unresolved(structure, rid)


def unresolved_atom_names(
    structure: "ProteinStructure", rid: ResidueId
) -> frozenset[str]:
    binding = _ACTIVE.get()
    if binding is None or not binding.unresolved(structure, rid):
        return frozenset()
    placement = binding.placements[rid]
    site = structure.constitution.residue_site_at(
        structure.constitution.residue_index(rid)
    )
    geometry = structure.residue_geometry(structure.constitution.residue_index(rid))
    return placement.targets | frozenset(
        a.name for a in site.atom_sites if geometry.atom_geometry(a.name).occupancy == 0
    )


def unresolved_atom_refs(structure: "ProteinStructure") -> frozenset[AtomRef]:
    """Return coordinates to leave out of a whole-structure contact search."""
    binding = _ACTIVE.get()
    if binding is None:
        return frozenset()
    return frozenset(
        AtomRef(rid, name)
        for rid in binding.placements
        for name in unresolved_atom_names(structure, rid)
    )


def eligible_geometry_targets(
    structure: "ProteinStructure", requested: frozenset[ResidueId] | None
) -> frozenset[ResidueId] | None:
    binding = _ACTIVE.get()
    if binding is None:
        return requested
    blocked = {rid for rid in binding.placements if binding.unresolved(structure, rid)}
    if not blocked:
        return requested
    return (
        requested
        if requested is not None
        else frozenset(s.residue_id for s in structure.constitution.iter_residues())
    ) - blocked


@contextmanager
def backbone_repair_run(binding: BackboneAvailability | None) -> Iterator[None]:
    token = _ACTIVE.set(binding)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
