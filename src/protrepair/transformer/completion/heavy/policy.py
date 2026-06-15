"""Shared heavy-atom completion planning policies."""

from typing import Protocol

from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHeavyAtomSemantics,
    ResidueTemplate,
)
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId


class ResidueRefinementTargeting(Protocol):
    """Minimal residue-targeting surface for refinement-gated repair policy."""

    def targets_residue(self, residue_id: ResidueId) -> bool:
        """Return whether the directive targets one residue."""

        ...


def should_add_terminal_oxt(
    residue: ResidueSite,
    *,
    component_library: ComponentLibrary,
) -> bool:
    """Return whether a residue can receive terminal OXT augmentation."""

    if residue.has_atom_site("OXT"):
        return False

    template = component_library.get(residue.component_id)
    if template is None:
        return False

    required_atom_names = ("N", "CA", "C", "O")
    return all(residue.has_atom_site(atom_name) for atom_name in required_atom_names)


def allows_refinement_gated_component_repair(
    template: ResidueTemplate,
    *,
    local_refinement: ResidueRefinementTargeting | None,
    residue_id: ResidueId,
) -> bool:
    """Return whether one refinement-gated ideal-geometry path may run."""

    heavy_atom_semantics = template.heavy_atom_semantics
    if not isinstance(heavy_atom_semantics, IdealGeometryHeavyAtomSemantics):
        return True

    if not heavy_atom_semantics.requires_local_refinement():
        return True

    if local_refinement is None:
        return False

    return local_refinement.targets_residue(residue_id)
