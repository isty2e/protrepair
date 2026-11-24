"""Shared heavy-atom completion planning policies."""

from typing import Protocol

from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHeavyAtomSemantics,
    ResidueTemplate,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.bound_peptide_ports import active_peptide_ports
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId


class ResidueRefinementTargeting(Protocol):
    """Minimal residue-targeting surface for refinement-gated repair policy."""

    def targets_residue(self, residue_id: ResidueId) -> bool:
        """Return whether the directive targets one residue."""

        ...


def should_add_terminal_oxt(
    residue: ResidueSite,
    *,
    structure: ProteinStructure,
    component_library: ComponentLibrary,
) -> bool:
    """Check whether a chain-end residue can receive OXT.

    Parameters
    ----------
    residue : ResidueSite
        Chain-end heavy inventory, including any atoms just repaired.
    structure : ProteinStructure
        Current carrier for rebinding the C port to the original occurrence.
    component_library : ComponentLibrary
        Definitions in effect for this repair.

    Returns
    -------
    bool
        Whether the end is unreserved and has the template and anchors for OXT.
    """

    if residue.has_atom_site("OXT"):
        return False
    ports = active_peptide_ports()
    if ports is not None and ports.reserves(
        structure, AtomRef(residue.residue_id, "C")
    ):
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
