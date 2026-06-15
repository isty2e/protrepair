"""Local bond-planning support resolution for continuous execution regions."""

from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    retained_non_polymer_rdkit_fallback_heavy_bond_definitions,
    retained_non_polymer_rdkit_fallback_supports_passive_context,
)
from protrepair.chemistry.single_atom_inorganic import is_single_atom_inorganic_residue
from protrepair.errors import RdkitUnavailableError
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot


class LocalBondPlanningSupportMode(Enum):
    """Continuous local bond-planning support mode for one included residue."""

    TEMPLATE = "template"
    TEMPLATE_LESS_PASSIVE_CONTEXT = "template_less_passive_context"
    NONBONDING_PASSIVE_CONTEXT = "nonbonding_passive_context"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class LocalBondPlanningSupportResolution:
    """Resolved local bond-planning support for one included residue."""

    mode: LocalBondPlanningSupportMode
    fallback_bond_definitions: tuple[BondDefinition, ...] = ()
    blocker_message: str = ""

    def __post_init__(self) -> None:
        if (
            self.mode
            is not LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
            and self.fallback_bond_definitions
        ):
            raise ValueError(
                "fallback bond definitions require template-less passive-context "
                "support"
            )
        if (
            self.mode is LocalBondPlanningSupportMode.UNSUPPORTED
            and not self.blocker_message
        ):
            return

        if (
            self.mode is not LocalBondPlanningSupportMode.UNSUPPORTED
            and self.blocker_message
        ):
            raise ValueError("only unsupported local bond planning may carry blockers")

    def supports_local_bond_planning(self) -> bool:
        """Return whether this residue admits continuous local bond planning."""

        return self.mode is not LocalBondPlanningSupportMode.UNSUPPORTED


def resolve_local_bond_planning_support(
    snapshot: ProteinStructureSnapshot,
    residue_index: ResidueIndex,
    residue_site: ResidueSite,
    *,
    movable_atom_indices: tuple[AtomIndex, ...],
    component_library: ComponentLibrary,
) -> LocalBondPlanningSupportResolution:
    """Resolve local bond-planning support for one included residue."""

    if component_library.get(residue_site.component_id) is not None:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.TEMPLATE
        )

    constitution = snapshot.structure.constitution
    movable_atom_index_set = set(movable_atom_indices)
    residue_has_movable_atom = any(
        constitution.atom_index_in_residue(residue_index, atom_site.name)
        in movable_atom_index_set
        for atom_site in residue_site.atom_sites
    )
    if is_single_atom_inorganic_residue(residue_site):
        if residue_has_movable_atom:
            return LocalBondPlanningSupportResolution(
                mode=LocalBondPlanningSupportMode.UNSUPPORTED,
                blocker_message=(
                    "continuous relaxation treats single-atom metal/ion residues "
                    "as nonbonding fixed context, not movable chemistry: "
                    f"{residue_site.component_id} "
                    f"({residue_site.residue_id.display_token()})"
                ),
            )

        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
        )

    if residue_has_movable_atom:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.UNSUPPORTED
        )

    heavy_atom_sites = tuple(
        atom_site for atom_site in residue_site.atom_sites if atom_site.element != "H"
    )
    if len(heavy_atom_sites) <= 1:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
        )

    if not residue_site.is_hetero:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.UNSUPPORTED
        )

    residue_geometry = snapshot.structure.residue_geometry(residue_index)
    formal_charge_by_atom_name = dict(
        snapshot.structure.residue_formal_charge_by_atom_name(residue_index)
    )
    try:
        if not retained_non_polymer_rdkit_fallback_supports_passive_context(
            residue_site,
            residue_geometry,
            formal_charge_by_atom_name=formal_charge_by_atom_name,
        ):
            return LocalBondPlanningSupportResolution(
                mode=LocalBondPlanningSupportMode.UNSUPPORTED
            )

        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT,
            fallback_bond_definitions=(
                retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
                    residue_site,
                    residue_geometry,
                    formal_charge_by_atom_name=formal_charge_by_atom_name,
                )
            ),
        )
    except (RdkitUnavailableError, RuntimeError, ValueError):
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.UNSUPPORTED
        )
