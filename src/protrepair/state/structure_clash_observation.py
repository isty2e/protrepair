"""Shared clash-observation helpers for structure state facts."""

from collections.abc import Iterable

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.clash_pair_generation import ContactDomain
from protrepair.diagnostics.clashes import StericClash
from protrepair.state.structure_axes import ClashPresenceState
from protrepair.state.structure_readiness import StructureChemistryReadinessFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId

__all__ = [
    "active_chemistry_readiness_facts",
    "clash_state_from_count",
    "heavy_heavy_clashes",
    "polymer_ligand_clashes",
    "protein_self_clashes",
    "total_clash_overlap_angstrom",
    "worst_clash_overlap_angstrom",
]


def active_chemistry_readiness_facts(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    chemistry_readiness_facts: "StructureChemistryReadinessFacts | None",
) -> "StructureChemistryReadinessFacts":
    """Return chemistry-readiness facts aligned with one structure carrier."""

    if chemistry_readiness_facts is None:
        return StructureChemistryReadinessFacts.from_structure(
            structure,
            component_library=component_library,
        )
    if chemistry_readiness_facts.carrier is not structure:
        raise ValueError(
            "clash observation requires chemistry-readiness facts for the same "
            "structure"
        )
    return chemistry_readiness_facts


def clash_state_from_count(clash_count: int) -> ClashPresenceState:
    """Return one clash state for a non-negative clash count."""

    return (
        ClashPresenceState.PRESENT
        if clash_count > 0
        else ClashPresenceState.NONE
    )


def worst_clash_overlap_angstrom(clashes: Iterable[StericClash]) -> float:
    """Return the maximum overlap across one clash iterable."""

    worst_overlap = 0.0
    for clash in clashes:
        if clash.overlap_angstrom > worst_overlap:
            worst_overlap = clash.overlap_angstrom
    return worst_overlap


def total_clash_overlap_angstrom(clashes: Iterable[StericClash]) -> float:
    """Return the total overlap across one clash iterable."""

    return sum(clash.overlap_angstrom for clash in clashes)


def protein_self_clashes(
    clashes: Iterable[StericClash],
) -> tuple[StericClash, ...]:
    """Return polymer-only clashes from one raw clash iterable."""

    return tuple(
        clash
        for clash in clashes
        if clash.left_domain is ContactDomain.POLYMER
        and clash.right_domain is ContactDomain.POLYMER
    )


def heavy_heavy_clashes(
    structure: ProteinStructure,
    clashes: Iterable[StericClash],
) -> tuple[StericClash, ...]:
    """Return clashes whose endpoints are both non-hydrogen atoms."""

    return tuple(
        clash
        for clash in clashes
        if _clash_endpoint_element(
            structure,
            residue_id=clash.left_residue_id,
            atom_name=clash.left_atom_name,
        )
        != "H"
        and _clash_endpoint_element(
            structure,
            residue_id=clash.right_residue_id,
            atom_name=clash.right_atom_name,
        )
        != "H"
    )


def _clash_endpoint_element(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    atom_name: str,
) -> str:
    """Return the element for one clash endpoint."""

    return structure.constitution.atom_site_at(
        structure.constitution.atom_index_in_residue(
            structure.constitution.residue_index(residue_id),
            atom_name,
        )
    ).element


def polymer_ligand_clashes(
    clashes: Iterable[StericClash],
) -> tuple[StericClash, ...]:
    """Return polymer-ligand clashes from one raw clash iterable."""

    return tuple(
        clash
        for clash in clashes
        if {clash.left_domain, clash.right_domain}
        == {ContactDomain.POLYMER, ContactDomain.RETAINED_NON_POLYMER}
    )
