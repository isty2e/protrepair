"""Completion target derivation over normalized residues and templates."""

from protrepair.chemistry import ResidueTemplate
from protrepair.structure.constitution import ResidueSite

BACKBONE_ATOM_NAMES: tuple[str, ...] = ("N", "CA", "C")
TERMINAL_EXCLUDED_ATOM_NAMES: frozenset[str] = frozenset({"OXT"})


def missing_atoms_for_completion(
    residue: ResidueSite,
    template: ResidueTemplate,
) -> tuple[str, ...]:
    """Return template atoms missing from one residue, excluding terminal OXT."""

    return template.missing_atom_names(
        residue.atom_site_names(),
        exclude_atom_names=TERMINAL_EXCLUDED_ATOM_NAMES,
    )
