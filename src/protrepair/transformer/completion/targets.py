"""Completion target derivation over normalized residues and templates."""

from protrepair.chemistry import ResidueTemplate
from protrepair.structure.constitution import ResidueSite, StructureConstitution
from protrepair.structure.labels import ResidueId

BACKBONE_ATOM_NAMES: tuple[str, ...] = ("N", "CA", "C")
TERMINAL_EXCLUDED_ATOM_NAMES: frozenset[str] = frozenset({"OXT"})


def require_non_water_hydrogen_targets(
    constitution: StructureConstitution, residue_ids: frozenset[ResidueId] | None
) -> None:
    """Reject an explicit request to prepare a water as a polymer or ligand."""
    if residue_ids is not None:
        for water in constitution.waters:
            if water.residue_id in residue_ids:
                raise ValueError(
                    f"hydrogen completion does not support retained water {water.residue_id.display_token()}"
                )


def missing_atoms_for_completion(
    residue: ResidueSite,
    template: ResidueTemplate,
) -> tuple[str, ...]:
    """Return template atoms missing from one residue, excluding terminal OXT."""

    return template.missing_atom_names(
        residue.atom_site_names(),
        exclude_atom_names=TERMINAL_EXCLUDED_ATOM_NAMES,
    )
