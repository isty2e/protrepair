"""Deterministic semantic summaries for permanent regression tests."""

import hashlib
from collections import Counter
from dataclasses import dataclass, fields, replace

from protrepair.structure import (
    ProteinStructure,
    ResidueSite,
)


@dataclass(frozen=True, slots=True)
class ChainSummary:
    """Chain-level semantic counts for stable regression assertions."""

    chain_id: str
    residue_count: int
    atom_count: int


@dataclass(frozen=True, slots=True)
class StructureSummary:
    """Deterministic semantic summary of a canonical structure."""

    chain_ids: tuple[str, ...]
    chain_summaries: tuple[ChainSummary, ...]
    residue_count: int
    ligand_count: int
    atom_count: int
    hydrogen_atom_count: int
    component_counts: tuple[tuple[str, int], ...]
    first_residue: str | None
    last_residue: str | None
    semantic_digest: str


def structure_summary_without_digest(summary: StructureSummary) -> StructureSummary:
    """Return one summary with coordinate-sensitive digest removed."""

    return replace(summary, semantic_digest="")


def structure_summaries_match_except_digest(
    actual: StructureSummary,
    expected: StructureSummary,
) -> bool:
    """Return whether two summaries differ only by semantic digest."""

    return structure_summary_without_digest(actual) == structure_summary_without_digest(
        expected
    )


def structure_summary_mismatch_report(
    actual: StructureSummary,
    expected: StructureSummary,
) -> str:
    """Return an actionable representative-summary mismatch report."""

    differing_fields = tuple(
        field.name
        for field in fields(StructureSummary)
        if getattr(actual, field.name) != getattr(expected, field.name)
    )
    if not differing_fields:
        return "structure summaries match"

    lines = ["structure summary mismatch:"]
    if differing_fields == ("semantic_digest",):
        lines.append(
            "only semantic_digest differs; all count/order/component fields match"
        )
        lines.append(
            "this is consistent with coordinate-only drift at digest precision"
        )

    for field_name in differing_fields:
        lines.append(f"{field_name}:")
        lines.append(f"  expected: {getattr(expected, field_name)!r}")
        lines.append(f"  actual:   {getattr(actual, field_name)!r}")

    return "\n".join(lines)


def semantic_digest_for_structure(
    structure: ProteinStructure,
    *,
    coordinate_decimal_places: int = 3,
) -> str:
    """Return the coordinate-sensitive semantic digest for one structure."""

    if coordinate_decimal_places < 0:
        raise ValueError("coordinate_decimal_places must be non-negative")

    return hashlib.sha256(
        "\n".join(
            _semantic_digest_lines(
                structure,
                coordinate_decimal_places=coordinate_decimal_places,
            )
        ).encode("utf-8")
    ).hexdigest()


def summarize_structure(structure: ProteinStructure) -> StructureSummary:
    """Summarize a canonical structure into a deterministic semantic digest."""

    component_counts: Counter[str] = Counter()
    chain_summaries: list[ChainSummary] = []
    digest_lines: list[str] = []
    hydrogen_atom_count = 0
    atom_count = 0

    for chain in structure.constitution.chains:
        chain_atom_count = 0
        for residue in chain.residues:
            residue_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(
                    residue.residue_id
                ),
            )

            component_counts[residue.component_id] += 1
            for atom_site in residue.atom_sites:
                atom_geometry = residue_geometry.atom_geometry(atom_site.name)

                atom_count += 1
                chain_atom_count += 1
                if atom_site.element == "H":
                    hydrogen_atom_count += 1

                digest_lines.append(
                    _semantic_digest_line(
                        "ATOM",
                        residue,
                        atom_site.name,
                        atom_site.element,
                        atom_geometry,
                        coordinate_decimal_places=3,
                    )
                )

        chain_summaries.append(
            ChainSummary(
                chain_id=chain.chain_id,
                residue_count=len(chain.residues),
                atom_count=chain_atom_count,
            )
        )

    for ligand in structure.constitution.ligands:
        ligand_geometry = structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(ligand.residue_id),
        )

        component_counts[ligand.component_id] += 1
        for atom_site in ligand.atom_sites:
            atom_geometry = ligand_geometry.atom_geometry(atom_site.name)

            atom_count += 1
            if atom_site.element == "H":
                hydrogen_atom_count += 1

            digest_lines.append(
                _semantic_digest_line(
                    "HETATM",
                    ligand,
                    atom_site.name,
                    atom_site.element,
                    atom_geometry,
                    coordinate_decimal_places=3,
                )
            )

    ordered_residues = tuple(
        structure.constitution.iter_residues(include_ligands=False)
    )
    first_residue = None
    last_residue = None
    if ordered_residues:
        first_residue = format_residue_token(ordered_residues[0])
        last_residue = format_residue_token(ordered_residues[-1])

    semantic_digest = hashlib.sha256(
        "\n".join(digest_lines).encode("utf-8")
    ).hexdigest()

    return StructureSummary(
        chain_ids=structure.chain_ids(),
        chain_summaries=tuple(chain_summaries),
        residue_count=len(ordered_residues),
        ligand_count=len(structure.constitution.ligands),
        atom_count=atom_count,
        hydrogen_atom_count=hydrogen_atom_count,
        component_counts=tuple(sorted(component_counts.items())),
        first_residue=first_residue,
        last_residue=last_residue,
        semantic_digest=semantic_digest,
    )


def _semantic_digest_lines(
    structure: ProteinStructure,
    *,
    coordinate_decimal_places: int,
) -> tuple[str, ...]:
    """Return coordinate digest lines in canonical structure order."""

    digest_lines: list[str] = []
    for chain in structure.constitution.chains:
        for residue in chain.residues:
            residue_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(
                    residue.residue_id
                ),
            )
            for atom_site in residue.atom_sites:
                atom_geometry = residue_geometry.atom_geometry(atom_site.name)
                digest_lines.append(
                    _semantic_digest_line(
                        "ATOM",
                        residue,
                        atom_site.name,
                        atom_site.element,
                        atom_geometry,
                        coordinate_decimal_places=coordinate_decimal_places,
                    )
                )

    for ligand in structure.constitution.ligands:
        ligand_geometry = structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(ligand.residue_id),
        )
        for atom_site in ligand.atom_sites:
            atom_geometry = ligand_geometry.atom_geometry(atom_site.name)
            digest_lines.append(
                _semantic_digest_line(
                    "HETATM",
                    ligand,
                    atom_site.name,
                    atom_site.element,
                    atom_geometry,
                    coordinate_decimal_places=coordinate_decimal_places,
                )
            )

    return tuple(digest_lines)


def _semantic_digest_line(
    record_name: str,
    residue: ResidueSite,
    atom_name: str,
    element: str,
    atom_geometry,
    *,
    coordinate_decimal_places: int,
) -> str:
    """Return one canonical digest line for one atom site."""

    coordinate_format = f"{{:.{coordinate_decimal_places}f}}"
    return "|".join(
        (
            record_name,
            residue.residue_id.display_token(),
            residue.component_id,
            atom_name,
            element,
            coordinate_format.format(atom_geometry.position.x),
            coordinate_format.format(atom_geometry.position.y),
            coordinate_format.format(atom_geometry.position.z),
        )
    )


def format_residue_token(residue: ResidueSite) -> str:
    """Return a compact residue token for stable summaries."""

    hetero_flag = "HET" if residue.is_hetero else "ATOM"
    return (
        f"{hetero_flag}:{residue.component_id}:"
        f"{residue.residue_id.display_token()}"
    )
