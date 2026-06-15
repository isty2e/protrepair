"""Deterministic semantic summaries for permanent regression tests."""

import hashlib
from collections import Counter
from dataclasses import dataclass

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
                    "|".join(
                        (
                            "ATOM",
                            residue.residue_id.display_token(),
                            residue.component_id,
                            atom_site.name,
                            atom_site.element,
                            f"{atom_geometry.position.x:.3f}",
                            f"{atom_geometry.position.y:.3f}",
                            f"{atom_geometry.position.z:.3f}",
                        )
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
                "|".join(
                    (
                        "HETATM",
                        ligand.residue_id.display_token(),
                        ligand.component_id,
                        atom_site.name,
                        atom_site.element,
                        f"{atom_geometry.position.x:.3f}",
                        f"{atom_geometry.position.y:.3f}",
                        f"{atom_geometry.position.z:.3f}",
                    )
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


def format_residue_token(residue: ResidueSite) -> str:
    """Return a compact residue token for stable summaries."""

    hetero_flag = "HET" if residue.is_hetero else "ATOM"
    return (
        f"{hetero_flag}:{residue.component_id}:"
        f"{residue.residue_id.display_token()}"
    )
