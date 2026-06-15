"""Tests for deterministic structure summary semantics."""

from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.structure_summary import summarize_structure

from protrepair.geometry import Vec3
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat


def test_structure_summary_keeps_polymer_residue_count_orthogonal_to_ligands() -> None:
    """Representative summaries should not count ligands as polymer residues."""

    structure = build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="HIS",
                        residue_id=ResidueId(chain_id="A", seq_num=2),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 1.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 1.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 1.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 1.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 2.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="FAD",
                residue_id=ResidueId(chain_id="A", seq_num=1058),
                atoms=(
                    atom_payload("C1", "C", Vec3(10.0, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(11.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(12.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="summary-polymer-ligand-orthogonality",
    )

    summary = summarize_structure(structure)

    assert summary.residue_count == 2
    assert summary.ligand_count == 1
    assert summary.first_residue == "ATOM:GLY:A:1"
    assert summary.last_residue == "ATOM:HIS:A:2"


def test_structure_summary_reports_ligand_only_structures_without_polymer_range() -> (
    None
):
    """Ligand-only structures should not fabricate a polymer residue range."""

    structure = build_canonical_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="FAD",
                residue_id=ResidueId(chain_id="A", seq_num=1058),
                atoms=(
                    atom_payload("C1", "C", Vec3(10.0, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(11.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(12.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="summary-ligand-only",
    )

    summary = summarize_structure(structure)

    assert summary.chain_summaries == ()
    assert summary.residue_count == 0
    assert summary.ligand_count == 1
    assert summary.atom_count == 3
    assert summary.first_residue is None
    assert summary.last_residue is None
