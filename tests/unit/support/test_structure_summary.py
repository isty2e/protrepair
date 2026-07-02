"""Tests for deterministic structure summary semantics."""

from dataclasses import replace

from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.structure_summary import (
    semantic_digest_for_structure,
    structure_summaries_match_except_digest,
    structure_summary_mismatch_report,
    summarize_structure,
)

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


def test_structure_summary_mismatch_report_identifies_digest_only_drift() -> None:
    """Digest-only representative drift should be diagnosable as coordinate-only."""

    expected = REPRESENTATIVE_CASES["1afc-hydrogen-his-protonated"].summary
    actual = replace(
        expected,
        semantic_digest="d5d0abbc29444683d4752a1342b49921896f585cca5602bb58b34254169ae0ff",
    )

    assert structure_summaries_match_except_digest(actual, expected)

    report = structure_summary_mismatch_report(actual, expected)

    assert "only semantic_digest differs" in report
    assert "coordinate-only drift" in report
    assert expected.semantic_digest in report
    assert actual.semantic_digest in report


def test_structure_summary_digest_helper_matches_default_summary_precision() -> None:
    """The standalone digest helper should preserve summary digest semantics."""

    structure = build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.004, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.006, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="summary-digest-helper",
    )
    summary = summarize_structure(structure)

    assert semantic_digest_for_structure(structure) == summary.semantic_digest
    assert semantic_digest_for_structure(
        structure,
        coordinate_decimal_places=2,
    ) != summary.semantic_digest
