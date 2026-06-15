"""Blueprint coverage relation tests."""

from tests.support.canonical_builders import (
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.io import FileFormat
from protrepair.relation.blueprint import StructureBlueprintCoverage
from protrepair.relation.sequence_alignment import (
    ObservedChainSequence,
    SequenceAlignmentRelation,
)
from protrepair.sources import UniProtSequenceRecord, UniProtSequenceReference
from protrepair.sources.projection import align_observed_chain_to_uniprot_record
from protrepair.structure import (
    ChainSite,
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
    ResidueSite,
)
from protrepair.structure.labels import ResidueId


def test_structure_blueprint_coverage_derives_internal_gap_and_mapping() -> None:
    """Coverage should distinguish realized mappings from one internal missing span."""

    chain = ChainSite(
        chain_id="A",
        residues=(
            ResidueSite(
                component_id="ALA",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atom_sites=(),
            ),
            ResidueSite(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=3),
                atom_sites=(),
            ),
        ),
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(chain),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACG",
        ),
    )

    coverage = StructureBlueprintCoverage.from_alignment(alignment)

    assert tuple(
        mapping.sequence_position for mapping in coverage.residue_mappings
    ) == (1, 3)
    assert tuple(mapping.relation for mapping in coverage.residue_mappings) == (
        SequenceAlignmentRelation.MATCH,
        SequenceAlignmentRelation.MATCH,
    )

    assert len(coverage.coverage_gaps) == 1
    gap = coverage.coverage_gaps[0]
    assert gap.absent_sequence_positions == (2,)
    assert gap.preceding_residue_id == ResidueId("A", 1)
    assert gap.following_residue_id == ResidueId("A", 3)
    assert gap.is_internal()


def test_structure_blueprint_coverage_represents_terminal_gap_without_lowering() -> (
    None
):
    """Coverage should keep terminal blueprint gaps as canonical relation truth."""

    chain = ChainSite(
        chain_id="A",
        residues=(
            ResidueSite(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=2),
                atom_sites=(),
            ),
            ResidueSite(
                component_id="SER",
                residue_id=ResidueId(chain_id="A", seq_num=3),
                atom_sites=(),
            ),
        ),
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(chain),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="AGS",
        ),
    )

    coverage = StructureBlueprintCoverage.from_alignment(alignment)

    assert len(coverage.coverage_gaps) == 1
    gap = coverage.coverage_gaps[0]
    assert gap.preceding_residue_id is None
    assert gap.following_residue_id == ResidueId("A", 2)
    assert gap.absent_sequence_positions == (1,)
    assert not gap.is_internal()


def test_structure_blueprint_coverage_keeps_nonlowerable_internal_gap_canonical() -> (
    None
):
    """Coverage should preserve internal gaps without inferring structure numbering."""

    chain = ChainSite(
        chain_id="A",
        residues=(
            ResidueSite(
                component_id="ALA",
                residue_id=ResidueId(chain_id="A", seq_num=10),
                atom_sites=(),
            ),
            ResidueSite(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=20),
                atom_sites=(),
            ),
        ),
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(chain),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDG",
        ),
    )

    coverage = StructureBlueprintCoverage.from_alignment(alignment)

    assert len(coverage.coverage_gaps) == 1
    gap = coverage.coverage_gaps[0]
    assert gap.is_internal()
    assert gap.absent_sequence_positions == (2, 3)
    assert gap.preceding_residue_id == ResidueId("A", 10)
    assert gap.following_residue_id == ResidueId("A", 20)


def test_structure_blueprint_coverage_from_structure_uses_attached_blueprint() -> None:
    """Coverage should be derivable directly from one structure's attached blueprint."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(),
                    ),
                    residue_payload(
                        component_id="GLU",
                        residue_id=ResidueId("A", 4),
                        atoms=(),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        polymer_blueprint=PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id="A",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="C"),
                        PolymerResidueSlot(sequence_position=3, token="D"),
                        PolymerResidueSlot(sequence_position=4, token="E"),
                    ),
                ),
            )
        ),
    )

    coverage = StructureBlueprintCoverage.from_structure(structure, "A")

    assert tuple(
        mapping.sequence_position for mapping in coverage.residue_mappings
    ) == (
        1,
        4,
    )
    assert len(coverage.coverage_gaps) == 1
    gap = coverage.coverage_gaps[0]
    assert gap.absent_sequence_positions == (2, 3)
    assert gap.preceding_residue_id == ResidueId("A", 1)
    assert gap.following_residue_id == ResidueId("A", 4)
