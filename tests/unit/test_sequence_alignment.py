"""Observed-to-external sequence alignment tests."""

from protrepair.relation.sequence_alignment import (
    ObservedChainSequence,
    ReferenceSequenceSourceFamily,
    SequenceDifferenceKind,
)
from protrepair.sources import (
    UniProtSequenceFamily,
    UniProtSequenceFamilyFetchResult,
    UniProtSequenceRecord,
    UniProtSequenceReference,
)
from protrepair.sources.projection import (
    align_observed_chain_to_uniprot_family,
    align_observed_chain_to_uniprot_record,
)
from protrepair.structure import (
    ChainSite,
    PolymerChainBlueprint,
    PolymerResidueSlot,
    ResidueSite,
)
from protrepair.structure.labels import (
    ResidueId,
)


def test_observed_chain_sequence_alignment_detects_internal_deletion() -> None:
    """Observed-vs-reference alignment should classify missing internal positions."""

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
    observed_sequence = ObservedChainSequence.from_chain(chain)
    reference_record = UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(accession="P12345"),
        primary_accession="P12345",
        isoform_accession=None,
        sequence="ACG",
    )

    alignment = align_observed_chain_to_uniprot_record(
        observed_sequence,
        reference_record,
    )

    assert alignment.match_count() == 2
    differences = alignment.differences()
    assert len(differences) == 1
    assert differences[0].kind is SequenceDifferenceKind.INTERNAL_DELETION
    assert differences[0].reference_sequence == "C"
    assert differences[0].reference_start == 2
    assert differences[0].reference_end == 2


def test_observed_chain_sequence_family_alignment_prefers_best_isoform() -> None:
    """Family alignment should prefer the isoform with the smallest edit burden."""

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
                residue_id=ResidueId(chain_id="A", seq_num=2),
                atom_sites=(),
            ),
        ),
    )
    observed_sequence = ObservedChainSequence.from_chain(chain)
    family = UniProtSequenceFamilyFetchResult(
        requested_reference=UniProtSequenceReference(
            accession="P12345",
            isoform_accession="P12345-2",
        ),
        family=UniProtSequenceFamily(
            canonical_record=UniProtSequenceRecord(
                uniprot_reference=UniProtSequenceReference(accession="P12345"),
                primary_accession="P12345",
                isoform_accession=None,
                sequence="ACG",
                declared_isoform_accessions=("P12345-2",),
            ),
            isoform_records=(
                UniProtSequenceRecord(
                    uniprot_reference=UniProtSequenceReference(
                        accession="P12345",
                        isoform_accession="P12345-2",
                    ),
                    primary_accession="P12345-2",
                    isoform_accession="P12345-2",
                    sequence="AG",
                ),
            ),
        ),
    )

    alignment_set = align_observed_chain_to_uniprot_family(
        observed_sequence,
        family,
    )

    primary_alignment = alignment_set.primary_alignment()
    assert primary_alignment is not None
    assert primary_alignment.reference_blueprint.sequence() == "AG"
    assert primary_alignment.reference_attachment is not None
    assert primary_alignment.reference_attachment.label == "P12345-2"
    assert primary_alignment.reference_attachment.source_family == (
        ReferenceSequenceSourceFamily.uniprot("P12345")
    )
    assert primary_alignment.differences() == ()
    interpretation = alignment_set.primary_interpretation()
    assert interpretation is not None
    assert interpretation.prefers_isoform
    assert interpretation.is_exact_match()


def test_observed_chain_sequence_alignment_classifies_terminal_truncation() -> None:
    """Leading reference-only segments should classify as N-terminal truncation."""

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
    observed_sequence = ObservedChainSequence.from_chain(chain)
    reference_record = UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(accession="P12345"),
        primary_accession="P12345",
        isoform_accession=None,
        sequence="AGS",
    )

    alignment = align_observed_chain_to_uniprot_record(
        observed_sequence,
        reference_record,
    )

    differences = alignment.differences()
    assert len(differences) == 1
    assert differences[0].kind is SequenceDifferenceKind.N_TERMINAL_TRUNCATION
    assert differences[0].reference_sequence == "A"


def test_observed_chain_sequence_align_to_blueprint_keeps_reference_positions() -> None:
    """Blueprint alignment should preserve explicit blueprint sequence positions."""

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
    observed_sequence = ObservedChainSequence.from_chain(chain)
    reference_blueprint = PolymerChainBlueprint(
        chain_id=None,
        residue_slots=(
            PolymerResidueSlot(sequence_position=9, token="A"),
            PolymerResidueSlot(sequence_position=10, token="G"),
            PolymerResidueSlot(sequence_position=11, token="S"),
        ),
    )

    alignment = observed_sequence.align_to_blueprint(reference_blueprint)

    assert tuple(
        column.reference_position
        for column in alignment.columns
        if column.reference_position is not None
    ) == (9, 10, 11)
