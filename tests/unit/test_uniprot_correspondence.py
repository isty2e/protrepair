"""UniProt correspondence ontology tests."""

import pytest

from protrepair.relation import (
    StructureEndpoint,
    StructureUniProtCorrespondence,
    StructureUniProtCorrespondenceSet,
    UniProtCorrespondenceConfidence,
)
from protrepair.scope import ResidueSetScope
from protrepair.sources import (
    UniProtSequenceFamily,
    UniProtSequenceFamilyFetchResult,
    UniProtSequenceFetchFailure,
    UniProtSequenceFetchFailureKind,
    UniProtSequenceFetchOutcome,
    UniProtSequenceRecord,
    UniProtSequenceReference,
)
from protrepair.structure.labels import (
    ResidueId,
)


def test_uniprot_sequence_reference_normalizes_accessions() -> None:
    """UniProt references should normalize accession casing and whitespace."""

    reference = UniProtSequenceReference(
        accession=" p12345 ",
        isoform_accession=" p12345-2 ",
        residue_start=10,
        residue_end=50,
    )

    assert reference.accession == "P12345"
    assert reference.isoform_accession == "P12345-2"
    assert reference.effective_accession() == "P12345-2"
    assert reference.references_interval()


def test_structure_uniprot_correspondence_set_preserves_ambiguity() -> None:
    """Correspondence sets should keep multiple same-region candidates."""

    structure_endpoint = StructureEndpoint.source(
        ResidueSetScope(
            residue_ids=(
                ResidueId(chain_id="A", seq_num=10),
                ResidueId(chain_id="A", seq_num=11),
            )
        )
    )
    high_candidate = StructureUniProtCorrespondence(
        structure_endpoint=structure_endpoint,
        uniprot_reference=UniProtSequenceReference(accession="P12345"),
        confidence=UniProtCorrespondenceConfidence.HIGH,
        sequence_identity=0.97,
        sequence_coverage=1.0,
    )
    exact_candidate = StructureUniProtCorrespondence(
        structure_endpoint=structure_endpoint,
        uniprot_reference=UniProtSequenceReference(accession="Q8ZZZ8"),
        confidence=UniProtCorrespondenceConfidence.EXACT,
        sequence_identity=1.0,
        sequence_coverage=1.0,
    )

    correspondence_set = StructureUniProtCorrespondenceSet(
        structure_endpoint=structure_endpoint,
        candidates=(high_candidate, exact_candidate),
    )

    assert correspondence_set.chain_id() == "A"
    assert correspondence_set.is_ambiguous()
    assert correspondence_set.best_candidates() == (exact_candidate,)
    assert correspondence_set.primary_candidate() is exact_candidate


def test_structure_uniprot_correspondence_rejects_multi_chain_regions() -> None:
    """UniProt correspondences should stay on one chain."""

    structure_endpoint = StructureEndpoint.source(
        ResidueSetScope(
            residue_ids=(
                ResidueId(chain_id="A", seq_num=10),
                ResidueId(chain_id="B", seq_num=10),
            )
        )
    )

    with pytest.raises(ValueError, match="single-chain polymer region"):
        StructureUniProtCorrespondence(
            structure_endpoint=structure_endpoint,
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            confidence=UniProtCorrespondenceConfidence.HIGH,
        )


def test_uniprot_sequence_fetch_outcome_requires_exactly_one_variant() -> None:
    """Fetch outcomes should be either success or failure, never both/neither."""

    reference = UniProtSequenceReference(accession="P12345")
    record = UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(accession="P12345"),
        primary_accession="P12345",
        isoform_accession=None,
        sequence="MPEPTIDE",
    )

    with pytest.raises(ValueError, match="exactly one"):
        UniProtSequenceFetchOutcome(requested_reference=reference)

    assert UniProtSequenceFetchOutcome.success(reference, record).is_success()
    assert (
        UniProtSequenceFetchOutcome.failure_result(
            failure=UniProtSequenceFetchFailure(
                requested_reference=reference,
                kind=UniProtSequenceFetchFailureKind.NOT_FOUND,
                message="not found",
            )
        ).is_success()
        is False
    )


def test_uniprot_sequence_family_record_preserves_requested_isoform_lookup() -> None:
    """Family records should expose the originally requested isoform record."""

    requested_reference = UniProtSequenceReference(
        accession="P12345",
        isoform_accession="P12345-2",
    )
    canonical_record = UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(accession="P12345"),
        primary_accession="P12345",
        isoform_accession=None,
        sequence="MPEPTIDE",
        declared_isoform_accessions=("P12345-1", "P12345-2"),
    )
    isoform_record = UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(
            accession="P12345",
            isoform_accession="P12345-2",
        ),
        primary_accession="P12345-2",
        isoform_accession="P12345-2",
        sequence="MPEPTIDER",
    )

    family = UniProtSequenceFamilyFetchResult(
        requested_reference=requested_reference,
        family=UniProtSequenceFamily(
            canonical_record=canonical_record,
            isoform_records=(isoform_record,),
        ),
    )

    assert family.requested_reference.accession == "P12345"
    assert family.requested_record() is isoform_record
