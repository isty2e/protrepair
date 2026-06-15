"""Source-aware projection into canonical blueprint and alignment nouns."""

from protrepair.relation.sequence_alignment import (
    ObservedChainSequence,
    ObservedSequenceAlignment,
    ObservedSequenceAlignmentSet,
    ReferenceSequenceAttachment,
    ReferenceSequenceCandidate,
    ReferenceSequenceSourceFamily,
)
from protrepair.relation.uniprot_reference import UniProtSequenceReference
from protrepair.sources.alphafold import AlphaFoldModelRecord
from protrepair.sources.uniprot import (
    UniProtSequenceFamilyFetchResult,
    UniProtSequenceRecord,
)
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)


def polymer_chain_blueprint_from_uniprot_record(
    record: UniProtSequenceRecord,
    *,
    interval_reference: UniProtSequenceReference | None = None,
    chain_id: str | None = None,
) -> PolymerChainBlueprint:
    """Project one UniProt record into one canonical polymer-chain blueprint."""

    if not isinstance(record, UniProtSequenceRecord):
        raise TypeError(
            "polymer_chain_blueprint_from_uniprot_record requires a "
            "UniProtSequenceRecord"
        )

    selection_reference = _selected_uniprot_reference(
        record,
        interval_reference=interval_reference,
    )
    effective_chain_id = None if chain_id is None else chain_id.strip() or None
    first_sequence_position = (
        1
        if selection_reference.residue_start is None
        else selection_reference.residue_start
    )
    selected_sequence = selection_reference.projected_sequence(record.sequence)
    return PolymerChainBlueprint(
        chain_id=effective_chain_id,
        residue_slots=tuple(
            PolymerResidueSlot(
                sequence_position=first_sequence_position + offset,
                token=token,
            )
            for offset, token in enumerate(selected_sequence)
        ),
    )


def polymer_blueprint_from_uniprot_record(
    record: UniProtSequenceRecord,
    *,
    interval_reference: UniProtSequenceReference | None = None,
    chain_id: str | None = None,
) -> PolymerBlueprint:
    """Project one UniProt record into one canonical polymer blueprint."""

    return PolymerBlueprint(
        chains=(
            polymer_chain_blueprint_from_uniprot_record(
                record,
                interval_reference=interval_reference,
                chain_id=chain_id,
            ),
        )
    )


def polymer_chain_blueprint_from_alphafold_model(
    model: AlphaFoldModelRecord,
    *,
    chain_id: str | None = None,
) -> PolymerChainBlueprint:
    """Project one AlphaFold model record into one canonical chain blueprint."""

    if not isinstance(model, AlphaFoldModelRecord):
        raise TypeError(
            "polymer_chain_blueprint_from_alphafold_model requires an "
            "AlphaFoldModelRecord"
        )

    effective_chain_id = None if chain_id is None else chain_id.strip() or None
    return PolymerChainBlueprint(
        chain_id=effective_chain_id,
        residue_slots=tuple(
            PolymerResidueSlot(sequence_position=sequence_position, token=token)
            for sequence_position, token in enumerate(model.sequence, start=1)
        ),
    )


def polymer_blueprint_from_alphafold_model(
    model: AlphaFoldModelRecord,
    *,
    chain_id: str | None = None,
) -> PolymerBlueprint:
    """Project one AlphaFold model record into one canonical polymer blueprint."""

    return PolymerBlueprint(
        chains=(polymer_chain_blueprint_from_alphafold_model(model, chain_id=chain_id),)
    )


def reference_sequence_candidate_from_uniprot_record(
    record: UniProtSequenceRecord,
    *,
    interval_reference: UniProtSequenceReference | None = None,
    chain_id: str | None = None,
) -> ReferenceSequenceCandidate:
    """Project one UniProt record into one source-attached reference candidate."""

    if not isinstance(record, UniProtSequenceRecord):
        raise TypeError(
            "reference_sequence_candidate_from_uniprot_record requires a "
            "UniProtSequenceRecord"
        )

    return ReferenceSequenceCandidate(
        blueprint=polymer_chain_blueprint_from_uniprot_record(
            record,
            interval_reference=interval_reference,
            chain_id=chain_id,
        ),
        attachment=ReferenceSequenceAttachment(
            label=record.effective_accession(),
            source_family=ReferenceSequenceSourceFamily.uniprot(
                record.uniprot_reference.accession
            ),
            is_canonical=record.uniprot_reference.isoform_accession is None,
        ),
    )


def align_observed_chain_to_uniprot_record(
    observed_sequence: ObservedChainSequence,
    record: UniProtSequenceRecord,
    *,
    interval_reference: UniProtSequenceReference | None = None,
) -> ObservedSequenceAlignment:
    """Align one observed chain sequence against one UniProt-backed candidate."""

    if not isinstance(observed_sequence, ObservedChainSequence):
        raise TypeError(
            "align_observed_chain_to_uniprot_record requires an "
            "ObservedChainSequence"
        )
    if not isinstance(record, UniProtSequenceRecord):
        raise TypeError(
            "align_observed_chain_to_uniprot_record requires a "
            "UniProtSequenceRecord"
        )

    return observed_sequence.align_to_candidate(
        reference_sequence_candidate_from_uniprot_record(
            record,
            interval_reference=interval_reference,
        )
    )


def align_observed_chain_to_uniprot_family(
    observed_sequence: ObservedChainSequence,
    family: UniProtSequenceFamilyFetchResult,
) -> ObservedSequenceAlignmentSet:
    """Align one observed chain sequence against one fetched UniProt family."""

    if not isinstance(observed_sequence, ObservedChainSequence):
        raise TypeError(
            "align_observed_chain_to_uniprot_family requires an "
            "ObservedChainSequence"
        )
    if not isinstance(family, UniProtSequenceFamilyFetchResult):
        raise TypeError(
            "align_observed_chain_to_uniprot_family requires a "
            "UniProtSequenceFamilyFetchResult"
        )

    return observed_sequence.align_to_candidates(
        tuple(
            reference_sequence_candidate_from_uniprot_record(
                record,
                interval_reference=family.requested_reference,
            )
            for record in family.successful_records()
        )
    )


def _selected_uniprot_reference(
    record: UniProtSequenceRecord,
    *,
    interval_reference: UniProtSequenceReference | None,
) -> UniProtSequenceReference:
    """Return one selection reference for projecting a raw UniProt record."""

    if interval_reference is None:
        return record.uniprot_reference

    if not isinstance(interval_reference, UniProtSequenceReference):
        raise TypeError(
            "UniProt interval projection requires a UniProtSequenceReference or None"
        )
    if interval_reference.accession != record.uniprot_reference.accession:
        raise ValueError(
            "UniProt interval projection requires the same base accession as the "
            "raw source record"
        )

    return UniProtSequenceReference(
        accession=record.uniprot_reference.accession,
        isoform_accession=record.uniprot_reference.isoform_accession,
        residue_start=interval_reference.residue_start,
        residue_end=interval_reference.residue_end,
    )
