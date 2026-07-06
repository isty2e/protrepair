from typing import cast

import pytest

from protrepair.relation.sequence_alignment import (
    ObservedChainSequence,
    ReferenceSequenceAttachment,
    ReferenceSequenceCandidate,
)
from protrepair.sources import (
    UniProtSequenceRecord,
    UniProtSequenceReference,
)
from protrepair.sources.projection import (
    polymer_blueprint_from_uniprot_record,
)
from protrepair.structure import (
    ChainSite,
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
    ResidueSite,
)
from protrepair.structure.labels import (
    ResidueId,
)


def test_uniprot_sequence_record_projects_polymer_blueprint_interval() -> None:
    """UniProt projections should support explicit requested intervals."""

    record = UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(
            accession="P12345",
            isoform_accession="P12345-2",
        ),
        primary_accession="P12345-2",
        isoform_accession="P12345-2",
        sequence="ACDEFGHIKLMN",
    )

    blueprint = polymer_blueprint_from_uniprot_record(
        record,
        interval_reference=UniProtSequenceReference(
            accession="P12345",
            isoform_accession="P12345-2",
            residue_start=10,
            residue_end=12,
        ),
    )

    assert blueprint == PolymerBlueprint(
        chains=(
            PolymerChainBlueprint(
                chain_id=None,
                residue_slots=(
                    PolymerResidueSlot(sequence_position=10, token="L"),
                    PolymerResidueSlot(sequence_position=11, token="M"),
                    PolymerResidueSlot(sequence_position=12, token="N"),
                ),
            ),
        )
    )


def test_unassigned_reference_blueprint_chain_identity_is_coherent() -> None:
    """Unassigned reference blueprints should support None lookups explicitly."""

    chain = PolymerChainBlueprint(
        chain_id=None,
        residue_slots=(
            PolymerResidueSlot(sequence_position=1, token="A"),
            PolymerResidueSlot(sequence_position=2, token="C"),
        ),
    )
    blueprint = PolymerBlueprint(chains=(chain,))

    assert blueprint.chain_ids() == (None,)
    assert blueprint.chain(None) is chain
    assert blueprint.select_chains((None,)) == blueprint


def test_polymer_chain_blueprint_rejects_malformed_chain_identity() -> None:
    """Blueprint chain ids should fail before accidental `.strip()` errors."""

    with pytest.raises(TypeError, match="chain_id"):
        PolymerChainBlueprint(
            chain_id=cast(str | None, 1),
            residue_slots=(
                PolymerResidueSlot(sequence_position=1, token="A"),
            ),
        )


def test_blank_reference_blueprint_chain_id_normalizes_to_unassigned() -> None:
    """Blank blueprint chain ids are unassigned reference identities."""

    chain = PolymerChainBlueprint(
        chain_id=" ",
        residue_slots=(
            PolymerResidueSlot(sequence_position=1, token="A"),
        ),
    )

    assert chain.chain_id is None


def test_polymer_blueprint_rejects_duplicate_unassigned_chains() -> None:
    """Two unassigned chains would make `chain(None)` ambiguous."""

    with pytest.raises(ValueError, match="repeat chain ids"):
        PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id=None,
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                    ),
                ),
                PolymerChainBlueprint(
                    chain_id=" ",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=2, token="C"),
                    ),
                ),
            )
        )


def test_polymer_blueprint_selects_mixed_assigned_and_unassigned_chains() -> None:
    """Reference blueprints may mix assigned and not-yet-assigned chain ids."""

    unassigned_chain = PolymerChainBlueprint(
        chain_id=None,
        residue_slots=(
            PolymerResidueSlot(sequence_position=1, token="A"),
        ),
    )
    assigned_chain = PolymerChainBlueprint(
        chain_id="B",
        residue_slots=(
            PolymerResidueSlot(sequence_position=1, token="G"),
        ),
    )
    blueprint = PolymerBlueprint(chains=(unassigned_chain, assigned_chain))

    assert blueprint.select_chains(("B", None)) == PolymerBlueprint(
        chains=(assigned_chain, unassigned_chain)
    )


def test_observed_chain_sequence_aligns_against_blueprint() -> None:
    """Observed sequences should align directly against polymer blueprints."""

    observed_sequence = ObservedChainSequence.from_chain(
        ChainSite(
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
    )
    reference_blueprint = PolymerChainBlueprint(
        chain_id=None,
        residue_slots=(
            PolymerResidueSlot(sequence_position=1, token="A"),
            PolymerResidueSlot(sequence_position=2, token="C"),
            PolymerResidueSlot(sequence_position=3, token="G"),
        ),
    )

    alignment = observed_sequence.align_to_blueprint(reference_blueprint)

    assert alignment.reference_blueprint is reference_blueprint
    assert alignment.reference_attachment is None
    assert alignment.match_count() == 2


def test_observed_chain_sequence_aligns_against_explicit_reference_candidates() -> None:
    """Observed sequences should align against blueprint candidates.

    This path should stay valid without a UniProt record attachment.
    """

    observed_sequence = ObservedChainSequence.from_chain(
        ChainSite(
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
    )

    alignment_set = observed_sequence.align_to_candidates(
        (
            ReferenceSequenceCandidate(
                blueprint=PolymerChainBlueprint(
                    chain_id=None,
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="G"),
                    ),
                ),
                attachment=ReferenceSequenceAttachment(label="candidate-1"),
            ),
            ReferenceSequenceCandidate(
                blueprint=PolymerChainBlueprint(
                    chain_id=None,
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="C"),
                    ),
                ),
                attachment=ReferenceSequenceAttachment(label="candidate-2"),
            ),
        )
    )

    primary_alignment = alignment_set.primary_alignment()

    assert primary_alignment is not None
    assert primary_alignment.reference_attachment is not None
    assert primary_alignment.reference_attachment.label == "candidate-1"
