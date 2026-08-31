"""Workflow requests grounded in external sequence and structure references."""

from dataclasses import dataclass

from protrepair.relation.blueprint import (
    StructureBlueprintCoverage,
    StructureBlueprintCoverageGap,
)
from protrepair.relation.sequence_alignment import (
    ObservedChainSequence,
    ObservedSequenceAlignment,
    ReferenceSequenceSourceFamily,
)
from protrepair.scope import AbsentResidueSpanScope, ResidueSetScope
from protrepair.sources.alphafold import AlphaFoldStructureArtifact
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.source_ingestion import ingest_alphafold_structure_artifact
from protrepair.workflow.contracts.span_policy import ExternalSpanGapSelectionPolicy

_DEFAULT_EXTERNAL_SPAN_GAP_SELECTION_POLICY = (
    ExternalSpanGapSelectionPolicy.internal_only()
)


@dataclass(frozen=True, slots=True)
class ExternalSpanReconstructionSpec:
    """One canonical absent-span reconstruction request backed by one donor.

    Parameters
    ----------
    scope : AbsentResidueSpanScope
        Contiguous missing source residues and their one or two source anchors.
    donor_structure : ProteinStructure
        Canonical structure supplying the conformation seed.
    donor_residue_ids : tuple[ResidueId, ...]
        Ordered, contiguous donor residues corresponding one-to-one with the
        missing source residues.

    Raises
    ------
    TypeError
        If a field does not satisfy its declared contract.
    ValueError
        If either span is non-contiguous, mismatched, or lacks required polymer
        donor context.
    """

    scope: AbsentResidueSpanScope
    donor_structure: ProteinStructure
    donor_residue_ids: tuple[ResidueId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AbsentResidueSpanScope):
            raise TypeError(
                "external span reconstruction specs require an AbsentResidueSpanScope"
            )
        if not isinstance(self.donor_structure, ProteinStructure):
            raise TypeError(
                "external span reconstruction specs require a donor ProteinStructure"
            )

        absent_residue_ids = self.scope.absent_residue_ids
        if not absent_residue_ids:
            raise ValueError(
                "external span reconstruction specs require absent residue ids"
            )
        target_residue_ids = tuple(
            residue_id
            for residue_id in (
                self.scope.preceding_residue_id,
                *absent_residue_ids,
                self.scope.following_residue_id,
            )
            if residue_id is not None
        )
        if len({residue_id.chain_id for residue_id in target_residue_ids}) != 1:
            raise ValueError(
                "external span reconstruction target residues must belong to one chain"
            )
        if any(
            not left_residue_id.immediately_precedes(right_residue_id)
            for left_residue_id, right_residue_id in zip(
                target_residue_ids,
                target_residue_ids[1:],
                strict=False,
            )
        ):
            raise ValueError(
                "external span reconstruction target residues must form one "
                "contiguous chain span"
            )

        donor_residue_ids = tuple(self.donor_residue_ids)
        if any(
            not isinstance(donor_residue_id, ResidueId)
            for donor_residue_id in donor_residue_ids
        ):
            raise TypeError(
                "external span reconstruction donor residue ids must be ResidueId "
                "values"
            )
        if len(donor_residue_ids) != len(set(donor_residue_ids)):
            raise ValueError(
                "external span reconstruction donor residue ids must not repeat"
            )
        if len(donor_residue_ids) != len(self.scope.absent_residue_ids):
            raise ValueError(
                "external span reconstruction requires one donor residue for "
                "each absent residue"
            )
        for donor_residue_id in donor_residue_ids:
            if (
                self.donor_structure.constitution.residue_or_ligand(donor_residue_id)
                is None
            ):
                raise ValueError(
                    "external span reconstruction donor structure must expose "
                    f"{donor_residue_id.display_token()}"
                )

        donor_chain_ids = {residue_id.chain_id for residue_id in donor_residue_ids}
        if len(donor_chain_ids) != 1:
            raise ValueError(
                "external span reconstruction donor residues must belong to one chain"
            )
        donor_chain_id = donor_residue_ids[0].chain_id
        if not self.donor_structure.constitution.has_chain(donor_chain_id):
            raise ValueError(
                "external span reconstruction donor residues must occupy a "
                "polymer chain"
            )
        donor_chain = self.donor_structure.constitution.chain(donor_chain_id)
        donor_chain_residue_ids = donor_chain.residue_ids()
        if any(
            residue_id not in donor_chain_residue_ids
            for residue_id in donor_residue_ids
        ):
            raise ValueError(
                "external span reconstruction donor residues must occupy a "
                "polymer chain"
            )
        if any(
            donor_chain.residue(residue_id).is_hetero
            for residue_id in donor_residue_ids
        ):
            raise ValueError(
                "external span reconstruction donor residues must be canonical "
                "polymer residues"
            )
        donor_positions = tuple(
            donor_chain_residue_ids.index(residue_id)
            for residue_id in donor_residue_ids
        )
        expected_positions = tuple(
            range(donor_positions[0], donor_positions[0] + len(donor_positions))
        )
        if donor_positions != expected_positions:
            raise ValueError(
                "external span reconstruction donor residues must occupy one "
                "contiguous chain span in order"
            )

        preceding_donor_residue_id = (
            None
            if donor_positions[0] == 0
            else donor_chain_residue_ids[donor_positions[0] - 1]
        )
        following_donor_residue_id = (
            None
            if donor_positions[-1] + 1 == len(donor_chain_residue_ids)
            else donor_chain_residue_ids[donor_positions[-1] + 1]
        )
        required_donor_flanking_residue_ids = tuple(
            residue_id
            for source_anchor_id, residue_id in (
                (self.scope.preceding_residue_id, preceding_donor_residue_id),
                (self.scope.following_residue_id, following_donor_residue_id),
            )
            if source_anchor_id is not None and residue_id is not None
        )
        if any(
            donor_chain.residue(residue_id).is_hetero
            for residue_id in required_donor_flanking_residue_ids
        ):
            raise ValueError(
                "external span reconstruction donor flanks must be canonical "
                "polymer residues"
            )
        if (
            self.scope.preceding_residue_id is not None
            and preceding_donor_residue_id is None
        ):
            raise ValueError(
                "external span reconstruction requires donor context before an "
                "anchored span"
            )
        if (
            self.scope.following_residue_id is not None
            and following_donor_residue_id is None
        ):
            raise ValueError(
                "external span reconstruction requires donor context after an "
                "anchored span"
            )

        object.__setattr__(self, "donor_residue_ids", donor_residue_ids)

    def donor_scope(self) -> ResidueSetScope:
        """Return the donor residue subset addressed by this request."""

        return ResidueSetScope(residue_ids=self.donor_residue_ids)

    def donor_flanking_residue_ids(
        self,
    ) -> tuple[ResidueId | None, ResidueId | None]:
        """Return donor residues corresponding to the source span anchors.

        Returns
        -------
        tuple[ResidueId | None, ResidueId | None]
            Preceding and following donor flanks; an unanchored source side is
            represented by ``None``.
        """

        donor_chain_residue_ids = self.donor_structure.constitution.chain(
            self.donor_residue_ids[0].chain_id
        ).residue_ids()
        first_position = donor_chain_residue_ids.index(self.donor_residue_ids[0])
        last_position = first_position + len(self.donor_residue_ids) - 1
        return (
            None
            if self.scope.preceding_residue_id is None or first_position == 0
            else donor_chain_residue_ids[first_position - 1],
            None
            if (
                self.scope.following_residue_id is None
                or last_position + 1 == len(donor_chain_residue_ids)
            )
            else donor_chain_residue_ids[last_position + 1],
        )


def _infer_absent_residue_ids_from_blueprint_coverage_gap(
    gap: StructureBlueprintCoverageGap,
) -> tuple[ResidueId, ...]:
    """Infer structure-local absent residue ids for one lowerable coverage gap."""

    absent_sequence_positions = gap.absent_sequence_positions
    gap_width = len(absent_sequence_positions)
    if gap.is_internal():
        preceding_residue_id = gap.preceding_residue_id
        following_residue_id = gap.following_residue_id
        assert preceding_residue_id is not None
        assert following_residue_id is not None
        expected_following_seq_num = preceding_residue_id.seq_num + gap_width + 1
        if following_residue_id.seq_num != expected_following_seq_num:
            raise ValueError(
                "blueprint coverage gap cannot be lowered when flanking residue "
                "numbering does not match the missing sequence span width"
            )
        first_absent_seq_num = preceding_residue_id.seq_num + 1
    elif gap.is_prefix_terminal():
        following_residue_id = gap.following_residue_id
        assert following_residue_id is not None
        first_absent_seq_num = following_residue_id.seq_num - gap_width
        if first_absent_seq_num <= 0:
            raise ValueError(
                "blueprint prefix coverage gap cannot be lowered when inferred "
                "residue numbering would become non-positive"
            )
    else:
        preceding_residue_id = gap.preceding_residue_id
        assert preceding_residue_id is not None
        first_absent_seq_num = preceding_residue_id.seq_num + 1

    return tuple(
        ResidueId(
            chain_id=gap.structure_chain_id,
            seq_num=first_absent_seq_num + offset,
        )
        for offset in range(gap_width)
    )


def lower_blueprint_coverage_gap_to_absent_span_scope(
    gap: StructureBlueprintCoverageGap,
) -> AbsentResidueSpanScope:
    """Lower one canonical blueprint coverage gap into one structure-local scope."""

    absent_residue_ids = _infer_absent_residue_ids_from_blueprint_coverage_gap(gap)
    return AbsentResidueSpanScope(
        absent_residue_ids=absent_residue_ids,
        preceding_residue_id=gap.preceding_residue_id,
        following_residue_id=gap.following_residue_id,
    )


def build_alphafold_span_reconstruction_specs(
    *,
    source_structure: ProteinStructure,
    alignment: ObservedSequenceAlignment,
    artifact: AlphaFoldStructureArtifact,
    gap_selection_policy: ExternalSpanGapSelectionPolicy | None = None,
) -> tuple[ExternalSpanReconstructionSpec, ...]:
    """Build explicit blueprint-gap reconstruction requests from AlphaFold.

    Parameters
    ----------
    source_structure : ProteinStructure
        Canonical structure used to derive the observed chain alignment.
    alignment : ObservedSequenceAlignment
        Observed-to-UniProt alignment defining source coverage gaps.
    artifact : AlphaFoldStructureArtifact
        AlphaFold model from the same UniProt accession family.
    gap_selection_policy : ExternalSpanGapSelectionPolicy | None
        Gap classes to include; defaults to internal spans only.

    Returns
    -------
    tuple[ExternalSpanReconstructionSpec, ...]
        Explicit requests in source coverage-gap order.

    Raises
    ------
    TypeError
        If an argument does not satisfy its declared contract.
    ValueError
        If source and donor references disagree or a selected gap cannot be
        mapped to contiguous canonical residue IDs.
    """

    if not isinstance(source_structure, ProteinStructure):
        raise TypeError(
            "build_alphafold_span_reconstruction_specs requires a ProteinStructure"
        )
    if not isinstance(alignment, ObservedSequenceAlignment):
        raise TypeError(
            "build_alphafold_span_reconstruction_specs requires an "
            "ObservedSequenceAlignment"
        )
    if not isinstance(artifact, AlphaFoldStructureArtifact):
        raise TypeError(
            "build_alphafold_span_reconstruction_specs requires an "
            "AlphaFoldStructureArtifact"
        )
    if gap_selection_policy is None:
        active_gap_selection_policy = _DEFAULT_EXTERNAL_SPAN_GAP_SELECTION_POLICY
    else:
        active_gap_selection_policy = gap_selection_policy
    if not isinstance(active_gap_selection_policy, ExternalSpanGapSelectionPolicy):
        raise TypeError(
            "build_alphafold_span_reconstruction_specs requires an "
            "ExternalSpanGapSelectionPolicy value"
        )
    reference_source_family = (
        None
        if alignment.reference_attachment is None
        else alignment.reference_attachment.source_family
    )
    if reference_source_family != ReferenceSequenceSourceFamily.uniprot(
        artifact.model.uniprot_reference.accession
    ):
        raise ValueError(
            "AlphaFold span reconstruction requires the alignment reference and "
            "artifact model to share the same UniProt accession family"
        )
    source_chain_id = alignment.observed_sequence.chain_id
    if not source_structure.constitution.has_chain(source_chain_id):
        raise ValueError(
            "AlphaFold span reconstruction alignment must reference a source chain"
        )
    source_chain = source_structure.constitution.chain(source_chain_id)
    if ObservedChainSequence.from_chain(source_chain) != alignment.observed_sequence:
        raise ValueError(
            "AlphaFold span reconstruction alignment must describe the active "
            "source chain"
        )
    reference_blueprint = alignment.reference_blueprint
    if (
        reference_blueprint.first_sequence_position() != 1
        or reference_blueprint.sequence() != artifact.model.sequence
    ):
        raise ValueError(
            "AlphaFold span reconstruction requires the alignment reference "
            "sequence to match the artifact model sequence"
        )

    coverage = StructureBlueprintCoverage.from_alignment(alignment)
    donor_structure = ingest_alphafold_structure_artifact(artifact)
    donor_chain_id = _require_single_supporting_chain_id(donor_structure)
    _validate_alphafold_donor_sequence(
        donor_structure,
        donor_chain_id=donor_chain_id,
        expected_sequence=artifact.model.sequence,
    )

    reconstruction_specs: list[ExternalSpanReconstructionSpec] = []
    for gap in coverage.coverage_gaps:
        if not active_gap_selection_policy.selects_gap(gap):
            continue

        source_scope = lower_blueprint_coverage_gap_to_absent_span_scope(gap)
        for residue_id in source_scope.absent_residue_ids:
            if source_chain.has_residue(residue_id):
                raise ValueError(
                    "AlphaFold span reconstruction requires absent source residues "
                    "to be missing from the source chain"
                )

        donor_residue_ids = _supporting_residue_ids_for_reference_positions(
            supporting_structure=donor_structure,
            supporting_chain_id=donor_chain_id,
            reference_positions=gap.absent_sequence_positions,
        )
        reconstruction_specs.append(
            ExternalSpanReconstructionSpec(
                scope=source_scope,
                donor_structure=donor_structure,
                donor_residue_ids=donor_residue_ids,
            )
        )

    return tuple(reconstruction_specs)


def _supporting_residue_ids_for_reference_positions(
    *,
    supporting_structure: ProteinStructure,
    supporting_chain_id: str,
    reference_positions: tuple[int, ...],
) -> tuple[ResidueId, ...]:
    """Return donor residue ids keyed by reference positions in one support chain."""

    supporting_chain = supporting_structure.constitution.chain(supporting_chain_id)
    donor_residue_ids: list[ResidueId] = []
    for reference_position in reference_positions:
        donor_residue_id = ResidueId(
            chain_id=supporting_chain_id,
            seq_num=reference_position,
        )
        if not supporting_chain.has_residue(donor_residue_id):
            raise ValueError(
                "AlphaFold support structure does not expose donor residue "
                f"{donor_residue_id.display_token()}"
            )

        donor_residue_ids.append(donor_residue_id)

    return tuple(donor_residue_ids)


def _require_single_supporting_chain_id(
    structure: ProteinStructure,
) -> str:
    """Return the sole polymer chain id expected from one AlphaFold template."""

    if len(structure.constitution.chains) != 1:
        raise ValueError(
            "AlphaFold span reconstruction currently requires a single-chain "
            "supporting structure"
        )

    return structure.constitution.chains[0].chain_id


def _validate_alphafold_donor_sequence(
    structure: ProteinStructure,
    *,
    donor_chain_id: str,
    expected_sequence: str,
) -> None:
    """Require donor residue identities to match AlphaFold model metadata."""

    observed_sequence = ObservedChainSequence.from_chain(
        structure.constitution.chain(donor_chain_id)
    )
    expected_residue_ids = tuple(
        ResidueId(donor_chain_id, sequence_position)
        for sequence_position in range(1, len(expected_sequence) + 1)
    )
    if (
        tuple(residue.residue_id for residue in observed_sequence.residues)
        != expected_residue_ids
        or "".join(observed_sequence.tokens()) != expected_sequence
    ):
        raise ValueError(
            "AlphaFold span reconstruction requires the artifact structure "
            "sequence to match its model metadata"
        )
