"""Workflow requests grounded in external sequence and structure references."""

from dataclasses import dataclass

from protrepair.relation.blueprint import (
    StructureBlueprintCoverage,
    StructureBlueprintCoverageGap,
)
from protrepair.relation.sequence_alignment import (
    ObservedSequenceAlignment,
    ReferenceSequenceSourceFamily,
)
from protrepair.relation.supporting_role import SupportingStructureRole
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
    """One explicit blueprint-gap reconstruction request backed by one donor."""

    blueprint_coverage_gap: StructureBlueprintCoverageGap
    donor_structure: ProteinStructure
    donor_residue_ids: tuple[ResidueId, ...]
    supporting_role: SupportingStructureRole = SupportingStructureRole.TEMPLATE

    def __post_init__(self) -> None:
        if not isinstance(
            self.blueprint_coverage_gap,
            StructureBlueprintCoverageGap,
        ):
            raise TypeError(
                "external span reconstruction specs require a "
                "StructureBlueprintCoverageGap"
            )
        if not isinstance(self.donor_structure, ProteinStructure):
            raise TypeError(
                "external span reconstruction specs require a donor ProteinStructure"
            )
        if not isinstance(self.supporting_role, SupportingStructureRole):
            raise TypeError(
                "external span reconstruction specs require a "
                "SupportingStructureRole value"
            )

        absent_span_scope = lower_blueprint_coverage_gap_to_absent_span_scope(
            self.blueprint_coverage_gap
        )
        donor_residue_ids = tuple(self.donor_residue_ids)
        if len(donor_residue_ids) != len(set(donor_residue_ids)):
            raise ValueError(
                "external span reconstruction donor residue ids must not repeat"
            )
        if len(donor_residue_ids) != len(absent_span_scope.absent_residue_ids):
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

        object.__setattr__(self, "donor_residue_ids", donor_residue_ids)

    def absent_span_scope(self) -> AbsentResidueSpanScope:
        """Return the lowered source absent-span scope addressed by this request."""

        return lower_blueprint_coverage_gap_to_absent_span_scope(
            self.blueprint_coverage_gap
        )

    def donor_scope(self) -> ResidueSetScope:
        """Return the donor residue subset addressed by this request."""

        return ResidueSetScope(residue_ids=self.donor_residue_ids)


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
    supporting_role: SupportingStructureRole = SupportingStructureRole.TEMPLATE,
    gap_selection_policy: ExternalSpanGapSelectionPolicy | None = None,
) -> tuple[ExternalSpanReconstructionSpec, ...]:
    """Build explicit blueprint-gap reconstruction requests from AlphaFold."""

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
    if not isinstance(supporting_role, SupportingStructureRole):
        raise TypeError(
            "build_alphafold_span_reconstruction_specs requires a "
            "SupportingStructureRole value"
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

    coverage = StructureBlueprintCoverage.from_alignment(alignment)
    source_chain = source_structure.constitution.chain(coverage.structure_chain_id)
    donor_structure = ingest_alphafold_structure_artifact(artifact)
    donor_chain_id = _require_single_supporting_chain_id(donor_structure)

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
                blueprint_coverage_gap=gap,
                donor_structure=donor_structure,
                donor_residue_ids=donor_residue_ids,
                supporting_role=supporting_role,
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
