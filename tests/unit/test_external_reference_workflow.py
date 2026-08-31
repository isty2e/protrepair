"""External-reference workflow integration tests."""

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.geometry import AxisRotation, InternalCoordinateFrame, Vec3
from protrepair.io import FileFormat, read_structure_string, write_structure_string
from protrepair.scope import AbsentResidueSpanScope
from protrepair.sources import (
    AlphaFoldModelRecord,
    AlphaFoldStructureArtifact,
    ObservedChainSequence,
    UniProtSequenceRecord,
    UniProtSequenceReference,
    align_observed_chain_to_uniprot_record,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.workflow.contracts import (
    ExternalSpanGapSelectionPolicy,
    ExternalSpanReconstructionSpec,
    RequestedGoalCompletionVerdict,
    WorkflowTransformRequests,
    build_alphafold_span_reconstruction_specs,
)


def test_build_alphafold_span_reconstruction_specs_infers_internal_deletions() -> None:
    """AlphaFold builder should infer one explicit absent-span request."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                    _build_residue("PHE", "A", 5),
                    _build_residue("GLY", "A", 6),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    artifact = _alphafold_artifact(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue("CYS", "X", 2),
                        _build_residue("ASP", "X", 3),
                        _build_residue("GLU", "X", 4),
                        _build_residue("PHE", "X", 5),
                        _build_residue("GLY", "X", 6),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="alphafold-template",
        ),
        sequence="ACDEFG",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDEFG",
        ),
    )

    reconstruction_specs = build_alphafold_span_reconstruction_specs(
        source_structure=source_structure,
        alignment=alignment,
        artifact=artifact,
    )

    assert len(reconstruction_specs) == 1
    scope = reconstruction_specs[0].scope
    assert scope.absent_residue_ids == (
        ResidueId("A", 3),
        ResidueId("A", 4),
    )
    assert scope.preceding_residue_id == ResidueId("A", 2)
    assert scope.following_residue_id == ResidueId("A", 5)
    assert reconstruction_specs[0].donor_residue_ids == (
        ResidueId("X", 3),
        ResidueId("X", 4),
    )


def test_alphafold_span_builder_rejects_alignment_from_another_source() -> None:
    """Residue numbering alone must not attach a stale alignment to the source."""

    source_structure = _structure_with_internal_gap(second_component_id="CYS")
    alignment_source = _structure_with_internal_gap(second_component_id="SER")
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(alignment_source.constitution.chain("A")),
        _uniprot_record(sequence="ACDEFG"),
    )
    artifact = _alphafold_artifact(
        _complete_six_residue_structure(),
        sequence="ACDEFG",
    )

    with pytest.raises(ValueError, match="active source chain"):
        build_alphafold_span_reconstruction_specs(
            source_structure=source_structure,
            alignment=alignment,
            artifact=artifact,
        )


def test_alphafold_span_builder_rejects_reference_sequence_mismatch() -> None:
    """An accession match must not conceal different reference sequences."""

    source_structure = _structure_with_internal_gap(second_component_id="CYS")
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        _uniprot_record(sequence="ACDEFG"),
    )
    artifact = _alphafold_artifact(
        _complete_six_residue_structure(),
        sequence="ACDFFG",
    )

    with pytest.raises(ValueError, match="reference sequence"):
        build_alphafold_span_reconstruction_specs(
            source_structure=source_structure,
            alignment=alignment,
            artifact=artifact,
        )


def test_alphafold_span_builder_rejects_donor_sequence_mismatch() -> None:
    """AlphaFold metadata must agree with residue identities in its coordinates."""

    source_structure = _structure_with_internal_gap(second_component_id="CYS")
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        _uniprot_record(sequence="ACDEFG"),
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "GLU", "ASP", "PHE", "GLY"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="alphafold-template",
    )

    with pytest.raises(ValueError, match="artifact structure sequence"):
        build_alphafold_span_reconstruction_specs(
            source_structure=source_structure,
            alignment=alignment,
            artifact=_alphafold_artifact(donor_structure, sequence="ACDEFG"),
        )


def test_external_span_spec_rejects_noncontiguous_target_scope() -> None:
    """The canonical request must identify one contiguous source span."""

    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )

    with pytest.raises(ValueError, match="contiguous chain span"):
        ExternalSpanReconstructionSpec(
            scope=AbsentResidueSpanScope(
                preceding_residue_id=ResidueId("A", 2),
                absent_residue_ids=(ResidueId("A", 3),),
                following_residue_id=ResidueId("A", 5),
            ),
            donor_structure=donor_structure,
            donor_residue_ids=(ResidueId("X", 3),),
        )


def test_external_span_spec_rejects_hetero_residue_in_donor_chain() -> None:
    """Chain membership must not conceal a retained non-polymer donor role."""

    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ALA", "X", 1),
                    _build_residue("CYS", "X", 2, is_hetero=True),
                    _build_residue("ASP", "X", 3),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )

    with pytest.raises(ValueError, match="canonical polymer residues"):
        ExternalSpanReconstructionSpec(
            scope=AbsentResidueSpanScope(
                preceding_residue_id=ResidueId("A", 1),
                absent_residue_ids=(ResidueId("A", 2),),
                following_residue_id=ResidueId("A", 3),
            ),
            donor_structure=donor_structure,
            donor_residue_ids=(ResidueId("X", 2),),
        )


def test_external_span_spec_rejects_hetero_donor_flank() -> None:
    """An anchored span must use polymer donor context at that boundary."""

    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ALA", "X", 1, is_hetero=True),
                    _build_residue("CYS", "X", 2),
                    _build_residue("ASP", "X", 3),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )

    with pytest.raises(ValueError, match="donor flanks"):
        ExternalSpanReconstructionSpec(
            scope=AbsentResidueSpanScope(
                preceding_residue_id=ResidueId("A", 1),
                absent_residue_ids=(ResidueId("A", 2),),
                following_residue_id=ResidueId("A", 3),
            ),
            donor_structure=donor_structure,
            donor_residue_ids=(ResidueId("X", 2),),
        )


def test_external_span_spec_requires_donor_context_at_each_source_anchor() -> None:
    """An internal source span needs a donor residue outside each endpoint."""

    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("CYS", "X", 2),
                    _build_residue("ASP", "X", 3),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )

    with pytest.raises(ValueError, match="requires donor context before"):
        ExternalSpanReconstructionSpec(
            scope=AbsentResidueSpanScope(
                preceding_residue_id=ResidueId("A", 1),
                absent_residue_ids=(ResidueId("A", 2),),
                following_residue_id=ResidueId("A", 3),
            ),
            donor_structure=donor_structure,
            donor_residue_ids=(ResidueId("X", 2),),
        )


def test_transform_requests_reject_conflicting_span_scopes() -> None:
    """Different explicit spans must not overlap or depend on execution order."""

    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    first_reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId("A", 1),
            absent_residue_ids=(ResidueId("A", 2), ResidueId("A", 3)),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 2), ResidueId("X", 3)),
    )
    conflicting_reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId("A", 2),
            absent_residue_ids=(ResidueId("A", 3), ResidueId("A", 4)),
            following_residue_id=ResidueId("A", 5),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3), ResidueId("X", 4)),
    )

    with pytest.raises(ValueError, match="must not overlap or depend"):
        WorkflowTransformRequests(
            external_span_reconstructions=(
                first_reconstruction,
                conflicting_reconstruction,
            )
        )


def test_transform_requests_allow_alternative_donors_for_the_same_span() -> None:
    """Exact-scope donor alternatives must remain separate workflow branches."""

    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId("A", 1),
        absent_residue_ids=(ResidueId("A", 2),),
        following_residue_id=ResidueId("A", 3),
    )
    donor_alternatives = (
        ExternalSpanReconstructionSpec(
            scope=scope,
            donor_structure=donor_structure,
            donor_residue_ids=(ResidueId("X", 2),),
        ),
        ExternalSpanReconstructionSpec(
            scope=scope,
            donor_structure=donor_structure,
            donor_residue_ids=(ResidueId("X", 3),),
        ),
    )

    requests = WorkflowTransformRequests(
        external_span_reconstructions=donor_alternatives
    )

    assert requests.external_span_reconstructions == donor_alternatives


def test_workflow_selects_valid_alternative_donor_for_the_same_span() -> None:
    """A rejected donor branch must not suppress a valid same-scope candidate."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("ASP", "A", 3),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structures = tuple(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue(
                            "CYS",
                            "X",
                            2,
                            alpha_carbon_orientation_sign=orientation_sign,
                        ),
                        _build_residue("ASP", "X", 3),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name=source_name,
        )
        for orientation_sign, source_name in (
            (-1, "mirrored-donor"),
            (1, "valid-donor"),
        )
    )
    scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId("A", 1),
        absent_residue_ids=(ResidueId("A", 2),),
        following_residue_id=ResidueId("A", 3),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=tuple(
                ExternalSpanReconstructionSpec(
                    scope=scope,
                    donor_structure=donor_structure,
                    donor_residue_ids=(ResidueId("X", 2),),
                )
                for donor_structure in donor_structures
            ),
        ),
    )

    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 2))
    assert not result.issues
    assert tuple(repair.kind for repair in result.repairs) == (
        RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED,
    )


def test_external_span_spec_uses_only_donor_flanks_with_source_anchors() -> None:
    """Terminal requests must ignore donor context on the unanchored side."""

    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    prefix_spec = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 1),),
            following_residue_id=ResidueId("A", 2),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 2),),
    )
    suffix_spec = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId("A", 2),
            absent_residue_ids=(ResidueId("A", 3),),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 2),),
    )

    assert prefix_spec.donor_flanking_residue_ids() == (
        None,
        ResidueId("X", 3),
    )
    assert suffix_spec.donor_flanking_residue_ids() == (
        ResidueId("X", 1),
        None,
    )


def test_build_alphafold_span_reconstruction_specs_skips_terminal_gaps_by_default() -> (
    None
):
    """AlphaFold builder should keep terminal gaps opt-in by default."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ASP", "A", 3),
                    _build_residue("GLU", "A", 4),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    artifact = _alphafold_artifact(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue("CYS", "X", 2),
                        _build_residue("ASP", "X", 3),
                        _build_residue("GLU", "X", 4),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="alphafold-template",
        ),
        sequence="ACDE",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDE",
        ),
    )

    reconstruction_specs = build_alphafold_span_reconstruction_specs(
        source_structure=source_structure,
        alignment=alignment,
        artifact=artifact,
    )

    assert reconstruction_specs == ()


def test_build_alphafold_span_reconstruction_specs_can_select_prefix_terminal_gap() -> (
    None
):
    """AlphaFold builder should admit prefix gaps when policy opts in."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ASP", "A", 3),
                    _build_residue("GLU", "A", 4),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    artifact = _alphafold_artifact(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue("CYS", "X", 2),
                        _build_residue("ASP", "X", 3),
                        _build_residue("GLU", "X", 4),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="alphafold-template",
        ),
        sequence="ACDE",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDE",
        ),
    )

    reconstruction_specs = build_alphafold_span_reconstruction_specs(
        source_structure=source_structure,
        alignment=alignment,
        artifact=artifact,
        gap_selection_policy=ExternalSpanGapSelectionPolicy(
            include_internal=True,
            include_prefix_terminal=True,
            include_suffix_terminal=False,
        ),
    )

    assert len(reconstruction_specs) == 1
    scope = reconstruction_specs[0].scope
    assert scope.preceding_residue_id is None
    assert scope.following_residue_id == ResidueId("A", 3)
    assert scope.absent_residue_ids == (
        ResidueId("A", 1),
        ResidueId("A", 2),
    )
    assert reconstruction_specs[0].donor_residue_ids == (
        ResidueId("X", 1),
        ResidueId("X", 2),
    )


def test_build_alphafold_span_reconstruction_specs_consumes_planning_context() -> None:
    """AlphaFold builder should use explicit planning-context gap policy."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ASP", "A", 3),
                    _build_residue("GLU", "A", 4),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    artifact = _alphafold_artifact(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue("CYS", "X", 2),
                        _build_residue("ASP", "X", 3),
                        _build_residue("GLU", "X", 4),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="alphafold-template",
        ),
        sequence="ACDE",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDE",
        ),
    )

    reconstruction_specs = build_alphafold_span_reconstruction_specs(
        source_structure=source_structure,
        alignment=alignment,
        artifact=artifact,
        gap_selection_policy=ExternalSpanGapSelectionPolicy(
            include_internal=True,
            include_prefix_terminal=True,
            include_suffix_terminal=False,
        ),
    )

    assert len(reconstruction_specs) == 1
    assert reconstruction_specs[0].scope.preceding_residue_id is None


def test_build_alphafold_span_reconstruction_specs_can_select_suffix_terminal_gap() -> (
    None
):
    """AlphaFold builder should admit suffix gaps when policy opts in."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    artifact = _alphafold_artifact(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue("CYS", "X", 2),
                        _build_residue("ASP", "X", 3),
                        _build_residue("GLU", "X", 4),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="alphafold-template",
        ),
        sequence="ACDE",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDE",
        ),
    )

    reconstruction_specs = build_alphafold_span_reconstruction_specs(
        source_structure=source_structure,
        alignment=alignment,
        artifact=artifact,
        gap_selection_policy=ExternalSpanGapSelectionPolicy(
            include_internal=True,
            include_prefix_terminal=False,
            include_suffix_terminal=True,
        ),
    )

    assert len(reconstruction_specs) == 1
    scope = reconstruction_specs[0].scope
    assert scope.preceding_residue_id == ResidueId("A", 2)
    assert scope.following_residue_id is None
    assert scope.absent_residue_ids == (
        ResidueId("A", 3),
        ResidueId("A", 4),
    )
    assert reconstruction_specs[0].donor_residue_ids == (
        ResidueId("X", 3),
        ResidueId("X", 4),
    )


def test_process_structure_reconstructs_prefix_terminal_span_from_alphafold() -> None:
    """Workflow should adopt one prefix-terminal explicit span reconstruction."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ASP", "A", 3),
                    _build_residue("GLU", "A", 4),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    artifact = _alphafold_artifact(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue("CYS", "X", 2),
                        _build_residue("ASP", "X", 3),
                        _build_residue("GLU", "X", 4),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="alphafold-template",
        ),
        sequence="ACDE",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDE",
        ),
    )
    reconstruction_specs = build_alphafold_span_reconstruction_specs(
        source_structure=source_structure,
        alignment=alignment,
        artifact=artifact,
        gap_selection_policy=ExternalSpanGapSelectionPolicy(
            include_internal=True,
            include_prefix_terminal=True,
            include_suffix_terminal=False,
        ),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=reconstruction_specs,
        ),
    )

    chain = result.structure.constitution.chain("A")
    assert chain.residue_ids() == (
        ResidueId("A", 1),
        ResidueId("A", 2),
        ResidueId("A", 3),
        ResidueId("A", 4),
    )
    assert tuple(residue.component_id for residue in chain.residues) == (
        "ALA",
        "CYS",
        "ASP",
        "GLU",
    )
    assert tuple(repair.kind for repair in result.repairs) == (
        RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED,
    )
    assert result.repairs[0].details == "one-anchor donor projection"


def test_process_structure_reconstructs_multiple_absent_spans_from_alphafold() -> None:
    """Workflow should sequentially adopt all explicit absent-span reconstructions."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                    _build_residue("PHE", "A", 5),
                    _build_residue("GLY", "A", 6),
                    _build_residue("LYS", "A", 9),
                    _build_residue("LEU", "A", 10),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    artifact = _alphafold_artifact(
        build_structure(
            chains=(
                chain_payload(
                    "X",
                    (
                        _build_residue("ALA", "X", 1),
                        _build_residue("CYS", "X", 2),
                        _build_residue("ASP", "X", 3),
                        _build_residue("GLU", "X", 4),
                        _build_residue("PHE", "X", 5),
                        _build_residue("GLY", "X", 6),
                        _build_residue("HIS", "X", 7),
                        _build_residue("ILE", "X", 8),
                        _build_residue("LYS", "X", 9),
                        _build_residue("LEU", "X", 10),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="alphafold-template",
        ),
        sequence="ACDEFGHIKL",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source_structure.constitution.chain("A")),
        UniProtSequenceRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            primary_accession="P12345",
            isoform_accession=None,
            sequence="ACDEFGHIKL",
        ),
    )
    reconstruction_specs = build_alphafold_span_reconstruction_specs(
        source_structure=source_structure,
        alignment=alignment,
        artifact=artifact,
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=reconstruction_specs,
        ),
    )

    chain = result.structure.constitution.chain("A")
    assert chain.residue_ids() == (
        ResidueId("A", 1),
        ResidueId("A", 2),
        ResidueId("A", 3),
        ResidueId("A", 4),
        ResidueId("A", 5),
        ResidueId("A", 6),
        ResidueId("A", 7),
        ResidueId("A", 8),
        ResidueId("A", 9),
        ResidueId("A", 10),
    )
    assert tuple(residue.component_id for residue in chain.residues) == (
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
    )
    assert result.requested_goal_completion_verdict() is (
        RequestedGoalCompletionVerdict.NOT_REQUESTED
    )
    assert tuple(repair.kind for repair in result.repairs) == (
        RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED,
        RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED,
    )


def test_internal_span_reconstruction_closes_perturbed_donor_and_updates_topology() -> (
    None
):
    """A rigidly displaced torsion seed should close before atomic insertion."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    _build_residue(component_id, "A", seq_num)
                    for seq_num, component_id in (
                        (1, "ALA"),
                        (2, "CYS"),
                        (5, "PHE"),
                        (6, "GLY"),
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    source_explicit_bond = TopologyBond(
        atom_index_1=source_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "C")
        ),
        atom_index_2=source_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 2), "N")
        ),
        relationship_type=BondRelationshipType.COVALENT,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source_structure = ProteinStructure.from_payload(
        constitution=source_structure.constitution,
        geometry=source_structure.geometry,
        topology=StructureTopology(
            constitution=source_structure.constitution,
            atom_topologies=source_structure.topology.atom_topologies,
            bonds=(source_explicit_bond,),
        ),
        polymer_blueprint=source_structure.polymer_blueprint,
        provenance=source_structure.provenance,
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(
                        component_id,
                        "X",
                        seq_num,
                        donor_perturbation=True,
                        include_hydrogen=seq_num == 3,
                        b_factor=95.0,
                        excluded_atom_names=(
                            frozenset({"N"})
                            if seq_num == 2
                            else frozenset({"O"})
                            if seq_num == 5
                            else frozenset()
                        ),
                    )
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3), ResidueId("A", 4)),
            preceding_residue_id=ResidueId("A", 2),
            following_residue_id=ResidueId("A", 5),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3), ResidueId("X", 4)),
    )
    original_positions = tuple(
        position for _, position in source_structure.geometry.iter_positions()
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert not result.issues
    assert len(result.repairs) == 1
    assert "CCD closure" in (result.repairs[0].details or "")
    assert tuple(
        _structure_position(result.structure, atom_ref)
        for atom_ref in (
            AtomRef(ResidueId("A", 1), "N"),
            AtomRef(ResidueId("A", 2), "C"),
            AtomRef(ResidueId("A", 5), "N"),
            AtomRef(ResidueId("A", 6), "C"),
        )
    ) == (
        original_positions[0],
        original_positions[6],
        original_positions[8],
        original_positions[14],
    )

    inserted_residue = result.structure.constitution.residue_or_ligand(
        ResidueId("A", 3)
    )
    assert inserted_residue is not None
    assert all(not atom_site.is_hydrogen() for atom_site in inserted_residue.atom_sites)
    inserted_geometry = result.structure.residue_geometry(
        result.structure.constitution.residue_index(ResidueId("A", 3))
    )
    assert all(
        atom_geometry.b_factor is None
        for atom_geometry in inserted_geometry.atoms_by_name.values()
    )

    expected_bonds = (
        (AtomRef(ResidueId("A", 2), "C"), AtomRef(ResidueId("A", 3), "N")),
        (AtomRef(ResidueId("A", 3), "N"), AtomRef(ResidueId("A", 3), "CA")),
        (AtomRef(ResidueId("A", 3), "CA"), AtomRef(ResidueId("A", 3), "C")),
        (AtomRef(ResidueId("A", 3), "C"), AtomRef(ResidueId("A", 3), "O")),
        (AtomRef(ResidueId("A", 3), "C"), AtomRef(ResidueId("A", 4), "N")),
        (AtomRef(ResidueId("A", 4), "C"), AtomRef(ResidueId("A", 5), "N")),
    )
    for left_ref, right_ref in expected_bonds:
        left_index = result.structure.constitution.resolve_atom_index(left_ref)
        right_index = result.structure.constitution.resolve_atom_index(right_ref)
        assert left_index is not None
        assert right_index is not None
        bond = result.structure.topology.bond_between(left_index, right_index)
        assert bond is not None
        assert bond.provenance in {
            BondProvenance.TEMPLATE_RESOLVED,
            BondProvenance.SEQUENCE_INFERRED,
        }
    carried_bond = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C")),
        result.structure.constitution.atom_index(AtomRef(ResidueId("A", 2), "N")),
    )
    assert carried_bond is not None
    assert carried_bond.provenance is BondProvenance.SOURCE_EXPLICIT

    pdb_round_trip = read_structure_string(
        write_structure_string(result.structure, FileFormat.PDB),
        FileFormat.PDB,
    )
    for left_ref, right_ref in expected_bonds:
        round_trip_left_index = pdb_round_trip.constitution.resolve_atom_index(left_ref)
        round_trip_right_index = pdb_round_trip.constitution.resolve_atom_index(
            right_ref
        )
        assert round_trip_left_index is not None
        assert round_trip_right_index is not None
        assert (
            pdb_round_trip.topology.bond_between(
                round_trip_left_index,
                round_trip_right_index,
            )
            is not None
        )


def test_single_residue_span_reconstruction_uses_residue_span_event_scope() -> None:
    """A one-residue gap should remain a semantic span operation."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    _build_residue(component_id, "A", seq_num)
                    for seq_num, component_id in (
                        (1, "ALA"),
                        (2, "CYS"),
                        (4, "GLU"),
                        (5, "PHE"),
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(
                        component_id,
                        "X",
                        seq_num,
                        donor_perturbation=True,
                    )
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3),),
            preceding_residue_id=ResidueId("A", 2),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3),),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert not result.issues
    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 3))
    assert len(result.repairs) == 1
    assert result.repairs[0].scope.residue_ids == (ResidueId("A", 3),)


def test_span_reconstruction_rejects_unsupported_donor_component_atomically() -> None:
    """Insertion must not materialize a residue without canonical topology."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                    _build_residue("GLU", "A", 4),
                    _build_residue("PHE", "A", 5),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ALA", "X", 1),
                    _build_residue("CYS", "X", 2),
                    _build_residue("ZZZ", "X", 3),
                    _build_residue("GLU", "X", 4),
                    _build_residue("PHE", "X", 5),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3),),
            preceding_residue_id=ResidueId("A", 2),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3),),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 3)) is None
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
    )
    assert "ZZZ" in result.issues[0].message


def test_span_reconstruction_rejects_unmodeled_donor_atoms_atomically() -> None:
    """Known components must not insert heavy atoms outside canonical chemistry."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                    _build_residue("GLU", "A", 4),
                    _build_residue("PHE", "A", 5),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ALA", "X", 1),
                    _build_residue("CYS", "X", 2),
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("X", 3),
                        atoms=(
                            *tuple(
                                _build_atom(
                                    atom_name,
                                    _ideal_backbone_position(3, atom_name),
                                )
                                for atom_name in ("N", "CA", "C", "O")
                            ),
                            _build_atom(
                                "CBX",
                                _ideal_backbone_position(3, "CA").with_offset(
                                    0.0,
                                    1.0,
                                    0.0,
                                ),
                            ),
                        ),
                    ),
                    _build_residue("GLU", "X", 4),
                    _build_residue("PHE", "X", 5),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3),),
            preceding_residue_id=ResidueId("A", 2),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3),),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 3)) is None
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
    )
    assert "CBX" in result.issues[0].message


def test_span_reconstruction_rejects_mirrored_standard_alpha_carbon() -> None:
    """A mirrored standard donor residue must not enter the canonical structure."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                    _build_residue("GLU", "A", 4),
                    _build_residue("PHE", "A", 5),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ALA", "X", 1),
                    _build_residue("CYS", "X", 2),
                    _build_residue(
                        "ALA",
                        "X",
                        3,
                        alpha_carbon_orientation_sign=-1,
                    ),
                    _build_residue("GLU", "X", 4),
                    _build_residue("PHE", "X", 5),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3),),
            preceding_residue_id=ResidueId("A", 2),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3),),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 3)) is None
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
    )
    assert "alpha-carbon chirality" in result.issues[0].message


def test_span_reconstruction_failure_is_atomic_for_unreachable_anchor() -> None:
    """An unreachable anchor should leave the requested residue absent."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                    _build_residue("PHE", "A", 5, translation_x=100.0),
                    _build_residue("GLY", "A", 6, translation_x=100.0),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3), ResidueId("A", 4)),
            preceding_residue_id=ResidueId("A", 2),
            following_residue_id=ResidueId("A", 5),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3), ResidueId("X", 4)),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 3)) is None
    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 4)) is None
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
    )


def test_span_reconstruction_reports_missing_source_anchor_without_crashing() -> None:
    """A stale explicit request should degrade to one structured atomic failure."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (_build_residue("GLU", "A", 4),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId("A", 2),
            absent_residue_ids=(ResidueId("A", 3),),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3),),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert result.structure == source_structure
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
    )
    assert "source anchor A:2 is absent" in result.issues[0].message


def test_span_reconstruction_rejects_twisted_peptide_junction_atomically() -> None:
    """Endpoint closure must not conceal a nonplanar carbonyl junction."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    _build_residue(
                        component_id,
                        "A",
                        seq_num,
                        translation_x=2.5 if seq_num >= 5 else 0.0,
                    )
                    for seq_num, component_id in (
                        (1, "ALA"),
                        (2, "CYS"),
                        (5, "PHE"),
                        (6, "GLY"),
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3), ResidueId("A", 4)),
            preceding_residue_id=ResidueId("A", 2),
            following_residue_id=ResidueId("A", 5),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3), ResidueId("X", 4)),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 3)) is None
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
    )
    assert "peptide-junction" in result.issues[0].message


def test_terminal_span_reconstruction_rejects_invalid_internal_junction() -> None:
    """A one-anchor projection must still validate junctions inside the span."""

    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue("CYS", "A", 2),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    _build_residue("ALA", "X", 1),
                    _build_residue("CYS", "X", 2),
                    _build_residue("ASP", "X", 3),
                    _build_residue("GLU", "X", 4, translation_x=3.0),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="donor",
    )
    reconstruction = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            absent_residue_ids=(ResidueId("A", 3), ResidueId("A", 4)),
            preceding_residue_id=ResidueId("A", 2),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 3), ResidueId("X", 4)),
    )

    result = process_structure(
        source_structure,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(reconstruction,),
        ),
    )

    assert result.structure.constitution.residue_or_ligand(ResidueId("A", 3)) is None
    assert tuple(issue.kind for issue in result.issues) == (
        ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
    )
    assert "internal peptide-junction" in result.issues[0].message


def _structure_with_internal_gap(
    *,
    second_component_id: str,
) -> ProteinStructure:
    """Return one chain missing reference positions three and four."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _build_residue("ALA", "A", 1),
                    _build_residue(second_component_id, "A", 2),
                    _build_residue("PHE", "A", 5),
                    _build_residue("GLY", "A", 6),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="source",
    )


def _complete_six_residue_structure() -> ProteinStructure:
    """Return one complete six-residue AlphaFold test structure."""

    return build_structure(
        chains=(
            chain_payload(
                "X",
                tuple(
                    _build_residue(component_id, "X", seq_num)
                    for seq_num, component_id in enumerate(
                        ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY"),
                        start=1,
                    )
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="alphafold-template",
    )


def _uniprot_record(*, sequence: str) -> UniProtSequenceRecord:
    """Return one UniProt test record for the shared accession."""

    return UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(accession="P12345"),
        primary_accession="P12345",
        isoform_accession=None,
        sequence=sequence,
    )


def _alphafold_artifact(
    structure: ProteinStructure,
    *,
    sequence: str,
) -> AlphaFoldStructureArtifact:
    """Return one AlphaFold artifact wrapping a canonical template structure."""

    model = AlphaFoldModelRecord(
        uniprot_reference=UniProtSequenceReference(accession="P12345"),
        entry_id="AF-P12345-F1-model_v4",
        model_entity_id="1",
        provider_id="AlphaFoldDB",
        tool_used="AlphaFold Monomer",
        sequence=sequence,
        pdb_url="https://example.org/alphafold/model.pdb",
        source_api_url="https://example.org/alphafold/api",
    )
    return AlphaFoldStructureArtifact(
        model=model,
        file_format=FileFormat.PDB,
        structure_text=write_structure_string(structure, FileFormat.PDB),
        source_url="https://example.org/alphafold/model.pdb",
    )


def _build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    *,
    donor_perturbation: bool = False,
    include_hydrogen: bool = False,
    b_factor: float | None = None,
    translation_x: float = 0.0,
    alpha_carbon_orientation_sign: int | None = None,
    is_hetero: bool = False,
    excluded_atom_names: frozenset[str] = frozenset(),
) -> CanonicalResiduePayload:
    """Return one canonical backbone-complete test residue."""

    residue_id = ResidueId(chain_id=chain_id, seq_num=seq_num)
    atom_names = tuple(
        atom_name
        for atom_name in (
            ("N", "CA", "C", "O")
            + (("CB",) if alpha_carbon_orientation_sign is not None else ())
            + (("H",) if include_hydrogen else ())
        )
        if atom_name not in excluded_atom_names
    )
    return residue_payload(
        component_id=component_id,
        residue_id=residue_id,
        is_hetero=is_hetero,
        atoms=tuple(
            _build_atom(
                atom_name,
                _test_atom_position(
                    seq_num,
                    atom_name,
                    donor_perturbation=donor_perturbation,
                    translation_x=translation_x,
                    alpha_carbon_orientation_sign=(alpha_carbon_orientation_sign),
                ),
                b_factor=b_factor,
            )
            for atom_name in atom_names
        ),
    )


def _build_atom(
    atom_name: str,
    position: Vec3,
    *,
    b_factor: float | None = None,
) -> CanonicalAtomPayload:
    """Return one test atom with deterministic peptide geometry."""

    return atom_payload(
        name=atom_name,
        element=atom_name[0],
        position=position,
        b_factor=b_factor,
    )


def _test_atom_position(
    seq_num: int,
    atom_name: str,
    *,
    donor_perturbation: bool,
    translation_x: float,
    alpha_carbon_orientation_sign: int | None,
) -> Vec3:
    """Return an optional rigid/torsional perturbation of ideal test geometry."""

    if atom_name == "CB":
        assert alpha_carbon_orientation_sign in {-1, 1}
        position = _alpha_carbon_beta_position(
            seq_num,
            orientation_sign=alpha_carbon_orientation_sign,
        )
    elif atom_name == "H":
        position = _ideal_backbone_position(seq_num, "N").with_offset(0.0, 0.0, 1.0)
    else:
        position = _ideal_backbone_position(seq_num, atom_name)
    if donor_perturbation and seq_num >= 3:
        position = AxisRotation.from_points(
            _ideal_backbone_position(2, "CA"),
            _ideal_backbone_position(2, "C"),
        ).rotate_point(
            position,
            origin=_ideal_backbone_position(2, "CA"),
            theta_radians=0.35,
        )
    if donor_perturbation:
        position = Vec3(
            x=-position.y + 20.0,
            y=position.x - 8.0,
            z=position.z + 11.0,
        )
    return position.with_offset(translation_x, 0.0, 0.0)


def _alpha_carbon_beta_position(
    seq_num: int,
    *,
    orientation_sign: int,
) -> Vec3:
    """Return one tetrahedral beta-carbon position with selected handedness."""

    return InternalCoordinateFrame(
        _ideal_backbone_position(seq_num, "C"),
        _ideal_backbone_position(seq_num, "N"),
        _ideal_backbone_position(seq_num, "CA"),
    ).place(
        bond_length=1.521,
        bond_angle_degrees=110.4,
        dihedral_degrees=-117.0 if orientation_sign > 0 else 117.0,
    )


def _structure_position(structure: ProteinStructure, atom_ref: AtomRef) -> Vec3:
    """Return one test atom position after asserting canonical addressability."""

    atom_index = structure.constitution.resolve_atom_index(atom_ref)
    assert atom_index is not None
    return structure.geometry.position(atom_index)


def _ideal_backbone_position(seq_num: int, atom_name: str) -> Vec3:
    """Return one deterministic extended-peptide backbone position."""

    positions: dict[int, dict[str, Vec3]] = {
        1: {
            "N": Vec3(0.0, 0.0, 0.0),
            "CA": Vec3(1.458, 0.0, 0.0),
            "C": Vec3(2.010, 1.421, 0.0),
        }
    }
    for residue_number in range(1, seq_num + 1):
        residue_positions = positions[residue_number]
        residue_positions["O"] = InternalCoordinateFrame(
            residue_positions["N"],
            residue_positions["CA"],
            residue_positions["C"],
        ).place(
            bond_length=1.231,
            bond_angle_degrees=120.8,
            dihedral_degrees=135.0,
        )
        if residue_number == seq_num:
            break

        next_n = InternalCoordinateFrame(
            residue_positions["N"],
            residue_positions["CA"],
            residue_positions["C"],
        ).place(
            bond_length=1.329,
            bond_angle_degrees=116.2,
            dihedral_degrees=-45.0,
        )
        next_ca = InternalCoordinateFrame(
            residue_positions["CA"],
            residue_positions["C"],
            next_n,
        ).place(
            bond_length=1.458,
            bond_angle_degrees=121.7,
            dihedral_degrees=180.0,
        )
        next_c = InternalCoordinateFrame(
            residue_positions["C"],
            next_n,
            next_ca,
        ).place(
            bond_length=1.525,
            bond_angle_degrees=111.2,
            dihedral_degrees=-60.0,
        )
        positions[residue_number + 1] = {
            "N": next_n,
            "CA": next_ca,
            "C": next_c,
        }

    return positions[seq_num][atom_name]
