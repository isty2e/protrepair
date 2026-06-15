"""External-reference workflow integration tests."""

from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, write_structure_string
from protrepair.relation.sequence_alignment import ObservedChainSequence
from protrepair.relation.supporting_role import SupportingStructureRole
from protrepair.sources import (
    AlphaFoldModelRecord,
    AlphaFoldStructureArtifact,
    UniProtSequenceRecord,
    UniProtSequenceReference,
)
from protrepair.sources.projection import align_observed_chain_to_uniprot_record
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.workflow.contracts import (
    ExternalSpanGapSelectionPolicy,
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
    gap = reconstruction_specs[0].blueprint_coverage_gap
    assert gap.absent_sequence_positions == (3, 4)
    assert gap.preceding_residue_id == ResidueId("A", 2)
    assert gap.following_residue_id == ResidueId("A", 5)

    assert reconstruction_specs[0].absent_span_scope().absent_residue_ids == (
        ResidueId("A", 3),
        ResidueId("A", 4),
    )
    assert reconstruction_specs[0].donor_residue_ids == (
        ResidueId("X", 3),
        ResidueId("X", 4),
    )
    assert reconstruction_specs[0].supporting_role is SupportingStructureRole.TEMPLATE


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
    gap = reconstruction_specs[0].blueprint_coverage_gap
    assert gap.is_prefix_terminal()
    assert reconstruction_specs[0].absent_span_scope().absent_residue_ids == (
        ResidueId("A", 1),
        ResidueId("A", 2),
    )
    assert reconstruction_specs[0].donor_residue_ids == (
        ResidueId("X", 1),
        ResidueId("X", 2),
    )


def test_build_alphafold_span_reconstruction_specs_consumes_planning_context() -> (
    None
):
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
    assert reconstruction_specs[0].blueprint_coverage_gap.is_prefix_terminal()


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
    gap = reconstruction_specs[0].blueprint_coverage_gap
    assert gap.is_suffix_terminal()
    assert reconstruction_specs[0].absent_span_scope().absent_residue_ids == (
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
) -> CanonicalResiduePayload:
    """Return one canonical backbone-complete test residue."""

    residue_id = ResidueId(chain_id=chain_id, seq_num=seq_num)
    return residue_payload(
        component_id=component_id,
        residue_id=residue_id,
        atoms=(
            _build_atom("N", seq_num * 4 + 0),
            _build_atom("CA", seq_num * 4 + 1),
            _build_atom("C", seq_num * 4 + 2),
            _build_atom("O", seq_num * 4 + 3),
        ),
    )


def _build_atom(
    atom_name: str,
    atom_index: int,
) -> CanonicalAtomPayload:
    """Return one simple test atom with deterministic coordinates."""

    return atom_payload(
        name=atom_name,
        element=atom_name[0],
        position=Vec3(x=float(atom_index), y=0.0, z=0.0),
    )
