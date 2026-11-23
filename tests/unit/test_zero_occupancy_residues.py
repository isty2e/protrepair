"""Whole-residue zero occupancy preserves a source-known internal gap."""

from dataclasses import replace
from pathlib import Path

import gemmi
import pytest

from protrepair import process_structure
from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.errors import StructureNormalizationError
from protrepair.io import FileFormat, read_structure_string, write_structure_string
from protrepair.io.ingress_policy import OccupancyPolicy, StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.scope import AbsentResidueSpanScope, WholeStructureScope
from protrepair.sources import (
    AlphaFoldModelRecord,
    AlphaFoldStructureArtifact,
    ObservedChainSequence,
    UniProtSequenceRecord,
    UniProtSequenceReference,
    align_observed_chain_to_uniprot_record,
)
from protrepair.state import HydrogenCoverageState
from protrepair.structure import AtomRef, ProteinStructure, ResidueId, StructureIngress
from protrepair.structure.peptide import are_peptide_adjacent
from protrepair.structure.sequence_boundary import PeptideSequenceBoundary
from protrepair.transformer.completion.span_reconstruction import (
    ReconstructedSpanCandidate,
    SpanChemistryConflictError,
    SpanReconstructionFailure,
    SpanReconstructionFailureKind,
    reconstruct_donor_span,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    StructureIngressOptions,
    WorkflowTransformRequests,
    build_alphafold_span_reconstruction_specs,
    requested_process_goal,
)

LEFT, GAP, RIGHT = (ResidueId("A", number) for number in (3, 4, 5))
SOURCE_SCOPE = AbsentResidueSpanScope(LEFT, RIGHT, (GAP,))
OMIT_WHOLE = StructureNormalizationPolicy(omit_zero_occupancy_polymer_residues=True)
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/pdb/span-reconstruction/1ubq-short-gaps.pdb"
)


def _rows(
    *,
    zero: frozenset[int] = frozenset({4}),
    positions: range = range(3, 6),
) -> list[str]:
    rows = []
    for row in FIXTURE.read_text().splitlines():
        if not row.startswith("ATOM  ") or (number := int(row[22:26])) not in positions:
            continue
        rows.append(row[:54] + "  0.00" + row[60:] if number in zero else row)
    return rows


def _read(
    rows: list[str],
    file_format: FileFormat = FileFormat.PDB,
    *,
    policy: StructureNormalizationPolicy = OMIT_WHOLE,
) -> ProteinStructure:
    return read_structure_string(
        _coordinate_text(rows, file_format), file_format, policy=policy
    )


def _coordinate_text(rows: list[str], file_format: FileFormat) -> str:
    text = "\n".join((*rows, "END", ""))
    if file_format is FileFormat.PDB:
        return text
    raw = gemmi.read_pdb_string(text)
    raw.setup_entities()
    return raw.make_mmcif_document().as_string()


def _is_atom(row: str, number: int, name: str) -> bool:
    return int(row[22:26]) == number and row[12:16].strip() == name


@pytest.mark.parametrize("file_format", tuple(FileFormat))
def test_whole_zero_records_are_absent_but_keep_their_identity(file_format):
    rows = _rows()
    before = read_structure_string(_coordinate_text(rows, file_format), file_format)
    assert before.constitution.chain("A").residue(GAP).component_id == "PHE"
    source = _read(rows, file_format)
    ingress = source.provenance.ingress
    assert ingress.zero_occupancy_residue_omissions is not None
    (record,) = ingress.zero_occupancy_residue_omissions
    assert record.residue_id == GAP
    assert record.component_id == "PHE"
    assert record.preceding_residue_id == LEFT
    assert record.following_residue_id == RIGHT
    assert len(record.atoms) == len(
        before.constitution.chain("A").residue(GAP).atom_sites
    )
    assert source.constitution.chain("A").residue_ids() == (LEFT, RIGHT)
    assert ingress.observation is not None
    assert ingress.observation.constitution == source.constitution
    assert ingress.zero_occupancy_omissions is None
    for ref in (AtomRef(LEFT, "C"), AtomRef(RIGHT, "N")):
        assert before.geometry.position(
            before.constitution.atom_index(ref)
        ) == source.geometry.position(source.constitution.atom_index(ref))


@pytest.mark.parametrize("file_format", tuple(FileFormat))
def test_omitted_coordinates_do_not_become_missing_sequence_positions(file_format):
    if file_format is FileFormat.MMCIF:
        raw = gemmi.read_pdb_string(
            _coordinate_text(
                ["SEQRES   1 A    4  ILE PHE VAL THR", *_rows()], FileFormat.PDB
            )
        )
        raw.setup_entities()
        raw.assign_label_seq_id()
        selected = read_structure_string(
            raw.make_mmcif_document().as_string(), file_format, policy=OMIT_WHOLE
        )
        right_kind = PeptideSequenceBoundary.MISSING_CONTINUATION
    else:
        selected = _read(["SEQRES   1 A    3  ILE PHE VAL", *_rows()], file_format)
        right_kind = PeptideSequenceBoundary.CONSTRUCT_END
    assert selected.constitution.chain("A").residue_ids() == (LEFT, RIGHT)
    obs = selected.provenance.ingress.observation
    assert obs is not None
    assert {f.port: f.kind for f in obs.deposited_peptide_boundaries} == {
        AtomRef(LEFT, "N"): PeptideSequenceBoundary.CONSTRUCT_END,
        AtomRef(RIGHT, "C"): right_kind,
    }


def test_workflow_second_pass_does_not_repair_an_ingress_gap(tmp_path):
    path = tmp_path / "zero.pdb"
    path.write_text(_coordinate_text(_rows(), FileFormat.PDB))
    ingress = StructureIngressOptions(omit_zero_occupancy_polymer_residues=True)
    result = process_structure(path, ingress=ingress)
    assert result.structure.constitution.residue_or_ligand(GAP) is None
    assert not result.repairs


def test_a_close_pair_across_a_recorded_gap_is_not_a_peptide():
    rows = _rows()
    left_carbon = next(row for row in rows if _is_atom(row, 3, "C"))
    close = (
        float(left_carbon[30:38]) + 1.330,
        float(left_carbon[38:46]),
        float(left_carbon[46:54]),
    )
    rows = [
        row[:30] + "".join(f"{x:8.3f}" for x in close) + row[54:]
        if _is_atom(row, 5, "N")
        else row
        for row in rows
    ]
    source = _read(rows)
    chain = source.constitution.chain("A")
    assert not are_peptide_adjacent(
        chain.residue(LEFT), chain.residue(RIGHT), structure=source
    )
    first, second = (
        source.constitution.atom_index(ref)
        for ref in (AtomRef(LEFT, "C"), AtomRef(RIGHT, "N"))
    )
    assert source.geometry.position(first).distance_to(
        source.geometry.position(second)
    ) == pytest.approx(1.33)
    for fmt in FileFormat:
        reread = read_structure_string(write_structure_string(source, fmt), fmt)
        left = reread.constitution.atom_index(AtomRef(LEFT, "C"))
        right = reread.constitution.atom_index(AtomRef(RIGHT, "N"))
        assert reread.topology.bond_between(left, right) is None
        assert reread.provenance.ingress.zero_occupancy_residue_omissions is None


@pytest.mark.parametrize("file_format", tuple(FileFormat))
def test_nonzero_altloc_not_selected_still_prevents_whole_omission(file_format):
    rows = [
        row[:16] + "A" + row[17:] if int(row[22:26]) == 4 else row for row in _rows()
    ]
    original = next(row for row in rows if _is_atom(row, 4, "CA"))
    rows.append(
        original[:6]
        + " 9998"
        + original[11:16]
        + "B"
        + original[17:54]
        + "  0.70"
        + original[60:]
    )
    result = _read(
        rows,
        file_format,
        policy=replace(OMIT_WHOLE, occupancy_policy=OccupancyPolicy.LOWEST),
    )
    index = result.constitution.atom_index(AtomRef(GAP, "CA"))
    assert result.geometry.atom_geometry(index).altloc == "A"
    assert result.geometry.atom_geometry(index).occupancy == 0.0
    assert result.provenance.ingress.zero_occupancy_residue_omissions == ()


@pytest.mark.parametrize("token", ("  1.00", "      "))
def test_positive_or_unspecified_hydrogen_disqualifies_a_whole_gap(token):
    rows = _rows()
    atom = next(row for row in rows if _is_atom(row, 4, "N"))
    rows.append(
        atom[:6]
        + " 9998"
        + atom[11:12]
        + " H  "
        + atom[16:54]
        + token
        + atom[60:76]
        + " H  "
    )
    result = _read(rows)
    assert result.constitution.resolve_atom_index(AtomRef(GAP, "H")) is not None
    assert result.provenance.ingress.zero_occupancy_residue_omissions == ()


@pytest.mark.parametrize("token", ("?", ".", "1e-60"))
def test_only_positive_source_evidence_can_establish_a_cif_flank(token):
    document = gemmi.cif.read_string(_coordinate_text(_rows(), FileFormat.MMCIF))
    block = document.sole_block()
    positions = block.find_values("_atom_site.auth_seq_id")
    names = block.find_values("_atom_site.label_atom_id")
    index = next(
        i for i, value in enumerate(positions) if value == "3" and names[i] == "CA"
    )
    block.find_values("_atom_site.occupancy")[index] = token
    source = read_structure_string(
        document.as_string(), FileFormat.MMCIF, policy=OMIT_WHOLE
    )
    assert source.constitution.residue_or_ligand(GAP) is not None
    assert source.provenance.ingress.zero_occupancy_residue_omissions == ()


def test_pdb_blank_backbone_flank_does_not_prove_observation():
    rows = [
        row[:54] + "      " + row[60:] if _is_atom(row, 3, "CA") else row
        for row in _rows()
    ]
    assert _read(rows).provenance.ingress.zero_occupancy_residue_omissions == ()


@pytest.mark.parametrize(
    "zero", (frozenset({3, 4}), frozenset({3, 4, 5}), frozenset({5}))
)
def test_runs_and_terminal_slots_are_not_eligible(zero):
    source = _read(_rows(zero=zero))
    assert source.constitution.chain("A").residue_ids() == (LEFT, GAP, RIGHT)
    assert source.provenance.ingress.zero_occupancy_residue_omissions == ()


def test_signed_zero_is_admitted_but_positive_underflow_stays_materialized():
    signed = [
        row[:54] + " -0.00" + row[60:] if _is_atom(row, 4, "N") else row
        for row in _rows()
    ]
    assert _read(signed).constitution.residue_or_ligand(GAP) is None
    tiny = [
        row[:54] + " 1e-60" + row[60:] if _is_atom(row, 4, "N") else row
        for row in _rows()
    ]
    assert _read(tiny).constitution.residue_or_ligand(GAP) is not None


def test_mixed_component_source_slot_is_not_eligible_even_when_all_zero():
    rows = _rows()
    residue = next(row for row in rows if _is_atom(row, 4, "CA"))
    rows.insert(
        next(i for i, row in enumerate(rows) if int(row[22:26]) == 5),
        residue[:6] + " 9998" + residue[11:17] + "ALA" + residue[20:],
    )
    assert _read(rows).provenance.ingress.zero_occupancy_residue_omissions == ()


def test_declared_sequence_conflict_stays_materialized():
    consistent = "SEQRES   1 A    3  ILE PHE VAL"
    conflicting = "SEQRES   1 A    3  ILE ALA VAL"
    assert _read([consistent, *_rows()]).constitution.residue_or_ligand(GAP) is None
    assert (
        _read([conflicting, *_rows()]).constitution.residue_or_ligand(GAP) is not None
    )


def test_cif_sequence_component_must_agree_with_the_observed_slot():
    document = _cif_with_declared_sequence()
    original = read_structure_string(
        document.as_string(), FileFormat.MMCIF, policy=OMIT_WHOLE
    )
    assert original.constitution.residue_or_ligand(GAP) is None

    components = document.sole_block().find_values("_entity_poly_seq.mon_id")
    assert tuple(components) == ("ILE", "PHE", "VAL")
    components[1] = "ALA"
    conflicting = read_structure_string(
        document.as_string(), FileFormat.MMCIF, policy=OMIT_WHOLE
    )
    assert conflicting.constitution.residue_or_ligand(GAP) is not None


def _cif_with_declared_sequence() -> gemmi.cif.Document:
    rows = ["SEQRES   1 A    3  ILE PHE VAL", *_rows()]
    raw = gemmi.read_pdb_string(_coordinate_text(rows, FileFormat.PDB))
    raw.setup_entities()
    raw.assign_label_seq_id()
    return raw.make_mmcif_document()


@pytest.mark.parametrize(
    ("author_position", "atom_name", "tag", "other_label"),
    (
        ("4", "N", "label_seq_id", "3"),
        ("4", "CZ", "label_seq_id", "3"),
        ("5", "O", "label_seq_id", "2"),
        ("4", "CZ", "label_asym_id", "other"),
        ("4", "CZ", "label_entity_id", "other"),
    ),
)
def test_conflicting_cif_atom_row_label_keeps_the_gap_materialized(
    author_position, atom_name, tag, other_label
):
    document = _cif_with_declared_sequence()
    block = document.sole_block()
    positions = block.find_values("_atom_site.auth_seq_id")
    names = block.find_values("_atom_site.label_atom_id")
    row = next(
        i
        for i, name in enumerate(names)
        if positions[i] == author_position and name == atom_name
    )
    block.find_values(f"_atom_site.{tag}")[row] = other_label

    contents = document.as_string()
    before = read_structure_string(contents, FileFormat.MMCIF)
    result = read_structure_string(contents, FileFormat.MMCIF, policy=OMIT_WHOLE)

    assert result.constitution.residue_or_ligand(GAP) is not None
    assert result.provenance.ingress.zero_occupancy_residue_omissions == ()
    assert (result.constitution, result.geometry, result.topology) == (
        before.constitution,
        before.geometry,
        before.topology,
    )


@pytest.mark.parametrize(("label", "withheld"), (("2", True), ("3", False)))
def test_declared_cif_atom_label_must_support_sequence_even_if_other_rows_are_missing(
    label, withheld
):
    document = _cif_with_declared_sequence()
    block = document.sole_block()
    labels = block.find_values("_atom_site.label_seq_id")
    positions = block.find_values("_atom_site.auth_seq_id")
    names = block.find_values("_atom_site.label_atom_id")
    for i, name in enumerate(names):
        labels[i] = label if positions[i] == "4" and name == "CZ" else "?"

    result = read_structure_string(
        document.as_string(), FileFormat.MMCIF, policy=OMIT_WHOLE
    )
    assert (result.constitution.residue_or_ligand(GAP) is None) is withheld


def test_cif_label_in_later_model_does_not_prevent_omission():
    document = _cif_with_declared_sequence()
    loop = document.sole_block().find_loop("_atom_site.id").get_loop()
    row = list(loop.values[: loop.width()])
    row[loop.tags.index("_atom_site.id")] = "9999"
    row[loop.tags.index("_atom_site.pdbx_PDB_model_num")] = "2"
    row[loop.tags.index("_atom_site.label_seq_id")] = "99"
    loop.add_row(row)

    result = read_structure_string(
        document.as_string(), FileFormat.MMCIF, policy=OMIT_WHOLE
    )
    assert result.constitution.residue_or_ligand(GAP) is None


@pytest.mark.parametrize("unmapped", ("terminal_residue", "water"))
def test_pdb_seqres_without_exact_source_correspondence_keeps_the_residue(unmapped):
    rows = ["SEQRES   1 A    3  ILE PHE VAL", *_rows()]
    if unmapped == "terminal_residue":
        rows[0] = "SEQRES   1 A    4  ILE PHE VAL LYS"
    else:
        rows.append(
            "HETATM 9999  O   HOH A 100      30.000  30.000  30.000  1.00 20.00           O  "
        )

    result = _read(rows)
    assert result.constitution.residue_or_ligand(GAP) is not None
    assert result.provenance.ingress.zero_occupancy_residue_omissions == ()


@pytest.mark.parametrize("boundary", ("insertion", "break", "declared_link"))
def test_only_an_unbroken_ordinary_path_becomes_an_internal_gap(boundary):
    rows = _rows()
    if boundary == "insertion":
        rows = [
            row[:22] + "   3A" + row[27:] if int(row[22:26]) == 4 else row
            for row in rows
        ]
    elif boundary == "break":
        rows.insert(
            next(i for i, row in enumerate(rows) if int(row[22:26]) == 4), "TER"
        )
    else:
        rows.append(f"CONECT{_serial(rows, 3, 'C'):5d}{_serial(rows, 5, 'N'):5d}")
    assert _read(rows).provenance.ingress.zero_occupancy_residue_omissions == ()


def _serial(rows: list[str], number: int, name: str) -> int:
    return next(int(row[6:11]) for row in rows if _is_atom(row, number, name))


@pytest.mark.parametrize("charge", ("0+", "1+"))
def test_eligible_zero_residue_with_declared_charge_refuses(charge):
    rows = [row[:78] + charge if _is_atom(row, 4, "N") else row for row in _rows()]
    with pytest.raises(StructureNormalizationError, match=r"formal charge at A:4.N"):
        _read(rows)


def test_connection_to_an_excluded_chain_is_still_protected():
    rows = _rows()
    atom = next(row for row in rows if _is_atom(row, 4, "CA"))
    rows.extend(
        (
            "HETATM 9998  C1  LIG B   9" + atom[27:54] + "  1.00" + atom[60:],
            f"CONECT{_serial(rows, 4, 'CA'):5d}{9998:5d}",
        )
    )
    with pytest.raises(
        StructureNormalizationError, match=r"pdb_conect connection at A:4.CA"
    ):
        _read(rows, policy=replace(OMIT_WHOLE, selected_chain_ids=("A",)))


def test_independent_receipts_and_one_unfiltered_preflight():
    rows = [
        row[:54] + "  0.00" + row[60:] if _is_atom(row, 3, "CG2") else row
        for row in _rows()
    ]
    sidechain = StructureNormalizationPolicy(
        omit_zero_occupancy_sidechain_heavy_atoms=True
    )
    partial = _read(rows, policy=sidechain)
    assert partial.constitution.residue_or_ligand(GAP) is not None
    assert partial.provenance.ingress.zero_occupancy_residue_omissions is None
    with pytest.raises(StructureNormalizationError, match="re-reading"):
        apply_structure_normalization_policy(partial, policy=OMIT_WHOLE)

    both = replace(OMIT_WHOLE, omit_zero_occupancy_sidechain_heavy_atoms=True)
    result = _read(rows, policy=both)
    assert result.constitution.residue_or_ligand(GAP) is None
    assert result.constitution.resolve_atom_index(AtomRef(LEFT, "CG2")) is None
    assert len(result.provenance.ingress.zero_occupancy_residue_omissions or ()) == 1
    assert len(result.provenance.ingress.zero_occupancy_omissions or ()) == 1
    with pytest.raises(
        StructureNormalizationError, match="cannot preserve.*pdb_conect"
    ):
        _read(
            [*rows, f"CONECT{_serial(rows, 3, 'CG2'):5d}{_serial(rows, 4, 'CB'):5d}"],
            policy=both,
        )
    with pytest.raises(ValueError, match="requires a source observation"):
        StructureIngress(FileFormat.PDB, zero_occupancy_residue_omissions=())


def _donor(*, component: str = "PHE", zero: bool = False) -> ProteinStructure:
    rows = _rows(zero=frozenset({4}) if zero else frozenset())
    if component == "ALA":
        rows = [
            row[:17] + "ALA" + row[20:] if int(row[22:26]) == 4 else row
            for row in rows
            if int(row[22:26]) != 4 or row[12:16].strip() in {"N", "CA", "C", "O", "CB"}
        ]
    return read_structure_string(_coordinate_text(rows, FileFormat.PDB), FileFormat.PDB)


def _reconstruct(
    source: ProteinStructure,
    donor: ProteinStructure,
    scope: AbsentResidueSpanScope = SOURCE_SCOPE,
    donor_ids: tuple[ResidueId, ...] = (GAP,),
) -> ReconstructedSpanCandidate | SpanReconstructionFailure:
    return reconstruct_donor_span(
        source,
        scope=scope,
        donor_structure=donor,
        donor_residue_ids=donor_ids,
        donor_preceding_residue_id=LEFT,
        donor_following_residue_id=RIGHT,
    )


def test_known_gap_uses_existing_explicit_reconstruction_and_keeps_history():
    source = _read(_rows())
    donor = _donor()
    result = process_structure(
        source,
        ingress=StructureIngressOptions(omit_zero_occupancy_polymer_residues=True),
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(SOURCE_SCOPE, donor, (GAP,)),
            ),
        ),
    )
    assert result.structure.constitution.chain("A").residue(GAP).component_id == "PHE"
    assert any(
        event.kind is RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED
        for event in result.repairs
    )
    assert result.structure.provenance.ingress is source.provenance.ingress
    assert (
        apply_structure_normalization_policy(result.structure, policy=OMIT_WHOLE)
        is result.structure
    )
    stale = _reconstruct(result.structure, donor)
    assert isinstance(stale, SpanReconstructionFailure)
    assert stale.kind is SpanReconstructionFailureKind.INVALID_TARGET_STATE

    candidate = _reconstruct(source, donor)
    assert isinstance(candidate, ReconstructedSpanCandidate)
    with pytest.raises(
        SpanChemistryConflictError, match="rebuilt for another structure"
    ):
        candidate.materialize(
            source.with_provenance(source.provenance), build_default_component_library()
        )
    ordinary_gap = _read(
        [row for row in _rows() if int(row[22:26]) != 4],
        policy=StructureNormalizationPolicy(),
    )
    unbound = _reconstruct(ordinary_gap, donor)
    assert isinstance(unbound, ReconstructedSpanCandidate)
    with pytest.raises(
        SpanChemistryConflictError, match="not admitted for the recorded gap"
    ):
        unbound.materialize(source, build_default_component_library())


@pytest.mark.parametrize(
    ("component", "zero", "reason"),
    (
        ("PHE", True, "positive donor occupancy"),
        ("ALA", False, "requires PHE, not donor ALA"),
    ),
)
def test_bad_known_gap_donor_refused_on_both_paths(component, zero, reason):
    source = _read(_rows())
    donor = _donor(component=component, zero=zero)
    direct = _reconstruct(source, donor)
    assert isinstance(direct, SpanReconstructionFailure)
    assert reason in direct.message
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(SOURCE_SCOPE, donor, (GAP,)),
            ),
        ),
    )
    assert any(
        issue.kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
        and reason in issue.message
        for issue in result.issues
    )
    assert not result.repairs
    assert result.structure == source


def test_widened_request_cannot_insert_an_extra_slot_inside_the_recorded_gap():
    scope = AbsentResidueSpanScope(LEFT, RIGHT, (GAP, ResidueId("A", 4, "A")))
    outcome = _reconstruct(_read(_rows()), _donor(), scope, (GAP, RIGHT))
    assert isinstance(outcome, SpanReconstructionFailure)
    assert outcome.kind is SpanReconstructionFailureKind.INVALID_TARGET_STATE
    assert "recorded gap and flanks" in outcome.message


@pytest.mark.parametrize(("atom_name", "accepted"), (("CA", False), ("CG2", True)))
def test_donor_check_reads_only_the_used_flank_inventory(atom_name, accepted):
    rows = [
        row[:54] + "  0.00" + row[60:] if _is_atom(row, 3, atom_name) else row
        for row in _rows(zero=frozenset())
    ]
    donor = _read(rows, policy=StructureNormalizationPolicy())
    outcome = _reconstruct(_read(_rows()), donor)
    if accepted:
        assert isinstance(outcome, ReconstructedSpanCandidate)
    else:
        assert isinstance(outcome, SpanReconstructionFailure)
        assert "positive donor occupancy at A:3.CA" in outcome.message


def test_gap_facing_backbone_does_not_become_a_free_terminus():
    source = _read(_rows())
    library = build_default_component_library()
    context = PolymerMicrostateContext(source)
    for residue_id, kind in (
        (LEFT, PolymerChemicalSite.BACKBONE_C),
        (RIGHT, PolymerChemicalSite.BACKBONE_N),
    ):
        component = source.constitution.chain("A").residue(residue_id).component_id
        site = context.backbone_site(
            residue_id, library.require(component), kind, assume_free_chain_ends=True
        )
        assert site.linkage is PeptideLinkage.UNKNOWN
        assert not site.free_terminal_assumption

    result = process_structure(
        source,
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    chain = result.structure.constitution.chain("A")
    assert not chain.residue(LEFT).has_atom_site("OXT")
    assert not {"H1", "H2", "H3"} & set(chain.residue(RIGHT).atom_site_names())
    assert result.structure.constitution.residue_or_ligand(GAP) is None


def test_alphafold_builder_donor_cannot_change_the_recorded_source_component():
    source = _read(_rows())
    donor = _donor(component="ALA")
    reference = UniProtSequenceReference(accession="P12345")
    record = UniProtSequenceRecord(
        uniprot_reference=reference,
        primary_accession="P12345",
        isoform_accession=None,
        sequence="IAV",
    )
    alignment = align_observed_chain_to_uniprot_record(
        ObservedChainSequence.from_chain(source.constitution.chain("A")), record
    )
    url = "https://example.org/alphafold/model.pdb"
    model = AlphaFoldModelRecord(
        uniprot_reference=reference,
        entry_id="AF-P12345-F1-model_v4",
        model_entity_id="1",
        provider_id="AlphaFoldDB",
        tool_used="AlphaFold Monomer",
        sequence="IAV",
        pdb_url=url,
        source_api_url="https://example.org/alphafold/api",
    )
    rows = write_structure_string(donor, FileFormat.PDB).splitlines()
    sequence_numbered = "\n".join(
        row[:22] + f"{int(row[22:26]) - 2:4d}" + row[26:]
        if row.startswith("ATOM  ")
        else row
        for row in rows
    )
    artifact = AlphaFoldStructureArtifact(
        model=model,
        file_format=FileFormat.PDB,
        structure_text=sequence_numbered,
        source_url=url,
    )
    specs = build_alphafold_span_reconstruction_specs(
        source_structure=source, alignment=alignment, artifact=artifact
    )
    (spec,) = specs
    assert spec.scope == SOURCE_SCOPE
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=specs
        ),
    )
    assert not result.repairs
    assert any(
        issue.kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
        and "requires PHE, not donor ALA" in issue.message
        for issue in result.issues
    )
