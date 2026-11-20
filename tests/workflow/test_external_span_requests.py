"""Reporting and scheduling of explicit donor-backed span requests."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.hydrogen_lengths import reference

from protrepair import process_structure
from protrepair.analysis import HydrogenLengthConvention
from protrepair.diagnostics import IssueSeverity, RepairEventKind, ValidationIssueKind
from protrepair.diagnostics.events import ValidationIssue
from protrepair.io import FileFormat, read_structure_string
from protrepair.scope import AbsentResidueSpanScope, WholeStructureScope
from protrepair.state import BackboneHeavyAtomCompletenessState
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.peptide import are_peptide_adjacent
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionReport,
    ExternalSpanReconstructionSpec,
    ExternalSpanReconstructionStatus,
    HydrogenLengthRequest,
    OrphanFragmentPolicy,
    ProcessResult,
    WorkflowPlanningContext,
    WorkflowSpanDonorAvailability,
    WorkflowTransformRequests,
    requested_process_goal,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/pdb/span-reconstruction/1ubq.pdb"
)
pytestmark = pytest.mark.workflow


@pytest.fixture
def donor_rows() -> tuple[str, ...]:
    return tuple(
        row for row in _FIXTURE.read_text().splitlines() if row.startswith("ATOM  ")
    )


def _read_rows(rows: tuple[str, ...]) -> ProteinStructure:
    return read_structure_string("\n".join((*rows, "END", "")), FileFormat.PDB)


def _separate_chain_spans(
    rows: tuple[str, ...], chains: str = "A", *, missing_first_anchor: bool = False
) -> tuple[ProteinStructure, tuple[ExternalSpanReconstructionSpec, ...]]:
    fragment = tuple(row for row in rows if 3 <= int(row[22:26]) <= 7)
    donor = _read_rows(fragment)
    source_rows: list[str] = []
    requests: list[ExternalSpanReconstructionSpec] = []
    serial = 0
    for index, chain in enumerate(chains):
        for row in fragment:
            number = int(row[22:26])
            if number == 5 or (number == 4 and index == 0 and missing_first_anchor):
                continue
            serial += 1
            x = float(row[30:38]) + 100.0 * index
            source_rows.append(
                f"{row[:6]}{serial:5d}{row[11:21]}{chain}{row[22:30]}{x:8.3f}{row[38:]}"
            )
        source_rows.append("TER")

        requests.append(_span_request(donor, ResidueId(chain, 5)))
    return _read_rows(tuple(source_rows)), tuple(requests)


def _span_request(
    donor: ProteinStructure, target: ResidueId
) -> ExternalSpanReconstructionSpec:
    return ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId(target.chain_id, target.seq_num - 1),
            following_residue_id=ResidueId(target.chain_id, target.seq_num + 1),
            absent_residue_ids=(target,),
        ),
        donor_structure=donor,
        donor_residue_ids=(ResidueId("A", target.seq_num),),
    )


def _report(result: ProcessResult) -> ExternalSpanReconstructionReport:
    report = result.external_span_reconstruction_report
    assert report is not None
    return report


def _budget(limit: int) -> WorkflowPlanningContext:
    return WorkflowPlanningContext(
        span_donor_availability=WorkflowSpanDonorAvailability.AVAILABLE,
        max_speculative_nodes=limit,
    )


@pytest.mark.parametrize("bad_alternative", [False, True])
def test_span_report_attributes_only_the_selected_donor(
    donor_rows: tuple[str, ...], *, bad_alternative: bool
) -> None:
    source, (request,) = _separate_chain_spans(donor_rows)
    fragment = tuple(row for row in donor_rows if 3 <= int(row[22:26]) <= 7)
    alternative_rows = tuple(
        (
            row[:12] + " QX " + row[16:]
            if bad_alternative
            else row[:30] + f"{float(row[30:38]) + 0.04:8.3f}" + row[38:]
        )
        if int(row[22:26]) == 5 and row[12:16].strip() == "CB"
        else row
        for row in fragment
    )
    alternative = replace(request, donor_structure=_read_rows(alternative_rows))
    assert alternative != request
    assert (
        alternative.donor_structure.provenance.ingress.source_name
        == request.donor_structure.provenance.ingress.source_name
    )

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(alternative, request, replace(request)),
        ),
    )

    outcomes = _report(result).outcomes
    assert tuple(outcome.request_index for outcome in outcomes) == (0, 1)
    assert {outcome.status for outcome in outcomes} == {
        ExternalSpanReconstructionStatus.APPLIED,
        ExternalSpanReconstructionStatus.PRESENT_WITHOUT_THIS_REQUEST,
    }
    assert all(outcome.attempt_count == 1 for outcome in outcomes)
    assert all(outcome.missing_residue_ids == () for outcome in outcomes)
    assert outcomes[1].successful_attempt_count == 1
    assert outcomes[0].successful_attempt_count == int(not bad_alternative)
    if bad_alternative:
        assert (
            outcomes[0].status
            is ExternalSpanReconstructionStatus.PRESENT_WITHOUT_THIS_REQUEST
        )
        failure = outcomes[0].last_failure
        assert failure is not None
        assert failure.kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
        assert failure.scope.residue_ids == request.scope.absent_residue_ids
        assert failure.provenance_origins
        assert failure not in result.issues
    else:
        assert all(outcome.last_failure is None for outcome in outcomes)
        cb = AtomRef(ResidueId("A", 5), "CB")
        selected_cb = result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(cb)
        ).position
        returned_donor_indices = []
        for index, spec in enumerate((alternative, request)):
            control = process_structure(
                source,
                transform_requests=WorkflowTransformRequests(
                    external_span_reconstructions=(spec,)
                ),
            )
            control_cb = control.structure.geometry.atom_geometry(
                control.structure.constitution.atom_index(cb)
            ).position
            if selected_cb == control_cb:
                returned_donor_indices.append(index)

        (returned_index,) = returned_donor_indices
        assert (
            outcomes[returned_index].status is ExternalSpanReconstructionStatus.APPLIED
        )
    assert not result.has_errors()


def test_span_report_distinguishes_present_and_unattempted_requests(
    donor_rows: tuple[str, ...],
) -> None:
    source, (request,) = _separate_chain_spans(donor_rows)
    assert process_structure(source).external_span_reconstruction_report is None

    requests = WorkflowTransformRequests(external_span_reconstructions=(request,))
    already_present = process_structure(
        request.donor_structure, transform_requests=requests
    )
    (present,) = _report(already_present).outcomes
    assert present.status is ExternalSpanReconstructionStatus.ALREADY_PRESENT
    assert present.missing_residue_ids == ()
    assert present.attempt_count == 0
    assert not already_present.repairs

    exhausted = process_structure(
        source, transform_requests=requests, planning_context=_budget(1)
    )
    (unattempted,) = _report(exhausted).outcomes
    assert unattempted.status is ExternalSpanReconstructionStatus.NOT_ATTEMPTED
    assert unattempted.missing_residue_ids == request.scope.absent_residue_ids
    assert unattempted.attempt_count == 0
    assert unattempted.last_failure is None
    assert exhausted.branch_report is not None
    assert exhausted.branch_report.search_exhausted()


def test_span_report_keeps_actual_failure_after_inherited_issue_is_filtered(
    donor_rows: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, (request,) = _separate_chain_spans(donor_rows)
    present, missing = ResidueId("A", 4), ResidueId("A", 5)
    partial_span = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            preceding_residue_id=ResidueId("A", 3),
            absent_residue_ids=(present, missing),
            following_residue_id=ResidueId("A", 6),
        ),
        donor_structure=request.donor_structure,
        donor_residue_ids=(present, missing),
    )
    inherited_issue = ValidationIssue(
        kind=ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN,
        severity=IssueSeverity.WARNING,
        message="inherited hydrogen issue",
    )
    monkeypatch.setattr(
        "protrepair.workflow.engine.processing.diagnose_source_microstate_contradictions",
        lambda *args, **kwargs: (inherited_issue,),
    )

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(partial_span,)
        ),
    )

    (refusal,) = _report(result).outcomes
    assert refusal.status is ExternalSpanReconstructionStatus.FAILED
    assert refusal.missing_residue_ids == (missing,)
    assert refusal.attempt_count == 1
    assert refusal.successful_attempt_count == 0
    assert refusal.last_failure is not None
    assert refusal.last_failure.kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
    assert refusal.last_failure in result.issues
    assert inherited_issue not in result.issues
    assert not result.repairs


def test_span_report_distinguishes_success_on_an_unselected_branch(
    donor_rows: tuple[str, ...],
) -> None:
    source, (first, second) = _separate_chain_spans(donor_rows, "AB")
    different_donor = _read_rows(donor_rows)
    alternative = replace(first, donor_structure=different_donor)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(first, second, alternative),
        ),
        planning_context=_budget(3),
    )

    outcomes = _report(result).outcomes
    assert sum(outcome.successful_attempt_count for outcome in outcomes) >= 2
    (retained,) = tuple(
        outcome
        for outcome in outcomes
        if outcome.status is ExternalSpanReconstructionStatus.APPLIED
    )
    (graft,) = tuple(
        event
        for event in result.repairs
        if event.kind is RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED
    )
    assert graft.scope.residue_ids == retained.scope.absent_residue_ids
    (not_retained,) = tuple(
        outcome
        for outcome in outcomes
        if outcome.status is ExternalSpanReconstructionStatus.NOT_RETAINED
    )
    assert not_retained.missing_residue_ids == not_retained.scope.absent_residue_ids
    assert not_retained.successful_attempt_count == 1
    assert not_retained.last_failure is None


@pytest.mark.parametrize("missing_first_anchor", [False, True])
def test_span_only_runs_every_request_and_keeps_a_failed_first_attempt(
    donor_rows: tuple[str, ...], *, missing_first_anchor: bool
) -> None:
    source, requests = _separate_chain_spans(
        donor_rows, "ABCDEFGH", missing_first_anchor=missing_first_anchor
    )

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=requests
        ),
    )

    outcomes = _report(result).outcomes
    assert (
        tuple(outcome.status for outcome in outcomes[1:])
        == (ExternalSpanReconstructionStatus.APPLIED,) * 7
    )
    assert outcomes[0].status is (
        ExternalSpanReconstructionStatus.FAILED
        if missing_first_anchor
        else ExternalSpanReconstructionStatus.APPLIED
    )
    assert all(outcome.attempt_count == 1 for outcome in outcomes)
    assert result.has_errors() is missing_first_anchor
    if missing_first_anchor:
        failure = outcomes[0].last_failure
        assert failure is not None
        assert failure.severity is IssueSeverity.ERROR
        assert failure in result.issues
        assert (
            result.structure.constitution.residue_or_ligand(ResidueId("A", 5)) is None
        )
    assert result.branch_report is not None
    assert not result.branch_report.search_exhausted()
    assert result.branch_report.preferred_outcome().node_id.value == len(requests)

    expected = requests[1:] if missing_first_anchor else requests
    assert tuple(
        event.scope.residue_ids
        for event in result.repairs
        if event.kind is RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED
    ) == tuple(request.scope.absent_residue_ids for request in expected)
    for residue in source.constitution.iter_residues():
        for site in residue.atom_sites:
            ref = AtomRef(residue.residue_id, site.name)
            assert result.structure.geometry.atom_geometry(
                result.structure.constitution.atom_index(ref)
            ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))


def test_span_only_budget_spends_nodes_in_request_order(
    donor_rows: tuple[str, ...],
) -> None:
    source, requests = _separate_chain_spans(donor_rows, "ABCD")
    ordered = (requests[2], requests[0], requests[3], requests[1])

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=ordered,
            orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
        ),
        planning_context=_budget(3),
        hydrogen_lengths=HydrogenLengthRequest(),
    )

    outcomes = _report(result).outcomes
    assert tuple(outcome.status for outcome in outcomes) == (
        ExternalSpanReconstructionStatus.APPLIED,
        ExternalSpanReconstructionStatus.APPLIED,
        ExternalSpanReconstructionStatus.NOT_ATTEMPTED,
        ExternalSpanReconstructionStatus.NOT_ATTEMPTED,
    )
    assert tuple(outcome.attempt_count for outcome in outcomes) == (1, 1, 0, 0)
    assert tuple(outcome.scope for outcome in outcomes) == tuple(
        request.scope for request in ordered
    )
    assert result.branch_report is not None
    assert result.branch_report.search_exhausted()


@pytest.mark.parametrize("gaps", [(4, 6), (6, 4)])
def test_span_only_accepts_distinct_scopes_with_a_shared_anchor(
    donor_rows: tuple[str, ...], gaps: tuple[int, int]
) -> None:
    fragment = tuple(row for row in donor_rows if 3 <= int(row[22:26]) <= 7)
    donor = _read_rows(fragment)
    source = _read_rows(tuple(row for row in fragment if int(row[22:26]) not in gaps))
    requests = tuple(_span_request(donor, ResidueId("A", residue)) for residue in gaps)

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=requests
        ),
    )

    outcomes = _report(result).outcomes
    assert all(
        outcome.status is ExternalSpanReconstructionStatus.APPLIED
        for outcome in outcomes
    )
    assert sum(outcome.attempt_count for outcome in outcomes) == 2
    assert (
        tuple(
            event.scope.residue_ids[0].seq_num
            for event in result.repairs
            if event.kind is RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED
        )
        == gaps
    )
    chain = result.structure.constitution.chain("A")
    for index in range(3, 7):
        assert are_peptide_adjacent(
            chain.residue(ResidueId("A", index)),
            chain.residue(ResidueId("A", index + 1)),
            structure=result.structure,
        )


@pytest.mark.parametrize("mixed_with", ["histidine", "goal", "terminal_repair"])
def test_mixed_span_workflow_keeps_competing_siblings(
    donor_rows: tuple[str, ...], mixed_with: str
) -> None:
    source, requests = _separate_chain_spans(donor_rows, "AB")
    goals = (
        (
            requested_process_goal(
                scope=WholeStructureScope(),
                value=BackboneHeavyAtomCompletenessState.COMPLETE,
            ),
        )
        if mixed_with == "goal"
        else ()
    )
    hydrogen_lengths = (
        HydrogenLengthRequest(
            (reference(),), convention=HydrogenLengthConvention.NUCLEAR, repair=True
        )
        if mixed_with == "terminal_repair"
        else None
    )

    result = process_structure(
        source,
        requested_goals=goals,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=requests,
            protonate_histidines=mixed_with == "histidine",
        ),
        planning_context=_budget(3),
        hydrogen_lengths=hydrogen_lengths,
    )

    outcomes = _report(result).outcomes
    assert {outcome.status for outcome in outcomes} == {
        ExternalSpanReconstructionStatus.APPLIED,
        ExternalSpanReconstructionStatus.NOT_RETAINED,
    }
    assert all(outcome.attempt_count == 1 for outcome in outcomes)
    assert result.branch_report is not None
    assert result.branch_report.search_exhausted()
