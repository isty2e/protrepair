"""Native-return facts and their local candidate-selection transport."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from protrepair.diagnostics import IssueSeverity, ValidationIssue, ValidationIssueKind
from protrepair.errors import RefinementError
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import RegionTransformationResult, StructureDelta
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.refinement.acceptance import (
    AssessedRefinementResult,
    FocusRefinementQualityMetrics,
    RefinementAcceptanceMetrics,
    RefinementAcceptanceVerdict,
)
from protrepair.transformer.refinement.local_pipeline import assessment, runtime
from protrepair.transformer.refinement.local_pipeline.candidates import (
    RefinementExecutionBatch,
    RefinementExecutionCandidate,
)
from protrepair.transformer.refinement.local_pipeline.lineage import (
    RefinementCandidateLineage,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
)
from protrepair.transformer.refinement.speculative_planning import (
    EvaluatedSpeculativeProposal,
    SpeculativeExecution,
)


def _record(**changes) -> ContinuousRelaxationExecution:
    record = ContinuousRelaxationExecution(
        settings=ContinuousRelaxationSettings(
            profile=ContinuousRelaxationProfile.RDKIT_UFF
        ),
        movable_scope=AtomSetScope((AtomRef(ResidueId("A", 1), "H"),)),
        minimizer_status=1,
        backend_version=" 2026.03.2 ",
    )
    return replace(record, **changes)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("minimizer_status", True, TypeError),
        ("minimizer_status", 0.0, TypeError),
        ("minimizer_status", None, TypeError),
        ("minimizer_status", -1, ValueError),
        ("backend_version", "  ", ValueError),
        ("backend_version", None, TypeError),
        ("settings", None, TypeError),
        ("movable_scope", ResidueSetScope((ResidueId("A", 1),)), TypeError),
    ],
)
def test_native_return_rejects_invalid_facts(field, value, error):
    with pytest.raises(error):
        _record(**{field: value})


def test_invocation_identity_deduplication_and_immutability():
    first, second = _record(), _record()
    assert first != second
    assert first.backend_version == "2026.03.2"
    assert ContinuousRelaxationExecution.deduplicated(
        (second, first, second, first)
    ) == (second, first)
    with pytest.raises(FrozenInstanceError):
        first.minimizer_status = 0  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError, match="execution evidence"):
        ContinuousRelaxationExecution.deduplicated((None,))  # pyright: ignore[reportArgumentType]


def test_native_motion_authority_is_order_independent_but_not_subset_based():
    first, second = (AtomRef(ResidueId("A", 1), name) for name in ("H", "N"))
    record = _record(movable_scope=AtomSetScope((first, second)))
    assert record.has_motion_scope(AtomSetScope((second, first)))
    assert not record.has_motion_scope(AtomSetScope((first,)))
    assert not record.has_motion_scope(
        AtomSetScope((first, AtomRef(ResidueId("B", 1), "N")))
    )


@pytest.fixture(scope="module")
def local_request_and_candidate():
    source = add_hydrogens(
        read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    ).structure
    residue_id = ResidueId("A", 10)
    snapshot = ProteinStructureSnapshot.from_structure(source)
    atoms = source.constitution.atom_indices_for_residue_index(
        source.constitution.residue_index(residue_id)
    )
    context = ProteinTransformationContext.from_snapshot_atom_input(
        snapshot,
        AtomInput(
            tuple(atoms), AtomInputBasis.RESIDUEWISE, ResidueSetScope((residue_id,))
        ),
    )
    request = LocalRefinementRequest.from_context(
        context,
        spec=ContinuousRelaxationSettings(
            profile=ContinuousRelaxationProfile.RDKIT_UFF, context_radius_angstrom=0
        ),
        component_library=None,
        restraint_library=None,
    )
    return request, RefinementExecutionCandidate(
        context, RefinementCandidateLineage(), source
    )


def _metrics(clashes):
    return RefinementAcceptanceMetrics(FocusRefinementQualityMetrics(clashes, 0))


@pytest.mark.parametrize("accepted", (False, True))
@pytest.mark.parametrize("failed_sibling", (False, True))
def test_local_selection_retains_unselected_and_rejected_native_returns(
    local_request_and_candidate, monkeypatch, accepted, failed_sibling
):
    request, first = local_request_and_candidate
    second, failing = replace(first), replace(first)
    scope = first.context.atom_input.observed_atom_scope(first.context.source_snapshot)
    records = tuple(
        _record(settings=request.spec, movable_scope=scope) for _ in range(2)
    )
    visited = []

    def execute(self, *, request):
        visited.append(self)
        if self is failing:
            raise RefinementError("pre-native sibling failure")
        index = 0 if self is first else 1
        source = self.fallback_structure
        return SpeculativeExecution(
            self,
            RegionTransformationResult(
                source,
                StructureDelta(source.constitution, source.constitution),
                (),
                str(index),
                relaxation_executions=(records[index], records[index]),
            ),
        )

    def assess(candidate, *, request, before_metrics):
        rejection = (
            None
            if accepted
            else ValidationIssue(
                kind=ValidationIssueKind.REFINEMENT_REJECTED,
                severity=IssueSeverity.INFO,
                message="controlled rejection after native return",
            )
        )
        return EvaluatedSpeculativeProposal(
            candidate,
            AssessedRefinementResult(
                candidate.outcome,
                before_metrics,
                _metrics(1 if candidate.proposal is first else 2),
                RefinementAcceptanceVerdict.ACCEPTED
                if accepted
                else RefinementAcceptanceVerdict.REJECTED,
                rejection_issue=rejection,
            ),
        )

    candidates = (first, failing, second) if failed_sibling else (first, second)
    monkeypatch.setattr(RefinementExecutionCandidate, "execute", execute)
    monkeypatch.setattr(assessment, "assess_refinement_candidate", assess)
    monkeypatch.setattr(
        assessment,
        "measure_refinement_acceptance_metrics_for_scope",
        lambda *a, **k: _metrics(5),
    )
    monkeypatch.setattr(
        assessment,
        "revalidate_dependent_hydrogens_after_refinement",
        lambda result, **k: result,
    )
    monkeypatch.setattr(
        runtime,
        "build_refinement_execution_batch",
        lambda request: RefinementExecutionBatch(candidates),
    )

    result, profile = runtime.execute_local_refinement_pipeline_profiled(request)
    assert len(visited) == len(candidates)
    assert all(
        actual is expected for actual, expected in zip(visited, candidates, strict=True)
    )
    assert result.backend_name == "0"
    assert result.refined_structure is first.fallback_structure
    assert result.relaxation_executions == records
    assert profile.execution_candidate_count == len(candidates)
    assert profile.continuous_backend_candidate_count == len(candidates)


def test_region_return_metadata_does_not_change_structure_value_semantics(
    local_request_and_candidate,
):
    _, candidate = local_request_and_candidate
    source = candidate.fallback_structure
    result = RegionTransformationResult(
        source, StructureDelta(source.constitution, source.constitution), (), "fake"
    )
    record = _record()
    with_metadata = replace(result, relaxation_executions=(record, record))
    assert with_metadata == result
    assert repr(with_metadata) == repr(result)
    assert with_metadata.relaxation_executions == (record,)
    with pytest.raises(TypeError, match="execution evidence"):
        replace(result, relaxation_executions=(None,))
