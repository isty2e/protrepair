"""Native-return facts and their local candidate-selection transport."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from protrepair.diagnostics import IssueSeverity, ValidationIssue, ValidationIssueKind
from protrepair.errors import ModelInvariantError, RefinementError
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import (
    MovedAtomDelta,
    RegionTransformationResult,
    StructureDelta,
)
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.result import ContinuousRelaxationResult
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.refinement.acceptance import (
    AssessedRefinementResult,
    FocusRefinementQualityMetrics,
    RefinementAcceptanceMetrics,
    RefinementAcceptanceVerdict,
    WholeStructureParserCompatibilityMetrics,
)
from protrepair.transformer.refinement.local_pipeline import (
    assessment,
    construction,
    runtime,
)
from protrepair.transformer.refinement.local_pipeline.candidates import (
    RefinementExecutionBatch,
    RefinementExecutionCandidate,
    RefinementExecutionMode,
)
from protrepair.transformer.refinement.local_pipeline.construction import (
    PreparedRefinementCandidateBase,
)
from protrepair.transformer.refinement.local_pipeline.lineage import (
    CandidateConstructionStageKind,
    RefinementCandidateLineage,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
)
from protrepair.transformer.refinement.result import LocalRefinementResult
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
        first.minimizer_status = 0  # pyrefly: ignore[read-only]
    with pytest.raises(TypeError, match="execution evidence"):
        ContinuousRelaxationExecution.deduplicated((None,))  # pyrefly: ignore[bad-argument-type]


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


@pytest.mark.parametrize(
    "preparation_kind",
    (
        None,
        CandidateConstructionStageKind.DISCRETE_PREPARATION,
        CandidateConstructionStageKind.PARSER_WITNESS_PRE_UNTANGLE,
    ),
)
def test_preparation_keeps_original_continuous_start_only_when_it_moves(
    local_request_and_candidate: tuple[
        LocalRefinementRequest, RefinementExecutionCandidate
    ],
    monkeypatch: pytest.MonkeyPatch,
    preparation_kind: CandidateConstructionStageKind | None,
) -> None:
    request, original = local_request_and_candidate
    atom_index = request.context.atom_input.atom_indices[0]
    prepared_context = replace(request.context)
    lineage = RefinementCandidateLineage()
    if preparation_kind is not None:
        lineage = lineage.with_step(
            kind=preparation_kind, moved_atom_indices=(atom_index,)
        )

    prepared = PreparedRefinementCandidateBase(prepared_context, lineage)
    seed = replace(
        original,
        context=prepared_context,
        lineage=lineage.with_step(
            kind=CandidateConstructionStageKind.DISCRETE_SEEDING,
            moved_atom_indices=(atom_index,),
        ),
    )
    monkeypatch.setattr(
        construction, "prepare_refinement_candidate_base", lambda *a, **k: prepared
    )
    monkeypatch.setattr(
        PreparedRefinementCandidateBase,
        "build_seed_execution_candidates",
        lambda *a, **k: (seed,),
    )

    batch = construction.build_refinement_execution_batch(request)
    continuous = tuple(
        candidate
        for candidate in batch.candidates
        if candidate.execution_mode is RefinementExecutionMode.CONTINUOUS_RELAXATION
    )
    assert continuous[0].context is prepared_context
    assert continuous[0].lineage is lineage
    assert continuous[1] is seed
    assert all(
        candidate.fallback_structure is original.fallback_structure
        for candidate in batch.candidates
    )
    if preparation_kind is None:
        assert len(continuous) == len(batch.candidates) == 2
        assert batch.original_start is None
        assert batch.all_candidates() == batch.candidates
    else:
        assert len(continuous) == 2
        assert len(batch.candidates) == 3
        assert len(batch.all_candidates()) == 4
        assert (
            batch.candidates[0].execution_mode is RefinementExecutionMode.DISCRETE_ONLY
        )
        assert batch.original_start is not None
        assert batch.original_start.context is request.context
        assert batch.original_start.lineage == RefinementCandidateLineage()
        assert batch.all_candidates()[-1] is batch.original_start
        assert (
            batch.without_candidate(batch.candidates[0]).original_start
            is batch.original_start
        )
        assert batch.without_candidate(batch.original_start).original_start is None


@pytest.mark.parametrize(
    "case,selected_backend",
    (
        ("rescue", "parent"),
        ("prepared-improves", "prepared"),
        ("all-rejected", "discrete_preconditioning"),
        ("parent-equal", "discrete_preconditioning"),
        ("prepared-fails", "parent"),
        ("parent-fails", "discrete_preconditioning"),
        ("prepared-equal", "parent"),
        ("discrete-improves", "discrete_preconditioning"),
    ),
)
@pytest.mark.parametrize(
    "preparation_kind",
    (
        CandidateConstructionStageKind.DISCRETE_PREPARATION,
        CandidateConstructionStageKind.PARSER_WITNESS_PRE_UNTANGLE,
    ),
)
def test_original_start_uses_common_selection_and_its_own_delta(
    local_request_and_candidate: tuple[
        LocalRefinementRequest, RefinementExecutionCandidate
    ],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    selected_backend: str,
    preparation_kind: CandidateConstructionStageKind,
) -> None:
    request, original = local_request_and_candidate
    preparation_index, native_index = request.context.atom_input.atom_indices[:2]
    prepared_context = replace(request.context)
    prepared = PreparedRefinementCandidateBase(
        prepared_context,
        RefinementCandidateLineage().with_step(
            kind=preparation_kind,
            moved_atom_indices=(preparation_index,),
        ),
    )
    visited: list[str] = []
    records: list[ContinuousRelaxationExecution] = []
    before_reads: list[ProteinStructure] = []

    def execute_continuous(self, *, request):
        name = "parent" if self.context is original.context else "prepared"
        visited.append(name)
        if (name == "parent" and case == "parent-fails") or (
            name == "prepared" and case == "prepared-fails"
        ):
            raise RefinementError(f"{name} start not executable")

        record = _record(
            settings=request.spec,
            movable_scope=self.context.atom_input.observed_atom_scope(
                self.context.source_snapshot
            ),
        )
        records.append(record)
        source = self.context.source_snapshot.structure
        return SpeculativeExecution(
            self,
            ContinuousRelaxationResult(
                source,
                StructureDelta(
                    source.constitution,
                    source.constitution,
                    moved_atoms=(MovedAtomDelta(native_index, native_index),),
                ),
                (),
                name,
                execution=record,
            ),
        )

    def assess(candidate, *, request, before_metrics):
        name = candidate.outcome.backend_name
        rejected = True
        clashes = 6
        if name == "discrete_preconditioning":
            if case == "discrete-improves":
                rejected, clashes = False, 3
            elif case == "prepared-equal":
                rejected, clashes = False, 5
        elif name == "prepared":
            if case == "prepared-improves":
                rejected, clashes = False, 3
            elif case == "prepared-equal":
                rejected, clashes = False, 5
        elif case != "all-rejected":
            rejected, clashes = False, 5 if case == "parent-equal" else 1
        return EvaluatedSpeculativeProposal(
            candidate,
            AssessedRefinementResult(
                candidate.outcome,
                before_metrics,
                _metrics(clashes),
                RefinementAcceptanceVerdict.REJECTED
                if rejected
                else RefinementAcceptanceVerdict.ACCEPTED,
                rejection_issue=ValidationIssue(
                    kind=ValidationIssueKind.REFINEMENT_REJECTED,
                    severity=IssueSeverity.INFO,
                    message=f"rejected {name}",
                )
                if rejected
                else None,
            ),
        )

    def before_metrics(source: ProteinStructure, **kwargs):
        before_reads.append(source)
        return _metrics(5)

    monkeypatch.setattr(
        construction, "prepare_refinement_candidate_base", lambda *a, **k: prepared
    )
    monkeypatch.setattr(
        PreparedRefinementCandidateBase,
        "build_seed_execution_candidates",
        lambda *a, **k: (),
    )
    monkeypatch.setattr(
        RefinementExecutionCandidate, "execute_continuous", execute_continuous
    )
    monkeypatch.setattr(assessment, "assess_refinement_candidate", assess)
    monkeypatch.setattr(
        assessment,
        "measure_refinement_acceptance_metrics_for_scope",
        before_metrics,
    )
    monkeypatch.setattr(
        assessment,
        "revalidate_dependent_hydrogens_after_refinement",
        lambda result, **k: result,
    )

    result, profile = runtime.execute_local_refinement_pipeline_profiled(request)
    suppressed = case in {"prepared-improves", "discrete-improves"}
    assert visited == (["prepared"] if suppressed else ["prepared", "parent"])
    assert len(before_reads) == 1
    assert result.backend_name == selected_backend
    assert result.relaxation_executions == tuple(records)
    assert len(records) == (
        1 if suppressed or case in {"parent-fails", "prepared-fails"} else 2
    )
    assert profile.execution_candidate_count == 3
    assert profile.continuous_backend_candidate_count == 2
    expected_indices = (native_index,)
    if case in {"all-rejected", "parent-equal", "parent-fails"}:
        expected_indices = ()
        assert result.issues[-1].message == "rejected discrete_preconditioning"
    elif case == "discrete-improves":
        expected_indices = (preparation_index,)
    elif selected_backend == "prepared":
        expected_indices = (preparation_index, native_index)
    assert result.delta.moved_atoms == tuple(
        MovedAtomDelta(index, index) for index in expected_indices
    )


def test_equal_quality_accepted_parser_shortcut_still_skips_rescue(
    local_request_and_candidate: tuple[
        LocalRefinementRequest, RefinementExecutionCandidate
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, original = local_request_and_candidate
    lineage = RefinementCandidateLineage().with_step(
        kind=CandidateConstructionStageKind.PARSER_WITNESS_PRE_UNTANGLE,
        moved_atom_indices=request.context.atom_input.atom_indices[:1],
    )
    prepared = replace(original, context=replace(original.context), lineage=lineage)
    discrete = replace(prepared, execution_mode=RefinementExecutionMode.DISCRETE_ONLY)
    metrics = replace(
        _metrics(5),
        parser_compatibility=WholeStructureParserCompatibilityMetrics(
            rdkit_sanitize_readable=False,
            extra_proximity_bond_count=1,
            extra_heavy_proximity_bond_count=0,
        ),
    )

    def assess(candidate, *, request, before_metrics):
        return EvaluatedSpeculativeProposal(
            candidate,
            AssessedRefinementResult(
                candidate.outcome,
                before_metrics,
                metrics,
                RefinementAcceptanceVerdict.ACCEPTED,
            ),
        )

    def unexpected_native(*args, **kwargs):
        pytest.fail("accepted parser shortcut must not run any native candidate")

    monkeypatch.setattr(assessment, "assess_refinement_candidate", assess)
    monkeypatch.setattr(
        assessment,
        "measure_refinement_acceptance_metrics_for_scope",
        lambda *a, **k: metrics,
    )
    monkeypatch.setattr(
        RefinementExecutionCandidate, "execute_continuous", unexpected_native
    )
    assessed, _, _ = assessment.execute_and_assess_refinement_candidate_batch(
        RefinementExecutionBatch((discrete, prepared), original_start=original),
        request=request,
    )
    assert len(assessed.evaluated_proposals) == 1
    assert assessed.evaluated_proposals[0].execution.proposal is discrete
    assert assessed.relaxation_executions == ()


@pytest.mark.parametrize(
    "invalid", ("lineage", "mode", "duplicate", "parent", "movement")
)
def test_original_start_rejects_invalid_batch_roles(
    local_request_and_candidate: tuple[
        LocalRefinementRequest, RefinementExecutionCandidate
    ],
    invalid: str,
) -> None:
    request, original = local_request_and_candidate
    source = original.fallback_structure
    lineage = RefinementCandidateLineage().with_step(
        kind=CandidateConstructionStageKind.DISCRETE_PREPARATION,
        moved_atom_indices=request.context.atom_input.atom_indices[:1],
    )
    prepared = replace(original, lineage=lineage)
    primaries = (prepared,)
    if invalid == "lineage":
        original = replace(original, lineage=lineage)
    elif invalid == "mode":
        original = replace(
            original, execution_mode=RefinementExecutionMode.DISCRETE_ONLY
        )
    elif invalid == "duplicate":
        primaries = (prepared, original)
    elif invalid == "parent":
        other_parent = ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=source.topology,
            polymer_blueprint=source.polymer_blueprint,
            provenance=source.provenance,
        )
        primaries = (replace(prepared, fallback_structure=other_parent),)
    elif invalid == "movement":
        primaries = (replace(prepared, lineage=RefinementCandidateLineage()),)

    with pytest.raises(ModelInvariantError):
        RefinementExecutionBatch(primaries, original_start=original)


@pytest.mark.parametrize("accepted", (False, True))
@pytest.mark.parametrize("failed_sibling", (False, True))
@pytest.mark.parametrize("nested", (False, True))
def test_local_selection_retains_unselected_and_rejected_native_returns(
    local_request_and_candidate, monkeypatch, accepted, failed_sibling, nested
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
        outcome: RegionTransformationResult = ContinuousRelaxationResult(
            source,
            StructureDelta(source.constitution, source.constitution),
            (),
            str(index),
            execution=records[index],
        )
        if nested:
            outcome = LocalRefinementResult.from_selection(
                outcome, relaxation_executions=(records[index],)
            )
        return SpeculativeExecution(self, outcome)

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
    without_metadata = LocalRefinementResult.from_selection(
        result, relaxation_executions=()
    )
    with_metadata = LocalRefinementResult.from_selection(
        result, relaxation_executions=(record, record)
    )
    assert with_metadata == without_metadata
    assert repr(with_metadata) == repr(without_metadata)
    assert with_metadata.relaxation_executions == (record,)
    with pytest.raises(TypeError, match="execution evidence"):
        replace(with_metadata, relaxation_executions=(None,))  # pyrefly: ignore[bad-argument-type]


def test_native_result_requires_one_fact_and_selection_retains_it(
    local_request_and_candidate,
):
    _, candidate = local_request_and_candidate
    source = candidate.fallback_structure
    core = RegionTransformationResult(
        source, StructureDelta(source.constitution, source.constitution), (), "rdkit"
    )
    record = _record()
    native = ContinuousRelaxationResult.from_transformation(core, execution=record)
    assert native.execution is record
    assert native.refined_structure is core.refined_structure
    assert native.delta is core.delta
    assert native.issues == core.issues
    assert native.backend_name == core.backend_name
    assert native.backend_version == core.backend_version
    with pytest.raises(TypeError, match="one native return"):
        replace(native, execution=None)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(ValueError, match="selected native return"):
        LocalRefinementResult.from_selection(native, relaxation_executions=())
    with pytest.raises(ValueError, match="selected native return"):
        LocalRefinementResult.from_selection(native, relaxation_executions=(_record(),))
    selected = LocalRefinementResult.from_selection(
        native, relaxation_executions=(record, record)
    )
    assert selected.refined_structure is source
    assert selected.relaxation_executions == (record,)
    with pytest.raises(ValueError, match="selected native return"):
        LocalRefinementResult.from_selection(selected, relaxation_executions=())
    nested = LocalRefinementResult.from_selection(
        selected, relaxation_executions=(record,)
    )
    assert nested.relaxation_executions == (record,)
    other = _record()
    multiple = LocalRefinementResult.from_selection(
        selected, relaxation_executions=(record, other)
    )
    with pytest.raises(ValueError, match="selected native return"):
        LocalRefinementResult.from_selection(multiple, relaxation_executions=(record,))
    retained = LocalRefinementResult.from_selection(
        multiple, relaxation_executions=(record, other, record)
    )
    assert retained.relaxation_executions == (record, other)
