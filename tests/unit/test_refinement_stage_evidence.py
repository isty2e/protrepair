"""Native returns survive refinement projections and workflow reporting."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.refinement_type_fixtures import build_toy_structure

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.io import read_structure
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.artifacts import StructureDelta
from protrepair.transformer.base import ProjectedCodomainState
from protrepair.transformer.completion.hydrogen import repair as hydrogen_repair
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.continuous.binding import ManualContinuousRelaxationBinding
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.outcome import (
    RefinementStage,
    RefinementStageOutcome,
    merge_refinement_stage_outcome,
)
from protrepair.transformer.refinement.result import (
    LocalRefinementResult,
    RefinementTransformationResult,
)
from protrepair.transformer.refinement.spec import (
    BackboneWindowRefinementSpec,
    RepairRefinementSpec,
)
from protrepair.workflow.actions import backbone_window_refinement as backbone
from protrepair.workflow.actions import local_refinement as local
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer


@pytest.fixture(scope="module")
def source():
    return read_structure(Path("tests/fixtures/pdb/1aho.pdb"))


def _evidence(profile=ContinuousRelaxationProfile.RDKIT_UFF, *, status=1):
    return ContinuousRelaxationExecution(
        settings=ContinuousRelaxationSettings(profile=profile),
        movable_scope=AtomSetScope(
            (AtomRef(ResidueId("A", 10), "CA"), AtomRef(ResidueId("A", 10), "CB"))
        ),
        minimizer_status=status,
        backend_version="2026.03.2",
    )


def test_result_updates_and_stage_projection_preserve_execution_identity(source):
    first, second = _evidence(), _evidence()
    carrier = RefinementTransformationResult(
        source, (), (), relaxation_executions=(first,)
    )
    issue = ValidationIssue(
        kind=ValidationIssueKind.REFINEMENT_REJECTED,
        severity=IssueSeverity.WARNING,
        message="retained parent after candidate rejection",
    )
    assert carrier.with_appended_issues((issue,)).relaxation_executions == (first,)
    assert carrier.with_structure(source).relaxation_executions == (first,)
    region = LocalRefinementResult(
        source,
        StructureDelta(source.constitution, source.constitution),
        (issue,),
        "rdkit",
        relaxation_executions=(second, second),
    )
    assert region == replace(region, relaxation_executions=())
    outcome = RefinementStageOutcome.from_result(region, stage=RefinementStage.REPAIR)
    result = merge_refinement_stage_outcome(carrier, outcome)
    assert result.structure is source
    assert result.issues == (issue,)
    assert result.relaxation_executions == (first, second)


def test_ligand_hydrogen_cleanup_preserves_native_history(monkeypatch):
    source = build_toy_structure()
    assert source.constitution.ligands
    receipt = _evidence()
    carrier = RefinementTransformationResult(
        source, (), (), relaxation_executions=(receipt,)
    )
    issue = ValidationIssue(
        kind=ValidationIssueKind.STERIC_CLASH,
        severity=IssueSeverity.WARNING,
        message="remaining hydrogen contact",
    )
    monkeypatch.setattr(
        hydrogen_repair,
        "cleanup_residual_hydrogen_clashes",
        lambda *args, **kwargs: (source, (issue,)),
    )

    result = hydrogen_repair._apply_hydrogen_cleanup_stage(
        carrier,
        component_library=build_default_component_library(),
        target_residue_ids=None,
        fixed_hydrogen_atom_refs=frozenset(),
    )
    assert result.structure is source
    assert result.issues == (issue,)
    assert RefinementTransformationResult.observed_executions(result) == (receipt,)


@pytest.mark.parametrize("stage", ["local", "backbone"])
def test_workflow_refinement_lift_retains_native_history(source, stage, monkeypatch):
    first, second = _evidence(), _evidence()
    carrier = RefinementTransformationResult(
        source, (), (), relaxation_executions=(first,)
    )
    context = TransformerExecutionContext(
        component_library=build_default_component_library(),
        original_structure=source,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
    )
    if stage == "local":
        transformer = local.LocalRefinementTransformer.from_repair_refinement(
            RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_atoms(second.movable_scope.atom_refs),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationProfile.RDKIT_MMFF
                ),
            )
        )
        monkeypatch.setattr(
            local,
            "apply_repair_stage_local_refinement",
            lambda result, **kwargs: replace(
                result, relaxation_executions=(*result.relaxation_executions, second)
            ),
        )
    else:
        transformer = backbone.BackboneWindowRefinementTransformer(
            BackboneWindowRefinementSpec((ResidueId("A", 10), ResidueId("A", 11)))
        )
        monkeypatch.setattr(
            backbone,
            "execute_backbone_window_refinement",
            lambda *args, **kwargs: LocalRefinementResult(
                source,
                StructureDelta(source.constitution, source.constitution),
                (),
                "rdkit",
                relaxation_executions=(second,),
            ),
        )

    result = transformer.execute(carrier, context=context)
    assert isinstance(result, RefinementTransformationResult)
    assert result.structure is source
    assert result.relaxation_executions == (first, second)
    with pytest.raises(TypeError, match="refinement codomain"):
        transformer.lift_projected_codomain(
            carrier,
            ProjectedCodomainState(transformer.workflow_scope, source),
            context=context,
        )

    following = HeavyAtomCompletionTransformer.from_completion_scope(
        scope=ResidueSetScope((ResidueId("A", 10),))
    )
    unchanged = following.lift_projected_codomain(
        result,
        ProjectedCodomainState(following.workflow_scope, source),
        context=context,
    )
    assert isinstance(unchanged, RefinementTransformationResult)
    assert unchanged.relaxation_executions == (first, second)
