"""Tests for explicit local-refinement hookup after repair stages."""

from typing import cast

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

import protrepair.transformer.refinement.local_pipeline.candidates as local_candidates
from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssue,
    ValidationIssueKind,
)
from protrepair.diagnostics.clashes import prepare_clash_detection_basis
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import (
    MovedAtomDelta,
    RegionTransformationResult,
    StructureDelta,
)
from protrepair.transformer.completion.heavy import repair_heavy_atoms
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.local_pipeline.candidates import (
    RefinementExecutionCandidate,
)
from protrepair.transformer.refinement.local_pipeline.lineage import (
    RefinementCandidateLineage,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
)
from protrepair.transformer.refinement.repair_stage import (
    apply_repair_stage_local_refinement,
)
from protrepair.workflow.contracts.result import ProcessResult

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def test_repair_heavy_atoms_stages_prerequisites_for_explicit_local_refinement() -> (
    None
):
    """Heavy repair should add local prerequisites before explicit refinement."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_backbone_structure(
        component_id="ALA",
        source_name="repair-heavy-local-refinement",
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )

    result = repair_heavy_atoms(structure, local_refinement=local_refinement)

    residue_site = result.structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    assert residue_site.has_atom_site("CB")
    assert any(atom_site.element == "H" for atom_site in residue_site.atom_sites)
    assert not any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED for issue in result.issues
    )
    assert any(
        repair.kind is RepairEventKind.LOCAL_REFINEMENT_APPLIED
        and repair.details is not None
        and repair.details.startswith("repair-stage local refinement via ")
        for repair in result.repairs
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
def test_local_refinement_stage_stages_passive_context_hydrogens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local refinement should hydrogenate passive template-less context first."""

    structure, residue_id, ligand_residue_id = (
        _passive_context_retained_non_polymer_structure()
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    refinement_call_count = 0

    def fake_refine_local_region(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    ):
        del allow_retained_non_polymer_rdkit_fallback
        del retained_non_polymer_chemistry_evidence
        nonlocal refinement_call_count
        refinement_call_count += 1
        snapshot = context.source_snapshot
        ligand_site = snapshot.structure.constitution.residue_or_ligand(
            ligand_residue_id
        )
        assert ligand_site is not None
        assert any(atom_site.element == "H" for atom_site in ligand_site.atom_sites)
        assert component_library is not None
        assert spec.force_field is ContinuousRelaxationForceField.UFF
        return RegionTransformationResult(
            refined_structure=snapshot.structure,
            delta=StructureDelta(
                before_constitution=snapshot.structure.constitution,
                after_constitution=snapshot.structure.constitution,
                moved_atoms=(
                    MovedAtomDelta(
                        before_atom_index=snapshot.structure.constitution.atom_index(
                            AtomRef(residue_id=residue_id, atom_name="CA")
                        ),
                        after_atom_index=snapshot.structure.constitution.atom_index(
                            AtomRef(residue_id=residue_id, atom_name="CA")
                        ),
                    ),
                ),
            ),
            issues=(),
            backend_name="rdkit",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )

    result = apply_repair_stage_local_refinement(
        ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        ),
        local_refinement=local_refinement,
        component_library=build_default_component_library(),
    )
    ligand_site = result.structure.constitution.residue_or_ligand(ligand_residue_id)
    assert ligand_site is not None

    assert refinement_call_count == 1
    assert any(atom_site.element == "H" for atom_site in ligand_site.atom_sites)
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == ligand_residue_id
        for repair in result.repairs
    )
    assert any(
        repair.kind is RepairEventKind.LOCAL_REFINEMENT_APPLIED
        for repair in result.repairs
    )
    assert not any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED for issue in result.issues
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
def test_local_refinement_stage_respects_strict_passive_context_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict local refinement must surface the canonical fallback blocker."""

    structure, residue_id, ligand_residue_id = (
        _passive_context_retained_non_polymer_structure()
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )

    def fake_refine_local_region(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    ):
        del (
            context,
            spec,
            component_library,
            allow_retained_non_polymer_rdkit_fallback,
            retained_non_polymer_chemistry_evidence,
        )
        raise AssertionError("strict fallback blocker should stop local refinement")

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )

    result = apply_repair_stage_local_refinement(
        ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        ),
        local_refinement=local_refinement,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    ligand_site = result.structure.constitution.residue_or_ligand(ligand_residue_id)
    assert ligand_site is not None

    assert all(atom_site.element != "H" for atom_site in ligand_site.atom_sites)
    assert not any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == ligand_residue_id
        for repair in result.repairs
    )
    fallback_issues = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_BLOCKED
    )
    assert len(fallback_issues) == 1
    assert fallback_issues[0].residue_id == ligand_residue_id
    assert not any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
        for issue in result.issues
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit evidence chemistry")
def test_local_refinement_stage_threads_retained_ligand_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict local refinement should stage retained ligand Hs from evidence."""

    structure, residue_id, ligand_residue_id = (
        _passive_context_retained_non_polymer_structure()
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    retained_evidence = (
        RetainedNonPolymerChemistryOverride(
            residue_id=ligand_residue_id,
            smiles="CO",
            heavy_atom_names=("C1", "O1"),
        ).to_evidence(),
    )
    refinement_call_count = 0

    def fail_passive_fallback(*args, **kwargs):
        raise AssertionError("evidence-backed local refinement called fallback")

    def fake_refine_local_region(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    ):
        del spec
        nonlocal refinement_call_count
        refinement_call_count += 1
        assert allow_retained_non_polymer_rdkit_fallback is False
        assert retained_non_polymer_chemistry_evidence == retained_evidence
        snapshot = context.source_snapshot
        ligand_site = snapshot.structure.constitution.residue_or_ligand(
            ligand_residue_id
        )
        assert ligand_site is not None
        assert any(atom_site.element == "H" for atom_site in ligand_site.atom_sites)
        assert component_library is not None
        return RegionTransformationResult(
            refined_structure=snapshot.structure,
            delta=StructureDelta(
                before_constitution=snapshot.structure.constitution,
                after_constitution=snapshot.structure.constitution,
            ),
            issues=(),
            backend_name="rdkit",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )
    monkeypatch.setattr(
        "protrepair.state.retained_non_polymer_chemistry."
        "infer_retained_non_polymer_rdkit_fallback",
        fail_passive_fallback,
    )

    result = apply_repair_stage_local_refinement(
        ProcessResult(
            structure=structure,
            repairs=(),
            issues=(),
            analyses=None,
        ),
        local_refinement=local_refinement,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
        retained_non_polymer_chemistry_evidence=retained_evidence,
    )
    ligand_site = result.structure.constitution.residue_or_ligand(ligand_residue_id)
    assert ligand_site is not None

    assert refinement_call_count == 1
    assert any(atom_site.element == "H" for atom_site in ligand_site.atom_sites)
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == ligand_residue_id
        for repair in result.repairs
    )


def test_refinement_candidate_threads_retained_ligand_policy_to_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution must use the same retained-ligand policy as readiness binding."""

    structure, residue_id, ligand_residue_id = (
        _passive_context_retained_non_polymer_structure()
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    retained_evidence = (
        RetainedNonPolymerChemistryOverride(
            residue_id=ligand_residue_id,
            smiles="CO",
            heavy_atom_names=("C1", "O1"),
        ).to_evidence(),
    )
    readiness_policies: list[bool] = []
    readiness_evidence: list[tuple[RetainedNonPolymerChemistryEvidence, ...]] = []
    problem_policies: list[bool] = []
    problem_evidence: list[tuple[RetainedNonPolymerChemistryEvidence, ...]] = []
    sentinel_problem = cast(ContinuousRelaxationProblem, object())

    def fake_derive_atom_scope_facts(
        *args,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
        **kwargs,
    ) -> object:
        del args, kwargs
        readiness_policies.append(allow_retained_non_polymer_rdkit_fallback)
        readiness_evidence.append(retained_non_polymer_chemistry_evidence)
        return object()

    def fake_require_atom_scope_execution(facts: object) -> None:
        del facts

    def fake_problem_from_inputs(
        *args,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
        **kwargs,
    ) -> ContinuousRelaxationProblem:
        del args, kwargs
        problem_policies.append(allow_retained_non_polymer_rdkit_fallback)
        problem_evidence.append(retained_non_polymer_chemistry_evidence)
        return sentinel_problem

    class RecordingBackend:
        observed_problem: ContinuousRelaxationProblem | None = None

        def relax(
            self,
            problem: ContinuousRelaxationProblem,
            *,
            restraint_library,
        ) -> RegionTransformationResult:
            self.observed_problem = problem
            return RegionTransformationResult(
                refined_structure=structure,
                delta=StructureDelta(
                    before_constitution=structure.constitution,
                    after_constitution=structure.constitution,
                ),
                issues=(),
                backend_name="fake",
            )

    monkeypatch.setattr(
        local_candidates,
        "derive_atom_scope_continuous_relaxation_facts",
        fake_derive_atom_scope_facts,
    )
    monkeypatch.setattr(
        local_candidates,
        "require_atom_scope_continuous_relaxation_execution",
        fake_require_atom_scope_execution,
    )
    monkeypatch.setattr(
        local_candidates.ContinuousRelaxationProblem,
        "from_inputs",
        fake_problem_from_inputs,
    )

    component_library = build_default_component_library()
    backend = RecordingBackend()
    request = LocalRefinementRequest(
        context=ProteinTransformationContext.from_snapshot_atom_input(
            snapshot,
            local_refinement.resolve_atom_input(snapshot),
        ),
        spec=ContinuousRelaxationSettings(
            backend_name="rdkit",
            force_field=ContinuousRelaxationForceField.UFF,
        ),
        component_library=component_library,
        restraint_library=build_default_restraint_library(),
        backend=backend,
        clash_basis=prepare_clash_detection_basis(
            structure,
            component_library=component_library,
        ),
        allow_retained_non_polymer_rdkit_fallback=False,
        retained_non_polymer_chemistry_evidence=retained_evidence,
    )
    candidate = RefinementExecutionCandidate(
        context=request.context,
        lineage=RefinementCandidateLineage(),
        fallback_structure=structure,
    )

    candidate.execute_continuous(request=request)

    assert readiness_policies == [False]
    assert readiness_evidence == [retained_evidence]
    assert problem_policies == [False]
    assert problem_evidence == [retained_evidence]
    assert backend.observed_problem is sentinel_problem


def test_add_hydrogens_can_run_explicit_local_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrogenation should run explicit refinement after hydrogens are added."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_backbone_structure(
        component_id="GLY",
        source_name="repair-hydrogen-local-refinement",
    )
    settings = ContinuousRelaxationConfig().bind(ContinuousRelaxationForceField.UFF)
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    refinement_call_count = 0

    def fake_refine_local_region(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    ):
        del allow_retained_non_polymer_rdkit_fallback
        del retained_non_polymer_chemistry_evidence
        nonlocal refinement_call_count
        refinement_call_count += 1
        snapshot = context.source_snapshot
        atom_input = context.atom_input
        residue_site = snapshot.structure.constitution.residue_or_ligand(residue_id)
        assert residue_site is not None
        assert any(atom_site.element == "H" for atom_site in residue_site.atom_sites)
        assert atom_input.referenced_residue_ids() == (residue_id,)
        assert spec == settings
        assert component_library is not None
        return RegionTransformationResult(
            refined_structure=snapshot.structure,
            delta=StructureDelta(
                before_constitution=snapshot.structure.constitution,
                after_constitution=snapshot.structure.constitution,
                moved_atoms=(
                    MovedAtomDelta(
                        before_atom_index=snapshot.structure.constitution.atom_index(
                            AtomRef(
                                residue_id=residue_id,
                                atom_name=residue_site.atom_sites[0].name,
                            )
                        ),
                        after_atom_index=snapshot.structure.constitution.atom_index(
                            AtomRef(
                                residue_id=residue_id,
                                atom_name=residue_site.atom_sites[0].name,
                            )
                        ),
                    ),
                ),
            ),
            issues=(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.INVALID_GEOMETRY,
                    severity=IssueSeverity.WARNING,
                    message="mock hydrogen refinement issue",
                    residue_id=residue_id,
                ),
            ),
            backend_name="rdkit",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )

    result = add_hydrogens(structure, local_refinement=local_refinement)
    result_residue_site = result.structure.constitution.residue_or_ligand(residue_id)
    assert result_residue_site is not None

    assert any(atom_site.element == "H" for atom_site in result_residue_site.atom_sites)
    assert refinement_call_count == 1
    assert any(
        issue.kind is ValidationIssueKind.INVALID_GEOMETRY for issue in result.issues
    )
    assert any(
        repair.kind is RepairEventKind.LOCAL_REFINEMENT_APPLIED
        for repair in result.repairs
    )


def test_add_hydrogens_binds_recommended_mmff_at_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recommended workflow refinement should bind MMFF only after H exists."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_backbone_structure(
        component_id="GLY",
        source_name="repair-hydrogen-recommended-mmff",
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=RecommendedContinuousRelaxationBinding(),
    )
    refinement_call_count = 0

    def fake_refine_local_region(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    ):
        del allow_retained_non_polymer_rdkit_fallback
        del retained_non_polymer_chemistry_evidence
        nonlocal refinement_call_count
        refinement_call_count += 1
        snapshot = context.source_snapshot
        atom_input = context.atom_input
        residue_site = snapshot.structure.constitution.residue_or_ligand(residue_id)
        assert residue_site is not None
        assert any(atom_site.element == "H" for atom_site in residue_site.atom_sites)
        assert atom_input.referenced_residue_ids() == (residue_id,)
        assert spec.force_field is ContinuousRelaxationForceField.MMFF
        assert component_library is not None
        return RegionTransformationResult(
            refined_structure=snapshot.structure,
            delta=StructureDelta(
                before_constitution=snapshot.structure.constitution,
                after_constitution=snapshot.structure.constitution,
            ),
            issues=(),
            backend_name="rdkit",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )

    add_hydrogens(structure, local_refinement=local_refinement)

    assert refinement_call_count == 1


def test_add_hydrogens_retries_recommended_mmff_with_uff_on_parameterization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recommended refinement should retry UFF after MMFF parameterization failure."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_backbone_structure(
        component_id="GLY",
        source_name="repair-hydrogen-recommended-mmff-fallback",
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=RecommendedContinuousRelaxationBinding(),
    )
    attempted_force_fields: list[ContinuousRelaxationForceField] = []

    def fake_refine_local_region(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    ):
        del allow_retained_non_polymer_rdkit_fallback
        del retained_non_polymer_chemistry_evidence
        assert component_library is not None
        attempted_force_fields.append(spec.force_field)
        if spec.force_field is ContinuousRelaxationForceField.MMFF:
            raise RefinementError(
                "RDKit MMFF could not parameterize the selected "
                "continuous-relaxation region"
            )

        return RegionTransformationResult(
            refined_structure=context.source_snapshot.structure,
            delta=StructureDelta(
                before_constitution=context.source_snapshot.structure.constitution,
                after_constitution=context.source_snapshot.structure.constitution,
            ),
            issues=(),
            backend_name="rdkit",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )

    add_hydrogens(structure, local_refinement=local_refinement)

    assert attempted_force_fields == [
        ContinuousRelaxationForceField.MMFF,
        ContinuousRelaxationForceField.UFF,
    ]


def test_add_hydrogens_skips_recommended_refinement_when_mmff_and_uff_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recommended refinement should reject cleanly if both fallback attempts fail."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_backbone_structure(
        component_id="GLY",
        source_name="repair-hydrogen-recommended-mmff-double-failure",
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=RecommendedContinuousRelaxationBinding(),
    )
    attempted_force_fields: list[ContinuousRelaxationForceField] = []

    def fake_refine_local_region(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    ):
        del context
        del allow_retained_non_polymer_rdkit_fallback
        del retained_non_polymer_chemistry_evidence
        assert component_library is not None
        attempted_force_fields.append(spec.force_field)
        if spec.force_field is ContinuousRelaxationForceField.MMFF:
            raise RefinementError(
                "RDKit MMFF could not parameterize the selected "
                "continuous-relaxation region"
            )

        raise RefinementError("RDKit UFF could not build a force field")

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )

    result = add_hydrogens(structure, local_refinement=local_refinement)
    result_residue_site = result.structure.constitution.residue_or_ligand(residue_id)
    assert result_residue_site is not None

    assert attempted_force_fields == [
        ContinuousRelaxationForceField.MMFF,
        ContinuousRelaxationForceField.UFF,
    ]
    assert any(atom_site.element == "H" for atom_site in result_residue_site.atom_sites)
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED for issue in result.issues
    )
    assert all(
        repair.kind is not RepairEventKind.LOCAL_REFINEMENT_APPLIED
        for repair in result.repairs
    )


def build_backbone_structure(
    *,
    component_id: str,
    source_name: str,
) -> ProteinStructure:
    """Return one single-residue canonical backbone fixture."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id=component_id,
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.40, 1.20, 0.0)),
                            atom_payload("O", "O", Vec3(2.10, 2.35, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name=source_name,
    )


def _passive_context_retained_non_polymer_structure() -> (
    tuple[ProteinStructure, ResidueId, ResidueId]
):
    """Return a local-refinement fixture with one passive template-less ligand."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    ligand_residue_id = ResidueId(chain_id="L", seq_num=1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, 1.4, 0.0)),
                            atom_payload("OG", "O", Vec3(1.4, 2.6, 0.0)),
                            atom_payload("H1", "H", Vec3(-0.7, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(0.0, 0.7, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, -0.7, 0.0)),
                            atom_payload("HA", "H", Vec3(1.4, -0.9, 0.0)),
                            atom_payload("HB1", "H", Vec3(0.8, 1.9, 0.8)),
                            atom_payload("HB2", "H", Vec3(2.0, 1.9, -0.8)),
                            atom_payload("HG", "H", Vec3(1.4, 3.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ligand_residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                    atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="repair-local-refinement-template-less-passive-context",
    )
    return structure, residue_id, ligand_residue_id
