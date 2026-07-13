"""Unit tests for public local-refinement API resolution and boundaries."""

import ast
import importlib
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)

from protrepair.chemistry import (
    BondDefinition,
    ChemicalComponentDefinition,
    ComponentLibrary,
    HeavyAtomSemantics,
    HydrogenSemantics,
    ResidueTemplate,
)
from protrepair.chemistry.internal_coordinates import InternalCoordinateProgram
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.diagnostics import (
    IssueSeverity,
    ValidationIssue,
    ValidationIssueKind,
)
from protrepair.diagnostics.clashes import ClashDetectionBasis
from protrepair.diagnostics.parser_readability import (
    RDKitNoConectSanitizeReadabilityMetrics,
)
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.relation import (
    DensityEvidence,
    StructureEndpoint,
)
from protrepair.relation.supporting_role import SupportingStructureRole
from protrepair.scope import ResidueSetScope, Scope
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.artifacts import (
    MovedAtomDelta,
    RegionTransformationResult,
    StructureDelta,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    LocalTransformationContextSpec,
    SupportingStructureSpec,
    transform_local_region,
)
from protrepair.transformer.refinement.acceptance import (
    AssessedRefinementResult,
    FocusRefinementQualityMetrics,
    RefinementAcceptanceMetrics,
    RefinementAcceptanceVerdict,
    WholeStructureParserCompatibilityMetrics,
    WholeStructureProximityBurdenMetrics,
    refinement_has_new_severe_restraint_backed_bond_length_failure,
    refinement_has_new_stereochemistry_failure,
    refinement_metrics_regressed,
    refinement_metrics_rejected,
)
from protrepair.transformer.refinement.candidate_selection import (
    materialize_assessed_refinement_candidate,
)
from protrepair.transformer.refinement.local_pipeline.backend import (
    resolve_continuous_relaxation_backend,
)
from protrepair.transformer.refinement.local_pipeline.construction import (
    PreparedRefinementCandidateBase,
)
from protrepair.transformer.refinement.local_pipeline.lineage import (
    CandidateConstructionStageKind,
    RefinementCandidateLineage,
)
from protrepair.workflow.contracts.result import ProcessResult


def test_refine_local_region_rejects_unknown_backend() -> None:
    """Public API should reject unknown backend names at the boundary."""

    with pytest.raises(RefinementError, match="not implemented"):
        transform_local_region(
            build_toy_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_residues(
                    (ResidueId(chain_id="A", seq_num=1),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                config=ContinuousRelaxationConfig(backend_name="imaginary"),
            ),
            component_library=build_toy_component_library(),
        )


def test_refine_local_region_rejects_internal_canonical_spec_at_public_boundary() -> (
    None
):
    """Direct refinement should require the stage-typed public spec."""

    with pytest.raises(TypeError, match="DirectRegionTransformationSpec"):
        transform_local_region(
            build_toy_structure(),
            ContinuousRelaxationSettings(
                backend_name="rdkit",
                force_field=ContinuousRelaxationForceField.UFF,
            ),  # type: ignore[arg-type]
            component_library=build_toy_component_library(),
        )


def test_backend_resolution_normalizes_whitespace_and_case() -> None:
    """Backend resolution should canonicalize user-facing backend tokens."""

    backend = resolve_continuous_relaxation_backend("  RDKIT  ")
    assert type(backend).__name__ == "RdkitContinuousRelaxationBackend"


def test_refine_local_region_rejects_hydrogen_incomplete_current_domain() -> None:
    """Continuous relaxation should reject hydrogen-less current domains."""

    with pytest.raises(
        RefinementError,
        match="hydrogens to be fully realized before any force field can be bound",
    ):
        transform_local_region(
            build_hydrogenless_angle_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_residues(
                    (ResidueId(chain_id="A", seq_num=1),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
            ),
            component_library=build_angle_component_library(),
        )


def test_refine_local_region_rejects_supporting_structure_contexts_for_now() -> None:
    """Direct API should reject richer contexts until contextual execution exists."""

    with pytest.raises(
        NotImplementedError,
        match="source-only transformation contexts",
    ):
        transform_local_region(
            build_angle_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_residues(
                    (ResidueId(chain_id="A", seq_num=1),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                context=LocalTransformationContextSpec(
                    supporting_structures=(
                        SupportingStructureSpec(
                            role=SupportingStructureRole.DONOR,
                            structure=build_angle_structure(),
                        ),
                    )
                ),
            ),
            component_library=build_angle_component_library(),
        )


def test_refine_local_region_rejects_external_evidence_contexts_for_now() -> None:
    """Direct API should reject external evidence until contextual execution exists."""

    with pytest.raises(
        NotImplementedError,
        match="source-only transformation contexts",
    ):
        transform_local_region(
            build_angle_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_residues(
                    (ResidueId(chain_id="A", seq_num=1),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                context=LocalTransformationContextSpec(
                    external_evidence=(
                        DensityEvidence(
                            target_structure_endpoint=StructureEndpoint.source(
                                ResidueSetScope(
                                    residue_ids=(ResidueId(chain_id="A", seq_num=1),)
                                )
                            )
                        ),
                    )
                ),
            ),
            component_library=build_angle_component_library(),
        )


def test_refine_local_region_rejects_selected_region_geometry_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance gating should reject backend results that worsen focus geometry."""

    structure = build_angle_structure()
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=0.0),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library: RestraintLibrary,
        ) -> RegionTransformationResult:
            del restraint_library
            structure = problem.region.snapshot.structure
            residue_id = ResidueId(chain_id="A", seq_num=1)
            residue_site = structure.constitution.residue_or_ligand(residue_id)
            assert residue_site is not None
            residue_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(residue_id),
            )
            refined_structure = structure.with_updated_residue_facets(
                residue_site=residue_site,
                residue_geometry=residue_geometry.with_atom_geometry(
                    "A3",
                    residue_geometry.atom_geometry("A3").with_position(
                        Vec3(5.0, 0.0, 0.0)
                    ),
                ),
                formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
                    constitution=structure.constitution,
                    residue_index=structure.constitution.residue_index(residue_id),
                ),
            )
            return RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=refined_structure.constitution,
                    moved_atoms=tuple(
                        MovedAtomDelta(
                            before_atom_index=atom_index,
                            after_atom_index=atom_index,
                        )
                        for atom_index in problem.region.movable_atom_indices
                    ),
                ),
                issues=(),
                backend_name="fake",
            )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.backend.resolve_continuous_relaxation_backend",
        lambda backend_name: FakeBackend(),
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=build_angle_component_library(),
    )

    assert result.refined_structure == structure
    assert result.delta.moved_atoms == ()
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED for issue in result.issues
    )


def test_refine_local_region_accepts_selected_region_clash_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance gating should keep backend results that improve focus clashes."""

    structure = build_clashy_structure()
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library: RestraintLibrary,
        ) -> RegionTransformationResult:
            del restraint_library
            structure = problem.region.snapshot.structure
            residue_id = ResidueId(chain_id="A", seq_num=1)
            residue_site = structure.constitution.residue_or_ligand(residue_id)
            assert residue_site is not None
            residue_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(residue_id),
            )
            refined_structure = structure.with_updated_residue_facets(
                residue_site=residue_site,
                residue_geometry=residue_geometry.with_atom_geometry(
                    "H1",
                    residue_geometry.atom_geometry("H1").with_position(
                        Vec3(2.8, 0.0, 0.0)
                    ),
                ),
                formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
                    constitution=structure.constitution,
                    residue_index=structure.constitution.residue_index(residue_id),
                ),
            )
            return RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=refined_structure.constitution,
                    moved_atoms=tuple(
                        MovedAtomDelta(
                            before_atom_index=atom_index,
                            after_atom_index=atom_index,
                        )
                        for atom_index in problem.region.movable_atom_indices
                    ),
                ),
                issues=(
                    ValidationIssue.for_residue(
                        kind=ValidationIssueKind.STERIC_CLASH,
                        severity=IssueSeverity.WARNING,
                        message="mock backend issue",
                        residue_id=ResidueId("A", 1),
                    ),
                ),
                backend_name="fake",
            )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.backend.resolve_continuous_relaxation_backend",
        lambda backend_name: FakeBackend(),
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=build_clashy_component_library(),
    )

    refined_h1 = result.refined_structure.geometry.atom_geometry(
        result.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "H1")
        )
    )

    assert refined_h1.position == Vec3(2.8, 0.0, 0.0)
    assert result.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=result.refined_structure.constitution.atom_index(
                AtomRef(ResidueId("A", 1), "H1")
            ),
            after_atom_index=result.refined_structure.constitution.atom_index(
                AtomRef(ResidueId("A", 1), "H1")
            ),
        ),
    )
    assert all(
        issue.kind is not ValidationIssueKind.REFINEMENT_REJECTED
        for issue in result.issues
    )


def test_refine_local_region_rejects_unresolved_full_structure_sanitize_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance should reject unresolved parser-visible sanitize failure."""

    structure = build_clashy_structure()
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library: RestraintLibrary,
        ) -> RegionTransformationResult:
            del restraint_library
            structure = problem.region.snapshot.structure
            residue_id = ResidueId(chain_id="A", seq_num=1)
            residue_site = structure.constitution.residue_or_ligand(residue_id)
            assert residue_site is not None
            residue_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(residue_id),
            )
            refined_structure = structure.with_updated_residue_facets(
                residue_site=residue_site,
                residue_geometry=residue_geometry.with_atom_geometry(
                    "H1",
                    residue_geometry.atom_geometry("H1").with_position(
                        Vec3(2.8, 0.0, 0.0)
                    ),
                ),
                formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
                    constitution=structure.constitution,
                    residue_index=structure.constitution.residue_index(residue_id),
                ),
            )
            return RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=refined_structure.constitution,
                    moved_atoms=tuple(
                        MovedAtomDelta(
                            before_atom_index=atom_index,
                            after_atom_index=atom_index,
                        )
                        for atom_index in problem.region.movable_atom_indices
                    ),
                ),
                issues=(),
                backend_name="fake",
            )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.backend.resolve_continuous_relaxation_backend",
        lambda backend_name: FakeBackend(),
    )

    sanitize_readability_by_structure = {id(structure): True}

    def _measure_sanitize_readability_metrics(
        candidate_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> RDKitNoConectSanitizeReadabilityMetrics:
        del component_library
        sanitize_readable = sanitize_readability_by_structure.get(
            id(candidate_structure),
            False,
        )
        return RDKitNoConectSanitizeReadabilityMetrics(
            sanitize_readable=sanitize_readable,
            extra_proximity_bond_count=0 if sanitize_readable else 1,
            extra_heavy_proximity_bond_count=0 if sanitize_readable else 1,
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.acceptance.measure_rdkit_no_conect_sanitize_readability_metrics",
        _measure_sanitize_readability_metrics,
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=build_clashy_component_library(),
    )

    assert result.refined_structure == structure
    assert result.delta.moved_atoms == ()
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED for issue in result.issues
    )


def test_refine_local_region_can_adopt_discrete_preconditioning_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-backend movement should be selectable when continuous FF regresses."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    moved_atom_ref = AtomRef(residue_id, "H1")
    structure = build_clashy_structure()
    moved_atom_index = structure.constitution.atom_index(moved_atom_ref)
    discrete_position = Vec3(2.8, 0.0, 0.0)
    backend_position = Vec3(0.2, 0.0, 0.0)
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_atoms((moved_atom_ref,)),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
    )

    def fake_prepare_refinement_candidate_base(
        context: ProteinTransformationContext,
        *,
        component_library: ComponentLibrary,
        clash_basis: ClashDetectionBasis,
    ) -> PreparedRefinementCandidateBase:
        del component_library, clash_basis
        discrete_structure = _structure_with_updated_atom_positions(
            context.source_snapshot.structure,
            residue_id,
            {"H1": discrete_position},
        )
        return PreparedRefinementCandidateBase(
            context=ProteinTransformationContext(
                source_snapshot=context.source_snapshot.with_structure(
                    discrete_structure
                ),
                atom_input=context.atom_input,
                supporting_structures=context.supporting_structures,
                external_evidence=context.external_evidence,
                external_constraints=context.external_constraints,
            ),
            lineage=RefinementCandidateLineage().with_step(
                kind=CandidateConstructionStageKind.PARSER_WITNESS_PRE_UNTANGLE,
                moved_atom_indices=(moved_atom_index,),
            ),
        )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library: RestraintLibrary,
        ) -> RegionTransformationResult:
            del restraint_library
            backend_calls.append(problem.region.snapshot.structure)
            refined_structure = _structure_with_updated_atom_positions(
                problem.region.snapshot.structure,
                residue_id,
                {"H1": backend_position},
            )
            return RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=refined_structure.constitution,
                    moved_atoms=tuple(
                        MovedAtomDelta(
                            before_atom_index=atom_index,
                            after_atom_index=atom_index,
                        )
                        for atom_index in problem.region.movable_atom_indices
                    ),
                ),
                issues=(),
                backend_name="fake",
            )

    def fake_assess_refinement_result(
        selected_scope: Scope,
        component_library: ComponentLibrary,
        restraint_library: RestraintLibrary,
        result: RegionTransformationResult,
        *,
        before_metrics: RefinementAcceptanceMetrics,
        clash_basis: ClashDetectionBasis | None = None,
    ) -> AssessedRefinementResult:
        del selected_scope, component_library, restraint_library, clash_basis
        assessed_backend_names.append(result.backend_name)
        if result.backend_name == "discrete_preconditioning":
            return AssessedRefinementResult(
                executed_result=result,
                before_metrics=before_metrics,
                after_metrics=_refinement_metrics(rdkit_readable=True),
                verdict=RefinementAcceptanceVerdict.ACCEPTED,
            )

        return AssessedRefinementResult(
            executed_result=result,
            before_metrics=before_metrics,
            after_metrics=_refinement_metrics(
                rdkit_readable=False,
                extra_proximity_bonds=2,
                extra_heavy_proximity_bonds=2,
            ),
            verdict=RefinementAcceptanceVerdict.REJECTED,
            rejection_issue=ValidationIssue(
                kind=ValidationIssueKind.REFINEMENT_REJECTED,
                severity=IssueSeverity.INFO,
                message="mock continuous regression",
            ),
        )

    backend_calls: list[ProteinStructure] = []
    assessed_backend_names: list[str] = []
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.construction.prepare_refinement_candidate_base",
        fake_prepare_refinement_candidate_base,
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.backend.resolve_continuous_relaxation_backend",
        lambda backend_name: FakeBackend(),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.measure_refinement_acceptance_metrics_for_scope",
        lambda structure, **kwargs: _refinement_metrics(
            rdkit_readable=False,
            extra_proximity_bonds=1,
            extra_heavy_proximity_bonds=1,
        ),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.assess_refinement_result_with_before_metrics",
        fake_assess_refinement_result,
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.revalidate_dependent_hydrogens_after_refinement",
        lambda result, **kwargs: result,
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=build_clashy_component_library(),
    )

    assert backend_calls == []
    assert assessed_backend_names == ["discrete_preconditioning"]
    assert _atom_position(result.refined_structure, residue_id, "H1") == (
        discrete_position
    )
    assert result.backend_name == "discrete_preconditioning"
    assert result.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=moved_atom_index,
            after_atom_index=moved_atom_index,
        ),
    )


def test_refine_local_region_revalidates_dependent_h_after_h_only_parser_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted H-only parser failures should rederive stale dependent hydrogens."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_angle_structure()
    bad_hydrogen_position = Vec3(2.35, 1.05, 0.0)
    refreshed_hydrogen_position = Vec3(1.35, -1.10, 0.0)
    refined_heavy_position = Vec3(2.20, 1.25, 0.0)
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library: RestraintLibrary,
        ) -> RegionTransformationResult:
            del restraint_library
            refined_structure = _structure_with_updated_atom_positions(
                problem.region.snapshot.structure,
                residue_id,
                {
                    "A3": refined_heavy_position,
                    "H1": bad_hydrogen_position,
                },
            )
            return RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=refined_structure.constitution,
                    moved_atoms=tuple(
                        MovedAtomDelta(
                            before_atom_index=atom_index,
                            after_atom_index=atom_index,
                        )
                        for atom_index in problem.region.movable_atom_indices
                    ),
                ),
                issues=(),
                backend_name="fake",
            )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.backend.resolve_continuous_relaxation_backend",
        lambda backend_name: FakeBackend(),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.measure_refinement_acceptance_metrics_for_scope",
        lambda structure, **kwargs: _refinement_metrics(rdkit_readable=True),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.assess_refinement_result_with_before_metrics",
        _accepted_h_only_parser_failure,
    )
    monkeypatch.setattr(
        "protrepair.transformer.dependent_hydrogen.measure_refinement_acceptance_metrics_for_scope",
        lambda structure, **kwargs: _refinement_metrics(rdkit_readable=True),
    )
    monkeypatch.setattr(
        "protrepair.transformer.dependent_hydrogen.rdkit_no_conect_parser_failing_residue_ids",
        lambda structure, **kwargs: (residue_id,),
    )

    hydrogenation_calls: list[frozenset[ResidueId] | None] = []

    def fake_materialize_hydrogens_core(
        structure: ProteinStructure,
        component_library: ComponentLibrary | None = None,
        *,
        target_residue_ids: frozenset[ResidueId] | None = None,
        protonate_histidines: bool = False,
    ) -> ProcessResult:
        del component_library, protonate_histidines
        hydrogenation_calls.append(target_residue_ids)
        return ProcessResult(
            structure=_structure_with_updated_atom_positions(
                structure,
                residue_id,
                {"H1": refreshed_hydrogen_position},
            ),
            repairs=(),
            issues=(),
            analyses=None,
        )

    monkeypatch.setattr(
        "protrepair.transformer.completion.hydrogen.core.materialize_hydrogens_core",
        fake_materialize_hydrogens_core,
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=build_angle_component_library(),
    )

    assert hydrogenation_calls == [frozenset((residue_id,))]
    assert _atom_position(result.refined_structure, residue_id, "A3") == (
        refined_heavy_position
    )
    assert _atom_position(result.refined_structure, residue_id, "H1") == (
        refreshed_hydrogen_position
    )


def test_refine_local_region_skips_dependent_h_with_heavy_parser_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heavy parser blockers make dependent-H rematerialization non-actionable."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_angle_structure()
    bad_hydrogen_position = Vec3(2.35, 1.05, 0.0)
    refined_heavy_position = Vec3(2.20, 1.25, 0.0)
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library: RestraintLibrary,
        ) -> RegionTransformationResult:
            del restraint_library
            refined_structure = _structure_with_updated_atom_positions(
                problem.region.snapshot.structure,
                residue_id,
                {
                    "A3": refined_heavy_position,
                    "H1": bad_hydrogen_position,
                },
            )
            return RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=refined_structure.constitution,
                    moved_atoms=tuple(
                        MovedAtomDelta(
                            before_atom_index=atom_index,
                            after_atom_index=atom_index,
                        )
                        for atom_index in problem.region.movable_atom_indices
                    ),
                ),
                issues=(),
                backend_name="fake",
            )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.backend.resolve_continuous_relaxation_backend",
        lambda backend_name: FakeBackend(),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.measure_refinement_acceptance_metrics_for_scope",
        lambda structure, **kwargs: _refinement_metrics(rdkit_readable=True),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.assess_refinement_result_with_before_metrics",
        _accepted_mixed_parser_failure,
    )

    def fake_materialize_hydrogens_core(
        structure: ProteinStructure,
        component_library: ComponentLibrary | None = None,
        *,
        target_residue_ids: frozenset[ResidueId] | None = None,
        protonate_histidines: bool = False,
    ) -> ProcessResult:
        del structure, component_library, target_residue_ids, protonate_histidines
        raise AssertionError(
            "dependent-H materialization should not run while heavy parser "
            "blockers remain"
        )

    monkeypatch.setattr(
        "protrepair.transformer.completion.hydrogen.core.materialize_hydrogens_core",
        fake_materialize_hydrogens_core,
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=build_angle_component_library(),
    )

    assert _atom_position(result.refined_structure, residue_id, "A3") == (
        refined_heavy_position
    )
    assert _atom_position(result.refined_structure, residue_id, "H1") == (
        bad_hydrogen_position
    )


def test_refine_local_region_rejects_dependent_hydrogen_revalidation_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependent-H revalidation candidates should not bypass acceptance gates."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_angle_structure()
    bad_hydrogen_position = Vec3(2.35, 1.05, 0.0)
    refreshed_hydrogen_position = Vec3(1.35, -1.10, 0.0)
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library: RestraintLibrary,
        ) -> RegionTransformationResult:
            del restraint_library
            refined_structure = _structure_with_updated_atom_positions(
                problem.region.snapshot.structure,
                residue_id,
                {"H1": bad_hydrogen_position},
            )
            return RegionTransformationResult(
                refined_structure=refined_structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=refined_structure.constitution,
                    moved_atoms=tuple(
                        MovedAtomDelta(
                            before_atom_index=atom_index,
                            after_atom_index=atom_index,
                        )
                        for atom_index in problem.region.movable_atom_indices
                    ),
                ),
                issues=(),
                backend_name="fake",
            )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.backend.resolve_continuous_relaxation_backend",
        lambda backend_name: FakeBackend(),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.measure_refinement_acceptance_metrics_for_scope",
        lambda structure, **kwargs: _refinement_metrics(rdkit_readable=True),
    )
    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.assess_refinement_result_with_before_metrics",
        _accepted_h_only_parser_failure,
    )
    monkeypatch.setattr(
        "protrepair.transformer.dependent_hydrogen.measure_refinement_acceptance_metrics_for_scope",
        lambda structure, **kwargs: _refinement_metrics(
            rdkit_readable=False,
            extra_proximity_bonds=2,
        ),
    )
    monkeypatch.setattr(
        "protrepair.transformer.dependent_hydrogen.rdkit_no_conect_parser_failing_residue_ids",
        lambda structure, **kwargs: (residue_id,),
    )

    def fake_materialize_hydrogens_core(
        structure: ProteinStructure,
        component_library: ComponentLibrary | None = None,
        *,
        target_residue_ids: frozenset[ResidueId] | None = None,
        protonate_histidines: bool = False,
    ) -> ProcessResult:
        del component_library, target_residue_ids, protonate_histidines
        return ProcessResult(
            structure=_structure_with_updated_atom_positions(
                structure,
                residue_id,
                {"H1": refreshed_hydrogen_position},
            ),
            repairs=(),
            issues=(),
            analyses=None,
        )

    monkeypatch.setattr(
        "protrepair.transformer.completion.hydrogen.core.materialize_hydrogens_core",
        fake_materialize_hydrogens_core,
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=build_angle_component_library(),
    )

    assert _atom_position(result.refined_structure, residue_id, "H1") == (
        bad_hydrogen_position
    )


def test_refinement_acceptance_metrics_keep_quality_axes_orthogonal() -> None:
    """Acceptance metrics should expose focus, global proximity, and parser axes."""

    metrics = RefinementAcceptanceMetrics(
        focus_quality=FocusRefinementQualityMetrics(
            clash_count=4,
            geometry_outlier_count=2,
            clash_overlap_sum_angstrom=1.5,
        ),
        whole_structure_proximity=WholeStructureProximityBurdenMetrics(
            near_covalent_contact_count=7,
            worst_near_covalent_overlap_angstrom=1.2,
            total_near_covalent_overlap_angstrom=3.4,
        ),
        parser_compatibility=WholeStructureParserCompatibilityMetrics(
            rdkit_sanitize_readable=False,
            extra_proximity_bond_count=5,
            extra_heavy_proximity_bond_count=3,
        ),
    )

    assert metrics.focus_quality.clash_count == 4
    assert metrics.whole_structure_proximity.near_covalent_contact_count == 7
    assert metrics.parser_compatibility.extra_heavy_proximity_bond_count == 3
    assert metrics.focus_clash_count == 4
    assert metrics.whole_structure_near_covalent_contact_count == 7
    assert metrics.whole_structure_parser_extra_heavy_proximity_bond_count == 3


def test_refinement_acceptance_metrics_reject_flat_constructor_kwargs() -> None:
    """Canonical acceptance metrics should be built from orthogonal records."""

    metrics_constructor = importlib.import_module(
        "protrepair.transformer.refinement.acceptance"
    ).__dict__["RefinementAcceptanceMetrics"]
    with pytest.raises(TypeError, match="focus_clash_count"):
        metrics_constructor(
            focus_clash_count=0,
            focus_geometry_outlier_count=0,
        )


def test_refinement_acceptance_policy_avoids_flat_projection_properties() -> None:
    """Production acceptance policy should read the orthogonal metric records."""

    policy_paths = (
        Path("src/protrepair/transformer/refinement/acceptance.py"),
        Path("src/protrepair/transformer/dependent_hydrogen.py"),
        Path("src/protrepair/transformer/refinement/local_pipeline/assessment.py"),
    )
    flat_projection_attributes = {
        "focus_clash_count",
        "focus_geometry_outlier_count",
        "focus_restraint_backed_geometry_outlier_count",
        "focus_fallback_geometry_outlier_count",
        "focus_severe_restraint_backed_bond_length_outlier_count",
        "focus_clash_overlap_sum_angstrom",
        "focus_near_covalent_contact_count",
        "focus_worst_near_covalent_overlap_angstrom",
        "focus_total_near_covalent_overlap_angstrom",
        "focus_stereochemistry_violation_count",
        "whole_structure_near_covalent_contact_count",
        "whole_structure_worst_near_covalent_overlap_angstrom",
        "whole_structure_total_near_covalent_overlap_angstrom",
        "whole_structure_rdkit_sanitize_readable",
        "whole_structure_parser_extra_proximity_bond_count",
        "whole_structure_parser_extra_heavy_proximity_bond_count",
    }
    violations: list[str] = []
    for policy_path in policy_paths:
        tree = ast.parse(policy_path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in flat_projection_attributes
            ):
                violations.append(f"{policy_path}:{node.lineno}:{node.attr}")

    assert not violations


def _acceptance_metrics(
    *,
    focus_clash_count: int,
    focus_geometry_outlier_count: int,
    focus_restraint_backed_geometry_outlier_count: int = 0,
    focus_fallback_geometry_outlier_count: int = 0,
    focus_severe_restraint_backed_bond_length_outlier_count: int = 0,
    focus_clash_overlap_sum_angstrom: float = 0.0,
    focus_near_covalent_contact_count: int = 0,
    focus_worst_near_covalent_overlap_angstrom: float = 0.0,
    focus_total_near_covalent_overlap_angstrom: float = 0.0,
    focus_stereochemistry_violation_count: int = 0,
    whole_structure_near_covalent_contact_count: int = 0,
    whole_structure_worst_near_covalent_overlap_angstrom: float = 0.0,
    whole_structure_total_near_covalent_overlap_angstrom: float = 0.0,
    whole_structure_rdkit_sanitize_readable: bool | None = None,
    whole_structure_parser_extra_proximity_bond_count: int = 0,
    whole_structure_parser_extra_heavy_proximity_bond_count: int = 0,
) -> RefinementAcceptanceMetrics:
    """Return canonical metrics from compact test fixture values."""

    return RefinementAcceptanceMetrics(
        focus_quality=FocusRefinementQualityMetrics(
            clash_count=focus_clash_count,
            geometry_outlier_count=focus_geometry_outlier_count,
            restraint_backed_geometry_outlier_count=(
                focus_restraint_backed_geometry_outlier_count
            ),
            fallback_geometry_outlier_count=(
                focus_fallback_geometry_outlier_count
            ),
            severe_restraint_backed_bond_length_outlier_count=(
                focus_severe_restraint_backed_bond_length_outlier_count
            ),
            clash_overlap_sum_angstrom=focus_clash_overlap_sum_angstrom,
            near_covalent_contact_count=focus_near_covalent_contact_count,
            worst_near_covalent_overlap_angstrom=(
                focus_worst_near_covalent_overlap_angstrom
            ),
            total_near_covalent_overlap_angstrom=(
                focus_total_near_covalent_overlap_angstrom
            ),
            stereochemistry_violation_count=focus_stereochemistry_violation_count,
        ),
        whole_structure_proximity=WholeStructureProximityBurdenMetrics(
            near_covalent_contact_count=whole_structure_near_covalent_contact_count,
            worst_near_covalent_overlap_angstrom=(
                whole_structure_worst_near_covalent_overlap_angstrom
            ),
            total_near_covalent_overlap_angstrom=(
                whole_structure_total_near_covalent_overlap_angstrom
            ),
        ),
        parser_compatibility=WholeStructureParserCompatibilityMetrics(
            rdkit_sanitize_readable=whole_structure_rdkit_sanitize_readable,
            extra_proximity_bond_count=(
                whole_structure_parser_extra_proximity_bond_count
            ),
            extra_heavy_proximity_bond_count=(
                whole_structure_parser_extra_heavy_proximity_bond_count
            ),
        ),
    )


def test_refinement_metrics_regressed_accepts_weighted_clash_improvement() -> None:
    """Acceptance gating should prioritize clash relief over geometry drift."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=3,
        focus_geometry_outlier_count=3,
        focus_clash_overlap_sum_angstrom=2.84,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=5,
        focus_clash_overlap_sum_angstrom=0.0,
    )

    assert not refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_metrics_regressed_rejects_stereochemistry_regression(
) -> None:
    """Acceptance gating should reject new stereochemistry violations."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_stereochemistry_violation_count=0,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_stereochemistry_violation_count=1,
    )

    assert refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_metrics_regressed_keeps_clash_relief_ahead_of_stereochemistry(
) -> None:
    """Clash relief can still improve ordering before hard-reject checks."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=1,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.8,
        focus_stereochemistry_violation_count=0,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_stereochemistry_violation_count=1,
    )

    assert not refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_metrics_rejected_rejects_new_stereochemistry_even_with_clash_relief(
) -> None:
    """New stereochemistry burden should hard-reject otherwise improved output."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=1,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.8,
        focus_stereochemistry_violation_count=0,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_stereochemistry_violation_count=1,
    )

    assert refinement_has_new_stereochemistry_failure(before_metrics, after_metrics)
    assert refinement_metrics_rejected(before_metrics, after_metrics)


def test_refinement_metrics_regressed_rejects_same_clash_geometry_regression() -> None:
    """Acceptance gating should reject geometry regression at equal clash burden."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=1,
        focus_geometry_outlier_count=2,
        focus_clash_overlap_sum_angstrom=0.50,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=1,
        focus_geometry_outlier_count=8,
        focus_clash_overlap_sum_angstrom=0.50,
    )

    assert refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_metrics_regressed_accepts_same_count_lower_overlap_geometry_cost(
) -> None:
    """Lower steric overlap should outrank geometry cost at equal clash count."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=1,
        focus_geometry_outlier_count=2,
        focus_clash_overlap_sum_angstrom=0.50,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=1,
        focus_geometry_outlier_count=8,
        focus_clash_overlap_sum_angstrom=0.45,
    )

    assert not refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_metrics_regressed_rejects_near_covalent_contact_regression() -> (
    None
):
    """Acceptance gating should reject parser-visible near-covalent regressions."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_near_covalent_contact_count=1,
        focus_worst_near_covalent_overlap_angstrom=1.10,
        focus_total_near_covalent_overlap_angstrom=1.10,
    )

    assert refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_metrics_regressed_accepts_near_covalent_relief() -> None:
    """Acceptance gating should prefer removing near-covalent burden first."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_near_covalent_contact_count=1,
        focus_worst_near_covalent_overlap_angstrom=1.25,
        focus_total_near_covalent_overlap_angstrom=1.25,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=3,
        focus_clash_overlap_sum_angstrom=0.0,
    )

    assert not refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_metrics_regressed_rejects_full_structure_sanitize_loss() -> None:
    """Acceptance ordering should treat parser-visible sanitize loss as regression."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        whole_structure_rdkit_sanitize_readable=True,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        whole_structure_rdkit_sanitize_readable=False,
    )

    assert refinement_metrics_regressed(before_metrics, after_metrics)


def test_refinement_rejects_unresolved_sanitize_without_global_relief(
) -> (
    None
):
    """Unreadable candidates should still fail when global burden does not improve."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        whole_structure_near_covalent_contact_count=5,
        whole_structure_worst_near_covalent_overlap_angstrom=1.50,
        whole_structure_total_near_covalent_overlap_angstrom=6.20,
        whole_structure_rdkit_sanitize_readable=False,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        whole_structure_near_covalent_contact_count=5,
        whole_structure_worst_near_covalent_overlap_angstrom=1.50,
        whole_structure_total_near_covalent_overlap_angstrom=6.20,
        whole_structure_rdkit_sanitize_readable=False,
    )

    assert refinement_metrics_rejected(before_metrics, after_metrics)


def test_refinement_accepts_unresolved_sanitize_with_global_relief(
) -> (
    None
):
    """Unreadable candidates may pass when global near-covalent burden drops."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=8,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=10.35,
        focus_near_covalent_contact_count=5,
        focus_worst_near_covalent_overlap_angstrom=2.01,
        focus_total_near_covalent_overlap_angstrom=7.75,
        whole_structure_near_covalent_contact_count=84,
        whole_structure_worst_near_covalent_overlap_angstrom=2.56,
        whole_structure_total_near_covalent_overlap_angstrom=107.81,
        whole_structure_rdkit_sanitize_readable=False,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=6,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_near_covalent_contact_count=0,
        focus_worst_near_covalent_overlap_angstrom=0.0,
        focus_total_near_covalent_overlap_angstrom=0.0,
        whole_structure_near_covalent_contact_count=75,
        whole_structure_worst_near_covalent_overlap_angstrom=1.84,
        whole_structure_total_near_covalent_overlap_angstrom=93.51,
        whole_structure_rdkit_sanitize_readable=False,
    )

    assert not refinement_metrics_rejected(before_metrics, after_metrics)


def test_refinement_accepts_angle_geometry_burden_when_ff_signal_improves() -> None:
    """Angle diagnostics alone should not veto major FF-quality relief."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=73,
        focus_geometry_outlier_count=0,
        focus_restraint_backed_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=98.89,
        focus_near_covalent_contact_count=31,
        focus_worst_near_covalent_overlap_angstrom=2.46,
        focus_total_near_covalent_overlap_angstrom=54.80,
        whole_structure_near_covalent_contact_count=133,
        whole_structure_worst_near_covalent_overlap_angstrom=2.46,
        whole_structure_total_near_covalent_overlap_angstrom=209.35,
        whole_structure_rdkit_sanitize_readable=False,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=59,
        focus_geometry_outlier_count=47,
        focus_restraint_backed_geometry_outlier_count=47,
        focus_severe_restraint_backed_bond_length_outlier_count=0,
        focus_clash_overlap_sum_angstrom=62.10,
        focus_near_covalent_contact_count=2,
        focus_worst_near_covalent_overlap_angstrom=1.61,
        focus_total_near_covalent_overlap_angstrom=3.20,
        whole_structure_near_covalent_contact_count=104,
        whole_structure_worst_near_covalent_overlap_angstrom=2.33,
        whole_structure_total_near_covalent_overlap_angstrom=157.75,
        whole_structure_rdkit_sanitize_readable=False,
    )

    assert not refinement_metrics_rejected(before_metrics, after_metrics)


def test_refinement_rejects_new_severe_restraint_backed_bond_length_failure() -> None:
    """Severe backed bond-length failures are credible enough to veto FF relief."""

    before_metrics = _acceptance_metrics(
        focus_clash_count=4,
        focus_geometry_outlier_count=0,
        focus_restraint_backed_geometry_outlier_count=0,
        focus_severe_restraint_backed_bond_length_outlier_count=0,
        focus_clash_overlap_sum_angstrom=5.0,
        focus_near_covalent_contact_count=2,
        focus_worst_near_covalent_overlap_angstrom=1.2,
        focus_total_near_covalent_overlap_angstrom=2.1,
    )
    after_metrics = _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=1,
        focus_restraint_backed_geometry_outlier_count=1,
        focus_severe_restraint_backed_bond_length_outlier_count=1,
        focus_clash_overlap_sum_angstrom=0.0,
        focus_near_covalent_contact_count=0,
        focus_worst_near_covalent_overlap_angstrom=0.0,
        focus_total_near_covalent_overlap_angstrom=0.0,
    )

    assert refinement_has_new_severe_restraint_backed_bond_length_failure(
        before_metrics,
        after_metrics,
    )
    assert refinement_metrics_rejected(before_metrics, after_metrics)


def _refinement_metrics(
    *,
    rdkit_readable: bool | None,
    extra_proximity_bonds: int = 0,
    extra_heavy_proximity_bonds: int = 0,
) -> RefinementAcceptanceMetrics:
    """Return compact refinement metrics for acceptance-boundary tests."""

    return _acceptance_metrics(
        focus_clash_count=0,
        focus_geometry_outlier_count=0,
        focus_clash_overlap_sum_angstrom=0.0,
        whole_structure_rdkit_sanitize_readable=rdkit_readable,
        whole_structure_parser_extra_proximity_bond_count=extra_proximity_bonds,
        whole_structure_parser_extra_heavy_proximity_bond_count=(
            extra_heavy_proximity_bonds
        ),
    )


def _accepted_h_only_parser_failure(
    selected_scope: Scope,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    result: RegionTransformationResult,
    *,
    before_metrics: RefinementAcceptanceMetrics,
    clash_basis: ClashDetectionBasis | None = None,
) -> AssessedRefinementResult:
    """Return an accepted assessment carrying an H-only parser failure."""

    del selected_scope, component_library, restraint_library, clash_basis
    return AssessedRefinementResult(
        executed_result=result,
        before_metrics=before_metrics,
        after_metrics=_refinement_metrics(
            rdkit_readable=False,
            extra_proximity_bonds=1,
        ),
        verdict=RefinementAcceptanceVerdict.ACCEPTED,
    )


def _accepted_mixed_parser_failure(
    selected_scope: Scope,
    component_library: ComponentLibrary,
    restraint_library: RestraintLibrary,
    result: RegionTransformationResult,
    *,
    before_metrics: RefinementAcceptanceMetrics,
    clash_basis: ClashDetectionBasis | None = None,
) -> AssessedRefinementResult:
    """Return an accepted assessment carrying mixed heavy and H parser failures."""

    del selected_scope, component_library, restraint_library, clash_basis
    return AssessedRefinementResult(
        executed_result=result,
        before_metrics=before_metrics,
        after_metrics=_refinement_metrics(
            rdkit_readable=False,
            extra_proximity_bonds=2,
            extra_heavy_proximity_bonds=1,
        ),
        verdict=RefinementAcceptanceVerdict.ACCEPTED,
    )


def _structure_with_updated_atom_positions(
    structure: ProteinStructure,
    residue_id: ResidueId,
    positions_by_atom_name: dict[str, Vec3],
) -> ProteinStructure:
    """Return a structure with selected residue-local atom positions replaced."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    for atom_name, position in positions_by_atom_name.items():
        residue_geometry = residue_geometry.with_atom_geometry(
            atom_name,
            residue_geometry.atom_geometry(atom_name).with_position(position),
        )

    return structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=(
            structure.residue_formal_charge_by_atom_name(residue_index)
        ),
    )


def _atom_position(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name: str,
) -> Vec3:
    """Return one residue-local atom position from a structure."""

    residue_index = structure.constitution.residue_index(residue_id)
    return structure.residue_geometry(residue_index).atom_geometry(atom_name).position


def build_toy_component_library() -> ComponentLibrary:
    """Return one tiny component library suitable for refinement API tests."""

    return ComponentLibrary(
        templates={
            "MOV": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="MOV",
                    atom_names=("C1", "H1"),
                    bonds=(BondDefinition("C1", "H1"),),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("C1",),
                ),
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=((("H1",), "class3", ("C1", "C1", "C1")),),
                ),
            )
        }
    )


def build_clashy_component_library() -> ComponentLibrary:
    """Return one tiny component library for a polymer clash fixture."""

    return ComponentLibrary(
        templates={
            "MOV": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="MOV",
                    atom_names=("C1", "H1"),
                    bonds=(BondDefinition("C1", "H1"),),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("C1",),
                ),
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=((("H1",), "class3", ("C1", "C1", "C1")),),
                ),
            ),
        }
    )


def build_angle_component_library() -> ComponentLibrary:
    """Return one three-heavy-atom residue library for geometry-gate tests."""

    return ComponentLibrary(
        templates={
            "ANG": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="ANG",
                    atom_names=("A1", "A2", "A3", "H1"),
                    bonds=(
                        BondDefinition("A1", "A2"),
                        BondDefinition("A2", "A3"),
                        BondDefinition("A2", "H1"),
                    ),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("A1", "A2", "A3"),
                ),
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=((("H1",), "class3", ("A1", "A3", "A2")),),
                ),
            )
        }
    )


def build_toy_structure() -> ProteinStructure:
    """Return one tiny one-residue structure for API boundary tests."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_clashy_structure() -> ProteinStructure:
    """Return one tiny local environment with one real focus-residue clash."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="B", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.1, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(-1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_angle_structure() -> ProteinStructure:
    """Return one hydrogenated residue with initially plausible geometry."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ANG",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("A1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("A2", "N", Vec3(1.40, 0.0, 0.0)),
                            atom_payload("A3", "O", Vec3(2.10, 1.20, 0.0)),
                            atom_payload("H1", "H", Vec3(1.35, -0.95, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_hydrogenless_angle_structure() -> ProteinStructure:
    """Return one geometry fixture that is illegal for force-field binding."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ANG",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("A1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("A2", "N", Vec3(1.40, 0.0, 0.0)),
                            atom_payload("A3", "O", Vec3(2.10, 1.20, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

def test_rejected_candidate_discards_pre_backend_moves() -> None:
    """Rejected candidates should discard pre-backend discrete movement too."""

    structure = build_clashy_structure()
    moved_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "H1")
    )
    executed_result = RegionTransformationResult(
        refined_structure=structure,
        delta=StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            moved_atoms=(
                MovedAtomDelta(
                    before_atom_index=moved_atom_index,
                    after_atom_index=moved_atom_index,
                ),
            ),
        ),
        issues=(),
        backend_name="fake",
    )
    rejected_assessment = AssessedRefinementResult(
        executed_result=executed_result,
        before_metrics=_acceptance_metrics(
            focus_clash_count=0,
            focus_geometry_outlier_count=0,
            focus_clash_overlap_sum_angstrom=0.0,
        ),
        after_metrics=_acceptance_metrics(
            focus_clash_count=0,
            focus_geometry_outlier_count=0,
            focus_clash_overlap_sum_angstrom=0.0,
            whole_structure_rdkit_sanitize_readable=False,
        ),
        verdict=RefinementAcceptanceVerdict.REJECTED,
        rejection_issue=ValidationIssue(
            kind=ValidationIssueKind.REFINEMENT_REJECTED,
            severity=IssueSeverity.INFO,
            message="parser-visible failure persists",
        ),
    )

    materialized_result = materialize_assessed_refinement_candidate(
        rejected_assessment,
        original_structure=structure,
        pre_backend_moved_atom_indices=(moved_atom_index,),
    )

    assert materialized_result.refined_structure == structure
    assert materialized_result.delta.moved_atoms == ()
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
        for issue in materialized_result.issues
    )
