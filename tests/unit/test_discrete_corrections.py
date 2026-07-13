"""Unit tests for internal discrete pre-refinement corrections."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.refinement_contract import build_refinement_inputs

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.clashes import ClashDetectionBasis
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.io.pdb_projection import RDKitNoConectPDBBlockProjector
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
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.discrete import (
    BranchedSidechainSeedProvenance,
    BranchedSidechainSeedTransformer,
    DiscretePreRefinementCorrectionTransformer,
    apply_discrete_pre_refinement_corrections,
)
from protrepair.transformer.discrete.axis_rotation import (
    score_discrete_correction_candidate,
    select_best_discrete_correction_candidate,
    snapshot_has_applicable_axis_rotation_correction,
)
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    transform_local_region,
)
from protrepair.transformer.refinement.acceptance import (
    AssessedRefinementResult,
    FocusRefinementQualityMetrics,
    RefinementAcceptanceMetrics,
    RefinementAcceptanceVerdict,
)


def test_apply_discrete_pre_refinement_corrections_prefers_fixture_asn_flip() -> None:
    """The Asn182 literature fixture should prefer one flipped amide candidate."""

    case = REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]
    component_library = build_default_component_library()
    structure = load_case_structure(case)
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_atoms(case.focus_atom_refs),
        context_radius_angstrom=3.0,
    )

    candidate = select_best_discrete_correction_candidate(
        snapshot=snapshot,
        atom_input=atom_input,
        structure=structure,
        residue_id=ResidueId("A", 182),
        component_library=component_library,
    )

    assert candidate is not None
    original_candidate = score_discrete_correction_candidate(
        structure=structure,
        residue_id=ResidueId("A", 182),
        component_library=component_library,
        moved_atom_indices=(),
    )
    assert (
        candidate.score.focus_clash_count == original_candidate.score.focus_clash_count
    )
    assert (
        candidate.score.focus_heavy_fractional_clash_overlap_sum
        < original_candidate.score.focus_heavy_fractional_clash_overlap_sum
    )

    corrected_snapshot = apply_discrete_pre_refinement_corrections(
        snapshot,
        atom_input,
        component_library=component_library,
    )
    moved_atom_indices = snapshot.moved_atom_indices_to(
        corrected_snapshot,
        atom_input.atom_indices,
    )
    moved_atom_names = {
        atom_ref.atom_name
        for atom_index in moved_atom_indices
        for atom_ref in (snapshot.structure.constitution.atom_ref_at(atom_index),)
        if atom_ref.residue_id == ResidueId("A", 182)
    }
    corrected_od1 = corrected_snapshot.structure.geometry.atom_geometry(
        corrected_snapshot.structure.constitution.atom_index(
            AtomRef(ResidueId("A", 182), "OD1")
        )
    )
    original_od1 = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 182), "OD1"))
    )

    assert {"OD1", "ND2"} <= moved_atom_names
    assert corrected_od1.position != original_od1.position


def test_discrete_pre_refinement_detects_applicable_asn_case() -> None:
    """Deterministic request transformer should report applicable Asn flip cases."""

    case = REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]
    component_library = build_default_component_library()
    snapshot, atom_input, _ = build_refinement_inputs(
        load_case_structure(case),
        LocalScopeSpec.from_atoms(case.focus_atom_refs),
        context_radius_angstrom=3.0,
    )
    transformer = DiscretePreRefinementCorrectionTransformer(component_library)

    assert snapshot_has_applicable_axis_rotation_correction(
        snapshot,
        atom_input,
        component_library,
    )
    assert transformer.is_applicable(
        ProteinTransformationContext.from_snapshot_atom_input(snapshot, atom_input)
    )


def test_residuewise_hydrogenated_flip_moves_attached_amide_hydrogens() -> None:
    """Residuewise hydrogenated Asn flips should carry attached amide hydrogens."""

    case = REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_residues(case.focus_residue_ids),
        context_radius_angstrom=3.0,
    )

    corrected_snapshot = apply_discrete_pre_refinement_corrections(
        snapshot,
        atom_input,
        component_library=component_library,
    )
    moved_atom_indices = snapshot.moved_atom_indices_to(
        corrected_snapshot,
        atom_input.atom_indices,
    )

    moved_atom_names = {
        atom_ref.atom_name
        for atom_index in moved_atom_indices
        for atom_ref in (snapshot.structure.constitution.atom_ref_at(atom_index),)
        if atom_ref.residue_id == ResidueId("A", 182)
    }
    assert {"1HD2", "2HD2"} <= moved_atom_names


def test_apply_discrete_pre_refinement_corrections_prefers_fixture_his_flip() -> None:
    """Hydrogenated His42 fixture should prefer one ring-flipped candidate."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-his42"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_residues(case.focus_residue_ids),
        context_radius_angstrom=4.0,
    )

    candidate = select_best_discrete_correction_candidate(
        snapshot=snapshot,
        atom_input=atom_input,
        structure=structure,
        residue_id=ResidueId("A", 42),
        component_library=component_library,
    )

    assert candidate is not None
    original_candidate = score_discrete_correction_candidate(
        structure=structure,
        residue_id=ResidueId("A", 42),
        component_library=component_library,
        moved_atom_indices=(),
    )
    assert (
        candidate.score.focus_clash_count < original_candidate.score.focus_clash_count
    )
    assert (
        candidate.score.focus_geometry_outlier_count
        == original_candidate.score.focus_geometry_outlier_count
    )

    corrected_snapshot = apply_discrete_pre_refinement_corrections(
        snapshot,
        atom_input,
        component_library=component_library,
    )
    moved_atom_indices = snapshot.moved_atom_indices_to(
        corrected_snapshot,
        atom_input.atom_indices,
    )
    moved_atom_names = {
        atom_ref.atom_name
        for atom_index in moved_atom_indices
        for atom_ref in (snapshot.structure.constitution.atom_ref_at(atom_index),)
        if atom_ref.residue_id == ResidueId("A", 42)
    }
    assert {"ND1", "CD2", "CE1", "NE2"} <= moved_atom_names
    assert "HE2" in moved_atom_names


def test_discrete_pre_refinement_respects_atomwise_terminal_selection() -> None:
    """Atomwise selections should require both terminal heavy atoms for flips."""

    case = REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]
    component_library = build_default_component_library()
    structure = load_case_structure(case)
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_atoms(
            (
                AtomRef(ResidueId("A", 182), "CB"),
                AtomRef(ResidueId("A", 182), "CG"),
            )
        ),
        context_radius_angstrom=3.0,
    )

    corrected_snapshot = apply_discrete_pre_refinement_corrections(
        snapshot,
        atom_input,
        component_library=component_library,
    )
    moved_atom_indices = snapshot.moved_atom_indices_to(
        corrected_snapshot,
        atom_input.atom_indices,
    )

    assert corrected_snapshot.structure == structure
    assert moved_atom_indices == ()


def test_hydrogenated_atomwise_his_flip_requires_ring_hydrogens() -> None:
    """Hydrogenated atomwise His flips should respect exact movable-atom selection."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-his42"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_atoms(case.focus_atom_refs),
        context_radius_angstrom=3.0,
    )

    corrected_snapshot = apply_discrete_pre_refinement_corrections(
        snapshot,
        atom_input,
        component_library=component_library,
    )
    moved_atom_indices = snapshot.moved_atom_indices_to(
        corrected_snapshot,
        atom_input.atom_indices,
    )

    assert corrected_snapshot.structure == structure
    assert moved_atom_indices == ()


def test_hydrogenated_atomwise_flip_requires_attached_hydrogens_in_selection() -> None:
    """Hydrogenated atomwise flips should respect exact movable-atom selection."""

    case = REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_atoms(case.focus_atom_refs),
        context_radius_angstrom=3.0,
    )

    corrected_snapshot = apply_discrete_pre_refinement_corrections(
        snapshot,
        atom_input,
        component_library=component_library,
    )
    moved_atom_indices = snapshot.moved_atom_indices_to(
        corrected_snapshot,
        atom_input.atom_indices,
    )

    assert corrected_snapshot.structure == structure
    assert moved_atom_indices == ()


def test_hydrogenated_atomwise_flip_supports_attached_hydrogen_closure_selection() -> (
    None
):
    """Attached-hydrogen closure should make hydrogenated flips explicit."""

    case = REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_atoms_with_attached_hydrogens(case.focus_atom_refs),
        component_library=component_library,
        context_radius_angstrom=3.0,
    )

    corrected_snapshot = apply_discrete_pre_refinement_corrections(
        snapshot,
        atom_input,
        component_library=component_library,
    )
    moved_atom_indices = snapshot.moved_atom_indices_to(
        corrected_snapshot,
        atom_input.atom_indices,
    )
    moved_atom_names = {
        atom_ref.atom_name
        for atom_index in moved_atom_indices
        for atom_ref in (snapshot.structure.constitution.atom_ref_at(atom_index),)
        if atom_ref.residue_id == ResidueId("A", 182)
    }

    assert {"OD1", "ND2", "1HD2", "2HD2"} <= moved_atom_names


def test_generate_pre_refinement_seed_candidates_supports_leu_residuewise_search() -> (
    None
):
    """Residuewise Leu selections should emit the full branched-sidechain seed grid."""

    case = REFINEMENT_BENCHMARK_CASES["1xgo-leu253"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_residues(case.focus_residue_ids),
        context_radius_angstrom=4.0,
    )
    seed_transformer = BranchedSidechainSeedTransformer(component_library)

    seed_candidates = seed_transformer.transform(
        ProteinTransformationContext.from_snapshot_atom_input(snapshot, atom_input)
    )

    assert len(seed_candidates) == 15
    assert all(
        isinstance(seed_candidate.provenance, BranchedSidechainSeedProvenance)
        for seed_candidate in seed_candidates
    )
    assert all(
        seed_candidate.provenance.residue_id == ResidueId("A", 253)
        for seed_candidate in seed_candidates
    )
    assert all(
        seed_candidate.provenance.component_id == "LEU"
        and len(seed_candidate.provenance.angle_degrees_by_step) == 2
        and any(
            angle_degrees != 0
            for angle_degrees in seed_candidate.provenance.angle_degrees_by_step
        )
        for seed_candidate in seed_candidates
    )
    assert any(
        {
            snapshot.structure.constitution.atom_ref_at(atom_index).atom_name
            for atom_index in snapshot.moved_atom_indices_to(
                seed_candidate.payload,
                atom_input.atom_indices,
            )
            for atom_ref in (snapshot.structure.constitution.atom_ref_at(atom_index),)
            if atom_ref.residue_id == ResidueId("A", 253)
        }
        >= {"CG", "CD1", "CD2"}
        for seed_candidate in seed_candidates
    )
    assert any(
        "1HD1"
        in {
            snapshot.structure.constitution.atom_ref_at(atom_index).atom_name
            for atom_index in snapshot.moved_atom_indices_to(
                seed_candidate.payload,
                atom_input.atom_indices,
            )
            for atom_ref in (snapshot.structure.constitution.atom_ref_at(atom_index),)
            if atom_ref.residue_id == ResidueId("A", 253)
        }
        for seed_candidate in seed_candidates
    )


def test_hydrogenated_atomwise_leu_seed_requires_attached_hydrogens_in_selection() -> (
    None
):
    """Hydrogenated atomwise Leu seeding should not move unselected side-chain Hs."""

    case = REFINEMENT_BENCHMARK_CASES["1xgo-leu253"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_atoms(case.focus_atom_refs),
        context_radius_angstrom=4.0,
    )
    seed_transformer = BranchedSidechainSeedTransformer(component_library)

    context = ProteinTransformationContext.from_snapshot_atom_input(
        snapshot,
        atom_input,
    )
    assert seed_transformer.transform(context) == ()
    assert not seed_transformer.is_applicable(context)


def test_apply_discrete_pre_refinement_corrections_supports_gln_amide_flip() -> None:
    """Synthetic glutamine geometry should also admit the same amide-flip logic."""

    component_library = build_default_component_library()
    structure = build_gln_flip_structure()
    snapshot, atom_input, _ = build_refinement_inputs(
        structure,
        LocalScopeSpec.from_atoms(
            (
                AtomRef(ResidueId("A", 1), "OE1"),
                AtomRef(ResidueId("A", 1), "NE2"),
            )
        ),
        context_radius_angstrom=3.0,
    )

    candidate = select_best_discrete_correction_candidate(
        snapshot=snapshot,
        atom_input=atom_input,
        structure=structure,
        residue_id=ResidueId("A", 1),
        component_library=component_library,
    )

    assert candidate is not None
    assert {"OE1", "NE2"} <= {
        structure.constitution.atom_ref_at(atom_index).atom_name
        for atom_index in candidate.moved_atom_indices
    }


def test_refine_local_region_preserves_discrete_correction_moves_when_backend_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public refinement should preserve discrete moves even when the backend no-ops."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-his42"]
    component_library = build_default_component_library()
    original_structure = load_case_structure(case)
    structure = add_hydrogens(
        original_structure,
        component_library=component_library,
    ).structure
    target_residue_id = ResidueId("A", 42)
    expected_input_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(target_residue_id, "ND1"))
    )

    expected_input_position = expected_input_geometry.position
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residues(case.focus_residue_ids),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=3.0),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library,
        ) -> RegionTransformationResult:
            del restraint_library
            return RegionTransformationResult(
                refined_structure=problem.region.snapshot.structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=problem.region.snapshot.structure.constitution,
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
        component_library=component_library,
    )

    moved_atom_names = {
        result.refined_structure.constitution.atom_site_at(
            moved_atom.after_atom_index
        ).name
        for moved_atom in result.delta.moved_atoms
        if result.refined_structure.constitution.residue_site_at(
            result.refined_structure.constitution.residue_index_for_atom_index(
                moved_atom.after_atom_index
            )
        ).residue_id
        == target_residue_id
    }
    refined_nd1 = result.refined_structure.geometry.atom_geometry(
        result.refined_structure.constitution.atom_index(
            AtomRef(target_residue_id, "ND1")
        )
    )

    assert {"ND1", "NE2"} <= moved_atom_names
    assert refined_nd1.position != expected_input_position


def test_refine_local_region_keeps_seed_candidates_when_baseline_candidate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded candidates should still be evaluated when the baseline candidate fails."""

    case = REFINEMENT_BENCHMARK_CASES["1xgo-leu253"]
    component_library = build_default_component_library()
    structure = add_hydrogens(
        load_case_structure(case),
        component_library=component_library,
        local_refinement=None,
    ).structure
    target_residue_id = ResidueId("A", 253)
    baseline_cd1_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(target_residue_id, "CD1"))
    )

    baseline_cd1_position = baseline_cd1_geometry.position
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residues(case.focus_residue_ids),
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(context_radius_angstrom=4.0),
    )

    class FakeBackend:
        def relax(
            self,
            problem,
            *,
            restraint_library,
        ) -> RegionTransformationResult:
            del restraint_library
            candidate_cd1_geometry = (
                problem.region.snapshot.structure.geometry.atom_geometry(
                    problem.region.snapshot.structure.constitution.atom_index(
                        AtomRef(target_residue_id, "CD1")
                    )
                )
            )

            assert candidate_cd1_geometry is not None

            candidate_cd1_position = candidate_cd1_geometry.position
            if candidate_cd1_position == baseline_cd1_position:
                raise RefinementError("baseline candidate rejected")

            return RegionTransformationResult(
                refined_structure=problem.region.snapshot.structure,
                delta=StructureDelta(
                    before_constitution=problem.region.snapshot.structure.constitution,
                    after_constitution=problem.region.snapshot.structure.constitution,
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

    def assess_without_rejection(
        selected_scope,
        component_library,
        restraint_library,
        result: RegionTransformationResult,
        *,
        before_metrics: RefinementAcceptanceMetrics,
        clash_basis: ClashDetectionBasis | None = None,
        pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
    ) -> AssessedRefinementResult:
        del (
            selected_scope,
            component_library,
            restraint_library,
            before_metrics,
            clash_basis,
            pdb_block_projector,
        )
        zero_metrics = RefinementAcceptanceMetrics(
            focus_quality=FocusRefinementQualityMetrics(
                clash_count=0,
                geometry_outlier_count=0,
                clash_overlap_sum_angstrom=0.0,
            ),
        )
        return AssessedRefinementResult(
            executed_result=result,
            before_metrics=zero_metrics,
            after_metrics=zero_metrics,
            verdict=RefinementAcceptanceVerdict.ACCEPTED,
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.local_pipeline.assessment.assess_refinement_result_with_before_metrics",
        assess_without_rejection,
    )

    result = transform_local_region(
        structure,
        spec,
        component_library=component_library,
    )
    refined_cd1 = result.refined_structure.geometry.atom_geometry(
        result.refined_structure.constitution.atom_index(
            AtomRef(target_residue_id, "CD1")
        )
    )

    assert refined_cd1 is not None

    assert refined_cd1.position != baseline_cd1_position
    assert any(
        result.refined_structure.constitution.residue_site_at(
            result.refined_structure.constitution.residue_index_for_atom_index(
                moved_atom.after_atom_index
            )
        ).residue_id
        == target_residue_id
        and result.refined_structure.constitution.atom_site_at(
            moved_atom.after_atom_index
        ).name
        in {"CG", "CD1", "CD2"}
        for moved_atom in result.delta.moved_atoms
    )


def build_gln_flip_structure() -> ProteinStructure:
    """Return one synthetic glutamine clash case for discrete amide correction."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLN",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(-1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(2.0, 0.2, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, -1.2, 0.0)),
                            atom_payload("CG", "C", Vec3(2.9, -1.2, 0.0)),
                            atom_payload("CD", "C", Vec3(4.4, -1.2, 0.0)),
                            atom_payload("OE1", "O", Vec3(5.4, -0.2, 0.0)),
                            atom_payload("NE2", "N", Vec3(5.4, -2.2, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(20.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(21.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(22.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(23.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(21.0, -1.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 3),
                        atoms=(
                            atom_payload("N", "N", Vec3(5.4, -2.2, 2.4)),
                            atom_payload("CA", "C", Vec3(7.0, 0.0, 2.4)),
                            atom_payload("C", "C", Vec3(8.2, 0.0, 2.4)),
                            atom_payload("O", "O", Vec3(9.0, 0.2, 2.4)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
