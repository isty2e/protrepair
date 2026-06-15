"""Tests for refinement benchmark strategy and preparation contracts."""

import pytest
from tests.support.refinement_benchmarks import (
    REFINEMENT_BENCHMARK_PROFILES,
    REFINEMENT_BENCHMARK_TRACKS,
    REFINEMENT_STRATEGIES,
    prepare_case_structure,
)
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES

from protrepair.chemistry import build_default_component_library
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.binding_policy import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import (
    LocalScopeLowering,
    LocalScopeSpec,
    atom_input_from_local_scope_spec,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective

pytestmark = pytest.mark.benchmark


def test_refinement_strategy_builds_atomwise_selection_for_atom_tight_case() -> None:
    """The atom-tight strategy should emit one exact atom-local scope spec."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    spec = REFINEMENT_STRATEGIES["atom-tight"].build_spec(
        case,
        execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
    )

    assert isinstance(spec.scope_spec, LocalScopeSpec)
    assert (
        spec.scope_spec.scope == LocalScopeSpec.from_atoms(case.focus_atom_refs).scope
    )
    assert spec.scope_spec.lowering is LocalScopeLowering.EXACT_ATOMS


def test_refinement_strategy_builds_residuewise_selection_for_residue_local_case() -> (
    None
):
    """Residue-local strategy presets should emit one residue-atom scope spec."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    spec = REFINEMENT_STRATEGIES["residue-local"].build_spec(
        case,
        execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_mmff"],
    )

    assert isinstance(spec.scope_spec, LocalScopeSpec)
    assert (
        spec.scope_spec.scope
        == LocalScopeSpec.from_residues(case.focus_residue_ids).scope
    )
    assert spec.scope_spec.lowering is LocalScopeLowering.RESIDUE_ATOMS
    assert spec.force_field is ContinuousRelaxationForceField.MMFF
    assert spec.config.backend_name == "rdkit"


def test_sidechain_local_strategy_builds_sidechain_lowering() -> None:
    """Sidechain-local strategy should lower residue focus to sidechain atoms only."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    spec = REFINEMENT_STRATEGIES["sidechain-local"].build_spec(
        case,
        execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
    )

    assert isinstance(spec.scope_spec, LocalScopeSpec)
    assert (
        spec.scope_spec.scope
        == LocalScopeSpec.from_residues(case.focus_residue_ids).scope
    )
    assert spec.scope_spec.lowering is LocalScopeLowering.RESIDUE_SIDECHAIN_ATOMS


def test_hydrogenated_track_adds_hydrogens_to_positive_fixture() -> None:
    """Hydrogenated benchmark track should materialize explicit hydrogens."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-his42"]
    heavy_structure, _ = prepare_case_structure(
        case,
        track=REFINEMENT_BENCHMARK_TRACKS["heavy-only"],
        component_library=None,
    )
    hydrogenated_structure, _ = prepare_case_structure(
        case,
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        component_library=None,
    )

    heavy_hydrogen_count = sum(
        1
        for atom_site in heavy_structure.iter_atom_sites(include_ligands=True)
        if atom_site.element == "H"
    )
    hydrogenated_hydrogen_count = sum(
        1
        for atom_site in hydrogenated_structure.iter_atom_sites(include_ligands=True)
        if atom_site.element == "H"
    )

    assert heavy_hydrogen_count == 0
    assert hydrogenated_hydrogen_count > 0


def test_residue_local_hydrogenated_benchmark_prefers_mmff() -> None:
    """Hydrogenated residue-local benchmark policy should prefer MMFF."""

    component_library = build_default_component_library()
    case = REFINEMENT_BENCHMARK_CASES["1bkr-his42"]
    strategy = REFINEMENT_STRATEGIES["residue-local"]
    structure, _ = prepare_case_structure(
        case,
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        component_library=component_library,
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    spec = strategy.build_spec(
        case,
        execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"],
    )
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        spec.scope_spec,
        component_library=component_library,
    )
    directive = RepairLocalRefinementDirective.from_atom_input(
        atom_input,
        binding=RecommendedContinuousRelaxationBinding(),
        config=spec.config,
    )

    bound_execution = directive.bind_execution(
        snapshot,
        component_library=component_library,
    )

    assert (
        bound_execution.binding_decision.settings.force_field
        is ContinuousRelaxationForceField.MMFF
    )


def test_recommended_binding_prefers_mmff_for_hydrogenated_atom_tight_benchmark() -> (
    None
):
    """Hydrogenated atom-tight benchmark policy should also start from MMFF."""

    component_library = build_default_component_library()
    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    strategy = REFINEMENT_STRATEGIES["atom-tight"]
    structure, _ = prepare_case_structure(
        case,
        track=REFINEMENT_BENCHMARK_TRACKS["hydrogenated"],
        component_library=component_library,
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    spec = strategy.build_spec(
        case,
        execution_profile=REFINEMENT_BENCHMARK_PROFILES["rdkit_mmff"],
    )
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        spec.scope_spec,
        component_library=component_library,
    )
    directive = RepairLocalRefinementDirective.from_atom_input(
        atom_input,
        binding=RecommendedContinuousRelaxationBinding(),
        config=spec.config,
    )

    bound_execution = directive.bind_execution(
        snapshot,
        component_library=component_library,
    )

    assert (
        bound_execution.binding_decision.settings.force_field
        is ContinuousRelaxationForceField.MMFF
    )
