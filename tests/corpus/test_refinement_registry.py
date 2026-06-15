"""Tests for refinement benchmark registries and fixture inventories."""

from pathlib import Path

import pytest
from tests.support.refinement_benchmarks import REFINEMENT_BENCHMARK_PROFILES
from tests.support.refinement_cases import (
    ALL_REFINEMENT_BENCHMARK_CASES,
    ALL_REFINEMENT_FIXTURE_SOURCES,
    EXPLORATORY_REFINEMENT_BENCHMARK_CASES,
    EXPLORATORY_REFINEMENT_FIXTURE_SOURCES,
    RefinementBenchmarkExpectedOutcome,
)
from tests.support.refinement_corpus import (
    ALL_CORPUS_METADATA,
    CorpusCoverageTag,
    CorpusEvaluationMode,
    CorpusExecutionPolicy,
    CorpusPanel,
    CorpusTrackingPressure,
    corpus_case_ids_for_evaluation_mode,
    corpus_case_ids_for_execution_policy,
    corpus_case_ids_for_panel,
    corpus_case_ids_for_required_operator_family,
    corpus_case_ids_for_tag,
    corpus_case_ids_for_tracking_pressure,
    uncovered_tags_for_panel,
)
from tests.support.refinement_perturbation_benchmarks import (
    PERTURBATION_BENCHMARK_SCENARIOS,
    PERTURBATION_REFINEMENT_BENCHMARK_CASES,
    REFERENCE_REFINEMENT_BENCHMARK_CASES,
    REFINEMENT_PERTURBATION_SPECS,
    SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES,
    RefinementPerturbationExpectedOutcome,
    RefinementPerturbationMode,
)
from tests.support.request_builders import ingress_options
from tests.support.whole_structure_sources import WHOLE_STRUCTURE_CORPUS_SOURCES

from protrepair.io import read_structure
from protrepair.transformer.continuous.settings import ContinuousRelaxationProfile
from protrepair.transformer.refinement.spec import RefinementOperatorFamily
from protrepair.workflow.contracts import LigandPolicy

pytestmark = pytest.mark.corpus


def test_refinement_fixture_sources_use_unique_output_paths() -> None:
    """Extracted refinement fixture sources should not collide on disk."""

    output_paths = tuple(
        source.output_path for source in ALL_REFINEMENT_FIXTURE_SOURCES.values()
    )
    assert len(output_paths) == len(set(output_paths))


@pytest.mark.parametrize(
    ("fixture_path", "expected_residue"),
    (
        (
            Path("tests/fixtures/pdb/refinement/4pep_sep68_local.pdb"),
            ("SEP", "A", 68),
        ),
        (
            Path("tests/fixtures/pdb/refinement/1uhg_sep68_local.pdb"),
            ("SEP", "A", 68),
        ),
        (
            Path("tests/fixtures/pdb/refinement/8pyr_sep164_local.pdb"),
            ("SEP", "A", 164),
        ),
        (
            Path("tests/fixtures/pdb/refinement/8pyr_tpo170_local.pdb"),
            ("TPO", "A", 170),
        ),
        (
            Path("tests/fixtures/pdb/refinement/4jfx_ptr8_local.pdb"),
            ("PTR", "P", 8),
        ),
    ),
)
def test_nonstandard_exploratory_fixture_contains_expected_modified_residue(
    fixture_path: Path,
    expected_residue: tuple[str, str, int],
) -> None:
    """Checked-in nonstandard exploratory fixtures should contain the seed residue."""

    structure = read_structure(
        fixture_path,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )
    modified_residue_tokens = {
        (residue.component_id, residue.residue_id.chain_id, residue.residue_id.seq_num)
        for residue in structure.constitution.iter_residues(include_ligands=True)
        if residue.component_id in {"SEP", "TPO", "PTR", "CSO"}
    }

    assert expected_residue in modified_residue_tokens


def test_refinement_benchmark_cases_reference_existing_fixture_paths() -> None:
    """Benchmark case registries should point to checked-in fixture paths."""

    for case in ALL_REFINEMENT_BENCHMARK_CASES.values():
        assert case.fixture_path.exists()
        if case.reference_fixture_path is not None:
            assert case.reference_fixture_path.exists()


def test_low_resolution_sidechain_cases_are_operator_support_gaps() -> None:
    """Density-misfit sidechain fixtures should not look like ordinary no-ops."""

    assert set(EXPLORATORY_REFINEMENT_BENCHMARK_CASES) == {
        "3j5p-gln519-sidechain",
        "5k12-tyr471-sidechain",
    }
    for case in EXPLORATORY_REFINEMENT_BENCHMARK_CASES.values():
        assert (
            case.expected_outcome
            is RefinementBenchmarkExpectedOutcome.EXPECTED_OPERATOR_SUPPORT_GAP
        )
        assert case.expected_outcome.expects_no_op()
        assert case.expected_outcome.matches_status("no_op")


@pytest.mark.parametrize(
    ("fixture_path", "expected_residues"),
    (
        (
            Path("tests/fixtures/pdb/refinement/3j6b_terminal_helix_misthread_local.pdb"),
            {("SER", "9", 228), ("VAL", "9", 237)},
        ),
        (
            Path("tests/fixtures/pdb/refinement/3j9e_loop_backbone_error_local.pdb"),
            {("PRO", "D", 370), ("ASP", "D", 375)},
        ),
        (
            Path("tests/fixtures/pdb/refinement/2h6o_glycan_problematic_local.pdb"),
            {("ASN", "A", 195), ("NAG", "M", 1)},
        ),
        (
            Path("tests/fixtures/pdb/refinement/3j5p_gln519_sidechain_local.pdb"),
            {("GLN", "C", 519)},
        ),
        (
            Path("tests/fixtures/pdb/refinement/5xm5_metal_sites_local.pdb"),
            {("ZN", "A", 201), ("ZN", "B", 202)},
        ),
        (
            Path("tests/fixtures/pdb/refinement/5k12_tyr471_sidechain_local.pdb"),
            {("TYR", "F", 471)},
        ),
        (
            Path("tests/fixtures/pdb/refinement/7s9d_prestin_segment_local.pdb"),
            {("LYS", "A", 615), ("GLY", "A", 636)},
        ),
    ),
)
def test_exploratory_refinement_fixture_contains_expected_seed_neighborhood(
    fixture_path: Path,
    expected_residues: set[tuple[str, str, int]],
) -> None:
    """Checked-in exploratory fixtures should retain their anchor residue content."""

    structure = read_structure(
        fixture_path,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )
    observed_residues = {
        (residue.component_id, residue.residue_id.chain_id, residue.residue_id.seq_num)
        for residue in structure.constitution.iter_residues(include_ligands=True)
    }
    assert expected_residues.issubset(observed_residues)


def test_exploratory_refinement_fixture_sources_reference_checked_in_paths() -> None:
    """Exploratory extraction recipes should point to checked-in fixture paths."""

    for source in EXPLORATORY_REFINEMENT_FIXTURE_SOURCES.values():
        assert source.output_path.exists()


def test_whole_structure_corpus_sources_reference_checked_in_paths() -> None:
    """Whole-structure corpus sources should point to checked-in paths."""

    for source in WHOLE_STRUCTURE_CORPUS_SOURCES.values():
        assert source.output_path.exists()


def test_whole_structure_corpus_sources_have_metadata_owners() -> None:
    """Downloaded whole-structure sources should stay queryable by corpus metadata."""

    assert set(WHOLE_STRUCTURE_CORPUS_SOURCES).issubset(ALL_CORPUS_METADATA)


def test_multichar_chain_whole_structure_sources_use_mmcif() -> None:
    """Multi-character chain subsets need mmCIF fixtures to preserve chain IDs."""

    for case_id in ("3j9z-chain-sb", "6gz3-chain-bj"):
        source = WHOLE_STRUCTURE_CORPUS_SOURCES[case_id]

        assert source.output_path.suffix == ".cif"
        assert source.download_format_priority[0] == "cif"


def test_reference_backed_refinement_benchmark_registry_only_contains_references() -> (
    None
):
    """Reference-backed perturb registry should contain only reference cases."""

    assert "1xgo-leu253" in REFERENCE_REFINEMENT_BENCHMARK_CASES
    for case in REFERENCE_REFINEMENT_BENCHMARK_CASES.values():
        assert case.reference_fixture_path is not None


def test_perturbation_benchmark_registry_separates_reference_provenance() -> None:
    """Perturbation fixtures should distinguish external and self references."""

    assert "1xgo-leu253" in REFERENCE_REFINEMENT_BENCHMARK_CASES
    assert (
        "1jd0-gln92-self-reference" in SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES
    )
    assert "4jfx-ptr8-self-reference" in SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES
    assert set(PERTURBATION_REFINEMENT_BENCHMARK_CASES) == (
        set(REFERENCE_REFINEMENT_BENCHMARK_CASES)
        | set(SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES)
    )

    for case in SELF_REFERENCE_REFINEMENT_BENCHMARK_CASES.values():
        assert case.reference_fixture_path == case.fixture_path

    for scenario in PERTURBATION_BENCHMARK_SCENARIOS.values():
        assert scenario.scenario_id in PERTURBATION_BENCHMARK_SCENARIOS
        assert scenario.case.reference_fixture_path is not None

    assert (
        PERTURBATION_BENCHMARK_SCENARIOS[
            "4jfx-ptr8-rigid-medium-self-reference"
        ].expected_outcome
        is RefinementPerturbationExpectedOutcome.RUNNABLE_SUCCESS
    )


def test_refinement_perturbation_registry_exposes_builtin_modes() -> None:
    """Perturbation benchmark registry should expose deterministic built-in modes."""

    assert REFINEMENT_PERTURBATION_SPECS["rigid_small"].mode is (
        RefinementPerturbationMode.RANDOM_RIGID
    )
    assert REFINEMENT_PERTURBATION_SPECS["clash_shove"].mode is (
        RefinementPerturbationMode.DIRECTED_SHOVE
    )


def test_refinement_benchmark_profile_registry_exposes_builtin_profiles() -> None:
    """Benchmark profile registry should expose built-in comparison profiles."""

    assert REFINEMENT_BENCHMARK_PROFILES["rdkit_uff"].profile is (
        ContinuousRelaxationProfile.RDKIT_UFF
    )
    assert REFINEMENT_BENCHMARK_PROFILES["rdkit_mmff"].profile is (
        ContinuousRelaxationProfile.RDKIT_MMFF
    )
    assert set(REFINEMENT_BENCHMARK_PROFILES) == {"rdkit_uff", "rdkit_mmff"}


def test_refinement_corpus_active_panel_contains_current_execution_subset() -> None:
    """Active corpus panel should expose current representative and local cases."""

    active_case_ids = corpus_case_ids_for_panel(CorpusPanel.ACTIVE)

    assert "1afc-hydrogen-his-protonated" in active_case_ids
    assert "1bkr-his42" in active_case_ids
    assert "1xgo-leu253" in active_case_ids


def test_refinement_corpus_query_surfaces_expected_coverage_axes() -> None:
    """Coverage-tag queries should expose the current case owners."""

    assert corpus_case_ids_for_tag(CorpusCoverageTag.BACKBONE_DRIVEN_CLASH) == (
        "1xgo-leu253",
        "3g8l-asn182",
        "3j7h-binding-site-backbone",
        "3j9e-loop-backbone-error",
        "6l54-chain-c",
        "6z3r-chain-c",
        "7s9d-prestin-segment",
    )
    assert corpus_case_ids_for_tag(CorpusCoverageTag.COFACTOR_KEPT) == (
        "1cjc-heavy-keep-ligand",
        "1cjc-hydrogen-keep-ligand",
    )


def test_refinement_corpus_active_panel_keeps_expected_backbone_owners() -> None:
    """Active panel should keep the currently runnable backbone-clash owners."""

    active_case_ids = set(corpus_case_ids_for_panel(CorpusPanel.ACTIVE))

    assert {"1xgo-leu253", "3g8l-asn182"}.issubset(active_case_ids)
    assert "3j7h-binding-site-backbone" not in active_case_ids
    assert "3j9e-loop-backbone-error" not in active_case_ids


def test_refinement_corpus_active_panel_still_exposes_known_gaps() -> None:
    """Current active panel should still report the expected uncovered axes."""

    uncovered_tags = uncovered_tags_for_panel(
        CorpusPanel.ACTIVE,
        tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
                CorpusCoverageTag.RETAINED_NON_POLYMER_KEPT,
                CorpusCoverageTag.NOT_APPLICABLE_CHEMISTRY,
                CorpusCoverageTag.NO_CONECT_INGEST_SENSITIVE,
                CorpusCoverageTag.PERTURB_REFERENCE_PAIRED,
            }
        ),
    )

    assert uncovered_tags == frozenset()

    exploratory_gaps = uncovered_tags_for_panel(
        CorpusPanel.ACTIVE,
        tags=frozenset(
            {
                CorpusCoverageTag.COFACTOR_KEPT,
                CorpusCoverageTag.LOW_RESOLUTION,
            }
        ),
    )

    assert exploratory_gaps == frozenset({CorpusCoverageTag.LOW_RESOLUTION})


def test_refinement_corpus_exploratory_panel_seeds_cryo_em_and_special_chemistry() -> (
    None
):
    """Exploratory corpus should seed upcoming cryo-EM and special-chemistry work."""

    exploratory_case_ids = corpus_case_ids_for_panel(CorpusPanel.EXPLORATORY)

    assert "3j6b-terminal-helix-misthread" in exploratory_case_ids
    assert "3j5p-gln519-sidechain" in exploratory_case_ids
    assert "3j9e-loop-backbone-error" in exploratory_case_ids
    assert "3j7h-binding-site-backbone" in exploratory_case_ids
    assert "2h6o-glycan-problematic" in exploratory_case_ids
    assert "5k12-tyr471-sidechain" in exploratory_case_ids
    assert "5xm5-metal-sites" in exploratory_case_ids
    assert "7s9d-prestin-segment" in exploratory_case_ids
    assert "3ja8-whole-structure" in exploratory_case_ids
    assert "6eyc-whole-structure" in exploratory_case_ids
    assert "3j9z-chain-sb" in exploratory_case_ids
    assert "5n9y-whole-structure" in exploratory_case_ids
    assert "5owx-whole-structure" in exploratory_case_ids
    assert "5wsn-whole-structure" in exploratory_case_ids
    assert "6d83-whole-structure" in exploratory_case_ids
    assert "6dmb-chain-a" in exploratory_case_ids
    assert "6gz3-chain-bj" in exploratory_case_ids
    assert "6l54-chain-c" in exploratory_case_ids
    assert "6z3r-chain-c" in exploratory_case_ids


def test_refinement_corpus_execution_policy_separates_gating_from_tracking() -> None:
    """Execution policy should distinguish must-pass panel cases from tracking-only."""

    must_pass_case_ids = corpus_case_ids_for_execution_policy(
        CorpusExecutionPolicy.MUST_PASS
    )
    tracking_case_ids = corpus_case_ids_for_execution_policy(
        CorpusExecutionPolicy.TRACKING_ONLY
    )

    assert "1afc-hydrogen-his-protonated" in must_pass_case_ids
    assert "3j6b-terminal-helix-misthread" not in must_pass_case_ids
    assert "3j6b-terminal-helix-misthread" in tracking_case_ids
    assert "2h6o-glycan-problematic" in tracking_case_ids


def test_refinement_corpus_tracking_pressure_queries_surface_current_failure_classes(
) -> None:
    """Tracking pressure queries should expose the current failure buckets."""

    assert corpus_case_ids_for_tracking_pressure(
        CorpusTrackingPressure.OPERATOR_CAPABILITY
    ) == (
        "3j6b-terminal-helix-misthread",
        "3j9e-loop-backbone-error",
        "7s9d-prestin-segment",
    )
    assert corpus_case_ids_for_tracking_pressure(
        CorpusTrackingPressure.CHEMISTRY_BLOCKER
    ) == (
        "2h6o-glycan-problematic",
        "2q6f-cys143-pje",
        "2z62-whole-structure",
        "5xm5-metal-sites",
    )
    assert "3ja8-whole-structure" in corpus_case_ids_for_tracking_pressure(
        CorpusTrackingPressure.SCALING_GAP
    )


def test_refinement_corpus_maps_backbone_windows_to_operator_contract() -> None:
    """Backbone-window exploratory cases should name the missing operator family."""

    assert corpus_case_ids_for_required_operator_family(
        RefinementOperatorFamily.BACKBONE_WINDOW_REFINEMENT
    ) == (
        "3j6b-terminal-helix-misthread",
        "3j9e-loop-backbone-error",
        "7s9d-prestin-segment",
    )


def test_refinement_corpus_evaluation_mode_separates_operator_from_admissibility() -> (
    None
):
    """Evaluation mode should distinguish operator cases from chemistry gating."""

    chemistry_admissibility_case_ids = corpus_case_ids_for_evaluation_mode(
        CorpusEvaluationMode.CHEMISTRY_ADMISSIBILITY
    )
    operator_benchmark_case_ids = corpus_case_ids_for_evaluation_mode(
        CorpusEvaluationMode.OPERATOR_BENCHMARK
    )

    assert chemistry_admissibility_case_ids == (
        "2h6o-glycan-problematic",
        "2q6f-cys143-pje",
        "2z62-whole-structure",
        "5xm5-metal-sites",
    )
    assert "5xm5-metal-sites" not in operator_benchmark_case_ids
    assert "3j6b-terminal-helix-misthread" in operator_benchmark_case_ids
