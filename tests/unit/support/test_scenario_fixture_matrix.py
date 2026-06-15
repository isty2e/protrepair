"""Scenario fixture matrix coverage guards."""

from tests.support.scenario_fixture_matrix import (
    REQUIRED_SCENARIO_CONTEXT_COVERAGE,
    REQUIRED_SCENARIO_DEFECT_COVERAGE,
    REQUIRED_SCENARIO_PHASE_COVERAGE,
    SCENARIO_FIXTURE_MATRIX,
    ScenarioContextTag,
    ScenarioDefectTag,
    ScenarioFixtureKind,
    ScenarioFixtureStatus,
    ScenarioPlanningPhase,
    collected_scenario_context_tags,
    collected_scenario_defect_tags,
    collected_scenario_phase_tags,
)

from protrepair.workflow.planning.capability import (
    WorkflowActionDonorRequirement,
    WorkflowActionEffectClass,
    WorkflowActionLocality,
)
from protrepair.workflow.planning.default_action_registry import (
    WORKFLOW_ACTION_REGISTRY,
)


def test_scenario_fixture_matrix_covers_required_defect_families() -> None:
    """The matrix should cover every target defect family."""

    assert REQUIRED_SCENARIO_DEFECT_COVERAGE <= collected_scenario_defect_tags()


def test_scenario_fixture_matrix_covers_required_context_families() -> None:
    """The matrix should cover every target planning context family."""

    assert REQUIRED_SCENARIO_CONTEXT_COVERAGE <= collected_scenario_context_tags()


def test_scenario_fixture_matrix_covers_required_planner_phases() -> None:
    """The matrix should exercise every target planner phase."""

    assert REQUIRED_SCENARIO_PHASE_COVERAGE <= collected_scenario_phase_tags()


def test_scenario_fixture_matrix_contains_literature_backed_shortlist_cases() -> None:
    """The matrix should keep the agreed literature-backed shortlist visible."""

    expected_fixture_ids = {
        "1bkr-thr101",
        "1bkr-his42",
        "1lxa-his125",
        "1uaq-gln55",
        "1xgo-leu253",
        "3g8l-asn182",
        "1ywr-his148",
        "1jd0-gln92",
        "1ywr-asn155",
        "2q6f-cys143-pje",
    }

    actual_fixture_ids = {
        fixture_id
        for fixture_id, entry in SCENARIO_FIXTURE_MATRIX.items()
        if entry.kind is ScenarioFixtureKind.LITERATURE_LOCAL
    }

    assert expected_fixture_ids <= actual_fixture_ids


def test_scenario_fixture_matrix_contains_representative_whole_structure_cases() -> (
    None
):
    """The matrix should keep representative whole-structure workflows visible."""

    expected_fixture_ids = {
        "1aho-heavy-default",
        "1aho-hydrogen-default",
        "1cjc-heavy-keep-ligand",
        "1cjc-hydrogen-keep-ligand",
        "1afc-hydrogen-his-protonated",
    }

    actual_fixture_ids = {
        fixture_id
        for fixture_id, entry in SCENARIO_FIXTURE_MATRIX.items()
        if entry.kind is ScenarioFixtureKind.REPRESENTATIVE_WHOLE
    }

    assert expected_fixture_ids <= actual_fixture_ids


def test_coverage_scenarios_do_not_collapse_back_into_mixed_phase_bags() -> None:
    """Coverage scenarios should stay distinct from later planning phases."""

    coverage_only_defects = {
        ScenarioDefectTag.MISSING_SIDECHAIN_ATOMS,
        ScenarioDefectTag.MISSING_MAINCHAIN_ATOMS,
        ScenarioDefectTag.MISSING_INTERNAL_RESIDUE_SPAN,
        ScenarioDefectTag.MISSING_PREFIX_TERMINAL_SPAN,
        ScenarioDefectTag.MISSING_SUFFIX_TERMINAL_SPAN,
    }

    coverage_entries = [
        entry
        for entry in SCENARIO_FIXTURE_MATRIX.values()
        if coverage_only_defects & set(entry.defect_tags)
    ]

    assert coverage_entries
    assert all(
        entry.planning_phases == (ScenarioPlanningPhase.COVERAGE,)
        for entry in coverage_entries
    )


def test_negative_controls_do_not_disappear_from_matrix() -> None:
    """Negative-control scenarios should remain explicit matrix entries."""

    negative_fixture_ids = {
        fixture_id
        for fixture_id, entry in SCENARIO_FIXTURE_MATRIX.items()
        if entry.status is ScenarioFixtureStatus.NEGATIVE_CONTROL
    }

    assert {
        "2q6f-cys143-pje",
        "synthetic-unresolved-ligand-clashing-serine-hydrogen",
    } <= negative_fixture_ids


def test_selected_scenarios_match_current_capability_registry() -> None:
    """Canonical scenario expectations should match at least one current capability."""

    expected_fixture_ids = {
        "synthetic-sidechain-gap-existing-residue",
        "synthetic-internal-blueprint-gap",
        "3g8l-asn182",
        "synthetic-ligand-clashing-serine-hydrogen",
        "1aho-hydrogen-default",
        "1cjc-hydrogen-keep-ligand",
    }

    for fixture_id in expected_fixture_ids:
        expectation = SCENARIO_FIXTURE_MATRIX[fixture_id].capability_expectation
        assert not expectation.expects_no_action_by_default
        assert any(
            expectation.matches_capability(registry_entry.capability)
            for registry_entry in WORKFLOW_ACTION_REGISTRY
        )


def test_terminal_gap_scenarios_require_external_donor_and_opt_in_context() -> None:
    """Terminal-gap donor cases should stay explicit about donor and opt-in policy."""

    for fixture_id, context_tag in (
        (
            "synthetic-prefix-terminal-blueprint-gap",
            ScenarioContextTag.PREFIX_TERMINAL_GAP_OPT_IN,
        ),
        (
            "synthetic-suffix-terminal-blueprint-gap",
            ScenarioContextTag.SUFFIX_TERMINAL_GAP_OPT_IN,
        ),
    ):
        entry = SCENARIO_FIXTURE_MATRIX[fixture_id]
        assert context_tag in entry.context_tags
        assert entry.capability_expectation.admissible_donor_requirements == (
            WorkflowActionDonorRequirement.EXTERNAL_DONOR,
        )
        assert entry.capability_expectation.preferred_localities == (
            WorkflowActionLocality.RESIDUE_SPAN,
        )


def test_holo_interaction_scenarios_keep_policy_and_post_augmentation_flags() -> None:
    """Holo interaction cases should keep explicit interaction-policy expectations."""

    interaction_entries = [
        entry
        for entry in SCENARIO_FIXTURE_MATRIX.values()
        if ScenarioDefectTag.LIGAND_AWARE_INTERACTION_CLASH in entry.defect_tags
        or ScenarioDefectTag.POST_HYDROGEN_INTERACTION_CLASH in entry.defect_tags
    ]

    assert interaction_entries
    assert all(
        entry.capability_expectation.requires_holo_interaction_policy
        for entry in interaction_entries
    )
    assert (
        SCENARIO_FIXTURE_MATRIX[
            "synthetic-ligand-clashing-serine-hydrogen"
        ].capability_expectation.expects_post_augmentation_correction
        is True
    )


def test_whole_structure_baselines_encode_no_action_by_default() -> None:
    """Representative baselines should keep anti-over-surgery pressure."""

    for fixture_id in ("1aho-heavy-default", "1cjc-heavy-keep-ligand"):
        expectation = SCENARIO_FIXTURE_MATRIX[fixture_id].capability_expectation
        assert expectation.expects_no_action_by_default is True
        assert expectation.reducible_deficit_families == ()
        assert expectation.preferred_localities == ()
        assert expectation.admissible_effect_classes == ()


def test_literature_local_intrinsic_cases_prefer_local_geometry_revision() -> None:
    """Literature local correction cases should stay local and geometry-revising."""

    intrinsic_entries = [
        entry
        for entry in SCENARIO_FIXTURE_MATRIX.values()
        if entry.kind is ScenarioFixtureKind.LITERATURE_LOCAL
        and ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY in entry.defect_tags
        and ScenarioDefectTag.LIGAND_AWARE_INTERACTION_CLASH not in entry.defect_tags
    ]

    assert intrinsic_entries
    assert all(
        WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY
        in entry.capability_expectation.admissible_effect_classes
        for entry in intrinsic_entries
    )
    assert all(
        WorkflowActionLocality.LOCAL_SCOPE
        in entry.capability_expectation.preferred_localities
        for entry in intrinsic_entries
    )
