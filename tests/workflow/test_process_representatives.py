"""Representative workflow replay tests."""

import sys
from pathlib import Path

import pytest
from tests.support.release_gate import (
    STRICT_RDKIT_RELEASE_GATE_ENV,
    strict_rdkit_release_gate_enabled,
)
from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.request_builders import (
    ingress_options,
    transform_requests,
    whole_structure_requested_goals,
)
from tests.support.structure_summary import (
    semantic_digest_for_structure,
    structure_summaries_match_except_digest,
    structure_summary_mismatch_report,
    summarize_structure,
)
from tests.support.whole_structure_sources import WHOLE_STRUCTURE_CORPUS_SOURCES

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.parser_readability import (
    diagnose_rdkit_no_conect_sanitize_readability,
    measure_rdkit_no_conect_sanitize_readability,
    rdkit_no_conect_extra_proximity_bond_clusters,
    rdkit_no_conect_parser_failing_residue_ids,
    rdkit_no_conect_parser_problem_witnesses,
)
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import (
    LigandPolicy,
    ProcessResult,
)
from protrepair.workflow.engine import normalize_source_structure

WORKFLOW_REPRESENTATIVE_CASE_IDS: tuple[str, ...] = (
    "1aho-hydrogen-default",
    "1cjc-heavy-keep-ligand",
    "1cjc-hydrogen-keep-ligand",
    "1afc-hydrogen-his-protonated",
)
WORKFLOW_RDKIT_COORDINATE_DIGESTS_2DP: dict[str, dict[str, frozenset[str]]] = {
    # RDKit force-field coordinates can drift across backend/scientific-stack
    # environments even when topology, atom ordering, and workflow issues are
    # unchanged. Keep this as a backend-version-bounded regression check over
    # known digest sets, not a portable semantic digest.
    "1afc-hydrogen-his-protonated": {
        "2026.03.2": frozenset(
            {
                "222fcdf1b738d90a5ef2762a82317b0fcabbfc78210ed3eeb9f54847e0522834",
            }
        ),
        "2026.03.3": frozenset(
            {
                "222fcdf1b738d90a5ef2762a82317b0fcabbfc78210ed3eeb9f54847e0522834",
            }
        ),
    },
}
pytestmark = pytest.mark.workflow


@pytest.mark.representative_regression
@pytest.mark.parametrize("case_id", WORKFLOW_REPRESENTATIVE_CASE_IDS)
def test_process_structure_preserves_representative_semantics(
    case_id: str,
) -> None:
    """Workflow replay should preserve the stored representative summary."""

    expected = REPRESENTATIVE_CASES[case_id]
    result = run_workflow_representative_case(expected.input_path, case_id)
    assert not result.has_errors()
    summary = summarize_structure(result.structure)

    if case_id in WORKFLOW_RDKIT_COORDINATE_DIGESTS_2DP:
        assert structure_summaries_match_except_digest(
            summary,
            expected.summary,
        ), structure_summary_mismatch_report(summary, expected.summary)
    else:
        assert summary == expected.summary, structure_summary_mismatch_report(
            summary,
            expected.summary,
        )


@pytest.mark.representative_regression
@pytest.mark.parametrize(
    "case_id",
    tuple(WORKFLOW_RDKIT_COORDINATE_DIGESTS_2DP),
)
def test_process_structure_preserves_rdkit_coordinate_digest(case_id: str) -> None:
    """RDKit-backed coordinate digest should match registered backend versions."""

    expected = REPRESENTATIVE_CASES[case_id]
    result = run_workflow_representative_case(expected.input_path, case_id)
    assert not result.has_errors()
    summary = summarize_structure(result.structure)
    assert structure_summaries_match_except_digest(
        summary,
        expected.summary,
    ), structure_summary_mismatch_report(summary, expected.summary)
    _assert_rdkit_coordinate_digest_matches(
        case_id,
        semantic_digest_for_structure(
            result.structure,
            coordinate_decimal_places=2,
        ),
    )


def _assert_rdkit_coordinate_digest_matches(
    case_id: str,
    actual_digest: str,
) -> None:
    expected_by_version = WORKFLOW_RDKIT_COORDINATE_DIGESTS_2DP[case_id]
    rdkit_version = _rdkit_version()
    if rdkit_version not in expected_by_version:
        message = (
            f"{case_id} coordinate digest is RDKit-version-bound; "
            f"no 2dp digest is registered for RDKit {rdkit_version!r}"
        )
        if strict_rdkit_release_gate_enabled():
            pytest.fail(message)
        pytest.skip(message)

    assert actual_digest in expected_by_version[rdkit_version]


def _rdkit_version() -> str | None:
    try:
        from rdkit import rdBase
    except ImportError:
        return None

    return str(rdBase.rdkitVersion)


def test_unknown_rdkit_coordinate_digest_skips_outside_release_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unregistered RDKit versions stay skippable outside release strict mode."""

    monkeypatch.delenv(STRICT_RDKIT_RELEASE_GATE_ENV, raising=False)
    _patch_rdkit_version(monkeypatch, "2099.99.9")

    with pytest.raises(pytest.skip.Exception):
        _assert_rdkit_coordinate_digest_matches(
            "1afc-hydrogen-his-protonated",
            "unused",
        )


def test_registered_rdkit_coordinate_digest_accepts_current_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each registered RDKit backend should accept the current-code digest."""

    for rdkit_version in ("2026.03.2", "2026.03.3"):
        _patch_rdkit_version(monkeypatch, rdkit_version)
        _assert_rdkit_coordinate_digest_matches(
            "1afc-hydrogen-his-protonated",
            "222fcdf1b738d90a5ef2762a82317b0fcabbfc78210ed3eeb9f54847e0522834",
        )


def test_unknown_rdkit_coordinate_digest_fails_release_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release strict mode should fail instead of skipping unknown RDKit."""

    monkeypatch.setenv(STRICT_RDKIT_RELEASE_GATE_ENV, "1")
    _patch_rdkit_version(monkeypatch, "2099.99.9")

    with pytest.raises(pytest.fail.Exception):
        _assert_rdkit_coordinate_digest_matches(
            "1afc-hydrogen-his-protonated",
            "unused",
        )


def test_unknown_rdkit_coordinate_digest_ignores_non_release_truthy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict release behavior is opt-in only for the documented env value."""

    monkeypatch.setenv(STRICT_RDKIT_RELEASE_GATE_ENV, "true")
    _patch_rdkit_version(monkeypatch, "2099.99.9")

    with pytest.raises(pytest.skip.Exception):
        _assert_rdkit_coordinate_digest_matches(
            "1afc-hydrogen-his-protonated",
            "unused",
        )


def test_missing_rdkit_coordinate_digest_fails_release_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release strict mode treats missing RDKit as missing digest coverage."""

    monkeypatch.setenv(STRICT_RDKIT_RELEASE_GATE_ENV, "1")
    _patch_rdkit_version(monkeypatch, None)

    with pytest.raises(pytest.fail.Exception):
        _assert_rdkit_coordinate_digest_matches(
            "1afc-hydrogen-his-protonated",
            "unused",
        )


def _patch_rdkit_version(
    monkeypatch: pytest.MonkeyPatch,
    rdkit_version: str | None,
) -> None:
    """Patch the module-local RDKit version probe regardless of collection name."""

    monkeypatch.setattr(
        sys.modules[__name__],
        "_rdkit_version",
        lambda: rdkit_version,
    )


@pytest.mark.representative_regression
def test_process_structure_repairs_1afc_to_no_conect_rdkit_readable_output() -> None:
    """Hydrogenated 1AFC workflow output should survive no-CONECT RDKit sanitize."""

    expected = REPRESENTATIVE_CASES["1afc-hydrogen-his-protonated"]
    result = run_workflow_representative_case(
        expected.input_path,
        "1afc-hydrogen-his-protonated",
    )

    assert (
        measure_rdkit_no_conect_sanitize_readability(result.structure)
        is True
    )


@pytest.mark.representative_regression
def test_process_structure_explicit_repair_keeps_3j6b_no_conect_rdkit_readable_output(
) -> None:
    """Explicit 3J6B repair should keep no-CONECT RDKit sanitize readability."""

    source = Path(
        "tests/fixtures/pdb/refinement/"
        "3j6b_terminal_helix_misthread_local.pdb"
    )

    assert (
        measure_rdkit_no_conect_sanitize_readability(
            process_structure(source).structure
        )
        is True
    )

    result = process_structure(
        source,
        transform_requests=transform_requests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues(
                    (
                        ResidueId("9", 149),
                        ResidueId("9", 152),
                        ResidueId("9", 235),
                    )
                ),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )

    assert measure_rdkit_no_conect_sanitize_readability(result.structure) is True
    assert not result.issues


@pytest.mark.representative_regression
def test_process_structure_2dn2_heme_cofactor_output_stays_no_conect_rdkit_readable(
) -> None:
    """2DN2 should stay RDKit-readable after template-backed HEM hydrogenation."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["2dn2-whole-structure"]
    result = process_structure(
        source.output_path,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=(source.selected_source_chain_ids or None),
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
    )

    assert measure_rdkit_no_conect_sanitize_readability(result.structure) is True


@pytest.mark.representative_regression
def test_process_structure_2qls_heme_cofactor_uses_template_backed_hydrogens(
) -> None:
    """2QLS HEM ligands should use bundled template hydrogens, not generic fallback."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["2qls-whole-structure"]
    result = process_structure(
        source.output_path,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=(source.selected_source_chain_ids or None),
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
    )

    hem_ligands = tuple(
        ligand
        for ligand in result.structure.constitution.ligands
        if ligand.component_id == "HEM"
    )
    assert hem_ligands

    for ligand in hem_ligands:
        hydrogen_names = tuple(
            atom.name
            for atom in ligand.atom_sites
            if atom.element == "H"
        )
        assert len(hydrogen_names) == 32
        assert all(not name.startswith("H0") for name in hydrogen_names)
        assert "HHA" in hydrogen_names
        assert "H2A" in hydrogen_names


@pytest.mark.representative_regression
def test_process_structure_2qls_output_stays_no_conect_rdkit_readable() -> None:
    """2QLS should repair the LEU-local parser failure in default workflow."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["2qls-whole-structure"]
    result = process_structure(
        source.output_path,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=(source.selected_source_chain_ids or None),
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
    )

    assert measure_rdkit_no_conect_sanitize_readability(result.structure) is True
    assert not any(
        issue.kind is ValidationIssueKind.PARSER_READABILITY
        for issue in result.issues
    )
    assert not diagnose_rdkit_no_conect_sanitize_readability(result.structure)
    assert not rdkit_no_conect_parser_failing_residue_ids(result.structure)


@pytest.mark.representative_regression
def test_2qls_parser_witness_exposes_extra_leu32_proximity_bond() -> None:
    """2QLS parser diagnostics should explain the RDKit-inferred extra bond."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["2qls-whole-structure"]
    component_library = build_default_component_library()
    structure = normalize_source_structure(
        source.output_path,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=("D",),
    )
    structure = add_hydrogens(
        structure,
        component_library=component_library,
    ).structure
    structure = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
    ).structure

    witnesses = rdkit_no_conect_parser_problem_witnesses(
        structure,
        component_library=component_library,
    )
    leu32_witnesses = tuple(
        witness
        for witness in witnesses
        if witness.residue_id == ResidueId("D", 32)
    )

    assert len(leu32_witnesses) == 1
    extra_bond_tokens = tuple(
        bond.display_token()
        for bond in leu32_witnesses[0].extra_proximity_bonds()
    )
    assert (
        "D:32.CD1-D:32.CD2" in extra_bond_tokens
        or "D:32.CD2-D:32.CD1" in extra_bond_tokens
    )
    clusters = rdkit_no_conect_extra_proximity_bond_clusters(
        structure,
        component_library=component_library,
    )
    assert clusters[0].residue_ids == (ResidueId("D", 32),)


@pytest.mark.representative_regression
def test_process_structure_2dn2_has_no_parser_readability_issue() -> None:
    """2DN2 should stay free of parser-readability issues after HEM repair."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["2dn2-whole-structure"]
    result = process_structure(
        source.output_path,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=(source.selected_source_chain_ids or None),
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
    )

    assert not any(
        issue.kind is ValidationIssueKind.PARSER_READABILITY
        for issue in result.issues
    )


@pytest.mark.representative_regression
def test_process_structure_6nbb_nad_output_stays_no_conect_rdkit_readable() -> None:
    """6NBB should stay RDKit-readable after NAD hydrogen placement."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["6nbb-whole-structure"]
    result = process_structure(
        source.output_path,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        selected_source_chain_ids=(source.selected_source_chain_ids or None),
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE,
        ),
    )

    assert measure_rdkit_no_conect_sanitize_readability(result.structure) is True
    assert not any(
        issue.kind is ValidationIssueKind.PARSER_READABILITY
        and issue.residue_id == ResidueId("A", 401)
        for issue in result.issues
    )

def run_workflow_representative_case(
    source: Path,
    case_id: str,
) -> ProcessResult:
    """Run one representative workflow scenario through the public API."""

    if case_id == "1cjc-heavy-keep-ligand":
        return process_structure(
            source,
            ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        )

    if case_id == "1aho-hydrogen-default":
        return process_structure(
            source,
            requested_goals=whole_structure_requested_goals(
                HydrogenCoverageState.COMPLETE,
            ),
        )

    if case_id == "1cjc-hydrogen-keep-ligand":
        return process_structure(
            source,
            ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
            requested_goals=whole_structure_requested_goals(
                HydrogenCoverageState.COMPLETE,
            ),
        )

    if case_id == "1afc-hydrogen-his-protonated":
        return process_structure(
            source,
            requested_goals=whole_structure_requested_goals(
                HydrogenCoverageState.COMPLETE,
            ),
            transform_requests=transform_requests(
                protonate_histidines=True,
            ),
        )

    raise ValueError(f"unsupported workflow representative case: {case_id}")
