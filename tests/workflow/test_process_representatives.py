"""Representative workflow replay tests."""

from pathlib import Path

import pytest
from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.request_builders import (
    ingress_options,
    transform_requests,
    whole_structure_requested_goals,
)
from tests.support.structure_summary import summarize_structure
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
pytestmark = pytest.mark.workflow


@pytest.mark.representative_regression
@pytest.mark.parametrize("case_id", WORKFLOW_REPRESENTATIVE_CASE_IDS)
def test_process_structure_preserves_representative_semantics(
    case_id: str,
) -> None:
    """Workflow replay should preserve the stored representative summary."""

    expected = REPRESENTATIVE_CASES[case_id]
    result = run_workflow_representative_case(expected.input_path, case_id)
    summary = summarize_structure(result.structure)

    assert summary == expected.summary
    assert not result.has_errors()


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
