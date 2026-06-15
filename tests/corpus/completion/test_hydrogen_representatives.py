"""Representative regression coverage for hydrogen placement."""

import pytest
from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.request_builders import ingress_options
from tests.support.structure_summary import summarize_structure

from protrepair.io import read_structure
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

HYDROGEN_REPRESENTATIVE_CASE_IDS: tuple[str, ...] = (
    "1aho-hydrogen-default",
    "1cjc-hydrogen-keep-ligand",
    "1afc-hydrogen-his-protonated",
)
pytestmark = pytest.mark.corpus


@pytest.mark.representative_regression
@pytest.mark.parametrize("case_id", HYDROGEN_REPRESENTATIVE_CASE_IDS)
def test_add_hydrogens_preserves_representative_semantics(case_id: str) -> None:
    """Hydrogen placement should preserve stored representative summaries."""

    expected = REPRESENTATIVE_CASES[case_id]
    expected_summary = (
        expected.hydrogen_direct_summary
        if expected.hydrogen_direct_summary is not None
        else expected.summary
    )
    result = run_hydrogen_representative_case(case_id)
    summary = summarize_structure(result.structure)

    assert summary == expected_summary
    assert not result.has_errors()


def run_hydrogen_representative_case(case_id: str) -> TransformationResult:
    """Hydrogenate one representative input using its scenario ingress policy."""

    expected = REPRESENTATIVE_CASES[case_id]
    structure = read_structure(
        expected.input_path,
        policy=options_for_case(case_id).structure_normalization_policy(),
    )
    return add_hydrogens(
        structure,
        protonate_histidines=expected.protonate_histidines,
    )


def options_for_case(case_id: str) -> StructureIngressOptions:
    """Return ingress options matching one representative hydrogen scenario."""

    if case_id == "1cjc-hydrogen-keep-ligand":
        return ingress_options(ligand_policy=LigandPolicy.KEEP)

    return StructureIngressOptions()
