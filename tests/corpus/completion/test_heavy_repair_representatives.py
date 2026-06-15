"""Representative regression coverage for heavy-atom repair."""

import pytest
from tests.support.representative_cases import REPRESENTATIVE_CASES
from tests.support.structure_summary import summarize_structure

from protrepair.io import read_structure
from protrepair.io.ingress_policy import LigandHandling, StructureNormalizationPolicy
from protrepair.transformer.completion.heavy import repair_heavy_atoms

HEAVY_REPAIR_REPRESENTATIVE_CASE_IDS: tuple[str, ...] = (
    "1aho-heavy-default",
    "1cjc-heavy-keep-ligand",
    "1aar-heavy-chain-1",
)
pytestmark = pytest.mark.corpus


@pytest.mark.representative_regression
@pytest.mark.parametrize("case_id", HEAVY_REPAIR_REPRESENTATIVE_CASE_IDS)
def test_repair_heavy_atoms_preserves_representative_semantics(
    case_id: str,
) -> None:
    """Heavy-atom repair should preserve the stored representative summary."""

    expected = REPRESENTATIVE_CASES[case_id]
    structure = read_structure(
        expected.input_path,
        policy=normalization_policy_for_case(case_id),
    )

    result = repair_heavy_atoms(structure)
    summary = summarize_structure(result.structure)

    assert summary == expected.summary
    assert not result.has_errors()


def normalization_policy_for_case(
    case_id: str,
) -> StructureNormalizationPolicy:
    """Return normalization policy matching one representative heavy scenario."""

    if case_id == "1cjc-heavy-keep-ligand":
        return StructureNormalizationPolicy(
            ligand_handling=LigandHandling.KEEP,
        )

    if case_id == "1aar-heavy-chain-1":
        return StructureNormalizationPolicy(selected_chain_ids=("A",))

    return StructureNormalizationPolicy()
