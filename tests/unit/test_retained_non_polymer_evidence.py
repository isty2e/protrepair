"""Unit tests for retained non-polymer RDKit evidence inference."""

import pytest

from protrepair.chemistry.inference import (
    retained_non_polymer_evidence as evidence_inference,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_expected_hydrogen_atom_names,
    template_with_hydrogens,
    template_without_hydrogens,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.structure.labels import ResidueId

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required dependency import guard
    Chem = None

RDKIT_AVAILABLE = Chem is not None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_evidence_template_without_hydrogens_returns_defensive_copy() -> None:
    """Caller mutation of one returned heavy template must not poison caches."""

    _clear_evidence_template_caches()
    first_template = template_without_hydrogens("CO")

    first_template.GetAtomWithIdx(0).SetFormalCharge(1)
    first_template.GetAtomWithIdx(0).SetProp("poisoned", "yes")

    second_template = template_without_hydrogens("CO")

    assert second_template.GetAtomWithIdx(0).GetFormalCharge() == 0
    assert not second_template.GetAtomWithIdx(0).HasProp("poisoned")


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_evidence_template_with_hydrogens_returns_defensive_copy() -> None:
    """Caller mutation of one returned H template must not affect inference."""

    _clear_evidence_template_caches()
    evidence = RetainedNonPolymerChemistryEvidence(
        residue_id=ResidueId("L", 1),
        smiles="CO",
        heavy_atom_names=("C1", "O1"),
    )
    first_template = template_with_hydrogens(evidence.smiles)
    hydrogen_atom = next(
        atom for atom in first_template.GetAtoms() if atom.GetAtomicNum() == 1
    )

    hydrogen_atom.SetAtomicNum(6)
    hydrogen_atom.SetProp("poisoned", "yes")

    second_template = template_with_hydrogens(evidence.smiles)

    assert any(atom.GetAtomicNum() == 1 for atom in second_template.GetAtoms())
    assert all(not atom.HasProp("poisoned") for atom in second_template.GetAtoms())
    assert retained_non_polymer_evidence_expected_hydrogen_atom_names(
        evidence
    ) == ("H001", "H002", "H003", "H004")


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_mutated_heavy_template_does_not_affect_hydrogenated_template() -> None:
    """A mutated heavy-template projection must not change later H inference."""

    _clear_evidence_template_caches()
    evidence = RetainedNonPolymerChemistryEvidence(
        residue_id=ResidueId("L", 1),
        smiles="CO",
        heavy_atom_names=("C1", "O1"),
    )
    heavy_template = template_without_hydrogens(evidence.smiles)

    heavy_template.GetAtomWithIdx(1).SetAtomicNum(6)

    assert retained_non_polymer_evidence_expected_hydrogen_atom_names(
        evidence
    ) == ("H001", "H002", "H003", "H004")


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_evidence_template_caches_are_bounded() -> None:
    """Arbitrary evidence SMILES must not grow RDKit template caches unbounded."""

    assert (
        evidence_inference._cached_template_without_hydrogens.cache_info().maxsize
        == 256
    )
    assert (
        evidence_inference._cached_template_with_hydrogens.cache_info().maxsize
        == 256
    )


def _clear_evidence_template_caches() -> None:
    """Clear old and new cache owners to keep tests order-independent."""

    for function_name in (
        "template_without_hydrogens",
        "template_with_hydrogens",
        "_cached_template_without_hydrogens",
        "_cached_template_with_hydrogens",
    ):
        function = getattr(evidence_inference, function_name, None)
        cache_clear = getattr(function, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
