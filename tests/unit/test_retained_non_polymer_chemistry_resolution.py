"""Unit tests for retained non-polymer chemistry resolution contracts."""

import pytest

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    RetainedNonPolymerChemistryResolution,
)


def test_unresolved_retained_chemistry_cannot_carry_resolved_facts() -> None:
    """Unresolved retained chemistry must not smuggle topology facts."""

    with pytest.raises(ValueError, match="resolved topology"):
        RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
            heavy_bond_definitions=(BondDefinition("C1", "O1"),),
        )


def test_resolved_retained_chemistry_cannot_carry_failure_reason() -> None:
    """A resolved chemistry source and a failure reason are contradictory."""

    with pytest.raises(ValueError, match="failure reason"):
        RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE,
            failure_reason="bad smiles",
        )


def test_projection_diagnostics_are_rdkit_fallback_only() -> None:
    """Name-projection diagnostics belong to generated RDKit fallback chemistry."""

    with pytest.raises(ValueError, match="only valid for RDKit fallback"):
        RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.TEMPLATE,
            hydrogen_name_projection_candidate_limit=4,
        )


def test_projection_candidate_count_requires_limit() -> None:
    """Projection candidate counts are meaningless without the evaluated limit."""

    with pytest.raises(ValueError, match="candidate_limit is required"):
        RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK,
            rdkit_backend_version="2026.03.2",
            hydrogen_name_projection_candidate_count=2,
        )


def test_rdkit_fallback_requires_backend_version() -> None:
    """RDKit fallback chemistry must identify the backend that inferred it."""

    with pytest.raises(ValueError, match="backend version"):
        RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK,
        )


def test_rdkit_backend_version_is_rdkit_fallback_only() -> None:
    """Backend-version provenance belongs only to RDKit fallback chemistry."""

    with pytest.raises(ValueError, match="only valid for RDKit fallback"):
        RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.TEMPLATE,
            rdkit_backend_version="2026.03.2",
        )


def test_rdkit_backend_version_is_normalized() -> None:
    """Resolution stores a canonical backend-version token."""

    resolution = RetainedNonPolymerChemistryResolution(
        source=RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK,
        rdkit_backend_version=" 2026.03.2 ",
    )

    assert resolution.rdkit_backend_version == "2026.03.2"


def test_rdkit_backend_version_must_not_be_blank() -> None:
    """Blank backend-version provenance is equivalent to missing provenance."""

    with pytest.raises(ValueError, match="backend version"):
        RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK,
            rdkit_backend_version="   ",
        )
