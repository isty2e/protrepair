"""Unit tests for literature-backed element radii ownership."""

from protrepair.chemistry import (
    COVALENT_RADII_SOURCE,
    DEFAULT_COVALENT_RADIUS_ANGSTROM,
    DEFAULT_VAN_DER_WAALS_RADIUS_ANGSTROM,
    VAN_DER_WAALS_RADII_SOURCE,
    covalent_radius_angstrom,
    van_der_waals_radius_angstrom,
)
from protrepair.transformer.completion.hydrogen.geometry import (
    rotatable_hydrogen_vdw_radius_angstrom,
)


def test_van_der_waals_radii_match_bondi_consensus_values() -> None:
    """Supported vdW radii should match the centralized Bondi-family owner."""

    assert "10.1021/j100785a001" in VAN_DER_WAALS_RADII_SOURCE
    assert "10.1021/j100881a503" in VAN_DER_WAALS_RADII_SOURCE
    assert "10.1021/jp8111556" in VAN_DER_WAALS_RADII_SOURCE
    assert van_der_waals_radius_angstrom("H") == 1.20
    assert van_der_waals_radius_angstrom("C") == 1.70
    assert van_der_waals_radius_angstrom("O") == 1.52
    assert van_der_waals_radius_angstrom("Se") == 1.90
    assert van_der_waals_radius_angstrom("ZN") == 1.39
    assert van_der_waals_radius_angstrom("Mg") == 1.73
    assert van_der_waals_radius_angstrom("Ca") == 2.31
    assert van_der_waals_radius_angstrom("Fe") == 2.00
    assert van_der_waals_radius_angstrom("I") == 1.98


def test_covalent_radii_match_cordero_values_for_supported_elements() -> None:
    """Supported covalent radii should come from the centralized Cordero owner."""

    assert "10.1039/B801115J" in COVALENT_RADII_SOURCE
    assert covalent_radius_angstrom("H") == 0.31
    assert covalent_radius_angstrom("C") == 0.76
    assert covalent_radius_angstrom("N") == 0.71
    assert covalent_radius_angstrom("P") == 1.07
    assert covalent_radius_angstrom("Se") == 1.20
    assert covalent_radius_angstrom("ZN") == 1.22
    assert covalent_radius_angstrom("Mg") == 1.41
    assert covalent_radius_angstrom("Ca") == 1.76
    assert covalent_radius_angstrom("Fe") == 1.32
    assert covalent_radius_angstrom("CL") == 1.02


def test_unsupported_element_radii_keep_explicit_default_policy() -> None:
    """Table additions must not change the unknown-element fallback contract."""

    assert van_der_waals_radius_angstrom("XX") == DEFAULT_VAN_DER_WAALS_RADIUS_ANGSTROM
    assert covalent_radius_angstrom("XX") == DEFAULT_COVALENT_RADIUS_ANGSTROM


def test_rotatable_hydrogen_scoring_uses_shared_vdw_owner() -> None:
    """Hydrogen scoring should not keep a duplicate vdW table."""

    assert rotatable_hydrogen_vdw_radius_angstrom("S") == van_der_waals_radius_angstrom(
        "S"
    )
