"""Unit tests for literature-backed element radii ownership."""

import pytest

from protrepair.chemistry import (
    COVALENT_RADII_SOURCE,
    RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_SOURCE,
    RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_VERSION,
    VAN_DER_WAALS_RADII_SOURCE,
    ElementRadiusLookup,
    ElementRadiusResolution,
    ElementRadiusResolutionStatus,
    RadiusKind,
    UnknownElementRadiusError,
    covalent_radius_angstrom,
    normalize_radius_element_symbol,
    prepare_radius_lookup,
    resolve_element_radius,
    van_der_waals_radius_angstrom,
)
from protrepair.transformer.completion.hydrogen.geometry import (
    rotatable_hydrogen_vdw_radius_angstrom,
)


def test_van_der_waals_radii_match_rdkit_periodic_table_snapshot() -> None:
    """Supported vdW radii should match the static RDKit PeriodicTable snapshot."""

    assert RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_VERSION == "2026.03.2"
    assert "rdkit==2026.3.2" in RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_SOURCE
    assert "GetRvdw" in VAN_DER_WAALS_RADII_SOURCE
    assert van_der_waals_radius_angstrom("H") == 1.20
    assert van_der_waals_radius_angstrom("C") == 1.70
    assert van_der_waals_radius_angstrom("N") == 1.60
    assert van_der_waals_radius_angstrom("O") == 1.55
    assert van_der_waals_radius_angstrom("P") == 1.95
    assert van_der_waals_radius_angstrom("Se") == 1.90
    assert van_der_waals_radius_angstrom("ZN") == 2.10
    assert van_der_waals_radius_angstrom("Mg") == 2.20
    assert van_der_waals_radius_angstrom("Ca") == 2.40
    assert van_der_waals_radius_angstrom("Fe") == 2.05
    assert van_der_waals_radius_angstrom("I") == 2.10


def test_covalent_radii_match_rdkit_periodic_table_snapshot() -> None:
    """Supported covalent radii should match the static RDKit PeriodicTable snapshot."""

    assert "GetRcovalent" in COVALENT_RADII_SOURCE
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


def test_unsupported_element_radii_resolve_explicitly_unknown() -> None:
    """Unsupported element symbols should not fall back to carbon-like defaults."""

    vdw_resolution = resolve_element_radius("XX", RadiusKind.VAN_DER_WAALS)
    covalent_resolution = resolve_element_radius("XX", RadiusKind.COVALENT)

    assert vdw_resolution.status is ElementRadiusResolutionStatus.UNKNOWN
    assert vdw_resolution.requested_element_symbol == "XX"
    assert vdw_resolution.normalized_element_symbol == "XX"
    assert vdw_resolution.radius_angstrom is None
    assert covalent_resolution.status is ElementRadiusResolutionStatus.UNKNOWN
    with pytest.raises(UnknownElementRadiusError, match="van_der_waals radius"):
        van_der_waals_radius_angstrom("XX")
    with pytest.raises(UnknownElementRadiusError, match="covalent radius"):
        covalent_radius_angstrom("XX")


def test_invalid_element_radii_resolve_separately_from_unsupported_symbols() -> None:
    """Invalid element strings should stay distinguishable from unsupported symbols."""

    blank_resolution = resolve_element_radius("  ", RadiusKind.VAN_DER_WAALS)
    pseudo_resolution = resolve_element_radius(" C1 ", RadiusKind.VAN_DER_WAALS)

    assert blank_resolution.status is ElementRadiusResolutionStatus.UNKNOWN
    assert blank_resolution.requested_element_symbol == ""
    assert blank_resolution.normalized_element_symbol is None
    assert pseudo_resolution.status is ElementRadiusResolutionStatus.UNKNOWN
    assert pseudo_resolution.requested_element_symbol == "C1"
    assert pseudo_resolution.normalized_element_symbol is None
    assert normalize_radius_element_symbol(" C1 ") is None


def test_deuterium_and_tritium_alias_to_hydrogen_radius() -> None:
    """D/T are isotope aliases of H for element-radius diagnostics."""

    deuterium_resolution = resolve_element_radius("D", RadiusKind.VAN_DER_WAALS)
    tritium_resolution = resolve_element_radius("t", RadiusKind.COVALENT)

    assert deuterium_resolution.status is ElementRadiusResolutionStatus.ALIASED
    assert deuterium_resolution.requested_element_symbol == "D"
    assert deuterium_resolution.normalized_element_symbol == "H"
    assert deuterium_resolution.radius_angstrom == van_der_waals_radius_angstrom("H")
    assert tritium_resolution.status is ElementRadiusResolutionStatus.ALIASED
    assert tritium_resolution.requested_element_symbol == "T"
    assert tritium_resolution.normalized_element_symbol == "H"
    assert tritium_resolution.radius_angstrom == covalent_radius_angstrom("H")


def test_prepared_radius_lookup_records_unresolved_elements_once() -> None:
    """Hot-loop callers should be able to pre-resolve unique element symbols."""

    lookup = prepare_radius_lookup(
        ("C", "c", "D", "XX", "XX", "C1", ""),
        RadiusKind.VAN_DER_WAALS,
    )

    assert lookup.has_unresolved_elements()
    assert lookup.unresolved_element_symbols == ("", "C1", "XX")
    assert lookup.radius_angstrom("C") == van_der_waals_radius_angstrom("C")
    assert lookup.radius_angstrom("c") == van_der_waals_radius_angstrom("C")
    assert lookup.radius_angstrom("D") == van_der_waals_radius_angstrom("H")
    with pytest.raises(UnknownElementRadiusError, match="prepared van_der_waals"):
        lookup.radius_angstrom("XX")
    with pytest.raises(
        UnknownElementRadiusError,
        match="radius policy test has unresolved van_der_waals radius",
    ) as error_info:
        lookup.require_complete("radius policy test")

    error_message = str(error_info.value)
    assert error_message.count("XX") == 1
    assert "C1" in error_message


def test_radius_resolution_value_objects_reject_invariant_drift() -> None:
    """Radius facts should not be constructible in contradictory states."""

    with pytest.raises(ValueError, match="unknown radius resolution"):
        ElementRadiusResolution(
            kind=RadiusKind.VAN_DER_WAALS,
            requested_element_symbol="XX",
            normalized_element_symbol="XX",
            status=ElementRadiusResolutionStatus.UNKNOWN,
            radius_angstrom=1.7,
            source=VAN_DER_WAALS_RADII_SOURCE,
        )

    with pytest.raises(ValueError, match="requires radius and source"):
        ElementRadiusResolution(
            kind=RadiusKind.VAN_DER_WAALS,
            requested_element_symbol="C",
            normalized_element_symbol="C",
            status=ElementRadiusResolutionStatus.KNOWN,
            radius_angstrom=None,
            source=None,
        )

    with pytest.raises(ValueError, match="cannot describe an alias"):
        ElementRadiusResolution(
            kind=RadiusKind.VAN_DER_WAALS,
            requested_element_symbol="D",
            normalized_element_symbol="H",
            status=ElementRadiusResolutionStatus.KNOWN,
            radius_angstrom=1.2,
            source=VAN_DER_WAALS_RADII_SOURCE,
        )

    with pytest.raises(ValueError, match="requires distinct symbols"):
        ElementRadiusResolution(
            kind=RadiusKind.VAN_DER_WAALS,
            requested_element_symbol="H",
            normalized_element_symbol="H",
            status=ElementRadiusResolutionStatus.ALIASED,
            radius_angstrom=1.2,
            source=VAN_DER_WAALS_RADII_SOURCE,
        )

    with pytest.raises(ValueError, match="prepared radius values must be positive"):
        ElementRadiusLookup(
            kind=RadiusKind.VAN_DER_WAALS,
            radius_by_element_symbol={"C": 0.0},
        )

    with pytest.raises(ValueError, match="invalid radius lookup key"):
        ElementRadiusLookup(
            kind=RadiusKind.VAN_DER_WAALS,
            radius_by_element_symbol={"C1": 1.7},
        )

    with pytest.raises(ValueError, match="cannot overlap resolved"):
        ElementRadiusLookup(
            kind=RadiusKind.VAN_DER_WAALS,
            radius_by_element_symbol={"C": 1.7},
            unresolved_element_symbols=(" c ",),
        )


def test_rotatable_hydrogen_scoring_uses_shared_vdw_owner() -> None:
    """Hydrogen scoring should not keep a duplicate vdW table."""

    assert rotatable_hydrogen_vdw_radius_angstrom("S") == van_der_waals_radius_angstrom(
        "S"
    )
