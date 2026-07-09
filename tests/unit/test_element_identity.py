"""Unit tests for source-preserving chemical element identity."""

import pytest

from protrepair.structure.constitution import AtomSite
from protrepair.structure.element import ElementIdentity


@pytest.mark.parametrize(
    ("source_symbol", "chemical_symbol", "isotope_mass_number"),
    (
        ("H", "H", None),
        (" d ", "H", 2),
        ("t", "H", 3),
        ("C", "C", None),
        ("zn", "ZN", None),
    ),
)
def test_element_identity_separates_source_chemical_and_isotope_axes(
    source_symbol: str,
    chemical_symbol: str,
    isotope_mass_number: int | None,
) -> None:
    """Element identity should normalize without erasing isotope provenance."""

    identity = ElementIdentity(source_symbol)

    assert identity.source_symbol == source_symbol.strip().upper()
    assert identity.chemical_symbol == chemical_symbol
    assert identity.isotope_mass_number == isotope_mass_number
    assert identity.is_hydrogen() is (chemical_symbol == "H")
    assert identity.is_isotope_alias() is (isotope_mass_number is not None)


@pytest.mark.parametrize("source_symbol", ("", " "))
def test_element_identity_rejects_blank_source_symbol(source_symbol: str) -> None:
    """Canonical element identity requires a nonblank source symbol."""

    with pytest.raises(ValueError, match="must not be blank"):
        ElementIdentity(source_symbol)


def test_element_identity_rejects_non_string_source_symbol() -> None:
    """Canonical element identity should reject untyped source payloads."""

    with pytest.raises(TypeError, match="string source symbol"):
        ElementIdentity(1)  # type: ignore[arg-type]


def test_element_identity_rejects_unicode_casefold_alias() -> None:
    """Unicode lookalikes must not normalize into valid element symbols."""

    with pytest.raises(ValueError, match="must be ASCII"):
        ElementIdentity("ſ")


@pytest.mark.parametrize(
    ("source_symbol", "expected_mass_number"),
    (("H", None), ("D", 2), ("T", 3)),
)
def test_atom_site_preserves_hydrogen_isotope_identity(
    source_symbol: str,
    expected_mass_number: int | None,
) -> None:
    """Atom sites should cache hydrogen class without rewriting source identity."""

    atom_site = AtomSite(name=" H1 ", element=source_symbol)

    assert atom_site.name == "H1"
    assert atom_site.element == source_symbol
    assert atom_site.element_identity.source_symbol == source_symbol
    assert atom_site.element_identity.chemical_symbol == "H"
    assert atom_site.element_identity.isotope_mass_number == expected_mass_number
    assert atom_site.is_hydrogen()
