"""Canonical chemical element identity for structure atom sites."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class ElementIdentity:
    """Source-preserving element identity with explicit isotope semantics."""

    source_symbol: str
    chemical_symbol: str
    isotope_mass_number: int | None

    def __init__(self, source_symbol: str) -> None:
        if not isinstance(source_symbol, str):
            raise TypeError("element identity requires a string source symbol")

        stripped_source_symbol = source_symbol.strip()
        if not stripped_source_symbol:
            raise ValueError("element identity source symbol must not be blank")
        if not stripped_source_symbol.isascii():
            raise ValueError("element identity source symbol must be ASCII")

        normalized_source_symbol = stripped_source_symbol.upper()

        chemical_symbol = normalized_source_symbol
        isotope_mass_number: int | None = None
        if normalized_source_symbol == "D":
            chemical_symbol = "H"
            isotope_mass_number = 2
        elif normalized_source_symbol == "T":
            chemical_symbol = "H"
            isotope_mass_number = 3

        object.__setattr__(self, "source_symbol", normalized_source_symbol)
        object.__setattr__(self, "chemical_symbol", chemical_symbol)
        object.__setattr__(self, "isotope_mass_number", isotope_mass_number)

    def is_hydrogen(self) -> bool:
        """Return whether this identity has hydrogen chemical behavior."""

        return self.chemical_symbol == "H"

    def is_isotope_alias(self) -> bool:
        """Return whether source spelling encodes an explicit isotope alias."""

        return self.isotope_mass_number is not None
