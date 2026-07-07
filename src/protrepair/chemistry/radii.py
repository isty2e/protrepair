"""Literature-backed element radii used by chemistry and diagnostics.

The codebase previously carried separate hard-coded radius tables in clash,
geometry, and hydrogen-placement helpers. Those values were already close to
standard literature tables, but the owner and provenance were unclear.

This module centralizes the radius families that ProtRepair currently relies on:

- van der Waals radii: Bondi, J. Phys. Chem. 1964, 68, 441-451
  DOI: 10.1021/j100785a001; Bondi, J. Phys. Chem. 1966, 70, 3006-3007
  DOI: 10.1021/j100881a503; Mantina et al., J. Phys. Chem. A 2009,
  113, 5806-5812 DOI: 10.1021/jp8111556; Batsanov, Inorg. Mater.
  2001, 37, 871-885
- covalent radii: Cordero et al., Dalton Trans. 2008, 2832-2838
  DOI: 10.1039/B801115J

Watchpoint:
- Later literature revisits hydrogen vdW radii, but ProtRepair intentionally keeps
  Bondi's 1.20 A hydrogen value for clash gating to avoid silently loosening
  existing steric thresholds during the 48jm line of work.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

VAN_DER_WAALS_RADII_SOURCE = (
    "Bondi, J. Phys. Chem. 1964, 68, 441-451; DOI: 10.1021/j100785a001; "
    "Bondi, J. Phys. Chem. 1966, 70, 3006-3007; DOI: 10.1021/j100881a503; "
    "Mantina et al., J. Phys. Chem. A 2009, 113, 5806-5812; "
    "DOI: 10.1021/jp8111556; Batsanov, Inorg. Mater. 2001, 37, 871-885"
)

COVALENT_RADII_SOURCE = (
    "Cordero et al., Dalton Trans. 2008, 2832-2838; DOI: 10.1039/B801115J"
)

BONDI_VAN_DER_WAALS_RADII_ANGSTROM: Mapping[str, float] = MappingProxyType(
    {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "MG": 1.73,
        "P": 1.80,
        "S": 1.80,
        "CA": 2.31,
        "FE": 2.00,
        "SE": 1.90,
        "ZN": 1.39,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
    }
)

CORDERO_COVALENT_RADII_ANGSTROM: Mapping[str, float] = MappingProxyType(
    {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "MG": 1.41,
        "P": 1.07,
        "S": 1.05,
        "CA": 1.76,
        "FE": 1.32,
        "SE": 1.20,
        "ZN": 1.22,
        "CL": 1.02,
        "BR": 1.20,
        "I": 1.39,
    }
)

_RADIUS_ELEMENT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "D": "H",
        "T": "H",
    }
)


class RadiusKind(str, Enum):
    """Closed radius family axis used by chemistry diagnostics."""

    VAN_DER_WAALS = "van_der_waals"
    COVALENT = "covalent"


class ElementRadiusResolutionStatus(str, Enum):
    """Resolution status for one element-radius lookup."""

    KNOWN = "known"
    ALIASED = "aliased"
    UNKNOWN = "unknown"


class UnknownElementRadiusError(ValueError):
    """Raised when an element has no radius under the requested radius kind."""


@dataclass(frozen=True, slots=True)
class ElementRadiusResolution:
    """Resolved or unresolved radius lookup for one element symbol."""

    kind: RadiusKind
    requested_element_symbol: str
    normalized_element_symbol: str | None
    status: ElementRadiusResolutionStatus
    radius_angstrom: float | None
    source: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RadiusKind):
            raise TypeError("kind must be a RadiusKind")
        if not isinstance(self.status, ElementRadiusResolutionStatus):
            raise TypeError("status must be an ElementRadiusResolutionStatus")
        if self.radius_angstrom is not None and self.radius_angstrom <= 0.0:
            raise ValueError("radius_angstrom must be positive when present")
        if self.status is ElementRadiusResolutionStatus.UNKNOWN:
            if self.radius_angstrom is not None or self.source is not None:
                raise ValueError("unknown radius resolution cannot carry a radius")
            return

        if self.radius_angstrom is None or self.source is None:
            raise ValueError("known radius resolution requires radius and source")
        if not self.source.strip():
            raise ValueError("known radius resolution source must be non-empty")
        if self.normalized_element_symbol is None:
            raise ValueError("known radius resolution requires a normalized symbol")
        if self.status is ElementRadiusResolutionStatus.KNOWN:
            if self.requested_element_symbol != self.normalized_element_symbol:
                raise ValueError("known radius resolution cannot describe an alias")
        elif self.requested_element_symbol == self.normalized_element_symbol:
            raise ValueError("aliased radius resolution requires distinct symbols")

    def is_known(self) -> bool:
        """Return whether this lookup resolved to a concrete radius."""

        return self.status is not ElementRadiusResolutionStatus.UNKNOWN

    def require_radius(self) -> float:
        """Return the concrete radius or raise with stable unknown-element wording."""

        if self.radius_angstrom is None:
            raise UnknownElementRadiusError(
                f"no {self.kind.value} radius for element "
                f"{self.requested_element_symbol!r}"
            )

        return self.radius_angstrom


@dataclass(frozen=True, slots=True)
class ElementRadiusLookup:
    """Prepared radius lookup for a fixed element set and one radius kind."""

    kind: RadiusKind
    radius_by_element_symbol: Mapping[str, float]
    unresolved_element_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RadiusKind):
            raise TypeError("kind must be a RadiusKind")
        for element_symbol, radius_angstrom in self.radius_by_element_symbol.items():
            if not _valid_radius_lookup_key(element_symbol):
                raise ValueError(f"invalid radius lookup key {element_symbol!r}")
            if radius_angstrom <= 0.0:
                raise ValueError("prepared radius values must be positive")
        object.__setattr__(
            self,
            "radius_by_element_symbol",
            MappingProxyType(dict(self.radius_by_element_symbol)),
        )
        normalized_unresolved_element_symbols = tuple(
            sorted(
                {
                    _radius_lookup_key(symbol)
                    for symbol in self.unresolved_element_symbols
                }
            )
        )
        if any(
            element_symbol in self.radius_by_element_symbol
            for element_symbol in normalized_unresolved_element_symbols
        ):
            raise ValueError("unresolved radius keys cannot overlap resolved keys")
        object.__setattr__(
            self,
            "unresolved_element_symbols",
            normalized_unresolved_element_symbols,
        )

    def has_unresolved_elements(self) -> bool:
        """Return whether at least one requested element failed to resolve."""

        return bool(self.unresolved_element_symbols)

    def radius_angstrom(self, element: str) -> float:
        """Return one prepared radius or raise for an unresolved element."""

        lookup_key = _radius_lookup_key(element)
        try:
            return self.radius_by_element_symbol[lookup_key]
        except KeyError as error:
            raise UnknownElementRadiusError(
                f"no prepared {self.kind.value} radius for element {lookup_key!r}"
            ) from error


def van_der_waals_radius_angstrom(element: str) -> float:
    """Return the vdW radius for one element, or raise when unsupported."""

    return element_radius_angstrom(element, RadiusKind.VAN_DER_WAALS)


def covalent_radius_angstrom(element: str) -> float:
    """Return the Cordero covalent radius for one element, or raise when unsupported."""

    return element_radius_angstrom(element, RadiusKind.COVALENT)


def element_radius_angstrom(element: str, kind: RadiusKind) -> float:
    """Return one concrete radius or raise when the element is unsupported."""

    return resolve_element_radius(element, kind).require_radius()


def resolve_element_radius(element: str, kind: RadiusKind) -> ElementRadiusResolution:
    """Return explicit radius resolution for one element symbol."""

    if not isinstance(kind, RadiusKind):
        raise TypeError("kind must be a RadiusKind")

    requested_element_symbol = _radius_lookup_key(element)
    normalized_element_symbol = normalize_radius_element_symbol(element)
    if normalized_element_symbol is None:
        return ElementRadiusResolution(
            kind=kind,
            requested_element_symbol=requested_element_symbol,
            normalized_element_symbol=None,
            status=ElementRadiusResolutionStatus.UNKNOWN,
            radius_angstrom=None,
            source=None,
        )

    radius_angstrom = _radius_table(kind).get(normalized_element_symbol)
    if radius_angstrom is None:
        return ElementRadiusResolution(
            kind=kind,
            requested_element_symbol=requested_element_symbol,
            normalized_element_symbol=normalized_element_symbol,
            status=ElementRadiusResolutionStatus.UNKNOWN,
            radius_angstrom=None,
            source=None,
        )

    return ElementRadiusResolution(
        kind=kind,
        requested_element_symbol=requested_element_symbol,
        normalized_element_symbol=normalized_element_symbol,
        status=(
            ElementRadiusResolutionStatus.ALIASED
            if requested_element_symbol != normalized_element_symbol
            else ElementRadiusResolutionStatus.KNOWN
        ),
        radius_angstrom=radius_angstrom,
        source=_radius_source(kind),
    )


def prepare_radius_lookup(
    elements: Iterable[str],
    kind: RadiusKind,
) -> ElementRadiusLookup:
    """Return a prepared radius lookup for unique element symbols."""

    radius_by_element_symbol: dict[str, float] = {}
    unresolved_element_symbols: set[str] = set()
    for element in elements:
        lookup_key = _radius_lookup_key(element)
        if (
            lookup_key in radius_by_element_symbol
            or lookup_key in unresolved_element_symbols
        ):
            continue

        resolution = resolve_element_radius(element, kind)
        if resolution.radius_angstrom is None:
            unresolved_element_symbols.add(lookup_key)
            continue

        radius_by_element_symbol[lookup_key] = resolution.radius_angstrom

    return ElementRadiusLookup(
        kind=kind,
        radius_by_element_symbol=radius_by_element_symbol,
        unresolved_element_symbols=tuple(unresolved_element_symbols),
    )


def normalize_radius_element_symbol(element: str) -> str | None:
    """Return the canonical radius-table symbol for one raw element string."""

    lookup_key = _radius_lookup_key(element)
    if lookup_key in _RADIUS_ELEMENT_ALIASES:
        return _RADIUS_ELEMENT_ALIASES[lookup_key]
    if not _valid_radius_lookup_key(lookup_key):
        return None

    return lookup_key


def _radius_lookup_key(element: str) -> str:
    """Return the stable lookup key for one raw element string."""

    return element.strip().upper()


def _valid_radius_lookup_key(element_symbol: str) -> bool:
    """Return whether one normalized symbol has element-symbol shape."""

    return 1 <= len(element_symbol) <= 2 and all(
        "A" <= character <= "Z" for character in element_symbol
    )


def _radius_table(kind: RadiusKind) -> Mapping[str, float]:
    """Return the active radius table for one radius kind."""

    if kind is RadiusKind.VAN_DER_WAALS:
        return BONDI_VAN_DER_WAALS_RADII_ANGSTROM
    if kind is RadiusKind.COVALENT:
        return CORDERO_COVALENT_RADII_ANGSTROM

    raise TypeError("kind must be a RadiusKind")


def _radius_source(kind: RadiusKind) -> str:
    """Return the provenance text for one radius kind."""

    if kind is RadiusKind.VAN_DER_WAALS:
        return VAN_DER_WAALS_RADII_SOURCE
    if kind is RadiusKind.COVALENT:
        return COVALENT_RADII_SOURCE

    raise TypeError("kind must be a RadiusKind")
