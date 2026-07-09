"""Static RDKit PeriodicTable element radii used by diagnostics.

The lookup path intentionally does not import RDKit at runtime. Radius values are
a committed snapshot of ``rdkit.Chem.GetPeriodicTable().GetRvdw`` and
``GetRcovalent`` for atomic numbers 1 through 118, generated from the release
constraint version recorded below.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType

RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_VERSION = "2026.03.2"
RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_SOURCE = (
    "RDKit PeriodicTable GetRvdw/GetRcovalent snapshot generated from "
    "rdkit==2026.3.2 (rdBase.rdkitVersion=2026.03.2) for atomic numbers 1-118"
)
VAN_DER_WAALS_RADII_SOURCE = (
    f"{RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_SOURCE}; radius API: GetRvdw; "
    "upstream source: BODR v10.1"
)
COVALENT_RADII_SOURCE = (
    f"{RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_SOURCE}; radius API: GetRcovalent; "
    "upstream source: Cordero et al. DOI 10.1039/B801115J"
)

_RDKIT_VAN_DER_WAALS_SOURCE_DEFAULT_ELEMENT_SYMBOLS = frozenset(
    {"DS", "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG"}
)
_RDKIT_COVALENT_SOURCE_DEFAULT_ELEMENT_SYMBOLS = frozenset(
    {
        "BK",
        "CF",
        "ES",
        "FM",
        "MD",
        "NO",
        "LR",
        "RF",
        "DB",
        "SG",
        "BH",
        "HS",
        "MT",
        "DS",
        "RG",
        "CN",
    }
)

RDKIT_PERIODIC_TABLE_VAN_DER_WAALS_RADII_ANGSTROM: Mapping[str, float] = (
    MappingProxyType(
        {
            "H": 1.2,
            "HE": 1.4,
            "LI": 2.2,
            "BE": 1.9,
            "B": 1.8,
            "C": 1.7,
            "N": 1.6,
            "O": 1.55,
            "F": 1.5,
            "NE": 1.54,
            "NA": 2.4,
            "MG": 2.2,
            "AL": 2.1,
            "SI": 2.1,
            "P": 1.95,
            "S": 1.8,
            "CL": 1.8,
            "AR": 1.88,
            "K": 2.8,
            "CA": 2.4,
            "SC": 2.3,
            "TI": 2.15,
            "V": 2.05,
            "CR": 2.05,
            "MN": 2.05,
            "FE": 2.05,
            "CO": 2.0,
            "NI": 2.0,
            "CU": 2.0,
            "ZN": 2.1,
            "GA": 2.1,
            "GE": 2.1,
            "AS": 2.05,
            "SE": 1.9,
            "BR": 1.9,
            "KR": 2.02,
            "RB": 2.9,
            "SR": 2.55,
            "Y": 2.4,
            "ZR": 2.3,
            "NB": 2.15,
            "MO": 2.1,
            "TC": 2.05,
            "RU": 2.05,
            "RH": 2.0,
            "PD": 2.05,
            "AG": 2.1,
            "CD": 2.2,
            "IN": 2.2,
            "SN": 2.25,
            "SB": 2.2,
            "TE": 2.1,
            "I": 2.1,
            "XE": 2.16,
            "CS": 3.0,
            "BA": 2.7,
            "LA": 2.5,
            "CE": 2.48,
            "PR": 2.47,
            "ND": 2.45,
            "PM": 2.43,
            "SM": 2.42,
            "EU": 2.4,
            "GD": 2.38,
            "TB": 2.37,
            "DY": 2.35,
            "HO": 2.33,
            "ER": 2.32,
            "TM": 2.3,
            "YB": 2.28,
            "LU": 2.27,
            "HF": 2.25,
            "TA": 2.2,
            "W": 2.1,
            "RE": 2.05,
            "OS": 2.0,
            "IR": 2.0,
            "PT": 2.05,
            "AU": 2.1,
            "HG": 2.05,
            "TL": 2.2,
            "PB": 2.3,
            "BI": 2.3,
            "PO": 2.0,
            "AT": 2.0,
            "RN": 2.0,
            "FR": 2.0,
            "RA": 2.0,
            "AC": 2.0,
            "TH": 2.4,
            "PA": 2.0,
            "U": 2.3,
            "NP": 2.0,
            "PU": 2.0,
            "AM": 2.0,
            "CM": 2.0,
            "BK": 2.0,
            "CF": 2.0,
            "ES": 2.0,
            "FM": 2.0,
            "MD": 2.0,
            "NO": 2.0,
            "LR": 2.0,
            "RF": 2.0,
            "DB": 2.0,
            "SG": 2.0,
            "BH": 2.0,
            "HS": 2.0,
            "MT": 2.0,
            "DS": 2.0,
            "RG": 2.0,
            "CN": 2.0,
            "NH": 2.0,
            "FL": 2.0,
            "MC": 2.0,
            "LV": 2.0,
            "TS": 2.0,
            "OG": 2.0,
        }
    )
)

RDKIT_PERIODIC_TABLE_COVALENT_RADII_ANGSTROM: Mapping[str, float] = MappingProxyType(
    {
        "H": 0.31,
        "HE": 0.28,
        "LI": 1.28,
        "BE": 0.96,
        "B": 0.84,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "NE": 0.58,
        "NA": 1.66,
        "MG": 1.41,
        "AL": 1.21,
        "SI": 1.11,
        "P": 1.07,
        "S": 1.05,
        "CL": 1.02,
        "AR": 1.06,
        "K": 2.03,
        "CA": 1.76,
        "SC": 1.7,
        "TI": 1.6,
        "V": 1.52,
        "CR": 1.39,
        "MN": 1.39,
        "FE": 1.32,
        "CO": 1.26,
        "NI": 1.24,
        "CU": 1.32,
        "ZN": 1.22,
        "GA": 1.22,
        "GE": 1.2,
        "AS": 1.19,
        "SE": 1.2,
        "BR": 1.2,
        "KR": 1.16,
        "RB": 2.2,
        "SR": 1.95,
        "Y": 1.9,
        "ZR": 1.75,
        "NB": 1.64,
        "MO": 1.54,
        "TC": 1.47,
        "RU": 1.46,
        "RH": 1.42,
        "PD": 1.39,
        "AG": 1.45,
        "CD": 1.44,
        "IN": 1.42,
        "SN": 1.39,
        "SB": 1.39,
        "TE": 1.38,
        "I": 1.39,
        "XE": 1.4,
        "CS": 2.44,
        "BA": 2.15,
        "LA": 2.07,
        "CE": 2.04,
        "PR": 2.03,
        "ND": 2.01,
        "PM": 1.99,
        "SM": 1.98,
        "EU": 1.98,
        "GD": 1.96,
        "TB": 1.94,
        "DY": 1.92,
        "HO": 1.92,
        "ER": 1.89,
        "TM": 1.9,
        "YB": 1.87,
        "LU": 1.87,
        "HF": 1.75,
        "TA": 1.7,
        "W": 1.62,
        "RE": 1.51,
        "OS": 1.44,
        "IR": 1.41,
        "PT": 1.36,
        "AU": 1.36,
        "HG": 1.32,
        "TL": 1.45,
        "PB": 1.46,
        "BI": 1.48,
        "PO": 1.4,
        "AT": 1.5,
        "RN": 1.5,
        "FR": 2.6,
        "RA": 2.2,
        "AC": 2.15,
        "TH": 2.06,
        "PA": 2.0,
        "U": 1.96,
        "NP": 1.9,
        "PU": 1.87,
        "AM": 1.8,
        "CM": 1.69,
        "BK": 1.9,
        "CF": 1.9,
        "ES": 1.9,
        "FM": 1.9,
        "MD": 1.9,
        "NO": 1.9,
        "LR": 1.9,
        "RF": 1.9,
        "DB": 1.9,
        "SG": 1.9,
        "BH": 1.9,
        "HS": 1.9,
        "MT": 1.9,
        "DS": 1.9,
        "RG": 1.9,
        "CN": 1.9,
        "NH": 1.36,
        "FL": 1.43,
        "MC": 1.62,
        "LV": 1.75,
        "TS": 1.65,
        "OG": 1.57,
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
    """Availability status for one element-radius lookup."""

    RESOLVED = "resolved"
    UNKNOWN = "unknown"


class ElementRadiusDataQuality(str, Enum):
    """Upstream source quality for one resolved radius value."""

    SOURCE_REPORTED = "source_reported"
    SOURCE_DEFAULT = "source_default"


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
    data_quality: ElementRadiusDataQuality | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RadiusKind):
            raise TypeError("kind must be a RadiusKind")
        if not isinstance(self.status, ElementRadiusResolutionStatus):
            raise TypeError("status must be an ElementRadiusResolutionStatus")
        if self.data_quality is not None and not isinstance(
            self.data_quality,
            ElementRadiusDataQuality,
        ):
            raise TypeError("data_quality must be an ElementRadiusDataQuality")
        if self.radius_angstrom is not None and (
            not isfinite(self.radius_angstrom) or self.radius_angstrom <= 0.0
        ):
            raise ValueError("radius_angstrom must be finite and positive when present")

        requested_element_symbol = _radius_lookup_key(
            self.requested_element_symbol
        )
        if requested_element_symbol != self.requested_element_symbol:
            raise ValueError("requested element symbol must be a canonical lookup key")
        expected_normalized_element_symbol = normalize_radius_element_symbol(
            requested_element_symbol
        )
        if self.normalized_element_symbol != expected_normalized_element_symbol:
            raise ValueError(
                "normalized element symbol must match the canonical radius resolver"
            )

        if self.status is ElementRadiusResolutionStatus.UNKNOWN:
            if (
                self.radius_angstrom is not None
                or self.source is not None
                or self.data_quality is not None
            ):
                raise ValueError(
                    "unknown radius resolution cannot carry radius source or quality"
                )
            if (
                expected_normalized_element_symbol is not None
                and expected_normalized_element_symbol in _radius_table(self.kind)
            ):
                raise ValueError(
                    "unknown radius resolution cannot describe a known table element"
                )
            return

        if (
            self.radius_angstrom is None
            or self.source is None
            or self.data_quality is None
        ):
            raise ValueError(
                "resolved radius resolution requires radius source and source quality"
            )
        if not self.source.strip():
            raise ValueError("resolved radius resolution source must be non-empty")
        if self.normalized_element_symbol is None:
            raise ValueError("resolved radius resolution requires a normalized symbol")
        if self.source != _radius_source(self.kind):
            raise ValueError(
                "resolved radius resolution must use the canonical radius source"
            )
        expected_radius_angstrom = _radius_table(self.kind).get(
            self.normalized_element_symbol
        )
        if self.radius_angstrom != expected_radius_angstrom:
            raise ValueError(
                "resolved radius resolution must match the active radius table"
            )
        expected_data_quality = _radius_data_quality(
            self.kind,
            self.normalized_element_symbol,
        )
        if self.data_quality is not expected_data_quality:
            raise ValueError(
                "resolved radius resolution must match canonical source quality"
            )

    def is_resolved(self) -> bool:
        """Return whether this lookup resolved to a concrete radius."""

        return self.status is ElementRadiusResolutionStatus.RESOLVED

    def is_alias(self) -> bool:
        """Return whether the requested symbol resolved through an alias."""

        return (
            self.is_resolved()
            and self.requested_element_symbol != self.normalized_element_symbol
        )

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
    source_default_element_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RadiusKind):
            raise TypeError("kind must be a RadiusKind")
        for element_symbol, radius_angstrom in self.radius_by_element_symbol.items():
            if not _valid_radius_lookup_key(element_symbol):
                raise ValueError(f"invalid radius lookup key {element_symbol!r}")
            if not isfinite(radius_angstrom) or radius_angstrom <= 0.0:
                raise ValueError("prepared radius values must be finite and positive")
            normalized_element_symbol = normalize_radius_element_symbol(element_symbol)
            expected_radius_angstrom = (
                None
                if normalized_element_symbol is None
                else _radius_table(self.kind).get(normalized_element_symbol)
            )
            if radius_angstrom != expected_radius_angstrom:
                raise ValueError(
                    "prepared radius values must match the active radius table"
                )
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
        normalized_source_default_element_symbols = tuple(
            sorted(
                {
                    _radius_lookup_key(symbol)
                    for symbol in self.source_default_element_symbols
                }
            )
        )
        expected_source_default_element_symbols = tuple(
            sorted(
                element_symbol
                for element_symbol in self.radius_by_element_symbol
                if _radius_data_quality(
                    self.kind,
                    normalize_radius_element_symbol(element_symbol),
                )
                is ElementRadiusDataQuality.SOURCE_DEFAULT
            )
        )
        if (
            normalized_source_default_element_symbols
            != expected_source_default_element_symbols
        ):
            raise ValueError(
                "prepared source-default radius keys must match canonical source "
                "quality"
            )
        object.__setattr__(
            self,
            "source_default_element_symbols",
            normalized_source_default_element_symbols,
        )

    def has_unresolved_elements(self) -> bool:
        """Return whether at least one requested element failed to resolve."""

        return bool(self.unresolved_element_symbols)

    def has_source_defaults(self) -> bool:
        """Return whether any prepared radius is an upstream source default."""

        return bool(self.source_default_element_symbols)

    def require_complete(self, context: str) -> None:
        """Raise one aggregate error if any requested element lacks a radius."""

        if self.has_unresolved_elements():
            raise self.unresolved_radius_error(context)

    def require_kind(self, expected_kind: RadiusKind, context: str) -> None:
        """Raise when this lookup cannot serve a radius-family consumer."""

        if not isinstance(expected_kind, RadiusKind):
            raise TypeError("expected_kind must be a RadiusKind")
        if self.kind is not expected_kind:
            raise ValueError(
                f"{context} requires {expected_kind.value} radius lookup; "
                f"got {self.kind.value}"
            )

    def unresolved_radius_error(self, context: str) -> UnknownElementRadiusError:
        """Return an aggregate unresolved-radius error for this prepared lookup."""

        unresolved = ", ".join(
            repr(element_symbol) for element_symbol in self.unresolved_element_symbols
        )
        return UnknownElementRadiusError(
            f"{context} has unresolved {self.kind.value} radius for element(s): "
            f"{unresolved}"
        )

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
    """Return the RDKit covalent radius for one element, or raise when unsupported."""

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
            data_quality=None,
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
            data_quality=None,
        )

    return ElementRadiusResolution(
        kind=kind,
        requested_element_symbol=requested_element_symbol,
        normalized_element_symbol=normalized_element_symbol,
        status=ElementRadiusResolutionStatus.RESOLVED,
        radius_angstrom=radius_angstrom,
        source=_radius_source(kind),
        data_quality=_radius_data_quality(kind, normalized_element_symbol),
    )


def prepare_radius_lookup(
    elements: Iterable[str],
    kind: RadiusKind,
) -> ElementRadiusLookup:
    """Return a prepared radius lookup for unique element symbols."""

    radius_by_element_symbol: dict[str, float] = {}
    unresolved_element_symbols: set[str] = set()
    source_default_element_symbols: set[str] = set()
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
        if resolution.data_quality is ElementRadiusDataQuality.SOURCE_DEFAULT:
            source_default_element_symbols.add(lookup_key)

    return ElementRadiusLookup(
        kind=kind,
        radius_by_element_symbol=radius_by_element_symbol,
        unresolved_element_symbols=tuple(unresolved_element_symbols),
        source_default_element_symbols=tuple(source_default_element_symbols),
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

    stripped_element = element.strip()
    if not stripped_element.isascii():
        return stripped_element

    return stripped_element.upper()


def _valid_radius_lookup_key(element_symbol: str) -> bool:
    """Return whether one normalized symbol has element-symbol shape."""

    return 1 <= len(element_symbol) <= 2 and all(
        "A" <= character <= "Z" for character in element_symbol
    )


def _radius_table(kind: RadiusKind) -> Mapping[str, float]:
    """Return the active radius table for one radius kind."""

    if kind is RadiusKind.VAN_DER_WAALS:
        return RDKIT_PERIODIC_TABLE_VAN_DER_WAALS_RADII_ANGSTROM
    if kind is RadiusKind.COVALENT:
        return RDKIT_PERIODIC_TABLE_COVALENT_RADII_ANGSTROM

    raise TypeError("kind must be a RadiusKind")


def _radius_source(kind: RadiusKind) -> str:
    """Return the provenance text for one radius kind."""

    if kind is RadiusKind.VAN_DER_WAALS:
        return VAN_DER_WAALS_RADII_SOURCE
    if kind is RadiusKind.COVALENT:
        return COVALENT_RADII_SOURCE

    raise TypeError("kind must be a RadiusKind")


def _radius_data_quality(
    kind: RadiusKind,
    normalized_element_symbol: str | None,
) -> ElementRadiusDataQuality:
    """Return pinned upstream source quality for one resolved radius value."""

    if normalized_element_symbol is None:
        raise ValueError("resolved radius quality requires a normalized symbol")
    source_default_symbols = (
        _RDKIT_VAN_DER_WAALS_SOURCE_DEFAULT_ELEMENT_SYMBOLS
        if kind is RadiusKind.VAN_DER_WAALS
        else _RDKIT_COVALENT_SOURCE_DEFAULT_ELEMENT_SYMBOLS
        if kind is RadiusKind.COVALENT
        else None
    )
    if source_default_symbols is None:
        raise TypeError("kind must be a RadiusKind")
    if normalized_element_symbol in source_default_symbols:
        return ElementRadiusDataQuality.SOURCE_DEFAULT

    return ElementRadiusDataQuality.SOURCE_REPORTED
