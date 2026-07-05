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

from collections.abc import Mapping
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

DEFAULT_VAN_DER_WAALS_RADIUS_ANGSTROM = 1.70

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

DEFAULT_COVALENT_RADIUS_ANGSTROM = 0.77


def van_der_waals_radius_angstrom(element: str) -> float:
    """Return the literature-backed van der Waals radius for one element symbol."""

    return BONDI_VAN_DER_WAALS_RADII_ANGSTROM.get(
        element.strip().upper(),
        DEFAULT_VAN_DER_WAALS_RADIUS_ANGSTROM,
    )


def covalent_radius_angstrom(element: str) -> float:
    """Return the Cordero single-bond covalent radius for one element symbol."""

    return CORDERO_COVALENT_RADII_ANGSTROM.get(
        element.strip().upper(),
        DEFAULT_COVALENT_RADIUS_ANGSTROM,
    )
