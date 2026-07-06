"""Histidine protonation request and assignment contracts."""

import math
from dataclasses import dataclass
from typing import TypeAlias

from protrepair.structure.slots import ResidueIndex

DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO = 0.2

__all__ = [
    "DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO",
    "DisabledHistidineProtonationRequest",
    "HistidineDeltaProtonationAssignment",
    "HistidineProtonationRequest",
    "PrasRatioHistidineProtonationRequest",
]


@dataclass(frozen=True, slots=True)
class DisabledHistidineProtonationRequest:
    """Request no additional histidine protonation."""


@dataclass(frozen=True, slots=True)
class PrasRatioHistidineProtonationRequest:
    """Request PRAS-style deterministic histidine delta protonation."""

    ratio: float = DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO

    def __post_init__(self) -> None:
        ratio = self.ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("histidine protonation ratio must be an int or float")
        if not math.isfinite(float(ratio)):
            raise ValueError("histidine protonation ratio must be finite")
        if not 0.0 <= float(ratio) <= 1.0:
            raise ValueError("histidine protonation ratio must be in [0.0, 1.0]")

        object.__setattr__(self, "ratio", float(ratio))


HistidineProtonationRequest: TypeAlias = (
    DisabledHistidineProtonationRequest | PrasRatioHistidineProtonationRequest
)
"""Boundary-level request variants for histidine protonation resolution."""


@dataclass(frozen=True, slots=True)
class HistidineDeltaProtonationAssignment:
    """Resolved assignment to append the histidine delta HD1 atom."""

    residue_index: ResidueIndex

    def __post_init__(self) -> None:
        if not isinstance(self.residue_index, ResidueIndex):
            raise TypeError(
                "histidine protonation assignment residue_index must be a ResidueIndex"
            )
