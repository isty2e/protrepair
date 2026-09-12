"""Histidine protonation request and assignment contracts."""

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias, TypeGuard

from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ResidueIndex

DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO = 0.2


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

_HISTIDINE_PROTONATION_REQUEST_TYPES = (
    DisabledHistidineProtonationRequest,
    PrasRatioHistidineProtonationRequest,
)


@dataclass(frozen=True, slots=True)
class HistidineDeltaProtonationAssignment:
    """Identify a chain-local PRAS target for protonating both ring nitrogens.

    Parameters
    ----------
    residue_index : ResidueIndex
        Histidine slot in the chain used to resolve the ratio request.

    Raises
    ------
    TypeError
        The index is not a canonical residue slot.
    """

    residue_index: ResidueIndex

    def __post_init__(self) -> None:
        if not isinstance(self.residue_index, ResidueIndex):
            raise TypeError(
                "histidine protonation assignment residue_index must be a ResidueIndex"
            )

    def microstate_constraints(self) -> MicrostateConstraints:
        """Request the complete cationic imidazole protonation state.

        Returns
        -------
        MicrostateConstraints
            One H on each ring nitrogen. The resolver chooses compatible charge
            localization and Kekule bonds rather than independently appending HD1.
        """
        return MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1)))


def normalize_histidine_protonation_request(
    histidine_protonation: HistidineProtonationRequest | None = None,
    *,
    protonate_histidines: bool = False,
) -> HistidineProtonationRequest:
    """Normalize boundary histidine protonation inputs into a closed request value."""

    if not isinstance(protonate_histidines, bool):
        raise TypeError("protonate_histidines must be a bool")
    if histidine_protonation is None:
        if protonate_histidines:
            return PrasRatioHistidineProtonationRequest()

        return DisabledHistidineProtonationRequest()
    if not _is_histidine_protonation_request(histidine_protonation):
        raise TypeError(
            "histidine_protonation must be a HistidineProtonationRequest value"
        )
    if protonate_histidines:
        raise ValueError(
            "histidine_protonation and protonate_histidines must not both be set"
        )

    return histidine_protonation


def resolve_histidine_protonation_assignments(
    chain: ChainSite,
    request: HistidineProtonationRequest,
) -> tuple[HistidineDeltaProtonationAssignment, ...]:
    """Resolve one chain-local request into deterministic delta-protonation targets."""

    if not isinstance(chain, ChainSite):
        raise TypeError("histidine protonation resolution requires a ChainSite")
    if not _is_histidine_protonation_request(request):
        raise TypeError("histidine protonation request must use a supported variant")
    if isinstance(request, DisabledHistidineProtonationRequest):
        return ()

    histidine_indices = tuple(
        ResidueIndex(residue_offset)
        for residue_offset, residue in enumerate(chain.residues)
        if residue.component_id == "HIS"
    )
    selected_count = math.floor(
        Decimal(len(histidine_indices)) * Decimal(str(request.ratio))
    )
    return tuple(
        HistidineDeltaProtonationAssignment(residue_index)
        for residue_index in histidine_indices[:selected_count]
    )


def _is_histidine_protonation_request(
    value: object,
) -> TypeGuard[HistidineProtonationRequest]:
    """Return whether value belongs to the closed request variant set."""

    return type(value) in _HISTIDINE_PROTONATION_REQUEST_TYPES


def histidine_microstate_requests(
    structure: ProteinStructure,
    request: HistidineProtonationRequest,
) -> dict[tuple[ResidueId, PolymerChemicalSite], MicrostateConstraints | None]:
    """Translate a ratio request into complete ring choices and explicit resets.

    Parameters
    ----------
    structure : ProteinStructure
        Polymer chains defining deterministic ratio membership.
    request : HistidineProtonationRequest
        Disabled means no new choice. A ratio replaces the previous ratio;
        unselected HIS return to original evidence and preparation preferences.

    Returns
    -------
    dict
        Site-keyed requests for PolymerMicrostatePreparation. Selected HIS request
        both ring protons; None resets saved choices without overriding source H.

    Raises
    ------
    TypeError
        The request is not a supported canonical variant.
    """
    if not _is_histidine_protonation_request(request):
        raise TypeError("histidine protonation request must use a supported variant")
    if isinstance(request, DisabledHistidineProtonationRequest):
        return {}
    requests = {}
    for chain in structure.constitution.chains:
        selected = {
            assignment.residue_index.value: assignment.microstate_constraints()
            for assignment in resolve_histidine_protonation_assignments(chain, request)
        }
        for index, residue in enumerate(chain.residues):
            if residue.component_id == "HIS":
                requests[(residue.residue_id, PolymerChemicalSite.SIDECHAIN)] = (
                    selected.get(index)
                )
    return requests


__all__ = [
    "DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO",
    "DisabledHistidineProtonationRequest",
    "HistidineDeltaProtonationAssignment",
    "HistidineProtonationRequest",
    "PrasRatioHistidineProtonationRequest",
    "normalize_histidine_protonation_request",
    "resolve_histidine_protonation_assignments",
]
