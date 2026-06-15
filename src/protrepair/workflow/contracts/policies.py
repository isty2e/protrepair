"""Workflow and processing policy enums."""

from enum import Enum

from protrepair.io.ingress_policy import MutationPolicy, OccupancyPolicy
from protrepair.transformer.completion.policies import OrphanFragmentPolicy

__all__ = [
    "CTerminalOxtPolicy",
    "HydrogenPolicy",
    "LigandPolicy",
    "MutationPolicy",
    "OccupancyPolicy",
    "OrphanFragmentPolicy",
]


class LigandPolicy(str, Enum):
    """Policy for retaining or dropping ligands."""

    DROP = "drop"
    KEEP = "keep"


class HydrogenPolicy(str, Enum):
    """Policy for hydrogen handling during processing."""

    PRESERVE = "preserve"
    ADD_MISSING = "add_missing"


class CTerminalOxtPolicy(str, Enum):
    """Policy for C-terminal OXT augmentation during workflow processing."""

    PRESERVE = "preserve"
    AUGMENT_AUTHENTIC_SOURCE_BOUNDARIES = (
        "augment_authentic_source_boundaries"
    )
