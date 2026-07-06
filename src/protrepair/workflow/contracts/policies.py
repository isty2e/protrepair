"""Workflow and processing policy enums."""

from enum import Enum

from protrepair.io.ingress_policy import MutationPolicy, OccupancyPolicy
from protrepair.transformer.completion.policies import OrphanFragmentPolicy

__all__ = [
    "LigandPolicy",
    "MutationPolicy",
    "OccupancyPolicy",
    "OrphanFragmentPolicy",
]


class LigandPolicy(str, Enum):
    """Policy for retaining or dropping ligands."""

    DROP = "drop"
    KEEP = "keep"
