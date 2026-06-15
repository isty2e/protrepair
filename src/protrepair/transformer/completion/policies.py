"""Completion transformer execution policy contracts."""

from enum import Enum

__all__ = ["OrphanFragmentPolicy"]


class OrphanFragmentPolicy(str, Enum):
    """Policy for disconnected template fragments during atom completion."""

    REBUILD = "rebuild"
    PRESERVE = "preserve"
    SALVAGE_WHEN_SAFE = "salvage_when_safe"
