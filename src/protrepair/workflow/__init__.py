"""Workflow processing entrypoints."""

from protrepair.workflow.engine import (
    normalize_source_structure,
    process_canonical_structure,
    process_structure_source,
)

__all__ = [
    "normalize_source_structure",
    "process_canonical_structure",
    "process_structure_source",
]
