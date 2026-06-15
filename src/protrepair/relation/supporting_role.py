"""Semantic roles for supporting structures at workflow and relation boundaries."""

from enum import Enum


class SupportingStructureRole(str, Enum):
    """Closed roles for additional structures in one transformation context."""

    REFERENCE = "reference"
    DONOR = "donor"
    TEMPLATE = "template"
    EVIDENCE = "evidence"
