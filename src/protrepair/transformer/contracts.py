"""Canonical structure-transformer contracts."""

from protrepair.transformer.atom_input import (
    AtomInput,
    AtomInputBasis,
    AtomInputRealization,
)
from protrepair.transformer.base import (
    BranchingContextOperation,
    ContextOperation,
    DeterministicContextOperation,
    ProjectedCodomainState,
    ProjectedDomainState,
    ProteinTransformer,
    TransformationCandidate,
)
from protrepair.transformer.context import (
    ProteinTransformationContext,
    SupportingStructureContext,
)
from protrepair.transformer.result import TransformationResult

__all__ = [
    "AtomInput",
    "AtomInputBasis",
    "AtomInputRealization",
    "BranchingContextOperation",
    "ContextOperation",
    "DeterministicContextOperation",
    "ProjectedCodomainState",
    "ProjectedDomainState",
    "ProteinTransformationContext",
    "ProteinTransformer",
    "SupportingStructureContext",
    "TransformationCandidate",
    "TransformationResult",
]
