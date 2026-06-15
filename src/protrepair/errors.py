"""Project-specific exceptions for the ``protrepair`` package."""


class PrasError(Exception):
    """Base exception for the ``protrepair`` package."""


class PackingError(PrasError):
    """Raised when a side-chain packing request or result is invalid."""


class RefinementError(PrasError):
    """Raised when a local refinement request or backend execution is invalid."""


class ModelInvariantError(PrasError):
    """Raised when a canonical model invariant is violated."""


class AtomNotFoundError(PrasError):
    """Raised when a residue does not contain a requested atom."""


class ResidueNotFoundError(PrasError):
    """Raised when a chain does not contain a requested residue."""


class ChainNotFoundError(PrasError):
    """Raised when a structure does not contain a requested chain."""


class UnknownComponentError(PrasError):
    """Raised when a component library cannot resolve a component identifier."""


class GemmiUnavailableError(PrasError):
    """Raised when gemmi-backed I/O is requested without gemmi installed."""


class RdkitUnavailableError(PrasError):
    """Raised when RDKit-backed refinement is requested without RDKit."""


class UnsupportedFileFormatError(PrasError):
    """Raised when a structure file format cannot be inferred or serialized."""


class StructureNormalizationError(PrasError):
    """Raised when ingress normalization cannot produce a canonical structure."""
