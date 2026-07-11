"""Project-specific exceptions for the ``protrepair`` package."""


class ProtrepairError(Exception):
    """Base exception for the ``protrepair`` package."""


class PackingError(ProtrepairError):
    """Raised when a side-chain packing request or result is invalid."""


class RefinementError(ProtrepairError):
    """Raised when a local refinement request or backend execution is invalid."""


class ModelInvariantError(ProtrepairError):
    """Raised when a canonical model invariant is violated."""


class AtomNotFoundError(ProtrepairError):
    """Raised when a residue does not contain a requested atom."""


class ResidueNotFoundError(ProtrepairError):
    """Raised when a chain does not contain a requested residue."""


class ChainNotFoundError(ProtrepairError):
    """Raised when a structure does not contain a requested chain."""


class UnknownComponentError(ProtrepairError):
    """Raised when a component library cannot resolve a component identifier."""


class UnknownElementRadiusError(ProtrepairError, ValueError):
    """Raised when an element has no radius under the requested radius kind."""


class RdkitUnavailableError(ProtrepairError):
    """Raised when a required RDKit backend cannot be imported."""


class UnsupportedFileFormatError(ProtrepairError):
    """Raised when a structure file format cannot be inferred or serialized."""


class StructureInputTooLargeError(ProtrepairError):
    """Raised when a structure input exceeds the public ingress size limit."""


class StructureNormalizationError(ProtrepairError):
    """Raised when ingress normalization cannot produce a canonical structure."""
