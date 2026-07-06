"""Geometry placement exceptions."""

from protrepair.errors import ProtrepairError


class GeometryPlacementError(ProtrepairError, ValueError):
    """Raised when a geometry placement primitive is geometrically undefined.

    This is a package-domain error for degenerate placement frames or vectors.
    Malformed coordinate payloads remain ordinary input validation failures.
    """
