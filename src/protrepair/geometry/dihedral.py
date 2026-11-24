"""Signed dihedral measurements on fixed coordinates."""

from math import atan2, degrees, isfinite

import numpy as np

from protrepair.geometry.vector import Vec3


def measured_dihedral_degrees(
    first: Vec3, second: Vec3, third: Vec3, fourth: Vec3
) -> float | None:
    """Return the signed dihedral, or None for a degenerate or nonfinite plane."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        left = first.to_array() - second.to_array()
        axis = third.to_array() - second.to_array()
        right = fourth.to_array() - third.to_array()
        norm = float(np.linalg.norm(axis))
        if not isfinite(norm) or norm <= 1.0e-10:
            return None
        unit = axis / norm
        left -= np.dot(left, unit) * unit
        right -= np.dot(right, unit) * unit
        for plane in (left, right):
            norm = float(np.linalg.norm(plane))
            if not isfinite(norm) or norm <= 1.0e-10:
                return None
        cosine = float(np.dot(left, right))
        sine = float(np.dot(np.cross(unit, left), right))
    if not isfinite(cosine) or not isfinite(sine):
        return None
    return degrees(atan2(sine, cosine))
