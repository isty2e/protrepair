"""Covalent geometry of a peptide junction, including amide-N substituents."""

from dataclasses import dataclass
from math import acos, atan2, degrees, isfinite

import numpy as np

from protrepair.geometry import Vec3


@dataclass(frozen=True, slots=True)
class PeptideJunctionGeometry:
    """Geometry screened at a newly materialized peptide connection.

    Parameters
    ----------
    preceding_ca, carbonyl_c, carbonyl_o : Vec3
        Carbonyl-side alpha carbon, carbon, and oxygen positions.
    nitrogen, following_ca : Vec3
        Amide-side nitrogen and alpha carbon positions.
    nitrogen_substituents : tuple[Vec3, ...]
        Other heavy neighbors of N, excluding CA and the preceding carbonyl C.
        The caller must supply a complete covalent neighborhood.
    """

    preceding_ca: Vec3
    carbonyl_c: Vec3
    carbonyl_o: Vec3
    nitrogen: Vec3
    following_ca: Vec3
    nitrogen_substituents: tuple[Vec3, ...] = ()

    def is_plausible(self) -> bool:
        """Return whether lengths, angles, and both amide planes are admissible.

        Returns
        -------
        bool
            False for degenerate or non-finite geometry, C-N outside 1.1-1.6 A,
            junction angles outside 85-145 degrees, or torsions more than
            30 degrees from either cis or trans planarity.
        """

        if not 1.1 <= self.carbonyl_c.distance_to(self.nitrogen) <= 1.6:
            return False
        angles = (
            _angle(self.preceding_ca, self.carbonyl_c, self.nitrogen),
            _angle(self.carbonyl_o, self.carbonyl_c, self.nitrogen),
            _angle(self.carbonyl_c, self.nitrogen, self.following_ca),
            *(
                _angle(self.carbonyl_c, self.nitrogen, neighbor)
                for neighbor in self.nitrogen_substituents
            ),
        )
        if not all(angle is not None and 85.0 <= angle <= 145.0 for angle in angles):
            return False
        torsions = (
            _dihedral(
                self.preceding_ca, self.carbonyl_c, self.nitrogen, self.following_ca
            ),
            _dihedral(
                self.carbonyl_o, self.carbonyl_c, self.nitrogen, self.following_ca
            ),
            *(
                _dihedral(self.carbonyl_c, self.nitrogen, self.following_ca, neighbor)
                for neighbor in self.nitrogen_substituents
            ),
        )
        return all(
            angle is not None and min(abs(angle), abs(180.0 - abs(angle))) <= 30.0
            for angle in torsions
        )


def _angle(first: Vec3, center: Vec3, third: Vec3) -> float | None:
    with np.errstate(over="ignore", invalid="ignore"):
        left = first.to_array() - center.to_array()
        right = third.to_array() - center.to_array()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if not isfinite(denominator) or denominator <= 1.0e-10:
            return None
        cosine = float(np.dot(left, right)) / denominator
    if not isfinite(cosine):
        return None
    return degrees(acos(min(1.0, max(-1.0, cosine))))


def _dihedral(first: Vec3, second: Vec3, third: Vec3, fourth: Vec3) -> float | None:
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
