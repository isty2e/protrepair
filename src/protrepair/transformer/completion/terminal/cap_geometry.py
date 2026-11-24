"""Full-methyl peptide cap poses and their junction geometry checks."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from math import acos, degrees, isfinite

import numpy as np

from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.diagnostics.geometry import HeavyGeometryPolicy
from protrepair.diagnostics.peptide_geometry import PeptideJunctionGeometry
from protrepair.geometry import Vec3
from protrepair.geometry.internal_coordinates import InternalCoordinateFrame

GEOMETRY = HeavyGeometryPolicy()
PHI_OFFSETS = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
METHYL_OFFSETS = (0, 30, -30, 60)


def _unit(v: np.ndarray) -> np.ndarray:
    d = float(np.linalg.norm(v))
    if not isfinite(d) or d <= 1.0e-10:
        raise ValueError("degenerate_frame")
    return v / d


def _third(center: Vec3, a: Vec3, b: Vec3, length: float) -> Vec3:
    c = center.to_array()
    u, v = _unit(a.to_array() - c), _unit(b.to_array() - c)
    _unit(np.cross(u, v))
    return Vec3.from_iterable(c - length * _unit(u + v))


def _place(a: Vec3, b: Vec3, c: Vec3, d: float, angle: float, dihedral: float) -> Vec3:
    _unit(np.cross(b.to_array() - a.to_array(), c.to_array() - b.to_array()))
    return InternalCoordinateFrame(a, b, c).place(
        bond_length=d, bond_angle_degrees=angle, dihedral_degrees=dihedral
    )


def _angle(a: Vec3, b: Vec3, c: Vec3) -> float:
    return degrees(
        acos(
            float(
                np.clip(
                    _unit(a.to_array() - b.to_array())
                    @ _unit(c.to_array() - b.to_array()),
                    -1,
                    1,
                )
            )
        )
    )


@dataclass(frozen=True, slots=True)
class PeptideCapPose:
    """A rigidly attached blocking group and, when needed, a new amide H."""

    cap: dict[str, Vec3]
    flank_h: Vec3 | None
    phi_offset: int
    methyl_offset: int


@dataclass(frozen=True, slots=True)
class PeptideCapFrame:
    """Observed anchors for an N- or C-directed amide closure."""

    side: str
    positions: dict[str, Vec3]
    n_substituent: str | None

    def poses(self) -> Iterator[PeptideCapPose]:
        """Yield staggered poses first, then alternate methyl phases."""
        p = self.positions
        azimuths = (
            PHI_OFFSETS if self.side == "N" and self.n_substituent is None else (0,)
        )
        for phase in METHYL_OFFSETS:
            for az in azimuths:
                extra = None
                if self.side == "N":
                    end = p["N"]
                    anchor = (
                        _place(p["C"], p["CA"], end, 1.329, 121.7, 180 + az)
                        if self.n_substituent is None
                        else _third(end, p["CA"], p[self.n_substituent], 1.329)
                    )
                    methyl = _place(p["CA"], end, anchor, 1.525, 116.2, 180)
                    coords = {
                        "C": anchor,
                        "O": _place(p["CA"], end, anchor, 1.231, 123, 0),
                        "CH3": methyl,
                    }
                    if self.n_substituent is None:
                        extra = _third(end, p["CA"], anchor, 1.01)
                else:
                    end = p["C"]
                    anchor = _third(end, p["CA"], p["O"], 1.329)
                    methyl = _place(p["CA"], end, anchor, 1.458, 121.7, 180)
                    coords = {
                        "N": anchor,
                        "C": methyl,
                        "HN2": _third(anchor, end, methyl, 1.01),
                    }
                for i, base in enumerate((60, 180, 300), 1):
                    coords[f"H{i}"] = _place(
                        end, anchor, methyl, 1.09, 109.4712206, base + phase
                    )
                yield PeptideCapPose(coords, extra, az, phase)

    def verify(self, pose: PeptideCapPose, *, trans: bool = True) -> None:
        """Check the two-sided geometry; known deposited caps can also be cis."""
        p, q = self.positions, pose.cap
        if self.side == "N":
            before, c, o, n, after = q["CH3"], q["C"], q["O"], p["N"], p["CA"]
            subst = (
                p[self.n_substituent]
                if self.n_substituent is not None
                else pose.flank_h
            )
            heavy = (subst,) if self.n_substituent == "CD" and subst is not None else ()
            methyl, anchor, carbonyl = before, c, True
        else:
            before, c, o, n, after = p["CA"], p["C"], p["O"], q["N"], q["C"]
            subst = q["HN2"]
            heavy = ()
            methyl, anchor, carbonyl = after, n, False
        if not PeptideJunctionGeometry(before, c, o, n, after, heavy).is_plausible():
            raise ValueError("amide_junction_geometry")
        omega = abs(InternalCoordinateFrame.torsion(before, c, n, after))
        if trans and omega < 150:
            raise ValueError("nontrans_junction")

        def check_angle(a: Vec3, b: Vec3, d: Vec3) -> None:
            if (
                not GEOMETRY.minimum_bond_angle_degrees
                <= _angle(a, b, d)
                <= GEOMETRY.maximum_bond_angle_degrees
            ):
                raise ValueError("internal_angle")

        tol = GEOMETRY.bond_length_tolerance_angstrom
        if (
            abs(c.distance_to(o) - 1.23) > tol
            or abs(methyl.distance_to(anchor) - (1.52 if carbonyl else 1.46)) > tol
            or abs(
                p["CA"].distance_to(p["C"] if self.side == "C" else p["N"])
                - (1.52 if self.side == "C" else 1.46)
            )
            > tol
        ):
            raise ValueError("internal_heavy_bond")
        if subst is None or (
            not heavy
            and not source_hydrogen_bond_length_is_compatible(
                n.distance_to(subst), 1.01
            )
        ):
            raise ValueError("amide_h_or_substituent")
        check_angle(before, c, o)
        check_angle(c, n, subst)
        check_angle(after, n, subst)
        off = abs(InternalCoordinateFrame.torsion(c, n, after, subst))
        if min(off, 180 - off) > 30:
            raise ValueError("amide_n_pyramid")
        hs = [q[f"H{i}"] for i in (1, 2, 3)]
        if any(
            not source_hydrogen_bond_length_is_compatible(methyl.distance_to(h), 1.09)
            for h in hs
        ):
            raise ValueError("methyl_h_length")
        for a, b in combinations([anchor, *hs], 2):
            check_angle(a, methyl, b)
