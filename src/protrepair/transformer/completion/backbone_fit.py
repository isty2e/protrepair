"""Fixed-anchor reconstruction of peptide carbonyls and alpha carbons."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, product
from math import cos, radians, sin

import numpy as np
from numpy.typing import NDArray

from protrepair.diagnostics.peptide_geometry import PeptideJunctionGeometry
from protrepair.geometry import GeometryPlacementError, RigidTransform, Vec3

_Array = NDArray[np.float64]
_CA = (1.525, 0.0, 0.0)
_C = (0.0, 0.0, 0.0)
_O = (1.231 * cos(radians(120.8)), -1.231 * sin(radians(120.8)), 0.0)
_N = (1.329 * cos(radians(116.2)), 1.329 * sin(radians(116.2)), 0.0)
_CARBONYL = np.array((_CA, _C, _O, _N))
_ALPHA = np.array(
    ((-0.525, 1.363, 0), (0, 0, 0), (1.526, 0, 0), (-0.529, -0.774, -1.205))
)
_CARBONYL.setflags(write=False)
_ALPHA.setflags(write=False)
_BONDS = frozenset(
    frozenset(pair)
    for pair in (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("-C", "N"),
        ("C", "+N"),
        ("N", "CD"),
    )
)


def _amide(cis: bool, proline: bool) -> _Array:
    axis = -np.asarray(_N) / 1.329
    side = np.asarray(_CA) - _N
    side -= (side @ axis) * axis
    side *= (1 if cis else -1) / np.linalg.norm(side)
    ca = np.asarray(_N) + 1.458 * (
        cos(radians(121.7)) * axis + sin(radians(121.7)) * side
    )
    points = [*_CARBONYL, ca]
    if proline:
        points.append(
            np.asarray(_N)
            + 1.47 * (cos(radians(126)) * axis - sin(radians(126)) * side)
        )
    return np.array(points)


def _frame_fit(reference: _Array, anchors: list[int], points: _Array) -> RigidTransform:
    if (
        len(anchors) < 3
        or np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-5) < 2
    ):
        raise GeometryPlacementError("collinear or missing peptide-plane anchors")
    fit = RigidTransform.fit(reference[anchors], points)
    if fit is None:
        raise GeometryPlacementError("singular peptide reference frame")
    return fit


@dataclass(frozen=True, slots=True)
class _Distances:
    indices: NDArray[np.intp]
    lengths: _Array
    sigmas: _Array
    incidence: _Array
    fixed: _Array

    def residual(self, variables: _Array) -> tuple[_Array, _Array]:
        xyz = np.concatenate((variables, self.fixed))
        delta = xyz[self.indices[:, 0]] - xyz[self.indices[:, 1]]
        norms = np.linalg.norm(delta, axis=1)
        deriv = delta / (np.maximum(norms, 1e-12) * self.sigmas)[:, None]
        return (norms - self.lengths) / self.sigmas, (
            deriv[:, None, :] * self.incidence[:, :, None]
        ).reshape(-1, variables.size)

    def fit(self, seed: _Array) -> tuple[float, _Array] | None:
        x = seed.copy()
        damping = 0.08
        for _ in range(42):
            r, j = self.residual(x)
            gradient = np.einsum("ri,r->i", j, r)
            hessian = np.einsum("ri,rj->ij", j, j)
            step = np.linalg.solve(
                hessian + np.diag(damping * np.maximum(hessian.diagonal(), 1)), gradient
            )
            if np.max(np.abs(gradient)) < 1e-4 or np.max(np.abs(step)) < 1e-6:
                sv = np.linalg.svd(j, compute_uv=False)
                if (
                    damping >= 1e4
                    or len(sv) < x.size
                    or sv[-1] <= 0.015 * sv[0]
                    or np.max(np.abs(r)) > 5
                ):
                    return None
                return float(r @ r), x
            trial = x - step.reshape(x.shape)
            nr, _ = self.residual(trial)
            if nr @ nr < r @ r:
                x, damping = trial, max(1e-7, 0.28 * damping)
            else:
                damping *= 5
        return None


@dataclass(frozen=True, slots=True)
class BackboneFit:
    """Selected positive anchors; - and + name verified peptide neighbors."""

    anchors: Mapping[str, Vec3]
    component: str
    following: str

    def place(self, names: frozenset[str]) -> dict[str, Vec3]:
        """Reconstruct a carbonyl site or solve jointly for CA and C."""
        if names.intersection(self.anchors):
            raise ValueError("a target cannot supply an anchor")
        if names in ({"C"}, {"O"}):
            name = next(iter(names))
            keys = ("CA", "C", "O", "+N")
            ids = [i for i, k in enumerate(keys) if k != name and k in self.anchors]
            if len(ids) != 3:
                raise GeometryPlacementError("missing carbonyl plane")
            fit = _frame_fit(
                _CARBONYL,
                ids,
                np.array([self.anchors[keys[i]].to_array() for i in ids]),
            )
            out = {name: Vec3.coerce(fit.apply(_CARBONYL[keys.index(name)]))}
            self._verify(out)
            return out
        if names != {"CA", "C"}:
            raise GeometryPlacementError("unsupported backbone mask")
        needed = {"-CA", "-C", "-O", "N", "O", "+N", "+CA"}
        if not needed <= self.anchors.keys() or (
            self.component != "GLY" and "CB" not in self.anchors
        ):
            raise GeometryPlacementError("missing coupled backbone anchors")
        roots = [
            root
            for left, right in product((False, True), repeat=2)
            for root in self._branch(left, right)
        ]
        if not roots:
            raise GeometryPlacementError("no admissible coupled placement")
        _, best = min(roots, key=lambda item: item[0])
        if any(np.linalg.norm(root - best, axis=1).max() > 0.35 for _, root in roots):
            raise GeometryPlacementError("competing coupled placements")
        return {k: Vec3.coerce(v) for k, v in zip(("CA", "C"), best, strict=True)}

    def _verify(self, candidate: dict[str, Vec3]) -> None:
        points = {**self.anchors, **candidate}
        edges = [("CA", "C", "O", "+N", "+CA", "+CD", self.following)]
        if "CA" in candidate:
            edges.append(("-CA", "-C", "-O", "N", "CA", "CD", self.component))
        for ca, c, o, n, right, cd, typ in edges:
            required = {ca, c, o, n, right} | ({cd} if typ == "PRO" else set())
            if not required <= points.keys():
                raise GeometryPlacementError("incomplete amide neighborhood")
            if not PeptideJunctionGeometry(
                points[ca],
                points[c],
                points[o],
                points[n],
                points[right],
                (points[cd],) if typ == "PRO" else (),
            ).is_plausible():
                raise GeometryPlacementError("implausible peptide junction")
        for a, b, ideal in (
            ("CA", "C", 1.525),
            ("C", "O", 1.231),
            ("C", "+N", 1.329),
            ("CA", "N", 1.458),
            ("CA", "CB", 1.526),
        ):
            if (
                (a in candidate or b in candidate)
                and a in points
                and b in points
                and abs(points[a].distance_to(points[b]) - ideal) > 0.10
            ):
                raise GeometryPlacementError("inconsistent covalent bond length")
        if self.component != "GLY" and {"N", "CA", "C", "CB"} <= points.keys():
            ca_xyz = points["CA"].to_array()
            n_xyz, c_xyz, b_xyz = (
                points[k].to_array() - ca_xyz for k in ("N", "C", "CB")
            )
            if np.dot(np.cross(n_xyz, c_xyz), b_xyz) <= 0:
                raise GeometryPlacementError("inverted alpha carbon")

    def _branch(self, left_cis: bool, right_cis: bool) -> list[tuple[float, _Array]]:
        names = ["CA", "C", *sorted(self.anchors)]
        points = np.array([self.anchors[name].to_array() for name in names[2:]])
        pairs: dict[tuple[int, int], tuple[float, float]] = {}
        seeds: list[_Array] = []
        fixed_score = 0.0
        templates = (
            (
                ("-CA", "-C", "-O", "N", "CA", "CD"),
                _amide(left_cis, self.component == "PRO"),
            ),
            (
                ("CA", "C", "O", "+N", "+CA", "+CD"),
                _amide(right_cis, self.following == "PRO"),
            ),
            (("N", "CA", "C", "CB"), _ALPHA),
        )
        for keys, ref in templates:
            keys = keys[: len(ref)]
            valid = [i for i, k in enumerate(keys) if k in names]
            anchors = [i for i in valid if keys[i] in self.anchors]
            if len(anchors) >= 3 and "C" in keys and "CA" in keys:
                fit = _frame_fit(
                    ref,
                    anchors,
                    np.array([self.anchors[keys[i]].to_array() for i in anchors]),
                )
                seeds.append(fit.apply_many(ref[[keys.index("CA"), keys.index("C")]]))
            for a, b in combinations(valid, 2):
                i, j = sorted((names.index(keys[a]), names.index(keys[b])))
                length = float(np.linalg.norm(ref[a] - ref[b]))
                sigma = 0.032 if frozenset((keys[a], keys[b])) in _BONDS else 0.085
                if ref is not _ALPHA and a < 3 and b >= 4:
                    sigma = 0.16
                if i >= 2:
                    if sigma == 0.16:
                        err = abs(
                            np.linalg.norm(points[i - 2] - points[j - 2]) - length
                        )
                        if err > 0.65:
                            return []
                        fixed_score += (err / sigma) ** 2
                elif (i, j) not in pairs or sigma < pairs[i, j][1]:
                    pairs[i, j] = (length, sigma)
        indices = np.array(list(pairs), dtype=np.intp)
        values = np.array(list(pairs.values()))
        incidence = (
            np.eye(len(names))[indices[:, 0], :2]
            - np.eye(len(names))[indices[:, 1], :2]
        )
        objective = _Distances(indices, values[:, 0], values[:, 1], incidence, points)
        results: list[tuple[float, _Array]] = []
        for seed in seeds:
            outcome = objective.fit(seed)
            if outcome is None:
                continue
            score, x = outcome
            try:
                self._verify(
                    {k: Vec3.coerce(v) for k, v in zip(names[:2], x, strict=True)}
                )
            except GeometryPlacementError:
                continue
            results.append((float(score + fixed_score), x))
        return results
