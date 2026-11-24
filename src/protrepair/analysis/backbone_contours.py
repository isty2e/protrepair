"""Residue-class Top8000 contours used to screen generated peptide planes."""

from enum import Enum
from functools import cache
from gzip import decompress
from hashlib import sha256
from importlib.resources import files
from io import BytesIO
from math import floor, isfinite
from types import MappingProxyType

import numpy as np
import numpy.typing as npt


class BackboneContour(str, Enum):
    """Reference distribution for an ordinary, non-proline peptide N."""

    GENERAL = "general-noGPIVpreP"
    GLYCINE = "gly-sym"
    PRE_PROLINE = "prepro-noGP"
    ILE_VAL = "ileval-nopreP"

    @classmethod
    def for_nonproline(cls, component: str, following: str) -> "BackboneContour":
        """Select the contour from canonical backbone families on a linked peptide.

        Glycine keeps its own contour before proline. The cis/trans proline
        references are outside the generated-ACE search.
        """
        if component == "PRO":
            raise ValueError("proline requires a cis/trans backbone reference")
        if component == "GLY":
            return cls.GLYCINE
        if following == "PRO":
            return cls.PRE_PROLINE
        if component in ("ILE", "VAL"):
            return cls.ILE_VAL
        return cls.GENERAL

    @property
    def allowed_floor(self) -> float:
        """Return the lower boundary of the published allowed region."""
        return 0.0005 if self is BackboneContour.GENERAL else 0.001

    def density(self, phi: float, psi: float) -> float:
        """Interpolate the relative density at signed phi and psi (degrees)."""
        if not (isfinite(phi) and isfinite(psi)):
            raise ValueError("backbone angles must be finite")
        grid = self._grid()
        a, b = (phi + 179) / 2, (psi + 179) / 2
        i, j = floor(a), floor(b)
        u, v = a - i, b - j
        return sum(
            float(grid[(i + di) % 180, (j + dj) % 180]) * wi * wj
            for di, wi in enumerate((1 - u, u))
            for dj, wj in enumerate((1 - v, v))
        )

    @cache
    def _grid(self) -> npt.NDArray[np.float64]:
        resource = (
            files("protrepair.analysis")
            .joinpath("resources")
            .joinpath(f"rama8000-{self.value}.data.gz")
        )
        data = decompress(resource.read_bytes())
        if sha256(data).hexdigest() != _GRID_DIGESTS[self]:
            raise ValueError(f"invalid {self.value} backbone contour")

        x, y, values = np.loadtxt(BytesIO(data), dtype=np.float64, unpack=True)
        grid = np.zeros((180, 180), dtype=np.float64)
        grid[((x + 179) / 2).astype(int), ((y + 179) / 2).astype(int)] = values
        grid.setflags(write=False)
        return grid


# The sparse source tables omit zero-density bins. A damaged file must not
# silently turn an occupied bin into a zero.
_GRID_DIGESTS = MappingProxyType(
    {
        BackboneContour.GENERAL: "ccdbc6a201ca2510119e77b0dd169ca2cdea7f2c9ed348555a3b1129f8c2b00a",
        BackboneContour.GLYCINE: "89c75a5ac036ff3309c51b46a2412f30137827abaff4606209ed41e04fcba637",
        BackboneContour.PRE_PROLINE: "1cc10b92911d47ca775f0a8131a029b42634dbe26364cece4cfdc8c85b9c3fab",
        BackboneContour.ILE_VAL: "567e1318128a50b8f44362427bf717d564749936d22d5dbfd34e5b0c17819276",
    }
)
