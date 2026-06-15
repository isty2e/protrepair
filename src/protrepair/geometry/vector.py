"""Canonical vector primitives shared across ProtRepair geometry consumers."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Vec3:
    """Immutable 3D coordinate or vector value object."""

    x: float
    y: float
    z: float

    @classmethod
    def from_iterable(cls, coordinates: Iterable[float]) -> "Vec3":
        """Construct one vector from a length-3 iterable."""

        values = tuple(float(value) for value in coordinates)
        if len(values) != 3:
            raise ValueError("Vec3 requires exactly three coordinates")

        return cls(x=values[0], y=values[1], z=values[2])

    @classmethod
    def coerce(
        cls,
        coordinates: "Vec3 | Sequence[float] | npt.NDArray[np.float64]",
    ) -> "Vec3":
        """Return a canonical vector from any accepted coordinate-like input."""

        if isinstance(coordinates, cls):
            return coordinates

        return cls.from_iterable(coordinates)

    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert this vector to one NumPy array."""

        return np.asarray((self.x, self.y, self.z), dtype=np.float64)

    def distance_to(self, other: "Vec3") -> float:
        """Return the Euclidean distance to another vector."""

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def with_offset(self, dx: float, dy: float, dz: float) -> "Vec3":
        """Return a translated vector."""

        return type(self)(x=self.x + dx, y=self.y + dy, z=self.z + dz)

    def __iter__(self) -> Iterator[float]:
        """Yield coordinates in Cartesian order."""

        yield self.x
        yield self.y
        yield self.z


CoordinateLike = Vec3 | Sequence[float] | npt.NDArray[np.float64]
