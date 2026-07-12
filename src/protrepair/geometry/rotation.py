"""Pure axis-rotation geometry primitives."""

from dataclasses import dataclass
from math import cos, sin

import numpy as np
import numpy.typing as npt

from protrepair.geometry.vector import CoordinateLike, Vec3

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AxisRotation:
    """Behaviorful Rodrigues rotation around one Cartesian axis."""

    axis: Vec3

    @classmethod
    def from_points(
        cls,
        start: CoordinateLike,
        end: CoordinateLike,
    ) -> "AxisRotation":
        """Build one axis rotation from two points spanning the axis."""

        start_vector = Vec3.coerce(start)
        end_vector = Vec3.coerce(end)
        return cls(
            axis=Vec3(
                x=end_vector.x - start_vector.x,
                y=end_vector.y - start_vector.y,
                z=end_vector.z - start_vector.z,
            )
        )

    def rotate_vector(
        self,
        vector: CoordinateLike,
        *,
        theta_radians: float,
    ) -> FloatArray:
        """Return one vector rotated around this axis."""

        vector_array = Vec3.coerce(vector).to_array()
        axis_array = self.axis.to_array()
        axis_length = float(np.linalg.norm(axis_array))
        # A zero rotation axis is identity by contract, not a placement failure.
        if axis_length == 0.0:
            return vector_array

        unit_axis = axis_array / axis_length
        x_norm = float(unit_axis[0])
        y_norm = float(unit_axis[1])
        z_norm = float(unit_axis[2])
        sin_theta = sin(theta_radians)
        cos_theta = cos(theta_radians)
        one_minus_cos = 1.0 - cos_theta

        rotation_matrix = np.asarray(
            (
                (
                    cos_theta + x_norm * x_norm * one_minus_cos,
                    x_norm * y_norm * one_minus_cos - z_norm * sin_theta,
                    x_norm * z_norm * one_minus_cos + y_norm * sin_theta,
                ),
                (
                    x_norm * y_norm * one_minus_cos + z_norm * sin_theta,
                    cos_theta + y_norm * y_norm * one_minus_cos,
                    y_norm * z_norm * one_minus_cos - x_norm * sin_theta,
                ),
                (
                    x_norm * z_norm * one_minus_cos - y_norm * sin_theta,
                    y_norm * z_norm * one_minus_cos + x_norm * sin_theta,
                    cos_theta + z_norm * z_norm * one_minus_cos,
                ),
            ),
            dtype=np.float64,
        )
        return vector_array @ rotation_matrix

    def rotate_point(
        self,
        point: CoordinateLike,
        *,
        origin: CoordinateLike,
        theta_radians: float,
    ) -> Vec3:
        """Return one point rotated around this axis relative to one origin."""

        origin_vector = Vec3.coerce(origin).to_array()
        point_vector = Vec3.coerce(point).to_array()
        rotated = self.rotate_vector(
            point_vector - origin_vector,
            theta_radians=theta_radians,
        )
        return Vec3.from_iterable(rotated + origin_vector)
