"""Pure rigid-body geometry primitives."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from protrepair.geometry.vector import CoordinateLike, Vec3

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """Rigid transform that aligns one point cloud onto another."""

    rotation_matrix: FloatArray
    moving_centroid: FloatArray
    target_centroid: FloatArray

    @classmethod
    def fit(
        cls,
        moving_points: FloatArray,
        target_points: FloatArray,
        *,
        minimum_point_count: int = 3,
        minimum_rank: int = 2,
    ) -> "RigidTransform | None":
        """Fit one rigid transform from moving points onto target points."""

        if moving_points.shape != target_points.shape:
            raise ValueError(
                "rigid transform fit requires point clouds with matching shapes"
            )

        if moving_points.ndim != 2 or moving_points.shape[1] != 3:
            raise ValueError(
                "rigid transform fit requires point clouds with shape (n, 3)"
            )

        if moving_points.shape[0] < minimum_point_count:
            return None

        moving_centroid = moving_points.mean(axis=0)
        target_centroid = target_points.mean(axis=0)
        centered_moving_points = moving_points - moving_centroid
        if np.linalg.matrix_rank(centered_moving_points) < minimum_rank:
            return None

        centered_target_points = target_points - target_centroid
        covariance_matrix = centered_moving_points.T @ centered_target_points
        left_vectors, _, right_vectors_t = np.linalg.svd(covariance_matrix)
        orientation_sign = np.sign(np.linalg.det(left_vectors @ right_vectors_t))
        handedness_matrix = np.diag((1.0, 1.0, orientation_sign))
        rotation_matrix = left_vectors @ handedness_matrix @ right_vectors_t
        return cls(
            rotation_matrix=rotation_matrix,
            moving_centroid=moving_centroid,
            target_centroid=target_centroid,
        )

    def apply(self, point: CoordinateLike) -> FloatArray:
        """Apply this rigid transform to one Cartesian point."""

        point_array = Vec3.coerce(point).to_array()
        return (
            (point_array - self.moving_centroid) @ self.rotation_matrix
            + self.target_centroid
        )

    def apply_many(self, points: FloatArray) -> FloatArray:
        """Apply this rigid transform to one point cloud."""

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("rigid transform expects point clouds with shape (n, 3)")

        return (
            (points - self.moving_centroid) @ self.rotation_matrix
            + self.target_centroid
        )
