"""Contract tests for axis rotation geometry."""

from math import pi

import numpy as np
import pytest

from protrepair.geometry import AxisRotation, Vec3


def test_zero_axis_rotation_is_identity() -> None:
    """Zero-axis rotation is an intentional identity, not placement failure."""

    vector = Vec3(1.0, 2.0, 3.0)
    rotated = AxisRotation(axis=Vec3(0.0, 0.0, 0.0)).rotate_vector(
        vector,
        theta_radians=pi / 3.0,
    )

    assert rotated == pytest.approx(vector.to_array())


def test_nonzero_axis_rotation_preserves_vector_norm() -> None:
    """A defined rotation should preserve norm without using identity fallback."""

    vector = Vec3(1.0, 0.0, 0.0)
    rotated = AxisRotation(axis=Vec3(0.0, 0.0, 1.0)).rotate_vector(
        vector,
        theta_radians=pi / 2.0,
    )

    assert float(np.linalg.norm(rotated)) == pytest.approx(1.0)
    assert rotated != pytest.approx(vector.to_array())
