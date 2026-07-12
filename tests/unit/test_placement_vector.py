"""Invariant tests for shared geometry-placement vector algebra."""

from math import pi

import numpy as np
import numpy.typing as npt
import pytest

from protrepair.geometry.placement_vector import (
    PLACEMENT_VECTOR_NORM_EPSILON,
    angle_between_vectors_radians_or_none,
    deterministic_unit_orthogonal_or_none,
    normalized_vector_or_none,
    scaled_point_from_origin_or_none,
    vector_norm,
)


def _vector(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
    return np.asarray((x, y, z), dtype=np.float64)


def test_vector_norm_returns_euclidean_length() -> None:
    """The shared norm primitive should preserve ordinary 3D geometry."""

    assert vector_norm(_vector(3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_normalized_vector_rejects_norm_at_degenerate_threshold() -> None:
    """The shared threshold should be closed on the degenerate side."""

    assert (
        normalized_vector_or_none(
            _vector(PLACEMENT_VECTOR_NORM_EPSILON, 0.0, 0.0)
        )
        is None
    )


def test_normalized_vector_accepts_norm_above_degenerate_threshold() -> None:
    """A representable norm above the threshold should retain its direction."""

    above_threshold = float(
        np.nextafter(PLACEMENT_VECTOR_NORM_EPSILON, np.inf)
    )
    normalized = normalized_vector_or_none(_vector(above_threshold, 0.0, 0.0))

    assert normalized == pytest.approx(_vector(1.0, 0.0, 0.0))


@pytest.mark.parametrize("non_finite", (float("nan"), float("inf")))
def test_normalized_vector_rejects_non_finite_direction(
    non_finite: float,
) -> None:
    """Non-finite direction data must not leak into placement coordinates."""

    assert normalized_vector_or_none(_vector(non_finite, 0.0, 0.0)) is None


def test_deterministic_orthogonal_vector_uses_stable_cartesian_tie_break() -> None:
    """Equal orthogonal candidates should preserve Cartesian axis order."""

    orthogonal = deterministic_unit_orthogonal_or_none(_vector(1.0, 0.0, 0.0))

    assert orthogonal == pytest.approx(_vector(0.0, 0.0, 1.0))


def test_deterministic_orthogonal_vector_rejects_zero_axis() -> None:
    """A zero axis has no canonical orthogonal direction."""

    assert deterministic_unit_orthogonal_or_none(_vector(0.0, 0.0, 0.0)) is None


@pytest.mark.parametrize("non_finite", (float("nan"), float("inf")))
def test_deterministic_orthogonal_vector_rejects_non_finite_axis(
    non_finite: float,
) -> None:
    """A non-finite axis must not produce a nominal orthogonal direction."""

    assert (
        deterministic_unit_orthogonal_or_none(_vector(non_finite, 0.0, 0.0))
        is None
    )


def test_deterministic_orthogonal_vector_is_unit_and_perpendicular() -> None:
    """An arbitrary defined axis should produce a normalized perpendicular."""

    axis = _vector(1.0, 2.0, 3.0)
    orthogonal = deterministic_unit_orthogonal_or_none(axis)

    assert orthogonal is not None
    assert vector_norm(orthogonal) == pytest.approx(1.0)
    assert float(np.dot(axis, orthogonal)) == pytest.approx(0.0, abs=1e-15)


def test_deterministic_orthogonal_vector_preserves_negative_axis_sign() -> None:
    """Cartesian tie-breaking should remain deterministic for negative axes."""

    orthogonal = deterministic_unit_orthogonal_or_none(_vector(-1.0, 0.0, 0.0))

    assert orthogonal == pytest.approx(_vector(0.0, 0.0, -1.0))


def test_scaled_point_rejects_coincident_candidate() -> None:
    """Coincident origin and candidate have no scaleable direction."""

    origin = _vector(1.0, 2.0, 3.0)

    assert scaled_point_from_origin_or_none(origin, origin.copy(), 1.5) is None


def test_scaled_point_preserves_requested_distance() -> None:
    """Defined point scaling should preserve direction and requested distance."""

    origin = _vector(1.0, 2.0, 3.0)
    scaled = scaled_point_from_origin_or_none(
        origin,
        _vector(3.0, 2.0, 3.0),
        1.5,
    )

    assert scaled is not None
    assert vector_norm(scaled - origin) == pytest.approx(1.5)
    assert scaled == pytest.approx(_vector(2.5, 2.0, 3.0))


def test_scaled_point_rejects_norm_overflow() -> None:
    """Finite inputs whose norm overflows must not collapse onto the origin."""

    assert (
        scaled_point_from_origin_or_none(
            _vector(0.0, 0.0, 0.0),
            _vector(1.0e308, 1.0e308, 0.0),
            1.0,
        )
        is None
    )


@pytest.mark.parametrize("non_finite", (float("nan"), float("inf")))
def test_scaled_point_rejects_non_finite_distance(non_finite: float) -> None:
    """A non-finite requested distance must not leak into coordinates."""

    assert (
        scaled_point_from_origin_or_none(
            _vector(0.0, 0.0, 0.0),
            _vector(1.0, 0.0, 0.0),
            non_finite,
        )
        is None
    )


@pytest.mark.parametrize(
    ("right_vector", "expected_angle"),
    (
        pytest.param((1.0, 0.0, 0.0), 0.0, id="parallel"),
        pytest.param((-1.0, 0.0, 0.0), pi, id="antiparallel"),
        pytest.param((0.0, 1.0, 0.0), pi / 2.0, id="orthogonal"),
    ),
)
def test_angle_between_vectors_preserves_canonical_extrema(
    right_vector: tuple[float, float, float],
    expected_angle: float,
) -> None:
    """Shared angle algebra should preserve clamped canonical extrema."""

    angle = angle_between_vectors_radians_or_none(
        _vector(1.0, 0.0, 0.0),
        _vector(*right_vector),
    )

    assert angle == pytest.approx(expected_angle)


def test_angle_between_vectors_rejects_non_finite_direction() -> None:
    """Non-finite vectors must not be misclassified as a finite angle."""

    assert (
        angle_between_vectors_radians_or_none(
            _vector(float("inf"), 0.0, 0.0),
            _vector(1.0, 0.0, 0.0),
        )
        is None
    )


def test_angle_between_vectors_rejects_zero_direction() -> None:
    """A zero operand has no defined angle."""

    assert (
        angle_between_vectors_radians_or_none(
            _vector(0.0, 0.0, 0.0),
            _vector(1.0, 0.0, 0.0),
        )
        is None
    )


def test_angle_between_near_parallel_vectors_remains_finite() -> None:
    """Near-parallel rounding should stay inside the acos domain."""

    angle = angle_between_vectors_radians_or_none(
        _vector(1.0, 1.0e-14, 0.0),
        _vector(1.0, 0.0, 0.0),
    )

    assert angle is not None
    assert 0.0 <= angle <= pi


def test_normalized_vector_preserves_negative_direction() -> None:
    """Normalization should not discard vector orientation."""

    normalized = normalized_vector_or_none(_vector(-2.0, 0.0, 0.0))

    assert normalized == pytest.approx(_vector(-1.0, 0.0, 0.0))
