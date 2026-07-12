"""Non-throwing NumPy vector algebra for geometry placement primitives."""

from math import acos, isfinite, sqrt

import numpy as np
import numpy.typing as npt

PlacementVector = npt.NDArray[np.float64]
PLACEMENT_VECTOR_NORM_EPSILON = 1e-12

_CARTESIAN_UNIT_AXES = (
    np.array((1.0, 0.0, 0.0), dtype=np.float64),
    np.array((0.0, 1.0, 0.0), dtype=np.float64),
    np.array((0.0, 0.0, 1.0), dtype=np.float64),
)


def vector_norm(vector: PlacementVector) -> float:
    """Return the Euclidean norm of one canonical placement vector."""

    return float(np.linalg.norm(vector))


def normalized_vector_or_none(
    vector: PlacementVector,
) -> PlacementVector | None:
    """Return one unit vector, or ``None`` when its direction is undefined."""

    norm = vector_norm(vector)
    if not isfinite(norm) or norm <= PLACEMENT_VECTOR_NORM_EPSILON:
        return None

    return np.asarray(vector / norm, dtype=np.float64)


def deterministic_unit_orthogonal_or_none(
    axis: PlacementVector,
) -> PlacementVector | None:
    """Return a deterministic unit vector orthogonal to a non-zero axis."""

    axis_norm = vector_norm(axis)
    if not isfinite(axis_norm) or axis_norm <= PLACEMENT_VECTOR_NORM_EPSILON:
        return None

    reference_axis = min(
        _CARTESIAN_UNIT_AXES,
        key=lambda candidate: abs(float(np.dot(axis, candidate))),
    )
    orthogonal = np.asarray(np.cross(axis, reference_axis), dtype=np.float64)
    return normalized_vector_or_none(orthogonal)


def scaled_point_from_origin_or_none(
    origin: PlacementVector,
    candidate: PlacementVector,
    distance: float,
) -> PlacementVector | None:
    """Scale a candidate point to one distance from an origin, if defined."""

    origin_components = _finite_components_or_none(origin)
    candidate_components = _finite_components_or_none(candidate)
    if (
        origin_components is None
        or candidate_components is None
        or not isfinite(distance)
    ):
        return None

    direction_x = candidate_components[0] - origin_components[0]
    direction_y = candidate_components[1] - origin_components[1]
    direction_z = candidate_components[2] - origin_components[2]
    direction_norm = sqrt(
        direction_x * direction_x
        + direction_y * direction_y
        + direction_z * direction_z
    )
    if (
        not isfinite(direction_norm)
        or direction_norm <= PLACEMENT_VECTOR_NORM_EPSILON
    ):
        return None

    scale = distance / direction_norm
    scaled_components = (
        origin_components[0] + direction_x * scale,
        origin_components[1] + direction_y * scale,
        origin_components[2] + direction_z * scale,
    )
    if not all(isfinite(component) for component in scaled_components):
        return None

    return np.asarray(scaled_components, dtype=np.float64)


def angle_between_vectors_radians_or_none(
    left_vector: PlacementVector,
    right_vector: PlacementVector,
) -> float | None:
    """Return the clamped angle between two non-zero vectors, if defined."""

    left_norm = vector_norm(left_vector)
    right_norm = vector_norm(right_vector)
    if (
        not isfinite(left_norm)
        or not isfinite(right_norm)
        or left_norm <= PLACEMENT_VECTOR_NORM_EPSILON
        or right_norm <= PLACEMENT_VECTOR_NORM_EPSILON
    ):
        return None

    cosine = float(np.dot(left_vector, right_vector)) / (left_norm * right_norm)
    if not isfinite(cosine):
        return None

    return acos(min(1.0, max(-1.0, cosine)))


def _finite_components_or_none(
    vector: PlacementVector,
) -> tuple[float, float, float] | None:
    """Return finite components from one canonical 3D placement vector."""

    components = (float(vector[0]), float(vector[1]), float(vector[2]))
    return components if all(isfinite(component) for component in components) else None
