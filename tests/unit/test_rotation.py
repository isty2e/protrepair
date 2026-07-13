"""Contract tests for axis rotation geometry."""

from math import pi

import numpy as np
import pytest
from tests.support.canonical_builders import atom_payload, residue_payload

import protrepair.geometry.rotation as rotation_module
from protrepair.geometry import AxisRotation, Vec3
from protrepair.structure.labels import ResidueId
from protrepair.transformer.discrete.orientation import (
    rotate_residue_atoms_about_axis,
)


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


@pytest.mark.parametrize(
    "axis",
    (Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)),
)
def test_batch_point_rotation_matches_individual_rotation(axis: Vec3) -> None:
    """Batch rotation should preserve the established pointwise contract."""

    rotation = AxisRotation(axis=axis)
    origin = Vec3(1.0, 2.0, 3.0)
    points = (
        Vec3(2.0, 2.0, 3.0),
        Vec3(1.0, 4.0, 3.0),
        Vec3(-1.0, 2.0, 5.0),
    )

    actual = rotation.rotate_points(
        points,
        origin=origin,
        theta_radians=pi / 2.0,
    )
    expected = tuple(
        rotation.rotate_point(
            point,
            origin=origin,
            theta_radians=pi / 2.0,
        )
        for point in points
    )

    for actual_point, expected_point in zip(actual, expected, strict=True):
        assert actual_point.to_array() == pytest.approx(expected_point.to_array())


@pytest.mark.parametrize(
    ("axis", "origin", "point"),
    (
        (
            Vec3(0.0, 0.0, 0.0),
            Vec3(1e308, -1e308, 1e308),
            Vec3(-1e308, 1e308, 0.0),
        ),
        (
            Vec3(1e-320, 0.0, 0.0),
            Vec3(float("inf"), 0.0, 0.0),
            Vec3(1.0, 2.0, 3.0),
        ),
    ),
)
def test_batch_degenerate_axis_preserves_pointwise_arithmetic(
    axis: Vec3,
    origin: Vec3,
    point: Vec3,
) -> None:
    """Degenerate-axis batching should retain cancellation and NaN semantics."""

    rotation = AxisRotation(axis=axis)
    with np.errstate(over="ignore", invalid="ignore"):
        actual = rotation.rotate_points(
            (point,),
            origin=origin,
            theta_radians=pi / 3.0,
        )[0]
        expected = rotation.rotate_point(
            point,
            origin=origin,
            theta_radians=pi / 3.0,
        )

    assert np.array_equal(
        actual.to_array(),
        expected.to_array(),
        equal_nan=True,
    )


def test_residue_rotation_constructs_one_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One residue rotation should share trigonometry across selected atoms."""

    residue_site, residue_geometry, formal_charges = residue_payload(
        component_id="HIS",
        residue_id=ResidueId("A", 1),
        atoms=(
            atom_payload("CB", "C", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CG", "C", Vec3(0.0, 0.0, 1.0)),
            atom_payload("ND1", "N", Vec3(1.0, 0.0, 1.0)),
            atom_payload("CD2", "C", Vec3(0.0, 1.0, 1.0)),
            atom_payload("CE1", "C", Vec3(-1.0, 0.0, 1.0)),
        ),
    )
    cosine_calls: list[float] = []
    original_cosine = rotation_module.cos

    def recording_cosine(theta_radians: float) -> float:
        cosine_calls.append(theta_radians)
        return original_cosine(theta_radians)

    monkeypatch.setattr(rotation_module, "cos", recording_cosine)

    _, _, _, moved_atom_names = rotate_residue_atoms_about_axis(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=formal_charges,
        axis_atom_names=("CB", "CG"),
        rotated_atom_names={"ND1", "CD2", "CE1"},
        theta_radians=pi,
    )

    assert moved_atom_names == ("ND1", "CD2", "CE1")
    assert cosine_calls == [pi]
