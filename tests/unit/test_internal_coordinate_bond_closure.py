"""Analytic closure retains anchors and exposes both feasible solutions."""

from math import degrees

import numpy as np
import pytest

from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3


@pytest.mark.parametrize("torsion", (-175.0, -90.0, -1.0, 0.0, 45.0, 179.0))
def test_closure_candidates_satisfy_both_lengths_and_fixed_angle(
    torsion: float,
) -> None:
    frame = InternalCoordinateFrame(Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(1, 0, 0))
    closing = Vec3(0, 2, 1)
    known = frame.place(
        bond_length=1.5, bond_angle_degrees=110.0, dihedral_degrees=torsion
    )
    closing_length = float(np.linalg.norm(known.to_array() - closing.to_array()))
    candidates = frame.bond_closure_candidates(
        bond_length=1.5,
        bond_angle_degrees=110.0,
        preferred_dihedral_degrees=torsion,
        closing_anchor=closing,
        closing_bond_length=closing_length,
    )
    np.testing.assert_allclose(candidates[0].to_array(), known.to_array(), atol=1e-12)
    for candidate in candidates:
        a = Vec3.coerce(frame.atom_b).to_array() - Vec3.coerce(frame.atom_c).to_array()
        b = candidate.to_array() - Vec3.coerce(frame.atom_c).to_array()
        assert np.linalg.norm(b) == pytest.approx(1.5)
        assert np.linalg.norm(
            candidate.to_array() - closing.to_array()
        ) == pytest.approx(closing_length)
        angle = degrees(
            float(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        )
        assert angle == pytest.approx(110.0)


def test_infeasible_fixed_angle_uses_finite_nearest_two_sphere_solution() -> None:
    frame = InternalCoordinateFrame(Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(1, 0, 0))
    closing = Vec3(1.8, 0.0, 0.0)
    candidates = frame.bond_closure_candidates(
        bond_length=1.0,
        bond_angle_degrees=110.0,
        preferred_dihedral_degrees=60.0,
        closing_anchor=closing,
        closing_bond_length=0.5,
    )
    assert len(candidates) == 1
    assert np.linalg.norm(
        candidates[0].to_array() - Vec3.coerce(frame.atom_c).to_array()
    ) == pytest.approx(1.0)
    assert np.linalg.norm(
        candidates[0].to_array() - closing.to_array()
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("length", (float("inf"), float("nan"), 0.0, -1.0))
@pytest.mark.filterwarnings("error")
def test_invalid_closure_length_is_typed_geometry_failure(length: float) -> None:
    frame = InternalCoordinateFrame(Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(1, 0, 0))
    with pytest.raises(GeometryPlacementError):
        frame.bond_closure_candidates(
            bond_length=1.5,
            bond_angle_degrees=110.0,
            preferred_dihedral_degrees=0.0,
            closing_anchor=Vec3(0, 2, 1),
            closing_bond_length=length,
        )


def test_physically_disjoint_bond_spheres_do_not_fabricate_closure() -> None:
    frame = InternalCoordinateFrame(Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(1, 0, 0))
    with pytest.raises(GeometryPlacementError, match="incompatible"):
        frame.bond_closure_candidates(
            bond_length=1.0,
            bond_angle_degrees=110.0,
            preferred_dihedral_degrees=0.0,
            closing_anchor=Vec3(10, 1, 0),
            closing_bond_length=1.0,
        )


@pytest.mark.parametrize("value", (float("inf"), float("nan")))
@pytest.mark.filterwarnings("error")
def test_non_finite_closing_anchor_has_typed_failure_without_warning(
    value: float,
) -> None:
    frame = InternalCoordinateFrame(Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(1, 0, 0))
    with pytest.raises(GeometryPlacementError, match="finite"):
        frame.bond_closure_candidates(
            bond_length=1.0,
            bond_angle_degrees=110.0,
            preferred_dihedral_degrees=0.0,
            closing_anchor=Vec3(value, 1.0, 0.0),
            closing_bond_length=1.0,
        )


def test_tangent_spheres_have_finite_single_solution() -> None:
    frame = InternalCoordinateFrame(Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(1, 0, 0))
    candidates = frame.bond_closure_candidates(
        bond_length=1.0,
        bond_angle_degrees=110.0,
        preferred_dihedral_degrees=0.0,
        closing_anchor=Vec3(3, 0, 0),
        closing_bond_length=1.0,
    )
    assert candidates == (Vec3(2, 0, 0),)
