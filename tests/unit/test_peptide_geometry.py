"""Peptide screening must include amide substituents, not only the backbone."""

from dataclasses import replace

import numpy as np
import pytest

from protrepair.diagnostics.peptide_geometry import PeptideJunctionGeometry
from protrepair.geometry import Vec3
from protrepair.geometry.rotation import AxisRotation


def _junction() -> PeptideJunctionGeometry:
    return PeptideJunctionGeometry(
        preceding_ca=Vec3(-0.65, 1.1, 0.0),
        carbonyl_c=Vec3(0.0, 0.0, 0.0),
        carbonyl_o=Vec3(-0.65, -1.1, 0.0),
        nitrogen=Vec3(1.33, 0.0, 0.0),
        following_ca=Vec3(2.05, -1.2, 0.0),
        nitrogen_substituents=(Vec3(2.05, 1.2, 0.0),),
    )


@pytest.mark.parametrize(
    "angle,expected",
    [(0.0, True), (20.0, True), (29.9, True), (30.1, False), (60.0, False)],
)
def test_amide_substituent_plane_is_screened(angle: float, expected: bool) -> None:
    junction = _junction()
    substituent = AxisRotation.from_points(
        junction.nitrogen, junction.following_ca
    ).rotate_point(
        junction.nitrogen_substituents[0],
        origin=junction.nitrogen,
        theta_radians=float(np.deg2rad(angle)),
    )
    assert replace(junction, nitrogen_substituents=()).is_plausible()
    assert (
        replace(junction, nitrogen_substituents=(substituent,)).is_plausible()
        is expected
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e308])
@pytest.mark.parametrize(
    "field", ["carbonyl_c", "following_ca", "nitrogen_substituents"]
)
def test_unusable_junction_geometry_is_rejected(field: str, value: float) -> None:
    point = Vec3(value, value, value)
    junction = _junction()
    if field == "nitrogen_substituents":
        junction = replace(junction, nitrogen_substituents=(point,))
    elif field == "carbonyl_c":
        junction = replace(junction, carbonyl_c=point)
    else:
        junction = replace(junction, following_ca=point)
    assert not junction.is_plausible()


def test_degenerate_amide_plane_is_rejected() -> None:
    junction = _junction()
    assert not replace(
        junction, nitrogen_substituents=(junction.nitrogen,)
    ).is_plausible()
