"""Compare frozen chemistry exactly, allowing roundoff in derived restraints."""

from dataclasses import replace

import pytest

from protrepair.chemistry.nonstandard.registry import NonstandardComponentRecord


def assert_component_reference_matches(
    actual: NonstandardComponentRecord, expected: NonstandardComponentRecord
) -> None:
    assert (
        replace(
            actual,
            bond_targets=expected.bond_targets,
            angle_targets=expected.angle_targets,
        )
        == expected
    )
    for found, reference in zip(
        actual.bond_targets, expected.bond_targets, strict=True
    ):
        assert found.target_distance_angstrom == pytest.approx(
            reference.target_distance_angstrom, rel=0.0, abs=1e-8
        )
        assert (
            replace(found, target_distance_angstrom=reference.target_distance_angstrom)
            == reference
        )
    for found, reference in zip(
        actual.angle_targets, expected.angle_targets, strict=True
    ):
        assert found.target_angle_degrees == pytest.approx(
            reference.target_angle_degrees, rel=0.0, abs=1e-6
        )
        assert (
            replace(found, target_angle_degrees=reference.target_angle_degrees)
            == reference
        )
