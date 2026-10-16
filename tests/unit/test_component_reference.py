"""CCD comparisons distinguish derived roundoff from changed chemical data."""

from dataclasses import replace

import pytest
from tests.support.component_reference import assert_component_reference_matches

from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)


@pytest.mark.parametrize("kind", ("bond", "angle"))
@pytest.mark.parametrize("roundoff_only", (False, True))
def test_reference_comparison_bounds_derived_numeric_differences(kind, roundoff_only):
    expected = build_bundled_retained_non_polymer_registry().get("NAG")
    assert expected is not None
    delta = 1e-13 if roundoff_only else 1e-3
    if kind == "bond":
        first, *rest = expected.bond_targets
        actual = replace(
            expected,
            bond_targets=(
                replace(
                    first,
                    target_distance_angstrom=first.target_distance_angstrom + delta,
                ),
                *rest,
            ),
        )
    else:
        first, *rest = expected.angle_targets
        actual = replace(
            expected,
            angle_targets=(
                replace(first, target_angle_degrees=first.target_angle_degrees + delta),
                *rest,
            ),
        )
    if roundoff_only:
        assert_component_reference_matches(actual, expected)
    else:
        with pytest.raises(AssertionError):
            assert_component_reference_matches(actual, expected)


@pytest.mark.parametrize("kind", ("charge", "aromaticity"))
def test_reference_comparison_keeps_non_numeric_chemistry_exact(kind):
    expected = build_bundled_retained_non_polymer_registry().get("NAG")
    assert expected is not None
    if kind == "charge":
        first, *rest = expected.atoms
        actual = replace(
            expected,
            atoms=(replace(first, formal_charge=first.formal_charge + 1), *rest),
        )
    else:
        first, *rest = expected.bonds
        actual = replace(
            expected, bonds=(replace(first, aromatic=not first.aromatic), *rest)
        )
    with pytest.raises(AssertionError):
        assert_component_reference_matches(actual, expected)
