"""Angle identity and matching independent of geometric target values."""

from itertools import permutations

import pytest

from protrepair.chemistry import build_default_restraint_library
from protrepair.chemistry.restraint.template import AngleRestraintTarget


@pytest.mark.parametrize("names", tuple(permutations((" n ", "Ca", " c "))))
def test_angle_identity_preserves_center_and_normalizes_endpoints(names):
    target = AngleRestraintTarget(" c ", "ca", " n ", 110.0)
    other_value = AngleRestraintTarget("N", "CA", "C", 120.0, 2.0)
    assert target.canonical_triplet() == other_value.canonical_triplet()
    assert target.canonical_triplet() == ("CA", "C", "N")
    left, center, right = names
    assert target.matches(
        atom_name_1=left, center_atom_name=center, atom_name_2=right
    ) is (center.strip().upper() == "CA")


def test_dictionary_angle_matching_agrees_with_unordered_endpoint_contract():
    for template in build_default_restraint_library().templates.values():
        variants = (template, *template.sidechain_variants)
        for variant in variants:
            for target in variant.angle_targets:
                names = (
                    target.atom_name_1,
                    target.center_atom_name,
                    target.atom_name_2,
                )
                for left, center, right in permutations((*names, "ABSENT"), 3):
                    expected = center == target.center_atom_name and {left, right} == {
                        target.atom_name_1,
                        target.atom_name_2,
                    }
                    assert (
                        target.matches(
                            atom_name_1=f" {left.lower()} ",
                            center_atom_name=f" {center.lower()} ",
                            atom_name_2=f" {right.lower()} ",
                        )
                        is expected
                    )
                    assert (
                        target.canonical_triplet() == (center, *sorted((left, right)))
                    ) is expected
