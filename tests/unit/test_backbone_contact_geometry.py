"""Geometry required to waive a steric penetration during backbone rebuilding."""

from math import cos, radians, sin

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.structure import AtomRef, FileFormat, ResidueId
from protrepair.transformer.completion.backbone_environment import BackboneEnvironment

_RID = ResidueId("A", 1)


def _environment(component, atoms, extra=()):
    site = residue_payload(component_id=component, residue_id=_RID, atoms=tuple(atoms))
    s = build_structure(
        chains=(chain_payload("A", (site, *extra)),), source_format=FileFormat.PDB
    )
    return BackboneEnvironment(s, build_default_component_library())


@pytest.mark.parametrize(
    ("component", "parent", "h_name", "nominal"),
    (
        ("ALA", "CA", "HA", 1.09),
        ("ALA", "N", "H", 1.01),
        ("SER", "OG", "HG", 0.98),
        ("CYS", "SG", "HG", 1.34),
    ),
)
@pytest.mark.parametrize("element", ("H", "D", "T"))
def test_direct_bond_uses_the_parent_element_and_preserves_isotopes(
    component, parent, h_name, nominal, element
):
    ref = AtomRef(_RID, element + h_name[1:])
    heavy = AtomRef(_RID, parent)
    origin = Vec3(0, 0, 0)
    env = _environment(
        component,
        [
            atom_payload(parent, parent[0], origin),
            atom_payload(ref.atom_name, element, Vec3(20, 20, 20), occupancy=0),
        ],
    )
    for offset, clash in ((-0.27, True), (-0.24, False), (0.11, False), (0.14, True)):
        p = Vec3(nominal + offset, 0, 0)
        assert env.clash(ref, p, heavy, origin, {ref: p}) is clash
        assert env.clash(heavy, origin, ref, p, {ref: p}) is clash


@pytest.mark.parametrize(("angle", "clash"), ((65, True), (85, True), (100, False)))
def test_h_h_on_a_common_proposed_parent_needs_a_valid_angle(angle, clash):
    ca, ha, hb = (AtomRef(_RID, k) for k in ("CA", "HA2", "HA3"))
    p, q = (
        Vec3(0.9, 0, 0),
        Vec3(0.9 * cos(radians(angle)), 0.9 * sin(radians(angle)), 0),
    )
    env = _environment(
        "GLY",
        [
            atom_payload("CA", "C", Vec3(100, 100, 100), occupancy=0),
            atom_payload("HA2", "H", p),
            atom_payload("HA3", "H", q),
        ],
    )
    assert p.distance_to(q) < 1.5
    assert env.clash(ha, p, hb, q, {})
    assert env.clash(ha, p, hb, q, {ca: Vec3(0, 0, 0)}) is clash
    assert env.clash(hb, q, ha, p, {ca: Vec3(0, 0, 0)}) is clash


@pytest.mark.parametrize(
    "complication", ("none", "other_heavy", "different_alt", "charged_h")
)
def test_inferred_oxygen_h_needs_compatible_identity_and_one_plausible_parent(
    complication,
):
    og, cb, h = (AtomRef(_RID, k) for k in ("OG", "CB", "HG"))
    center, p, q = (
        Vec3(0, 0, 0),
        Vec3(0, 0.98, 0),
        Vec3(1.43 * cos(radians(15)), -1.43 * sin(radians(15)), 0),
    )
    extra = ()
    if complication == "other_heavy":
        extra = (
            residue_payload(
                component_id="ALA",
                residue_id=ResidueId("A", 2),
                atoms=(atom_payload("CB", "C", Vec3(0, 2.08, 0)),),
            ),
        )
    env = _environment(
        "SER",
        [
            atom_payload("OG", "O", center, altloc="A"),
            atom_payload("CB", "C", q),
            atom_payload(
                "HG",
                "H",
                p,
                altloc="B" if complication == "different_alt" else None,
                formal_charge=1 if complication == "charged_h" else None,
            ),
        ],
        extra,
    )
    assert env.hydrogen_parent(h) == og
    assert env.clash(h, p, cb, q, {}) is (complication != "none")
