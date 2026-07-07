"""Unit tests for near-covalent contact classification."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import UnknownElementRadiusError
from protrepair.diagnostics.clash_pair_generation import ContactDomain
from protrepair.diagnostics.clashes import StericClash
from protrepair.diagnostics.near_covalent import detect_near_covalent_contacts
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.structure.labels import ResidueId


def test_near_covalent_contacts_report_unknown_radii_once() -> None:
    """Near-covalent classification should aggregate unknown covalent radii."""

    left_id = ResidueId("A", 1)
    right_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNX",
                        residue_id=left_id,
                        atoms=(atom_payload("X1", "XX", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="UNY",
                        residue_id=right_id,
                        atoms=(atom_payload("Y1", "C1", Vec3(1.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    clash = StericClash(
        left_residue_id=left_id,
        left_component_id="UNX",
        left_atom_name="X1",
        left_domain=ContactDomain.POLYMER,
        right_residue_id=right_id,
        right_component_id="UNY",
        right_atom_name="Y1",
        right_domain=ContactDomain.POLYMER,
        distance_angstrom=1.0,
        allowed_distance_angstrom=2.5,
        overlap_angstrom=1.5,
    )

    with pytest.raises(UnknownElementRadiusError) as error_info:
        detect_near_covalent_contacts(structure, clashes=(clash, clash))

    message = str(error_info.value)
    assert "near-covalent contact detection has unresolved covalent radius" in message
    assert message.count("XX") == 1
    assert "C1" in message
