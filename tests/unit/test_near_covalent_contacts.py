"""Unit tests for near-covalent contact classification."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import UnknownElementRadiusError
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics.clash_pair_generation import ContactDomain
from protrepair.diagnostics.clashes import ClashPolicy, detect_clashes
from protrepair.diagnostics.near_covalent import (
    NearCovalentContact,
    NearCovalentContactPolicy,
    detect_near_covalent_contacts,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.structure.labels import ResidueId


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf")),
)
def test_near_covalent_policy_rejects_non_finite_minimum_overlap(
    value: float,
) -> None:
    """Near-covalent overlap policy must remain finite."""

    with pytest.raises(ValueError, match="minimum_overlap_angstrom must be finite"):
        NearCovalentContactPolicy(minimum_overlap_angstrom=value)


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf")),
)
def test_near_covalent_policy_rejects_non_finite_covalent_margin(
    value: float,
) -> None:
    """Near-covalent margin policy must remain finite."""

    with pytest.raises(
        ValueError,
        match="covalent_distance_margin_angstrom must be finite",
    ):
        NearCovalentContactPolicy(covalent_distance_margin_angstrom=value)


def test_near_covalent_contact_rejects_non_finite_measurements() -> None:
    """Materialized near-covalent facts must contain finite measurements."""

    contact = NearCovalentContact(
        left_residue_id=ResidueId("A", 1),
        left_component_id="ALA",
        left_atom_name="CB",
        left_domain=ContactDomain.POLYMER,
        right_residue_id=ResidueId("B", 1),
        right_component_id="ALA",
        right_atom_name="CB",
        right_domain=ContactDomain.POLYMER,
        distance_angstrom=1.0,
        covalent_distance_cutoff_angstrom=1.5,
        overlap_angstrom=0.5,
    )

    with pytest.raises(ValueError, match="finite distance"):
        replace(contact, distance_angstrom=float("nan"))
    with pytest.raises(ValueError, match="finite cutoff"):
        replace(contact, covalent_distance_cutoff_angstrom=float("inf"))
    with pytest.raises(ValueError, match="finite overlap"):
        replace(contact, overlap_angstrom=float("nan"))


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
    with pytest.raises(UnknownElementRadiusError) as error_info:
        detect_near_covalent_contacts(
            structure,
            component_library=build_standard_component_library(),
        )

    message = str(error_info.value)
    assert "near-covalent contact detection has unresolved covalent radius" in message
    assert message.count("XX") == 1
    assert "C1" in message


def test_near_covalent_contacts_do_not_depend_on_vdw_clash_output() -> None:
    """Covalent-radius proximity should be detected even without a vdW clash."""

    left_id = ResidueId("A", 1)
    right_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="FRX",
                        residue_id=left_id,
                        atoms=(atom_payload("FR1", "FR", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="FRY",
                        residue_id=right_id,
                        atoms=(atom_payload("FR1", "FR", Vec3(4.40, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    component_library = build_standard_component_library()

    clash_report = detect_clashes(
        structure,
        component_library=component_library,
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=10.0),
    )
    contacts = detect_near_covalent_contacts(
        structure,
        component_library=component_library,
        pair_policy=ClashPolicy(heavy_overlap_tolerance_angstrom=10.0),
    )

    assert not clash_report.clashes
    assert len(contacts) == 1
    assert contacts[0].distance_angstrom == pytest.approx(4.40)
    assert contacts[0].covalent_distance_cutoff_angstrom == pytest.approx(5.65)
    assert contacts[0].overlap_angstrom == pytest.approx(1.25)


def test_near_covalent_contacts_keep_retained_ions_out_by_default() -> None:
    """Metal-like retained sites should not enter polymer-only diagnostics."""

    polymer_id = ResidueId("A", 1)
    ligand_id = ResidueId("Z", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=polymer_id,
                        atoms=(atom_payload("O", "O", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="ZN",
                residue_id=ligand_id,
                atoms=(atom_payload("ZN", "ZN", Vec3(2.10, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    component_library = build_standard_component_library()

    default_contacts = detect_near_covalent_contacts(
        structure,
        component_library=component_library,
    )
    ligand_contacts = detect_near_covalent_contacts(
        structure,
        component_library=component_library,
        pair_policy=ClashPolicy(include_ligands=True),
    )

    assert default_contacts == ()
    assert len(ligand_contacts) == 1
    assert {
        ligand_contacts[0].left_domain,
        ligand_contacts[0].right_domain,
    } == {ContactDomain.POLYMER, ContactDomain.RETAINED_NON_POLYMER}
