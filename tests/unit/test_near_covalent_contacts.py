"""Unit tests for near-covalent contact classification."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import (
    UnknownElementRadiusError,
    build_default_component_library,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics.clash_pair_generation import ContactDomain
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    detect_clashes,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.near_covalent import (
    NearCovalentContact,
    NearCovalentContactPolicy,
    detect_near_covalent_contacts,
    detect_near_covalent_contacts_from_context,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


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


def test_near_covalent_contacts_honor_same_residue_policy() -> None:
    """Same-residue nonbonded heavy pairs should follow their explicit policy."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNL",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("C2", "C", Vec3(1.5, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    component_library = build_standard_component_library()

    ignored = detect_near_covalent_contacts(
        structure,
        component_library=component_library,
    )
    included = detect_near_covalent_contacts(
        structure,
        component_library=component_library,
        policy=NearCovalentContactPolicy(ignore_same_residue=False),
    )

    assert ignored == ()
    assert len(included) == 1
    assert {included[0].left_atom_name, included[0].right_atom_name} == {"C1", "C2"}


def test_near_covalent_contacts_still_ignore_template_bonded_pair() -> None:
    """Enabling same-residue diagnostics must not report a known template bond."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    contacts = detect_near_covalent_contacts(
        structure,
        component_library=build_standard_component_library(),
        policy=NearCovalentContactPolicy(ignore_same_residue=False),
    )

    assert contacts == ()


@pytest.mark.parametrize(
    "relationship_type",
    (
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.METAL_COORDINATION,
    ),
)
def test_near_covalent_contacts_exclude_expected_topology_relationships(
    relationship_type: BondRelationshipType,
) -> None:
    """Canonical close-contact relationships should not become pathologies."""

    structure = _polymer_ligand_contact_structure()
    structure = _with_topology_relationship(structure, relationship_type)

    contacts = detect_near_covalent_contacts(
        structure,
        component_library=build_default_component_library(),
        pair_policy=ClashPolicy(include_ligands=True),
    )

    assert contacts == ()


def test_near_covalent_contacts_do_not_hide_unknown_topology_relationship() -> None:
    """UNKNOWN topology must not silently legitimize an unexplained close pair."""

    structure = _with_topology_relationship(
        _polymer_ligand_contact_structure(),
        BondRelationshipType.UNKNOWN,
    )

    contacts = detect_near_covalent_contacts(
        structure,
        component_library=build_default_component_library(),
        pair_policy=ClashPolicy(include_ligands=True),
    )

    assert len(contacts) == 1


def test_6nbb_zinc_coordination_is_not_near_covalent_pathology() -> None:
    """Source-explicit 6NBB zinc coordination should remain topology truth."""

    structure = read_structure(Path("tests/fixtures/corpus/pdb6nbb.ent"))
    cysteine_id = ResidueId("A", 46)
    zinc_id = ResidueId("A", 402)
    coordination_bond = structure.topology.bond_between(
        structure.constitution.atom_index(AtomRef(cysteine_id, "SG")),
        structure.constitution.atom_index(AtomRef(zinc_id, "ZN")),
    )

    assert coordination_bond is not None
    assert (
        coordination_bond.relationship_type
        is BondRelationshipType.METAL_COORDINATION
    )

    contacts = detect_near_covalent_contacts(
        structure,
        component_library=build_default_component_library(),
        focus_residue_ids=frozenset((cysteine_id,)),
        pair_policy=ClashPolicy(include_ligands=True),
    )

    assert not any(
        {
            (contact.left_residue_id, contact.left_atom_name),
            (contact.right_residue_id, contact.right_atom_name),
        }
        == {(cysteine_id, "SG"), (zinc_id, "ZN")}
        for contact in contacts
    )


def test_context_projection_rejects_mismatched_structure_address_space() -> None:
    """Prepared geometry and canonical topology must share one address space."""

    source_structure = _polymer_ligand_contact_structure()
    mismatched_structure = build_structure(
        chains=(
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("B", 1),
                        atoms=(atom_payload("CB", "C", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    component_library = build_default_component_library()
    context = prepare_clash_detection_context(
        source_structure,
        component_library=component_library,
        policy=ClashPolicy(include_ligands=True),
    )

    with pytest.raises(ValueError, match="matching structure address space"):
        detect_near_covalent_contacts_from_context(
            mismatched_structure,
            context,
        )


def _polymer_ligand_contact_structure() -> ProteinStructure:
    """Return one unexplained polymer-ligand near-covalent contact."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(atom_payload("CB", "C", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="NAD",
                residue_id=ResidueId("A", 401),
                atoms=(atom_payload("C5N", "C", Vec3(1.5, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _with_topology_relationship(
    structure: ProteinStructure,
    relationship_type: BondRelationshipType,
) -> ProteinStructure:
    """Return the contact fixture with one canonical topology relationship."""

    topology = StructureTopology(
        constitution=structure.constitution,
        atom_topologies=structure.topology.atom_topologies,
        bonds=(
            TopologyBond(
                atom_index_1=structure.constitution.atom_index(
                    AtomRef(ResidueId("A", 1), "CB")
                ),
                atom_index_2=structure.constitution.atom_index(
                    AtomRef(ResidueId("A", 401), "C5N")
                ),
                relationship_type=relationship_type,
            ),
        ),
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
