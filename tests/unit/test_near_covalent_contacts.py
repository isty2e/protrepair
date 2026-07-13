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
from protrepair.diagnostics.clash_pair_generation import (
    ContactDomain,
    PreparedAtomSitePairIndex,
)
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    detect_clashes,
    detect_clashes_from_context,
    prepare_clash_detection_basis,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.near_covalent import (
    NearCovalentContact,
    NearCovalentContactPolicy,
    detect_near_covalent_contacts,
    detect_near_covalent_contacts_from_context,
    prepare_near_covalent_contact_basis,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import AtomGeometry
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


def test_context_projection_rejects_mismatched_structure_constitution() -> None:
    """Prepared geometry and canonical topology must share one constitution."""

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

    with pytest.raises(ValueError, match="immutable constitution"):
        detect_near_covalent_contacts_from_context(
            mismatched_structure,
            context,
        )


def test_context_projection_rejects_same_address_replacement_constitution() -> None:
    """Address equality must not admit chemistry facts from another snapshot."""

    source_structure = _polymer_ligand_contact_structure()
    replacement_structure = _polymer_ligand_contact_structure(ligand_element="N")
    assert source_structure.constitution is not replacement_structure.constitution
    assert (
        source_structure.constitution.address_space_key
        == replacement_structure.constitution.address_space_key
    )
    context = prepare_clash_detection_context(
        source_structure,
        component_library=build_default_component_library(),
        policy=ClashPolicy(include_ligands=True),
    )

    with pytest.raises(ValueError, match="immutable constitution"):
        detect_near_covalent_contacts_from_context(
            replacement_structure,
            context,
        )


def test_context_projection_rejects_replaced_geometry() -> None:
    """Coordinate-bound contexts must not be paired with a newer geometry."""

    source_structure = _polymer_ligand_contact_structure()
    ligand_id = ResidueId("A", 401)
    moved_ligand_geometry = source_structure.residue_geometry(
        source_structure.constitution.residue_index(ligand_id)
    ).with_atom_geometry(
        "C5N",
        AtomGeometry(position=Vec3(10.0, 0.0, 0.0)),
    )
    moved_structure = source_structure.with_updated_residue_geometries(
        ((ligand_id, moved_ligand_geometry),)
    )
    context = prepare_clash_detection_context(
        source_structure,
        component_library=build_default_component_library(),
        policy=ClashPolicy(include_ligands=True),
    )

    with pytest.raises(ValueError, match="created by a preparation factory"):
        replace(context, geometry=moved_structure.geometry)
    with pytest.raises(ValueError, match="original immutable geometry"):
        detect_near_covalent_contacts_from_context(
            moved_structure,
            context,
        )


def test_prepared_pair_index_requires_near_covalent_basis() -> None:
    """Shared spatial frames require the basis that owns their cell contract."""

    structure = _polymer_ligand_contact_structure()
    context = prepare_clash_detection_context(
        structure,
        component_library=build_default_component_library(),
        policy=ClashPolicy(include_ligands=True),
    )
    prepared_pair_index = PreparedAtomSitePairIndex(
        atom_sites=context.atom_sites,
        focus_residue_ids=None,
    )

    with pytest.raises(ValueError, match="require a near-covalent basis"):
        detect_near_covalent_contacts_from_context(
            structure,
            context,
            prepared_pair_index=prepared_pair_index,
        )


def test_prepared_near_covalent_basis_cannot_be_reassembled_with_replace() -> None:
    """Cached contact facets must only be assembled by their preparation factory."""

    structure = _polymer_ligand_contact_structure()
    clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=build_default_component_library(),
        policy=ClashPolicy(include_ligands=True),
    )
    near_covalent_basis = prepare_near_covalent_contact_basis(
        structure,
        pair_policy=clash_basis.policy,
    )

    with pytest.raises(ValueError, match="created by a preparation factory"):
        replace(near_covalent_basis, candidate_cell_size_angstrom=0.5)


def test_prepared_pair_index_preserves_independent_contact_metrics() -> None:
    """Shared spatial preparation must not merge metric-specific semantics."""

    structure = _polymer_ligand_contact_structure()
    component_library = build_default_component_library()
    clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
        policy=ClashPolicy(include_ligands=True),
    )
    near_covalent_basis = prepare_near_covalent_contact_basis(
        structure,
        pair_policy=clash_basis.policy,
    )
    shared_cell_size_angstrom = max(
        clash_basis.candidate_cell_size_angstrom,
        near_covalent_basis.candidate_cell_size_angstrom,
    )
    context = clash_basis.bind_context(
        structure,
        candidate_cell_size_angstrom=shared_cell_size_angstrom,
    )
    focus_residue_ids = frozenset({ResidueId("A", 1)})
    prepared_pair_index = PreparedAtomSitePairIndex(
        atom_sites=context.atom_sites,
        focus_residue_ids=focus_residue_ids,
    )

    expected_clashes = detect_clashes_from_context(
        context,
        focus_residue_ids=focus_residue_ids,
    )
    expected_near_covalent_contacts = detect_near_covalent_contacts_from_context(
        structure,
        context,
        focus_residue_ids=focus_residue_ids,
        basis=near_covalent_basis,
    )

    assert (
        detect_clashes_from_context(
            context,
            focus_residue_ids=focus_residue_ids,
            prepared_pair_index=prepared_pair_index,
        )
        == expected_clashes
    )
    assert (
        detect_near_covalent_contacts_from_context(
            structure,
            context,
            focus_residue_ids=focus_residue_ids,
            basis=near_covalent_basis,
            prepared_pair_index=prepared_pair_index,
        )
        == expected_near_covalent_contacts
    )


def test_near_covalent_basis_ignores_clash_metric_thresholds() -> None:
    """Shared pair scope must not couple near-covalent facts to clash cutoffs."""

    structure = _polymer_ligand_contact_structure()
    component_library = build_default_component_library()
    source_clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
        policy=ClashPolicy(
            include_ligands=True,
            heavy_overlap_tolerance_angstrom=10.0,
        ),
    )
    near_covalent_basis = prepare_near_covalent_contact_basis(
        structure,
        pair_policy=source_clash_basis.policy,
    )
    independent_clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
        policy=ClashPolicy(
            include_ligands=True,
            heavy_overlap_tolerance_angstrom=0.0,
        ),
    )
    context = independent_clash_basis.bind_context(
        structure,
        candidate_cell_size_angstrom=max(
            independent_clash_basis.candidate_cell_size_angstrom,
            near_covalent_basis.candidate_cell_size_angstrom,
        ),
    )

    assert near_covalent_basis.bind_context(context).pair_policy == (
        independent_clash_basis.policy.as_contact_pair_policy()
    )


def test_prepared_pair_index_rejects_another_focus_or_coordinate_frame() -> None:
    """Spatial indexes must remain bound to one atom-site frame and focus."""

    structure = _polymer_ligand_contact_structure()
    component_library = build_default_component_library()
    clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
        policy=ClashPolicy(include_ligands=True),
    )
    first_context = clash_basis.bind_context(structure)
    second_context = clash_basis.bind_context(structure)
    focus_residue_ids = frozenset({ResidueId("A", 1)})
    prepared_pair_index = PreparedAtomSitePairIndex(
        atom_sites=first_context.atom_sites,
        focus_residue_ids=focus_residue_ids,
    )

    with pytest.raises(ValueError, match="matching focus"):
        detect_clashes_from_context(
            first_context,
            focus_residue_ids=frozenset({ResidueId("A", 401)}),
            prepared_pair_index=prepared_pair_index,
        )
    with pytest.raises(ValueError, match="original atom-site frame"):
        detect_clashes_from_context(
            second_context,
            focus_residue_ids=focus_residue_ids,
            prepared_pair_index=prepared_pair_index,
        )


def test_near_covalent_basis_rejects_replaced_topology() -> None:
    """Cached topology exclusions must not cross a topology replacement."""

    structure = _polymer_ligand_contact_structure()
    component_library = build_default_component_library()
    clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
        policy=ClashPolicy(include_ligands=True),
    )
    near_covalent_basis = prepare_near_covalent_contact_basis(
        structure,
        pair_policy=clash_basis.policy,
    )
    topology_updated_structure = _with_topology_relationship(
        structure,
        BondRelationshipType.COVALENT,
    )
    replacement_clash_basis = prepare_clash_detection_basis(
        topology_updated_structure,
        component_library=component_library,
        policy=ClashPolicy(include_ligands=True),
    )
    context = replacement_clash_basis.bind_context(
        topology_updated_structure,
        candidate_cell_size_angstrom=max(
            replacement_clash_basis.candidate_cell_size_angstrom,
            near_covalent_basis.candidate_cell_size_angstrom,
        ),
    )

    assert not clash_basis.is_compatible_with(topology_updated_structure)
    with pytest.raises(ValueError, match="immutable constitution and topology"):
        clash_basis.bind_context(topology_updated_structure)
    with pytest.raises(ValueError, match="original immutable topology"):
        detect_near_covalent_contacts_from_context(
            topology_updated_structure,
            context,
            basis=near_covalent_basis,
        )
    with pytest.raises(ValueError, match="clash context.*immutable topology"):
        near_covalent_basis.bind_context(context)

    updated_near_covalent_basis = prepare_near_covalent_contact_basis(
        topology_updated_structure,
        pair_policy=replacement_clash_basis.policy,
    )
    assert (
        detect_near_covalent_contacts_from_context(
            topology_updated_structure,
            context,
            basis=updated_near_covalent_basis,
        )
        == ()
    )


def test_near_covalent_basis_rejects_context_from_replaced_constitution() -> None:
    """Cached chemistry facts must not mix with a same-address context."""

    source_structure = _polymer_ligand_contact_structure()
    replacement_structure = _polymer_ligand_contact_structure(ligand_element="N")
    assert source_structure.constitution is not replacement_structure.constitution
    assert (
        source_structure.constitution.address_space_key
        == replacement_structure.constitution.address_space_key
    )
    component_library = build_default_component_library()
    source_clash_basis = prepare_clash_detection_basis(
        source_structure,
        component_library=component_library,
        policy=ClashPolicy(include_ligands=True),
    )
    near_covalent_basis = prepare_near_covalent_contact_basis(
        source_structure,
        pair_policy=source_clash_basis.policy,
    )
    replacement_clash_basis = prepare_clash_detection_basis(
        replacement_structure,
        component_library=component_library,
        policy=ClashPolicy(include_ligands=True),
    )
    replacement_context = replacement_clash_basis.bind_context(
        replacement_structure,
        candidate_cell_size_angstrom=max(
            replacement_clash_basis.candidate_cell_size_angstrom,
            near_covalent_basis.candidate_cell_size_angstrom,
        ),
    )

    with pytest.raises(ValueError, match="original immutable constitution"):
        near_covalent_basis.bind_context(replacement_context)
    with pytest.raises(ValueError, match="created by a preparation factory"):
        replace(
            near_covalent_basis,
            constitution=replacement_structure.constitution,
            topology=replacement_structure.topology,
        )


def _polymer_ligand_contact_structure(
    *,
    ligand_element: str = "C",
) -> ProteinStructure:
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
                atoms=(
                    atom_payload("C5N", ligand_element, Vec3(1.5, 0.0, 0.0)),
                ),
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
