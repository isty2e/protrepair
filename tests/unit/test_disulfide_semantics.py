"""Topology-authoritative disulfide semantics across structure consumers."""

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics import (
    ClashPolicy,
    bind_clash_detection_context,
    prepare_clash_detection_basis,
)
from protrepair.diagnostics.clashes import StericClash
from protrepair.diagnostics.near_covalent import detect_near_covalent_contacts
from protrepair.geometry import Vec3
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state.hydrogen_expectation import (
    derive_structure_hydrogen_expectation_model,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.disulfide import (
    disulfide_atom_ref_pairs,
    disulfide_bonded_cysteine_residue_ids,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)


@pytest.mark.parametrize(
    "relationship_type",
    (BondRelationshipType.COVALENT, BondRelationshipType.DISULFIDE),
)
def test_covalent_like_cysteine_sg_topology_defines_disulfide_chemistry(
    relationship_type: BondRelationshipType,
) -> None:
    """Generic and specialized covalent CYS-SG bonds share disulfide semantics."""

    structure = sg_pair_structure(
        distance_angstrom=8.0,
        relationship_type=relationship_type,
    )
    left_id = ResidueId("A", 1, "A")
    right_id = ResidueId("B", 1)

    assert disulfide_atom_ref_pairs(structure) == frozenset(
        {
            (
                AtomRef(left_id, "SG"),
                AtomRef(right_id, "SG"),
            )
        }
    )
    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset(
        {left_id, right_id}
    )


@pytest.mark.parametrize(
    "relationship_type",
    (None, BondRelationshipType.UNKNOWN, BondRelationshipType.METAL_COORDINATION),
)
def test_close_cysteine_sg_geometry_does_not_create_disulfide_truth(
    relationship_type: BondRelationshipType | None,
) -> None:
    """Proximity and noncovalent topology must not define a CYS microstate."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=relationship_type,
    )

    assert disulfide_atom_ref_pairs(structure) == frozenset()
    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset()


@pytest.mark.parametrize("provenance", tuple(BondProvenance))
def test_disulfide_chemistry_does_not_depend_on_bond_provenance(
    provenance: BondProvenance,
) -> None:
    """Physical relationship and evidence provenance must remain orthogonal."""

    structure = sg_pair_structure(
        distance_angstrom=8.0,
        relationship_type=BondRelationshipType.DISULFIDE,
        provenance=provenance,
    )

    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset(
        {ResidueId("A", 1, "A"), ResidueId("B", 1)}
    )


def test_atom_name_alone_does_not_make_non_cysteine_sg_a_disulfide() -> None:
    """A covalent SG-SG relationship requires CYS identity at both endpoints."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=BondRelationshipType.DISULFIDE,
        right_component_id="SER",
    )

    assert disulfide_atom_ref_pairs(structure) == frozenset()
    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset()


def test_clash_basis_rebinds_disulfide_truth_from_current_topology() -> None:
    """Reusable clash bases must not cache topology from their source structure."""

    unbonded = sg_pair_structure(distance_angstrom=2.0)
    bonded = with_sg_relationship(unbonded, BondRelationshipType.DISULFIDE)
    component_library = build_default_component_library()
    basis = prepare_clash_detection_basis(
        unbonded,
        component_library=component_library,
    )

    unbonded_report = bind_clash_detection_context(
        unbonded,
        basis=basis,
    ).detect_clashes()
    bonded_report = bind_clash_detection_context(
        bonded,
        basis=basis,
    ).detect_clashes()

    assert has_sg_sg_clash(unbonded_report.clashes)
    assert not has_sg_sg_clash(bonded_report.clashes)


@pytest.mark.parametrize(
    ("relationship_type", "expect_near_covalent", "expect_disulfide"),
    (
        (None, True, False),
        (BondRelationshipType.UNKNOWN, True, False),
        (BondRelationshipType.METAL_COORDINATION, False, False),
        (BondRelationshipType.COVALENT, False, True),
        (BondRelationshipType.DISULFIDE, False, True),
    ),
)
def test_near_covalent_contact_and_disulfide_chemistry_remain_orthogonal(
    relationship_type: BondRelationshipType | None,
    expect_near_covalent: bool,
    expect_disulfide: bool,
) -> None:
    """Expected close contact and CYS microstate are independent projections."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=relationship_type,
    )
    contacts = detect_near_covalent_contacts(
        structure,
        component_library=build_default_component_library(),
        pair_policy=ClashPolicy(),
    )

    assert (
        any(
            {contact.left_atom_name, contact.right_atom_name} == {"SG"}
            for contact in contacts
        )
        is expect_near_covalent
    )
    assert bool(disulfide_bonded_cysteine_residue_ids(structure)) is expect_disulfide


@pytest.mark.parametrize(
    ("relationship_type", "expect_hg"),
    (
        (None, True),
        (BondRelationshipType.UNKNOWN, True),
        (BondRelationshipType.METAL_COORDINATION, True),
        (BondRelationshipType.COVALENT, False),
        (BondRelationshipType.DISULFIDE, False),
    ),
)
def test_hydrogen_expectation_consumes_topology_disulfide_semantics(
    relationship_type: BondRelationshipType | None,
    expect_hg: bool,
) -> None:
    """CYS HG expectation must not depend on SG distance alone."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=relationship_type,
    )
    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=build_default_component_library(),
    )

    for residue_id in (ResidueId("A", 1, "A"), ResidueId("B", 1)):
        expected_names = model.expected_hydrogen_atom_names_by_residue[residue_id]
        assert ("HG" in expected_names) is expect_hg


def test_retained_cysteine_disulfide_uses_shared_hydrogen_semantics() -> None:
    """Retained CYS expectation and completion must honor canonical disulfides."""

    left_id = ResidueId("L", 1)
    right_id = ResidueId("R", 1)
    structure = retained_cys_pair_structure(left_id, right_id)
    component_library = build_default_component_library()

    expectation = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=component_library,
    )
    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
    )

    for residue_id in (left_id, right_id):
        assert "HG" not in expectation.expected_hydrogen_atom_names_by_residue[
            residue_id
        ]
        residue = result.structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None and not residue.has_atom_site("HG")

    bond = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(AtomRef(left_id, "SG")),
        result.structure.constitution.atom_index(AtomRef(right_id, "SG")),
    )
    assert bond is not None
    assert bond.relationship_type is BondRelationshipType.DISULFIDE


@pytest.mark.parametrize("chemistry_mode", ("override", "fallback"))
def test_retained_disulfide_excludes_sg_h_across_chemistry_sources(
    chemistry_mode: str,
) -> None:
    """Explicit evidence and fallback must not protonate a bonded CYS SG."""

    left_id = ResidueId("L", 1)
    right_id = ResidueId("R", 1)
    structure = retained_sulfide_pair_structure(left_id, right_id)
    chemistry_evidence = (
        tuple(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="CS",
                heavy_atom_names=("C1", "SG"),
            ).to_evidence()
            for residue_id in (left_id, right_id)
        )
        if chemistry_mode == "override"
        else ()
    )
    component_library = ComponentLibrary(templates={})

    expectation = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=chemistry_evidence,
    )
    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
        chemistry_evidence=chemistry_evidence,
    )

    for residue_id in (left_id, right_id):
        resolution = expectation.resolution_for_retained_non_polymer(residue_id)
        assert all(
            "SG" not in (bond.atom_name_1, bond.atom_name_2)
            for bond in resolution.hydrogen_bond_definitions
        )
        assert not has_hydrogen_topology_bond_to(
            result.structure,
            AtomRef(residue_id, "SG"),
        )


def sg_pair_structure(
    *,
    distance_angstrom: float,
    relationship_type: BondRelationshipType | None = None,
    left_component_id: str = "CYS",
    right_component_id: str = "CYS",
    provenance: BondProvenance = BondProvenance.SOURCE_EXPLICIT,
) -> ProteinStructure:
    """Return a cross-chain CYS pair with one insertion-coded endpoint."""

    left_id = ResidueId("A", 1, "A")
    right_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id=left_component_id,
                        residue_id=left_id,
                        atoms=(atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id=right_component_id,
                        residue_id=right_id,
                        atoms=(
                            atom_payload(
                                "SG",
                                "S",
                                Vec3(distance_angstrom, 0.0, 0.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="topology-authoritative-disulfide",
    )
    if relationship_type is None:
        return structure

    return with_sg_relationship(
        structure,
        relationship_type,
        provenance=provenance,
    )


def retained_cys_pair_structure(
    left_id: ResidueId,
    right_id: ResidueId,
) -> ProteinStructure:
    """Return two retained CYS components with an explicit SG-SG bond."""

    structure = build_structure(
        chains=(),
        ligands=(
            retained_cys_payload(left_id, x_offset=0.0),
            retained_cys_payload(right_id, x_offset=8.0),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-cys-disulfide",
    )
    return with_sg_relationship(
        structure,
        BondRelationshipType.DISULFIDE,
        left_atom_ref=AtomRef(left_id, "SG"),
        right_atom_ref=AtomRef(right_id, "SG"),
    )


def retained_sulfide_pair_structure(
    left_id: ResidueId,
    right_id: ResidueId,
) -> ProteinStructure:
    """Return retained CYS-labelled methylthiol fragments joined at SG."""

    structure = build_structure(
        chains=(),
        ligands=tuple(
            residue_payload(
                component_id="CYS",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(x_offset, 0.0, 0.0)),
                    atom_payload("SG", "S", Vec3(x_offset + 1.8, 0.0, 0.0)),
                ),
                is_hetero=True,
            )
            for residue_id, x_offset in ((left_id, 0.0), (right_id, 8.0))
        ),
        source_format=FileFormat.PDB,
        source_name="retained-cys-sulfide",
    )
    return with_sg_relationship(
        structure,
        BondRelationshipType.DISULFIDE,
        left_atom_ref=AtomRef(left_id, "SG"),
        right_atom_ref=AtomRef(right_id, "SG"),
    )


def retained_cys_payload(
    residue_id: ResidueId,
    *,
    x_offset: float,
) -> CanonicalResiduePayload:
    """Return one retained CYS payload with nondegenerate heavy geometry."""

    return residue_payload(
        component_id="CYS",
        residue_id=residue_id,
        atoms=(
            atom_payload("N", "N", Vec3(x_offset - 1.2, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(x_offset, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(x_offset + 1.3, 0.3, 0.0)),
            atom_payload("O", "O", Vec3(x_offset + 2.0, 1.1, 0.0)),
            atom_payload("CB", "C", Vec3(x_offset, 1.4, 0.5)),
            atom_payload("SG", "S", Vec3(x_offset, 2.8, 0.7)),
        ),
        is_hetero=True,
    )


def with_sg_relationship(
    structure: ProteinStructure,
    relationship_type: BondRelationshipType,
    *,
    provenance: BondProvenance = BondProvenance.SOURCE_EXPLICIT,
    left_atom_ref: AtomRef | None = None,
    right_atom_ref: AtomRef | None = None,
) -> ProteinStructure:
    """Return the structure with one canonical relationship between CYS SG atoms."""

    left_index = structure.constitution.atom_index(
        left_atom_ref or AtomRef(ResidueId("A", 1, "A"), "SG")
    )
    right_index = structure.constitution.atom_index(
        right_atom_ref or AtomRef(ResidueId("B", 1), "SG")
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=left_index,
                    atom_index_2=right_index,
                    relationship_type=relationship_type,
                    provenance=provenance,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def has_sg_sg_clash(clashes: tuple[StericClash, ...]) -> bool:
    """Return whether one clash tuple contains a CYS SG-SG pair."""

    return any(
        clash.left_atom_name == "SG" and clash.right_atom_name == "SG"
        for clash in clashes
    )


def has_hydrogen_topology_bond_to(
    structure: ProteinStructure,
    atom_ref: AtomRef,
) -> bool:
    """Return whether topology bonds one hydrogen to the requested atom."""

    atom_index = structure.constitution.atom_index(atom_ref)
    return any(
        structure.constitution.atom_site_at(other_index).is_hydrogen()
        for bond in structure.topology.bonds
        if bond.involves(atom_index)
        for other_index in bond.endpoint_pair()
        if other_index != atom_index
    )
