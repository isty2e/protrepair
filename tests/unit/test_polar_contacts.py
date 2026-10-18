"""Chemistry-backed contact classification, independent of residue membership."""

from dataclasses import replace

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.polar_contacts import PolarContactChemistry
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    detect_clashes_from_context,
    prepare_clash_detection_basis,
    prepare_clash_detection_context,
    prepare_projected_clash_detection_context,
    probable_hydrogen_bond,
)
from protrepair.diagnostics.near_covalent import (
    NearCovalentContactPolicy,
    detect_near_covalent_contacts,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.rotatable import (
    build_rotatable_hydrogen_environments,
)
from protrepair.transformer.completion.hydrogen.scoring import RotatableHydrogenSearch
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

RID = ResidueId("A", 1)
LIBRARY = build_default_component_library()


def with_fixture_topology(
    structure: ProteinStructure, topology: StructureTopology
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def molecular_fixture(
    atoms: tuple[CanonicalAtomPayload, ...],
    bonds: tuple[tuple[str, str, int], ...],
    *,
    component: str = "UNK",
) -> ProteinStructure:
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id=component,
                        residue_id=RID,
                        atoms=atoms,
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return with_fixture_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(AtomRef(RID, left)),
                    atom_index_2=structure.constitution.atom_index(AtomRef(RID, right)),
                    order=order,
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                )
                for left, right, order in bonds
            ),
        ),
    )


def glu_fixture(
    *, explicit_h: bool = True, hydrogen_element: str = "H"
) -> ProteinStructure:
    atoms = (
        atom_payload("N", "N", Vec3(0, 0, 0)),
        atom_payload("H", hydrogen_element, Vec3(1, 0, 0)),
        atom_payload("CA", "C", Vec3(-1, 1, 0)),
        atom_payload("CB", "C", Vec3(0, 2, 0)),
        atom_payload("CG", "C", Vec3(1, 3, 0)),
        atom_payload("CD", "C", Vec3(2.8, 1.2, 0)),
        atom_payload("OE1", "O", Vec3(2.8, 0, 0)),
        atom_payload("OE2", "O", Vec3(4, 1.2, 0), formal_charge=-1),
    )
    bonds = (("N", "H", 1),) if explicit_h else ()
    return molecular_fixture(atoms, bonds, component="GLU")


@pytest.mark.parametrize("component", ["GLU", "UNK"])
def test_empty_residue_has_no_supported_polar_contacts(component: str) -> None:
    structure = molecular_fixture((), (), component=component)
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    assert not chemistry.hydrogen_anchors(RID)
    assert not chemistry.acceptor_names(RID)
    context = prepare_clash_detection_context(structure, component_library=LIBRARY)
    assert not detect_clashes_from_context(context).clashes


@pytest.mark.parametrize("explicit_h", [True, False])
def test_glu_intramolecular_contact_uses_chemistry_not_residue_identity(
    explicit_h: bool,
) -> None:
    structure = glu_fixture(explicit_h=explicit_h)
    context = prepare_clash_detection_context(structure, component_library=LIBRARY)
    sites = {site.atom_name: site for site in context.atom_sites}
    assert probable_hydrogen_bond(sites["H"], sites["OE1"], 1.8)
    assert probable_hydrogen_bond(sites["OE1"], sites["H"], 1.8)
    assert not any(
        {clash.left_atom_name, clash.right_atom_name} == {"H", "OE1"}
        for clash in detect_clashes_from_context(context).clashes
    )
    chemistry = sites["H"].context.polar_chemistry
    assert chemistry.bond_hop_distance(AtomRef(RID, "H"), AtomRef(RID, "OE1")) == 6
    assert not any(
        {contact.left_atom_name, contact.right_atom_name} == {"H", "OE1"}
        for contact in detect_near_covalent_contacts(
            structure,
            component_library=LIBRARY,
            policy=NearCovalentContactPolicy(
                ignore_same_residue=False, covalent_distance_margin_angstrom=1.0
            ),
        )
    )


@pytest.mark.parametrize("element", ["H", "D", "T"])
def test_reused_polar_chemistry_rechecks_hbond_angle_and_full_clash_report(element):
    source = glu_fixture(hydrogen_element=element)
    observation = StructureChemistryObservation(source, component_library=LIBRARY)
    chemistry = observation.polar_contact_chemistry
    geometry = source.residue_geometry(source.constitution.residue_index(RID))
    moved = source.with_updated_residue_geometries(
        (
            (
                RID,
                geometry.with_atom_geometry(
                    "N", replace(geometry.atom_geometry("N"), position=Vec3(1, 1, 0))
                ),
            ),
        )
    )
    assert observation.for_structure(moved).polar_contact_chemistry is chemistry

    for structure, expected_hbond in ((source, True), (moved, False)):
        fresh = prepare_clash_detection_context(structure, component_library=LIBRARY)
        basis = prepare_clash_detection_basis(
            structure,
            component_library=LIBRARY,
            polar_chemistry=chemistry,
        )
        projected = prepare_projected_clash_detection_context(
            structure,
            residue_ids=(RID,),
            component_library=LIBRARY,
            polar_chemistry=chemistry,
        )
        for context in (basis.bind_context(structure), projected):
            sites = {site.atom_name: site for site in context.atom_sites}
            assert sites["H"].context.polar_chemistry is chemistry
            assert (
                probable_hydrogen_bond(sites["H"], sites["OE1"], 1.8) is expected_hbond
            )
            assert context.detect_clashes() == fresh.detect_clashes()
            assert (
                any(
                    {clash.left_atom_name, clash.right_atom_name} == {"H", "OE1"}
                    for clash in context.detect_clashes().clashes
                )
                is not expected_hbond
            )


@pytest.mark.parametrize("change", ["constitution", "topology", "library"])
@pytest.mark.parametrize("projected", [False, True])
def test_clash_factories_reject_polar_chemistry_bound_to_other_inputs(
    change, projected
):
    source = glu_fixture()
    chemistry = PolarContactChemistry(source.constitution, source.topology, LIBRARY)
    structure, library = source, LIBRARY
    if change == "constitution":
        structure = ProteinStructure.from_payload(
            constitution=glu_fixture().constitution,
            topology=source.topology,
            geometry=source.geometry,
            provenance=source.provenance,
            polymer_blueprint=source.polymer_blueprint,
        )
    elif change == "topology":
        structure = with_fixture_topology(
            source,
            StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=(),
            ),
        )
    else:
        library = ComponentLibrary()
    with pytest.raises(ValueError, match="polar chemistry.*graph.*library"):
        if projected:
            prepare_projected_clash_detection_context(
                structure,
                residue_ids=(RID,),
                component_library=library,
                polar_chemistry=chemistry,
            )
        else:
            prepare_clash_detection_basis(
                structure,
                component_library=library,
                polar_chemistry=chemistry,
            )


@pytest.mark.parametrize("element", ["H", "D", "T"])
def test_explicit_carbon_hydrogen_never_reassigned_to_nearer_nitrogen(
    element: str,
) -> None:
    structure = molecular_fixture(
        (
            atom_payload("CA", "C", Vec3(-0.1, 0, 0)),
            atom_payload("N", "N", Vec3(0, 0, 0)),
            atom_payload("HA2", element, Vec3(1, 0, 0)),
            atom_payload("C", "C", Vec3(4, 0, 0)),
            atom_payload("O", "O", Vec3(2.8, 0, 0)),
        ),
        (("CA", "HA2", 1), ("C", "O", 2)),
        component="GLY",
    )
    context = prepare_clash_detection_context(structure, component_library=LIBRARY)
    sites = {site.atom_name: site for site in context.atom_sites}
    assert sites["HA2"].context.hydrogen_anchor_by_name["HA2"] == "CA"
    assert not probable_hydrogen_bond(sites["HA2"], sites["O"], 1.8)
    assert any(
        {c.left_atom_name, c.right_atom_name} == {"HA2", "O"}
        for c in detect_clashes_from_context(context).clashes
    )


@pytest.mark.parametrize(
    ("atoms", "bonds", "expected"),
    [
        ((("C", "C", 0), ("O", "O", 0)), (("C", "O", 2),), {"O"}),
        (
            (
                ("C", "C", 0),
                ("O", "O", 0),
                ("N", "N", 0),
                ("H1", "H", 0),
                ("H2", "H", 0),
            ),
            (("C", "O", 2), ("C", "N", 1), ("N", "H1", 1), ("N", "H2", 1)),
            {"O"},
        ),
        (
            (
                ("C", "C", 0),
                ("N", "N", 1),
                ("H1", "H", 0),
                ("H2", "H", 0),
                ("H3", "H", 0),
            ),
            (("C", "N", 1), ("N", "H1", 1), ("N", "H2", 1), ("N", "H3", 1)),
            set(),
        ),
        ((("C", "C", 0), ("O", "O", -1)), (("C", "O", 1),), {"O"}),
        (
            (("C", "C", 0), ("O", "O", -1), ("H", "H", 0)),
            (("C", "O", 1), ("O", "H", 1)),
            set(),
        ),
        (
            (("C", "C", 0), ("O1", "O", 0), ("O2", "O", -1), ("H", "H", 0)),
            (("C", "O1", 1), ("C", "O2", 1), ("O2", "H", 1)),
            set(),
        ),
        (
            (("C", "C", 0), ("O1", "O", 0), ("O2", "O", -1)),
            (("C", "O1", 1), ("C", "O2", 2)),
            set(),
        ),
        (
            (("C", "C", 0), ("O1", "O", 0), ("O2", "O", 0), ("H", "H", 0)),
            (("C", "O1", 2), ("C", "O2", 1), ("O2", "H", 1)),
            {"O1"},
        ),
        (
            (("C", "C", 0), ("O", "O", 0), ("H", "H", 0)),
            (("C", "O", 1), ("O", "H", 1)),
            {"O"},
        ),
        (
            (("O", "O", 0), ("H1", "H", 0), ("H2", "H", 0)),
            (("O", "H1", 1), ("O", "H2", 1)),
            {"O"},
        ),
        ((("N", "N", 0), ("O", "O", 0), ("S", "S", 0)), (), set()),
    ],
)
def test_acceptor_motifs_require_charge_valence_and_resonance(
    atoms: tuple[tuple[str, str, int], ...],
    bonds: tuple[tuple[str, str, int], ...],
    expected: set[str],
) -> None:
    structure = molecular_fixture(
        tuple(
            atom_payload(name, element, Vec3(i, 0, 0), formal_charge=charge)
            for i, (name, element, charge) in enumerate(atoms)
        ),
        bonds,
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, ComponentLibrary()
    )
    assert chemistry.acceptor_names(RID) == expected


def test_ambiguous_canonical_hydrogen_prevents_template_fallback() -> None:
    structure = glu_fixture()
    bonds = (
        *structure.topology.bonds,
        TopologyBond(
            atom_index_1=structure.constitution.atom_index(AtomRef(RID, "H")),
            atom_index_2=structure.constitution.atom_index(AtomRef(RID, "CA")),
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        ),
    )
    structure = with_fixture_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    assert "H" not in chemistry.hydrogen_anchors(RID)


def test_pdb_unknown_conect_is_resolved_only_in_contact_projection() -> None:
    original = glu_fixture()
    raw = read_structure_string(
        write_structure_string(original, FileFormat.PDB), FileFormat.PDB
    )
    topology = raw.topology
    chemistry = PolarContactChemistry(raw.constitution, topology, LIBRARY)
    assert chemistry.hydrogen_anchors(RID)["H"] == "N"
    assert chemistry.bond_hop_distance(AtomRef(RID, "H"), AtomRef(RID, "N")) == 1
    assert "OE1" in chemistry.acceptor_names(RID)
    assert raw.topology is topology
    assert any(
        b.relationship_type is BondRelationshipType.UNKNOWN for b in topology.bonds
    )


@pytest.mark.parametrize("order", [None, 2])
def test_unknown_mismatched_attachment_never_falls_back_to_named_template(
    order: int | None,
) -> None:
    structure = glu_fixture()
    unknown = TopologyBond(
        atom_index_1=structure.constitution.atom_index(AtomRef(RID, "H")),
        atom_index_2=structure.constitution.atom_index(AtomRef(RID, "CA")),
        order=order,
        relationship_type=BondRelationshipType.UNKNOWN,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    structure = with_fixture_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(unknown,),
        ),
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    assert not chemistry.hydrogen_anchors(RID)


@pytest.mark.parametrize("element", ["H", "D", "T"])
def test_canonical_donor_isotope_and_unknown_attachment(element: str) -> None:
    structure = molecular_fixture(
        (
            atom_payload("N1", "N", Vec3(0, 0, 0)),
            atom_payload("HX", element, Vec3(1, 0, 0)),
            atom_payload("O1", "O", Vec3(2.8, 0, 0)),
            atom_payload("C1", "C", Vec3(4, 0, 0)),
        ),
        (("N1", "HX", 1), ("O1", "C1", 2)),
    )
    context = prepare_clash_detection_context(
        structure, component_library=ComponentLibrary()
    )
    sites = {site.atom_name: site for site in context.atom_sites}
    assert probable_hydrogen_bond(sites["HX"], sites["O1"], 1.8)
    detached = with_fixture_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                b
                for b in structure.topology.bonds
                if not b.involves(structure.constitution.atom_index(AtomRef(RID, "HX")))
            ),
        ),
    )
    sites = {
        site.atom_name: site
        for site in prepare_clash_detection_context(
            detached, component_library=ComponentLibrary()
        ).atom_sites
    }
    assert not probable_hydrogen_bond(sites["HX"], sites["O1"], 1.8)


def test_metal_relationship_does_not_supply_polar_contact_exemption() -> None:
    structure = molecular_fixture(
        (
            atom_payload("O", "O", Vec3(0, 0, 0)),
            atom_payload("C", "C", Vec3(1.2, 0, 0)),
            atom_payload("ZN", "ZN", Vec3(0, 0, 2)),
        ),
        (("O", "C", 2),),
    )
    metal = TopologyBond(
        atom_index_1=structure.constitution.atom_index(AtomRef(RID, "O")),
        atom_index_2=structure.constitution.atom_index(AtomRef(RID, "ZN")),
        relationship_type=BondRelationshipType.METAL_COORDINATION,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        order=None,
    )
    structure = with_fixture_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(*structure.topology.bonds, metal),
        ),
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, ComponentLibrary()
    )
    assert not chemistry.acceptor_names(RID)
    assert not chemistry.supports_donor(AtomRef(RID, "O"))


def test_coordinate_rebinding_reuses_chemistry_but_charge_invalidates_basis() -> None:
    structure = glu_fixture()
    basis = prepare_clash_detection_basis(
        structure, component_library=LIBRARY, policy=ClashPolicy()
    )
    context = basis.bind_context(structure)
    moved = structure.with_updated_residue_geometries(
        (
            (
                RID,
                structure.residue_geometry(
                    structure.constitution.residue_index(RID)
                ).with_atom_geometry(
                    "H",
                    AtomGeometry(position=Vec3(-1, 0, 0)),
                ),
            ),
        )
    )
    rebound = basis.bind_context(moved)
    assert (
        rebound.atom_sites[0].context.polar_chemistry
        is context.atom_sites[0].context.polar_chemistry
    )
    sites = {site.atom_name: site for site in rebound.atom_sites}
    assert not probable_hydrogen_bond(sites["H"], sites["OE1"], 3.8)
    atom_topologies = list(structure.topology.atom_topologies)
    index = structure.constitution.atom_index(AtomRef(RID, "OE2"))
    source_charge = atom_topologies[index.value]
    assert source_charge is not None
    atom_topologies[index.value] = replace(source_charge, formal_charge=1)
    changed = with_fixture_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=tuple(atom_topologies),
            bonds=structure.topology.bonds,
        ),
    )
    with pytest.raises(ValueError, match="compatible immutable"):
        basis.bind_context(changed)
    new_chemistry = PolarContactChemistry(
        changed.constitution, changed.topology, LIBRARY
    )
    assert "OE2" not in new_chemistry.acceptor_names(RID)


@pytest.mark.parametrize("charge", [0, 1])
def test_rotatable_scoring_uses_the_same_local_acceptor_chemistry(charge: int) -> None:
    structure = molecular_fixture(
        (
            atom_payload("OG", "O", Vec3(0, 0, 0)),
            atom_payload("HG", "H", Vec3(1, 0, 0)),
            atom_payload("CB", "C", Vec3(-1.4, 0, 0)),
            atom_payload("CA", "C", Vec3(-2.5, 1, 0)),
            atom_payload("C", "C", Vec3(4, 0, 0)),
            atom_payload("O", "O", Vec3(2.8, 0, 0), formal_charge=charge),
        ),
        (("OG", "HG", 1),),
        component="SER",
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    payload = CompletionResiduePayload(
        residue_site=structure.constitution.residue_slots[0],
        residue_geometry=structure.residue_geometry(
            structure.constitution.residue_index(RID)
        ),
        formal_charge_by_atom_name=structure.residue_formal_charge_by_atom_name(
            structure.constitution.residue_index(RID)
        ),
    )
    environment = build_rotatable_hydrogen_environments(
        residues=(payload.without_atom_sites({"HG"}),),
        templates=(LIBRARY.require("SER"),),
        polar_chemistry=chemistry,
    )[0]
    search = RotatableHydrogenSearch(
        outer_anchor=Vec3(-2.5, 1, 0),
        inner_anchor=Vec3(-1.4, 0, 0),
        donor=Vec3(0, 0, 0),
        hydrogen=Vec3(1, 0, 0),
        bond_length=1,
        bond_angle_degrees=108.5,
        dihedral=0,
        partial_charge=0,
        sigma=0,
        epsilon=0,
    )
    assert search.clash_burden(Vec3(1, 0, 0), environment).clash_count == charge
    sites = {
        site.atom_name: site
        for site in prepare_clash_detection_context(
            structure, component_library=LIBRARY
        ).atom_sites
    }
    assert probable_hydrogen_bond(sites["HG"], sites["O"], 1.8) is (charge == 0)


@pytest.mark.parametrize(
    "component,n_name,h_name",
    [
        ("ASN", "ND2", "HD21"),
        ("GLN", "NE2", "HE21"),
        ("HIS", "NE2", "HE2"),
        ("TRP", "NE1", "HE1"),
    ],
)
def test_supported_amide_and_ring_templates_do_not_admit_protonated_nitrogen(
    component: str,
    n_name: str,
    h_name: str,
) -> None:
    template = LIBRARY.require(component)
    atoms = tuple(
        atom_payload(
            name,
            "O" if name.startswith("O") else "N" if name.startswith("N") else "C",
            Vec3(i, 0, 0),
        )
        for i, name in enumerate(template.definition.atom_names)
        if not name.startswith("H")
    )
    atoms += (atom_payload(h_name, "H", Vec3(0, 1, 0)),)
    structure = molecular_fixture(atoms, ((n_name, h_name, 1),), component=component)
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    assert n_name not in chemistry.acceptor_names(RID)


@pytest.mark.parametrize("name,element", [("HB3", "H"), ("3HB", "H"), ("DB3", "D")])
def test_standard_methylene_naming_variants_keep_their_supported_parent(
    name: str, element: str
) -> None:
    structure = molecular_fixture(
        (
            atom_payload("CA", "C", Vec3(0, 0, 0)),
            atom_payload("CB", "C", Vec3(1.5, 0, 0)),
            atom_payload("CG", "C", Vec3(2.5, 1, 0)),
            atom_payload(name, element, Vec3(1.5, -1, 0)),
        ),
        (),
        component="PRO",
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    assert chemistry.hydrogen_anchors(RID)[name] == "CB"
    assert chemistry.bond_hop_distance(AtomRef(RID, name), AtomRef(RID, "CA")) == 2


def test_ambiguous_template_naming_family_cannot_create_an_attachment() -> None:
    structure = molecular_fixture(
        (
            atom_payload("OG1", "O", Vec3(0, 0, 0)),
            atom_payload("CG2", "C", Vec3(2, 0, 0)),
            atom_payload("HG3", "H", Vec3(1, 0, 0)),
        ),
        (),
        component="THR",
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    assert "HG3" not in chemistry.hydrogen_anchors(RID)


@pytest.mark.parametrize(
    "component,name,parent",
    [
        ("VAL", "HG11", "CG1"),
        ("VAL", "HG23", "CG2"),
        ("ILE", "HD13", "CD1"),
        ("THR", "HG21", "CG2"),
    ],
)
def test_pdb_prefix_suffix_numbering_preserves_the_branch(
    component: str, name: str, parent: str
) -> None:
    structure = molecular_fixture(
        (
            atom_payload(parent, "C", Vec3(0, 0, 0)),
            atom_payload(name, "H", Vec3(1, 0, 0)),
        ),
        (),
        component=component,
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, LIBRARY
    )
    assert chemistry.hydrogen_anchors(RID)[name] == parent


@pytest.mark.parametrize(
    "smiles,element,expected_count",
    [
        ("CN", "N", 1),
        ("C[NH3+]", "N", 0),
        ("CC(=O)N", "N", 0),
        ("c1ccncc1", "N", 1),
        ("c1cc[nH]c1", "N", 0),
        ("CSC", "S", 1),
        ("CS", "S", 0),
        ("CSSC", "S", 0),
    ],
)
def test_explicit_molecular_graph_supports_distinct_polar_motifs(
    smiles: str,
    element: str,
    expected_count: int,
) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atoms = tuple(
        atom_payload(
            f"X{a.GetIdx()}",
            a.GetSymbol().upper(),
            Vec3(a.GetIdx(), 0, 0),
            formal_charge=a.GetFormalCharge(),
        )
        for a in molecule.GetAtoms()
    )
    structure = molecular_fixture(atoms, ())
    structure = with_fixture_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(RID, f"X{b.GetBeginAtomIdx()}")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(RID, f"X{b.GetEndAtomIdx()}")
                    ),
                    order=1 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()),
                    aromatic=b.GetIsAromatic(),
                    provenance=BondProvenance.EVIDENCE_RESOLVED,
                )
                for b in molecule.GetBonds()
            ),
        ),
    )
    chemistry = PolarContactChemistry(
        structure.constitution, structure.topology, ComponentLibrary()
    )
    assert (
        sum(
            a.name in chemistry.acceptor_names(RID)
            for a in structure.constitution.atom_slots
            if a.element == element
        )
        == expected_count
    )


@pytest.mark.parametrize(
    "component,oxygen_names",
    [("ASP", ("OD1", "OD2")), ("GLU", ("OE1", "OE2")), ("ALA", ("O", "OXT"))],
)
def test_stock_carboxyl_oxygens_are_acceptors_before_charge_publication(
    component: str, oxygen_names: tuple[str, str]
) -> None:
    template = LIBRARY.require(component)
    source = molecular_fixture(
        tuple(
            atom_payload(
                name,
                "N" if name.startswith("N") else "O" if name.startswith("O") else "C",
                Vec3(i, 0, 0),
            )
            for i, name in enumerate(template.definition.atom_names)
            if name != "OXT" or component == "ALA"
        ),
        (),
        component=component,
    )
    topology = source.topology
    chemistry = PolarContactChemistry(source.constitution, topology, LIBRARY)

    assert set(oxygen_names) <= chemistry.acceptor_names(RID)
    assert source.topology is topology
    assert all(
        topology.formal_charge(source.constitution.atom_index(AtomRef(RID, name)))
        is None
        for name in oxygen_names
    )


@pytest.mark.parametrize("oxygen_order", [1, 2])
@pytest.mark.parametrize("acid_hydrogen", [False, True])
def test_unknown_carboxyl_charge_does_not_hide_unprotonated_oxygen(
    oxygen_order: int, acid_hydrogen: bool
) -> None:
    atoms = tuple(
        atom_payload(name, element, Vec3(i, 0, 0))
        for i, (name, element) in enumerate(
            (("R", "C"), ("C", "C"), ("O1", "O"), ("O2", "O"))
        )
    )
    bonds = (("R", "C", 1), ("C", "O1", oxygen_order), ("C", "O2", 1))
    if acid_hydrogen:
        atoms += (atom_payload("HO", "D", Vec3(3, 1, 0)),)
        bonds += (("O2", "HO", 1),)
    source = molecular_fixture(atoms, bonds)
    chemistry = PolarContactChemistry(
        source.constitution, source.topology, ComponentLibrary()
    )

    assert chemistry.acceptor_names(RID) == ({"O1"} if acid_hydrogen else {"O1", "O2"})


@pytest.mark.parametrize("extra", [None, "H"])
def test_carboxyl_inference_requires_a_complete_trigonal_carbon(
    extra: str | None,
) -> None:
    atoms = (
        atom_payload("C", "C", Vec3(0, 0, 0)),
        atom_payload("O1", "O", Vec3(1, 0, 0)),
        atom_payload("O2", "O", Vec3(-1, 0, 0)),
    )
    bonds = (("C", "O1", 1), ("C", "O2", 1))
    if extra is not None:
        atoms += (
            atom_payload("R", "C", Vec3(0, 1, 0)),
            atom_payload("H", "H", Vec3(0, -1, 0)),
        )
        bonds += (("C", "R", 1), ("C", "H", 1))
    source = molecular_fixture(atoms, bonds)
    chemistry = PolarContactChemistry(
        source.constitution, source.topology, ComponentLibrary()
    )

    assert not chemistry.acceptor_names(RID)


@pytest.mark.parametrize(
    "name,element", [("HG12", "H"), ("HG13", "H"), ("DG13", "D"), ("TG13", "T")]
)
def test_branched_methylene_suffix_preserves_the_template_parent(
    name: str, element: str
) -> None:
    source = molecular_fixture(
        (
            atom_payload("CB", "C", Vec3(-1, 0, 0)),
            atom_payload("CG1", "C", Vec3(0, 0, 0)),
            atom_payload("CG2", "C", Vec3(0.2, 0, 0)),
            atom_payload("CD1", "C", Vec3(0, 1, 0)),
            atom_payload(name, element, Vec3(1, 0, 0)),
        ),
        (),
        component="ILE",
    )
    chemistry = PolarContactChemistry(source.constitution, source.topology, LIBRARY)

    assert chemistry.hydrogen_anchors(RID)[name] == "CG1"
    assert chemistry.bond_hop_distance(AtomRef(RID, name), AtomRef(RID, "CG1")) == 1


@pytest.mark.parametrize(
    "component,name,parents",
    [("ILE", "HG24", ("CG1", "CG2")), ("ARG", "HH13", ("NH1", "NH2"))],
)
def test_methylene_suffix_shift_does_not_create_methyl_or_amide_aliases(
    component, name, parents
) -> None:
    source = molecular_fixture(
        tuple(
            atom_payload(parent, "N" if parent.startswith("N") else "C", Vec3(i, 0, 0))
            for i, parent in enumerate(parents)
        )
        + (atom_payload(name, "H", Vec3(1, 1, 0)),),
        (),
        component=component,
    )
    chemistry = PolarContactChemistry(source.constitution, source.topology, LIBRARY)

    assert name not in chemistry.hydrogen_anchors(RID)


@pytest.mark.parametrize(
    "center_charge,peer_charge,peer_order", [(1, None, 2), (0, 1, 1), (0, -1, 2)]
)
def test_unknown_carboxyl_inference_preserves_declared_charge_conflicts(
    center_charge, peer_charge, peer_order
) -> None:
    source = molecular_fixture(
        (
            atom_payload("R", "C", Vec3(0, 1, 0)),
            atom_payload("C", "C", Vec3(0, 0, 0), formal_charge=center_charge),
            atom_payload("O1", "O", Vec3(1, 0, 0)),
            atom_payload("O2", "O", Vec3(-1, 0, 0), formal_charge=peer_charge),
        ),
        (("C", "R", 1), ("C", "O1", 1), ("C", "O2", peer_order)),
    )
    chemistry = PolarContactChemistry(
        source.constitution, source.topology, ComponentLibrary()
    )

    assert "O1" not in chemistry.acceptor_names(RID)


def test_methylene_alias_does_not_override_a_declared_source_parent() -> None:
    source = molecular_fixture(
        (
            atom_payload("CG1", "C", Vec3(0, 0, 0)),
            atom_payload("CG2", "C", Vec3(2, 0, 0)),
            atom_payload("HG13", "H", Vec3(1, 0, 0)),
        ),
        (("CG2", "HG13", 1),),
        component="ILE",
    )
    chemistry = PolarContactChemistry(source.constitution, source.topology, LIBRARY)

    assert chemistry.hydrogen_anchors(RID)["HG13"] == "CG2"
