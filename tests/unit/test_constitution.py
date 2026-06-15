from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.geometry import Vec3
from protrepair.structure import AtomSite
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import (
    AtomIndex,
    ChainIndex,
    ResidueIndex,
)


def test_atom_site_drops_geometry_and_charge_payload() -> None:
    atom_site, _atom_geometry, _formal_charge = atom_payload(
        name=" ca ",
        element=" c ",
        position=Vec3(1.0, 2.0, 3.0),
        occupancy=0.5,
        b_factor=12.0,
        formal_charge=-1,
        altloc=" A ",
    )

    assert atom_site == AtomSite(name="CA", element="C")


def test_residue_constitution_preserves_identity_and_atom_order() -> None:
    residue_site, residue_geometry, _formal_charge_by_atom_name = residue_payload(
        component_id="ser",
        residue_id=ResidueId(chain_id="A", seq_num=10, insertion_code=None),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
        ),
    )

    assert residue_site.component_id == "SER"
    assert residue_site.atom_site_names() == ("N", "CA", "CB")
    assert residue_site.atom_site("CA") == AtomSite(name="CA", element="C")
    assert residue_geometry.position("CA") == Vec3(1.0, 0.0, 0.0)


def test_structure_constitution_collects_chain_and_ligand_scaffold() -> None:
    polymer_residue = residue_payload(
        component_id="gly",
        residue_id=ResidueId(chain_id="A", seq_num=1, insertion_code=None),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
        ),
    )
    ligand = residue_payload(
        component_id="SO4",
        residue_id=ResidueId(chain_id="Z", seq_num=900, insertion_code=None),
        atoms=(atom_payload("S", "S", Vec3(4.0, 0.0, 0.0)),),
        is_hetero=True,
    )
    structure = build_structure(
        chains=(chain_payload("A", (polymer_residue,)),),
        ligands=(ligand,),
        source_format=FileFormat.PDB,
        source_name="example.pdb",
    )

    constitution = structure.constitution

    assert constitution.chain_ids() == ("A",)
    assert constitution.chain("A").residue_ids() == (polymer_residue[0].residue_id,)
    assert constitution.residue_or_ligand(ligand[0].residue_id) == ligand[0]
    assert constitution.resolve_atom_site(
        AtomRef(residue_id=polymer_residue[0].residue_id, atom_name="CA")
    ) == AtomSite(name="CA", element="C")
    assert structure.constitution.resolve_atom_site(
        AtomRef(residue_id=polymer_residue[0].residue_id, atom_name="CA")
    ) == AtomSite(name="CA", element="C")
    assert tuple(structure.iter_atom_sites(include_ligands=True)) == (
        AtomSite(name="N", element="N"),
        AtomSite(name="CA", element="C"),
        AtomSite(name="S", element="S"),
    )


def test_structure_preserves_formal_charge_payload_on_constitution_scaffold() -> None:
    residue = residue_payload(
        component_id="ser",
        residue_id=ResidueId(chain_id="A", seq_num=10, insertion_code=None),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0), formal_charge=1),
        ),
    )
    structure = build_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
    )

    residue_id = residue[0].residue_id
    assert structure.constitution.chain("A") == chain_payload("A", (residue,))[0]
    assert structure.constitution.residue_or_ligand(residue_id) == residue[0]
    assert (
        structure.topology.formal_charge(
            structure.constitution.atom_index(
                AtomRef(residue_id=residue_id, atom_name="CA")
            )
        )
        == 1
    )


def test_structure_constitution_exposes_stable_slot_indices() -> None:
    residue_a = residue_payload(
        component_id="gly",
        residue_id=ResidueId(chain_id="A", seq_num=1, insertion_code=None),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
        ),
    )
    residue_b = residue_payload(
        component_id="ser",
        residue_id=ResidueId(chain_id="A", seq_num=2, insertion_code=None),
        atoms=(atom_payload("CB", "C", Vec3(2.0, 0.0, 0.0)),),
    )
    ligand = residue_payload(
        component_id="SO4",
        residue_id=ResidueId(chain_id="Z", seq_num=900, insertion_code=None),
        atoms=(atom_payload("S", "S", Vec3(4.0, 0.0, 0.0)),),
        is_hetero=True,
    )
    structure = build_structure(
        chains=(chain_payload("A", (residue_a, residue_b)),),
        ligands=(ligand,),
        source_format=FileFormat.PDB,
    )

    constitution = structure.constitution

    assert constitution.chain_index("A") == ChainIndex(0)
    assert constitution.chain_site_at(ChainIndex(0)).chain_id == "A"

    assert constitution.residue_index(residue_a[0].residue_id) == ResidueIndex(0)
    assert constitution.residue_index(residue_b[0].residue_id) == ResidueIndex(1)
    assert constitution.residue_index(ligand[0].residue_id) == ResidueIndex(2)
    assert constitution.residue_site_at(ResidueIndex(2)) == ligand[0]

    ca_ref = AtomRef(residue_id=residue_a[0].residue_id, atom_name="CA")
    ligand_ref = AtomRef(residue_id=ligand[0].residue_id, atom_name="S")
    missing_ref = AtomRef(residue_id=residue_b[0].residue_id, atom_name="CG")
    assert constitution.atom_index(ca_ref) == AtomIndex(1)
    assert constitution.atom_index(ligand_ref) == AtomIndex(3)
    assert constitution.resolve_atom_index(ca_ref) == AtomIndex(1)
    assert constitution.resolve_atom_index(missing_ref) is None
    assert constitution.atom_ref_at(AtomIndex(3)) == ligand_ref
    assert constitution.atom_site_at(AtomIndex(1)) == AtomSite(
        name="CA",
        element="C",
    )
