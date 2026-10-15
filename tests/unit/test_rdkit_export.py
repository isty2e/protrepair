import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.errors import (
    ProtrepairError,
    RdkitProjectionError,
    RdkitUnavailableError,
)
from protrepair.geometry import Vec3
from protrepair.io import rdkit as rdkit_export
from protrepair.io import to_rdkit_molecule
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


def _structure(
    atoms: tuple[CanonicalAtomPayload, ...],
    edges: tuple[tuple[int, int, int | None, BondRelationshipType, bool], ...],
) -> ProteinStructure:
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("LONG_CHAIN", 7, "A"),
                atoms=atoms,
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                TopologyBond(
                    AtomIndex(first),
                    AtomIndex(second),
                    order=order,
                    relationship_type=relationship,
                    aromatic=aromatic,
                )
                for first, second, order, relationship, aromatic in edges
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _smiles_structure(smiles: str) -> ProteinStructure:
    parsed = Chem.MolFromSmiles(smiles)
    assert parsed is not None
    molecule = Chem.AddHs(parsed)
    Chem.Kekulize(molecule, clearAromaticFlags=False)
    atoms = tuple(
        atom_payload(
            f"A{atom.GetIdx()}",
            {2: "D", 3: "T"}.get(atom.GetIsotope(), atom.GetSymbol()),
            Vec3(atom.GetIdx() * 0.123456789, 0.987654321, -0.000012345),
            formal_charge=atom.GetFormalCharge(),
            occupancy=0.75,
            b_factor=12.5,
            altloc="B",
        )
        for atom in molecule.GetAtoms()
    )
    edges = tuple(
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            None
            if bond.GetBondType() == Chem.BondType.DATIVE
            else int(bond.GetBondTypeAsDouble()),
            BondRelationshipType.METAL_COORDINATION
            if bond.GetBondType() == Chem.BondType.DATIVE
            else BondRelationshipType.COVALENT,
            bond.GetIsAromatic(),
        )
        for bond in molecule.GetBonds()
    )
    return _structure(atoms, edges)


@pytest.mark.parametrize(
    "smiles",
    ("C", "C=O", "C#N", "c1ccccc1", "c1cc[nH]c1", "[NH4+]", "[Na+].[Cl-]"),
)
def test_native_export_preserves_selected_organic_chemistry(smiles):
    structure = _smiles_structure(smiles)
    native = to_rdkit_molecule(structure)
    expected = Chem.AddHs(Chem.MolFromSmiles(smiles))

    assert Chem.MolToSmiles(native) == Chem.MolToSmiles(expected)
    assert native.GetNumAtoms() == structure.geometry.atom_count()
    assert not any(atom.GetNumImplicitHs() for atom in native.GetAtoms())
    assert not any(atom.GetNumRadicalElectrons() for atom in native.GetAtoms())
    assert Chem.SanitizeMol(native) == Chem.SanitizeFlags.SANITIZE_NONE


@pytest.mark.parametrize(
    "smiles",
    ("c1ccccc1", "n1ccccc1", "c1cc[nH]c1", "c1ccc2ccccc2c1", "C=Cc1ccccc1"),
)
def test_native_export_normalizes_kekule_input_without_changing_selected_chemistry(
    smiles,
):
    source = _smiles_structure(smiles)
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=tuple(replace(bond, aromatic=False) for bond in source.topology.bonds),
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    native = to_rdkit_molecule(source)
    expected = Chem.AddHs(Chem.MolFromSmiles(smiles))

    assert Chem.MolToSmiles(native) == Chem.MolToSmiles(expected)
    assert any(bond.GetIsAromatic() for bond in native.GetBonds())
    assert native.GetNumBonds() == len(topology.bonds)
    assert source.topology is topology
    assert not any(bond.aromatic for bond in source.topology.bonds)
    for bond in topology.bonds:
        first, second = (index.value for index in bond.endpoint_pair())
        projected = native.GetBondBetweenAtoms(first, second)
        assert projected.GetIntProp("protrepair_bond_order") == bond.order
        assert not projected.GetBoolProp("protrepair_aromatic")
        if not projected.GetIsAromatic():
            assert projected.GetBondTypeAsDouble() == bond.order

    assert tuple(
        tuple(native.GetConformer().GetAtomPosition(index))
        for index in range(native.GetNumAtoms())
    ) == tuple(tuple(atom.position) for atom in source.geometry.atom_geometries)
    assert not any(
        atom.GetNumRadicalElectrons() or atom.GetNumImplicitHs()
        for atom in native.GetAtoms()
    )


def test_native_export_copies_full_precision_identity_scalars_and_isotopes():
    structure = _smiles_structure("[2H]O[3H]")
    native = to_rdkit_molecule(structure)
    conformer = native.GetConformer()
    assert conformer.Is3D()
    assert [atom.GetIsotope() for atom in native.GetAtoms()] == [2, 0, 3]

    for index, geometry in enumerate(structure.geometry.atom_geometries):
        atom = native.GetAtomWithIdx(index)
        point = conformer.GetAtomPosition(index)
        assert (point.x, point.y, point.z) == tuple(geometry.position)
        assert atom.GetBoolProp("protrepair_formal_charge_known")
        info = atom.GetPDBResidueInfo()
        assert info.GetName().strip() == f"A{index}"
        assert info.GetResidueName() == "LIG"
        assert info.GetChainId() == "LONG_CHAIN"
        assert info.GetResidueNumber() == 7
        assert info.GetInsertionCode() == "A"
        assert info.GetIsHeteroAtom()
        assert info.GetAltLoc() == "B"
        assert info.GetOccupancy() == 0.75
        assert info.GetTempFactor() == 12.5

    conformer.SetAtomPosition(0, (9.0, 9.0, 9.0))
    native.GetAtomWithIdx(0).SetFormalCharge(1)
    assert structure.geometry.atom_geometries[0].position != Vec3(9.0, 9.0, 9.0)
    assert structure.topology.formal_charge(AtomIndex(0)) == 0
    assert to_rdkit_molecule(structure).GetAtomWithIdx(0).GetFormalCharge() == 0


def test_native_export_marks_unknown_nonmetal_charge_without_assigning_it():
    structure = _smiles_structure("C")
    topologies = tuple(None for _ in structure.topology.atom_topologies)
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=topologies,
            bonds=structure.topology.bonds,
        ),
    )
    native = to_rdkit_molecule(structure)
    assert all(atom.GetFormalCharge() == 0 for atom in native.GetAtoms())
    assert not any(
        atom.GetBoolProp("protrepair_formal_charge_known") for atom in native.GetAtoms()
    )
    assert all(
        structure.topology.formal_charge(AtomIndex(value)) is None
        for value in range(native.GetNumAtoms())
    )


@pytest.mark.parametrize("known_charge", (None, 0))
def test_native_export_refuses_sanitation_charge_reinterpretation(known_charge):
    structure = _structure(
        (
            atom_payload("C", "C", Vec3(0, 0, 0), formal_charge=0),
            atom_payload("N", "N", Vec3(1, 0, 0), formal_charge=known_charge),
            atom_payload("O1", "O", Vec3(2, 1, 0), formal_charge=known_charge),
            atom_payload("O2", "O", Vec3(2, -1, 0), formal_charge=known_charge),
            *(
                atom_payload(f"H{i}", "H", Vec3(-1, i, 0), formal_charge=0)
                for i in range(3)
            ),
        ),
        (
            (0, 1, 1, BondRelationshipType.COVALENT, False),
            (1, 2, 2, BondRelationshipType.COVALENT, False),
            (1, 3, 2, BondRelationshipType.COVALENT, False),
            *((0, i, 1, BondRelationshipType.COVALENT, False) for i in range(4, 7)),
        ),
    )
    with pytest.raises(ProtrepairError, match="sanitation changed formal charge"):
        to_rdkit_molecule(structure)
    assert structure.topology.formal_charge(AtomIndex(1)) == known_charge


def test_native_export_refuses_implicit_organometallic_bond_retyping():
    structure = _smiles_structure("[Zn+2]<-[OH2]")
    bonds = tuple(
        TopologyBond(
            *bond.endpoint_pair(),
            order=1,
            relationship_type=BondRelationshipType.COVALENT,
        )
        for bond in structure.topology.bonds
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    with pytest.raises(ProtrepairError, match="sanitation changed bond chemistry"):
        to_rdkit_molecule(structure)
    assert all(
        bond.relationship_type is BondRelationshipType.COVALENT for bond in bonds
    )


def test_native_export_does_not_hide_incompatible_integral_aromatic_orders():
    structure = _smiles_structure("c1ccccc1")
    bonds = tuple(
        replace(bond, order=2) if bond.aromatic else bond
        for bond in structure.topology.bonds
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    with pytest.raises(ProtrepairError):
        to_rdkit_molecule(structure)


@pytest.mark.parametrize("smiles", ("n1ccccc1", "c1cc[nH]c1", "c1ccc2ccccc2c1"))
def test_native_export_preserves_uniform_order_aromatic_semantics(smiles):
    source = _smiles_structure(smiles)
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=tuple(
            replace(bond, order=1) if bond.aromatic else bond
            for bond in source.topology.bonds
        ),
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    native = to_rdkit_molecule(source)
    expected = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert Chem.MolToSmiles(native) == Chem.MolToSmiles(expected)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in native.GetAtoms())
    assert source.topology is topology


@pytest.mark.parametrize(
    "smiles", ("[Zn+2]<-[OH2]", "[Fe+3]<-n1ccccc1", "[Zn+2]<-[SH2]")
)
def test_native_export_orients_declared_coordination_without_ff_admission(smiles):
    structure = _smiles_structure(smiles)
    native = to_rdkit_molecule(structure)
    dative = [
        bond for bond in native.GetBonds() if bond.GetBondType() == Chem.BondType.DATIVE
    ]
    assert len(dative) == 1
    assert dative[0].GetEndAtom().GetSymbol() in {"Fe", "Zn"}
    assert dative[0].GetBeginAtom().GetSymbol() in {"N", "O", "S"}
    assert dative[0].GetProp("protrepair_relationship_type") == "metal_coordination"
    assert not any(atom.GetNumImplicitHs() for atom in native.GetAtoms())


def test_native_export_derives_tetrahedral_stereo_from_coordinates():
    def tetrahedron(sign):
        return _structure(
            (
                atom_payload("C", "C", Vec3(0, 0, 0), formal_charge=0),
                atom_payload("F", "F", Vec3(sign, 0, 0), formal_charge=0),
                atom_payload("CL", "CL", Vec3(0, 1, 0), formal_charge=0),
                atom_payload("BR", "BR", Vec3(0, 0, 1), formal_charge=0),
                atom_payload("H", "H", Vec3(-sign, -1, -1), formal_charge=0),
            ),
            tuple(
                (0, end, 1, BondRelationshipType.COVALENT, False) for end in range(1, 5)
            ),
        )

    labels = {
        to_rdkit_molecule(tetrahedron(sign)).GetAtomWithIdx(0).GetProp("_CIPCode")
        for sign in (-1, 1)
    }
    assert labels == {"R", "S"}


@pytest.mark.parametrize(
    "relationship,order,aromatic,message",
    (
        (BondRelationshipType.UNKNOWN, 1, False, "unresolved relationship"),
        (BondRelationshipType.COVALENT, None, False, "unresolved bond order"),
        (BondRelationshipType.COVALENT, 5, False, "unsupported bond order"),
        (BondRelationshipType.COVALENT, 3, True, "aromatic"),
    ),
)
def test_native_export_refuses_unresolved_or_unsupported_edges(
    relationship, order, aromatic, message
):
    structure = _structure(
        (
            atom_payload("H1", "H", Vec3(0, 0, 0), formal_charge=0),
            atom_payload("H2", "H", Vec3(0, 0, 1), formal_charge=0),
        ),
        ((0, 1, order, relationship, aromatic),),
    )
    with pytest.raises(ProtrepairError, match=message):
        to_rdkit_molecule(structure)


@pytest.mark.parametrize("symbol", ("Fe", "Zn", "Na"))
def test_native_export_refuses_unknown_metal_charge(symbol):
    structure = _structure((atom_payload("M", symbol, Vec3(0, 0, 0)),), ())
    with pytest.raises(ProtrepairError, match="unknown metal charge"):
        to_rdkit_molecule(structure)
    assert structure.topology.formal_charge(AtomIndex(0)) is None


@pytest.mark.parametrize("elements", (("Zn", "Fe"), ("O", "N")))
def test_native_export_refuses_ambiguous_coordination_direction(elements):
    structure = _structure(
        tuple(
            atom_payload(f"A{index}", element, Vec3(index, 0, 0), formal_charge=0)
            for index, element in enumerate(elements)
        ),
        ((0, 1, None, BondRelationshipType.METAL_COORDINATION, False),),
    )
    with pytest.raises(ProtrepairError, match="exactly one metal"):
        to_rdkit_molecule(structure)


def test_native_export_reports_hydrogen_bond_projection_loss():
    structure = _smiles_structure("O.O")
    additional = TopologyBond(
        AtomIndex(0),
        AtomIndex(1),
        order=None,
        relationship_type=BondRelationshipType.HYDROGEN_BOND,
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(*structure.topology.bonds, additional),
        ),
    )
    native = to_rdkit_molecule(structure)
    assert native.GetIntProp("protrepair_omitted_hydrogen_bonds") == 1
    assert native.GetBondBetweenAtoms(0, 1) is None
    assert native.GetNumAtoms() == 6
    assert len(structure.topology.bonds) == 5


def test_native_export_refuses_radicals_instead_of_inventing_missing_hydrogens():
    structure = _structure(
        (atom_payload("C", "C", Vec3(0, 0, 0), formal_charge=0),), ()
    )
    with pytest.raises(ProtrepairError, match="radical"):
        to_rdkit_molecule(structure)


def test_native_export_does_not_use_pdb_or_hydrogenation(monkeypatch):
    structure = _smiles_structure("C")

    def unexpected(*args, **kwargs):
        pytest.fail("native export must not parse PDB or add H")

    monkeypatch.setattr(Chem, "MolFromPDBBlock", unexpected)
    monkeypatch.setattr(Chem, "AddHs", unexpected)
    native = to_rdkit_molecule(structure)
    assert native.GetNumAtoms() == 5


def test_native_export_preserves_identity_after_atom_reordering():
    original = _smiles_structure("[Zn+2]<-[OH2]")
    values = tuple(reversed(range(original.geometry.atom_count())))
    remap = {old: new for new, old in enumerate(values)}
    structure = _structure(
        tuple(
            atom_payload(
                original.constitution.atom_site_at(AtomIndex(value)).name,
                original.constitution.atom_site_at(AtomIndex(value)).element,
                original.geometry.atom_geometry(AtomIndex(value)).position,
                formal_charge=original.topology.formal_charge(AtomIndex(value)),
            )
            for value in values
        ),
        tuple(
            (
                remap[bond.atom_index_1.value],
                remap[bond.atom_index_2.value],
                bond.order,
                bond.relationship_type,
                bond.aromatic,
            )
            for bond in original.topology.bonds
        ),
    )
    native = to_rdkit_molecule(structure)
    dative = next(
        bond for bond in native.GetBonds() if bond.GetBondType() == Chem.BondType.DATIVE
    )
    assert dative.GetEndAtomIdx() == remap[0]
    assert dative.GetBeginAtomIdx() == remap[1]
    for value, site in enumerate(structure.constitution.atom_slots):
        atom = native.GetAtomWithIdx(value)
        assert atom.GetPDBResidueInfo().GetName().strip() == site.name
        position = native.GetConformer().GetAtomPosition(value)
        assert (position.x, position.y, position.z) == tuple(
            structure.geometry.atom_geometry(AtomIndex(value)).position
        )


def test_native_export_refuses_aromatic_evidence_for_nonaromatic_closed_valence():
    structure = _smiles_structure("CC")
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                replace(bond, aromatic=True)
                if bond.endpoint_pair() == (AtomIndex(0), AtomIndex(1))
                else bond
                for bond in structure.topology.bonds
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    with pytest.raises(RdkitProjectionError):
        to_rdkit_molecule(structure)


def test_native_export_preserves_resolved_disulfide_relationship():
    structure = _smiles_structure("CSSC")
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                replace(bond, relationship_type=BondRelationshipType.DISULFIDE)
                if bond.endpoint_pair() == (AtomIndex(1), AtomIndex(2))
                else bond
                for bond in structure.topology.bonds
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    bond = to_rdkit_molecule(structure).GetBondBetweenAtoms(1, 2)
    assert bond.GetBondType() == Chem.BondType.SINGLE
    assert bond.GetProp("protrepair_relationship_type") == "disulfide"
    assert bond.GetIntProp("protrepair_bond_order") == 1


def test_native_export_keeps_unknown_b_factor_distinct_from_explicit_zero():
    structure = _structure(
        (
            atom_payload("H1", "H", Vec3(0, 0, 0), formal_charge=0),
            atom_payload("H2", "H", Vec3(0, 0, 1), formal_charge=0, b_factor=0.0),
        ),
        ((0, 1, 1, BondRelationshipType.COVALENT, False),),
    )
    native = to_rdkit_molecule(structure)
    assert not native.GetAtomWithIdx(0).GetBoolProp("protrepair_b_factor_known")
    assert native.GetAtomWithIdx(1).GetBoolProp("protrepair_b_factor_known")
    assert all(
        atom.GetPDBResidueInfo().GetTempFactor() == 0 for atom in native.GetAtoms()
    )
    assert structure.geometry.atom_geometry(AtomIndex(0)).b_factor is None


def test_native_export_distinguishes_explicit_neutral_metal_from_unknown_charge():
    structure = _smiles_structure("[Fe+0]")
    native = to_rdkit_molecule(structure)
    assert native.GetAtomWithIdx(0).GetFormalCharge() == 0
    assert native.GetAtomWithIdx(0).GetBoolProp("protrepair_formal_charge_known")


@pytest.mark.parametrize("value", (None, "protein.pdb", 1))
def test_native_export_rejects_noncanonical_input_without_loading_it(value):
    with pytest.raises(TypeError, match="ProteinStructure"):
        to_rdkit_molecule(value)


def test_native_export_supports_empty_canonical_structure():
    native = to_rdkit_molecule(_structure((), ()))
    assert native.GetNumAtoms() == 0
    assert native.GetNumBonds() == 0
    assert native.GetConformer().Is3D()
    assert native.GetIntProp("protrepair_omitted_hydrogen_bonds") == 0


def test_native_export_refuses_aromatic_coordination():
    structure = _smiles_structure("[Zn+2]<-[OH2]")
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                replace(bond, aromatic=True)
                if bond.relationship_type is BondRelationshipType.METAL_COORDINATION
                else bond
                for bond in structure.topology.bonds
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    with pytest.raises(ProtrepairError, match="aromatic coordination"):
        to_rdkit_molecule(structure)


def test_public_import_and_non_native_io_work_without_operational_rdkit():
    program = """
import builtins

original_import = builtins.__import__

def unavailable(name, *args, **kwargs):
    if name == "rdkit" or name.startswith("rdkit."):
        raise ImportError("injected unavailable RDKit")
    return original_import(name, *args, **kwargs)

builtins.__import__ = unavailable

import protrepair
from protrepair.errors import RdkitUnavailableError
from protrepair.io import FileFormat, read_structure_string, to_rdkit_molecule

structure = read_structure_string(
    "ATOM      1  N   GLY A   1      11.104  13.207   9.487"
    "  1.00 20.00           N  \\nEND\\n",
    FileFormat.PDB,
)
assert structure.geometry.atom_count() == 1
try:
    to_rdkit_molecule(structure)
except RdkitUnavailableError:
    pass
else:
    raise AssertionError("native export did not report unavailable RDKit")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("missing_module", ("Chem", "rdqueries"))
def test_native_export_unavailable_backend_uses_typed_failure(
    monkeypatch, missing_module
):
    structure = _smiles_structure("C")
    monkeypatch.setattr(rdkit_export, missing_module, None)
    with pytest.raises(RdkitUnavailableError):
        to_rdkit_molecule(structure)


def test_native_export_radical_refusal_identifies_the_atom_and_open_valence():
    source = _structure((atom_payload("C", "C", Vec3(0, 0, 0), formal_charge=0),), ())
    with pytest.raises(RdkitProjectionError) as caught:
        to_rdkit_molecule(source)
    assert source.constitution.atom_ref_at(AtomIndex(0)).display_token() in str(
        caught.value
    )
    assert "open valence" in str(caught.value)


@pytest.mark.parametrize(
    "relationship,order",
    (
        (BondRelationshipType.UNKNOWN, 1),
        (BondRelationshipType.COVALENT, None),
        (BondRelationshipType.METAL_COORDINATION, None),
    ),
)
def test_native_export_bond_refusal_identifies_both_endpoints(relationship, order):
    source = _structure(
        (
            atom_payload("H1", "H", Vec3(0, 0, 0), formal_charge=0),
            atom_payload("H2", "H", Vec3(0, 0, 1), formal_charge=0),
        ),
        ((0, 1, order, relationship, False),),
    )
    with pytest.raises(RdkitProjectionError) as caught:
        to_rdkit_molecule(source)
    for value in (0, 1):
        assert source.constitution.atom_ref_at(AtomIndex(value)).display_token() in str(
            caught.value
        )


@pytest.mark.parametrize(
    "name,smiles,padded_name",
    (
        ("CA", "C", " CA "),
        ("CA", "[Ca+2]", "CA  "),
        ("CL", "[Cl-]", "CL  "),
        ("H1", "[H][H]", " H1 "),
        ("1HG1", "[H][H]", "1HG1"),
        ("ABCDE", "C", "ABCDE"),
    ),
)
def test_native_residue_metadata_has_element_aware_pdb_names(name, smiles, padded_name):
    original = _smiles_structure(smiles)
    source = _structure(
        tuple(
            atom_payload(
                name if value == 0 else atom.name,
                atom.element,
                original.geometry.atom_geometry(AtomIndex(value)).position,
                formal_charge=original.topology.formal_charge(AtomIndex(value)),
            )
            for value, atom in enumerate(original.constitution.atom_slots)
        ),
        tuple(
            (
                bond.atom_index_1.value,
                bond.atom_index_2.value,
                bond.order,
                bond.relationship_type,
                bond.aromatic,
            )
            for bond in original.topology.bonds
        ),
    )
    native = to_rdkit_molecule(source)
    assert native.GetAtomWithIdx(0).GetPDBResidueInfo().GetName() == padded_name
    assert source.constitution.atom_site_at(AtomIndex(0)).name == name
    if len(name) <= 4:
        assert Chem.MolToPDBBlock(native).splitlines()[0][12:16] == padded_name


def test_native_metadata_binary_round_trip_requires_explicit_property_flags():
    native = to_rdkit_molecule(_smiles_structure("[2H]O[3H]"))
    without_properties = Chem.Mol(native.ToBinary(Chem.PropertyPickleOptions.NoProps))
    with pytest.raises(KeyError):
        without_properties.GetAtomWithIdx(0).GetBoolProp(
            "protrepair_formal_charge_known"
        )

    restored = Chem.Mol(native.ToBinary(Chem.PropertyPickleOptions.AllProps))
    assert restored.GetIntProp("protrepair_omitted_hydrogen_bonds") == 0
    for original, actual in zip(native.GetAtoms(), restored.GetAtoms(), strict=True):
        assert actual.GetPropsAsDict() == original.GetPropsAsDict()
        assert actual.GetIsotope() == original.GetIsotope()
    for original, actual in zip(native.GetBonds(), restored.GetBonds(), strict=True):
        assert actual.GetPropsAsDict() == original.GetPropsAsDict()
