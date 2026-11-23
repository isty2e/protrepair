"""File validity, native stereo fidelity and value semantics of named CCD graphs."""

from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from protrepair.chemistry.inference.ccd_graph import component_heavy_molecule
from protrepair.chemistry.inference.rdkit_graph import (
    require_compatible_pose_stereochemistry,
)
from protrepair.io import read_retained_component_definitions
from protrepair.sources.chemistry import (
    RetainedNonPolymerChemistryOverride,
    override_by_residue_id,
)
from protrepair.structure import ResidueId
from protrepair.workflow.contracts import StructureIngressOptions

if TYPE_CHECKING:
    from typing_extensions import assert_type

DATA = Path(__file__).resolve().parents[1] / "fixtures/chemistry/named-ccd/ligands.cif"
RID = ResidueId("A", 999)
AMINE = """data_AMN
_chem_comp.id AMN
_chem_comp.pdbx_formal_charge 0
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
AMN CM C
AMN NT N
AMN HA H
AMN HB H
AMN HC H
AMN HD H
AMN HE H
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
AMN CM NT SING
AMN CM HA SING
AMN CM HB SING
AMN CM HC SING
AMN NT HD SING
AMN NT HE SING
"""


def test_collection_lookup_and_assignment_use_model_values(tmp_path):
    definitions = read_retained_component_definitions(DATA)
    model = definitions[" aq4 "]
    assert set(definitions) == {"AQ4", "LAC", "FUM"}
    with pytest.raises(KeyError):
        definitions["absent"]
    with pytest.raises(TypeError, match="ResidueId"):
        model.for_residue("A:999")  # pyrefly: ignore[bad-argument-type]
    other = tmp_path / "other.cif"
    other.write_text(DATA.read_text())
    copy = read_retained_component_definitions(other)["AQ4"]
    left = StructureIngressOptions(
        retained_non_polymer_chemistry_overrides=[model.for_residue(RID)]
    )
    right = StructureIngressOptions(
        retained_non_polymer_chemistry_overrides=[copy.for_residue(RID)]
    )
    assert {left: "chosen"}[right] == "chosen"
    with pytest.raises(ValueError, match="duplicate component ID AQ4"):
        read_retained_component_definitions(DATA, other)

    other.write_text(AMINE)
    assert set(read_retained_component_definitions(other, DATA)) == {
        "AMN",
        "AQ4",
        "FUM",
        "LAC",
    }
    smiles = RetainedNonPolymerChemistryOverride(RID, "CO", ("C", "O"))
    if TYPE_CHECKING:
        assert_type(
            override_by_residue_id([smiles])[RID], RetainedNonPolymerChemistryOverride
        )
    for args in (
        [model.for_residue(RID)] * 2,
        [smiles, model.for_residue(RID)],
        [model.for_residue(RID), smiles],
    ):
        with pytest.raises(ValueError, match="repeat"):
            StructureIngressOptions(retained_non_polymer_chemistry_overrides=args)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("AMN NT HE SING", ""),
        ("AMN NT HE SING", "AMN NT HE DOUB"),
        ("AMN NT HE SING", "AMN NT no_atom SING"),
        ("AMN CM HA SING", "AMN CM CM SING"),
        ("AMN CM NT SING", "AMN CM NT SING\nAMN NT CM SING"),
        ("AMN NT N", "AMN NT N\nAMN nt N"),
        ("AMN HA H", "AMN HA D"),
        ("AMN CM NT SING", "AMN CM NT METAL"),
        ("AMN CM NT SING", "XYZ CM NT SING"),
        ("pdbx_formal_charge 0", "pdbx_formal_charge 1"),
    ],
)
def test_reader_rejects_bad_chemistry(tmp_path, before, after):
    file = tmp_path / "input.cif"
    file.write_text(AMINE.replace(before, after))
    with pytest.raises(ValueError, match="component AMN"):
        read_retained_component_definitions(file)


def test_omitted_charge_is_zero_but_an_explicit_null_is_not(tmp_path):
    file = tmp_path / "input.cif"
    file.write_text(AMINE)
    assert (
        sum(a.charge for a in read_retained_component_definitions(file)["AMN"].atoms)
        == 0
    )
    doc = gemmi.cif.read_string(AMINE)
    block = doc.sole_block()
    table = block.get_mmcif_category("_chem_comp_atom.")
    table["charge"] = ["?"] + ["0"] * 6
    block.set_mmcif_category("_chem_comp_atom.", table, raw=True)
    doc.write_file(str(file))
    with pytest.raises(ValueError, match="AMN"):
        read_retained_component_definitions(file)


def test_disconnected_closed_shell_fragments_and_coordinates_are_not_definitions(
    tmp_path,
):
    file = tmp_path / "input.cif"
    file.write_text(
        AMINE.replace("AMN CM C", "AMN F1 F\nAMN F2 F\nAMN CM C") + "AMN F1 F2 SING\n"
    )
    with pytest.raises(ValueError, match="disconnected"):
        read_retained_component_definitions(file)
    file.write_text("data_coordinates\n_atom_site.id 1\n")
    with pytest.raises(ValueError, match="not a CCD"):
        read_retained_component_definitions(file)


@pytest.mark.parametrize(
    ("code", "declared", "opposite"), [("LAC", "R", "S"), ("FUM", "E", "Z")]
)
def test_declared_stereo_controls_the_pose_without_using_ideal_geometry_as_a_target(
    tmp_path, code, declared, opposite
):
    block = gemmi.cif.read_file(str(DATA)).find_block(code)
    assert block is not None
    doc = gemmi.cif.Document()
    block = doc.add_copied_block(block)
    xyz = {
        row[0]: tuple(float(x) for x in tuple(row)[1:])
        for row in block.find(
            "_chem_comp_atom.",
            [
                "atom_id",
                "pdbx_model_Cartn_x_ideal",
                "pdbx_model_Cartn_y_ideal",
                "pdbx_model_Cartn_z_ideal",
            ],
        )
    }
    models = []
    for wrong in (False, True):
        if wrong:
            cat = "_chem_comp_atom." if code == "LAC" else "_chem_comp_bond."
            column = block.find_values(cat + "pdbx_stereo_config")
            column[list(column).index(declared)] = opposite
        file = tmp_path / "model.cif"
        doc.write_file(str(file))
        model = read_retained_component_definitions(file)[code]
        models.append(model.for_residue(RID))
        native = component_heavy_molecule(model)
        pose = Chem.Conformer(native.GetNumAtoms())
        pose.Set3D(True)
        for i, name in enumerate(model.heavy_atom_names):
            pose.SetAtomPosition(i, xyz[name])
        native.AddConformer(pose)
        if wrong:
            with pytest.raises(ValueError, match="stereochemistry contradicts"):
                require_compatible_pose_stereochemistry(native)
        else:
            require_compatible_pose_stereochemistry(native)
    assert len(set(models)) == 2


def test_row_order_does_not_change_the_graph_or_slot_assignment(tmp_path):
    doc = gemmi.cif.read_string(AMINE)
    block = doc.sole_block()
    file = tmp_path / "model.cif"
    doc.write_file(str(file))
    before = read_retained_component_definitions(file)["AMN"]
    for cat in ("_chem_comp_atom.", "_chem_comp_bond."):
        cols = block.get_mmcif_category(cat)
        block.set_mmcif_category(cat, {k: list(reversed(v)) for k, v in cols.items()})
    doc.write_file(str(file))
    after = read_retained_component_definitions(file)["AMN"]
    assert after.for_residue(RID) == before.for_residue(RID)
    assert after.occupied_slots({"DB": "CM", "HD": "NT", "X": "NT"}) == {
        "DB": "HB",
        "HD": "HD",
        "X": "HE",
    }


def _write_graph(path, atoms, edges):
    doc = gemmi.cif.Document()
    b = doc.add_new_block("TST")
    b.set_pair("_chem_comp.id", "TST")
    b.set_pair("_chem_comp.pdbx_formal_charge", "?")
    atom_loop = b.init_loop(
        "_chem_comp_atom.",
        ["comp_id", "atom_id", "type_symbol", "charge", "pdbx_stereo_config"],
    )
    h_bonds = []
    for name, element, charge, h_count, stereo in atoms:
        atom_loop.add_row(["TST", name, element, str(charge), stereo])
        for i in range(h_count):
            h = f"H{name}{i}"
            atom_loop.add_row(["TST", h, "H", "0", "N"])
            h_bonds.append((name, h, "SING", "N"))
    bond_loop = b.init_loop(
        "_chem_comp_bond.",
        ["comp_id", "atom_id_1", "atom_id_2", "value_order", "pdbx_stereo_config"],
    )
    for edge in (*edges, *h_bonds):
        bond_loop.add_row(["TST", *edge])
    doc.write_file(str(path))
    return read_retained_component_definitions(path)["TST"]


def test_dependent_atom_stereo_survives_e_z_assignment(tmp_path):
    # The middle carbon's two alkenyl arms differ only in double-bond stereo.
    counts = (3, 1, 1, 1, 1, 1, 3)
    atoms = [
        (f"C{i}", "C", 0, h, "R" if i == 4 else "N") for i, h in enumerate(counts, 1)
    ]
    atoms.append(("F", "F", 0, 0, "N"))
    bonds = [
        (
            f"C{i}",
            f"C{i + 1}",
            "DOUB" if i in (2, 5) else "SING",
            {2: "E", 5: "Z"}.get(i, "N"),
        )
        for i in range(1, 7)
    ]
    model = _write_graph(
        tmp_path / "stereo.cif", atoms, [*bonds, ("C4", "F", "SING", "N")]
    )
    mol = component_heavy_molecule(model)
    rdCIPLabeler.AssignCIPLabels(mol)
    assert (
        mol.GetAtomWithIdx(model.heavy_atom_names.index("C4")).GetProp("_CIPCode")
        == "R"
    )
    for a, b, label in (("C2", "C3", "E"), ("C5", "C6", "Z")):
        bond = mol.GetBondBetweenAtoms(
            model.heavy_atom_names.index(a), model.heavy_atom_names.index(b)
        )
        assert bond.GetProp("_CIPCode") == label


def test_aromatic_carbanion_keeps_its_explicit_hydrogens(tmp_path):
    atoms = [(f"C{i}", "C", -1 if i == 0 else 0, 1, "N") for i in range(5)]
    bonds = [
        (f"C{i}", f"C{(i + 1) % 5}", "DOUB" if i in (1, 3) else "SING", "N")
        for i in range(5)
    ]
    model = _write_graph(tmp_path / "ring.cif", atoms, bonds)
    mol = component_heavy_molecule(model)
    assert all(a.GetIsAromatic() and a.GetTotalNumHs() == 1 for a in mol.GetAtoms())
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == -1
