"""External component assets must not fabricate charge or covalent chemistry."""

from pathlib import Path

import pytest

from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.io.nonstandard_component_ingress import (
    parse_bond_row,
    read_nonstandard_component_registry,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import BondRelationshipType


def _component_asset(
    *, charge: str = "0", charge_column: bool = True, bond_flag: str = ""
) -> str:
    charge_tag = "_chem_comp_atom.charge\n" if charge_column else ""
    first_charge = f" {charge}" if charge_column else ""
    second_charge = " 0" if charge_column else ""
    flag_tag, flag_value = bond_flag.split("=", 1) if bond_flag else ("", "")
    flag_column = f"_chem_comp_bond.{flag_tag}\n" if flag_tag else ""
    flag_cell = f" {flag_value}" if flag_tag else ""
    return (
        "data_TEST\n"
        "_chem_comp.id TEST\n"
        "_chem_comp.type NON-POLYMER\n"
        "_chem_comp.name 'External component'\n"
        "loop_\n"
        "_chem_comp_atom.atom_id\n"
        "_chem_comp_atom.type_symbol\n"
        f"{charge_tag}FE FE{first_charge}\nN1 N{second_charge}\n"
        "loop_\n"
        "_chem_comp_bond.atom_id_1\n"
        "_chem_comp_bond.atom_id_2\n"
        "_chem_comp_bond.value_order\n"
        "_chem_comp_bond.pdbx_aromatic_flag\n"
        f"{flag_column}FE N1 SING N{flag_cell}\n"
    )


@pytest.mark.parametrize("charge", ("?", ".", "'?'", "'.'"))
def test_component_asset_refuses_unknown_formal_charge(
    tmp_path: Path, charge: str
) -> None:
    path = tmp_path / "unknown-charge.cif"
    path.write_text(_component_asset(charge=charge), encoding="utf-8")

    with pytest.raises(ValueError, match=r"TEST.*FE.*formal charge"):
        read_nonstandard_component_registry(path)


@pytest.mark.parametrize("charge", ("-2", "0", "+3", "'0'"))
def test_component_asset_retains_known_formal_charge(
    tmp_path: Path, charge: str
) -> None:
    path = tmp_path / "known-charge.cif"
    path.write_text(_component_asset(charge=charge), encoding="utf-8")

    record = read_nonstandard_component_registry(path).get("TEST")
    assert record is not None
    assert record.atoms[0].formal_charge == int(charge.strip("'"))
    assert record.to_idealized_component().atoms[0].formal_charge == int(
        charge.strip("'")
    )


def test_component_asset_uses_dictionary_default_only_for_absent_charge_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "default-charge.cif"
    path.write_text(_component_asset(charge_column=False), encoding="utf-8")

    record = read_nonstandard_component_registry(path).get("TEST")
    assert record is not None
    assert tuple(atom.formal_charge for atom in record.atoms) == (0, 0)


@pytest.mark.parametrize("flag", ("pdbx_metal_coordination_flag", "pdbx_metal_pi_flag"))
@pytest.mark.parametrize("value", ("Y", "y", "'Y'", "?", ".", "X"))
def test_component_asset_refuses_non_covalent_or_unresolved_metal_flags(
    tmp_path: Path, flag: str, value: str
) -> None:
    path = tmp_path / "coordinate-bond.cif"
    path.write_text(_component_asset(bond_flag=f"{flag}={value}"), encoding="utf-8")

    with pytest.raises(ValueError, match=r"TEST.*FE.*N1.*metal"):
        read_nonstandard_component_registry(path)


@pytest.mark.parametrize(
    "flag", ("", "pdbx_metal_coordination_flag=N", "pdbx_metal_pi_flag='n'")
)
def test_component_asset_keeps_legacy_or_explicit_non_coordinate_bonds(
    tmp_path: Path, flag: str
) -> None:
    path = tmp_path / "covalent-bond.cif"
    path.write_text(_component_asset(bond_flag=flag), encoding="utf-8")

    record = read_nonstandard_component_registry(path).get("TEST")
    assert record is not None
    bond = record.to_idealized_component().bonds[0]
    assert (bond.atom_name_1, bond.atom_name_2, bond.order, bond.aromatic) == (
        "FE",
        "N1",
        1,
        False,
    )


def test_component_asset_failure_does_not_publish_a_partial_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.cif"
    valid = _component_asset().replace("TEST", "GOOD")
    path.write_text(valid + _component_asset(charge="?"), encoding="utf-8")

    with pytest.raises(ValueError, match=r"TEST.*FE.*formal charge"):
        read_nonstandard_component_registry(path)


@pytest.mark.parametrize("element", ("C", "H"))
def test_unknown_charge_refusal_is_not_limited_to_metal_atoms(
    tmp_path: Path, element: str
) -> None:
    path = tmp_path / "unknown-nonmetal.cif"
    path.write_text(
        _component_asset(charge="?").replace("FE FE ?", f"FE {element} ?"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"TEST.*FE.*formal charge"):
        read_nonstandard_component_registry(path)


@pytest.mark.parametrize("reversed_endpoints", (False, True))
@pytest.mark.parametrize("flags", (("N", "Y"), ("Y", "N"), ("N", "?")))
def test_both_metal_flags_are_checked_independently(
    tmp_path: Path, reversed_endpoints: bool, flags: tuple[str, str]
) -> None:
    coordination, metal_pi = flags
    text = _component_asset(bond_flag=f"pdbx_metal_coordination_flag={coordination}")
    text = text.replace(
        "_chem_comp_bond.pdbx_metal_coordination_flag\n",
        "_chem_comp_bond.pdbx_metal_coordination_flag\n"
        "_chem_comp_bond.pdbx_metal_pi_flag\n",
    ).replace(f"SING N {coordination}\n", f"SING N {coordination} {metal_pi}\n")
    if reversed_endpoints:
        text = text.replace("FE N1 SING", "N1 FE SING")
    path = tmp_path / "both-flags.cif"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="metal"):
        read_nonstandard_component_registry(path)


def test_direct_bond_lowering_cannot_erase_a_coordinate_flag() -> None:
    with pytest.raises(ValueError, match="metal coordination"):
        parse_bond_row(
            atom_name_1="N1",
            atom_name_2="FE",
            order_label="SING",
            aromatic_flag="N",
            component_id="TEST",
            metal_coordination_flag="Y",
        )


def test_explicit_null_charge_in_a_scalar_cif_category_is_not_defaulted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scalar.cif"
    path.write_text(
        "data_TEST\n_chem_comp.id TEST\n_chem_comp.type NON-POLYMER\n"
        "_chem_comp.name 'Single atom'\n_chem_comp_atom.atom_id FE\n"
        "_chem_comp_atom.type_symbol FE\n_chem_comp_atom.charge ?\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"TEST.*FE.*formal charge"):
        read_nonstandard_component_registry(path)


@pytest.mark.parametrize(
    "column",
    ("atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"),
)
def test_incomplete_bond_columns_cannot_hide_declared_coordination(
    tmp_path: Path, column: str
) -> None:
    text = _component_asset(bond_flag="pdbx_metal_coordination_flag=Y")
    tags = ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"]
    values = ["FE", "N1", "SING", "N", "Y"]
    del values[tags.index(column)]
    text = text.replace(f"_chem_comp_bond.{column}\n", "")
    text = text.replace("FE N1 SING N Y\n", " ".join(values) + "\n")
    path = tmp_path / "incomplete-bonds.cif"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=r"TEST.*incomplete bond columns"):
        read_nonstandard_component_registry(path)


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_2d6c_ingress_keeps_unresolved_atoms_and_typed_coordination(
    file_format: FileFormat,
) -> None:
    source = read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb2d6c.ent")
    restored = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    for chain in ("A", "B"):
        for seq, count in ((154, 43), (155, 5)):
            residue_id = ResidueId(chain, seq)
            original = source.constitution.residue_or_ligand(residue_id)
            roundtrip = restored.constitution.residue_or_ligand(residue_id)
            assert original is not None and roundtrip is not None
            assert roundtrip.atom_site_names() == original.atom_site_names()
            assert len(roundtrip.atom_sites) == count
            assert not any(atom.is_hydrogen() for atom in roundtrip.atom_sites)
            for atom in original.atom_sites:
                ref = AtomRef(residue_id, atom.name)
                first = source.constitution.atom_index(ref)
                second = restored.constitution.atom_index(ref)
                assert restored.topology.formal_charge(second) is None
                assert restored.geometry.position(second) == source.geometry.position(
                    first
                )

        iron = restored.constitution.atom_index(AtomRef(ResidueId(chain, 154), "FE"))
        for seq, name in ((93, "NE2"), (155, "N1")):
            partner = restored.constitution.atom_index(
                AtomRef(ResidueId(chain, seq), name)
            )
            bond = restored.topology.bond_between(iron, partner)
            assert bond is not None
            assert bond.relationship_type is BondRelationshipType.METAL_COORDINATION
            assert bond.source_metadata is not None
            distance = restored.geometry.position(iron).distance_to(
                restored.geometry.position(partner)
            )
            decimal_places = 2 if file_format is FileFormat.PDB else 4
            assert bond.source_metadata.reported_distance_angstrom == round(
                distance, decimal_places
            )
