"""Standard carbonyl chemistry survives topology and force-field projection."""

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry.component.topology import (
    template_resolved_topology_bonds_for_new_atoms,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure_string, write_structure_string
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.continuous.rdkit import build_rdkit_molecule
from protrepair.transformer.local import LocalScopeSpec


@pytest.mark.parametrize(
    "component_id",
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS "
        "MET PHE PRO SER THR TRP TYR VAL"
    ).split(),
)
def test_standard_backbone_carbonyl_is_a_double_bond(component_id: str) -> None:
    definition = build_standard_component_library().require(component_id).definition
    carbonyl = next(
        bond
        for bond in definition.bonds
        if {bond.atom_name_1, bond.atom_name_2} == {"C", "O"}
    )
    assert carbonyl.order == 2
    assert not carbonyl.aromatic
    terminal = next(
        bond
        for bond in definition.bonds
        if {bond.atom_name_1, bond.atom_name_2} == {"C", "OXT"}
    )
    assert terminal.order == 1


@pytest.mark.parametrize(
    "component_id,carbon,oxygen", [("ASN", "CG", "OD1"), ("GLN", "CD", "OE1")]
)
def test_new_amide_oxygen_projects_a_double_bond(
    component_id: str, carbon: str, oxygen: str
) -> None:
    residue_id = ResidueId("A", 1)
    library = build_standard_component_library()
    definition = library.require(component_id).definition

    def structure(include_oxygen: bool) -> ProteinStructure:
        names = [
            name
            for name in definition.atom_names
            if name != "OXT" and (include_oxygen or name != oxygen)
        ]
        return build_structure(
            chains=(
                chain_payload(
                    "A",
                    (
                        residue_payload(
                            residue_id=residue_id,
                            component_id=component_id,
                            atoms=tuple(
                                atom_payload(name, name[0], Vec3(float(i), 0.0, 0.0))
                                for i, name in enumerate(names)
                            ),
                        ),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
        )

    before, after = structure(False), structure(True)
    bonds = template_resolved_topology_bonds_for_new_atoms(
        source_constitution=before.constitution,
        target_constitution=after.constitution,
        component_library=library,
    )
    assert len(bonds) == 1
    bond = bonds[0]
    assert {
        after.constitution.atom_ref_at(bond.atom_index_1),
        after.constitution.atom_ref_at(bond.atom_index_2),
    } == {AtomRef(residue_id, carbon), AtomRef(residue_id, oxygen)}
    assert bond.order == 2


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize(
    "residue_number,carbon_name,oxygen_name",
    ((4, "C", "O"), (25, "CG", "OD1"), (49, "CD", "OE1")),
)
def test_actual_rdkit_carbonyl_is_sp2_without_artificial_radicals(
    file_format: FileFormat,
    residue_number: int,
    carbon_name: str,
    oxygen_name: str,
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures/pdb/span-reconstruction/1ubq-short-gaps.pdb"
    )
    structure = read_structure_string(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not line.startswith("CONECT")
        ),
        FileFormat.PDB,
    )
    if file_format is FileFormat.MMCIF:
        structure = read_structure_string(
            write_structure_string(structure, FileFormat.MMCIF), FileFormat.MMCIF
        )
    library = build_standard_component_library()
    residue_id = ResidueId("A", residue_number)
    problem = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        component_library=library,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    carbon, oxygen = (
        mapping[structure.constitution.atom_index(AtomRef(residue_id, name))]
        for name in (carbon_name, oxygen_name)
    )
    assert molecule.GetBondBetweenAtoms(carbon, oxygen).GetBondTypeAsDouble() == 2.0
    for index in (carbon, oxygen):
        atom = molecule.GetAtomWithIdx(index)
        assert atom.GetHybridization() == Chem.HybridizationType.SP2
        assert atom.GetNumRadicalElectrons() == 0
    parameters = rdForceFieldHelpers.GetUFFBondStretchParams(molecule, carbon, oxygen)
    assert parameters is not None
    assert 1.15 < parameters[1] < 1.30
