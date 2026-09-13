"""Fixed standard rings retain chemical valence through repair and projection."""

from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.io import FileFormat, read_structure_string, write_structure_string
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondProvenance, StructureTopology
from protrepair.transformer.completion.heavy import repair_heavy_atoms
from protrepair.transformer.completion.hydrogen import materialize_hydrogens_core
from protrepair.transformer.continuous.rdkit import build_rdkit_molecule
from protrepair.transformer.local import LocalScopeSpec

# Heavy ring edges from the wwPDB CCD; the localized double edges are one Kekule form.
_RING_ORDERS = {
    "PHE": {
        ("CG", "CD1"): 2,
        ("CG", "CD2"): 1,
        ("CD1", "CE1"): 1,
        ("CD2", "CE2"): 2,
        ("CE1", "CZ"): 2,
        ("CE2", "CZ"): 1,
    },
    "TYR": {
        ("CG", "CD1"): 2,
        ("CG", "CD2"): 1,
        ("CD1", "CE1"): 1,
        ("CD2", "CE2"): 2,
        ("CE1", "CZ"): 2,
        ("CE2", "CZ"): 1,
    },
    "TRP": {
        ("CG", "CD1"): 2,
        ("CG", "CD2"): 1,
        ("CD1", "NE1"): 1,
        ("NE1", "CE2"): 1,
        ("CE2", "CD2"): 2,
        ("CD2", "CE3"): 1,
        ("CE2", "CZ2"): 1,
        ("CE3", "CZ3"): 2,
        ("CZ2", "CH2"): 2,
        ("CZ3", "CH2"): 1,
    },
}
_RESIDUE_NUMBERS = {"PHE": 22, "TYR": 15, "TRP": 107}
_CORPUS = Path(__file__).resolve().parents[1] / "fixtures/corpus/pdb1afc.ent"


def _input_text(component_id: str, omitted: frozenset[str] = frozenset()) -> str:
    number = _RESIDUE_NUMBERS[component_id]
    return (
        "\n".join(
            line
            for line in _CORPUS.read_text().splitlines()
            if line.startswith("ATOM")
            and line[21] == "A"
            and number - 1 <= int(line[22:26]) <= number + 1
            and not (int(line[22:26]) == number and line[12:16].strip() in omitted)
        )
        + "\nEND\n"
    )


def _ring_names(component_id: str) -> frozenset[str]:
    return frozenset(name for pair in _RING_ORDERS[component_id] for name in pair)


def _assert_native_ring(structure: ProteinStructure, component_id: str) -> None:
    residue_id = ResidueId("A", _RESIDUE_NUMBERS[component_id])
    problem = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        component_library=build_standard_component_library(),
    )
    molecule, mapping = build_rdkit_molecule(problem)
    indices = {
        name: mapping[structure.constitution.atom_index(AtomRef(residue_id, name))]
        for name in _ring_names(component_id)
    }
    for index in indices.values():
        atom = molecule.GetAtomWithIdx(index)
        assert atom.GetIsAromatic()
        assert atom.GetHybridization() == Chem.HybridizationType.SP2
        assert atom.GetNumRadicalElectrons() == 0
        assert atom.GetFormalCharge() == 0

    for left, right in _RING_ORDERS[component_id]:
        bond = molecule.GetBondBetweenAtoms(indices[left], indices[right])
        assert bond.GetIsAromatic()
        assert bond.GetBondTypeAsDouble() == 1.5
        parameters = rdForceFieldHelpers.GetUFFBondStretchParams(
            molecule, indices[left], indices[right]
        )
        assert parameters is not None
        assert 1.30 < parameters[1] < 1.45

    if component_id == "TRP":
        nitrogen = molecule.GetAtomWithIdx(indices["NE1"])
        assert nitrogen.GetTotalValence() == 3
        assert sum(atom.GetAtomicNum() == 1 for atom in nitrogen.GetNeighbors()) == 1


@pytest.mark.parametrize("component_id", tuple(_RING_ORDERS))
def test_standard_ring_definition_has_ccd_kekule_orders(component_id: str) -> None:
    definition = build_standard_component_library().require(component_id).definition
    actual = {
        frozenset((bond.atom_name_1, bond.atom_name_2)): bond
        for bond in definition.bonds
    }
    for pair, order in _RING_ORDERS[component_id].items():
        bond = actual[frozenset(pair)]
        assert bond.order == order
        assert not bond.aromatic  # Integral graph, not a forced backend aromatic type.
    assert actual[frozenset(("CB", "CG"))].order == 1
    if component_id == "TYR":
        assert actual[frozenset(("CZ", "OH"))].order == 1
    assert not definition.formal_charges


@pytest.mark.parametrize("component_id", tuple(_RING_ORDERS))
@pytest.mark.parametrize("file_format", (None, FileFormat.PDB, FileFormat.MMCIF))
def test_hydrogenated_ring_is_aromatic_in_actual_ff_input(
    component_id: str, file_format: FileFormat | None
) -> None:
    source = read_structure_string(_input_text(component_id), FileFormat.PDB)
    structure = materialize_hydrogens_core(source).structure
    for index in range(len(source.geometry.atom_geometries)):
        atom_ref = source.constitution.atom_ref_at(AtomIndex(index))
        restored_index = structure.constitution.atom_index(atom_ref)
        assert (
            structure.geometry.atom_geometries[restored_index.value]
            == (source.geometry.atom_geometries[index])
        )
    if file_format is not None:
        structure = read_structure_string(
            write_structure_string(structure, file_format), file_format
        )
    residue_id = ResidueId("A", _RESIDUE_NUMBERS[component_id])
    for (left, right), order in _RING_ORDERS[component_id].items():
        bond = structure.topology.bond_between(
            structure.constitution.atom_index(AtomRef(residue_id, left)),
            structure.constitution.atom_index(AtomRef(residue_id, right)),
        )
        assert bond is not None
        assert bond.order == order
        assert bond.provenance is (
            BondProvenance.TEMPLATE_RESOLVED
            if file_format is None
            else BondProvenance.SOURCE_EXPLICIT
        )
    _assert_native_ring(structure, component_id)


@pytest.mark.parametrize(
    "component_id,missing_atom",
    [
        (component, atom)
        for component in _RING_ORDERS
        for atom in sorted(_ring_names(component))
    ],
)
def test_repaired_ring_atom_restores_its_chemical_bonds(
    component_id: str, missing_atom: str
) -> None:
    source = read_structure_string(
        _input_text(component_id, frozenset((missing_atom,))), FileFormat.PDB
    )
    result = repair_heavy_atoms(source)
    residue_id = ResidueId("A", _RESIDUE_NUMBERS[component_id])
    residue = result.structure.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    assert residue.has_atom_site(missing_atom)
    hydrogenated = materialize_hydrogens_core(result.structure).structure
    for file_format in (None, FileFormat.PDB, FileFormat.MMCIF):
        projected = (
            hydrogenated
            if file_format is None
            else read_structure_string(
                write_structure_string(hydrogenated, file_format), file_format
            )
        )
        for (left, right), order in _RING_ORDERS[component_id].items():
            bond = projected.topology.bond_between(
                projected.constitution.atom_index(AtomRef(residue_id, left)),
                projected.constitution.atom_index(AtomRef(residue_id, right)),
            )
            assert bond is not None
            assert bond.order == order
            assert bond.provenance is (
                BondProvenance.TEMPLATE_RESOLVED
                if file_format is None
                else BondProvenance.SOURCE_EXPLICIT
            )
    _assert_native_ring(hydrogenated, component_id)


@pytest.mark.parametrize("component_id", tuple(_RING_ORDERS))
def test_custom_component_orders_are_not_replaced_by_standard_defaults(
    component_id: str,
) -> None:
    template = build_standard_component_library().require(component_id)
    custom = replace(
        template,
        definition=replace(
            template.definition,
            bonds=tuple(replace(bond, order=1) for bond in template.definition.bonds),
        ),
    )
    library = build_standard_component_library().with_template(custom)
    source = read_structure_string(
        _input_text(component_id, _ring_names(component_id)), FileFormat.PDB
    )
    structure = repair_heavy_atoms(source, component_library=library).structure
    residue_id = ResidueId("A", _RESIDUE_NUMBERS[component_id])
    for left, right in _RING_ORDERS[component_id]:
        bond = structure.topology.bond_between(
            structure.constitution.atom_index(AtomRef(residue_id, left)),
            structure.constitution.atom_index(AtomRef(residue_id, right)),
        )
        assert bond is not None
        assert bond.order == 1


@pytest.mark.parametrize("component_id", ("PHE", "TYR"))
def test_alternate_source_kekule_form_is_retained_and_native_equivalent(
    component_id: str,
) -> None:
    original = materialize_hydrogens_core(
        read_structure_string(_input_text(component_id), FileFormat.PDB)
    ).structure
    residue_id = ResidueId("A", _RESIDUE_NUMBERS[component_id])
    ring_indices = {
        original.constitution.atom_index(AtomRef(residue_id, name))
        for name in _ring_names(component_id)
    }
    bonds = tuple(
        replace(bond, order=3 - bond.order)
        if bond.atom_index_1 in ring_indices
        and bond.atom_index_2 in ring_indices
        and bond.order is not None
        else bond
        for bond in original.topology.bonds
    )
    alternate = ProteinStructure.from_payload(
        constitution=original.constitution,
        geometry=original.geometry,
        topology=StructureTopology(
            constitution=original.constitution,
            atom_topologies=original.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=original.polymer_blueprint,
        provenance=original.provenance,
    )
    # mmCIF explicitly distinguishes source single orders from unspecified order.
    restored = read_structure_string(
        write_structure_string(alternate, FileFormat.MMCIF), FileFormat.MMCIF
    )
    for (left, right), order in _RING_ORDERS[component_id].items():
        bond = restored.topology.bond_between(
            restored.constitution.atom_index(AtomRef(residue_id, left)),
            restored.constitution.atom_index(AtomRef(residue_id, right)),
        )
        assert bond is not None
        assert bond.order == 3 - order
        assert not bond.aromatic
    assert restored.constitution == original.constitution
    assert restored.topology.atom_topologies == original.topology.atom_topologies
    _assert_native_ring(restored, component_id)


@pytest.mark.parametrize("order", (1, 3))
def test_source_order_conflict_is_not_hidden_by_forced_aromatic_binding(
    order: int,
) -> None:
    original = materialize_hydrogens_core(
        read_structure_string(_input_text("PHE"), FileFormat.PDB)
    ).structure
    residue_id = ResidueId("A", _RESIDUE_NUMBERS["PHE"])
    pair = frozenset(
        original.constitution.atom_index(AtomRef(residue_id, name))
        for name in ("CG", "CD1")
    )
    changed = ProteinStructure.from_payload(
        constitution=original.constitution,
        geometry=original.geometry,
        topology=StructureTopology(
            constitution=original.constitution,
            atom_topologies=original.topology.atom_topologies,
            bonds=tuple(
                replace(bond, order=order)
                if frozenset(bond.endpoint_pair()) == pair
                else bond
                for bond in original.topology.bonds
            ),
        ),
        polymer_blueprint=original.polymer_blueprint,
        provenance=original.provenance,
    )
    restored = read_structure_string(
        write_structure_string(changed, FileFormat.MMCIF), FileFormat.MMCIF
    )
    problem = build_continuous_relaxation_problem(
        restored,
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        component_library=build_standard_component_library(),
    )
    molecule, mapping = build_rdkit_molecule(problem)
    indices = [
        mapping[restored.constitution.atom_index(AtomRef(residue_id, name))]
        for name in ("CG", "CD1")
    ]
    bond = molecule.GetBondBetweenAtoms(*indices)
    assert bond.GetBondTypeAsDouble() == order
    assert not bond.GetIsAromatic()
    assert restored.constitution == original.constitution
    assert restored.topology.atom_topologies == original.topology.atom_topologies


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_ring_resolution_preserves_source_phenolate_charge_and_hydrogens(
    file_format: FileFormat,
) -> None:
    hydrogenated = materialize_hydrogens_core(
        read_structure_string(_input_text("TYR"), FileFormat.PDB)
    ).structure
    number = _RESIDUE_NUMBERS["TYR"]
    lines = []
    for line in write_structure_string(hydrogenated, FileFormat.PDB).splitlines():
        if not line.startswith("ATOM"):
            continue
        if int(line[22:26]) == number:
            if line[12:16].strip() == "HH":
                continue
            if line[12:16].strip() == "OH":
                line = line[:78].ljust(78) + "1-"
        lines.append(line)

    source = read_structure_string("\n".join(lines), FileFormat.PDB)
    restored = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    assert restored.constitution == source.constitution
    assert restored.geometry == source.geometry
    assert restored.topology.atom_topologies == source.topology.atom_topologies
    oxygen = restored.constitution.atom_index(AtomRef(ResidueId("A", number), "OH"))
    oxygen_topology = restored.topology.atom_topologies[oxygen.value]
    assert oxygen_topology is not None
    assert oxygen_topology.formal_charge == -1
    assert (
        restored.constitution.resolve_atom_index(AtomRef(ResidueId("A", number), "HH"))
        is None
    )
    _assert_native_ring(restored, "TYR")


@pytest.mark.parametrize("component_id", tuple(_RING_ORDERS))
def test_incomplete_ring_retains_integral_bonds_in_local_ff_scope(
    component_id: str,
) -> None:
    missing = _ring_names(component_id).difference({"CG", "CD1"})
    structure = read_structure_string(
        _input_text(component_id, missing), FileFormat.PDB
    )
    residue_id = ResidueId("A", _RESIDUE_NUMBERS[component_id])
    refs = tuple(AtomRef(residue_id, name) for name in ("CG", "CD1"))
    problem = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(refs),
        component_library=build_standard_component_library(),
        context_radius_angstrom=0.1,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    left, right = (mapping[structure.constitution.atom_index(ref)] for ref in refs)
    bond = molecule.GetBondBetweenAtoms(left, right)
    assert bond.GetBondTypeAsDouble() == 2.0
    assert not bond.GetIsAromatic()
