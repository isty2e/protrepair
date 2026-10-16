"""Curated isolated ligands retain source evidence and realize chemical graphs."""

from dataclasses import replace
from math import isfinite
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.nonstandard_component_ingress import (
    read_nonstandard_component_registry,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
)
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import BondProvenance
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.rdkit import (
    build_rdkit_force_field,
    build_rdkit_molecule,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec

FIXTURES = Path(__file__).parents[1] / "fixtures"
SOURCES = {"SAN": "1aj0", "CMP": "1hw5"}
HYDROGEN_COUNTS = {"SAN": 8, "CMP": 12}


def _source_text(component_id: str) -> str:
    text = (FIXTURES / f"corpus/pdb{SOURCES[component_id]}.ent").read_text()
    return (
        "\n".join(
            line
            for line in text.splitlines()
            if line.startswith("HETATM") and line[17:20] == component_id
        )
        + "\nEND\n"
    )


def _source(component_id: str) -> ProteinStructure:
    return read_structure_string(_source_text(component_id), FileFormat.PDB)


def _assert_source_preserved(
    source: ProteinStructure, target: ProteinStructure
) -> None:
    for old, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old)
        new = target.constitution.atom_index(ref)
        assert target.constitution.atom_site_at(
            new
        ) == source.constitution.atom_site_at(old)
        assert target.geometry.atom_geometry(new) == source.geometry.atom_geometry(old)
        if source.topology.formal_charge(old) is not None:
            assert target.topology.formal_charge(new) == source.topology.formal_charge(
                old
            )
    for old in source.topology.bonds:
        if old.provenance is not BondProvenance.SOURCE_EXPLICIT:
            continue
        refs = tuple(source.constitution.atom_ref_at(i) for i in old.endpoint_pair())
        new = target.topology.bond_between(
            *(target.constitution.atom_index(ref) for ref in refs)
        )
        assert new is not None and new.source_metadata == old.source_metadata
        assert new.provenance is old.provenance


@pytest.mark.parametrize("component_id", ("ACE", "SAN", "CMP", "FUC", "GAL"))
def test_curated_records_match_frozen_primary_ccd(component_id):
    frozen = read_nonstandard_component_registry(
        FIXTURES / f"chemistry/retained/{component_id}.cif"
    ).get(component_id)
    bundled = build_bundled_retained_non_polymer_registry().get(component_id)
    assert bundled is not None and frozen is not None
    assert (
        replace(
            bundled,
            bond_targets=frozen.bond_targets,
            angle_targets=frozen.angle_targets,
        )
        == frozen
    )
    for actual, expected in zip(bundled.bond_targets, frozen.bond_targets, strict=True):
        assert actual.target_distance_angstrom == pytest.approx(
            expected.target_distance_angstrom, rel=0.0, abs=1e-8
        )
        assert (
            replace(actual, target_distance_angstrom=expected.target_distance_angstrom)
            == expected
        )
    for actual, expected in zip(
        bundled.angle_targets, frozen.angle_targets, strict=True
    ):
        assert actual.target_angle_degrees == pytest.approx(
            expected.target_angle_degrees, rel=0.0, abs=1e-6
        )
        assert (
            replace(actual, target_angle_degrees=expected.target_angle_degrees)
            == expected
        )
    template = build_default_component_library().require(component_id)
    assert dict(template.heavy_atom_charge_requirements()) == {
        atom.atom_name: atom.formal_charge
        for atom in bundled.atoms
        if not atom.is_hydrogen()
    }


@pytest.mark.parametrize("component_id", ("SAN", "CMP"))
@pytest.mark.parametrize("strict", (False, True))
def test_actual_standalone_ligands_complete_graph_without_fallback(
    component_id, strict
):
    source = _source(component_id)
    result = add_retained_non_polymer_hydrogens(
        source, allow_retained_non_polymer_rdkit_fallback=not strict
    )
    assert not result.issues
    _assert_source_preserved(source, result.structure)
    facts = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert len(facts) == len(source.constitution.ligands)
    assert all(
        fact.heavy_topology_source is RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
        and fact.heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
        and fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
        for fact in facts
    )
    record = build_bundled_retained_non_polymer_registry().get(component_id)
    assert record is not None
    expected = {
        frozenset((b.atom_name_1, b.atom_name_2)): (b.order, b.aromatic)
        for b in record.bonds
    }
    for residue in result.structure.constitution.ligands:
        assert (
            sum(a.is_hydrogen() for a in residue.atom_sites)
            == HYDROGEN_COUNTS[component_id]
        )
        observed = {}
        for bond in result.structure.topology.bonds:
            first, second = (
                result.structure.constitution.atom_ref_at(i)
                for i in bond.endpoint_pair()
            )
            if first.residue_id == second.residue_id == residue.residue_id:
                observed[frozenset((first.atom_name, second.atom_name))] = (
                    bond.order,
                    bond.aromatic,
                )
        assert observed == expected
    repeated = add_retained_non_polymer_hydrogens(result.structure)
    assert not repeated.issues and not repeated.repairs
    assert repeated.structure.constitution == result.structure.constitution
    assert repeated.structure.geometry == result.structure.geometry
    assert repeated.structure.topology == result.structure.topology


@pytest.mark.parametrize("component_id", ("SAN", "CMP"))
@pytest.mark.parametrize(
    "family", (ContinuousRelaxationForceField.UFF, ContinuousRelaxationForceField.MMFF)
)
def test_actual_standalone_graph_binds_closed_shell_native_field(component_id, family):
    source = _source(component_id)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    structure = result.structure
    refs = tuple(
        structure.constitution.atom_ref_at(i)
        for i, _ in structure.geometry.iter_positions()
    )
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_atoms(refs),
        component_library=build_default_component_library(),
        force_field=family,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    assert Chem.GetFormalCharge(molecule) == 0
    assert all(a.GetNumRadicalElectrons() == 0 for a in molecule.GetAtoms())
    assert molecule.GetNumAtoms() == len(refs)
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    field = build_rdkit_force_field(
        molecule,
        problem=problem,
        restraint_library=build_default_restraint_library(),
        rdkit_atom_index_by_structure_atom_index=mapping,
    )
    assert isfinite(field.CalcEnergy()) and all(map(isfinite, field.CalcGrad()))


def test_actual_cyclic_amp_stereo_uses_deposited_and_generated_coordinates():
    result = add_retained_non_polymer_hydrogens(_source("CMP"))
    assert not result.issues
    record = build_bundled_retained_non_polymer_registry().get("CMP")
    assert record is not None
    ideal = {atom.atom_name: np.asarray(atom.ideal_position) for atom in record.atoms}
    for residue in result.structure.constitution.ligands:
        actual = {
            atom.name: result.structure.geometry.position(
                result.structure.constitution.atom_index(
                    AtomRef(residue.residue_id, atom.name)
                )
            ).to_array()
            for atom in residue.atom_sites
        }
        for atom in record.atoms:
            if atom.stereo_config is None:
                continue
            neighbors = [
                b.atom_name_2 if b.atom_name_1 == atom.atom_name else b.atom_name_1
                for b in record.bonds
                if atom.atom_name in (b.atom_name_1, b.atom_name_2)
            ]
            assert len(neighbors) == 4
            ideal_volume = np.linalg.det(
                np.asarray([ideal[n] - ideal[neighbors[-1]] for n in neighbors[:3]])
            )
            actual_volume = np.linalg.det(
                np.asarray([actual[n] - actual[neighbors[-1]] for n in neighbors[:3]])
            )
            assert ideal_volume * actual_volume > 0


@pytest.mark.parametrize("component_id", ("SAN", "CMP"))
@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_standalone_egress_preserves_realized_bonds_and_hydrogen_inventory(
    component_id, file_format
):
    result = add_retained_non_polymer_hydrogens(_source(component_id))
    assert not result.issues
    text = write_structure_string(result.structure, file_format)
    recovered = read_structure_string(text, file_format)
    assert recovered.constitution == result.structure.constitution
    for bond in result.structure.topology.bonds:
        refs = tuple(
            result.structure.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
        )
        assert (
            recovered.topology.bond_between(
                *(recovered.constitution.atom_index(ref) for ref in refs)
            )
            is not None
        )
    resolved = resolve_retained_non_polymer_topology(
        recovered, build_default_component_library()
    )
    assert not resolved.issues
    _assert_source_preserved(recovered, resolved.structure)
    facts = (
        StructureChemistryObservation(resolved.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert all(
        f.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
        and f.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
        for f in facts
    )
    repeated = add_retained_non_polymer_hydrogens(resolved.structure)
    assert not repeated.issues
    _assert_source_preserved(recovered, repeated.structure)


@pytest.mark.parametrize("component_id", ("SAN", "CMP"))
def test_actual_standalone_no_conect_pdb_is_readable(component_id):
    result = add_retained_non_polymer_hydrogens(_source(component_id))
    assert not result.issues
    text = write_structure_string(result.structure, FileFormat.PDB)
    block = (
        "\n".join(line for line in text.splitlines() if not line.startswith("CONECT"))
        + "\n"
    )
    molecule = Chem.MolFromPDBBlock(
        block, sanitize=True, removeHs=False, proximityBonding=True
    )
    assert molecule is not None
    assert molecule.GetNumAtoms() == sum(
        len(r.atom_sites) for r in result.structure.constitution.ligands
    )


@pytest.mark.parametrize(
    "component_id,atom_name,charge", (("SAN", "N1", "1+"), ("CMP", "O2P", "1-"))
)
@pytest.mark.parametrize("topology_only", (False, True))
def test_neutral_standalone_charge_conflict_cannot_publish_template_graph(
    component_id, atom_name, charge, topology_only
):
    text = _source_text(component_id)
    text = (
        "\n".join(
            line[:78].ljust(78) + charge
            if line.startswith("HETATM") and line[12:16].strip() == atom_name
            else line
            for line in text.splitlines()
        )
        + "\n"
    )
    source = read_structure_string(text, FileFormat.PDB)
    execute = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = execute(
        source,
        build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.issues and not result.repairs
    assert result.structure == source
    facts = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert all(
        f.heavy_atom_topology_availability_state
        is not TopologyAvailabilityState.PRESENT
        for f in facts
    )


@pytest.mark.parametrize("component_id", ("SAN", "CMP"))
@pytest.mark.parametrize("defect", ("missing", "extra", "wrong_element"))
def test_standalone_heavy_inventory_distinguishes_partial_from_conflicting_chemistry(
    component_id, defect
):
    lines = _source_text(component_id).splitlines()[:-1]
    first = lines[0]
    first_name = first[12:16]
    if defect == "missing":
        lines = [line for line in lines if line[12:16] != first_name]
    elif defect == "wrong_element":
        lines = [
            line[:76].ljust(76) + " F" + line[78:]
            if line[12:16] == first_name
            else line
            for line in lines
        ]
    else:
        lines.insert(1, first[:6] + "99999" + first[11:12] + " XX " + first[16:])
    source = read_structure_string("\n".join(lines) + "\nEND\n", FileFormat.PDB)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((source.constitution.ligands[0].residue_id,)),
    )
    template = build_default_component_library().require(component_id)
    assert not template.has_complete_heavy_scaffold(source.constitution.ligands[0])
    _assert_source_preserved(source, result.structure)
    if defect == "missing":
        assert template.has_compatible_heavy_scaffold(source.constitution.ligands[0])
        facts = (
            StructureChemistryObservation(result.structure)
            .whole_structure_facts()[1]
            .retained_non_polymer_facts
        )
        assert facts[0].heavy_atom_topology_availability_state is not (
            TopologyAvailabilityState.PRESENT
        )
    else:
        assert not template.has_compatible_heavy_scaffold(
            source.constitution.ligands[0]
        )
        assert result.issues and not result.repairs
        assert result.structure == source


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_standalone_original_hydrogen_alias_isotope_geometry_and_edges_survive(isotope):
    prepared = add_retained_non_polymer_hydrogens(_source("SAN"))
    assert not prepared.issues
    lines = write_structure_string(prepared.structure, FileFormat.PDB).splitlines()
    lines = [
        line[:12] + " HX " + line[16:76].ljust(60) + isotope.rjust(2) + line[78:]
        if line.startswith("HETATM") and line[12:16].strip() == "HN11"
        else line
        for line in lines
    ]
    source = read_structure_string("\n".join(lines) + "\n", FileFormat.PDB)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    _assert_source_preserved(source, result.structure)
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    repeated = add_retained_non_polymer_hydrogens(result.structure)
    assert not repeated.issues and not repeated.repairs


def test_standalone_conflict_is_local_in_mixed_batch():
    san = _source_text("SAN").splitlines()[:-1]
    san = [
        line[:78].ljust(78) + "1+" if line[12:16].strip() == "N1" else line
        for line in san
    ]
    source = read_structure_string(
        "\n".join(san + _source_text("CMP").splitlines()) + "\n", FileFormat.PDB
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.issues
    _assert_source_preserved(source, result.structure)
    for residue in result.structure.constitution.ligands:
        assert sum(a.is_hydrogen() for a in residue.atom_sites) == (
            0 if residue.component_id == "SAN" else 12
        )


@pytest.mark.parametrize("strict", (False, True))
def test_cyclic_amp_anion_override_precedes_neutral_ccd_default(strict):
    text = (
        "\n".join(
            line[:78].ljust(78) + "1-"
            if line.startswith("HETATM") and line[12:16].strip() == "O2P"
            else line
            for line in _source_text("CMP").splitlines()
        )
        + "\n"
    )
    source = read_structure_string(text, FileFormat.PDB)
    # Named stereo and traversal are mapped from the frozen CCD graph.
    smiles = "[P@]1(=O)([O-])OC[C@H]2O[C@@H](n3cnc4c(N)ncnc34)[C@H](O)[C@@H]2O1"
    names = (
        "P",
        "O1P",
        "O2P",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
        "C2'",
        "O2'",
        "C3'",
        "O3'",
    )
    evidence = tuple(
        RetainedNonPolymerChemistryOverride(
            residue.residue_id, smiles, names
        ).to_evidence()
        for residue in source.constitution.ligands
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=evidence,
        allow_retained_non_polymer_rdkit_fallback=not strict,
    )
    assert not result.issues
    _assert_source_preserved(source, result.structure)
    facts = (
        StructureChemistryObservation(
            result.structure, retained_non_polymer_chemistry_evidence=evidence
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert all(
        fact.heavy_topology_source
        is RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
        and fact.heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
        and fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
        for fact in facts
    )
    for residue in result.structure.constitution.ligands:
        assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 11
        anchor = result.structure.constitution.atom_index(
            AtomRef(residue.residue_id, "O2P")
        )
        assert result.structure.topology.formal_charge(anchor) == -1
        assert not any(
            anchor in bond.endpoint_pair()
            and any(
                result.structure.constitution.atom_site_at(index).is_hydrogen()
                for index in bond.endpoint_pair()
            )
            for bond in result.structure.topology.bonds
        )
    repeated = add_retained_non_polymer_hydrogens(
        result.structure,
        chemistry_evidence=evidence,
        allow_retained_non_polymer_rdkit_fallback=not strict,
    )
    assert not repeated.issues and not repeated.repairs
    assert repeated.structure == result.structure


@pytest.mark.parametrize("component_id", ("SAN", "CMP"))
def test_standalone_preparation_is_independent_of_source_atom_order(component_id):
    lines = _source_text(component_id).splitlines()[:-1]
    reversed_residues = {}
    for line in lines:
        reversed_residues.setdefault(line[21:27], []).append(line)
    text = (
        "\n".join(
            line for residue in reversed_residues.values() for line in reversed(residue)
        )
        + "\nEND\n"
    )
    source = read_structure_string(text, FileFormat.PDB)
    result = add_retained_non_polymer_hydrogens(source)
    reference = add_retained_non_polymer_hydrogens(_source(component_id))
    assert not result.issues
    _assert_source_preserved(source, result.structure)
    assert result.structure.constitution == reference.structure.constitution
    for index, position in result.structure.geometry.iter_positions():
        assert position == reference.structure.geometry.position(index)
    assert result.structure.topology == reference.structure.topology


@pytest.mark.parametrize("topology_only", (False, True))
@pytest.mark.parametrize("with_source_h", (False, True))
def test_standalone_source_connection_conflict_is_not_erased(
    topology_only, with_source_h
):
    if with_source_h:
        prepared = add_retained_non_polymer_hydrogens(_source("SAN"))
        text = write_structure_string(prepared.structure, FileFormat.PDB)
        endpoint_names = ("HN11", "N2")
    else:
        text = _source_text("SAN")
        endpoint_names = ("N1", "N2")
    serials = {
        line[12:16].strip(): int(line[6:11])
        for line in text.splitlines()
        if line.startswith("HETATM")
    }
    lines = [line for line in text.splitlines() if line != "END"]
    lines.append("CONECT" + "".join(f"{serials[name]:5d}" for name in endpoint_names))
    source = read_structure_string("\n".join(lines) + "\nEND\n", FileFormat.PDB)
    execute = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = execute(source, build_default_component_library())
    assert result.issues and not result.repairs
    assert result.structure == source
