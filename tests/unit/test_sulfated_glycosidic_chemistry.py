"""Sulfated pyranose application follows deposited donor and acceptor roles."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import (
    IdealGeometryHydrogenSemantics,
    build_default_component_library,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.chemistry.retained_non_polymer.glycosidic import (
    GlycosidicTemplateBinding,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
    retained_non_polymer_record_to_template,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.nonstandard_component_ingress import (
    read_nonstandard_component_registry,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import BondRelationshipType, StructureTopology
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _source_text():
    return (
        "\n".join(
            line
            for line in (FIXTURES / "corpus/pdb4oee.ent").read_text().splitlines()
            if line.startswith(("HETATM", "LINK")) and line[17:20] in {"GNX", "IDY"}
        )
        + "\nEND\n"
    )


def _source():
    return read_structure_string(_source_text(), FileFormat.PDB)


def _assert_source_preserved(source, structure):
    for old, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old)
        new = structure.constitution.atom_index(ref)
        assert structure.constitution.atom_site_at(
            new
        ) == source.constitution.atom_site_at(old)
        assert structure.geometry.atom_geometry(new) == source.geometry.atom_geometry(
            old
        )
        if source.topology.formal_charge(old) is not None:
            assert structure.topology.formal_charge(
                new
            ) == source.topology.formal_charge(old)
    for bond in source.topology.bonds:
        refs = tuple(
            source.constitution.atom_ref_at(index) for index in bond.endpoint_pair()
        )
        current = structure.topology.bond_between(
            *(structure.constitution.atom_index(ref) for ref in refs)
        )
        assert current is not None
        assert current.provenance == bond.provenance
        assert current.source_metadata == bond.source_metadata


@pytest.mark.parametrize("name", ("GNX", "IDY"))
def test_sulfated_bundled_chemistry_matches_frozen_ccd(name):
    frozen = read_nonstandard_component_registry(
        FIXTURES / f"chemistry/retained/{name}.cif"
    ).get(name)
    bundled = build_bundled_retained_non_polymer_registry().get(name)
    assert bundled == frozen and bundled is not None


@pytest.mark.parametrize("strict", (False, True))
def test_sulfated_link_realizes_coupled_hydrogen_inventory_and_ready_graph(strict):
    source = _source()
    result = add_retained_non_polymer_hydrogens(
        source, allow_retained_non_polymer_rdkit_fallback=not strict
    )
    assert not result.issues
    assert tuple(
        sum(atom.is_hydrogen() for atom in residue.atom_sites)
        for residue in result.structure.constitution.ligands
    ) == (11, 12)
    _assert_source_preserved(source, result.structure)
    for residue in result.structure.constitution.ligands:
        anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
            constitution=result.structure.constitution,
            residue_index=result.structure.constitution.residue_index(
                residue.residue_id
            ),
        )
        if residue.component_id == "IDY":
            assert "O4" not in anchors.values()
            assert sum(parent == "C7" for parent in anchors.values()) == 3
        else:
            assert not residue.has_atom_site("O1") and not residue.has_atom_site("HO1")
    link = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(AtomRef(ResidueId("B", 1), "O4")),
        result.structure.constitution.atom_index(AtomRef(ResidueId("B", 2), "C1")),
    )
    assert (
        link is not None
        and link.order == 1
        and link.relationship_type is BondRelationshipType.COVALENT
    )
    facts = (
        StructureChemistryObservation(
            result.structure, component_library=build_default_component_library()
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert len(facts) == 2
    assert all(
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
        and fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
        for fact in facts
    )
    repeated = add_retained_non_polymer_hydrogens(result.structure)
    assert (
        repeated.structure is result.structure
        and not repeated.repairs
        and not repeated.issues
    )


def test_sulfated_ring_stereo_matches_ccd_using_all_actual_neighbors():
    result = add_retained_non_polymer_hydrogens(_source())
    assert not result.issues
    checked = 0
    for residue in result.structure.constitution.ligands:
        record = read_nonstandard_component_registry(
            FIXTURES / f"chemistry/retained/{residue.component_id}.cif"
        ).get(residue.component_id)
        assert record is not None
        component = record.to_idealized_component()
        for atom in component.heavy_atoms():
            if atom.stereo_config is None:
                continue
            neighbors = sorted(component.bonded_atom_names(atom.atom_name))
            assert len(neighbors) == 4
            ideal, actual = [], []
            for name in neighbors:
                neighbor = component.atom_with_ideal_position(name)
                assert neighbor is not None
                ideal.append(np.asarray(neighbor.ideal_position))
                ref = AtomRef(residue.residue_id, name)
                if (
                    residue.component_id == "GNX"
                    and atom.atom_name == "C1"
                    and name == "O1"
                ):
                    ref = AtomRef(ResidueId("B", 1), "O4")
                actual.append(
                    result.structure.geometry.position(
                        result.structure.constitution.atom_index(ref)
                    ).to_array()
                )
            volumes = [
                float(
                    np.linalg.det(
                        np.asarray([point - points[3] for point in points[:3]])
                    )
                )
                for points in (ideal, actual)
            ]
            assert volumes[0] * volumes[1] > 0
            checked += 1
    assert checked == 10


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_sulfated_graph_and_hydrogens_survive_output_and_topology_resolution(
    file_format,
):
    source = _source()
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    recovered = read_structure_string(
        write_structure_string(result.structure, file_format), file_format
    )
    resolved = resolve_retained_non_polymer_topology(
        recovered, build_default_component_library()
    )
    assert not resolved.issues
    assert len(resolved.structure.constitution.atom_slots) == len(
        result.structure.constitution.atom_slots
    )
    facts = (
        StructureChemistryObservation(
            resolved.structure, component_library=build_default_component_library()
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert all(
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
        and fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
        for fact in facts
    )
    for old, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old)
        assert resolved.structure.geometry.position(
            resolved.structure.constitution.atom_index(ref)
        ) == source.geometry.position(old)


@pytest.mark.parametrize("name,expected_h", (("GNX", 13), ("IDY", 12)))
def test_isolated_sulfated_records_keep_their_noncontextual_chemistry(name, expected_h):
    record = read_nonstandard_component_registry(
        FIXTURES / f"chemistry/retained/{name}.cif"
    ).get(name)
    assert record is not None
    component = record.to_idealized_component()
    atoms = []
    for atom in component.heavy_atoms():
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(
                atom.atom_name, atom.element, Vec3.from_iterable(atom.ideal_position)
            )
        )
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id=name,
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=tuple(atoms),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    result = add_retained_non_polymer_hydrogens(
        source, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert not result.issues
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[0].atom_sites
        )
        == expected_h
    )
    _assert_source_preserved(source, result.structure)


@pytest.mark.parametrize("topology_only", (False, True))
@pytest.mark.parametrize(
    "mutation",
    (
        "missing_acceptor",
        "missing_donor",
        "acceptor_charge",
        "sulfate_charge",
        "unknown_link",
        "double_link",
        "methyl_donor",
    ),
)
def test_sulfated_contradiction_refuses_both_connected_partners(
    mutation, topology_only
):
    text = _source_text()
    if mutation in {"acceptor_charge", "sulfate_charge"}:
        name = "O4" if mutation == "acceptor_charge" else "OS3"
        text = (
            "\n".join(
                line[:78] + "1-"
                if line.startswith("HETATM")
                and line[17:20] == "IDY"
                and line[12:16].strip() == name
                else line
                for line in text.splitlines()
            )
            + "\n"
        )
    source = read_structure_string(text, FileFormat.PDB)
    if mutation.startswith("missing"):
        ref = (
            AtomRef(ResidueId("B", 1), "C7")
            if mutation == "missing_acceptor"
            else AtomRef(ResidueId("B", 2), "C3")
        )
        source = source.without_atom_refs((ref,))
    if mutation in {"unknown_link", "double_link", "methyl_donor"}:
        original = source.topology.bond_between(
            source.constitution.atom_index(AtomRef(ResidueId("B", 1), "O4")),
            source.constitution.atom_index(AtomRef(ResidueId("B", 2), "C1")),
        )
        assert original is not None
        bond = original
        if mutation == "unknown_link":
            bond = replace(
                bond,
                relationship_type=BondRelationshipType.UNKNOWN,
                source_metadata=None,
            )
        elif mutation == "double_link":
            bond = replace(bond, order=2, source_metadata=None)
        else:
            bond = replace(
                bond,
                atom_index_1=source.constitution.atom_index(
                    AtomRef(ResidueId("B", 1), "C1")
                ),
                atom_index_2=source.constitution.atom_index(
                    AtomRef(ResidueId("B", 2), "O4")
                ),
                source_metadata=None,
            )
        source = ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=tuple(
                    bond if old == original else old for old in source.topology.bonds
                ),
            ),
            polymer_blueprint=source.polymer_blueprint,
            provenance=source.provenance,
        )
    transformer = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transformer(source, build_default_component_library())
    assert result.structure is source and result.issues and not result.repairs
    _assert_source_preserved(source, result.structure)


def test_methylated_acceptor_is_admitted_by_binding_with_frozen_custom_library():
    source = _source()
    library = build_default_component_library()
    for name in ("GNX", "IDY"):
        record = read_nonstandard_component_registry(
            FIXTURES / f"chemistry/retained/{name}.cif"
        ).get(name)
        assert record is not None
        library = library.with_template(
            replace(
                retained_non_polymer_record_to_template(record),
                hydrogen_semantics=IdealGeometryHydrogenSemantics(
                    record.to_idealized_component()
                ),
            )
        )
    residue = source.constitution.ligands[0]
    binding = GlycosidicTemplateBinding(
        source, residue, library.require("IDY"), library
    )
    assert binding.external_bonds == (
        (AtomRef(ResidueId("B", 1), "O4"), AtomRef(ResidueId("B", 2), "C1")),
    )
    assert "HO4" not in binding.template.expected_hydrogen_atom_names()
    assert {"C7", "O1"} <= set(binding.template.expected_heavy_atom_names())


def test_invalid_sulfated_override_never_falls_back_to_new_template():
    source = _source()
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((ResidueId("B", 2),)),
        chemistry_evidence=(
            RetainedNonPolymerChemistryEvidence(
                residue_id=ResidueId("B", 2),
                smiles="not-a-smiles",
                heavy_atom_names=source.constitution.ligands[1].atom_site_names(),
            ),
        ),
    )
    assert result.structure is source and result.issues and not result.repairs


@pytest.mark.parametrize("topology_only", (False, True))
def test_valid_isolated_idy_override_cannot_protonate_linked_o4(topology_only):
    prepared = add_retained_non_polymer_hydrogens(_source())
    assert not prepared.issues
    source = prepared.structure
    evidence = RetainedNonPolymerChemistryEvidence(
        residue_id=ResidueId("B", 1),
        smiles="COC1OC(C(=O)O)C(O)C(O)C1OS(=O)(=O)O",
        heavy_atom_names=(
            "C7",
            "O1",
            "C1",
            "O5",
            "C5",
            "C6",
            "O6A",
            "O6B",
            "C4",
            "O4",
            "C3",
            "O3",
            "C2",
            "O2",
            "S",
            "OS1",
            "OS2",
            "OS3",
        ),
    )
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=build_default_component_library(),
        evidence=evidence,
    )
    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert "external" in resolution.failure_reason
    transformer = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transformer(
        source,
        build_default_component_library(),
        target_residue_ids=frozenset((evidence.residue_id,)),
        chemistry_evidence=(evidence,),
    )
    assert result.structure is source and result.issues and not result.repairs
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(evidence.residue_id),
    )
    assert "O4" not in anchors.values()


@pytest.mark.parametrize(
    "component,center",
    [(name, f"C{i}") for name in ("IDY", "GNX") for i in range(1, 6)],
)
def test_wrong_actual_sulfated_tetrahedron_refuses_the_connected_pair(
    component, center
):
    source = _source()
    residue = next(
        r for r in source.constitution.ligands if r.component_id == component
    )
    record = read_nonstandard_component_registry(
        FIXTURES / f"chemistry/retained/{component}.cif"
    ).get(component)
    assert record is not None
    ideal = record.to_idealized_component()
    refs = [
        AtomRef(ResidueId("B", 1), "O4")
        if component == "GNX" and center == "C1" and name == "O1"
        else AtomRef(residue.residue_id, name)
        for name in sorted(ideal.bonded_atom_names(center))
        if (atom := ideal.atom(name)) is not None and not atom.is_hydrogen()
    ]
    a, b, c = (
        source.geometry.position(source.constitution.atom_index(ref)).to_array()
        for ref in refs
    )
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    old = geometry.position(center).to_array()
    flipped = old - 2 * np.dot(old - a, normal) * normal
    source = source.with_updated_residue_facets(
        residue,
        residue_geometry=geometry.with_atom_geometry(
            center,
            geometry.atom_geometry(center).with_position(Vec3.from_iterable(flipped)),
        ),
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source and result.issues and not result.repairs
    assert any("stereo" in issue.message for issue in result.issues)


def _deposit_idy_hydrogen(*, name="H7", isotope="H", occupied=False):
    source = _source()
    prepared = add_retained_non_polymer_hydrogens(source)
    assert not prepared.issues
    selected = AtomRef(ResidueId("B", 1), name)
    structure = prepared.structure.without_atom_refs(
        tuple(
            prepared.structure.constitution.atom_ref_at(i)
            for i, _ in prepared.structure.geometry.iter_positions()
            if prepared.structure.constitution.atom_site_at(i).is_hydrogen()
            and prepared.structure.constitution.atom_ref_at(i) != selected
        )
    )
    residue = structure.constitution.residue_or_ligand(selected.residue_id)
    assert residue is not None
    geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue.residue_id)
    )
    residue = residue.with_atom_sites(
        replace(atom, element=isotope) if atom.name == name else atom
        for atom in residue.atom_sites
    )
    if occupied:
        residue = residue.with_atom_site(AtomSite("HO4", isotope))
        geometry = geometry.with_atom_geometry(
            "HO4",
            AtomGeometry(
                Vec3.from_iterable(geometry.position("O4").to_array() + (0, 0, 1))
            ),
        )
    structure = structure.with_updated_residue_facets(
        residue, residue_geometry=geometry
    )
    return read_structure_string(
        write_structure_string(structure, FileFormat.MMCIF), FileFormat.MMCIF
    )


@pytest.mark.parametrize("name", ("H7", "HO6B", "HOS3"))
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_original_methyl_and_acid_h_isotopes_are_preserved(name, isotope):
    source = _deposit_idy_hydrogen(name=name, isotope=isotope)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    _assert_source_preserved(source, result.structure)
    assert tuple(
        sum(atom.is_hydrogen() for atom in residue.atom_sites)
        for residue in result.structure.constitution.ligands
    ) == (11, 12)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("topology_only", (False, True))
def test_original_h_on_linked_hydroxyl_is_not_removed_to_admit_sulfated_chemistry(
    isotope, topology_only
):
    source = _deposit_idy_hydrogen(isotope=isotope, occupied=True)
    transformer = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transformer(source, build_default_component_library())
    assert result.structure is source and result.issues and not result.repairs
    _assert_source_preserved(source, result.structure)


@pytest.mark.parametrize("mutation", ("extra", "element"))
def test_sulfated_incompatible_acceptor_inventory_refuses_both_partners(mutation):
    source = _source()
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    if mutation == "extra":
        residue = residue.with_atom_site(AtomSite("X1", "C"))
        geometry = geometry.with_atom_geometry("X1", AtomGeometry(Vec3(-10, -6, -8)))
    else:
        residue = residue.with_atom_sites(
            replace(atom, element="N") if atom.name == "O3" else atom
            for atom in residue.atom_sites
        )
    source = source.with_updated_residue_facets(residue, residue_geometry=geometry)
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source and result.issues and not result.repairs


def test_sulfated_preparation_is_independent_of_deposited_atom_order():
    text = _source_text().splitlines()
    reversed_text = []
    for component in ("GNX", "IDY"):
        reversed_text.extend(
            reversed(
                [
                    line
                    for line in text
                    if line.startswith("HETATM") and line[17:20] == component
                ]
            )
        )
    reversed_text.extend(line for line in text if line.startswith("LINK"))
    sources = (
        _source(),
        read_structure_string("\n".join(reversed_text) + "\nEND\n", FileFormat.PDB),
    )
    results = [add_retained_non_polymer_hydrogens(source) for source in sources]
    assert all(not result.issues for result in results)
    for source, result in zip(sources, results, strict=True):
        _assert_source_preserved(source, result.structure)
    for index, position in results[0].structure.geometry.iter_positions():
        ref = results[0].structure.constitution.atom_ref_at(index)
        other = results[1].structure.constitution.atom_index(ref)
        assert results[1].structure.geometry.position(other) == position


def test_sulfated_no_conect_pdb_is_native_readable():
    result = add_retained_non_polymer_hydrogens(_source())
    assert not result.issues
    text = (
        "\n".join(
            line
            for line in write_structure_string(
                result.structure, FileFormat.PDB
            ).splitlines()
            if not line.startswith("CONECT")
        )
        + "\n"
    )
    molecule = Chem.MolFromPDBBlock(
        text, sanitize=True, removeHs=False, proximityBonding=True
    )
    assert molecule is not None and molecule.GetNumAtoms() == len(
        result.structure.constitution.atom_slots
    )
