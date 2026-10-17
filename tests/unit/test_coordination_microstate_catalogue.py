"""Registered graphs, source custody and generic coordination preparation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.request_builders import ingress_options

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.inference.coordination_microstates import (
    CoordinationMicrostates,
)
from protrepair.chemistry.inference.rdkit_graph import (
    require_compatible_pose_stereochemistry,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    template_without_hydrogens,
)
from protrepair.chemistry.retained_non_polymer import coordination
from protrepair.chemistry.retained_non_polymer import (
    coordination_catalogue as catalogue,
)
from protrepair.chemistry.retained_non_polymer.coordination_catalogue import (
    coordination_microstate_family,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.geometry import Vec3
from protrepair.io import (
    read_structure,
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState, TopologyAvailabilityState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    WorkflowTransformRequests,
    requested_process_goal,
)

LIGAND = ResidueId("L", 73)
METAL = ResidueId("Z", 4)
CONTROL_STATES = (
    ("n1c(C)[nH]cc1", "[nH]1c(C)ncc1", "n1c(C)[n-]cc1", "[n-]1c(C)ncc1"),
    ("n1c[nH]c(C)c1C", "[nH]1cnc(C)c1C", "n1c[n-]c(C)c1C", "[n-]1cnc(C)c1C"),
    ("n1cn(C)cc1",),
)


def _family(states):
    molecule = template_without_hydrogens(states[0])
    return CoordinationMicrostates(
        tuple(f"Q{i}" for i in range(molecule.GetNumAtoms())),
        tuple(states),
        "Independent synthetic substituted aromatic-N source/mapping control.",
    )


def _source(
    family, donor, *, reverse=False, donor_h=None, charge=None, attached_h=None
):
    molecule = Chem.AddHs(template_without_hydrogens(family.smiles_states[0]))
    assert EmbedMolecule(molecule, randomSeed=17) == 0
    conformer = molecule.GetConformer()
    atoms = []
    for i, name in enumerate(family.heavy_atom_names):
        atom = molecule.GetAtomWithIdx(i)
        point = conformer.GetAtomPosition(i)
        atoms.append(
            atom_payload(
                name,
                atom.GetSymbol().upper(),
                Vec3(point.x, point.y, point.z),
                occupancy=0.7,
                b_factor=11,
            )
        )
    donor_index = family.heavy_atom_names.index(donor)
    donor_point = np.array(conformer.GetAtomPosition(donor_index))
    neighbors = molecule.GetAtomWithIdx(donor_index).GetNeighbors()
    carbon_points = np.array(
        [
            list(conformer.GetAtomPosition(a.GetIdx()))
            for a in neighbors
            if a.GetAtomicNum() == 6
        ]
    )
    direction = donor_point - carbon_points.mean(axis=0)
    direction /= np.linalg.norm(direction)
    if donor_h is not None:
        atoms.append(
            atom_payload("SRC", donor_h, Vec3.from_iterable(donor_point + direction))
        )
    if attached_h is not None:
        _, point = attached_h
        atoms.append(atom_payload("OBSERVED_H", "H", point))
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="XQM",
                residue_id=LIGAND,
                is_hetero=True,
                atoms=tuple(reversed(atoms)) if reverse else tuple(atoms),
            ),
            residue_payload(
                component_id="ZN",
                residue_id=METAL,
                is_hetero=True,
                atoms=(
                    atom_payload(
                        "M",
                        "ZN",
                        Vec3.from_iterable(donor_point + 2.3 * direction),
                        formal_charge=charge,
                    ),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    bonds = [
        TopologyBond(
            source.constitution.atom_index(AtomRef(LIGAND, donor)),
            source.constitution.atom_index(AtomRef(METAL, "M")),
            order=None,
            relationship_type=BondRelationshipType.METAL_COORDINATION,
            provenance=BondProvenance.SOURCE_EXPLICIT,
        )
    ]
    if donor_h is not None:
        bonds.append(
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, donor)),
                source.constitution.atom_index(AtomRef(LIGAND, "SRC")),
                order=1,
                relationship_type=BondRelationshipType.COVALENT,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            )
        )
    if attached_h is not None:
        parent, _ = attached_h
        bonds.append(
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, parent)),
                source.constitution.atom_index(AtomRef(LIGAND, "OBSERVED_H")),
                order=1,
                relationship_type=BondRelationshipType.COVALENT,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            )
        )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(bonds),
        ),
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    return source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(
                source.provenance.ingress,
                observation=StructureObservation.from_source_facets(
                    constitution=source.constitution,
                    geometry=source.geometry,
                    topology=source.topology,
                ),
            ),
        )
    )


def _register(monkeypatch, family):
    original = coordination_microstate_family
    monkeypatch.setattr(
        coordination,
        "coordination_microstate_family",
        lambda key: family if key == "XQM" else original(key),
    )


@pytest.mark.parametrize("states", CONTROL_STATES)
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_registered_substituted_graphs_prepare_without_runtime_name_rules(
    states, reverse, file_format, monkeypatch
):
    family = _family(states)
    donor = family.heavy_atom_names[0]
    _register(monkeypatch, family)
    source = _source(family, donor, reverse=reverse, charge=2)
    result = process_structure(
        source,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
        transform_requests=WorkflowTransformRequests(
            allow_retained_non_polymer_rdkit_fallback=False
        ),
    )
    assert not result.has_errors(), result.issues
    prepared = result.structure
    # Pure ligand controls query retained facts, not polymer goal applicability.
    _, facts = StructureChemistryObservation(
        prepared,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    ).whole_structure_facts()
    ligand_fact = next(
        f for f in facts.retained_non_polymer_facts if f.residue_id == LIGAND
    )
    assert ligand_fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert (
        ligand_fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    residue_index = prepared.constitution.residue_index(LIGAND)
    anchors = prepared.topology.residue_hydrogen_anchor_by_name(
        constitution=prepared.constitution, residue_index=residue_index
    )
    assert anchors and donor not in anchors.values()
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(METAL, "M"))
        )
        == 2
    )
    for index in range(len(source.constitution.atom_slots)):
        ref = source.constitution.atom_ref_at(AtomIndex(index))
        before = source.geometry.atom_geometry(source.constitution.atom_index(ref))
        after = prepared.geometry.atom_geometry(prepared.constitution.atom_index(ref))
        assert after == before
    native = to_rdkit_molecule(prepared)
    assert not any(
        atom.GetNumRadicalElectrons() or atom.GetNumImplicitHs()
        for atom in native.GetAtoms()
    )
    assert any(bond.GetBondType() == Chem.BondType.DATIVE for bond in native.GetBonds())
    reread = read_structure_string(
        write_structure_string(prepared, file_format), file_format
    )
    assert len(reread.constitution.atom_slots) == len(prepared.constitution.atom_slots)
    link = reread.topology.bond_between(
        reread.constitution.atom_index(AtomRef(LIGAND, donor)),
        reread.constitution.atom_index(AtomRef(METAL, "M")),
    )
    assert link is not None
    if file_format is FileFormat.MMCIF:
        assert link.relationship_type is BondRelationshipType.METAL_COORDINATION
    else:
        # PDB LINK preserves endpoints, but does not encode a coordination type.
        assert link.source_metadata is not None
        assert link.source_metadata.record_type is SourceBondRecordType.PDB_LINK
        assert link.order is None
    repeated = add_retained_non_polymer_hydrogens(
        prepared,
        target_residue_ids=frozenset((LIGAND,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert not repeated.has_errors(), repeated.issues
    assert repeated.structure.constitution == prepared.constitution
    assert repeated.structure.geometry == prepared.geometry
    assert repeated.structure.topology == prepared.topology


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_generic_donor_process_refuses_source_h_without_deleting_it(
    isotope, monkeypatch
):
    family = _family(CONTROL_STATES[0])
    _register(monkeypatch, family)
    source = _source(family, family.heavy_atom_names[0], donor_h=isotope)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((LIGAND,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.issues
    assert result.structure == source
    source_h = result.structure.constitution.resolve_atom_site(AtomRef(LIGAND, "SRC"))
    assert source_h is not None and source_h.element == isotope


@pytest.mark.parametrize(
    "case",
    (
        "mapping",
        "changed_graph",
        "disconnected",
        "no_donor",
        "blank_provenance",
        "bond_stereo",
        "atom_stereo",
        "protonated_internal_donor",
        "covalent_metal",
        "unsupported_metal",
        "invalid_smiles",
        "duplicate_names",
    ),
)
def test_family_rejects_invalid_chemical_interpretation(case):
    states = ("n1c[nH]cc1",)
    names = ("A", "B", "C", "D", "E")
    provenance = "Synthetic control."
    if case == "mapping":
        names = names[:-1]
    elif case == "changed_graph":
        states += ("n1cc[nH]c1",)
    elif case == "disconnected":
        states = ("n1c[nH]cc1.C",)
        names += ("F",)
    elif case == "no_donor":
        states = ("c1cc[nH]c1",)
    elif case == "blank_provenance":
        provenance = " "
    elif case == "bond_stereo":
        states = ("n1c[nH]cc1/C=C/C", "n1c[nH]cc1/C=C\\C")
        names += ("F", "G", "H")
    elif case == "atom_stereo":
        states = ("n1c[nH]cc1[C@H](F)Cl", "n1c[nH]cc1[C@@H](F)Cl")
        names += ("F", "G", "H")
    elif case == "protonated_internal_donor":
        states = ("[Fe+2]<-[nH]1ccnc1",)
        names += ("F",)
    elif case == "covalent_metal":
        states = ("[Fe+2]-n1c[nH]cc1",)
        names += ("F",)
    elif case == "unsupported_metal":
        states = ("[Na+]<-n1c[nH]cc1",)
        names += ("F",)
    elif case == "invalid_smiles":
        states = ("not-a-molecule",)
    elif case == "duplicate_names":
        names = ("A", "B", "C", "D", " a ")
    with pytest.raises(ValueError):
        CoordinationMicrostates(names, states, provenance)


def test_unknown_component_lookup_does_not_require_native_chemistry(monkeypatch):
    monkeypatch.setattr(
        "protrepair.chemistry.inference.retained_non_polymer_evidence.Chem", None
    )
    assert coordination_microstate_family("UNREGISTERED-CONTROL") is None


@pytest.mark.parametrize(
    "case", ("valid", "boolean_schema", "unknown_schema", "duplicate", "blank_key")
)
def test_packaged_data_boundary_normalizes_keys_and_rejects_ambiguity(
    case, tmp_path, monkeypatch
):
    family = _family(CONTROL_STATES[0])
    record = {
        "component_id": " xqm ",
        "heavy_atom_names": family.heavy_atom_names,
        "smiles_states": family.smiles_states,
        "provenance": family.provenance,
    }
    payload = {"schema_version": 1, "components": [record]}
    if case == "boolean_schema":
        payload["schema_version"] = True
    elif case == "unknown_schema":
        payload["schema_version"] = 2
    elif case == "duplicate":
        payload["components"].append({**record, "component_id": "XQM"})
    elif case == "blank_key":
        record["component_id"] = " "
    (tmp_path / "coordination_microstates.json").write_text(json.dumps(payload))
    monkeypatch.setattr(catalogue, "files", lambda _: tmp_path)
    catalogue._catalogue_records.cache_clear()
    catalogue._load_family.cache_clear()
    try:
        if case == "valid":
            bound = coordination_microstate_family(" XqM ")
            assert bound is not None
            assert bound == family
            assert bound is coordination_microstate_family("XQM")
            assert bound.hypotheses(LIGAND, donor_atom_name="Q0")
        else:
            with pytest.raises(ValueError):
                coordination_microstate_family("XQM")
    finally:
        catalogue._catalogue_records.cache_clear()
        catalogue._load_family.cache_clear()


def test_active_template_does_not_invoke_coordination_catalogue(monkeypatch):
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="HIS",
                residue_id=LIGAND,
                is_hetero=True,
                atoms=(atom_payload("CA", "C", Vec3(0, 0, 0)),),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )

    def unexpected_catalogue(_):
        pytest.fail("an active component template must precede catalogue lookup")

    monkeypatch.setattr(
        coordination, "coordination_microstate_family", unexpected_catalogue
    )
    assert (
        coordination.coordinated_component_preparation(
            source, source.constitution.ligands[0], build_default_component_library()
        )
        is None
    )


def test_internal_metal_charge_role_does_not_require_atom_name_fe(monkeypatch):
    original = coordination_microstate_family("HME")
    assert original is not None
    names = tuple(
        "CENTER" if name == "FE" else name for name in original.heavy_atom_names
    )
    family = replace(original, heavy_atom_names=names)
    assert family.metal_atom_names == frozenset(("CENTER",))
    _register(monkeypatch, family)
    source = read_structure(Path("tests/fixtures/corpus/pdb2d6c.ent"))
    residue = next(r for r in source.constitution.ligands if r.component_id == "HME")
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    atoms = tuple(
        atom_payload(
            "CENTER" if atom.name == "FE" else atom.name,
            atom.element,
            geometry.position(atom.name),
        )
        for atom in residue.atom_sites
        if not atom.is_hydrogen()
    )
    isolated = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="XQM", residue_id=LIGAND, is_hetero=True, atoms=atoms
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    bound = coordination.coordinated_component_preparation(
        isolated, isolated.constitution.ligands[0], build_default_component_library()
    )
    assert bound is not None
    assert bound.unresolved_charge_atom_names() == ("CENTER",)
    assert dict(bound.formal_charge_requirements()).get("CENTER") is None
    assert len(bound.hypotheses) == 2


@pytest.mark.parametrize(
    "matched,opposite",
    (
        ("n1c[nH]cc1[C@H](F)Cl", "n1c[nH]cc1[C@@H](F)Cl"),
        ("n1c[nH]cc1/C=C/C", "n1c[nH]cc1/C=C\\C"),
        ("n1c[nH]cc1/C(Cl)=C(F)/Br", "n1c[nH]cc1/C(Cl)=C(F)\\Br"),
    ),
)
@pytest.mark.parametrize("contradictory", (False, True))
@pytest.mark.parametrize("explicit", (False, True))
def test_declared_stereo_is_checked_against_actual_source_geometry(
    matched, opposite, contradictory, explicit, monkeypatch
):
    expected = _family((matched,))
    _register(monkeypatch, expected)
    source = _source(_family((opposite if contradictory else matched,)), "Q0", charge=2)
    evidence = (
        RetainedNonPolymerChemistryEvidence(LIGAND, matched, expected.heavy_atom_names)
        if explicit
        else None
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=() if evidence is None else (evidence,),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    facts = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=build_default_component_library(),
        evidence=evidence,
        allow_rdkit_fallback=False,
    )
    if contradictory:
        assert result.structure == source
        assert any("stereo" in issue.message.lower() for issue in result.issues)
        assert facts.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        assert "stereo" in facts.failure_reason.lower()
    else:
        assert not result.has_errors(), result.issues
        assert result.structure.geometry.atom_count() > source.geometry.atom_count()
        assert facts.source is (
            RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
            if explicit
            else RetainedNonPolymerChemistryEvidenceSource.COORDINATION_PREPARATION
        )
        to_rdkit_molecule(result.structure)
    for index in range(source.geometry.atom_count()):
        ref = source.constitution.atom_ref_at(AtomIndex(index))
        current = result.structure.constitution.atom_index(ref)
        assert result.structure.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(AtomIndex(index))


@pytest.mark.parametrize("contradictory", (False, True))
def test_stereo_check_does_not_replace_observed_h_geometry(contradictory, monkeypatch):
    family = _family(("n1c[nH]cc1[C@H](F)Cl",))
    _register(monkeypatch, family)
    native = Chem.AddHs(template_without_hydrogens(family.smiles_states[0]))
    assert EmbedMolecule(native, randomSeed=17) == 0
    conformer = native.GetConformer()
    center = native.GetAtomWithIdx(5)
    hydrogen = next(a for a in center.GetNeighbors() if a.GetAtomicNum() == 1)
    point = np.array(conformer.GetAtomPosition(hydrogen.GetIdx()))
    if contradictory:
        anchors = [
            np.array(conformer.GetAtomPosition(a.GetIdx()))
            for a in center.GetNeighbors()
            if a.GetAtomicNum() != 1
        ]
        normal = np.cross(anchors[1] - anchors[0], anchors[2] - anchors[0])
        normal /= np.linalg.norm(normal)
        origin = np.array(conformer.GetAtomPosition(center.GetIdx()))
        vector = point - origin
        point = origin + vector - 2 * np.dot(vector, normal) * normal
    source = _source(
        family, "Q0", charge=2, attached_h=("Q5", Vec3.from_iterable(point))
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((LIGAND,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    # Varying the fourth H does not reverse the assigned heavy-triad handedness.
    # This guard diagnoses stereo contradictions, not poor H bond angles.
    assert not result.has_errors(), result.issues
    assert result.structure.geometry.atom_count() > source.geometry.atom_count()
    assert Chem.FindMolChiralCenters(to_rdkit_molecule(result.structure)) == [(5, "R")]
    ref = AtomRef(LIGAND, "OBSERVED_H")
    assert result.structure.geometry.atom_geometry(
        result.structure.constitution.atom_index(ref)
    ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))


def test_unspecified_stereo_does_not_override_source_handedness(monkeypatch):
    family = _family(("n1c[nH]cc1C(F)Cl",))
    _register(monkeypatch, family)
    source = _source(_family(("n1c[nH]cc1[C@H](F)Cl",)), "Q0", charge=2)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((LIGAND,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert not result.has_errors(), result.issues
    assert result.structure.geometry.atom_count() > source.geometry.atom_count()
    assert Chem.FindMolChiralCenters(to_rdkit_molecule(result.structure)) == [(5, "R")]


def test_geometrically_unassigned_stereo_is_not_a_confirmed_opposite():
    template = template_without_hydrogens("n1c[nH]cc1[C@H](F)Cl")
    full = Chem.AddHs(Chem.Mol(template))
    assert EmbedMolecule(full, randomSeed=17) == 0
    original = full.GetConformer()
    conformer = Chem.Conformer(template.GetNumAtoms())
    for i in range(template.GetNumAtoms()):
        point = original.GetAtomPosition(i)
        conformer.SetAtomPosition(i, (point.x, point.y, 0.0))
    template.AddConformer(conformer)
    observed = Chem.Mol(template)
    Chem.RemoveStereochemistry(observed)
    Chem.AssignStereochemistryFrom3D(observed)
    assert observed.GetAtomWithIdx(5).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    before = Chem.MolToSmiles(template)
    require_compatible_pose_stereochemistry(template)
    assert Chem.MolToSmiles(template) == before
