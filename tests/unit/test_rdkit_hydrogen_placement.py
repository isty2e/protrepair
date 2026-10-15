"""Native-only generated H correction on independently deposited geometry."""

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

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.inference.rdkit_graph import (
    has_declared_stereochemistry,
    molecule_with_pose_stereochemistry,
    require_compatible_pose_stereochemistry,
)
from protrepair.chemistry.inference.rdkit_hydrogen_geometry import (
    add_hydrogens_preserving_stereochemistry,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    require_retained_non_polymer_evidence_stereochemistry,
    template_without_hydrogens,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure import ProteinStructure
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)

SMILES = "[P@]1(=O)([O-])OC[C@H]2O[C@@H](n3cnc4c(N)ncnc34)[C@H](O)[C@@H]2O1"
NAMES = (
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


def _pose(index):
    source = read_structure(Path("tests/fixtures/corpus/pdb1hw5.ent"))
    residues = tuple(r for r in source.constitution.ligands if r.component_id == "CMP")
    residue = residues[index]
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    molecule = template_without_hydrogens(SMILES)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for i, name in enumerate(NAMES):
        conformer.SetAtomPosition(i, tuple(geometry.position(name)))
    molecule.AddConformer(conformer)
    return molecule


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("source_stereo", (False, True))
def test_generated_h_preserves_deposited_stereo_positions_and_native_lengths(
    index, source_stereo
):
    pose = _pose(index)
    if source_stereo:
        pose = molecule_with_pose_stereochemistry(pose)
    before = tuple(
        tuple(pose.GetConformer().GetAtomPosition(i)) for i in range(pose.GetNumAtoms())
    )
    raw = Chem.AddHs(pose, addCoords=True)
    raw_observation = molecule_with_pose_stereochemistry(raw)
    mismatched = {
        a.GetIdx()
        for a, b in zip(raw.GetAtoms(), raw_observation.GetAtoms(), strict=True)
        if a.GetChiralTag()
        and b.GetChiralTag()
        and a.GetChiralTag() != b.GetChiralTag()
    }
    corrected = add_hydrogens_preserving_stereochemistry(pose)
    require_compatible_pose_stereochemistry(corrected)
    assert corrected.GetNumAtoms() == raw.GetNumAtoms()
    assert (
        tuple(
            tuple(pose.GetConformer().GetAtomPosition(i))
            for i in range(pose.GetNumAtoms())
        )
        == before
    )
    for i, point in enumerate(before):
        assert tuple(corrected.GetConformer().GetAtomPosition(i)) == point
    for atom in corrected.GetAtoms():
        if atom.GetIdx() < pose.GetNumAtoms():
            continue
        parent = atom.GetNeighbors()[0].GetIdx()
        current = np.array(corrected.GetConformer().GetAtomPosition(atom.GetIdx()))
        original = np.array(raw.GetConformer().GetAtomPosition(atom.GetIdx()))
        center = np.array(corrected.GetConformer().GetAtomPosition(parent))
        assert np.linalg.norm(current - center) == pytest.approx(
            np.linalg.norm(original - center)
        )
        if parent not in mismatched:
            assert np.array_equal(current, original)
    # This geometric check does not reuse the native stereo comparator.
    for index in mismatched:
        atom = corrected.GetAtomWithIdx(index)
        anchors = [
            a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() < pose.GetNumAtoms()
        ]
        hydrogen = next(
            a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() >= pose.GetNumAtoms()
        )
        conformer = corrected.GetConformer()
        center = np.array(conformer.GetAtomPosition(index))
        original_vectors = np.array(
            [np.array(conformer.GetAtomPosition(i)) - center for i in anchors]
        )
        h_point = np.array(conformer.GetAtomPosition(hydrogen))
        tetra_vectors = np.array(
            [np.array(conformer.GetAtomPosition(i)) - h_point for i in anchors]
        )
        assert np.linalg.det(original_vectors) * np.linalg.det(tetra_vectors) > 0


@pytest.mark.parametrize("smiles", ("F/C=C/F", "F/C=C\\F"))
@pytest.mark.parametrize("opposite", (False, True))
def test_double_bond_pose_is_checked_without_changing_source(smiles, opposite):
    molecule = Chem.MolFromSmiles(smiles)
    selected_y = -1.0 if smiles == "F/C=C/F" else 1.0
    if opposite:
        selected_y *= -1.0
    points = (
        (-1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.3, 0.0, 0.0),
        (2.3, selected_y, 0.0),
    )
    conformer = Chem.Conformer(4)
    for index, point in enumerate(points):
        conformer.SetAtomPosition(index, point)
    molecule.AddConformer(conformer)
    source_smiles = Chem.MolToSmiles(molecule)
    if opposite:
        with pytest.raises(ValueError, match="bond stereochemistry"):
            add_hydrogens_preserving_stereochemistry(molecule)
    else:
        result = add_hydrogens_preserving_stereochemistry(molecule)
        require_compatible_pose_stereochemistry(result)
        assert result.GetNumAtoms() == 6
        for index, point in enumerate(points):
            assert tuple(result.GetConformer().GetAtomPosition(index)) == point
    assert Chem.MolToSmiles(molecule) == source_smiles
    for index, point in enumerate(points):
        assert tuple(molecule.GetConformer().GetAtomPosition(index)) == point


def test_undefined_stereo_does_not_create_an_opposite_pose_constraint():
    molecule = _pose(1)
    Chem.RemoveStereochemistry(molecule)
    assert not has_declared_stereochemistry(molecule)
    require_compatible_pose_stereochemistry(molecule)
    result = add_hydrogens_preserving_stereochemistry(molecule)
    raw = Chem.AddHs(molecule, addCoords=True)
    assert np.array_equal(
        result.GetConformer().GetPositions(), raw.GetConformer().GetPositions()
    )


@pytest.mark.parametrize("order", ("reverse", "rotate", "identity"))
def test_tetrahedral_placement_respects_native_atom_permutations(order):
    molecule = _pose(1)
    indices = list(range(molecule.GetNumAtoms()))
    if order == "reverse":
        indices.reverse()
    elif order == "rotate":
        indices = indices[3:] + indices[:3]
    molecule = Chem.RenumberAtoms(molecule, indices)
    before = molecule.GetConformer().GetPositions().copy()
    result = add_hydrogens_preserving_stereochemistry(molecule)
    require_compatible_pose_stereochemistry(result)
    assert np.array_equal(result.GetConformer().GetPositions()[: len(indices)], before)
    assert np.array_equal(molecule.GetConformer().GetPositions(), before)


@pytest.mark.parametrize("side", (-1, 0, 1))
@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize("order", ("identity", "reverse", "rotate", "interleave"))
def test_unassigned_center_uses_declared_stereo_on_either_side_of_anchor_plane(
    side, invert, order
):
    pose = _pose(1)
    conformer = pose.GetConformer()
    center = np.array(conformer.GetAtomPosition(7))
    anchors = np.array(
        [
            conformer.GetAtomPosition(a.GetIdx())
            for a in pose.GetAtomWithIdx(7).GetNeighbors()
        ]
    )
    normal = np.cross(anchors[1] - anchors[0], anchors[2] - anchors[0])
    normal /= np.linalg.norm(normal)
    displacement = np.dot(center - anchors[0], normal)
    conformer.SetAtomPosition(7, tuple(center + (side - 1) * displacement * normal))
    if invert:
        pose.GetAtomWithIdx(7).InvertChirality()
    indices = list(range(pose.GetNumAtoms()))
    if order == "reverse":
        indices.reverse()
    elif order == "rotate":
        indices = indices[3:] + indices[:3]
    elif order == "interleave":
        indices = indices[::2] + indices[1::2]
    pose = Chem.RenumberAtoms(pose, indices)
    index = indices.index(7)
    expected = pose.GetAtomWithIdx(index).GetChiralTag()
    assert (
        molecule_with_pose_stereochemistry(pose).GetAtomWithIdx(index).GetChiralTag()
        == Chem.ChiralType.CHI_UNSPECIFIED
    )
    original_positions = pose.GetConformer().GetPositions().copy()
    raw = Chem.AddHs(pose, addCoords=True)
    result = add_hydrogens_preserving_stereochemistry(pose)
    assert (
        molecule_with_pose_stereochemistry(result).GetAtomWithIdx(index).GetChiralTag()
        == expected
    )
    assert np.array_equal(pose.GetConformer().GetPositions(), original_positions)
    assert np.array_equal(
        result.GetConformer().GetPositions()[: len(indices)], original_positions
    )
    atom = result.GetAtomWithIdx(index)
    hydrogen = next(
        a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() >= len(indices)
    )
    geometry = result.GetConformer()
    point = np.array(geometry.GetAtomPosition(hydrogen))
    fixed = np.array(
        [
            geometry.GetAtomPosition(a.GetIdx())
            for a in atom.GetNeighbors()
            if a.GetIdx() != hydrogen
        ]
    )
    volume = np.linalg.det(fixed - point)
    expected_sign = -1 if expected == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 1
    assert expected_sign * volume > 0
    parent = np.array(geometry.GetAtomPosition(index))
    native_point = np.array(raw.GetConformer().GetAtomPosition(hydrogen))
    assert np.linalg.norm(point - parent) == pytest.approx(
        np.linalg.norm(native_point - parent)
    )


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("name", ("C4'", "C1'", "C2'", "C3'"))
def test_source_h_geometry_damage_is_not_always_a_native_configuration_change(
    index, name
):
    pose = _pose(index)
    source = add_hydrogens_preserving_stereochemistry(pose)
    center_index = NAMES.index(name)
    atom = source.GetAtomWithIdx(center_index)
    hydrogen = next(a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() >= len(NAMES))
    conformer = source.GetConformer()
    center = np.array(conformer.GetAtomPosition(center_index))
    before = np.array(conformer.GetAtomPosition(hydrogen))
    after = 2 * center - before
    anchors = np.array(
        [
            conformer.GetAtomPosition(a.GetIdx())
            for a in atom.GetNeighbors()
            if a.GetIdx() != hydrogen
        ]
    )
    assert np.linalg.det(anchors - before) * np.linalg.det(anchors - after) < 0
    geometry = ResidueGeometry(
        {
            **{
                atom_name: AtomGeometry(
                    Vec3.from_iterable(conformer.GetAtomPosition(i))
                )
                for i, atom_name in enumerate(NAMES)
            },
            "OBSERVED": AtomGeometry(Vec3.from_iterable(after)),
        }
    )
    evidence = RetainedNonPolymerChemistryEvidence(ResidueId("L", 1), SMILES, NAMES)
    heavy_tag = (
        molecule_with_pose_stereochemistry(pose)
        .GetAtomWithIdx(center_index)
        .GetChiralTag()
    )
    if heavy_tag == Chem.ChiralType.CHI_UNSPECIFIED:
        with pytest.raises(ValueError, match="stereo"):
            require_retained_non_polymer_evidence_stereochemistry(
                evidence, geometry, hydrogen_attachments=(("OBSERVED", name),)
            )
    else:
        assert heavy_tag == atom.GetChiralTag()
        require_retained_non_polymer_evidence_stereochemistry(
            evidence, geometry, hydrogen_attachments=(("OBSERVED", name),)
        )
    assert geometry.position("OBSERVED") == Vec3.from_iterable(after)


@pytest.mark.parametrize("isotope", (0, 2, 3))
def test_repeated_hydrogenation_keeps_existing_hydrogen_isotopes_and_positions(isotope):
    source = add_hydrogens_preserving_stereochemistry(_pose(1))
    hydrogen = next(atom for atom in source.GetAtoms() if atom.GetAtomicNum() == 1)
    hydrogen.SetIsotope(isotope)
    positions = source.GetConformer().GetPositions().copy()
    result = add_hydrogens_preserving_stereochemistry(source)
    assert result.GetNumAtoms() == source.GetNumAtoms()
    assert result.GetAtomWithIdx(hydrogen.GetIdx()).GetIsotope() == isotope
    assert np.array_equal(result.GetConformer().GetPositions(), positions)


def test_confirmed_mirrored_heavy_pose_is_rejected_without_moving_it():
    molecule = _pose(1)
    conformer = molecule.GetConformer()
    for index in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (-point.x, point.y, point.z))
    before = conformer.GetPositions().copy()
    with pytest.raises(ValueError, match="atom stereochemistry"):
        add_hydrogens_preserving_stereochemistry(molecule)
    assert np.array_equal(molecule.GetConformer().GetPositions(), before)


def test_inverted_new_h_is_corrected_without_changing_source_atoms(monkeypatch):
    add_hydrogens = Chem.AddHs
    captured = {}

    def inverted_placement(molecule, *args, **kwargs):
        result = add_hydrogens(molecule, *args, **kwargs)
        observed = Chem.Mol(result)
        Chem.RemoveStereochemistry(observed)
        Chem.AssignStereochemistryFrom3D(observed, replaceExistingTags=True)
        atom = result.GetAtomWithIdx(7)
        hydrogen = next(
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
            and neighbor.GetIdx() >= molecule.GetNumAtoms()
        )
        if observed.GetAtomWithIdx(7).GetChiralTag() == atom.GetChiralTag():
            conformer = result.GetConformer()
            point = 2 * np.array(conformer.GetAtomPosition(7)) - np.array(
                conformer.GetAtomPosition(hydrogen.GetIdx())
            )
            conformer.SetAtomPosition(hydrogen.GetIdx(), tuple(point))
        captured["raw"] = Chem.Mol(result)
        captured["hydrogen"] = hydrogen.GetIdx()
        return result

    monkeypatch.setattr(Chem, "AddHs", inverted_placement)
    source = _pose(1)
    result = add_hydrogens_preserving_stereochemistry(source)
    raw = captured["raw"]
    hydrogen_index = captured["hydrogen"]
    point = np.array(result.GetConformer().GetAtomPosition(hydrogen_index))
    old_point = np.array(raw.GetConformer().GetAtomPosition(hydrogen_index))
    center = np.array(result.GetConformer().GetAtomPosition(7))
    assert not np.array_equal(point, old_point)
    assert np.linalg.norm(point - center) == pytest.approx(
        np.linalg.norm(old_point - center)
    )
    for index in range(source.GetNumAtoms()):
        assert tuple(result.GetConformer().GetAtomPosition(index)) == tuple(
            source.GetConformer().GetAtomPosition(index)
        )
    atom = result.GetAtomWithIdx(7)
    anchors = [a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() != hydrogen_index]
    vectors = np.array(
        [
            np.array(result.GetConformer().GetAtomPosition(index)) - center
            for index in anchors
        ]
    )
    translated = vectors + center - point
    assert np.linalg.det(vectors) * np.linalg.det(translated) > 0


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_retained_cleanup_cannot_move_source_h_using_an_overridden_template_name(
    isotope,
):
    smiles = "N[C@@H](CO)C(=O)O"
    names = ("N", "CA", "CB", "OG", "C", "O", "OXT")
    native = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert EmbedMolecule(native, randomSeed=41) == 0
    hydrogen = next(
        atom
        for atom in native.GetAtomWithIdx(1).GetNeighbors()
        if atom.GetAtomicNum() == 1
    )
    point = Vec3.from_iterable(native.GetConformer().GetAtomPosition(hydrogen.GetIdx()))
    residue_id = ResidueId("L", 1)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="SER",
                residue_id=residue_id,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        name,
                        native.GetAtomWithIdx(index).GetSymbol(),
                        Vec3.from_iterable(
                            native.GetConformer().GetAtomPosition(index)
                        ),
                    )
                    for index, name in enumerate(names)
                )
                + (atom_payload("HG", isotope, point, occupancy=0.4, b_factor=17.0),),
            ),
            residue_payload(
                component_id="BLK",
                residue_id=ResidueId("Z", 1),
                is_hetero=True,
                atoms=(atom_payload("C1", "C", point),),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    source.constitution.atom_index(AtomRef(residue_id, "CA")),
                    source.constitution.atom_index(AtomRef(residue_id, "HG")),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        ),
        provenance=source.provenance,
    )
    evidence = RetainedNonPolymerChemistryEvidence(residue_id, smiles, names)
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((residue_id,)),
    )
    before = source.constitution.atom_index(AtomRef(residue_id, "HG"))
    after = result.structure.constitution.atom_index(AtomRef(residue_id, "HG"))
    assert any(event.residue_id == residue_id for event in result.repairs)
    assert result.structure.geometry.atom_geometry(
        after
    ) == source.geometry.atom_geometry(before)
    outer = add_hydrogens(result.structure, prepare_heavy_atoms=False)
    assert outer.structure.geometry.atom_geometry(
        outer.structure.constitution.atom_index(AtomRef(residue_id, "HG"))
    ) == source.geometry.atom_geometry(before)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("operation", ("prepare", "facts", "topology"))
def test_opposite_observed_h_is_not_silently_replaced_by_a_corrected_new_h(
    isotope, operation
):
    pose = _pose(1)
    corrected = add_hydrogens_preserving_stereochemistry(pose)
    atom = corrected.GetAtomWithIdx(7)
    hydrogen = next(a for a in atom.GetNeighbors() if a.GetIdx() >= pose.GetNumAtoms())
    conformer = corrected.GetConformer()
    center = np.array(conformer.GetAtomPosition(7))
    point = 2 * center - np.array(conformer.GetAtomPosition(hydrogen.GetIdx()))
    damaged = Chem.Mol(corrected)
    damaged.GetConformer().SetAtomPosition(hydrogen.GetIdx(), tuple(point))
    assert (
        molecule_with_pose_stereochemistry(damaged).GetAtomWithIdx(7).GetChiralTag()
        != atom.GetChiralTag()
    )
    residue_id = ResidueId("L", 17)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="CMP",
                residue_id=residue_id,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        name,
                        pose.GetAtomWithIdx(i).GetSymbol().upper(),
                        Vec3.from_iterable(pose.GetConformer().GetAtomPosition(i)),
                        formal_charge=-1 if name == "O2P" else None,
                    )
                    for i, name in enumerate(NAMES)
                )
                + (
                    atom_payload(
                        "OBSERVED",
                        isotope,
                        Vec3.from_iterable(point),
                        occupancy=0.4,
                        b_factor=17.0,
                    ),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=(
            TopologyBond(
                source.constitution.atom_index(AtomRef(residue_id, "C1'")),
                source.constitution.atom_index(AtomRef(residue_id, "OBSERVED")),
                order=1,
                relationship_type=BondRelationshipType.COVALENT,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    evidence = RetainedNonPolymerChemistryEvidence(residue_id, SMILES, NAMES)
    # The heavy-only projection agrees; the copied H makes the actual pose opposite.
    require_compatible_pose_stereochemistry(pose)
    if operation == "facts":
        facts = resolve_retained_non_polymer_chemistry(
            source,
            source.constitution.ligands[0],
            component_library=build_default_component_library(),
            evidence=evidence,
            allow_rdkit_fallback=False,
        )
        assert facts.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        assert "stereo" in facts.failure_reason.lower()
    else:
        execute = (
            add_retained_non_polymer_hydrogens
            if operation == "prepare"
            else resolve_retained_non_polymer_topology
        )
        result = execute(
            source,
            chemistry_evidence=(evidence,),
            component_library=build_default_component_library(),
            allow_retained_non_polymer_rdkit_fallback=False,
        )
        assert result.structure is source
        assert not result.repairs
        assert any("stereo" in issue.message.lower() for issue in result.issues)


@pytest.mark.parametrize("reverse_order", (False, True))
def test_mixed_ligand_batch_preserves_sources_and_refuses_only_opposite_pose(
    reverse_order,
):
    pose = _pose(1)
    payloads = []
    evidence = []
    for seq_num in (1, 2):
        residue_id = ResidueId("L", seq_num)
        atoms = []
        for index, name in enumerate(NAMES):
            point = pose.GetConformer().GetAtomPosition(index)
            atoms.append(
                atom_payload(
                    name,
                    pose.GetAtomWithIdx(index).GetSymbol(),
                    Vec3(point.x, point.y, point.z)
                    if seq_num == 1
                    else Vec3(-point.x, point.y, point.z),
                    formal_charge=pose.GetAtomWithIdx(index).GetFormalCharge(),
                )
            )
        payloads.append(
            residue_payload(
                component_id="CMP",
                residue_id=residue_id,
                is_hetero=True,
                atoms=tuple(atoms),
            )
        )
        evidence.append(RetainedNonPolymerChemistryEvidence(residue_id, SMILES, NAMES))
    source = build_structure(
        chains=(),
        ligands=tuple(reversed(payloads)) if reverse_order else tuple(payloads),
        source_format=FileFormat.MMCIF,
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=tuple(evidence),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.repairs
    assert all(event.residue_id == ResidueId("L", 1) for event in result.repairs)
    assert any(
        issue.residue_id == ResidueId("L", 2) and "stereo" in issue.message.lower()
        for issue in result.issues
    )
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        assert result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(ref)
        ) == source.geometry.atom_geometry(index)
    bad = source.constitution.residue_index(ResidueId("L", 2))
    bad_after = result.structure.constitution.residue_index(ResidueId("L", 2))
    assert result.structure.residue_geometry(bad_after) == source.residue_geometry(bad)

    observation = StructureChemistryObservation(
        result.structure,
        retained_non_polymer_chemistry_evidence=tuple(evidence),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    ready = observation.projected_facts((ResidueId("L", 1),))[1]
    fact = ready.retained_non_polymer_facts[0]
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    parent_index = result.structure.constitution.atom_index(
        AtomRef(ResidueId("L", 1), "C1'")
    )
    hydrogen_name = next(
        name
        for name, parent in result.structure.topology.residue_hydrogen_anchor_by_name(
            constitution=result.structure.constitution,
            residue_index=result.structure.constitution.residue_index(
                ResidueId("L", 1)
            ),
        ).items()
        if parent == "C1'"
    )
    hydrogen_index = result.structure.constitution.atom_index(
        AtomRef(ResidueId("L", 1), hydrogen_name)
    )
    original_geometry = result.structure.geometry.atom_geometry(hydrogen_index)
    center = result.structure.geometry.atom_geometry(parent_index).position
    flipped = Vec3.from_iterable(
        2 * center.to_array() - original_geometry.position.to_array()
    )
    changed = ProteinStructure.from_payload(
        constitution=result.structure.constitution,
        geometry=result.structure.geometry.with_atom_geometry(
            hydrogen_index, original_geometry.with_position(flipped)
        ),
        topology=result.structure.topology,
        provenance=result.structure.provenance,
    )
    changed_observation = observation.for_structure(changed)
    changed_facts = changed_observation.projected_facts((ResidueId("L", 1),))[1]
    changed_fact = changed_facts.retained_non_polymer_facts[0]
    assert (
        changed_fact.heavy_atom_topology_availability_state
        is not TopologyAvailabilityState.PRESENT
    )
    assert (
        changed_fact.hydrogen_topology_availability_state
        is not TopologyAvailabilityState.PRESENT
    )
    topology_result = resolve_retained_non_polymer_topology(
        changed,
        build_default_component_library(),
        chemistry_evidence=tuple(evidence),
        target_residue_ids=frozenset((ResidueId("L", 1),)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert topology_result.structure is changed
    assert not topology_result.repairs
    assert any("stereo" in issue.message.lower() for issue in topology_result.issues)


@pytest.mark.parametrize("readiness", (False, True))
def test_missing_mapped_coordinate_is_a_mapping_failure_not_a_key_error(readiness):
    source = read_structure(Path("tests/fixtures/corpus/pdb1hw5.ent"))
    residue = next(r for r in source.constitution.ligands if r.component_id == "CMP")
    evidence = RetainedNonPolymerChemistryEvidence(
        residue.residue_id, SMILES, ("ABSENT", *NAMES[1:])
    )
    if readiness:
        facts = resolve_retained_non_polymer_chemistry(
            source,
            residue,
            component_library=build_default_component_library(),
            evidence=evidence,
            allow_rdkit_fallback=False,
        )
        assert facts.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        assert "mapping" in facts.failure_reason.lower()
    else:
        with pytest.raises(ValueError, match="mapping"):
            require_retained_non_polymer_evidence_stereochemistry(
                evidence,
                source.residue_geometry(
                    source.constitution.residue_index(residue.residue_id)
                ),
            )
