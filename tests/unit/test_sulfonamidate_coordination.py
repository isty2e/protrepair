"""Common donor admission and source-qualified sulfonamidate preparation."""

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

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.component.graph import (
    BondDefinition,
    ChemicalComponentDefinition,
)
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.inference.coordination_donor import NitrogenCoordinationDonor
from protrepair.chemistry.inference.coordination_microstates import (
    CoordinationMicrostates,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    template_without_hydrogens,
)
from protrepair.chemistry.retained_non_polymer import coordination
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.errors import (
    RdkitProjectionError,
    UnsupportedCoordinationChemistryError,
)
from protrepair.geometry import Vec3
from protrepair.geometry.placement_vector import deterministic_unit_orthogonal_or_none
from protrepair.io import (
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.state.retained_non_polymer_chemistry import (
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)

LIGAND = ResidueId("L", 73)
METAL = ResidueId("Z", 4)


def _example(
    smiles="CCS(=O)(=O)[NH-]",
    *,
    source_h=None,
    donor_charge: int | None = None,
    metal_charge: int | None = 2,
    metal="ZN",
    reverse=False,
    mirror=False,
    coincident=False,
    excess_h=False,
    second_partner=False,
):
    native = Chem.AddHs(template_without_hydrogens(smiles))
    assert EmbedMolecule(native, randomSeed=17) == 0
    if mirror:
        conformer = native.GetConformer()
        for i in range(native.GetNumAtoms()):
            x, y, z = conformer.GetAtomPosition(i)
            conformer.SetAtomPosition(i, (-x, y, z))
    count = template_without_hydrogens(smiles).GetNumAtoms()
    names = tuple(f"Q{i}" for i in range(count))
    donor_index = next(
        a.GetIdx()
        for a in native.GetAtoms()
        if a.GetAtomicNum() == 7 and a.GetFormalCharge() == -1
    )
    donor = names[donor_index]
    point = np.array(native.GetConformer().GetAtomPosition(donor_index))
    sulfur = next(
        a
        for a in native.GetAtomWithIdx(donor_index).GetNeighbors()
        if a.GetAtomicNum() == 16
    )
    axis = np.array(native.GetConformer().GetAtomPosition(sulfur.GetIdx())) - point
    axis /= np.linalg.norm(axis)
    orthogonal = deterministic_unit_orthogonal_or_none(axis)
    assert orthogonal is not None
    metal_position = Vec3.from_iterable(
        point + 2 * (-0.5 * axis + np.sqrt(0.75) * orthogonal)
    )
    if coincident:
        metal_position = Vec3.from_iterable(point)
    atoms = [
        atom_payload(
            name,
            native.GetAtomWithIdx(i).GetSymbol().upper(),
            Vec3.from_iterable(native.GetConformer().GetAtomPosition(i)),
            occupancy=0.7,
            b_factor=11,
            formal_charge=donor_charge if i == donor_index else None,
        )
        for i, name in enumerate(names)
    ]
    if source_h:
        atoms.append(
            atom_payload(
                "SRC",
                source_h,
                Vec3.from_iterable(point + [0, 0, 1]),
                occupancy=0.7,
                b_factor=11,
            )
        )
    if excess_h:
        atoms.append(atom_payload("SRC2", "H", Vec3.from_iterable(point + [0, 1, 0])))
    additional = (
        ()
        if not second_partner
        else (
            residue_payload(
                component_id=metal,
                residue_id=ResidueId("Z", 5),
                is_hetero=True,
                atoms=(
                    atom_payload(
                        "M",
                        metal,
                        Vec3.from_iterable(point + [0, 2, 0]),
                        formal_charge=metal_charge,
                    ),
                ),
            ),
        )
    )
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
                component_id=metal,
                residue_id=METAL,
                is_hetero=True,
                atoms=(
                    atom_payload(
                        "M", metal, metal_position, formal_charge=metal_charge
                    ),
                ),
            ),
            *additional,
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
            source_metadata=SourceBondMetadata(
                SourceBondRecordType.MMCIF_STRUCT_CONN,
                source_id="site",
                reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
            ),
        )
    ]
    if second_partner:
        bonds.append(
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, donor)),
                source.constitution.atom_index(AtomRef(ResidueId("Z", 5), "M")),
                order=None,
                relationship_type=BondRelationshipType.METAL_COORDINATION,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    SourceBondRecordType.MMCIF_STRUCT_CONN,
                    source_id="site2",
                    reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
                ),
            )
        )
    if source_h:
        bonds.append(
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, donor)),
                source.constitution.atom_index(AtomRef(LIGAND, "SRC")),
                order=1,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            )
        )
    if excess_h:
        bonds.append(
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, donor)),
                source.constitution.atom_index(AtomRef(LIGAND, "SRC2")),
                order=1,
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
    source = source.with_provenance(
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
    return (
        source,
        CoordinationMicrostates(
            names, (smiles,), "Independent sulfonamidate source control."
        ),
        donor,
        native,
    )


def _register(monkeypatch, family):
    original = coordination.coordination_microstate_family
    monkeypatch.setattr(
        coordination,
        "coordination_microstate_family",
        lambda name: family if name == "XQM" else original(name),
    )


def _preserved(source, result):
    for i, slot in enumerate(source.constitution.atom_slots):
        original = AtomIndex(i)
        ref = source.constitution.atom_ref_at(original)
        index = result.constitution.atom_index(ref)
        assert result.constitution.atom_site_at(index) == slot
        assert result.geometry.atom_geometry(index) == source.geometry.atom_geometry(
            original
        )
        assert (
            result.topology.formal_charge(index)
            == source.topology.formal_charge(original)
            or source.topology.formal_charge(original) is None
        )


@pytest.mark.parametrize(
    "smiles", ["CS(=O)(=O)[NH-]", "CCS(=O)(=O)[NH-]", "c1ccccc1S(=O)(=O)[NH-]"]
)
@pytest.mark.parametrize("route", ["catalogue", "override"])
@pytest.mark.parametrize("reverse", [False, True])
def test_sulfonamidate_preparation_is_common_and_source_preserving(
    smiles, route, reverse, monkeypatch
):
    source, family, donor, _ = _example(smiles, reverse=reverse)
    _register(monkeypatch, family)
    evidence = (
        ()
        if route == "catalogue"
        else (family.hypotheses(LIGAND, donor_atom_name=donor)[0],)
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=evidence,
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.repairs
    _preserved(source, result.structure)
    residue = result.structure.constitution.residue_or_ligand(LIGAND)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        result.structure,
        residue,
        component_library=build_default_component_library(),
        evidence=None if not evidence else evidence[0],
        allow_rdkit_fallback=False,
    )
    assert resolution.has_coordination_topology(result.structure)
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert sum(parent == donor for parent in anchors.values()) == 1
    native = to_rdkit_molecule(result.structure)
    assert not any(a.GetNumRadicalElectrons() for a in native.GetAtoms())
    assert sum(b.GetBondType() == Chem.BondType.DATIVE for b in native.GetBonds()) == 1
    for fmt in (FileFormat.PDB, FileFormat.MMCIF):
        restored = read_structure_string(
            write_structure_string(result.structure, fmt), fmt
        )
        assert len(restored.constitution.atom_slots) == len(
            result.structure.constitution.atom_slots
        )


@pytest.mark.parametrize("isotope", ["H", "D", "T"])
def test_one_original_donor_h_is_preserved(isotope, monkeypatch):
    source, family, donor, _ = _example(source_h=isotope, donor_charge=-1)
    _register(monkeypatch, family)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((LIGAND,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.repairs
    _preserved(source, result.structure)
    atom = result.structure.constitution.resolve_atom_site(AtomRef(LIGAND, "SRC"))
    assert atom is not None
    assert atom.element == isotope


@pytest.mark.parametrize("route", ["catalogue", "override"])
def test_known_neutral_donor_charge_is_not_overwritten(route, monkeypatch):
    source, family, donor, _ = _example(donor_charge=0)
    _register(monkeypatch, family)
    evidence = (
        ()
        if route == "catalogue"
        else (family.hypotheses(LIGAND, donor_atom_name=donor)[0],)
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=evidence,
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure == source
    assert not result.repairs


@pytest.mark.parametrize("route", ["catalogue", "override", "template"])
@pytest.mark.parametrize("lost", ["charge", "hydrogen", "coordination"])
def test_original_source_evidence_cannot_be_hidden(route, lost, monkeypatch):
    original, family, donor, native = _example(
        source_h="D" if lost == "hydrogen" else None,
        donor_charge=0 if lost == "charge" else None,
    )
    current, _, _, _ = _example()
    if lost == "coordination":
        current = ProteinStructure.from_payload(
            constitution=current.constitution,
            geometry=current.geometry,
            topology=StructureTopology(
                constitution=current.constitution,
                atom_topologies=current.topology.atom_topologies,
                bonds=(),
            ),
            provenance=current.provenance,
            polymer_blueprint=current.polymer_blueprint,
        )
    current = current.with_provenance(
        replace(
            current.provenance,
            ingress=replace(
                current.provenance.ingress,
                observation=original.provenance.ingress.observation,
            ),
        )
    )
    _register(monkeypatch, family)
    library = build_default_component_library()
    if route == "template":
        library = library.with_template(_template(native, family))
    evidence = family.hypotheses(LIGAND, donor_atom_name=donor)[0]
    result = add_retained_non_polymer_hydrogens(
        current,
        library,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=(evidence,) if route == "override" else (),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure == current
    assert not result.repairs


def test_wrong_metal_is_unsupported_not_a_source_contradiction(monkeypatch):
    source, family, donor, _ = _example(metal="FE")
    evidence = family.hypotheses(LIGAND, donor_atom_name=donor)[0]
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((LIGAND,)), chemistry_evidence=(evidence,)
    )
    assert result.structure == source
    assert result.issues[0].kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION


def _template(native, family):
    names = (
        *family.heavy_atom_names,
        *(f"Y{i}" for i in range(len(family.heavy_atom_names), native.GetNumAtoms())),
    )
    component = IdealizedComponent(
        "XQM",
        None,
        tuple(
            IdealizedComponentAtom(
                names[a.GetIdx()],
                a.GetSymbol().upper(),
                a.GetFormalCharge(),
                ideal_position=tuple(native.GetConformer().GetAtomPosition(a.GetIdx())),
            )
            for a in native.GetAtoms()
        ),
        tuple(
            BondDefinition(
                names[b.GetBeginAtomIdx()],
                names[b.GetEndAtomIdx()],
                int(b.GetBondTypeAsDouble()),
            )
            for b in native.GetBonds()
        ),
    )
    return ResidueTemplate(
        ChemicalComponentDefinition(
            "XQM",
            component.heavy_atom_names(),
            component.heavy_bonds(),
            {a.atom_name: a.formal_charge for a in component.heavy_atoms()},
        ),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(component),
    )


def test_active_bound_template_keeps_authority_and_uses_partner_geometry(monkeypatch):
    source, family, donor, native = _example()
    _register(monkeypatch, family)
    library = build_default_component_library().with_template(_template(native, family))
    result = add_retained_non_polymer_hydrogens(
        source,
        library,
        target_residue_ids=frozenset((LIGAND,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.repairs
    _preserved(source, result.structure)
    residue = result.structure.constitution.residue_or_ligand(LIGAND)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        result.structure,
        residue,
        component_library=library,
    )
    assert resolution.source.value == "template"
    assert resolution.has_coordination_topology(result.structure)
    _assert_donor_geometry(result.structure, donor)


def _assert_donor_geometry(structure, donor):
    anchors = structure.topology.residue_hydrogen_anchor_by_name(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(LIGAND),
    )
    h = next(name for name, parent in anchors.items() if parent == donor)
    point = structure.geometry.position(
        structure.constitution.atom_index(AtomRef(LIGAND, donor))
    )
    hydrogen = structure.geometry.position(
        structure.constitution.atom_index(AtomRef(LIGAND, h))
    )
    metal = structure.geometry.position(
        structure.constitution.atom_index(AtomRef(METAL, "M"))
    )
    nh = hydrogen.to_array() - point.to_array()
    nz = metal.to_array() - point.to_array()
    angle = np.degrees(
        np.arccos(
            np.clip(np.dot(nh, nz) / np.linalg.norm(nh) / np.linalg.norm(nz), -1, 1)
        )
    )
    assert 90 < angle < 150
    assert 0.8 < np.linalg.norm(nh) < 1.2
    assert hydrogen.distance_to(metal) > 2


def test_isolated_neutral_template_does_not_publish_bound_chemistry(monkeypatch):
    source, family, donor, native = _example()
    _register(monkeypatch, family)
    neutral = Chem.AddHs(template_without_hydrogens("CCS(=O)(=O)N"))
    assert EmbedMolecule(neutral, randomSeed=17) == 0
    library = build_default_component_library().with_template(
        _template(neutral, family)
    )
    result = add_retained_non_polymer_hydrogens(
        source, library, target_residue_ids=frozenset((LIGAND,))
    )
    assert result.structure == source
    assert result.issues[0].kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
    evidence = family.hypotheses(LIGAND, donor_atom_name=donor)[0]
    overridden = add_retained_non_polymer_hydrogens(
        source,
        library,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=(evidence,),
    )
    assert overridden.repairs
    _assert_donor_geometry(overridden.structure, donor)


def test_unknown_source_zinc_charge_stays_unknown(monkeypatch):
    source, family, donor, _ = _example(metal_charge=None)
    _register(monkeypatch, family)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((LIGAND,))
    )
    assert result.repairs
    index = result.structure.constitution.atom_index(AtomRef(METAL, "M"))
    assert result.structure.topology.formal_charge(index) is None
    with pytest.raises(RdkitProjectionError, match="charge"):
        to_rdkit_molecule(result.structure)
    _assert_donor_geometry(result.structure, donor)


@pytest.mark.parametrize("route", ["catalogue", "override", "template"])
def test_degenerate_partner_geometry_does_not_publish_partial_h(route, monkeypatch):
    source, family, donor, native = _example(coincident=True)
    _register(monkeypatch, family)
    library = build_default_component_library()
    if route == "template":
        library = library.with_template(_template(native, family))
    evidence = family.hypotheses(LIGAND, donor_atom_name=donor)[0]
    result = add_retained_non_polymer_hydrogens(
        source,
        library,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=(evidence,) if route == "override" else (),
    )
    assert result.structure == source
    assert not result.repairs
    assert result.issues[0].kind is (
        ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        if route == "template"
        else ValidationIssueKind.CHEMISTRY_CONTRADICTION
    )


@pytest.mark.parametrize("route", ["catalogue", "override", "template"])
def test_two_observed_donor_h_are_not_deleted_to_fit_the_bound_state(
    route, monkeypatch
):
    source, family, donor, native = _example(source_h="H", excess_h=True)
    _register(monkeypatch, family)
    library = build_default_component_library()
    if route == "template":
        library = library.with_template(_template(native, family))
    evidence = family.hypotheses(LIGAND, donor_atom_name=donor)[0]
    result = add_retained_non_polymer_hydrogens(
        source,
        library,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=(evidence,) if route == "override" else (),
    )
    assert result.structure == source
    assert not result.repairs


@pytest.mark.parametrize(
    "smiles",
    [
        "[O-]C(=O)CCS(=O)(=O)[NH-]",
        "C[C@H](F)CS(=O)(=O)[NH-]",
        "C[C@@H](F)CS(=O)(=O)[NH-]",
    ],
)
def test_remote_charge_and_actual_stereo_are_not_lost(smiles, monkeypatch):
    source, family, donor, _ = _example(smiles)
    _register(monkeypatch, family)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((LIGAND,))
    )
    assert result.repairs
    _preserved(source, result.structure)
    native = to_rdkit_molecule(result.structure)
    assert (
        sum(a.GetFormalCharge() for a in native.GetAtoms())
        == sum(
            a.GetFormalCharge() for a in template_without_hydrogens(smiles).GetAtoms()
        )
        + 2
    )
    local = Chem.RWMol(native)
    for i in reversed(range(local.GetNumAtoms())):
        if local.GetAtomWithIdx(i).GetAtomicNum() == 30:
            local.RemoveAtom(i)
    Chem.SanitizeMol(local)
    Chem.AssignStereochemistryFrom3D(local, replaceExistingTags=True)
    Chem.AssignStereochemistry(local, cleanIt=True, force=True)
    expected = template_without_hydrogens(smiles)
    for atom in expected.GetAtoms():
        if not atom.HasProp("_CIPCode"):
            continue
        name = family.heavy_atom_names[atom.GetIdx()]
        observed = next(
            a
            for a in local.GetAtoms()
            if a.GetPDBResidueInfo().GetName().strip() == name
        )
        assert observed.GetProp("_CIPCode") == atom.GetProp("_CIPCode")


def test_declared_stereo_opposite_to_actual_geometry_is_atomic(monkeypatch):
    source, family, _, _ = _example("C[C@H](F)CS(=O)(=O)[NH-]", mirror=True)
    _register(monkeypatch, family)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((LIGAND,))
    )
    assert result.structure == source
    assert not result.repairs
    assert any("stereochemistry" in issue.message for issue in result.issues)


def test_original_1jd0_window_with_deposited_links_prepares_five_hydrogens():
    text = Path("tests/fixtures/pdb/refinement/1jd0_gln92_local.pdb").read_text()
    links = (
        "LINK         NE2 HIS A  94                ZN    ZN A 901"
        "     1555   1555  2.06",
        "LINK         NE2 HIS A  96                ZN    ZN A 901"
        "     1555   1555  2.04",
        "LINK         ND1 HIS A 119                ZN    ZN A 901"
        "     1555   1555  2.04",
        "LINK        ZN    ZN A 901                 N1  AZM A1400"
        "     1555   1555  1.96",
    )
    frozen = read_structure_string(text, FileFormat.PDB)
    source = read_structure_string("\n".join(links) + "\n" + text, FileFormat.PDB)
    assert source.constitution == frozen.constitution
    assert source.geometry == frozen.geometry
    assert source.topology.atom_topologies == frozen.topology.atom_topologies
    assert len(source.constitution.atom_slots) == 247
    residue_id = ResidueId("A", 1400)
    unchanged = add_retained_non_polymer_hydrogens(
        frozen,
        target_residue_ids=frozenset((residue_id,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert unchanged.structure == frozen
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((residue_id,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.repairs
    assert len(result.structure.constitution.atom_slots) == 252
    _preserved(source, result.structure)
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(residue_id, "N1"))
        )
        == -1
    )
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(residue_id),
    )
    assert sum(parent == "N1" for parent in anchors.values()) == 1
    assert (
        sum(
            b.relationship_type is BondRelationshipType.METAL_COORDINATION
            for b in result.structure.topology.bonds
        )
        == 4
    )


@pytest.mark.parametrize(
    "smiles", ["CS(=O)(=O)[NH-]", "CCS(=O)(=O)[NH-]", "c1ccccc1S(=O)(=O)[NH-]"]
)
def test_primary_sulfonamidate_qualifies_independent_scaffolds(smiles):
    molecule = template_without_hydrogens(smiles)
    donor = next(a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() == 7)
    role = NitrogenCoordinationDonor.from_molecule(molecule, donor)
    assert role is NitrogenCoordinationDonor.PRIMARY_SULFONAMIDATE
    assert role.hydrogen_count == 1
    assert role.admits_metal("ZN")
    assert not role.admits_metal("FE")
    hydrated = Chem.AddHs(molecule)
    assert NitrogenCoordinationDonor.from_molecule(hydrated, donor) is role


@pytest.mark.parametrize(
    "smiles",
    ["CS(=O)(=O)N", "CS(=O)(=O)[N-]C", "CS(=O)[NH-]", "COS(=O)(=O)[NH-]", "C[NH-]"],
)
def test_unadmitted_nitrogen_motifs_remain_explicitly_unsupported(smiles):
    molecule = template_without_hydrogens(smiles)
    donor = next(a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() == 7)
    with pytest.raises(UnsupportedCoordinationChemistryError):
        NitrogenCoordinationDonor.from_molecule(molecule, donor)


def test_common_family_binds_sulfonamidate_without_component_dispatch():
    smiles = "CCS(=O)(=O)[NH-]"
    molecule = template_without_hydrogens(smiles)
    names = tuple(f"Q{i}" for i in range(molecule.GetNumAtoms()))
    family = CoordinationMicrostates(
        names, (smiles,), "Independent ethylsulfonamidate chemistry."
    )
    donor = names[-1]
    assert family.admits_donor(donor)
    assert not family.admits_donor(names[0])
    assert (
        family.hypotheses(ResidueId("X", 1), donor_atom_name=donor)[0].smiles == smiles
    )


@pytest.mark.parametrize("route", ["catalogue", "override", "template"])
def test_bridged_donor_is_not_reduced_to_one_partner(route, monkeypatch):
    source, family, donor, native = _example(second_partner=True)
    _register(monkeypatch, family)
    library = build_default_component_library()
    if route == "template":
        library = library.with_template(_template(native, family))
    evidence = family.hypotheses(LIGAND, donor_atom_name=donor)[0]
    result = add_retained_non_polymer_hydrogens(
        source,
        library,
        target_residue_ids=frozenset((LIGAND,)),
        chemistry_evidence=(evidence,) if route == "override" else (),
    )
    assert result.structure == source
    assert not result.repairs


def test_bound_template_cannot_hide_a_remote_original_charge(monkeypatch):
    source, family, donor, native = _example("[O-]C(=O)CCS(=O)(=O)[NH-]")
    _register(monkeypatch, family)
    original_topologies = list(source.topology.atom_topologies)
    original_topologies[0] = AtomTopology(formal_charge=-1)
    observed_topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=tuple(original_topologies),
        bonds=source.topology.bonds,
    )
    observation = StructureObservation.from_source_facets(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=observed_topology,
    )
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=observation),
        )
    )
    neutral = Chem.AddHs(template_without_hydrogens("OC(=O)CCS(=O)(=O)[NH-]"))
    assert EmbedMolecule(neutral, randomSeed=17) == 0
    library = build_default_component_library().with_template(
        _template(neutral, family)
    )
    result = add_retained_non_polymer_hydrogens(
        source, library, target_residue_ids=frozenset((LIGAND,))
    )
    assert result.structure == source
    assert not result.repairs


def test_bound_template_does_not_publish_charged_remote_source_h():
    source, family, _, native = _example(source_h="H")
    hydrogen = source.constitution.atom_index(AtomRef(LIGAND, "SRC"))
    carbon = source.constitution.atom_index(AtomRef(LIGAND, "Q0"))
    parent_h = next(
        a for a in native.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1
    )
    position = Vec3.from_iterable(
        native.GetConformer().GetAtomPosition(parent_h.GetIdx())
    )
    geometries = source.geometry.with_atom_geometry(
        hydrogen, source.geometry.atom_geometry(hydrogen).with_position(position)
    )
    atoms = list(source.topology.atom_topologies)
    atoms[hydrogen.value] = AtomTopology(formal_charge=1)
    bonds = tuple(b for b in source.topology.bonds if hydrogen not in b.endpoint_pair())
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=tuple(atoms),
        bonds=(
            *bonds,
            TopologyBond(
                hydrogen, carbon, order=1, provenance=BondProvenance.SOURCE_EXPLICIT
            ),
        ),
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=geometries,
        topology=topology,
        provenance=replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=None),
        ),
        polymer_blueprint=source.polymer_blueprint,
    )
    library = build_default_component_library().with_template(_template(native, family))
    result = add_retained_non_polymer_hydrogens(
        source, library, target_residue_ids=frozenset((LIGAND,))
    )
    assert result.structure == source
    assert not result.repairs
