"""Mapped ligand chemistry must retain physical coordination and source evidence."""

from dataclasses import replace
from math import cos, pi, sin

import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
)
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.inference.idealized_stereochemistry import (
    component_reference_molecule,
)
from protrepair.chemistry.inference.rdkit_graph import (
    covalent_bond_definitions_from_molecule,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_heavy_bond_definitions,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import (
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.scope import WholeStructureScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
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
    resolve_retained_non_polymer_topology,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)

LIGAND_ID = ResidueId("L", 1)
ATOM_NAMES = ("FE", "N1", "C1", "C2", "C3", "C4", "C5")


def _evidence() -> RetainedNonPolymerChemistryEvidence:
    return RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "[Fe+3]<-n1ccccc1", ATOM_NAMES
    ).to_evidence()


def _source(
    *,
    typed: bool = False,
    reverse_slots: bool = False,
    iron_charge: int | None = None,
    metal_symbol: str = "FE",
    ring_size: int = 6,
) -> ProteinStructure:
    positions = (
        Vec3(0, 0, 0),
        Vec3(2.3, 0, 0),
        Vec3(2.995, 1.204, 0),
        Vec3(4.385, 1.204, 0),
        Vec3(5.08, 0, 0),
        Vec3(4.385, -1.204, 0),
        Vec3(2.995, -1.204, 0),
    )
    if ring_size == 5:
        radius = 1.39 / (2 * sin(pi / 5))
        positions = (
            positions[0],
            *(
                Vec3(
                    2.3 + radius + radius * cos(pi + i * 2 * pi / 5),
                    radius * sin(pi + i * 2 * pi / 5),
                    0,
                )
                for i in range(5)
            ),
        )
    names = ATOM_NAMES[: ring_size + 1]
    atoms = tuple(
        atom_payload(
            name,
            metal_symbol if name == "FE" else "N" if name == "N1" else "C",
            position,
            occupancy=0.8,
            b_factor=7.0,
            formal_charge=iron_charge if name == "FE" else None,
        )
        for name, position in zip(names, positions, strict=True)
    )
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="XXX",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=tuple(reversed(atoms)) if reverse_slots else atoms,
            ),
        ),
        source_format=FileFormat.MMCIF if typed else FileFormat.PDB,
    )
    relationship = (
        BondRelationshipType.METAL_COORDINATION
        if typed
        else BondRelationshipType.UNKNOWN
    )
    source_bond = TopologyBond(
        structure.constitution.atom_index(AtomRef(LIGAND_ID, "FE")),
        structure.constitution.atom_index(AtomRef(LIGAND_ID, "N1")),
        order=None,
        relationship_type=relationship,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=(
                SourceBondRecordType.MMCIF_STRUCT_CONN
                if typed
                else SourceBondRecordType.PDB_CONECT
            ),
            source_id="coord1" if typed else "CONECT",
            reported_relationship_type=relationship,
        ),
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(source_bond,),
        ),
        provenance=structure.provenance,
    )
    file_format = FileFormat.MMCIF if typed else FileFormat.PDB
    return read_structure_string(
        write_structure_string(structure, file_format=file_format),
        file_format=file_format,
    )


def _with_bonds(
    structure: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=structure.provenance,
    )


def _observed(structure: ProteinStructure) -> ProteinStructure:
    return read_structure_string(
        write_structure_string(structure, file_format=FileFormat.MMCIF),
        file_format=FileFormat.MMCIF,
    )


def _add_atom(
    structure: ProteinStructure, name: str, element: str, position: Vec3
) -> ProteinStructure:
    residue = structure.constitution.ligands[0]
    geometry = structure.residue_geometry(
        structure.constitution.residue_index(LIGAND_ID)
    )
    charges = tuple(
        (
            atom.name,
            structure.topology.formal_charge(
                structure.constitution.atom_index(AtomRef(LIGAND_ID, atom.name))
            ),
        )
        for atom in residue.atom_sites
    )
    return structure.with_updated_residue_facets(
        replace(residue, atom_sites=(*residue.atom_sites, AtomSite(name, element))),
        residue_geometry=ResidueGeometry(
            atoms_by_name={**geometry.atoms_by_name, name: AtomGeometry(position)}
        ),
        formal_charge_by_atom_name=charges,
    )


def _external_source() -> ProteinStructure:
    source = _source()
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND_ID))
    metal, metal_geometry, metal_charges = residue_payload(
        component_id="FE",
        residue_id=ResidueId("M", 1),
        is_hetero=True,
        atoms=(atom_payload("FE", "FE", Vec3(0, 0, 0)),),
    )
    split = source.with_ligand_facets(
        ligand_sites=(
            replace(
                residue,
                atom_sites=tuple(
                    atom for atom in residue.atom_sites if atom.name != "FE"
                ),
            ),
            metal,
        ),
        ligand_geometries=(
            ResidueGeometry(
                atoms_by_name={
                    name: atom
                    for name, atom in geometry.atoms_by_name.items()
                    if name != "FE"
                }
            ),
            metal_geometry,
        ),
        ligand_formal_charge_payloads=((), metal_charges),
    )
    bond = TopologyBond(
        split.constitution.atom_index(AtomRef(LIGAND_ID, "N1")),
        split.constitution.atom_index(AtomRef(metal.residue_id, "FE")),
        order=None,
        relationship_type=BondRelationshipType.METAL_COORDINATION,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            SourceBondRecordType.MMCIF_STRUCT_CONN,
            reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
        ),
    )
    return _observed(_with_bonds(split, (bond,)))


def _external_evidence() -> RetainedNonPolymerChemistryEvidence:
    return RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "n1ccccc1", ATOM_NAMES[1:]
    ).to_evidence()


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize("reverse_slots", (False, True))
def test_explicit_dative_chemistry_prepares_h_charge_and_typed_topology(
    typed, reverse_slots
):
    source = _source(typed=typed, reverse_slots=reverse_slots)
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )

    assert not result.issues
    prepared = result.structure
    residue = prepared.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 5
    for name in ATOM_NAMES:
        ref = AtomRef(LIGAND_ID, name)
        before = source.constitution.atom_index(ref)
        after = prepared.constitution.atom_index(ref)
        assert prepared.geometry.atom_geometry(after) == source.geometry.atom_geometry(
            before
        )
        assert prepared.topology.formal_charge(after) == (3 if name == "FE" else 0)
    bond = prepared.topology.bond_between(
        prepared.constitution.atom_index(AtomRef(LIGAND_ID, "FE")),
        prepared.constitution.atom_index(AtomRef(LIGAND_ID, "N1")),
    )
    assert bond is not None
    assert bond.relationship_type is BondRelationshipType.METAL_COORDINATION
    assert bond.order is None
    assert not bond.aromatic
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert bond.source_metadata == source.topology.bonds[0].source_metadata
    observation = prepared.provenance.ingress.observation
    assert observation is not None
    assert observation is source.provenance.ingress.observation
    assert observation.formal_charge(AtomRef(LIGAND_ID, "FE")) is None
    resolution = resolve_retained_non_polymer_chemistry(
        prepared, residue, component_library=ComponentLibrary({}), evidence=_evidence()
    )
    assert resolution.has_hydrogen_topology(prepared, residue)
    assert len(resolution.heavy_bond_definitions) == 6


@pytest.mark.parametrize(
    "replacement", (None, BondRelationshipType.UNKNOWN, BondRelationshipType.COVALENT)
)
def test_missing_or_wrong_coordination_invalidates_fresh_readiness(replacement):
    prepared = add_retained_non_polymer_hydrogens(
        _source(), ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    ).structure
    assert len(prepared.constitution.ligands[0].atom_sites) == 12
    bonds = tuple(
        bond
        for bond in prepared.topology.bonds
        if bond.relationship_type is not BondRelationshipType.METAL_COORDINATION
    )
    if replacement is not None:
        bonds += (
            TopologyBond(
                prepared.constitution.atom_index(AtomRef(LIGAND_ID, "FE")),
                prepared.constitution.atom_index(AtomRef(LIGAND_ID, "N1")),
                order=1 if replacement is BondRelationshipType.COVALENT else None,
                relationship_type=replacement,
                provenance=BondProvenance.EVIDENCE_RESOLVED,
            ),
        )
    stale = _with_bonds(prepared, bonds)
    for structure, needs_resolution in ((prepared, False), (stale, True)):
        fact = (
            StructureChemistryObservation(
                structure,
                component_library=ComponentLibrary({}),
                retained_non_polymer_chemistry_evidence=(_evidence(),),
            )
            .whole_structure_facts()[1]
            .retained_non_polymer_facts[0]
        )
        assert fact.requires_topology_resolution() is needs_resolution


def test_original_known_charge_cannot_be_hidden_by_current_unknown_assignment():
    source = _source(iron_charge=2)
    index = source.constitution.atom_index(AtomRef(LIGAND_ID, "FE"))
    atoms = list(source.topology.atom_topologies)
    atoms[index.value] = AtomTopology(formal_charge=None)
    hidden = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=tuple(atoms),
            bonds=source.topology.bonds,
        ),
        provenance=source.provenance,
    )
    result = add_retained_non_polymer_hydrogens(
        hidden, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )

    assert result.structure is hidden
    assert not result.repairs
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for issue in result.issues
    )
    observation = hidden.provenance.ingress.observation
    assert observation is not None
    assert observation.formal_charge(AtomRef(LIGAND_ID, "FE")) == 2


def test_typed_coordination_cannot_be_replaced_by_a_source_covalent_claim():
    source = _source(typed=True)
    source = _with_bonds(
        source,
        (
            replace(
                source.topology.bonds[0],
                relationship_type=BondRelationshipType.COVALENT,
                order=1,
            ),
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )

    assert result.structure is source
    assert result.issues
    assert not result.repairs


@pytest.mark.parametrize("smiles", ("[Fe+3]->n1ccccc1", "[Fe+3]<-[nH+]1ccccc1"))
def test_incompatible_declared_donor_refuses_without_fallback(smiles):
    source = _source()
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, smiles, ATOM_NAMES
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


@pytest.mark.parametrize("order", (2, 3))
def test_reported_covalent_order_cannot_be_discarded_for_dative_chemistry(order):
    source = _source()
    bond = source.topology.bonds[0]
    assert bond.source_metadata is not None
    source = _with_bonds(
        source,
        (
            replace(
                bond,
                order=order,
                source_metadata=replace(bond.source_metadata, reported_order=order),
            ),
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


@pytest.mark.parametrize("typed", (False, True))
def test_reported_single_is_retained_without_giving_the_donor_covalent_valence(typed):
    source = _source(typed=typed)
    bond = source.topology.bonds[0]
    assert bond.source_metadata is not None
    source = _with_bonds(
        source,
        (
            replace(
                bond,
                order=1,
                source_metadata=replace(bond.source_metadata, reported_order=1),
            ),
        ),
    )
    source = _observed(source)
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert not result.issues
    first = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "FE"))
    second = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "N1"))
    selected = result.structure.topology.bond_between(first, second)
    assert selected is not None
    assert selected.relationship_type is BondRelationshipType.METAL_COORDINATION
    assert selected.order is None
    assert selected.source_metadata is not None
    assert selected.source_metadata.reported_order == 1


@pytest.mark.parametrize("symbol", ("H", "D", "T"))
def test_source_donor_hydrogen_is_not_deleted_to_make_coordination_work(symbol):
    source = _add_atom(_source(typed=True), "HD", symbol, Vec3(1.3, 0, 0))
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND_ID, "N1")),
        source.constitution.atom_index(AtomRef(LIGAND_ID, "HD")),
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = _observed(_with_bonds(source, (*source.topology.bonds, bond)))
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


@pytest.mark.parametrize("charge", (-1, 0, 1))
@pytest.mark.parametrize("mask_original_charge", (False, True))
def test_mapped_chemistry_requires_neutral_bound_source_hydrogen(
    charge, mask_original_charge
):
    source = _source(typed=True)
    carbon = source.residue_geometry(
        source.constitution.residue_index(LIGAND_ID)
    ).position("C1")
    source = _add_atom(source, "HSRC", "H", carbon.with_offset(0, 1.1, 0))
    hydrogen = source.constitution.atom_index(AtomRef(LIGAND_ID, "HSRC"))
    charges = list(source.topology.atom_topologies)
    charges[hydrogen.value] = AtomTopology(charge)
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND_ID, "C1")),
        hydrogen,
        order=1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            SourceBondRecordType.MMCIF_STRUCT_CONN,
            reported_order=1,
            reported_relationship_type=BondRelationshipType.COVALENT,
        ),
    )
    source = _observed(
        ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=StructureTopology(
                constitution=source.constitution,
                atom_topologies=tuple(charges),
                bonds=(*source.topology.bonds, bond),
            ),
            provenance=source.provenance,
        )
    )
    if mask_original_charge:
        index = source.constitution.atom_index(AtomRef(LIGAND_ID, "HSRC"))
        charges = list(source.topology.atom_topologies)
        charges[index.value] = AtomTopology(0)
        source = ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=StructureTopology(
                constitution=source.constitution,
                atom_topologies=tuple(charges),
                bonds=source.topology.bonds,
            ),
            provenance=source.provenance,
        )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    if charge == 0:
        assert not result.issues
        assert result.repairs
    else:
        assert result.structure is source
        assert result.issues
        assert not result.repairs
        resolution = resolve_retained_non_polymer_chemistry(
            source,
            source.constitution.ligands[0],
            component_library=ComponentLibrary({}),
            evidence=_evidence(),
        )
        assert not resolution.source.is_resolved()


def test_anionic_aromatic_donor_keeps_its_charge_and_no_donor_hydrogen():
    source = _source(ring_size=5)
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "[Fe+2]<-[n-]1cccc1", ATOM_NAMES[:6]
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )
    assert not result.issues
    index = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "N1"))
    assert result.structure.topology.formal_charge(index) == -1
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND_ID),
    )
    assert sorted(anchors.values()) == ["C1", "C2", "C3", "C4"]


def test_disconnected_counterion_does_not_change_selected_coordination():
    source = _observed(_add_atom(_source(), "NA", "NA", Vec3(20, 0, 0)))
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "[Na+].[Fe+3]<-n1ccccc1", ("NA", *ATOM_NAMES)
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )
    assert not result.issues
    sodium = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "NA"))
    assert result.structure.topology.formal_charge(sodium) == 1
    assert not any(bond.involves(sodium) for bond in result.structure.topology.bonds)


def test_missing_original_external_partner_cannot_receive_isolated_success():
    source = _external_source()
    missing = source.without_atom_refs((AtomRef(ResidueId("M", 1), "FE"),))
    result = add_retained_non_polymer_hydrogens(
        missing,
        ComponentLibrary({}),
        chemistry_evidence=(_external_evidence(),),
        target_residue_ids=frozenset((LIGAND_ID,)),
    )
    assert result.structure is missing
    assert result.issues
    assert not result.repairs
    resolution = resolve_retained_non_polymer_chemistry(
        missing,
        missing.constitution.ligands[0],
        component_library=ComponentLibrary({}),
        evidence=_external_evidence(),
    )
    assert not resolution.source.is_resolved()


def test_missing_original_external_relationship_cannot_receive_isolated_success():
    source = _with_bonds(_external_source(), ())
    result = add_retained_non_polymer_hydrogens(
        source,
        ComponentLibrary({}),
        chemistry_evidence=(_external_evidence(),),
        target_residue_ids=frozenset((LIGAND_ID,)),
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


def test_external_aromatic_donor_cannot_bridge_two_metals():
    source = _external_source()
    sites = source.constitution.ligands
    metal = residue_payload(
        component_id="FE",
        residue_id=ResidueId("M", 2),
        is_hetero=True,
        atoms=(atom_payload("FE", "FE", Vec3(4.6, 0, 0)),),
    )
    source = source.with_ligand_facets(
        ligand_sites=(*sites, metal[0]),
        ligand_geometries=(
            *(
                source.residue_geometry(
                    source.constitution.residue_index(site.residue_id)
                )
                for site in sites
            ),
            metal[1],
        ),
    )
    extra = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND_ID, "N1")),
        source.constitution.atom_index(AtomRef(ResidueId("M", 2), "FE")),
        order=None,
        relationship_type=BondRelationshipType.METAL_COORDINATION,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = _observed(_with_bonds(source, (*source.topology.bonds, extra)))
    result = add_retained_non_polymer_hydrogens(
        source,
        ComponentLibrary({}),
        chemistry_evidence=(_external_evidence(),),
        target_residue_ids=frozenset((LIGAND_ID,)),
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


def test_original_donor_hydrogen_cannot_be_hidden_by_removing_current_atom():
    source = add_retained_non_polymer_hydrogens(
        _source(typed=True), ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    ).structure
    source = _add_atom(source, "HD", "D", Vec3(1.3, 0, 0))
    extra = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND_ID, "N1")),
        source.constitution.atom_index(AtomRef(LIGAND_ID, "HD")),
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = _observed(_with_bonds(source, (*source.topology.bonds, extra)))
    hidden = source.without_atom_refs((AtomRef(LIGAND_ID, "HD"),))
    result = add_retained_non_polymer_hydrogens(
        hidden, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert result.structure is hidden
    assert result.issues
    resolution = resolve_retained_non_polymer_chemistry(
        hidden,
        hidden.constitution.ligands[0],
        component_library=ComponentLibrary({}),
        evidence=_evidence(),
    )
    assert not resolution.has_hydrogen_topology(hidden, hidden.constitution.ligands[0])


def test_original_donor_h_cannot_be_hidden_by_rebinding_it_in_current_topology():
    prepared = add_retained_non_polymer_hydrogens(
        _source(typed=True), ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    ).structure
    carbon = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "C1"))
    nitrogen = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "N1"))
    original_bonds = []
    changed = False
    for bond in prepared.topology.bonds:
        if bond.involves(carbon):
            other = next(index for index in bond.endpoint_pair() if index != carbon)
            if prepared.constitution.atom_site_at(other).is_hydrogen():
                bond = replace(bond, atom_index_1=nitrogen, atom_index_2=other)
                changed = True
        original_bonds.append(bond)
    assert changed
    original = _observed(_with_bonds(prepared, tuple(original_bonds)))
    current = prepared.with_provenance(original.provenance)
    resolution = resolve_retained_non_polymer_chemistry(
        current,
        current.constitution.ligands[0],
        component_library=ComponentLibrary({}),
        evidence=_evidence(),
    )
    assert not resolution.source.is_resolved()
    assert "donor hydrogen" in resolution.failure_reason


def test_unreported_single_order_is_not_erased_to_make_coordination_work():
    source = _source()
    bond = replace(source.topology.bonds[0], order=1, source_metadata=None)
    expected = replace(
        bond, order=None, relationship_type=BondRelationshipType.METAL_COORDINATION
    )
    with pytest.raises(ValueError, match="bond order"):
        bond.with_resolved_chemistry(expected)


def test_zinc_preparation_preserves_explicit_charge_without_native_hydrides():
    source = _source(metal_symbol="ZN")
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "[Zn+2]<-n1ccccc1", ATOM_NAMES
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )
    assert not result.issues
    native = to_rdkit_molecule(result.structure)
    assert native.GetAtomWithIdx(0).GetFormalCharge() == 2
    assert all(n.GetAtomicNum() != 1 for n in native.GetAtomWithIdx(0).GetNeighbors())
    assert (
        source.topology.formal_charge(
            source.constitution.atom_index(AtomRef(LIGAND_ID, "FE"))
        )
        is None
    )


def test_refused_coordination_does_not_block_independent_ligand_preparation():
    source = _source(iron_charge=2)
    residue = source.constitution.ligands[0]
    neighbor_id = ResidueId("L", 2)
    neighbor = residue_payload(
        component_id="YYY",
        residue_id=neighbor_id,
        is_hetero=True,
        atoms=(
            atom_payload("C1", "C", Vec3(20, 0, 0)),
            atom_payload("O1", "O", Vec3(21.43, 0, 0)),
        ),
    )
    charges = tuple(
        (
            atom.name,
            source.topology.formal_charge(
                source.constitution.atom_index(AtomRef(LIGAND_ID, atom.name))
            ),
        )
        for atom in residue.atom_sites
    )
    source = _observed(
        source.with_ligand_facets(
            ligand_sites=(residue, neighbor[0]),
            ligand_geometries=(
                source.residue_geometry(source.constitution.residue_index(LIGAND_ID)),
                neighbor[1],
            ),
            ligand_formal_charge_payloads=(charges, neighbor[2]),
        )
    )
    evidence = (
        _evidence(),
        RetainedNonPolymerChemistryOverride(
            neighbor_id, "CO", ("C1", "O1")
        ).to_evidence(),
    )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=evidence
    )
    assert result.issues
    refused = result.structure.constitution.residue_or_ligand(LIGAND_ID)
    independent = result.structure.constitution.residue_or_ligand(neighbor_id)
    assert refused == residue
    assert independent is not None
    assert sum(atom.is_hydrogen() for atom in independent.atom_sites) == 4
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "FE"))
        )
        == 2
    )


@pytest.mark.parametrize("strict", (False, True))
def test_public_workflow_uses_explicit_coordination_without_proximity_fallback(strict):
    result = process_structure(
        _source(),
        ingress=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=(
                RetainedNonPolymerChemistryOverride(
                    LIGAND_ID, _evidence().smiles, ATOM_NAMES
                ),
            ),
        ),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
        transform_requests=WorkflowTransformRequests(
            allow_retained_non_polymer_rdkit_fallback=not strict
        ),
    )
    assert len(result.structure.constitution.ligands[0].atom_sites) == 12
    assert not result.has_errors()
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "FE"))
        )
        == 3
    )


@pytest.mark.parametrize("symbol", ("H", "D", "T"))
def test_complete_source_isotope_geometry_survives_topology_only_rebinding(symbol):
    source = add_retained_non_polymer_hydrogens(
        _source(typed=True), ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    ).structure
    residue = source.constitution.ligands[0]
    source = source.with_updated_residue_facets(
        replace(
            residue,
            atom_sites=tuple(
                AtomSite(atom.name, symbol) if atom.is_hydrogen() else atom
                for atom in reversed(residue.atom_sites)
            ),
        ),
        residue_geometry=source.residue_geometry(
            source.constitution.residue_index(LIGAND_ID)
        ),
        formal_charge_by_atom_name=tuple(
            (
                atom.name,
                source.topology.formal_charge(
                    source.constitution.atom_index(AtomRef(LIGAND_ID, atom.name))
                ),
            )
            for atom in residue.atom_sites
        ),
    )
    source = _observed(source)
    result = resolve_retained_non_polymer_topology(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert not result.issues
    assert result.structure.geometry == source.geometry
    assert result.structure.constitution == source.constitution
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        current = result.structure.constitution.atom_index(ref)
        assert result.structure.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(index)
    repeated = resolve_retained_non_polymer_topology(
        result.structure, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert repeated.structure is result.structure
    assert not repeated.issues


def test_source_aromatic_resonance_constraints_remain_covalent_with_coordination():
    source = _source()
    definitions = retained_non_polymer_evidence_heavy_bond_definitions(_evidence())
    target = next(
        bond
        for bond in definitions
        if frozenset((bond.atom_name_1, bond.atom_name_2)) == frozenset(("C1", "C2"))
    )
    order = 3 - target.order
    extra = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND_ID, "C1")),
        source.constitution.atom_index(AtomRef(LIGAND_ID, "C2")),
        order=order,
        relationship_type=BondRelationshipType.COVALENT,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            SourceBondRecordType.MMCIF_STRUCT_CONN,
            reported_order=order,
            reported_relationship_type=BondRelationshipType.COVALENT,
        ),
    )
    source = _with_bonds(source, (*source.topology.bonds, extra))
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert not result.issues
    actual = result.structure.topology.bond_between(
        *(
            result.structure.constitution.atom_index(AtomRef(LIGAND_ID, name))
            for name in ("C1", "C2")
        )
    )
    assert actual is not None and actual.order == order and actual.aromatic
    assert actual.source_metadata == extra.source_metadata


def test_native_hydrogenation_failure_has_no_successful_organic_fallback(monkeypatch):
    source = _source()

    def fail(*args, **kwargs):
        raise RuntimeError("controlled H placement failure")

    monkeypatch.setattr(Chem, "AddHs", fail)
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


def test_prepared_native_consumer_keeps_dative_and_does_not_invent_implicit_h():
    prepared = add_retained_non_polymer_hydrogens(
        _source(), ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    ).structure
    native = to_rdkit_molecule(prepared)
    donor = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "N1"))
    metal = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "FE"))
    assert (
        native.GetBondBetweenAtoms(donor.value, metal.value).GetBondType()
        == Chem.BondType.DATIVE
    )
    assert not any(
        atom.GetNumImplicitHs() or atom.GetNumRadicalElectrons()
        for atom in native.GetAtoms()
    )
    assert Chem.MolToSmiles(native) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(_evidence().smiles))
    )


def _selected_donor_case(smiles, *, charge=None):
    native = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert EmbedMolecule(native, randomSeed=17) == 0
    names = tuple(
        f"Q{i}" if atom.GetAtomicNum() != 1 else f"H{i}"
        for i, atom in enumerate(native.GetAtoms())
    )
    positions = native.GetConformer()
    component = IdealizedComponent(
        "QDN",
        None,
        tuple(
            IdealizedComponentAtom(
                names[atom.GetIdx()],
                atom.GetSymbol().upper(),
                atom.GetFormalCharge(),
                ideal_position=tuple(positions.GetAtomPosition(atom.GetIdx())),
            )
            for atom in native.GetAtoms()
        ),
        covalent_bond_definitions_from_molecule(
            native, atom_names_by_index=dict(enumerate(names))
        ),
    )
    template = ResidueTemplate(
        ChemicalComponentDefinition(
            "QDN",
            component.heavy_atom_names(),
            component.heavy_bonds(),
            {atom.atom_name: atom.formal_charge for atom in component.heavy_atoms()},
        ),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(component),
    )
    donor_index = next(
        atom.GetIdx() for atom in native.GetAtoms() if atom.GetAtomicNum() == 7
    )
    donor = AtomRef(LIGAND_ID, names[donor_index])
    metal_id = ResidueId("Z", 9)
    position = Vec3.from_iterable(positions.GetAtomPosition(donor_index))
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="QDN",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        names[atom.GetIdx()],
                        atom.GetSymbol().upper(),
                        Vec3.from_iterable(positions.GetAtomPosition(atom.GetIdx())),
                        formal_charge=charge if atom.GetIdx() == donor_index else None,
                    )
                    for atom in native.GetAtoms()
                    if atom.GetAtomicNum() != 1
                ),
            ),
            residue_payload(
                component_id="ZN",
                residue_id=metal_id,
                is_hetero=True,
                atoms=(
                    atom_payload(
                        "ZN", "ZN", Vec3(position.x + 2, position.y, position.z)
                    ),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    bond = TopologyBond(
        source.constitution.atom_index(donor),
        source.constitution.atom_index(AtomRef(metal_id, "ZN")),
        order=None,
        relationship_type=BondRelationshipType.METAL_COORDINATION,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            SourceBondRecordType.MMCIF_STRUCT_CONN,
            reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
        ),
    )
    return (
        _observed(_with_bonds(source, (bond,))),
        ComponentLibrary({"QDN": template}),
        donor,
    )


@pytest.mark.parametrize("smiles", ("n1ccccc1", "CS(=O)(=O)[NH-]", "CCS(=O)(=O)[NH-]"))
def test_selected_template_qualifies_donor_without_automatic_microstate_selection(
    smiles,
):
    source, library, donor = _selected_donor_case(smiles)
    result = add_retained_non_polymer_hydrogens(
        source,
        library,
        target_residue_ids=frozenset((LIGAND_ID,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert not result.issues
    assert result.repairs
    prepared = result.structure
    residue = prepared.constitution.residue_or_ligand(LIGAND_ID)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        prepared, residue, component_library=library
    )
    assert resolution.source.value == "template"
    assert resolution.has_coordination_topology(prepared)
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(ResidueId("Z", 9), "ZN"))
        )
        is None
    )
    assert prepared.topology.formal_charge(prepared.constitution.atom_index(donor)) == (
        -1 if "[NH-]" in smiles else None
    )
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        assert prepared.geometry.atom_geometry(
            prepared.constitution.atom_index(ref)
        ) == source.geometry.atom_geometry(index)
    repeated = add_retained_non_polymer_hydrogens(
        prepared,
        library,
        target_residue_ids=frozenset((LIGAND_ID,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert repeated.structure is prepared and not repeated.issues


def test_bound_template_does_not_erase_a_known_neutral_donor_charge():
    source, library, _ = _selected_donor_case("CS(=O)(=O)[NH-]", charge=0)
    result = add_retained_non_polymer_hydrogens(
        source, library, target_residue_ids=frozenset((LIGAND_ID,))
    )
    assert result.structure is source and result.issues and not result.repairs


def test_fallback_does_not_ignore_an_unadmitted_coordinated_amine():
    source, _, _ = _selected_donor_case("CCN")
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary(), target_residue_ids=frozenset((LIGAND_ID,))
    )
    assert result.structure is source and result.issues and not result.repairs
    residue = source.constitution.residue_or_ligand(LIGAND_ID)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=ComponentLibrary()
    )
    assert not resolution.source.is_resolved()


@pytest.mark.parametrize("route", ("template", "evidence"))
def test_coincident_partner_refuses_donor_h_without_publishing_partial_changes(route):
    smiles = "CS(=O)(=O)[NH-]"
    source, library, donor = _selected_donor_case(smiles)
    metal_id = ResidueId("Z", 9)
    metal = source.constitution.residue_or_ligand(metal_id)
    assert metal is not None
    metal_geometry = source.geometry.atom_geometry(
        source.constitution.atom_index(AtomRef(metal_id, "ZN"))
    )
    position = source.geometry.position(source.constitution.atom_index(donor))
    source = source.with_updated_residue_facets(
        metal,
        residue_geometry=ResidueGeometry(
            atoms_by_name={"ZN": metal_geometry.with_position(position)}
        ),
        formal_charge_by_atom_name=(),
    )
    evidence = (
        ()
        if route == "template"
        else (
            RetainedNonPolymerChemistryOverride(
                LIGAND_ID, smiles, library.require("QDN").expected_heavy_atom_names()
            ).to_evidence(),
        )
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        library,
        chemistry_evidence=evidence,
        target_residue_ids=frozenset((LIGAND_ID,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure is source and result.issues and not result.repairs


def test_selected_component_reference_is_defensive_against_native_mutation():
    _, library, _ = _selected_donor_case("n1ccccc1")
    semantics = library.require("QDN").hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    first = component_reference_molecule(semantics.component)
    expected = Chem.MolToSmiles(first)
    first.GetAtomWithIdx(0).SetFormalCharge(7)
    second = component_reference_molecule(semantics.component)
    assert Chem.MolToSmiles(second) == expected
    assert second.GetAtomWithIdx(0).GetFormalCharge() == 0


@pytest.mark.parametrize("external", (False, True))
@pytest.mark.parametrize("missing", ("N1", "FE"))
def test_missing_mapped_coordination_endpoint_is_residue_local(external, missing):
    source = _external_source() if external else _source(typed=True, reverse_slots=True)
    missing_id = ResidueId("M", 1) if external and missing == "FE" else LIGAND_ID
    source = source.without_atom_refs((AtomRef(missing_id, missing),))
    evidence = _external_evidence() if external else _evidence()
    residue = source.constitution.residue_or_ligand(LIGAND_ID)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=ComponentLibrary({}), evidence=evidence
    )
    assert not resolution.source.is_resolved()
    assert resolution.failure_reason
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(source, ComponentLibrary({}), chemistry_evidence=(evidence,))
        assert result.structure is source
        assert not result.repairs and result.issues


def test_incomplete_mapped_donor_does_not_abort_a_valid_neighbor():
    source = _source(typed=True).without_atom_refs((AtomRef(LIGAND_ID, "N1"),))
    neighbor_id = ResidueId("L", 2)
    neighbor, geometry, charges = residue_payload(
        component_id="MTH",
        residue_id=neighbor_id,
        is_hetero=True,
        atoms=(atom_payload("C1", "C", Vec3(20, 0, 0)),),
    )
    source = source.with_ligand_facets(
        ligand_sites=(*source.constitution.ligands, neighbor),
        ligand_geometries=(
            source.residue_geometry(source.constitution.residue_index(LIGAND_ID)),
            geometry,
        ),
        ligand_formal_charge_payloads=(
            source.residue_formal_charge_by_atom_name(
                source.constitution.residue_index(LIGAND_ID)
            ),
            charges,
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(_evidence(),)
    )
    assert result.structure.constitution.residue_or_ligand(LIGAND_ID) == (
        source.constitution.residue_or_ligand(LIGAND_ID)
    )
    assert result.structure.residue_geometry(
        result.structure.constitution.residue_index(LIGAND_ID)
    ) == source.residue_geometry(source.constitution.residue_index(LIGAND_ID))
    neighbor = result.structure.constitution.residue_or_ligand(neighbor_id)
    assert neighbor is not None
    assert sum(atom.is_hydrogen() for atom in neighbor.atom_sites) == 4
    assert any(issue.residue_id == LIGAND_ID for issue in result.issues)


@pytest.mark.parametrize("route", ("template", "evidence"))
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("change", ("none", "remove", "replace_isotope"))
def test_allowed_original_donor_h_retains_its_identity(route, isotope, change):
    smiles = "CS(=O)(=O)[NH-]"
    source, library, donor = _selected_donor_case(smiles)
    source = add_retained_non_polymer_hydrogens(
        source, library, target_residue_ids=frozenset((LIGAND_ID,))
    ).structure
    index = source.constitution.residue_index(LIGAND_ID)
    parents = source.topology.residue_hydrogen_anchor_by_name(
        constitution=source.constitution, residue_index=index
    )
    name = next(name for name, parent in parents.items() if parent == donor.atom_name)
    residue = source.constitution.residue_site_at(index)
    source = source.with_updated_residue_facets(
        replace(
            residue,
            atom_sites=tuple(
                AtomSite(atom.name, isotope) if atom.name == name else atom
                for atom in residue.atom_sites
            ),
        ),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
    )
    source = _observed(source)
    evidence = (
        RetainedNonPolymerChemistryOverride(
            LIGAND_ID, smiles, library.require("QDN").expected_heavy_atom_names()
        ).to_evidence()
        if route == "evidence"
        else None
    )
    if change == "remove":
        source = source.without_atom_refs((AtomRef(LIGAND_ID, name),))
    elif change == "replace_isotope":
        residue = source.constitution.residue_site_at(index)
        source = source.with_updated_residue_facets(
            replace(
                residue,
                atom_sites=tuple(
                    AtomSite(atom.name, "D" if isotope == "H" else "H")
                    if atom.name == name
                    else atom
                    for atom in residue.atom_sites
                ),
            ),
            residue_geometry=source.residue_geometry(index),
            formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
        )
    residue = source.constitution.residue_or_ligand(LIGAND_ID)
    assert residue is not None
    active_library = ComponentLibrary({}) if evidence is not None else library
    resolution = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=active_library, evidence=evidence
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        active_library,
        chemistry_evidence=() if evidence is None else (evidence,),
        target_residue_ids=frozenset((LIGAND_ID,)),
    )
    if change == "none":
        assert resolution.source.is_resolved()
        assert not result.issues
        assert result.structure.constitution.resolve_atom_site(
            AtomRef(LIGAND_ID, name)
        ) == source.constitution.resolve_atom_site(AtomRef(LIGAND_ID, name))
    else:
        assert not resolution.source.is_resolved()
        assert result.structure is source
        assert not result.repairs and result.issues
