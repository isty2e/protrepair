"""Source-compatible sulfate chemistry and native force-field admission."""

from dataclasses import replace
from math import isfinite
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.restraint.defaults import build_default_restraint_library
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.diagnostics.chemistry_contradictions import (
    diagnose_retained_non_polymer_template_chemistry_contradictions,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.nonstandard_component_ingress import (
    read_nonstandard_component_registry,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.topology import (
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
from protrepair.transformer.continuous.rdkit import (
    build_rdkit_force_field,
    build_rdkit_molecule,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec

SULFATE_ID = ResidueId("L", 1)
CCD_PATH = Path(__file__).parents[1] / "fixtures/chemistry/retained/SO4.cif"


def _sulfate(
    *, charges=(), missing=(), extra=(), elements=(), order=None, residue_id=SULFATE_ID
):
    record = read_nonstandard_component_registry(CCD_PATH).get("SO4")
    assert record is not None
    component = record.to_idealized_component()
    positions = {
        atom.atom_name: Vec3.coerce(atom.ideal_position)
        for atom in component.heavy_atoms()
        if atom.ideal_position is not None
    }
    assert len(positions) == len(component.heavy_atoms())
    charge_map = dict(charges)
    element_map = dict(elements)
    atoms = (
        tuple(
            atom_payload(
                atom.atom_name,
                element_map.get(atom.atom_name, atom.element),
                positions[atom.atom_name],
                formal_charge=charge_map.get(atom.atom_name),
            )
            for atom in component.heavy_atoms()
            if atom.atom_name not in missing
        )
        + extra
    )
    if order is not None:
        atoms = tuple({atom[0].name: atom for atom in atoms}[name] for name in order)
    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="SO4", residue_id=residue_id, atoms=atoms, is_hetero=True
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _with_bonds(structure, bonds):
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


def _fact(structure):
    observation = StructureChemistryObservation(
        structure, component_library=build_default_component_library()
    )
    return observation.whole_structure_facts()[1].retained_non_polymer_facts[0]


def test_default_sulfate_matches_frozen_ccd_graph_and_charge():
    frozen = read_nonstandard_component_registry(CCD_PATH).get("SO4")
    bundled = build_bundled_retained_non_polymer_registry().get("SO4")
    assert frozen is not None and bundled == frozen
    template = build_default_component_library().get("SO4")
    assert template is not None and template.can_add_hydrogens()
    assert template.expected_hydrogen_atom_names() == ()
    assert sum(template.definition.formal_charges.values()) == -2
    assert sorted(bond.order for bond in template.definition.bonds) == [1, 1, 2, 2]


@pytest.mark.parametrize("reversed_slots", (False, True))
def test_sulfate_graph_and_charge_publish_atomically_without_placement(reversed_slots):
    source = _sulfate(order=("O4", "O3", "O2", "O1", "S") if reversed_slots else None)
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert not result.issues
    assert result.structure.constitution is source.constitution
    assert result.structure.geometry is source.geometry
    assert result.structure.provenance is source.provenance
    assert result.structure.topology.formal_charge_entries()
    assert (
        sum(
            charge or 0
            for _, charge in result.structure.topology.formal_charge_entries()
        )
        == -2
    )
    assert (
        _fact(result.structure).heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    repeated = resolve_retained_non_polymer_topology(
        result.structure, build_default_component_library()
    )
    assert (
        repeated.structure is result.structure
        and not repeated.repairs
        and not repeated.issues
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_zero_h_template_preserves_conflicting_source_hydrogens(isotope):
    source = _sulfate(extra=(atom_payload("HX", isotope, Vec3(2, 0, 1)),))
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(source, build_default_component_library())
        assert result.structure is source
        assert any(
            issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
            for issue in result.issues
        )
    assert (
        _fact(source).hydrogen_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )


@pytest.mark.parametrize("name,charge", (("O3", 0), ("O1", -1), ("S", 6)))
def test_sulfate_explicit_charge_conflict_leaves_entire_residue_unchanged(name, charge):
    source = _sulfate(charges=((name, charge),))
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(source, build_default_component_library())
        assert result.structure is source
        assert any(
            issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
            for issue in result.issues
        )


@pytest.mark.parametrize(
    "kwargs",
    ({"elements": (("O4", "N"),)},),
)
def test_sulfate_wrong_element_scaffold_is_not_published(kwargs):
    source = _sulfate(**kwargs)
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure is source and result.issues
    assert (
        _fact(source).heavy_atom_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )


def test_partial_sulfate_publishes_only_local_charge_without_claiming_readiness():
    source = _sulfate(missing=("O4",))
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert not result.issues
    assert result.structure.constitution is source.constitution
    assert result.structure.geometry is source.geometry
    assert dict(
        result.structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
    ) == {"O3": -1}
    assert len(result.structure.topology.bonds) == 3
    assert (
        _fact(result.structure).heavy_atom_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )
    repeated = resolve_retained_non_polymer_topology(
        result.structure, build_default_component_library()
    )
    assert (
        repeated.structure is result.structure
        and not repeated.repairs
        and not repeated.issues
    )


def test_complete_bonds_without_ionic_charges_are_not_ready():
    source = _sulfate()
    template = build_default_component_library().get("SO4")
    assert template is not None
    bonds = tuple(
        TopologyBond(
            source.constitution.atom_index(AtomRef(SULFATE_ID, b.atom_name_1)),
            source.constitution.atom_index(AtomRef(SULFATE_ID, b.atom_name_2)),
            order=b.order,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        for b in template.definition.bonds
    )
    source = _with_bonds(source, bonds)
    fact = _fact(source)
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.ABSENT
    )
    assert fact.requires_topology_resolution()
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure.topology.bonds == source.topology.bonds
    assert result.repairs
    assert (
        _fact(result.structure).heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )


def test_sulfate_keeps_connectivity_only_source_metadata():
    source = _sulfate(charges=(("O3", -1), ("O4", -1), ("S", 0)))
    bond = TopologyBond(
        AtomIndex(0),
        AtomIndex(1),
        order=None,
        relationship_type=BondRelationshipType.UNKNOWN,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_CONECT,
            source_id="source-edge",
        ),
    )
    source = _with_bonds(source, (bond,))
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    resolved = result.structure.topology.bond_between(*bond.endpoint_pair())
    assert resolved == replace(
        bond, order=2, relationship_type=BondRelationshipType.COVALENT
    )
    assert result.structure.topology.atom_topologies == source.topology.atom_topologies


@pytest.mark.parametrize(
    "force_field",
    (ContinuousRelaxationForceField.UFF, ContinuousRelaxationForceField.MMFF),
)
def test_sulfate_native_graph_is_closed_shell_and_has_finite_force_field(force_field):
    source = _sulfate()
    prepared = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    ).structure
    scope = LocalScopeSpec.from_atoms(
        tuple(
            prepared.constitution.atom_ref_at(index)
            for index in prepared.constitution.atom_indices_for_residue(SULFATE_ID)
        )
    )
    problem = build_continuous_relaxation_problem(
        prepared,
        scope,
        component_library=build_default_component_library(),
        context_radius_angstrom=0,
        force_field=force_field,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    assert sum(a.GetFormalCharge() for a in molecule.GetAtoms()) == -2
    assert all(a.GetNumRadicalElectrons() == 0 for a in molecule.GetAtoms())
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    ff = build_rdkit_force_field(
        molecule,
        problem=problem,
        restraint_library=build_default_restraint_library(),
        rdkit_atom_index_by_structure_atom_index=mapping,
    )
    assert isfinite(ff.CalcEnergy()) and all(map(isfinite, ff.CalcGrad()))


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_sulfate_charge_survives_egress_and_reingress(file_format):
    prepared = resolve_retained_non_polymer_topology(
        _sulfate(), build_default_component_library()
    ).structure
    text = write_structure_string(prepared, file_format)
    restored = read_structure_string(text, file_format)
    assert (
        sum(charge or 0 for _, charge in restored.topology.formal_charge_entries())
        == -2
    )
    assert (
        restored.constitution.ligands[0].atom_sites
        == prepared.constitution.ligands[0].atom_sites
    )
    if file_format is FileFormat.PDB:
        molecule = Chem.MolFromPDBBlock(
            text, sanitize=True, removeHs=False, proximityBonding=True
        )
        assert molecule is not None
        assert sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) == -2
        assert all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())


def test_reported_sulfate_bond_order_conflict_does_not_publish_charges():
    bond = TopologyBond(
        AtomIndex(0),
        AtomIndex(1),
        order=1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN, reported_order=1
        ),
    )
    source = _with_bonds(_sulfate(), (bond,))
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure is source and result.issues


@pytest.mark.parametrize(
    "relationship,admitted",
    (
        (BondRelationshipType.COVALENT, False),
        (BondRelationshipType.DISULFIDE, False),
        (BondRelationshipType.UNKNOWN, False),
        (BondRelationshipType.METAL_COORDINATION, True),
        (BondRelationshipType.HYDROGEN_BOND, True),
    ),
)
@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize("provenance", tuple(BondProvenance))
@pytest.mark.parametrize("topology_only", (False, True))
def test_sulfate_external_relationships_are_not_reinterpreted(
    relationship, admitted, partial, provenance, topology_only
):
    sulfate = _sulfate(missing=("O1",) if partial else ())
    source = build_structure(
        chains=(),
        ligands=(
            (
                sulfate.constitution.ligands[0],
                sulfate.residue_geometry(ResidueIndex(0)),
                (),
            ),
            residue_payload(
                component_id="ZN",
                residue_id=ResidueId("Z", 2),
                is_hetero=True,
                atoms=(atom_payload("ZN", "ZN", Vec3(3, 0, 0), formal_charge=2),),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(SULFATE_ID, "O3")),
        source.constitution.atom_index(AtomRef(ResidueId("Z", 2), "ZN")),
        order=None,
        relationship_type=relationship,
        provenance=provenance,
        source_metadata=(
            SourceBondMetadata(
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                reported_relationship_type=relationship,
            )
            if provenance is BondProvenance.SOURCE_EXPLICIT
            else None
        ),
    )
    source = _with_bonds(source, (bond,))
    execute = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = execute(
        source,
        build_default_component_library(),
        target_residue_ids=frozenset((SULFATE_ID,)),
    )
    assert result.structure.topology.bond_between(*bond.endpoint_pair()) == bond
    if admitted:
        assert result.structure.geometry is source.geometry and not result.issues
        assert _fact(result.structure).heavy_atom_topology_availability_state is (
            TopologyAvailabilityState.ABSENT
            if partial
            else TopologyAvailabilityState.PRESENT
        )
    else:
        assert result.structure is source and result.issues
        assert f"external {relationship.value} relationship" in result.issues[0].message
        assert (
            _fact(source).heavy_atom_topology_availability_state
            is TopologyAvailabilityState.ABSENT
        )


def test_bisulfate_override_takes_precedence_over_zero_h_template():
    source = _sulfate()
    evidence = RetainedNonPolymerChemistryEvidence(
        SULFATE_ID, "S(=O)(=O)([O-])O", ("S", "O1", "O2", "O3", "O4")
    )
    result = add_retained_non_polymer_hydrogens(
        source, build_default_component_library(), chemistry_evidence=(evidence,)
    )
    assert not result.issues
    assert len(result.structure.constitution.atom_slots) == 6
    assert (
        sum(
            charge or 0
            for _, charge in result.structure.topology.formal_charge_entries()
        )
        == -1
    )
    observation = StructureChemistryObservation(
        result.structure, retained_non_polymer_chemistry_evidence=(evidence,)
    )
    fact = observation.whole_structure_facts()[1].retained_non_polymer_facts[0]
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )


def test_strict_fallback_still_admits_known_sulfate():
    result = resolve_retained_non_polymer_topology(
        _sulfate(),
        build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert not result.issues
    assert (
        _fact(result.structure).heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )


def test_sulfate_failure_is_residue_local_in_a_mixed_batch():
    bad = _sulfate(charges=(("O3", 0),))
    good_id = ResidueId("L", 2)
    good = _sulfate(residue_id=good_id)
    source = build_structure(
        chains=(),
        ligands=tuple(
            (
                structure.constitution.ligands[0],
                structure.residue_geometry(ResidueIndex(0)),
                structure.residue_formal_charge_by_atom_name(ResidueIndex(0)),
            )
            for structure in (bad, good)
        ),
        source_format=FileFormat.PDB,
    )
    result = add_retained_non_polymer_hydrogens(
        source, build_default_component_library()
    )
    assert result.structure.geometry is source.geometry
    assert result.structure.constitution is source.constitution
    assert all(issue.scope.residue_ids == (SULFATE_ID,) for issue in result.issues)
    assert result.issues
    assert all(
        result.structure.topology.formal_charge(index)
        == source.topology.formal_charge(index)
        for index in source.constitution.atom_indices_for_residue(SULFATE_ID)
    )
    assert (
        sum(
            result.structure.topology.formal_charge(index) or 0
            for index in source.constitution.atom_indices_for_residue(good_id)
        )
        == -2
    )


def test_sulfate_extra_heavy_atom_does_not_gain_a_partial_template_graph():
    source = _sulfate(extra=(atom_payload("CX", "C", Vec3(4, 0, 0)),))
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure is source and result.issues


def test_sulfate_source_h_is_not_hidden_by_already_realized_heavy_charges():
    source = _sulfate(
        charges=(("O3", -1), ("O4", -1)),
        extra=(atom_payload("HX", "H", Vec3(2, 0, 1)),),
    )
    fact = _fact(source)
    assert fact.hydrogen_topology_availability_state is TopologyAvailabilityState.ABSENT
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure is source and result.issues


def test_disconnected_ionic_override_can_publish_charges_without_bonds():
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=SULFATE_ID,
                is_hetero=True,
                atoms=(
                    atom_payload("NA", "NA", Vec3(0, 0, 0)),
                    atom_payload("CL", "CL", Vec3(4, 0, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    evidence = RetainedNonPolymerChemistryEvidence(
        SULFATE_ID, "[Na+].[Cl-]", ("NA", "CL")
    )
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library(), chemistry_evidence=(evidence,)
    )
    assert not result.issues and not result.structure.topology.bonds
    assert result.structure.topology.formal_charge(AtomIndex(0)) == 1
    assert result.structure.topology.formal_charge(AtomIndex(1)) == -1
    assert result.structure.geometry is source.geometry


def test_charged_override_conflict_preserves_source_charge_and_atoms():
    source = _sulfate(charges=(("O3", 0),))
    evidence = RetainedNonPolymerChemistryEvidence(
        SULFATE_ID, "S(=O)(=O)([O-])O", ("S", "O1", "O2", "O3", "O4")
    )
    result = add_retained_non_polymer_hydrogens(
        source, build_default_component_library(), chemistry_evidence=(evidence,)
    )
    assert result.structure is source and result.issues


@pytest.mark.parametrize("reported_zero", (False, True))
def test_sulfate_diagnostics_distinguish_unspecified_charge_from_declared_zero(
    reported_zero: bool,
) -> None:
    source: ProteinStructure = _sulfate(charges=(("O3", 0),) if reported_zero else ())
    template = build_default_component_library().require("SO4")
    issues = diagnose_retained_non_polymer_template_chemistry_contradictions(
        source.constitution.ligands[0],
        residue_geometry=source.residue_geometry(ResidueIndex(0)),
        source_formal_charge_by_atom_name=dict(
            source.residue_formal_charge_by_atom_name(ResidueIndex(0))
        ),
        template=template,
    )
    assert bool(issues) is reported_zero
