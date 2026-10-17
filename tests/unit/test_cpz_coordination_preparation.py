"""Named CPZ chemistry, source custody and atomic preparation conflicts."""

from dataclasses import replace
from math import sqrt

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.retained_non_polymer.coordination import (
    coordinated_component_preparation,
)
from protrepair.diagnostics.kinds import IssueSeverity
from protrepair.geometry import Vec3
from protrepair.io import (
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.scope import WholeStructureScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
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

CPZ = ResidueId("L", 73)
METAL = ResidueId("Z", -7)
# CCD CPZ ideal coordinates and named heavy graph, independent of SMILES traversal.
HEAVY = (
    ("C11", "C", (0.807, -0.463, 0.561)),
    ("C10", "C", (0.764, -0.484, 1.955)),
    ("C9", "C", (1.800, -1.079, 2.676)),
    ("C8", "C", (2.878, -1.652, 2.002)),
    ("C7", "C", (2.921, -1.631, 0.608)),
    ("C6", "C", (1.885, -1.036, -0.112)),
    ("C4", "C", (1.928, -1.014, -1.554)),
    ("C2", "C", (1.425, -0.120, -2.466)),
    ("N1", "N", (1.711, -0.512, -3.750)),
    ("C5", "C", (2.383, -1.634, -3.616)),
    ("N3", "N", (2.539, -1.978, -2.302)),
    ("CL", "CL", (1.748, -1.103, 4.396)),
)
EDGES = frozenset(
    frozenset(pair)
    for pair in (
        ("C11", "C10"),
        ("C11", "C6"),
        ("C10", "C9"),
        ("C9", "C8"),
        ("C9", "CL"),
        ("C8", "C7"),
        ("C7", "C6"),
        ("C6", "C4"),
        ("C4", "C2"),
        ("C4", "N3"),
        ("C2", "N1"),
        ("N1", "C5"),
        ("C5", "N3"),
    )
)


def _observed(structure):
    return structure.with_provenance(
        replace(
            structure.provenance,
            ingress=replace(
                structure.provenance.ingress,
                observation=StructureObservation.from_source_facets(
                    constitution=structure.constitution,
                    geometry=structure.geometry,
                    topology=structure.topology,
                ),
            ),
        )
    )


def _with_bonds(source, bonds):
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )


def _source(
    donor="N1",
    *,
    element="FE",
    charges=(),
    reverse=False,
    relationship=BondRelationshipType.METAL_COORDINATION,
    metal_charge=None,
):
    charges = dict(charges)
    atoms = tuple(
        atom_payload(
            name,
            symbol,
            Vec3(*xyz),
            occupancy=0.7,
            b_factor=11.0,
            formal_charge=charges.get(name),
        )
        for name, symbol, xyz in HEAVY
    )
    position = next(Vec3(*xyz) for name, _, xyz in HEAVY if name == donor)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="CPZ",
                residue_id=CPZ,
                is_hetero=True,
                atoms=tuple(reversed(atoms)) if reverse else atoms,
            ),
            residue_payload(
                component_id=element,
                residue_id=METAL,
                is_hetero=True,
                atoms=(
                    atom_payload(
                        element,
                        element,
                        position.with_offset(-1.5, 0, -1.7),
                        formal_charge=metal_charge,
                    ),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(CPZ, donor)),
        source.constitution.atom_index(AtomRef(METAL, element)),
        order=None,
        relationship_type=relationship,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="named-cpz-site",
            reported_relationship_type=relationship,
            reported_distance_angstrom=2.3,
        ),
    )
    return _observed(_with_bonds(source, (bond,)))


def _with_hydrogen(source, parent, isotope):
    index = source.constitution.residue_index(CPZ)
    residue = source.constitution.residue_site_at(index)
    geometry = source.residue_geometry(index)
    first, second = ("C2", "C5") if parent == "N1" else ("C4", "C5")
    offset = tuple(
        p - 0.5 * (a + b)
        for p, a, b in zip(
            geometry.position(parent),
            geometry.position(first),
            geometry.position(second),
            strict=True,
        )
    )
    length = sqrt(sum(value * value for value in offset))
    site = AtomSite("SOURCEH", isotope)
    changed = source.with_updated_residue_facets(
        residue.with_atom_site(site),
        residue_geometry=geometry.with_atom_geometry(
            site.name,
            AtomGeometry(
                geometry.position(parent).with_offset(
                    *(1.02 * value / length for value in offset)
                ),
                occupancy=0.4,
                b_factor=8.0,
            ),
        ),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
    )
    bond = TopologyBond(
        changed.constitution.atom_index(AtomRef(CPZ, parent)),
        changed.constitution.atom_index(AtomRef(CPZ, "SOURCEH")),
        order=1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="cpz-source-h",
            reported_relationship_type=BondRelationshipType.COVALENT,
            reported_order=1,
        ),
    )
    return _observed(_with_bonds(changed, (*changed.topology.bonds, bond)))


def _require_custody(source, prepared):
    assert prepared.provenance.ingress is source.provenance.ingress
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        current = prepared.constitution.atom_index(ref)
        assert prepared.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(index)
        assert prepared.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(index)
        charge = source.topology.formal_charge(index)
        assert charge is None or prepared.topology.formal_charge(current) == charge
    for bond in source.topology.bonds:
        refs = tuple(
            source.constitution.atom_ref_at(index) for index in bond.endpoint_pair()
        )
        current = prepared.topology.bond_between(
            *(prepared.constitution.atom_index(ref) for ref in refs)
        )
        assert current is not None
        assert current.source_metadata == bond.source_metadata
        assert current.relationship_type == bond.relationship_type
        assert current.provenance == bond.provenance
        assert current.order == bond.order
        assert current.aromatic == bond.aromatic


@pytest.mark.parametrize("donor", ("N1", "N3"))
@pytest.mark.parametrize("element", ("FE", "ZN", "CU", "CO", "NI", "MN", "CD", "HG"))
@pytest.mark.parametrize("reverse", (False, True))
def test_neutral_cpz_prepares_exact_named_graph(donor, element, reverse):
    source = _source(donor, element=element, reverse=reverse)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((CPZ,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert all(issue.severity is IssueSeverity.INFO for issue in result.issues)
    prepared = result.structure
    residue = prepared.constitution.residue_or_ligand(CPZ)
    assert residue is not None
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 7
    index = prepared.constitution.residue_index(CPZ)
    anchors = prepared.topology.residue_hydrogen_anchor_by_name(
        constitution=prepared.constitution, residue_index=index
    )
    other = "N3" if donor == "N1" else "N1"
    assert list(anchors.values()).count(other) == 1
    assert donor not in anchors.values()
    assert not {"C4", "C6", "C9", "CL"}.intersection(anchors.values())
    assert all(
        list(anchors.values()).count(name) == 1
        for name in ("C2", "C5", "C7", "C8", "C10", "C11")
    )
    heavy_names = {name for name, _, _ in HEAVY}
    edges = {
        frozenset(
            prepared.constitution.atom_ref_at(atom).atom_name
            for atom in bond.endpoint_pair()
        )
        for bond in prepared.topology.bonds
        if all(
            prepared.constitution.atom_ref_at(atom).residue_id == CPZ
            and prepared.constitution.atom_ref_at(atom).atom_name in heavy_names
            for atom in bond.endpoint_pair()
        )
    }
    assert edges == EDGES
    for first, second in (("C4", "C6"), ("C9", "CL")):
        bond = prepared.topology.bond_between(
            prepared.constitution.atom_index(AtomRef(CPZ, first)),
            prepared.constitution.atom_index(AtomRef(CPZ, second)),
        )
        assert bond is not None and bond.order == 1 and not bond.aromatic
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(METAL, element))
        )
        is None
    )
    resolution = resolve_retained_non_polymer_chemistry(
        prepared,
        residue,
        component_library=build_default_component_library(),
        allow_rdkit_fallback=False,
    )
    assert (
        resolution.source
        is RetainedNonPolymerChemistryEvidenceSource.COORDINATION_PREPARATION
    )
    assert resolution.has_hydrogen_topology(prepared, residue)
    assert resolution.has_formal_charge_topology(prepared, residue)
    assert resolution.has_coordination_topology(prepared)
    _require_custody(source, prepared)


@pytest.mark.parametrize("donor", ("N1", "N3"))
@pytest.mark.parametrize("charged", ("N1", "N3"))
def test_source_cpz_anion_is_not_neutralized(donor, charged):
    source = _source(donor, charges=((charged, -1),))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert all(issue.severity is IssueSeverity.INFO for issue in result.issues)
    prepared = result.structure
    residue = prepared.constitution.residue_or_ligand(CPZ)
    assert residue is not None
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 6
    _require_custody(source, prepared)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("donor", ("N1", "N3"))
def test_other_n_source_isotope_survives(donor, isotope):
    source = _with_hydrogen(_source(donor), "N3" if donor == "N1" else "N1", isotope)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert all(issue.severity is IssueSeverity.INFO for issue in result.issues)
    _require_custody(source, result.structure)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("donor", ("N1", "N3"))
def test_coordinating_n_source_h_is_atomic_refusal(donor, isotope):
    source = _with_hydrogen(_source(donor), donor, isotope)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is source
    assert not result.repairs
    assert result.issues


@pytest.mark.parametrize("atom,charge", (("N1", 1), ("CL", -1), ("C9", 1)))
def test_contradictory_source_charge_refuses_without_fallback(atom, charge):
    source = _source(charges=((atom, charge),))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is source and not result.repairs and result.issues


@pytest.mark.parametrize("missing", ("CL", "C6", "N3"))
def test_missing_cpz_heavy_atom_is_not_built_as_pim(missing):
    source = _source().without_atom_refs((AtomRef(CPZ, missing),))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is source and not result.repairs and result.issues


@pytest.mark.parametrize("atom", ("CL", "C4"))
def test_non_nitrogen_coordination_is_not_admitted(atom):
    source = _source(atom)
    with pytest.raises(ValueError, match="unsupported CPZ coordination"):
        coordinated_component_preparation(
            source, source.constitution.ligands[0], build_default_component_library()
        )


@pytest.mark.parametrize("element", ("CA", "MG"))
def test_unsupported_metal_remains_unsupported(element):
    source = _source(element=element)
    with pytest.raises(ValueError, match="unsupported CPZ coordination"):
        coordinated_component_preparation(
            source, source.constitution.ligands[0], build_default_component_library()
        )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_public_cpz_h_goal_and_actual_native_roundtrip(file_format):
    source = _source(metal_charge=2)
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        transform_requests=WorkflowTransformRequests(
            allow_retained_non_polymer_rdkit_fallback=False
        ),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert not result.has_errors()
    assert result.requested_goals_fully_satisfied()
    _require_custody(source, result.structure)
    native = to_rdkit_molecule(result.structure)
    assert native is not None
    assert not any(
        atom.GetNumRadicalElectrons() or atom.GetNumImplicitHs()
        for atom in native.GetAtoms()
    )
    text = write_structure_string(result.structure, file_format)
    reread = read_structure_string(text, file_format)
    residue = reread.constitution.residue_or_ligand(CPZ)
    assert residue is not None
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 7
    heavy_names = {name for name, _, _ in HEAVY}
    assert {
        frozenset(
            reread.constitution.atom_ref_at(atom).atom_name
            for atom in bond.endpoint_pair()
        )
        for bond in reread.topology.bonds
        if all(
            reread.constitution.atom_ref_at(atom).residue_id == CPZ
            and reread.constitution.atom_ref_at(atom).atom_name in heavy_names
            for atom in bond.endpoint_pair()
        )
    } == EDGES
    index = reread.constitution.residue_index(CPZ)
    anchors = reread.topology.residue_hydrogen_anchor_by_name(
        constitution=reread.constitution, residue_index=index
    )
    assert sorted(anchors.values()) == sorted(
        ("C2", "C5", "C7", "C8", "C10", "C11", "N3")
    )
    metal = reread.constitution.atom_index(AtomRef(METAL, "FE"))
    assert reread.topology.formal_charge(metal) == 2
    coordination = reread.topology.bond_between(
        metal, reread.constitution.atom_index(AtomRef(CPZ, "N1"))
    )
    assert coordination is not None
    assert coordination.relationship_type is BondRelationshipType.METAL_COORDINATION
    for name, _, _ in HEAVY:
        assert reread.geometry.position(
            reread.constitution.atom_index(AtomRef(CPZ, name))
        ) == result.structure.geometry.position(
            result.structure.constitution.atom_index(AtomRef(CPZ, name))
        )
    actual = Chem.MolFromPDBBlock(
        write_structure_string(reread, FileFormat.PDB),
        sanitize=True,
        removeHs=False,
        proximityBonding=True,
    )
    assert actual is not None
    assert actual.GetNumAtoms() == native.GetNumAtoms()


@pytest.mark.parametrize("name,element", (("CL", "C"), ("N3", "C"), ("C6", "N")))
def test_wrong_named_cpz_element_is_atomic_refusal(name, element):
    source = _source()
    index = source.constitution.residue_index(CPZ)
    residue = source.constitution.residue_site_at(index)
    changed = source.with_updated_residue_facets(
        residue.with_atom_site(AtomSite(name, element)),
        residue_geometry=source.residue_geometry(index),
    )
    result = add_retained_non_polymer_hydrogens(
        changed, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is changed and not result.repairs and result.issues


@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.UNKNOWN, BondRelationshipType.COVALENT)
)
def test_untyped_or_covalent_cpz_link_is_not_a_coordination_profile(relationship):
    source = _source(relationship=relationship)
    assert (
        coordinated_component_preparation(
            source, source.constitution.ligands[0], build_default_component_library()
        )
        is None
    )


def test_cpz_topology_only_and_repeated_preparation_preserve_payload():
    source = _source()
    prepared = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    ).structure
    for result in (
        resolve_retained_non_polymer_topology(
            prepared,
            build_default_component_library(),
            target_residue_ids=frozenset((CPZ,)),
        ),
        add_retained_non_polymer_hydrogens(
            prepared, target_residue_ids=frozenset((CPZ,))
        ),
    ):
        assert result.structure.constitution == prepared.constitution
        assert result.structure.geometry == prepared.geometry
        assert result.structure.topology == prepared.topology
        _require_custody(source, result.structure)


def test_current_charge_cannot_hide_original_cpz_charge():
    source = _source(charges=(("N1", 1),))
    atoms = list(source.topology.atom_topologies)
    atoms[source.constitution.atom_index(AtomRef(CPZ, "N1")).value] = AtomTopology(0)
    hidden = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=tuple(atoms),
            bonds=source.topology.bonds,
        ),
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    result = add_retained_non_polymer_hydrogens(
        hidden, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is hidden and not result.repairs and result.issues


def test_original_source_h_cannot_be_removed_to_enable_cpz():
    source = _with_hydrogen(_source(), "N1", "T")
    hidden = source.without_atom_refs((AtomRef(CPZ, "SOURCEH"),))
    result = add_retained_non_polymer_hydrogens(
        hidden, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is hidden and not result.repairs and result.issues


@pytest.mark.parametrize("names", (("C4", "C6"), ("C9", "CL")))
@pytest.mark.parametrize("order", (1, 2))
def test_cpz_source_substitution_orders_are_not_overwritten(names, order):
    source = _source()
    bond = TopologyBond(
        *(source.constitution.atom_index(AtomRef(CPZ, name)) for name in names),
        order=order,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="explicit-cpz-substitution",
            reported_relationship_type=BondRelationshipType.COVALENT,
            reported_order=order,
        ),
    )
    source = _observed(_with_bonds(source, (*source.topology.bonds, bond)))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    if order == 2:
        assert result.structure is source and not result.repairs and result.issues
    else:
        assert result.repairs
        _require_custody(source, result.structure)


def test_extra_cpz_substituent_is_not_accepted_by_component_name():
    source = _source()
    index = source.constitution.residue_index(CPZ)
    residue = source.constitution.residue_site_at(index)
    changed = source.with_updated_residue_facets(
        residue.with_atom_site(AtomSite("EXTRA", "C")),
        residue_geometry=source.residue_geometry(index).with_atom_geometry(
            "EXTRA", AtomGeometry(Vec3(10, 10, 10))
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        changed, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is changed and not result.repairs and result.issues


def test_active_cpz_template_owns_its_preparation():
    source = _source()
    names = tuple(name for name, _, _ in HEAVY)
    library = build_default_component_library().with_template(
        ResidueTemplate(ChemicalComponentDefinition("CPZ", names))
    )
    assert (
        coordinated_component_preparation(
            source, source.constitution.ligands[0], library
        )
        is None
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=library,
        target_residue_ids=frozenset((CPZ,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure is source and not result.repairs


def test_unbound_cpz_does_not_use_coordination_preference():
    source = _observed(_with_bonds(_source(), ()))
    assert (
        coordinated_component_preparation(
            source, source.constitution.ligands[0], build_default_component_library()
        )
        is None
    )


def test_multiple_ring_donors_refuse_a_single_cpz_profile():
    source = _source()
    bond = replace(
        source.topology.bonds[0],
        atom_index_1=source.constitution.atom_index(AtomRef(CPZ, "N3")),
    )
    source = _observed(_with_bonds(source, (*source.topology.bonds, bond)))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is source and not result.repairs and result.issues


def test_external_covalence_is_not_treated_as_a_free_cpz_n_h():
    source = _source()
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(CPZ, "N3")),
        source.constitution.atom_index(AtomRef(METAL, "FE")),
        order=1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="cpz-external-covalence",
            reported_relationship_type=BondRelationshipType.COVALENT,
            reported_order=1,
        ),
    )
    source = _observed(_with_bonds(source, (*source.topology.bonds, bond)))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is source and not result.repairs and result.issues


def test_changed_original_source_metadata_refuses_cpz():
    source = _with_hydrogen(_source(), "N3", "D")
    changed = _with_bonds(
        source,
        tuple(
            replace(
                bond, provenance=BondProvenance.EVIDENCE_RESOLVED, source_metadata=None
            )
            if bond.source_metadata is not None
            and bond.source_metadata.source_id == "cpz-source-h"
            else bond
            for bond in source.topology.bonds
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        changed, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is changed and not result.repairs and result.issues


def test_source_anion_and_non_donor_t_h_are_not_jointly_normalized():
    source = _with_hydrogen(_source(charges=(("N3", -1),)), "N3", "T")
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,))
    )
    assert result.structure is source and not result.repairs and result.issues


def test_explicit_cpz_anion_overrides_neutral_preference():
    source = _source()
    names = ("N1", "C5", "N3", "C4", "C6", "C7", "C8", "C9", "CL", "C10", "C11", "C2")
    evidence = RetainedNonPolymerChemistryOverride(
        CPZ, "n1c[n-]c(-c2ccc(Cl)cc2)c1", names
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((CPZ,)), chemistry_evidence=(evidence,)
    )
    assert not result.issues
    residue = result.structure.constitution.residue_or_ligand(CPZ)
    assert residue is not None
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 6
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(CPZ, "N3"))
        )
        == -1
    )
    _require_custody(source, result.structure)
