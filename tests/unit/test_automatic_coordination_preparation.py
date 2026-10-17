"""Context, source custody and partial-charge semantics of automatic preparation."""

from dataclasses import replace
from math import cos, pi, sin, sqrt

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.request_builders import ingress_options

from protrepair.api import process_structure
from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.retained_non_polymer.coordination import (
    coordinated_component_preparation,
)
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import RdkitProjectionError
from protrepair.geometry import Vec3
from protrepair.io import to_rdkit_molecule
from protrepair.scope import WholeStructureScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
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
from protrepair.structure.snapshot import ProteinStructureSnapshot
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
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportMode,
    resolve_local_bond_planning_support,
)
from protrepair.workflow.contracts import LigandPolicy, requested_process_goal

IMD = ResidueId("L", 73)
METAL = ResidueId("Z", -7)
NAMES = ("N1", "C2", "N3", "C4", "C5")
PIM_NAMES = (*NAMES, "C6", "C7", "C8", "C9", "C10", "C11")
PIM_POSITIONS = (
    Vec3(-0.962, 0.475, 1.830),
    Vec3(-0.545, 0.269, 3.099),
    Vec3(0.626, -0.306, 3.072),
    Vec3(1.006, -0.490, 1.803),
    Vec3(0.020, -0.011, 0.998),
    Vec3(0.008, -0.005, -0.481),
    Vec3(1.174, 0.290, -1.189),
    Vec3(1.157, 0.294, -2.568),
    Vec3(-0.011, 0.005, -3.250),
    Vec3(-1.170, -0.288, -2.553),
    Vec3(-1.165, -0.301, -1.174),
)


@pytest.fixture(params=("IMD", "PIM"))
def component_id(request):
    return request.param


def _observed(structure: ProteinStructure) -> ProteinStructure:
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
        ),
    )


def _with_topology(structure, topology):
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _source(
    donor="N1",
    *,
    component_id="IMD",
    relationship=BondRelationshipType.METAL_COORDINATION,
    element="FE",
    reverse=False,
    charges=(),
) -> ProteinStructure:
    names = PIM_NAMES if component_id == "PIM" else NAMES
    positions = (
        PIM_POSITIONS
        if component_id == "PIM"
        else tuple(
            Vec3(1.18 * cos(pi + i * 2 * pi / 5), 1.18 * sin(pi + i * 2 * pi / 5), 0)
            for i in range(5)
        )
    )
    donor_position = positions[names.index(donor)]
    metal_position = Vec3.from_iterable(3.2 * value for value in donor_position)
    charge_map = dict(charges)
    atoms = tuple(
        atom_payload(
            name,
            "N" if name.startswith("N") else "C",
            position,
            occupancy=0.7,
            b_factor=11.0,
            formal_charge=charge_map.get(name),
        )
        for name, position in zip(names, positions, strict=True)
    )
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id=component_id,
                residue_id=IMD,
                is_hetero=True,
                atoms=tuple(reversed(atoms)) if reverse else atoms,
            ),
            residue_payload(
                component_id=element,
                residue_id=METAL,
                is_hetero=True,
                atoms=(atom_payload(element, element, metal_position),),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(IMD, donor)),
        source.constitution.atom_index(AtomRef(METAL, element)),
        order=None,
        relationship_type=relationship,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="independent-ring-site",
            reported_relationship_type=relationship,
            reported_distance_angstrom=2.3,
        ),
    )
    return _observed(
        _with_topology(
            source,
            StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=(bond,),
            ),
        )
    )


def _with_charge(source, ref, charge, *, observed=False):
    index = source.constitution.atom_index(ref)
    atoms = list(source.topology.atom_topologies)
    atoms[index.value] = AtomTopology(charge)
    updated = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=tuple(atoms),
            bonds=source.topology.bonds,
        ),
    )
    return _observed(updated) if observed else updated


def _with_hydrogen(source, parent, isotope="H", *, observed=True):
    index = source.constitution.residue_index(IMD)
    residue = source.constitution.residue_site_at(index)
    geometry = source.residue_geometry(index)
    site = AtomSite("SOURCEH", isotope)
    first, second = ("C2", "C5") if parent == "N1" else ("C2", "C4")
    offset = tuple(
        coordinate - 0.5 * (a + b)
        for coordinate, a, b in zip(
            geometry.position(parent),
            geometry.position(first),
            geometry.position(second),
            strict=True,
        )
    )
    length = sqrt(sum(value * value for value in offset))
    updated = source.with_updated_residue_facets(
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
        updated.constitution.atom_index(AtomRef(IMD, parent)),
        updated.constitution.atom_index(AtomRef(IMD, site.name)),
        order=1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="source-h",
            reported_relationship_type=BondRelationshipType.COVALENT,
            reported_order=1,
        ),
    )
    updated = _with_topology(
        updated,
        StructureTopology(
            constitution=updated.constitution,
            atom_topologies=updated.topology.atom_topologies,
            bonds=(*updated.topology.bonds, bond),
        ),
    )
    return _observed(updated) if observed else updated


@pytest.mark.parametrize("donor", ("N1", "N3"))
@pytest.mark.parametrize("element", ("FE", "ZN", "CU", "CO", "NI", "MN", "CD", "HG"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("component_id,hydrogen_count", (("IMD", 4), ("PIM", 8)))
def test_bound_imidazole_prepares_without_override_or_metal_charge(
    donor, element, reverse, component_id, hydrogen_count
):
    source = _source(donor, element=element, reverse=reverse, component_id=component_id)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((IMD,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert all(issue.severity is IssueSeverity.INFO for issue in result.issues)
    prepared = result.structure
    residue = prepared.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == hydrogen_count
    index = prepared.constitution.residue_index(IMD)
    anchors = prepared.topology.residue_hydrogen_anchor_by_name(
        constitution=prepared.constitution, residue_index=index
    )
    assert donor not in anchors.values()
    other = "N3" if donor == "N1" else "N1"
    assert list(anchors.values()).count(other) == 1
    if component_id == "PIM":
        assert "C5" not in anchors.values()
        assert "C6" not in anchors.values()
        assert all(
            list(anchors.values()).count(name) == 1
            for name in ("C2", "C4", "C7", "C8", "C9", "C10", "C11")
        )
        phenyl = prepared.topology.bond_between(
            prepared.constitution.atom_index(AtomRef(IMD, "C5")),
            prepared.constitution.atom_index(AtomRef(IMD, "C6")),
        )
        assert phenyl is not None
        assert phenyl.order == 1
        assert not phenyl.aromatic
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(METAL, element))
        )
        is None
    )
    for original_index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(original_index)
        current = prepared.constitution.atom_index(ref)
        assert prepared.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(original_index)
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
    assert resolution.has_coordination_topology(prepared)
    assert resolution.has_formal_charge_topology(prepared, residue)


@pytest.mark.parametrize("donor", ("N1", "N3"))
@pytest.mark.parametrize("charged", ("N1", "N3"))
def test_source_imidazolate_charge_is_not_neutralized(donor, charged, component_id):
    source = _source(donor, charges=((charged, -1),), component_id=component_id)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    assert all(issue.severity is IssueSeverity.INFO for issue in result.issues)
    residue = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == (
        3 if component_id == "IMD" else 7
    )
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(IMD, charged))
        )
        == -1
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_source_other_n_hydrogen_identity_coordinates_and_isotope_survive(
    isotope, component_id
):
    source = _with_hydrogen(_source(component_id=component_id), "N3", isotope)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    assert all(issue.severity is IssueSeverity.INFO for issue in result.issues)
    ref = AtomRef(IMD, "SOURCEH")
    first = source.constitution.atom_index(ref)
    second = result.structure.constitution.atom_index(ref)
    assert result.structure.constitution.atom_site_at(
        second
    ) == source.constitution.atom_site_at(first)
    assert result.structure.geometry.atom_geometry(
        second
    ) == source.geometry.atom_geometry(first)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_source_donor_hydrogen_is_atomic_refusal_not_fallback(isotope, component_id):
    source = _with_hydrogen(_source(component_id=component_id), "N1", isotope)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs
    assert all(
        issue.kind is not ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED
        for issue in result.issues
    )


@pytest.mark.parametrize("charge", (-1, 1))
def test_charged_bound_source_hydrogen_is_not_neutral_h_preparation(
    charge, component_id
):
    source = _with_hydrogen(_source(component_id=component_id), "N3")
    source = _with_charge(source, AtomRef(IMD, "SOURCEH"), charge, observed=True)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is source
    assert not result.repairs
    assert result.issues


def test_current_neutralization_cannot_hide_original_charge(component_id):
    source = _source(charges=(("N1", 1),), component_id=component_id)
    hidden = _with_charge(source, AtomRef(IMD, "N1"), 0)
    result = add_retained_non_polymer_hydrogens(
        hidden, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is hidden
    assert not result.repairs
    assert result.issues


@pytest.mark.parametrize("change", ("removed", "renamed", "element"))
@pytest.mark.parametrize("hydrogenated", (False, True))
def test_original_heavy_identity_cannot_be_hidden_by_current_scaffold(
    change, hydrogenated, component_id
):
    current = _source(component_id=component_id)
    index = current.constitution.residue_index(IMD)
    residue = current.constitution.residue_site_at(index)
    geometry = current.residue_geometry(index)
    if change == "removed":
        original = current.with_updated_residue_facets(
            residue.with_atom_site(AtomSite("SOURCEC", "C")),
            residue_geometry=geometry.with_atom_geometry(
                "SOURCEC", AtomGeometry(Vec3(10, 10, 10))
            ),
        )
    elif change == "renamed":
        original = current.without_atom_refs((AtomRef(IMD, "C2"),))
        original = original.with_updated_residue_facets(
            residue.without_atom_sites(("C2",)).with_atom_site(
                AtomSite("SOURCEC", "C")
            ),
            residue_geometry=original.residue_geometry(index).with_atom_geometry(
                "SOURCEC", geometry.atom_geometry("C2")
            ),
        )
    else:
        original = current.with_updated_residue_facets(
            residue.with_atom_site(AtomSite("C2", "O")),
            residue_geometry=geometry,
        )
    original = _observed(original)
    if hydrogenated:
        prepared = add_retained_non_polymer_hydrogens(
            current, target_residue_ids=frozenset((IMD,))
        )
        assert prepared.repairs
        current = prepared.structure
    changed = current.with_provenance(original.provenance)
    residue = changed.constitution.residue_or_ligand(IMD)
    assert residue is not None

    resolution = resolve_retained_non_polymer_chemistry(
        changed,
        residue,
        component_library=build_default_component_library(),
        allow_rdkit_fallback=True,
    )
    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert "original heavy" in resolution.failure_reason

    _, chemistry = StructureChemistryObservation(
        changed, component_library=build_default_component_library()
    ).whole_structure_facts()
    fact = next(
        item for item in chemistry.retained_non_polymer_facts if item.residue_id == IMD
    )
    assert (
        fact.heavy_topology_source
        is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    )
    assert fact.chemistry_failure_reason

    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(
            changed,
            build_default_component_library(),
            target_residue_ids=frozenset((IMD,)),
        )
        assert result.structure is changed
        assert not result.repairs
        assert result.issues
        assert all(
            issue.kind is not ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED
            for issue in result.issues
        )


@pytest.mark.parametrize("missing", ("C2", "C5"))
def test_original_incomplete_scaffold_does_not_forbid_completed_current_atoms(
    missing, component_id
):
    current = _source(component_id=component_id)
    original = _observed(current.without_atom_refs((AtomRef(IMD, missing),)))
    completed = current.with_provenance(original.provenance)
    result = add_retained_non_polymer_hydrogens(
        completed, target_residue_ids=frozenset((IMD,))
    )
    assert result.repairs
    assert all(issue.severity is IssueSeverity.INFO for issue in result.issues)
    residue = result.structure.constitution.residue_or_ligand(IMD)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        result.structure,
        residue,
        component_library=build_default_component_library(),
        allow_rdkit_fallback=False,
    )
    assert (
        resolution.source
        is RetainedNonPolymerChemistryEvidenceSource.COORDINATION_PREPARATION
    )
    assert resolution.has_hydrogen_topology(result.structure, residue)


@pytest.mark.parametrize("order,aromatic", ((2, False), (1, True)))
def test_invalid_source_hydrogen_edge_cannot_supply_preparation_evidence(
    order, aromatic, component_id
):
    source = _with_hydrogen(_source(component_id=component_id), "N3")
    bonds = tuple(
        replace(
            bond,
            order=order,
            aromatic=aromatic,
            source_metadata=replace(bond.source_metadata, reported_order=order),
        )
        if bond.source_metadata is not None
        and bond.source_metadata.source_id == "source-h"
        else bond
        for bond in source.topology.bonds
    )
    source = _observed(
        _with_topology(
            source,
            StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=bonds,
            ),
        )
    )
    with pytest.raises(ValueError):
        coordinated_component_preparation(
            source,
            source.constitution.ligands[0],
            build_default_component_library(),
        )


def test_original_h_attachment_cannot_disappear_behind_current_generated_bond(
    component_id,
):
    source = _with_hydrogen(_source(component_id=component_id), "N3")
    bonds = tuple(
        replace(bond, provenance=BondProvenance.EVIDENCE_RESOLVED, source_metadata=None)
        if bond.source_metadata is not None
        and bond.source_metadata.source_id == "source-h"
        else bond
        for bond in source.topology.bonds
    )
    changed = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
    )
    with pytest.raises(ValueError):
        coordinated_component_preparation(
            changed,
            changed.constitution.ligands[0],
            build_default_component_library(),
        )


@pytest.mark.parametrize("donor", ("N1", "N3"))
@pytest.mark.parametrize("element", ("CA", "MG"))
def test_polymer_partner_admission_does_not_expand_automatic_imidazole(
    donor, element, component_id
):
    source = _source(donor, element=element, component_id=component_id)
    with pytest.raises(ValueError, match=f"unsupported {component_id} coordination"):
        coordinated_component_preparation(
            source, source.constitution.ligands[0], build_default_component_library()
        )
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((IMD,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


def test_untyped_external_connection_does_not_become_coordination(component_id):
    source = _source(
        relationship=BondRelationshipType.UNKNOWN, component_id=component_id
    )
    residue = source.constitution.ligands[0]
    assert (
        coordinated_component_preparation(
            source, residue, build_default_component_library()
        )
        is None
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is source
    assert source.topology.bonds[0].relationship_type is BondRelationshipType.UNKNOWN


def test_missing_original_partner_is_not_replaced_by_fallback(component_id):
    source = _source(component_id=component_id)
    changed = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(),
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        changed, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is changed
    assert result.issues
    assert not result.repairs


@pytest.mark.parametrize("order", (1, 2))
def test_coordination_order_without_matching_source_declaration_is_refused(
    order, component_id
):
    source = _source(component_id=component_id)
    bond = replace(source.topology.bonds[0], order=order, source_metadata=None)
    changed = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(bond,),
        ),
    )
    with pytest.raises(ValueError):
        coordinated_component_preparation(
            changed,
            changed.constitution.ligands[0],
            build_default_component_library(),
        )


def test_multiple_declared_coordination_order_is_not_used_as_site_evidence(
    component_id,
):
    source = _source(component_id=component_id)
    metadata = source.topology.bonds[0].source_metadata
    assert metadata is not None
    bond = replace(
        source.topology.bonds[0],
        source_metadata=replace(metadata, reported_order=2),
    )
    changed = _observed(
        _with_topology(
            source,
            StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=(bond,),
            ),
        )
    )
    with pytest.raises(ValueError):
        coordinated_component_preparation(
            changed,
            changed.constitution.ligands[0],
            build_default_component_library(),
        )


def test_explicit_mapped_imidazolate_has_precedence_over_neutral_preference(
    component_id,
):
    source = _source(component_id=component_id)
    names = PIM_NAMES if component_id == "PIM" else NAMES
    smiles = "n1c[n-]cc1-c2ccccc2" if component_id == "PIM" else "n1c[n-]cc1"
    evidence = RetainedNonPolymerChemistryOverride(IMD, smiles, names).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((IMD,)),
        chemistry_evidence=(evidence,),
    )
    assert not result.issues
    residue = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == (
        3 if component_id == "IMD" else 7
    )
    resolution = resolve_retained_non_polymer_chemistry(
        result.structure,
        residue,
        component_library=build_default_component_library(),
        evidence=evidence,
    )
    assert (
        resolution.source is RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
    )


def test_active_component_template_prevents_automatic_profile_replacement(component_id):
    source = _source(component_id=component_id)
    names = PIM_NAMES if component_id == "PIM" else NAMES
    library = build_default_component_library().with_template(
        ResidueTemplate(ChemicalComponentDefinition(component_id, names))
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
        target_residue_ids=frozenset((IMD,)),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure is source
    assert not result.repairs


def test_public_state_based_preparation_reobserves_actual_h_topology(component_id):
    source = _source(component_id=component_id)
    result = process_structure(
        source,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert not result.has_errors()
    residue = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == (
        4 if component_id == "IMD" else 8
    )
    _, chemistry = StructureChemistryObservation(
        result.structure, component_library=build_default_component_library()
    ).whole_structure_facts()
    fact = next(
        item for item in chemistry.retained_non_polymer_facts if item.residue_id == IMD
    )
    assert not fact.requires_hydrogen_completion()
    assert not fact.requires_topology_resolution()
    assert any(
        issue.kind is ValidationIssueKind.COORDINATED_COMPONENT_PREPARATION
        for issue in result.issues
    )


def test_topology_only_and_repeated_preparation_keep_realized_payload(component_id):
    source = _source(component_id=component_id)
    prepared = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    ).structure
    topology = resolve_retained_non_polymer_topology(
        prepared,
        build_default_component_library(),
        target_residue_ids=frozenset((IMD,)),
    )
    assert topology.structure.constitution == prepared.constitution
    assert topology.structure.geometry == prepared.geometry
    assert topology.structure.topology == prepared.topology
    repeated = add_retained_non_polymer_hydrogens(
        prepared, target_residue_ids=frozenset((IMD,))
    )
    assert repeated.structure.constitution == prepared.constitution
    assert repeated.structure.geometry == prepared.geometry
    assert repeated.structure.topology == prepared.topology


@pytest.mark.parametrize("order", (1, 2))
def test_pim_source_phenyl_attachment_order_is_not_replaced(order):
    source = _source(component_id="PIM")
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(IMD, "C5")),
        source.constitution.atom_index(AtomRef(IMD, "C6")),
        order=order,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="phenyl-source-order",
            reported_order=order,
            reported_relationship_type=BondRelationshipType.COVALENT,
        ),
    )
    source = _observed(
        _with_topology(
            source,
            StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=(*source.topology.bonds, bond),
            ),
        )
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    if order == 2:
        assert result.structure is source
        assert not result.repairs
        assert result.issues
        return

    prepared = result.structure
    current = prepared.topology.bond_between(
        *(
            prepared.constitution.atom_index(source.constitution.atom_ref_at(index))
            for index in bond.endpoint_pair()
        )
    )
    assert current == replace(
        bond,
        atom_index_1=prepared.constitution.atom_index(AtomRef(IMD, "C5")),
        atom_index_2=prepared.constitution.atom_index(AtomRef(IMD, "C6")),
    )
    assert (
        sum(atom.is_hydrogen() for atom in prepared.constitution.ligands[0].atom_sites)
        == 8
    )


@pytest.mark.parametrize("missing", ("C2", "C5", "C6", "C11"))
def test_incomplete_pim_scaffold_cannot_receive_automatic_chemistry(missing):
    source = _source(component_id="PIM")
    changed = source.without_atom_refs((AtomRef(IMD, missing),))
    result = add_retained_non_polymer_hydrogens(
        changed, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is changed
    assert result.issues
    assert not result.repairs


def test_extra_pim_substituent_is_not_accepted_by_component_name():
    source = _source(component_id="PIM")
    index = source.constitution.residue_index(IMD)
    residue = source.constitution.residue_site_at(index)
    changed = source.with_updated_residue_facets(
        residue.with_atom_site(AtomSite("EXTRA", "C")),
        residue_geometry=source.residue_geometry(index).with_atom_geometry(
            "EXTRA", AtomGeometry(Vec3(10, 10, 10))
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        changed, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is changed
    assert result.issues
    assert not result.repairs


def test_pim_two_ring_donors_do_not_become_one_supported_site():
    source = _source(component_id="PIM")
    second = replace(
        source.topology.bonds[0],
        atom_index_1=source.constitution.atom_index(AtomRef(IMD, "N3")),
    )
    source = _observed(
        _with_topology(
            source,
            StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=(*source.topology.bonds, second),
            ),
        )
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


@pytest.mark.parametrize("charge", (0, 1))
def test_pim_original_carbon_charge_is_not_silently_cleared(charge):
    source = _source(component_id="PIM", charges=(("C6", charge),))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    )
    if charge == 1:
        assert result.structure is source
        assert not result.repairs
        assert result.issues
    else:
        prepared = result.structure
        assert (
            prepared.topology.formal_charge(
                prepared.constitution.atom_index(AtomRef(IMD, "C6"))
            )
            == 0
        )
        assert len(prepared.constitution.ligands[0].atom_sites) == 19


def test_pim_native_organic_projection_needs_no_extra_h_or_radical_assignment():
    source = _source(component_id="PIM")
    prepared = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((IMD,))
    ).structure
    with pytest.raises(RdkitProjectionError, match="unknown metal charge"):
        to_rdkit_molecule(prepared)

    organic = prepared.without_atom_refs((AtomRef(METAL, "FE"),))
    molecule = to_rdkit_molecule(organic)
    assert molecule.GetNumAtoms() == 19
    assert molecule.GetNumBonds() == 20
    assert sum(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()) == 8
    assert all(atom.GetNumImplicitHs() == 0 for atom in molecule.GetAtoms())
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
    for index, position in organic.geometry.iter_positions():
        assert tuple(molecule.GetConformer().GetAtomPosition(index.value)) == tuple(
            position
        )


@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("prepared", (False, True))
@pytest.mark.parametrize("donor", ("N1", "N3"))
@pytest.mark.parametrize("reverse", (False, True))
def test_known_bound_imidazole_provides_passive_graph_without_coordinate_fallback(
    component_id, strict, cached, prepared, donor, reverse, monkeypatch
):
    source = _source(donor, component_id=component_id, reverse=reverse)
    if prepared:
        source = add_retained_non_polymer_hydrogens(
            source, target_residue_ids=frozenset((IMD,))
        ).structure

    def fail_fallback(*args, **kwargs):
        raise AssertionError("known coordination chemistry used coordinate fallback")

    monkeypatch.setattr(
        "protrepair.state.retained_non_polymer_chemistry."
        "infer_retained_non_polymer_rdkit_fallback",
        fail_fallback,
    )
    library = build_default_component_library()
    index = source.constitution.residue_index(IMD)
    residue = source.constitution.residue_site_at(index)
    chemistry = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=library, allow_rdkit_fallback=not strict
    )
    assert (
        chemistry.source
        is RetainedNonPolymerChemistryEvidenceSource.COORDINATION_PREPARATION
    )
    support = resolve_local_bond_planning_support(
        ProteinStructureSnapshot.from_structure(source),
        index,
        residue,
        movable_atom_indices=(),
        component_library=library,
        allow_retained_non_polymer_rdkit_fallback=not strict,
        retained_non_polymer_chemistry_resolution_by_residue_id=(
            {IMD: chemistry} if cached else None
        ),
    )
    assert support.mode is LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
    assert set(support.fallback_bond_definitions) == set(
        chemistry.heavy_bond_definitions
    )
    assert all(
        name != "FE"
        for bond in support.fallback_bond_definitions
        for name in (bond.atom_name_1, bond.atom_name_2)
    )
    assert (
        source.topology.formal_charge(
            source.constitution.atom_index(AtomRef(METAL, "FE"))
        )
        is None
    )


@pytest.mark.parametrize("strict", (False, True))
def test_passive_graph_does_not_hide_bound_source_h_contradiction(
    component_id, strict, monkeypatch
):
    source = _with_hydrogen(_source(component_id=component_id), "N1")

    def fail_fallback(*args, **kwargs):
        raise AssertionError("source contradiction used coordinate fallback")

    monkeypatch.setattr(
        "protrepair.state.retained_non_polymer_chemistry."
        "infer_retained_non_polymer_rdkit_fallback",
        fail_fallback,
    )
    index = source.constitution.residue_index(IMD)
    support = resolve_local_bond_planning_support(
        ProteinStructureSnapshot.from_structure(source),
        index,
        source.constitution.residue_site_at(index),
        movable_atom_indices=(),
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=not strict,
    )
    assert support.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    residue = source.constitution.residue_site_at(index)
    chemistry = resolve_retained_non_polymer_chemistry(
        source,
        residue,
        component_library=build_default_component_library(),
        allow_rdkit_fallback=not strict,
    )
    assert chemistry.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert chemistry.failure_reason
    assert chemistry.failure_reason in support.blocker_message


def test_known_coordination_does_not_authorize_movable_template_less_ligand(
    component_id, monkeypatch
):
    source = _source(component_id=component_id)

    def fail_resolution(*args, **kwargs):
        raise AssertionError("movable template-less chemistry consulted resolver")

    monkeypatch.setattr(
        "protrepair.transformer.continuous.support."
        "resolve_retained_non_polymer_chemistry",
        fail_resolution,
    )
    index = source.constitution.residue_index(IMD)
    support = resolve_local_bond_planning_support(
        ProteinStructureSnapshot.from_structure(source),
        index,
        source.constitution.residue_site_at(index),
        movable_atom_indices=(source.constitution.atom_index(AtomRef(IMD, "C2")),),
        component_library=build_default_component_library(),
    )
    assert support.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert not support.fallback_bond_definitions


def test_cached_coordination_resolution_cannot_replace_new_explicit_evidence(
    component_id,
):
    source = _source(component_id=component_id)
    index = source.constitution.residue_index(IMD)
    residue = source.constitution.residue_site_at(index)
    library = build_default_component_library()
    chemistry = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=library
    )
    evidence = RetainedNonPolymerChemistryOverride(
        IMD,
        "n1c[n-]cc1-c2ccccc2" if component_id == "PIM" else "n1c[n-]cc1",
        PIM_NAMES if component_id == "PIM" else NAMES,
    ).to_evidence()
    support = resolve_local_bond_planning_support(
        ProteinStructureSnapshot.from_structure(source),
        index,
        residue,
        movable_atom_indices=(),
        component_library=library,
        retained_non_polymer_chemistry_evidence=(evidence,),
        retained_non_polymer_chemistry_resolution_by_residue_id={IMD: chemistry},
    )
    assert support.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert not support.fallback_bond_definitions
