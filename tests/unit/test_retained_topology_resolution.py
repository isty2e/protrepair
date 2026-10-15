"""Retained chemistry publication preserves source evidence, not unknown fields."""

from dataclasses import replace
from math import isfinite

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem
from tests.support.request_builders import (
    ingress_options,
    whole_structure_requested_goals,
)
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)

from protrepair.api import process_structure
from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.restraint.defaults import build_default_restraint_library
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
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
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.rdkit import (
    build_rdkit_force_field,
    build_rdkit_molecule,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.repair_stage import (
    apply_repair_stage_local_refinement,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import LigandPolicy

LIGAND_ID = ResidueId("L", 1)


def test_local_prerequisites_finish_all_five_preparation_families():
    polymer_id = ResidueId("A", 1)
    complete_id = ResidueId("L", 2)
    named = _named_methanol()
    complete = named.constitution.ligands[0]
    source = build_bonded_structure(
        chains=(
            chain_payload(
                "A",
                residues=(
                    residue_payload(
                        component_id="ALA",
                        residue_id=polymer_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0, 0, 0)),
                            atom_payload("CA", "C", Vec3(1.4, 0, 0)),
                            atom_payload("C", "C", Vec3(2, 1.3, 0)),
                            atom_payload("O", "O", Vec3(1.5, 2.3, 0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(10, 0, 0)),
                    atom_payload("O1", "O", Vec3(11.4, 0, 0)),
                ),
            ),
            residue_payload(
                component_id="UNK",
                residue_id=complete_id,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        atom.name,
                        atom.element,
                        Vec3(
                            named.geometry.position(AtomIndex(index)).x + 20,
                            named.geometry.position(AtomIndex(index)).y,
                            named.geometry.position(AtomIndex(index)).z,
                        ),
                    )
                    for index, atom in enumerate(complete.atom_sites)
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = _with_bonds(
        source,
        (
            *source.topology.bonds,
            *(
                _source_bond(first + 6, second + 6)
                for first, second in ((0, 1), (1, 2), (0, 3), (0, 4), (0, 5))
            ),
        ),
    )
    library = build_default_component_library().with_template(
        ResidueTemplate(
            ChemicalComponentDefinition(
                "UNK",
                atom_names=("C1", "O1"),
                bonds=(BondDefinition("C1", "O1"),),
            )
        )
    )
    directive = RepairLocalRefinementDirective.from_residue_ids(
        (polymer_id, LIGAND_ID, complete_id),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    prepared = apply_repair_stage_local_refinement(
        TransformationResult(structure=source, repairs=(), issues=()),
        local_refinement=directive,
        component_library=library,
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    observation = StructureChemistryObservation(
        prepared.structure, component_library=library
    )
    polymer = prepared.structure.constitution.residue_or_ligand(polymer_id)
    assert polymer is not None
    assert polymer.has_atom_site("CB")
    assert polymer.has_atom_site("OXT")
    assert any(atom.is_hydrogen() for atom in polymer.atom_sites)
    facts = observation.whole_structure_facts()[1].retained_non_polymer_facts
    assert len(facts) == 2
    assert all(not fact.requires_hydrogen_completion() for fact in facts)
    assert all(not fact.requires_topology_resolution() for fact in facts)
    rejections = tuple(
        issue.message
        for issue in prepared.issues
        if issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
    )
    assert not rejections, rejections
    assert any(
        repair.kind is RepairEventKind.LOCAL_REFINEMENT_APPLIED
        for repair in prepared.repairs
    )


def _methanol() -> ProteinStructure:
    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.4, 0, 0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
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


def _source_bond(
    first: int = 0,
    second: int = 1,
    *,
    order: int | None = None,
    relationship: BondRelationshipType = BondRelationshipType.UNKNOWN,
) -> TopologyBond:
    return TopologyBond(
        AtomIndex(first),
        AtomIndex(second),
        order=order,
        relationship_type=relationship,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_CONECT,
            source_id="source-edge",
            reported_order=order,
            reported_relationship_type=relationship,
            reported_distance_angstrom=1.4,
        ),
    )


@pytest.mark.parametrize("use_override", (False, True))
def test_hydrogenation_resolves_connectivity_only_source_edges(use_override: bool):
    source_bond = _source_bond()
    source = _with_bonds(_methanol(), (source_bond,))
    evidence = (
        (
            RetainedNonPolymerChemistryOverride(
                LIGAND_ID, "CO", ("C1", "O1")
            ).to_evidence(),
        )
        if use_override
        else ()
    )

    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=evidence
    )

    bond = result.structure.topology.bond_between(AtomIndex(0), AtomIndex(1))
    assert bond == replace(
        source_bond, order=1, relationship_type=BondRelationshipType.COVALENT
    )
    assert len(result.structure.constitution.atom_slots) == 6
    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        restored = read_structure_string(
            write_structure_string(result.structure, file_format), file_format
        )
        first = restored.constitution.atom_index(AtomRef(LIGAND_ID, "C1"))
        second = restored.constitution.atom_index(AtomRef(LIGAND_ID, "O1"))
        restored_bond = restored.topology.bond_between(first, second)
        assert restored_bond is not None
        # A single PDB CONECT reports connectivity only; mmCIF retains order.
        assert restored_bond.order == (1 if file_format is FileFormat.MMCIF else None)


@pytest.mark.parametrize(
    ("order", "relationship"),
    (
        (2, BondRelationshipType.COVALENT),
        (None, BondRelationshipType.METAL_COORDINATION),
        (None, BondRelationshipType.HYDROGEN_BOND),
        (1, BondRelationshipType.DISULFIDE),
    ),
)
def test_hydrogenation_rejects_conflicting_source_chemistry_atomically(
    order, relationship
):
    source = _with_bonds(
        _methanol(), (_source_bond(order=order, relationship=relationship),)
    )
    result = add_retained_non_polymer_hydrogens(source, ComponentLibrary({}))

    assert result.structure is source
    assert not result.repairs
    expected_reason = "bond order" if order == 2 else "relationship"
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        and issue.residue_id == LIGAND_ID
        and expected_reason in issue.message
        for issue in result.issues
    )


@pytest.mark.parametrize("order", (None, 1))
@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.UNKNOWN, BondRelationshipType.COVALENT)
)
def test_bond_refinement_preserves_provenance_and_reported_attributes(
    order, relationship
):
    source = _source_bond(order=order, relationship=relationship)
    expected = TopologyBond(AtomIndex(0), AtomIndex(1))
    resolved = source.with_resolved_chemistry(expected)

    assert resolved.order == 1
    assert resolved.relationship_type is BondRelationshipType.COVALENT
    assert resolved.source_metadata is source.source_metadata
    assert resolved.provenance is source.provenance
    assert resolved.with_resolved_chemistry(expected) == resolved


def test_bond_refinement_rejects_different_endpoints():
    with pytest.raises(ValueError, match="endpoint"):
        _source_bond().with_resolved_chemistry(TopologyBond(AtomIndex(1), AtomIndex(2)))


def test_bond_refinement_can_revise_inferred_but_not_reported_order():
    expected = TopologyBond(AtomIndex(0), AtomIndex(1), order=2)
    inferred = replace(
        _source_bond(), order=1, relationship_type=BondRelationshipType.COVALENT
    )
    resolved = inferred.with_resolved_chemistry(expected)
    assert resolved.order == 2
    assert resolved.source_metadata is inferred.source_metadata
    assert resolved.source_metadata is not None
    assert resolved.source_metadata.reported_order is None

    reported = _source_bond(order=1, relationship=BondRelationshipType.COVALENT)
    with pytest.raises(ValueError, match="bond order"):
        reported.with_resolved_chemistry(expected)


@pytest.mark.parametrize("use_override", (False, True))
def test_complete_h_topology_resolution_is_coordinate_preserving_and_idempotent(
    use_override,
):
    library = ComponentLibrary({})
    evidence = (
        (
            RetainedNonPolymerChemistryOverride(
                LIGAND_ID, "CO", ("C1", "O1")
            ).to_evidence(),
        )
        if use_override
        else ()
    )
    hydrogenated = add_retained_non_polymer_hydrogens(
        _methanol(), library, chemistry_evidence=evidence
    ).structure
    source = _with_bonds(
        hydrogenated,
        tuple(
            _source_bond(bond.atom_index_1.value, bond.atom_index_2.value)
            for bond in hydrogenated.topology.bonds
        ),
    )
    result = resolve_retained_non_polymer_topology(
        source, library, chemistry_evidence=evidence
    )

    assert result.structure.constitution is source.constitution
    assert result.structure.geometry is source.geometry
    assert result.structure.topology.atom_topologies == source.topology.atom_topologies
    assert len(result.structure.topology.bonds) == 5
    assert all(
        bond.order == 1 and bond.relationship_type is BondRelationshipType.COVALENT
        for bond in result.structure.topology.bonds
    )
    assert tuple(
        bond.source_metadata for bond in result.structure.topology.bonds
    ) == tuple(bond.source_metadata for bond in source.topology.bonds)
    repeated = resolve_retained_non_polymer_topology(
        result.structure, library, chemistry_evidence=evidence
    )
    assert repeated.structure is result.structure
    assert not repeated.repairs


def _named_methanol(symbol: str = "H") -> ProteinStructure:
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.4, 0, 0)),
                    atom_payload("HX", symbol, Vec3(1.95, 0.75, 0)),
                    atom_payload("HA", "H", Vec3(-0.65, 0.8, 0)),
                    atom_payload("HB", "H", Vec3(-0.65, -0.8, 0)),
                    atom_payload("HC", "H", Vec3(0, 0, 1)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return _with_bonds(
        structure,
        tuple(_source_bond(*pair) for pair in ((0, 1), (1, 2), (0, 3), (0, 4), (0, 5))),
    )


def test_direct_hydrogenation_of_complete_source_only_resolves_topology():
    source = _named_methanol()
    library = ComponentLibrary({})
    result = add_retained_non_polymer_hydrogens(source, library)

    assert result.structure.geometry is source.geometry
    assert result.structure.constitution is source.constitution
    assert all(bond.order == 1 for bond in result.structure.topology.bonds)
    repeated = add_retained_non_polymer_hydrogens(result.structure, library)
    assert repeated.structure is result.structure
    assert not repeated.repairs


def test_complete_h_count_does_not_allow_changing_a_source_attachment():
    source = _named_methanol()
    bonds = tuple(
        bond
        for bond in source.topology.bonds
        if bond.endpoint_pair() != (AtomIndex(1), AtomIndex(2))
    )
    source = _with_bonds(source, (*bonds, _source_bond(0, 2)))
    library = ComponentLibrary({})
    fact = (
        StructureChemistryObservation(source, component_library=library)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert not fact.requires_hydrogen_completion()
    assert fact.requires_topology_resolution()

    result = resolve_retained_non_polymer_topology(source, library)
    assert result.structure is source
    assert not result.repairs
    assert result.issues


@pytest.mark.parametrize(
    "relationship",
    (
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
    ),
)
def test_extra_source_h_h_edge_is_not_a_ready_hydrogen_graph(relationship):
    library = ComponentLibrary({})
    source = resolve_retained_non_polymer_topology(_named_methanol(), library).structure
    source = _with_bonds(
        source,
        (
            *source.topology.bonds,
            _source_bond(2, 3, order=1, relationship=relationship),
        ),
    )
    fact = (
        StructureChemistryObservation(source, component_library=library)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.requires_topology_resolution()

    result = resolve_retained_non_polymer_topology(source, library)
    assert result.structure is source
    assert not result.repairs
    assert result.issues


def test_extra_h_disulfide_edge_is_not_a_ready_hydrogen_graph():
    library = ComponentLibrary({})
    source = resolve_retained_non_polymer_topology(_named_methanol(), library).structure
    source = _with_bonds(
        source,
        (
            *source.topology.bonds,
            _source_bond(0, 2, order=1, relationship=BondRelationshipType.DISULFIDE),
        ),
    )
    fact = (
        StructureChemistryObservation(source, component_library=library)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.requires_topology_resolution()

    result = resolve_retained_non_polymer_topology(source, library)
    assert result.structure is source
    assert not result.repairs
    assert result.issues


@pytest.mark.parametrize("symbol", ("H", "D", "T"))
@pytest.mark.parametrize("use_override", (False, True))
def test_topology_only_preserves_named_isotopic_h_and_readiness(symbol, use_override):
    source = _named_methanol(symbol)
    library = ComponentLibrary({})
    evidence = (
        (
            RetainedNonPolymerChemistryOverride(
                LIGAND_ID, "CO", ("C1", "O1")
            ).to_evidence(),
        )
        if use_override
        else ()
    )
    before = (
        StructureChemistryObservation(
            source,
            component_library=library,
            retained_non_polymer_chemistry_evidence=evidence,
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert before.requires_topology_resolution()
    assert not before.requires_hydrogen_completion()

    result = resolve_retained_non_polymer_topology(
        source, library, chemistry_evidence=evidence
    )
    assert result.structure.geometry is source.geometry
    assert result.structure.constitution is source.constitution
    assert result.repairs
    after = (
        StructureChemistryObservation(
            result.structure,
            component_library=library,
            retained_non_polymer_chemistry_evidence=evidence,
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert not after.requires_topology_resolution()
    assert not after.requires_hydrogen_completion()
    bond = result.structure.topology.bond_between(AtomIndex(1), AtomIndex(2))
    assert bond is not None
    assert bond.order == 1


def test_public_workflow_resolves_topology_without_replacing_existing_h():
    source = _named_methanol("D")
    result = process_structure(
        source,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure.geometry == source.geometry
    assert result.structure.constitution == source.constitution
    assert all(bond.order == 1 for bond in result.structure.topology.bonds)
    repeated = process_structure(
        result.structure,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert repeated.structure.topology == result.structure.topology
    assert repeated.structure.geometry == source.geometry
    assert not repeated.repairs


def test_topology_only_refuses_missing_h_and_strict_fallback():
    source = _methanol()
    result = resolve_retained_non_polymer_topology(source, ComponentLibrary({}))
    assert result.structure is source
    assert not result.repairs
    assert result.issues

    source = _named_methanol()
    result = resolve_retained_non_polymer_topology(
        source, ComponentLibrary({}), allow_retained_non_polymer_rdkit_fallback=False
    )
    assert result.structure is source
    assert not result.repairs
    assert any("disabled" in issue.message for issue in result.issues)


@pytest.mark.parametrize(
    "relationship",
    (
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.HYDROGEN_BOND,
        BondRelationshipType.METAL_COORDINATION,
    ),
)
@pytest.mark.parametrize("resolved_local", (False, True))
def test_external_relationship_is_preserved_without_becoming_a_local_h_anchor(
    relationship,
    resolved_local,
):
    ligand = _named_methanol()
    if resolved_local:
        ligand = resolve_retained_non_polymer_topology(
            ligand, ComponentLibrary({})
        ).structure
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        atom.name,
                        atom.element,
                        ligand.geometry.position(AtomIndex(index)),
                    )
                    for index, atom in enumerate(ligand.constitution.atom_slots)
                ),
            ),
            residue_payload(
                component_id="ION",
                residue_id=ResidueId("X", 9),
                is_hetero=True,
                atoms=(atom_payload("O", "O", Vec3(3, 1, 0)),),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    external = _source_bond(2, 6, relationship=relationship)
    source = _with_bonds(source, ligand.topology.bonds)
    observation = StructureChemistryObservation(
        source, component_library=ComponentLibrary({})
    )
    initial = observation.whole_structure_facts()[1].retained_non_polymer_facts[0]
    assert initial.requires_topology_resolution() is not resolved_local
    source = _with_bonds(source, (*ligand.topology.bonds, external))
    fact = (
        observation.for_structure(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    covalent_attachment = relationship in {
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
    }
    if covalent_attachment:
        assert "external" in fact.chemistry_failure_reason
        assert not fact.requires_topology_resolution()
    else:
        assert fact.requires_topology_resolution() is not resolved_local
    result = resolve_retained_non_polymer_topology(
        source, ComponentLibrary({}), target_residue_ids=frozenset((LIGAND_ID,))
    )
    assert (
        result.structure.topology.bond_between(AtomIndex(2), AtomIndex(6)) == external
    )
    if covalent_attachment:
        assert result.structure is source
        assert not result.repairs
    else:
        assert bool(result.repairs) is not resolved_local
        assert result.structure.geometry is source.geometry
        bond = result.structure.topology.bond_between(AtomIndex(1), AtomIndex(2))
        assert bond is not None
        assert bond.order == 1


def test_source_conflict_keeps_residue_while_neighbor_completes():
    neighbor_id = ResidueId("L", 2)
    source = build_structure(
        chains=(),
        ligands=tuple(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, y, 0)),
                    atom_payload("O1", "O", Vec3(1.4, y, 0)),
                ),
            )
            for residue_id, y in ((LIGAND_ID, 0), (neighbor_id, 10))
        ),
        source_format=FileFormat.PDB,
    )
    conflict = _source_bond(0, 1, order=2, relationship=BondRelationshipType.COVALENT)
    source = _with_bonds(source, (conflict, _source_bond(2, 3)))
    evidence = tuple(
        RetainedNonPolymerChemistryOverride(
            residue_id, "CO", ("C1", "O1")
        ).to_evidence()
        for residue_id in (LIGAND_ID, neighbor_id)
    )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=evidence
    )

    assert result.structure.constitution.ligands[0] == source.constitution.ligands[0]
    assert result.structure.residue_geometry(
        result.structure.constitution.residue_index(LIGAND_ID)
    ) == source.residue_geometry(source.constitution.residue_index(LIGAND_ID))
    assert (
        result.structure.topology.bond_between(AtomIndex(0), AtomIndex(1)) == conflict
    )
    assert len(result.structure.constitution.ligands[1].atom_sites) == 6
    assert any(event.residue_id == neighbor_id for event in result.repairs)
    assert not any(event.residue_id == LIGAND_ID for event in result.repairs)
    assert any(
        issue.residue_id == LIGAND_ID
        and issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        and "bond order" in issue.message
        for issue in result.issues
    )


def test_override_replaces_obsolete_model_edges_but_not_explicit_source_edges():
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.4, 0, 0)),
                    atom_payload("N1", "N", Vec3(1.4, 1.4, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = _with_bonds(
        source,
        (
            TopologyBond(AtomIndex(0), AtomIndex(1)),
            TopologyBond(AtomIndex(0), AtomIndex(2)),
        ),
    )
    evidence = (
        RetainedNonPolymerChemistryOverride(
            LIGAND_ID, "CON", ("C1", "O1", "N1")
        ).to_evidence(),
    )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=evidence
    )
    assert result.structure.topology.bond_between(AtomIndex(0), AtomIndex(2)) is None
    assert (
        result.structure.topology.bond_between(AtomIndex(1), AtomIndex(2)) is not None
    )

    explicit = _with_bonds(source, (_source_bond(0, 2),))
    refused = add_retained_non_polymer_hydrogens(
        explicit, ComponentLibrary({}), chemistry_evidence=evidence
    )
    assert refused.structure is explicit
    assert not refused.repairs


@pytest.mark.parametrize("source_explicit", (False, True))
def test_extra_heavy_edge_requires_resolution_even_with_complete_h(source_explicit):
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.4, 0, 0)),
                    atom_payload("N1", "N", Vec3(1.4, 1.4, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    library = ComponentLibrary({})
    evidence = (
        RetainedNonPolymerChemistryOverride(
            LIGAND_ID, "CON", ("C1", "O1", "N1")
        ).to_evidence(),
    )
    source = add_retained_non_polymer_hydrogens(
        source, library, chemistry_evidence=evidence
    ).structure
    extra = (
        _source_bond(0, 2)
        if source_explicit
        else TopologyBond(AtomIndex(0), AtomIndex(2))
    )
    source = _with_bonds(source, (*source.topology.bonds, extra))
    fact = (
        StructureChemistryObservation(
            source,
            component_library=library,
            retained_non_polymer_chemistry_evidence=evidence,
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert not fact.requires_hydrogen_completion()
    assert fact.requires_topology_resolution()

    result = resolve_retained_non_polymer_topology(
        source, library, chemistry_evidence=evidence
    )
    assert result.structure.geometry is source.geometry
    assert result.structure.constitution is source.constitution
    if source_explicit:
        assert result.structure is source
        assert result.issues
    else:
        assert (
            result.structure.topology.bond_between(AtomIndex(0), AtomIndex(2)) is None
        )
        assert result.repairs


def test_template_publishes_heavy_and_h_bonds_with_noncanonical_h_name():
    template = build_retained_non_polymer_component_library().get("LIG")
    assert template is not None
    library = ComponentLibrary(
        {
            "LIG": replace(
                template,
                definition=replace(
                    template.definition,
                    bonds=(BondDefinition("C1", "O1"), BondDefinition("C1", "N1")),
                ),
            )
        }
    )
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.4, 0, 0)),
                    atom_payload("N1", "N", Vec3(0, 1.3, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = _with_bonds(source, (_source_bond(0, 1), _source_bond(0, 2)))
    generated = add_retained_non_polymer_hydrogens(source, library).structure
    assert all(bond.order == 1 for bond in generated.topology.bonds)
    assert len(generated.topology.bonds) == 3

    residue_index = generated.constitution.residue_index(LIGAND_ID)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        "HX" if atom.is_hydrogen() else atom.name,
                        atom.element,
                        generated.residue_geometry(residue_index).position(atom.name),
                    )
                    for atom in generated.constitution.ligands[0].atom_sites
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = _with_bonds(source, (_source_bond(0, 1), _source_bond(0, 2)))
    result = resolve_retained_non_polymer_topology(source, library)
    assert result.structure.geometry is source.geometry
    assert result.structure.constitution is source.constitution
    assert len(result.structure.topology.bonds) == 3
    facts = StructureChemistryObservation(
        result.structure, component_library=library
    ).whole_structure_facts()[1]
    assert not facts.retained_non_polymer_facts[0].requires_topology_resolution()


@pytest.mark.parametrize("force_field_name", ("UFF", "MMFF"))
def test_resolved_graph_binds_native_force_fields_and_rdkit_reads_output(
    force_field_name,
):
    result = resolve_retained_non_polymer_topology(
        _named_methanol(), ComponentLibrary({})
    )
    problem = build_continuous_relaxation_problem(
        result.structure,
        LocalScopeSpec.from_residues((LIGAND_ID,)),
        component_library=ComponentLibrary(
            {
                "UNK": ResidueTemplate(
                    ChemicalComponentDefinition("UNK", atom_names=("C1", "O1"))
                )
            }
        ),
        force_field=ContinuousRelaxationForceField[force_field_name],
    )
    molecule, mapping = build_rdkit_molecule(problem)
    assert molecule.GetNumBonds() == 5
    assert molecule.GetNumAtoms() == 6
    force_field = build_rdkit_force_field(
        molecule,
        problem=problem,
        restraint_library=build_default_restraint_library(),
        rdkit_atom_index_by_structure_atom_index=mapping,
    )
    assert isfinite(force_field.CalcEnergy())
    assert all(isfinite(value) for value in force_field.CalcGrad())
    pdb = write_structure_string(result.structure, FileFormat.PDB)
    assert (
        Chem.MolFromPDBBlock(pdb, sanitize=True, removeHs=False, proximityBonding=True)
        is not None
    )
