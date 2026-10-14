"""Preserved H attachment truth without coordinate or protonation changes."""

from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.io import (
    FileFormat,
    read_structure,
    read_structure_string,
    write_structure_string,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState, TopologyAvailabilityState
from protrepair.state.topology import (
    HydrogenAttachmentResolutionFacts,
    HydrogenAttachmentResolutionState,
    TopologyAvailabilityAspect,
    TopologyAvailabilityFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.topology import (
    resolve_fixed_hydrogen_topology,
)
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.hydrogen_topology import (
    PolymerHydrogenTopologyResolutionTransformer,
)
from protrepair.workflow.contracts import RequestedGoalSet, WorkflowTransformRequests
from protrepair.workflow.contracts.request import requested_process_goal
from protrepair.workflow.planning.planner import plan_workflow_actions

_FIXTURE = Path("tests/fixtures/pdb/1aho.pdb")
_RID = ResidueId("A", 41)
pytestmark = pytest.mark.corpus


def _proline(
    *, alias: str = "HB3", element: str = "H", charge: str = ""
) -> ProteinStructure:
    lines = []
    for line in _FIXTURE.read_text().splitlines():
        if (
            not line.startswith("ATOM  ")
            or line[17:20] != "PRO"
            or int(line[22:26]) != 41
        ):
            continue
        if line[12:16].strip() == "HB3":
            line = f"{line[:12]}{alias:^4}{line[16:76]}{element:>2}{charge:>2}"
        lines.append(line)
    assert lines
    return read_structure_string("\n".join(lines) + "\nEND\n", FileFormat.PDB)


def _bond(
    structure: ProteinStructure, name: str, parent: str, residue_id: ResidueId = _RID
):
    return structure.topology.bond_between(
        structure.constitution.atom_index(AtomRef(residue_id, name)),
        structure.constitution.atom_index(AtomRef(residue_id, parent)),
    )


def _with_source_bonds(
    structure: ProteinStructure,
    specs: tuple[tuple[str, str, BondRelationshipType, int | None, bool], ...],
) -> ProteinStructure:
    by_pair = {bond.endpoint_pair(): bond for bond in structure.topology.bonds}
    for name, parent, relationship, order, aromatic in specs:
        bond = TopologyBond(
            structure.constitution.atom_index(AtomRef(_RID, name)),
            structure.constitution.atom_index(AtomRef(_RID, parent)),
            order=order,
            aromatic=aromatic,
            relationship_type=relationship,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                source_id="preserved-h",
                reported_order=order,
                reported_relationship_type=relationship,
            ),
        )
        by_pair[bond.endpoint_pair()] = bond
    topology = StructureTopology(
        constitution=structure.constitution,
        atom_topologies=structure.topology.atom_topologies,
        bonds=tuple(by_pair.values()),
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=replace(
            structure.provenance,
            ingress=replace(
                structure.provenance.ingress,
                observation=StructureObservation(
                    structure.constitution,
                    structure.geometry,
                    StructureTopology(
                        constitution=structure.constitution,
                        atom_topologies=topology.atom_topologies,
                        bonds=tuple(
                            replace(bond, aromatic=False)
                            for bond in topology.bonds
                            if bond.provenance is BondProvenance.SOURCE_EXPLICIT
                        ),
                    ),
                ),
            ),
        ),
    )


def _hydrogen_topology_state(structure: ProteinStructure) -> TopologyAvailabilityState:
    facts = TopologyAvailabilityFacts.from_projection(
        ProteinStructureSnapshot.from_structure(structure),
        residue_ids=(_RID,),
        component_library=build_default_component_library(),
    )
    return next(
        fact.state
        for fact in facts.residue_facts
        if fact.aspect is TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS
    )


@pytest.mark.parametrize(
    "has_hydrogens,targets",
    [
        (False, None),
        (False, frozenset()),
        (False, frozenset({_RID})),
        (True, frozenset()),
        (True, frozenset({ResidueId("A", 99)})),
    ],
)
def test_no_selected_present_h_skips_preparation(has_hydrogens, targets, monkeypatch):
    source = _proline()
    if not has_hydrogens:
        source = source.without_hydrogens()

    def unexpected_preparation(*args, **kwargs):
        raise AssertionError("no selected present H needs chemical preparation")

    monkeypatch.setattr(
        "protrepair.transformer.completion.hydrogen.topology.PolymerMicrostatePreparation",
        unexpected_preparation,
    )
    result = resolve_fixed_hydrogen_topology(
        source, build_default_component_library(), target_residue_ids=targets
    )
    assert result.structure is source
    assert not result.repairs and not result.issues


@pytest.mark.parametrize(
    "alias,element", [("HB3", "H"), ("3HB", "H"), ("DB3", "D"), ("3TB", "T")]
)
def test_present_alias_bonds_preserve_all_other_facets_and_are_idempotent(
    alias, element
):
    source = _proline(alias=alias, element=element)
    assert _bond(source, alias, "CB") is None
    assert _hydrogen_topology_state(source) is TopologyAvailabilityState.ABSENT
    library = build_default_component_library()
    query = PolymerMicrostatePreparation(PolymerMicrostateContext(source), library)
    assert query.fixed_hydrogen_topology_updates(_RID)
    assert _bond(source, alias, "CB") is None

    result = resolve_fixed_hydrogen_topology(source, library)
    after = result.structure
    bond = _bond(after, alias, "CB")
    assert bond is not None and bond.order == 1
    assert bond.relationship_type is BondRelationshipType.COVALENT
    assert bond.provenance is BondProvenance.TEMPLATE_RESOLVED
    assert after.constitution is source.constitution
    assert after.geometry is source.geometry
    assert after.provenance is source.provenance
    assert after.polymer_blueprint is source.polymer_blueprint
    assert after.topology.atom_topologies == source.topology.atom_topologies
    assert not result.issues
    assert result.repairs[0].kind is RepairEventKind.HYDROGEN_TOPOLOGY_RESOLVED
    assert _hydrogen_topology_state(after) is TopologyAvailabilityState.PRESENT
    detail = HydrogenAttachmentResolutionFacts.from_projection(
        ProteinStructureSnapshot.from_structure(after),
        residue_ids=(_RID,),
        component_library=library,
    )
    assert (
        detail.residue_facts[0].state
        is HydrogenAttachmentResolutionState.TOPOLOGY_RESOLVED
    )
    assert not detail.any_coordinate_inferred()
    again = resolve_fixed_hydrogen_topology(after, library)
    assert again.structure is after
    assert not again.repairs and not again.issues


@pytest.mark.parametrize(
    "relationship", [BondRelationshipType.UNKNOWN, BondRelationshipType.COVALENT]
)
def test_incomplete_source_bond_is_resolved_without_losing_reported_evidence(
    relationship,
):
    source = _with_source_bonds(_proline(), (("HB3", "CB", relationship, None, False),))
    before = _bond(source, "HB3", "CB")
    result = resolve_fixed_hydrogen_topology(source, build_default_component_library())
    after = _bond(result.structure, "HB3", "CB")
    assert before is not None and after is not None
    assert after == replace(
        before, relationship_type=BondRelationshipType.COVALENT, order=1
    )
    assert after.source_metadata is before.source_metadata
    assert result.structure.provenance is source.provenance


def test_typed_source_parent_can_resolve_an_arbitrary_h_name():
    source = _with_source_bonds(
        _proline(alias="QH"),
        (("QH", "CB", BondRelationshipType.COVALENT, None, False),),
    )
    result = resolve_fixed_hydrogen_topology(source, build_default_component_library())
    bond = _bond(result.structure, "QH", "CB")
    assert bond is not None and bond.order == 1
    assert not result.issues


@pytest.mark.parametrize(
    "specs",
    [
        (("HB3", "CB", BondRelationshipType.COVALENT, 2, False),),
        (("HB3", "CB", BondRelationshipType.COVALENT, 1, True),),
        (("HB3", "CB", BondRelationshipType.METAL_COORDINATION, None, False),),
        (("HB3", "CA", BondRelationshipType.UNKNOWN, None, False),),
        (("HB3", "CB", BondRelationshipType.HYDROGEN_BOND, None, False),),
        (
            ("HB3", "CB", BondRelationshipType.COVALENT, 1, False),
            ("HB3", "CA", BondRelationshipType.COVALENT, 1, False),
        ),
    ],
)
def test_conflicting_h_relationships_are_not_overwritten_or_declared_ready(specs):
    source = _with_source_bonds(_proline(), specs)
    result = resolve_fixed_hydrogen_topology(source, build_default_component_library())
    assert result.structure is source
    assert not result.repairs and result.issues
    assert _hydrogen_topology_state(source) is TopologyAvailabilityState.ABSENT


@pytest.mark.parametrize("alias,charge", [("QH", ""), ("HB3", "1+"), ("HB3", "1-")])
def test_unknown_or_charged_h_is_not_guessed_from_coordinates(alias, charge):
    source = _proline(alias=alias, charge=charge)
    result = resolve_fixed_hydrogen_topology(source, build_default_component_library())
    assert result.structure is source
    assert not result.repairs and result.issues
    assert _hydrogen_topology_state(source) is TopologyAvailabilityState.ABSENT


def test_scoped_resolution_does_not_repair_other_residues():
    source = read_structure(_FIXTURE)
    result = resolve_fixed_hydrogen_topology(
        source, build_default_component_library(), target_residue_ids=frozenset({_RID})
    )
    assert _bond(result.structure, "HB3", "CB") is not None
    assert _bond(result.structure, "HB3", "CB", ResidueId("A", 52)) is None
    assert result.structure.geometry is source.geometry
    assert {rid for event in result.repairs for rid in event.scope.residue_ids} == {
        _RID
    }


def test_known_nonstandard_template_resolves_prefix_h_without_geometry():
    record = build_bundled_nonstandard_registry().get("HYP")
    assert record is not None
    atoms = []
    for atom in record.atoms:
        if atom.is_hydrogen() and atom.atom_name != "HB3":
            continue
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(
                name="3HB" if atom.atom_name == "HB3" else atom.atom_name,
                element=atom.element,
                position=Vec3.from_iterable(atom.ideal_position),
            )
        )
    source = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="HYP",
                        residue_id=_RID,
                        atoms=tuple(atoms),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    result = resolve_fixed_hydrogen_topology(source, build_default_component_library())
    assert _bond(result.structure, "3HB", "CB") is not None
    assert result.structure.geometry is source.geometry
    assert not result.issues


@pytest.mark.parametrize("name", ("HG11", "1HG1"))
def test_branched_carbon_h_names_are_resolved_without_a_distance_cutoff(name):
    library = build_default_component_library()
    template = library.require("ILE")
    heavy = tuple(
        atom_payload(name=heavy_name, element=heavy_name[0], position=Vec3(i, 0, 0))
        for i, heavy_name in enumerate(template.expected_heavy_atom_names())
    )
    residue = residue_payload(
        component_id="ILE",
        residue_id=_RID,
        atoms=(*heavy, atom_payload(name=name, element="H", position=Vec3(100, 0, 0))),
    )
    source = build_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
    )
    result = resolve_fixed_hydrogen_topology(source, library)
    assert _bond(result.structure, name, "CG1") is not None
    assert result.structure.geometry is source.geometry
    assert not result.issues


@pytest.mark.parametrize(
    "component,hetero", [("UNK", False), ("PRO", True), ("ZN", True)]
)
def test_unknown_polymer_and_retained_components_do_not_use_polymer_h_grammar(
    component, hetero
):
    residue = residue_payload(
        component_id=component,
        residue_id=_RID,
        is_hetero=hetero,
        atoms=(
            atom_payload(name="CB", element="C", position=Vec3(0, 0, 0)),
            atom_payload(name="HB3", element="H", position=Vec3(1, 0, 0)),
        ),
    )
    source = build_structure(
        chains=() if hetero else (chain_payload("A", (residue,)),),
        ligands=(residue,) if hetero else (),
        source_format=FileFormat.PDB,
    )
    result = resolve_fixed_hydrogen_topology(source, build_default_component_library())
    assert result.structure is source and not result.repairs


def test_foreign_h_parent_refuses_only_affected_residue():
    lines = [line for line in _FIXTURE.read_text().splitlines() if line != "END"]
    serials = {
        (int(line[22:26]), line[12:16].strip()): int(line[6:11])
        for line in lines
        if line.startswith("ATOM  ")
    }
    lines.append(f"CONECT{serials[(41, 'HB3')]:5d}{serials[(52, 'CB')]:5d}")
    source = read_structure_string("\n".join(lines) + "\nEND\n", FileFormat.PDB)
    result = resolve_fixed_hydrogen_topology(
        source,
        build_default_component_library(),
        target_residue_ids=frozenset({_RID, ResidueId("A", 52)}),
    )
    assert _bond(result.structure, "HB3", "CB") is None
    assert _bond(result.structure, "HB3", "CB", ResidueId("A", 52)) is not None
    assert result.issues and result.issues[0].scope.residue_ids == (_RID,)
    assert result.structure.geometry is source.geometry


def test_direct_h_completion_records_source_bonds_even_if_placement_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise GeometryPlacementError("deliberate placement failure")

    monkeypatch.setattr(
        "protrepair.transformer.completion.hydrogen.core.generate_hydrogen_patch", fail
    )
    source = _proline()
    result = materialize_hydrogens_core(source)
    assert result.issues
    assert _bond(result.structure, "HB3", "CB") is not None
    ref = AtomRef(_RID, "HB3")
    assert result.structure.geometry.atom_geometry(
        result.structure.constitution.atom_index(ref)
    ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))


@pytest.mark.workflow
def test_workflow_repairs_complete_source_h_inventory_and_reassesses_topology():
    goals = (
        requested_process_goal(
            scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
        ),
    )
    source = read_structure(_FIXTURE)
    result = process_structure(_FIXTURE, requested_goals=goals)
    assert not result.has_errors()
    after = result.structure
    attached = {
        index
        for bond in after.topology.bonds
        if bond.relationship_type is BondRelationshipType.COVALENT
        and bond.order == 1
        and not bond.aromatic
        for index in bond.endpoint_pair()
    }
    assert all(
        AtomIndex(index) in attached
        for index, atom in enumerate(after.constitution.atom_slots)
        if atom.is_hydrogen()
    )
    for seq_num in (41, 52, 60):
        rid = ResidueId("A", seq_num)
        for name, parent in (("HB3", "CB"), ("HG3", "CG"), ("HD3", "CD")):
            assert _bond(after, name, parent, rid) is not None
            ref = AtomRef(rid, name)
            assert after.geometry.atom_geometry(
                after.constitution.atom_index(ref)
            ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))
    assert _hydrogen_topology_state(after) is TopologyAvailabilityState.PRESENT
    assert any(
        event.kind is RepairEventKind.HYDROGEN_TOPOLOGY_RESOLVED
        and _RID in event.scope.residue_ids
        for event in result.repairs
    )
    missing = _bond(after, "HB3", "CB")
    incomplete = ProteinStructure.from_payload(
        constitution=after.constitution,
        geometry=after.geometry,
        topology=StructureTopology(
            constitution=after.constitution,
            atom_topologies=after.topology.atom_topologies,
            bonds=tuple(bond for bond in after.topology.bonds if bond != missing),
        ),
        provenance=after.provenance,
        polymer_blueprint=after.polymer_blueprint,
    )
    plan = plan_workflow_actions(
        incomplete,
        requested_goals=RequestedGoalSet(goals),
        transform_requests=WorkflowTransformRequests(),
    )
    assert any(
        isinstance(action, PolymerHydrogenTopologyResolutionTransformer)
        for action in plan.transformers
    )
    local_plan = plan_workflow_actions(
        incomplete,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((_RID,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )
    assert any(
        isinstance(action, PolymerHydrogenTopologyResolutionTransformer)
        for action in local_plan.transformers
    )
    replanned = plan_workflow_actions(
        after,
        requested_goals=RequestedGoalSet(goals),
        transform_requests=WorkflowTransformRequests(),
    )
    assert not any(
        isinstance(action, PolymerHydrogenTopologyResolutionTransformer)
        for action in replanned.transformers
    )

    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        block = write_structure_string(after, file_format)
        roundtrip = read_structure_string(block, file_format)
        assert _bond(roundtrip, "HB3", "CB") is not None
    block = write_structure_string(after, FileFormat.PDB)
    for include_conect in (True, False):
        text = (
            block
            if include_conect
            else "\n".join(
                line for line in block.splitlines() if not line.startswith("CONECT")
            )
        )
        assert (
            Chem.MolFromPDBBlock(
                text, sanitize=True, removeHs=False, proximityBonding=True
            )
            is not None
        )
