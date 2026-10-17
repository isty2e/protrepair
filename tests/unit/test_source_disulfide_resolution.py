"""Explicit untyped sulfur graphs resolve without a distance-based guess."""

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
from tests.support.cysteine_michael_adduct import assert_original_preserved, with_bonds
from tests.support.request_builders import whole_structure_requested_goals

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state import HydrogenCoverageState
from protrepair.state.structure_topology import StructureDisulfideTopologyFacts
from protrepair.structure.disulfide import disulfide_atom_ref_pairs
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    TopologyBond,
)
from protrepair.transformer.base import ProjectedDomainState
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.disulfide_topology import (
    DisulfideTopologyResolutionTransformer,
)
from protrepair.workflow.contracts import RequestedGoalSet, WorkflowTransformRequests
from protrepair.workflow.planning.planner import plan_workflow_actions


def _source_pair(
    *, distance=8.0, charge=None, extra=None, sulfur_element="S", geometric_pair=False
):
    payloads = []
    positions = ((1, 0.0), (2, distance))
    if geometric_pair:
        positions += ((3, 20.0), (4, 22.05))
    for seq, x in positions:
        atoms = [
            atom_payload("CB", "C", Vec3(x, 1.8, 0.0)),
            atom_payload("SG", sulfur_element, Vec3(x, 0.0, 0.0), formal_charge=charge),
        ]
        if extra is not None and seq == 1:
            atoms.append(
                atom_payload(extra, "D" if extra == "DG" else "C", Vec3(x, -1.3, 0.0))
            )
        payloads.append(
            residue_payload(
                component_id="CYS", residue_id=ResidueId("A", seq), atoms=tuple(atoms)
            )
        )
    source = build_structure(
        chains=(chain_payload(chain_id="A", residues=tuple(payloads)),),
        source_format=FileFormat.PDB,
        source_name="untyped-sulfur-graph",
    )
    c = source.constitution
    bonds = [
        TopologyBond(
            c.atom_index(AtomRef(ResidueId("A", seq), "CB")),
            c.atom_index(AtomRef(ResidueId("A", seq), "SG")),
        )
        for seq in (1, 2)
    ]
    bonds.append(
        TopologyBond(
            c.atom_index(AtomRef(ResidueId("A", 1), "SG")),
            c.atom_index(AtomRef(ResidueId("A", 2), "SG")),
            order=None,
            relationship_type=BondRelationshipType.UNKNOWN,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
                source_id="sulfur-connection",
            ),
        )
    )
    if extra == "CX":
        bonds.append(
            TopologyBond(
                c.atom_index(AtomRef(ResidueId("A", 1), "SG")),
                c.atom_index(AtomRef(ResidueId("A", 1), extra)),
            )
        )
    return with_bonds(source, tuple(bonds))


@pytest.mark.parametrize("distance", (0.0, 2.05, 8.0))
def test_source_graph_resolution_is_not_a_distance_decision(distance):
    source = _source_pair(distance=distance)
    facts = StructureDisulfideTopologyFacts.from_structure(source)
    assert not disulfide_atom_ref_pairs(source)
    assert facts.has_promotable_candidates()
    planning = plan_workflow_actions(
        source,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )
    assert isinstance(planning.transformers[0], DisulfideTopologyResolutionTransformer)
    assert planning.state_deficit is not None
    assert planning.state_deficit.topology_resolution is not None
    assert planning.state_deficit.topology_resolution.promotable_disulfide_count == 1

    result = process_structure(source)

    pairs = disulfide_atom_ref_pairs(result.structure)
    assert len(pairs) == 1
    bond = result.structure.topology.bond_between(
        *(result.structure.constitution.atom_index(ref) for ref in next(iter(pairs)))
    )
    assert bond is not None and bond.relationship_type is BondRelationshipType.DISULFIDE
    assert bond.order == 1
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    original = source.topology.bond_between(
        *(source.constitution.atom_index(ref) for ref in next(iter(pairs)))
    )
    assert original is not None and bond.source_metadata == original.source_metadata
    assert_original_preserved(source, result.structure)
    assert process_structure(result.structure).structure == result.structure


def test_one_action_resolves_independent_source_and_geometric_evidence():
    source = _source_pair(geometric_pair=True)
    facts = StructureDisulfideTopologyFacts.from_structure(source)
    assert len(facts.source_atom_ref_pairs) == len(facts.promotable_candidates) == 1
    assert facts.promotable_candidate_count() == 2

    result = process_structure(source)

    assert len(disulfide_atom_ref_pairs(result.structure)) == 2
    assert {
        bond.provenance
        for bond in result.structure.topology.bonds
        if bond.relationship_type is BondRelationshipType.DISULFIDE
    } == {BondProvenance.SOURCE_EXPLICIT, BondProvenance.EVIDENCE_RESOLVED}


def test_stale_source_resolution_is_not_executable():
    source = _source_pair()
    action = DisulfideTopologyResolutionTransformer(
        source_atom_ref_pairs=StructureDisulfideTopologyFacts.from_structure(
            source
        ).source_atom_ref_pairs
    )
    context = TransformerExecutionContext(
        component_library=build_default_component_library(),
        original_structure=source,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    )
    assert action.accepts_projected_domain(
        ProjectedDomainState(scope=action.workflow_scope, state=source), context=context
    )
    changed = _source_pair(charge=-1)
    assert not action.accepts_projected_domain(
        ProjectedDomainState(scope=action.workflow_scope, state=changed),
        context=context,
    )


@pytest.mark.parametrize("order", (None, 1))
@pytest.mark.parametrize("reverse_slots", (False, True))
def test_source_resolution_preserves_reported_order_and_slot_independence(
    order, reverse_slots
):
    source = _source_pair()
    text = write_structure_string(source, FileFormat.MMCIF)
    source = read_structure_string(text, FileFormat.MMCIF)
    link = next(
        b
        for b in source.topology.bonds
        if b.relationship_type is BondRelationshipType.UNKNOWN
    )
    assert link.source_metadata is not None
    link = replace(
        link,
        order=order,
        source_metadata=replace(link.source_metadata, reported_order=order),
    )
    source = with_bonds(
        source,
        tuple(
            link if b.endpoint_pair() == link.endpoint_pair() else b
            for b in source.topology.bonds
        ),
    )
    if reverse_slots:
        for residue in tuple(source.constitution.chains[0].residues):
            source = source.with_updated_residue_facets(
                replace(residue, atom_sites=tuple(reversed(residue.atom_sites))),
                residue_geometry=source.residue_geometry(
                    source.constitution.residue_index(residue.residue_id)
                ),
                formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(
                    source.constitution.residue_index(residue.residue_id)
                ),
            )

    result = process_structure(source)

    assert len(disulfide_atom_ref_pairs(result.structure)) == 1
    assert_original_preserved(source, result.structure)


@pytest.mark.parametrize(
    "mutation",
    (
        "charge",
        "thiol-d",
        "extra-internal",
        "order",
        "aromatic",
        "reported-order",
        "inferred",
        "wrong-element",
        "competing-partner",
        "carbon-reported-order",
    ),
)
def test_unsupported_source_sulfur_graph_is_not_promoted(mutation):
    source = _source_pair(
        distance=2.05,
        charge=-1 if mutation == "charge" else None,
        extra="DG"
        if mutation == "thiol-d"
        else "CX"
        if mutation == "extra-internal"
        else None,
        sulfur_element="O" if mutation == "wrong-element" else "S",
    )
    link = next(
        b
        for b in source.topology.bonds
        if b.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    if mutation in {"order", "aromatic", "reported-order", "inferred"}:
        metadata = link.source_metadata
        assert metadata is not None
        link = replace(
            link,
            order=2 if mutation == "order" else link.order,
            aromatic=mutation == "aromatic",
            provenance=BondProvenance.TEMPLATE_RESOLVED
            if mutation == "inferred"
            else link.provenance,
            source_metadata=None
            if mutation == "inferred"
            else replace(metadata, reported_order=2)
            if mutation == "reported-order"
            else link.source_metadata,
        )
        source = with_bonds(
            source,
            tuple(
                link if b.endpoint_pair() == link.endpoint_pair() else b
                for b in source.topology.bonds
            ),
        )
    if mutation == "competing-partner":
        c = source.constitution
        source = with_bonds(
            source,
            (
                *source.topology.bonds,
                TopologyBond(
                    c.atom_index(AtomRef(ResidueId("A", 1), "SG")),
                    c.atom_index(AtomRef(ResidueId("A", 2), "CB")),
                    relationship_type=BondRelationshipType.UNKNOWN,
                ),
            ),
        )
    if mutation == "carbon-reported-order":
        carbon_bond = source.topology.bonds[0]
        carbon_bond = replace(
            carbon_bond,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT, reported_order=2
            ),
        )
        source = with_bonds(source, (carbon_bond, *source.topology.bonds[1:]))

    assert not StructureDisulfideTopologyFacts.from_structure(
        source
    ).has_promotable_candidates()
    assert not disulfide_atom_ref_pairs(process_structure(source).structure)


def test_conect_only_bpti_completes_hydrogens_and_roundtrips():
    text = (Path(__file__).parents[1] / "fixtures/corpus/pdb1bpi.ent").read_text()
    source = read_structure_string(
        "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith(("SSBOND", "HETATM"))
        ),
        FileFormat.PDB,
    )
    assert not disulfide_atom_ref_pairs(source)

    result = process_structure(
        source,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )

    assert result.requested_goals_fully_satisfied() is True
    assert len(disulfide_atom_ref_pairs(result.structure)) == 3
    assert all(
        not residue.has_atom_site("HG")
        for residue in result.structure.constitution.chains[0].residues
        if residue.component_id == "CYS"
    )
    assert_original_preserved(source, result.structure)
    assert (
        Chem.MolFromPDBBlock(
            write_structure_string(result.structure, FileFormat.PDB),
            sanitize=True,
            removeHs=False,
            proximityBonding=True,
        )
        is not None
    )
    for fmt in (FileFormat.PDB, FileFormat.MMCIF):
        restored = read_structure_string(
            write_structure_string(result.structure, fmt), fmt
        )
        assert len(disulfide_atom_ref_pairs(restored)) == 3
