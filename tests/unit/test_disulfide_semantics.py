"""Topology-authoritative disulfide semantics across structure consumers."""

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics import (
    ClashPolicy,
    bind_clash_detection_context,
    prepare_clash_detection_basis,
)
from protrepair.diagnostics.clashes import StericClash
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.diagnostics.near_covalent import detect_near_covalent_contacts
from protrepair.diagnostics.topology import (
    AmbiguousDisulfideFinding,
    DisulfideCandidate,
    LikelyDisulfideBond,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state.hydrogen_expectation import (
    derive_structure_hydrogen_expectation_model,
)
from protrepair.state.structure_topology import (
    DisulfideTopologyConflict,
    DisulfideTopologyConflictReason,
    StructureDisulfideTopologyFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.disulfide import (
    disulfide_atom_ref_pairs,
    disulfide_bonded_cysteine_residue_ids,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.continuous.bonds import inter_residue_bonds
from protrepair.workflow.actions.disulfide_topology import (
    DisulfideTopologyResolutionTransformer,
)
from protrepair.workflow.contracts import RequestedGoalSet, WorkflowTransformRequests
from protrepair.workflow.planning.planner import plan_workflow_actions


@pytest.mark.parametrize(
    "relationship_type",
    (BondRelationshipType.COVALENT, BondRelationshipType.DISULFIDE),
)
def test_covalent_like_cysteine_sg_topology_defines_disulfide_chemistry(
    relationship_type: BondRelationshipType,
) -> None:
    """Generic and specialized covalent CYS-SG bonds share disulfide semantics."""

    structure = sg_pair_structure(
        distance_angstrom=8.0,
        relationship_type=relationship_type,
    )
    left_id = ResidueId("A", 1, "A")
    right_id = ResidueId("B", 1)

    assert disulfide_atom_ref_pairs(structure) == frozenset(
        {
            (
                AtomRef(left_id, "SG"),
                AtomRef(right_id, "SG"),
            )
        }
    )
    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset(
        {left_id, right_id}
    )


@pytest.mark.parametrize(
    "relationship_type",
    (None, BondRelationshipType.UNKNOWN, BondRelationshipType.METAL_COORDINATION),
)
def test_close_cysteine_sg_geometry_does_not_create_disulfide_truth(
    relationship_type: BondRelationshipType | None,
) -> None:
    """Proximity and noncovalent topology must not define a CYS microstate."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=relationship_type,
    )

    assert disulfide_atom_ref_pairs(structure) == frozenset()
    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset()


@pytest.mark.parametrize("provenance", tuple(BondProvenance))
def test_disulfide_chemistry_does_not_depend_on_bond_provenance(
    provenance: BondProvenance,
) -> None:
    """Physical relationship and evidence provenance must remain orthogonal."""

    structure = sg_pair_structure(
        distance_angstrom=8.0,
        relationship_type=BondRelationshipType.DISULFIDE,
        provenance=provenance,
    )

    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset(
        {ResidueId("A", 1, "A"), ResidueId("B", 1)}
    )


def test_atom_name_alone_does_not_make_non_cysteine_sg_a_disulfide() -> None:
    """A covalent SG-SG relationship requires CYS identity at both endpoints."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=BondRelationshipType.DISULFIDE,
        right_component_id="SER",
    )

    assert disulfide_atom_ref_pairs(structure) == frozenset()
    assert disulfide_bonded_cysteine_residue_ids(structure) == frozenset()


def test_clash_basis_rebinds_disulfide_truth_from_current_topology() -> None:
    """Reusable clash bases must not cache topology from their source structure."""

    unbonded = sg_pair_structure(distance_angstrom=2.0)
    bonded = with_sg_relationship(unbonded, BondRelationshipType.DISULFIDE)
    component_library = build_default_component_library()
    basis = prepare_clash_detection_basis(
        unbonded,
        component_library=component_library,
    )

    unbonded_report = bind_clash_detection_context(
        unbonded,
        basis=basis,
    ).detect_clashes()
    bonded_report = bind_clash_detection_context(
        bonded,
        basis=basis,
    ).detect_clashes()

    assert has_sg_sg_clash(unbonded_report.clashes)
    assert not has_sg_sg_clash(bonded_report.clashes)


@pytest.mark.parametrize(
    ("relationship_type", "expect_near_covalent", "expect_disulfide"),
    (
        (None, True, False),
        (BondRelationshipType.UNKNOWN, True, False),
        (BondRelationshipType.METAL_COORDINATION, False, False),
        (BondRelationshipType.COVALENT, False, True),
        (BondRelationshipType.DISULFIDE, False, True),
    ),
)
def test_near_covalent_contact_and_disulfide_chemistry_remain_orthogonal(
    relationship_type: BondRelationshipType | None,
    expect_near_covalent: bool,
    expect_disulfide: bool,
) -> None:
    """Expected close contact and CYS microstate are independent projections."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=relationship_type,
    )
    contacts = detect_near_covalent_contacts(
        structure,
        component_library=build_default_component_library(),
        pair_policy=ClashPolicy(),
    )

    assert (
        any(
            {contact.left_atom_name, contact.right_atom_name} == {"SG"}
            for contact in contacts
        )
        is expect_near_covalent
    )
    assert bool(disulfide_bonded_cysteine_residue_ids(structure)) is expect_disulfide


@pytest.mark.parametrize(
    ("relationship_type", "expect_hg"),
    (
        (None, True),
        (BondRelationshipType.UNKNOWN, True),
        (BondRelationshipType.METAL_COORDINATION, True),
        (BondRelationshipType.COVALENT, False),
        (BondRelationshipType.DISULFIDE, False),
    ),
)
def test_hydrogen_expectation_consumes_topology_disulfide_semantics(
    relationship_type: BondRelationshipType | None,
    expect_hg: bool,
) -> None:
    """CYS HG expectation must not depend on SG distance alone."""

    structure = sg_pair_structure(
        distance_angstrom=2.0,
        relationship_type=relationship_type,
    )
    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=build_default_component_library(),
    )

    for residue_id in (ResidueId("A", 1, "A"), ResidueId("B", 1)):
        expected_names = model.expected_hydrogen_atom_names_by_residue[residue_id]
        assert ("HG" in expected_names) is expect_hg


def test_retained_cysteine_disulfide_uses_shared_hydrogen_semantics() -> None:
    """Retained CYS expectation and completion must honor canonical disulfides."""

    left_id = ResidueId("L", 1)
    right_id = ResidueId("R", 1)
    structure = retained_cys_pair_structure(left_id, right_id)
    component_library = build_default_component_library()

    expectation = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=component_library,
    )
    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
    )

    for residue_id in (left_id, right_id):
        assert "HG" not in expectation.expected_hydrogen_atom_names_by_residue[
            residue_id
        ]
        residue = result.structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None and not residue.has_atom_site("HG")

    bond = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(AtomRef(left_id, "SG")),
        result.structure.constitution.atom_index(AtomRef(right_id, "SG")),
    )
    assert bond is not None
    assert bond.relationship_type is BondRelationshipType.DISULFIDE


@pytest.mark.parametrize("chemistry_mode", ("override", "fallback"))
def test_retained_disulfide_excludes_sg_h_across_chemistry_sources(
    chemistry_mode: str,
) -> None:
    """Explicit evidence and fallback must not protonate a bonded CYS SG."""

    left_id = ResidueId("L", 1)
    right_id = ResidueId("R", 1)
    structure = retained_sulfide_pair_structure(left_id, right_id)
    chemistry_evidence = (
        tuple(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="CS",
                heavy_atom_names=("C1", "SG"),
            ).to_evidence()
            for residue_id in (left_id, right_id)
        )
        if chemistry_mode == "override"
        else ()
    )
    component_library = ComponentLibrary(templates={})

    expectation = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=component_library,
        retained_non_polymer_chemistry_evidence=chemistry_evidence,
    )
    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
        chemistry_evidence=chemistry_evidence,
    )

    for residue_id in (left_id, right_id):
        resolution = expectation.resolution_for_retained_non_polymer(residue_id)
        assert all(
            "SG" not in (bond.atom_name_1, bond.atom_name_2)
            for bond in resolution.hydrogen_bond_definitions
        )
        assert not has_hydrogen_topology_bond_to(
            result.structure,
            AtomRef(residue_id, "SG"),
        )


def test_unique_disulfide_geometry_is_promotable_but_not_topology_truth() -> None:
    """A unique short SG pair should remain evidence until transformed."""

    structure = sg_pair_structure(distance_angstrom=2.1)
    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert facts.promotable_candidates == (
        LikelyDisulfideBond(
            left_residue_id=ResidueId("A", 1, "A"),
            right_residue_id=ResidueId("B", 1),
            sg_distance_angstrom=2.1,
        ),
    )
    assert not structure.topology.bonds
    assert not inter_residue_bonds(ProteinStructureSnapshot.from_structure(structure))


@pytest.mark.parametrize(
    "relationship_type",
    (
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.METAL_COORDINATION,
        BondRelationshipType.HYDROGEN_BOND,
    ),
)
def test_existing_non_covalent_pair_blocks_disulfide_promotion(
    relationship_type: BondRelationshipType,
) -> None:
    """Existing relationship truth must not be silently overwritten by geometry."""

    structure = sg_pair_structure(
        distance_angstrom=2.1,
        relationship_type=relationship_type,
    )
    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert not facts.promotable_candidates
    assert len(facts.conflicts) == 1
    assert (
        facts.conflicts[0].reason
        is DisulfideTopologyConflictReason.EXISTING_PAIR_RELATIONSHIP
    )


def test_existing_covalent_pair_is_already_resolved_not_promotable() -> None:
    """Canonical covalent-like truth should make geometry evidence idempotent."""

    structure = sg_pair_structure(
        distance_angstrom=2.1,
        relationship_type=BondRelationshipType.COVALENT,
    )
    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert not facts.promotable_candidates
    assert not facts.conflicts


def test_other_inter_residue_covalent_partner_blocks_disulfide_promotion() -> None:
    """One sulfur cannot acquire a second covalent inter-residue partner."""

    structure = three_cysteine_structure((0.0, 2.1, 8.0))
    structure = with_sg_relationship(
        structure,
        BondRelationshipType.COVALENT,
        left_atom_ref=AtomRef(ResidueId("A", 1), "SG"),
        right_atom_ref=AtomRef(ResidueId("C", 1), "SG"),
    )
    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert not facts.promotable_candidates
    assert len(facts.conflicts) == 1
    assert (
        facts.conflicts[0].reason
        is DisulfideTopologyConflictReason.ENDPOINT_HAS_OTHER_COVALENT_PARTNER
    )


def test_ambiguous_three_cysteine_neighborhood_is_not_promotable() -> None:
    """Competing short partners should remain ambiguous diagnostic evidence."""

    facts = StructureDisulfideTopologyFacts.from_structure(
        three_cysteine_structure((0.0, 2.1, 2.8))
    )

    assert not facts.promotable_candidates
    assert not facts.conflicts
    assert facts.ambiguous_findings


def test_retained_cysteine_geometry_is_promotable() -> None:
    """Retained CYS uses the same evidence-resolution contract as polymer CYS."""

    structure = retained_close_cysteine_structure()
    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert len(facts.promotable_candidates) == 1
    assert facts.promotable_candidates[0].residue_pair() == (
        ResidueId("L", 1),
        ResidueId("R", 1),
    )


def test_overlapping_cysteine_sulfurs_are_not_disulfide_evidence() -> None:
    """A zero-length SG contact is invalid geometry, not bond evidence."""

    facts = StructureDisulfideTopologyFacts.from_structure(
        sg_pair_structure(distance_angstrom=0.0)
    )

    assert not facts.promotable_candidates
    assert not facts.ambiguous_findings


@pytest.mark.parametrize(
    ("distance_angstrom", "is_promotable"),
    ((2.5, True), (2.500_001, False)),
)
def test_likely_disulfide_cutoff_is_inclusive_only_at_2_5_angstrom(
    distance_angstrom: float,
    is_promotable: bool,
) -> None:
    """The literature-backed candidate cutoff should have an explicit boundary."""

    facts = StructureDisulfideTopologyFacts.from_structure(
        sg_pair_structure(distance_angstrom=distance_angstrom)
    )

    assert bool(facts.promotable_candidates) is is_promotable


def test_ambiguity_cutoff_is_inclusive_at_3_angstrom() -> None:
    """A sulfur with two partners at the permissive boundary stays unresolved."""

    facts = StructureDisulfideTopologyFacts.from_structure(
        three_cysteine_structure((0.0, 2.5, 5.5))
    )

    assert not facts.promotable_candidates
    assert tuple(finding.residue_id for finding in facts.ambiguous_findings) == (
        ResidueId("B", 1),
    )


def test_candidate_cutoff_does_not_override_existing_relationship() -> None:
    """Boundary-distance evidence still yields to canonical relationship truth."""

    facts = StructureDisulfideTopologyFacts.from_structure(
        sg_pair_structure(
            distance_angstrom=2.5,
            relationship_type=BondRelationshipType.UNKNOWN,
        )
    )

    assert not facts.promotable_candidates
    assert len(facts.conflicts) == 1


def test_disulfide_candidate_order_is_independent_of_constitution_order() -> None:
    """Candidate identity should use canonical residue ordering, not slot order."""

    later_id = ResidueId("Z", 9)
    earlier_id = ResidueId("A", -1, "B")
    facts = StructureDisulfideTopologyFacts.from_structure(
        cysteine_sulfur_structure(((later_id, 0.0), (earlier_id, 2.1)))
    )

    assert facts.promotable_candidates[0].residue_pair() == (
        earlier_id,
        later_id,
    )


def test_promotable_fact_batch_rejects_shared_cysteine_endpoints() -> None:
    """A topology-resolution batch cannot assign one sulfur twice."""

    structure = three_cysteine_structure((0.0, 2.1, 8.0))
    shared_id = ResidueId("A", 1)

    with pytest.raises(ValueError, match="share endpoints"):
        StructureDisulfideTopologyFacts(
            carrier=structure,
            promotable_candidates=(
                LikelyDisulfideBond(shared_id, ResidueId("B", 1), 2.1),
                LikelyDisulfideBond(shared_id, ResidueId("C", 1), 2.2),
            ),
            conflicts=(),
            ambiguous_findings=(),
        )


def test_resolved_pair_does_not_hide_independent_promotable_pair() -> None:
    """Resolution should remain local when another independent pair is pending."""

    structure = cysteine_sulfur_structure(
        (
            (ResidueId("A", 1), 0.0),
            (ResidueId("B", 1), 2.1),
            (ResidueId("C", 1), 10.0),
            (ResidueId("D", 1), 12.1),
        )
    )
    structure = with_sg_relationship(
        structure,
        BondRelationshipType.DISULFIDE,
        left_atom_ref=AtomRef(ResidueId("A", 1), "SG"),
        right_atom_ref=AtomRef(ResidueId("B", 1), "SG"),
    )

    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert tuple(
        candidate.residue_pair() for candidate in facts.promotable_candidates
    ) == ((ResidueId("C", 1), ResidueId("D", 1)),)


def test_mixed_polymer_retained_cysteine_pair_is_promotable() -> None:
    """Constitution placement must not split the disulfide evidence contract."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId("A", 1),
                        atoms=(atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="CYS",
                residue_id=ResidueId("L", 1),
                atoms=(atom_payload("SG", "S", Vec3(2.1, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="mixed-cysteine-pair",
    )

    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert len(facts.promotable_candidates) == 1


def test_intra_residue_cysteine_bond_does_not_block_disulfide_promotion() -> None:
    """The normal CYS SG-CB bond is not a competing inter-residue partner."""

    structure = cysteine_pair_with_intra_residue_sg_bond()

    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert len(facts.promotable_candidates) == 1
    assert not facts.conflicts


def test_disulfide_conflict_scope_includes_competing_partner() -> None:
    """Conflict facts should identify candidate and existing-bond residues."""

    structure = three_cysteine_structure((0.0, 2.1, 8.0))
    structure = with_sg_relationship(
        structure,
        BondRelationshipType.COVALENT,
        left_atom_ref=AtomRef(ResidueId("A", 1), "SG"),
        right_atom_ref=AtomRef(ResidueId("C", 1), "SG"),
    )

    conflict = StructureDisulfideTopologyFacts.from_structure(structure).conflicts[0]

    assert set(conflict.residue_ids()) == {
        ResidueId("A", 1),
        ResidueId("B", 1),
        ResidueId("C", 1),
    }


def test_topology_facts_reject_promotable_conflict_overlap() -> None:
    """One evidence pair cannot occupy two resolution dispositions."""

    structure = sg_pair_structure(
        distance_angstrom=2.1,
        relationship_type=BondRelationshipType.UNKNOWN,
    )
    conflict = StructureDisulfideTopologyFacts.from_structure(structure).conflicts[0]

    with pytest.raises(ValueError, match="both promotable and conflicting"):
        StructureDisulfideTopologyFacts(
            carrier=structure,
            promotable_candidates=(conflict.candidate,),
            conflicts=(
                DisulfideTopologyConflict(
                    candidate=conflict.candidate,
                    reason=conflict.reason,
                    conflicting_atom_ref_pairs=conflict.conflicting_atom_ref_pairs,
                ),
            ),
            ambiguous_findings=(),
        )


def test_disulfide_action_canonicalizes_candidate_order() -> None:
    """Atomic action identity should not depend on proposal enumeration order."""

    earlier = LikelyDisulfideBond(ResidueId("A", 1), ResidueId("B", 1), 2.1)
    later = LikelyDisulfideBond(ResidueId("C", 1), ResidueId("D", 1), 2.1)

    action = DisulfideTopologyResolutionTransformer(candidates=(later, earlier))

    assert action.candidates == (earlier, later)


def test_disulfide_action_rejects_duplicate_candidates() -> None:
    """Repeated evidence cannot produce repeated topology writes or events."""

    candidate = LikelyDisulfideBond(
        ResidueId("A", 1), ResidueId("B", 1), 2.1
    )

    with pytest.raises(ValueError, match="must not repeat"):
        DisulfideTopologyResolutionTransformer(candidates=(candidate, candidate))


def test_disulfide_action_rejects_shared_endpoints() -> None:
    """Direct action construction must preserve sulfur valence independently."""

    with pytest.raises(ValueError, match="share endpoints"):
        DisulfideTopologyResolutionTransformer(
            candidates=(
                LikelyDisulfideBond(
                    ResidueId("A", 1), ResidueId("B", 1), 2.1
                ),
                LikelyDisulfideBond(
                    ResidueId("A", 1), ResidueId("C", 1), 2.2
                ),
            )
        )


def test_far_source_disulfide_remains_execution_topology() -> None:
    """Canonical bond truth must not be invalidated by non-ideal coordinates."""

    structure = sg_pair_structure(
        distance_angstrom=8.0,
        relationship_type=BondRelationshipType.DISULFIDE,
    )

    facts = StructureDisulfideTopologyFacts.from_structure(structure)
    planned_bonds = inter_residue_bonds(
        ProteinStructureSnapshot.from_structure(structure)
    )

    assert not facts.promotable_candidates
    assert len(planned_bonds) == 1


def test_non_cysteine_sulfur_pair_is_not_disulfide_evidence() -> None:
    """Atom naming alone must not reinterpret another component as CYS."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="CYX",
                residue_id=ResidueId("L", 1),
                atoms=(atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
            residue_payload(
                component_id="CYS",
                residue_id=ResidueId("R", 1),
                atoms=(atom_payload("SG", "S", Vec3(2.1, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="non-cysteine-sulfur-pair",
    )

    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert not facts.promotable_candidates


def test_cysteine_without_sg_is_not_disulfide_evidence() -> None:
    """Incomplete cysteine sidechains must be repaired before topology inference."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="CYS",
                residue_id=ResidueId("L", 1),
                atoms=(atom_payload("CB", "C", Vec3(0.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
            residue_payload(
                component_id="CYS",
                residue_id=ResidueId("R", 1),
                atoms=(atom_payload("SG", "S", Vec3(2.1, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="missing-cysteine-sulfur",
    )

    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert not facts.promotable_candidates


def test_ambiguous_cluster_does_not_hide_independent_disulfide() -> None:
    """Ambiguity should block only its connected candidate component."""

    structure = cysteine_sulfur_structure(
        (
            (ResidueId("A", 1), 0.0),
            (ResidueId("B", 1), 2.1),
            (ResidueId("C", 1), 5.1),
            (ResidueId("D", 1), 20.0),
            (ResidueId("E", 1), 22.1),
        )
    )

    facts = StructureDisulfideTopologyFacts.from_structure(structure)

    assert tuple(
        candidate.residue_pair() for candidate in facts.promotable_candidates
    ) == ((ResidueId("D", 1), ResidueId("E", 1)),)
    assert tuple(finding.residue_id for finding in facts.ambiguous_findings) == (
        ResidueId("B", 1),
    )


def test_workflow_promotes_three_independent_disulfides_atomically() -> None:
    """Atomic resolution should scale without factorial pair permutations."""

    structure = cysteine_sulfur_structure(
        (
            (ResidueId("A", 1), 0.0),
            (ResidueId("B", 1), 2.1),
            (ResidueId("C", 1), 10.0),
            (ResidueId("D", 1), 12.1),
            (ResidueId("E", 1), 20.0),
            (ResidueId("F", 1), 22.1),
        )
    )

    planning = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )

    assert len(planning.transformers) == 1
    state_deficit = planning.state_deficit
    assert state_deficit is not None
    assert state_deficit.topology_resolution is not None
    assert state_deficit.topology_resolution.promotable_disulfide_count == 3
    action = planning.transformers[0]
    assert isinstance(action, DisulfideTopologyResolutionTransformer)
    assert len(action.candidates) == 3


def test_ambiguous_disulfide_findings_are_identity_sorted() -> None:
    """Ambiguity diagnostics should be stable across constitution order."""

    facts = StructureDisulfideTopologyFacts.from_structure(
        cysteine_sulfur_structure(
            (
                (ResidueId("C", 1), 0.0),
                (ResidueId("A", 1), 1.0),
                (ResidueId("B", 1), 2.0),
            )
        )
    )

    assert tuple(finding.residue_id for finding in facts.ambiguous_findings) == (
        ResidueId("A", 1),
        ResidueId("B", 1),
        ResidueId("C", 1),
    )
    assert all(
        tuple(candidate.residue_id for candidate in finding.candidates)
        == tuple(
            sorted(candidate.residue_id for candidate in finding.candidates)
        )
        for finding in facts.ambiguous_findings
    )


def test_workflow_resolves_independent_pair_while_preserving_conflict() -> None:
    """A local topology conflict must not suppress independent progress."""

    structure = cysteine_sulfur_structure(
        (
            (ResidueId("A", 1), 0.0),
            (ResidueId("B", 1), 2.1),
            (ResidueId("C", 1), 10.0),
            (ResidueId("D", 1), 12.1),
        )
    )
    structure = with_sg_relationship(
        structure,
        BondRelationshipType.UNKNOWN,
        left_atom_ref=AtomRef(ResidueId("A", 1), "SG"),
        right_atom_ref=AtomRef(ResidueId("B", 1), "SG"),
    )

    result = process_structure(structure)

    assert len(result.structure.topology.bonds) == 2
    assert any(
        bond.relationship_type is BondRelationshipType.DISULFIDE
        and bond.provenance is BondProvenance.EVIDENCE_RESOLVED
        for bond in result.structure.topology.bonds
    )
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for issue in result.issues
    )


def test_conflict_records_all_competing_covalent_partners() -> None:
    """Conflict evidence must not collapse multiple existing partners."""

    structure = cysteine_sulfur_structure(
        (
            (ResidueId("A", 1), 0.0),
            (ResidueId("B", 1), 2.1),
            (ResidueId("C", 1), 8.0),
            (ResidueId("D", 1), 12.0),
        )
    )
    a_sg = structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "SG"))
    competing_bonds = tuple(
        TopologyBond(
            atom_index_1=a_sg,
            atom_index_2=structure.constitution.atom_index(
                AtomRef(partner_id, "SG")
            ),
            relationship_type=BondRelationshipType.COVALENT,
            provenance=BondProvenance.SOURCE_EXPLICIT,
        )
        for partner_id in (ResidueId("C", 1), ResidueId("D", 1))
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=competing_bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )

    conflict = StructureDisulfideTopologyFacts.from_structure(structure).conflicts[0]

    assert len(conflict.conflicting_atom_ref_pairs) == 2
    assert set(conflict.residue_ids()) == {
        ResidueId("A", 1),
        ResidueId("B", 1),
        ResidueId("C", 1),
        ResidueId("D", 1),
    }


def test_source_disulfide_does_not_emit_topology_contradiction() -> None:
    """Matching source truth should remain resolved even at non-ideal distance."""

    structure = sg_pair_structure(
        distance_angstrom=2.1,
        relationship_type=BondRelationshipType.DISULFIDE,
    )

    result = process_structure(structure)

    assert not any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for issue in result.issues
    )


@pytest.mark.parametrize("distance_angstrom", (0.0, float("nan"), float("inf")))
def test_disulfide_evidence_rejects_nonphysical_distances(
    distance_angstrom: float,
) -> None:
    """Evidence values must be finite and positive before entering state facts."""

    with pytest.raises(ValueError, match="finite positive distance"):
        LikelyDisulfideBond(
            ResidueId("A", 1),
            ResidueId("B", 1),
            distance_angstrom,
        )


def test_ambiguous_disulfide_requires_multiple_partners() -> None:
    """A one-partner observation is not an ambiguity fact."""

    with pytest.raises(ValueError, match="multiple candidates"):
        AmbiguousDisulfideFinding(
            residue_id=ResidueId("A", 1),
            candidates=(DisulfideCandidate(ResidueId("B", 1), 2.1),),
        )


def test_likely_disulfide_rejects_self_pair() -> None:
    """One cysteine cannot be both endpoints of a disulfide candidate."""

    residue_id = ResidueId("A", 1)
    with pytest.raises(ValueError, match="two residues"):
        LikelyDisulfideBond(residue_id, residue_id, 2.1)


def test_reprocessing_promoted_disulfide_is_idempotent() -> None:
    """A second workflow pass must not duplicate topology or repair events."""

    first = process_structure(sg_pair_structure(distance_angstrom=2.1))
    second = process_structure(first.structure)

    assert second.structure.topology.bonds == first.structure.topology.bonds
    assert not any(
        repair.kind is RepairEventKind.DISULFIDE_TOPOLOGY_RESOLVED
        for repair in second.repairs
    )


def test_disulfide_topology_action_preserves_geometry_facet() -> None:
    """Topology resolution must not smuggle coordinate mutation into the action."""

    structure = sg_pair_structure(distance_angstrom=2.1)

    result = process_structure(structure)

    assert result.structure.geometry is structure.geometry
    assert result.structure.constitution is structure.constitution


def test_promoted_insertion_code_endpoint_roundtrips_through_pdb() -> None:
    """PDB CONECT projection must retain insertion-qualified residue identity."""

    result = process_structure(two_disulfide_pair_structure())
    roundtripped = read_structure_string(
        write_structure_string(result.structure, FileFormat.PDB),
        FileFormat.PDB,
    )
    endpoint_pairs = {
        tuple(
            sorted(
                (
                    roundtripped.constitution.atom_ref_at(bond.atom_index_1).residue_id,
                    roundtripped.constitution.atom_ref_at(bond.atom_index_2).residue_id,
                )
            )
        )
        for bond in roundtripped.topology.bonds
    }

    assert (ResidueId("A", 1, "A"), ResidueId("B", 1)) in endpoint_pairs


def test_workflow_promotes_independent_disulfides_atomically_and_idempotently() -> (
    None
):
    """Planning should resolve all independent candidates in one topology action."""

    structure = two_disulfide_pair_structure()
    planning = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )

    assert len(planning.transformers) == 1
    action = planning.transformers[0]
    assert isinstance(action, DisulfideTopologyResolutionTransformer)
    assert len(action.candidates) == 2

    result = process_structure(structure)
    evidence_bonds = tuple(
        bond
        for bond in result.structure.topology.bonds
        if bond.relationship_type is BondRelationshipType.DISULFIDE
        and bond.provenance is BondProvenance.EVIDENCE_RESOLVED
    )
    assert len(evidence_bonds) == 2
    assert sum(
        repair.kind is RepairEventKind.DISULFIDE_TOPOLOGY_RESOLVED
        for repair in result.repairs
    ) == 2
    assert len(
        inter_residue_bonds(
            ProteinStructureSnapshot.from_structure(result.structure)
        )
    ) == 2

    replanning = plan_workflow_actions(
        result.structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )
    assert not any(
        isinstance(transformer, DisulfideTopologyResolutionTransformer)
        for transformer in replanning.transformers
    )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_promoted_disulfides_project_through_egress(
    file_format: FileFormat,
) -> None:
    """Evidence-resolved bonds must reach both supported boundary formats."""

    result = process_structure(two_disulfide_pair_structure())
    serialized = write_structure_string(result.structure, file_format)
    roundtripped = read_structure_string(serialized, file_format)

    assert {
        tuple(
            sorted(
                (
                    roundtripped.constitution.atom_ref_at(bond.atom_index_1).residue_id,
                    roundtripped.constitution.atom_ref_at(bond.atom_index_2).residue_id,
                )
            )
        )
        for bond in roundtripped.topology.bonds
    } == {
        (ResidueId("A", 1, "A"), ResidueId("B", 1)),
        (ResidueId("C", 1), ResidueId("D", 1)),
    }
    if file_format is FileFormat.PDB:
        assert sum(
            line.startswith("CONECT") for line in serialized.splitlines()
        ) == 4
    else:
        assert serialized.count(" disulf ") == 2


def test_workflow_preserves_conflicting_topology_and_reports_contradiction() -> None:
    """Conflicting canonical relationships should survive with a typed issue."""

    structure = sg_pair_structure(
        distance_angstrom=2.1,
        relationship_type=BondRelationshipType.UNKNOWN,
    )
    result = process_structure(structure)

    assert result.structure.topology.bonds == structure.topology.bonds
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        and set(issue.scope.residue_ids)
        == {ResidueId("A", 1, "A"), ResidueId("B", 1)}
        for issue in result.issues
    )


def sg_pair_structure(
    *,
    distance_angstrom: float,
    relationship_type: BondRelationshipType | None = None,
    left_component_id: str = "CYS",
    right_component_id: str = "CYS",
    provenance: BondProvenance = BondProvenance.SOURCE_EXPLICIT,
) -> ProteinStructure:
    """Return a cross-chain CYS pair with one insertion-coded endpoint."""

    left_id = ResidueId("A", 1, "A")
    right_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id=left_component_id,
                        residue_id=left_id,
                        atoms=(atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id=right_component_id,
                        residue_id=right_id,
                        atoms=(
                            atom_payload(
                                "SG",
                                "S",
                                Vec3(distance_angstrom, 0.0, 0.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="topology-authoritative-disulfide",
    )
    if relationship_type is None:
        return structure

    return with_sg_relationship(
        structure,
        relationship_type,
        provenance=provenance,
    )


def retained_cys_pair_structure(
    left_id: ResidueId,
    right_id: ResidueId,
) -> ProteinStructure:
    """Return two retained CYS components with an explicit SG-SG bond."""

    structure = build_structure(
        chains=(),
        ligands=(
            retained_cys_payload(left_id, x_offset=0.0),
            retained_cys_payload(right_id, x_offset=8.0),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-cys-disulfide",
    )
    return with_sg_relationship(
        structure,
        BondRelationshipType.DISULFIDE,
        left_atom_ref=AtomRef(left_id, "SG"),
        right_atom_ref=AtomRef(right_id, "SG"),
    )


def retained_close_cysteine_structure() -> ProteinStructure:
    """Return two close retained CYS SG sites without canonical topology."""

    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="CYS",
                residue_id=ResidueId("L", 1),
                atoms=(atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
            residue_payload(
                component_id="CYS",
                residue_id=ResidueId("R", 1),
                atoms=(atom_payload("SG", "S", Vec3(2.1, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-close-cysteine-pair",
    )


def three_cysteine_structure(
    x_coordinates: tuple[float, float, float],
) -> ProteinStructure:
    """Return three cross-chain CYS SG sites at selected x coordinates."""

    return cysteine_sulfur_structure(
        tuple(
            (ResidueId(chain_id, 1), x_coordinate)
            for chain_id, x_coordinate in zip(
                ("A", "B", "C"),
                x_coordinates,
                strict=True,
            )
        )
    )


def two_disulfide_pair_structure() -> ProteinStructure:
    """Return two independent likely cross-chain disulfide pairs."""

    return cysteine_sulfur_structure(
        (
            (ResidueId("A", 1, "A"), 0.0),
            (ResidueId("B", 1), 2.1),
            (ResidueId("C", 1), 10.0),
            (ResidueId("D", 1), 12.1),
        )
    )


def cysteine_sulfur_structure(
    residue_coordinates: tuple[tuple[ResidueId, float], ...],
) -> ProteinStructure:
    """Return one-residue polymer chains containing only CYS SG sites."""

    return build_structure(
        chains=tuple(
            chain_payload(
                residue_id.chain_id,
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("SG", "S", Vec3(x_coordinate, 0.0, 0.0)),
                        ),
                    ),
                ),
            )
            for residue_id, x_coordinate in residue_coordinates
        ),
        source_format=FileFormat.PDB,
        source_name="cysteine-sulfur-structure",
    )


def cysteine_pair_with_intra_residue_sg_bond() -> ProteinStructure:
    """Return a close CYS pair with the first residue's normal SG-CB bond."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("CB", "C", Vec3(-1.8, 0.0, 0.0)),
                            atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId("B", 1),
                        atoms=(atom_payload("SG", "S", Vec3(2.1, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="cysteine-pair-with-local-sg-bond",
    )
    left_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "CB")
    )
    right_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "SG")
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=left_index,
                    atom_index_2=right_index,
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def retained_sulfide_pair_structure(
    left_id: ResidueId,
    right_id: ResidueId,
) -> ProteinStructure:
    """Return retained CYS-labelled methylthiol fragments joined at SG."""

    structure = build_structure(
        chains=(),
        ligands=tuple(
            residue_payload(
                component_id="CYS",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(x_offset, 0.0, 0.0)),
                    atom_payload("SG", "S", Vec3(x_offset + 1.8, 0.0, 0.0)),
                ),
                is_hetero=True,
            )
            for residue_id, x_offset in ((left_id, 0.0), (right_id, 8.0))
        ),
        source_format=FileFormat.PDB,
        source_name="retained-cys-sulfide",
    )
    return with_sg_relationship(
        structure,
        BondRelationshipType.DISULFIDE,
        left_atom_ref=AtomRef(left_id, "SG"),
        right_atom_ref=AtomRef(right_id, "SG"),
    )


def retained_cys_payload(
    residue_id: ResidueId,
    *,
    x_offset: float,
) -> CanonicalResiduePayload:
    """Return one retained CYS payload with nondegenerate heavy geometry."""

    return residue_payload(
        component_id="CYS",
        residue_id=residue_id,
        atoms=(
            atom_payload("N", "N", Vec3(x_offset - 1.2, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(x_offset, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(x_offset + 1.3, 0.3, 0.0)),
            atom_payload("O", "O", Vec3(x_offset + 2.0, 1.1, 0.0)),
            atom_payload("CB", "C", Vec3(x_offset, 1.4, 0.5)),
            atom_payload("SG", "S", Vec3(x_offset, 2.8, 0.7)),
        ),
        is_hetero=True,
    )


def with_sg_relationship(
    structure: ProteinStructure,
    relationship_type: BondRelationshipType,
    *,
    provenance: BondProvenance = BondProvenance.SOURCE_EXPLICIT,
    left_atom_ref: AtomRef | None = None,
    right_atom_ref: AtomRef | None = None,
) -> ProteinStructure:
    """Return the structure with one canonical relationship between CYS SG atoms."""

    left_index = structure.constitution.atom_index(
        left_atom_ref or AtomRef(ResidueId("A", 1, "A"), "SG")
    )
    right_index = structure.constitution.atom_index(
        right_atom_ref or AtomRef(ResidueId("B", 1), "SG")
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=left_index,
                    atom_index_2=right_index,
                    relationship_type=relationship_type,
                    provenance=provenance,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def has_sg_sg_clash(clashes: tuple[StericClash, ...]) -> bool:
    """Return whether one clash tuple contains a CYS SG-SG pair."""

    return any(
        clash.left_atom_name == "SG" and clash.right_atom_name == "SG"
        for clash in clashes
    )


def has_hydrogen_topology_bond_to(
    structure: ProteinStructure,
    atom_ref: AtomRef,
) -> bool:
    """Return whether topology bonds one hydrogen to the requested atom."""

    atom_index = structure.constitution.atom_index(atom_ref)
    return any(
        structure.constitution.atom_site_at(other_index).is_hydrogen()
        for bond in structure.topology.bonds
        if bond.involves(atom_index)
        for other_index in bond.endpoint_pair()
        if other_index != atom_index
    )
