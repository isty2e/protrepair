import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import build_default_restraint_library
from protrepair.diagnostics.kinds import IssueSeverity
from protrepair.diagnostics.parser_readability import (
    RDKitNoConectParserReadabilityProbe,
    RDKitParserProblemWitness,
    RDKitProximityBondWitness,
    prepare_rdkit_no_conect_known_bond_lookup,
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.state.structure_axes import (
    ParserCompatibilityProfile,
    ParserCompatibilityState,
)
from protrepair.state.structure_parser import StructureParserCompatibilityFacts
from protrepair.structure import ProteinStructure, StructureTopology
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
)
from protrepair.transformer.refinement.acceptance import (
    measure_refinement_acceptance_metrics,
    refinement_metrics_rejected,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import RequestedGoalReport, WorkflowPlanningContext
from protrepair.workflow.engine.reporting import evaluate_workflow_branch_quality_score
from protrepair.workflow.planning.assessment.deficits import (
    StructureParserCompatibilityDeficit,
    WorkflowDeficitDisposition,
)


def _structure(
    *,
    isotope: str = "H",
    displaced: bool = True,
    parent: int = 0,
    relationship: BondRelationshipType = BondRelationshipType.COVALENT,
    declared: bool = True,
    component_id: str = "UNK",
    names: tuple[str, str, str] = ("C1", "S1", "HOBS"),
) -> ProteinStructure:
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id=component_id,
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=(
                    atom_payload(names[0], "C", Vec3(0, 0, 0), b_factor=13.0),
                    atom_payload(names[1], "S", Vec3(1.8, 0, 0)),
                    atom_payload(
                        names[2],
                        isotope,
                        Vec3(1, 0.3, 0) if displaced else Vec3(-1.09, 0, 0),
                        occupancy=0.7,
                        b_factor=23.0,
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(AtomIndex(0), AtomIndex(1)),
                *(
                    (
                        TopologyBond(
                            AtomIndex(parent),
                            AtomIndex(2),
                            relationship_type=relationship,
                            provenance=BondProvenance.EVIDENCE_RESOLVED,
                        ),
                    )
                    if declared
                    else ()
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


@pytest.mark.parametrize("isotope", ["H", "D", "T"])
def test_readable_wrong_hydrogen_parent_is_not_zero_burden(isotope: str) -> None:
    source = _structure(isotope=isotope)
    probe = probe_rdkit_no_conect_parser_readability(source)

    assert probe.sanitize_readable is True
    assert probe.metrics().sanitize_readable is True
    assert probe.metrics().extra_proximity_bond_count == 1
    assert probe.metrics().extra_heavy_proximity_bond_count == 0
    assert probe.parser_problem_witnesses() == ()
    assert any(issue.severity is IssueSeverity.WARNING for issue in probe.issues())
    facts = StructureParserCompatibilityFacts.from_structure(source)
    assert facts.compatibility_state.is_compatible()
    assert facts.has_parser_visible_proximity_burden()


@pytest.mark.parametrize("isotope", ["H", "D", "T"])
def test_correct_or_declared_alternative_parent_is_not_misattachment(
    isotope: str,
) -> None:
    for source in (
        _structure(isotope=isotope, displaced=False),
        _structure(isotope=isotope, parent=1),
    ):
        probe = probe_rdkit_no_conect_parser_readability(source)
        assert probe.sanitize_readable is True
        assert probe.metrics().extra_proximity_bond_count == 0
        assert probe.issues() == ()


@pytest.mark.parametrize(
    "relationship",
    [
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.HYDROGEN_BOND,
        BondRelationshipType.METAL_COORDINATION,
    ],
)
def test_non_covalent_relationship_does_not_invent_hydrogen_parent(
    relationship: BondRelationshipType,
) -> None:
    probe = probe_rdkit_no_conect_parser_readability(
        _structure(relationship=relationship)
    )
    assert probe.sanitize_readable is True
    assert probe.metrics().extra_proximity_bond_count == 0


def test_unknown_hydrogen_parent_is_not_guessed_from_template_or_distance() -> None:
    probe = probe_rdkit_no_conect_parser_readability(_structure(declared=False))
    assert probe.sanitize_readable is True
    assert probe.metrics().extra_proximity_bond_count == 0


def test_prepared_attachment_probe_uses_fresh_coordinates_and_graph_custody() -> None:
    source = _structure()
    lookup = prepare_rdkit_no_conect_known_bond_lookup(source)
    corrected = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry.with_atom_geometry(
            AtomIndex(2),
            source.geometry.atom_geometry(AtomIndex(2)).with_position(
                Vec3(-1.09, 0, 0)
            ),
        ),
        topology=source.topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    assert (
        probe_rdkit_no_conect_parser_readability(source, known_bond_lookup=lookup)
        .metrics()
        .extra_proximity_bond_count
        == 1
    )
    assert (
        probe_rdkit_no_conect_parser_readability(corrected, known_bond_lookup=lookup)
        .metrics()
        .extra_proximity_bond_count
        == 0
    )
    assert corrected.topology is source.topology
    assert corrected.constitution is source.constitution
    assert corrected.provenance is source.provenance
    assert corrected.geometry.atom_geometry(AtomIndex(2)).occupancy == 0.7
    assert corrected.geometry.atom_geometry(AtomIndex(2)).b_factor == 23.0
    assert source.geometry.position(AtomIndex(2)) == Vec3(1, 0.3, 0)
    with pytest.raises(ValueError, match="original graph"):
        probe_rdkit_no_conect_parser_readability(
            _structure(parent=1), known_bond_lookup=lookup
        )


def test_explicit_hydrogen_parent_overrides_conflicting_component_template() -> None:
    # CYS's HB1 template attaches to CB, but this carrier explicitly declares SG.
    source = _structure(parent=1, component_id="CYS", names=("CB", "SG", "HB1"))
    lookup = prepare_rdkit_no_conect_known_bond_lookup(source)
    assert lookup.is_known_bond(
        AtomRef(ResidueId("L", 1), "SG"), AtomRef(ResidueId("L", 1), "HB1")
    )
    assert not lookup.is_known_bond(
        AtomRef(ResidueId("L", 1), "CB"), AtomRef(ResidueId("L", 1), "HB1")
    )


def test_failure_and_attachment_witnesses_count_the_same_edge_once() -> None:
    left, right = AtomRef(ResidueId("L", 1), "H"), AtomRef(ResidueId("L", 1), "S")
    attachment = RDKitProximityBondWitness(left, right, "H", "S", False)
    failed = RDKitParserProblemWitness(
        residue_id=left.residue_id,
        atom_names=("S",),
        messages=("Explicit valence is invalid",),
        proximity_bonds=(RDKitProximityBondWitness(right, left, "S", "H", False),),
    )
    probe = RDKitNoConectParserReadabilityProbe(False, (failed,), (), (attachment,))
    assert probe.metrics().sanitize_readable is False
    assert probe.metrics().extra_proximity_bond_count == 1
    assert probe.metrics().extra_heavy_proximity_bond_count == 0


def test_measured_refinement_rejects_wrong_h_parent_without_changing_structure() -> (
    None
):
    healthy, damaged = _structure(displaced=False), _structure()
    library = prepare_rdkit_no_conect_known_bond_lookup(healthy).component_library
    restraints = build_default_restraint_library()
    parent = measure_refinement_acceptance_metrics(
        healthy,
        focus_residue_ids=frozenset((ResidueId("L", 1),)),
        component_library=library,
        restraint_library=restraints,
    )
    candidate = measure_refinement_acceptance_metrics(
        damaged,
        focus_residue_ids=frozenset((ResidueId("L", 1),)),
        component_library=library,
        restraint_library=restraints,
    )
    assert parent.parser_compatibility.rdkit_sanitize_readable is True
    assert candidate.parser_compatibility.rdkit_sanitize_readable is True
    assert refinement_metrics_rejected(parent, candidate)
    assert not refinement_metrics_rejected(candidate, parent)
    assert not refinement_metrics_rejected(candidate, candidate)
    assert damaged.geometry.position(AtomIndex(2)) == Vec3(1, 0.3, 0)


@pytest.mark.parametrize("isotope", ["H", "D", "T"])
def test_readable_attachment_burden_reaches_workflow_quality(isotope: str) -> None:
    source = _structure(isotope=isotope)
    score = evaluate_workflow_branch_quality_score(
        TransformationResult(source, (), ()),
        requested_goal_report=RequestedGoalReport(()),
        planning_context=WorkflowPlanningContext(),
        component_library=None,
    )
    facts = StructureParserCompatibilityFacts.from_structure(source)
    deficit = StructureParserCompatibilityDeficit(
        profile=facts.profile,
        compatibility_state=facts.compatibility_state,
        extra_proximity_bond_count=facts.extra_proximity_bond_count,
        extra_heavy_proximity_bond_count=facts.extra_heavy_proximity_bond_count,
        disposition=WorkflowDeficitDisposition.OPTIONAL,
    )

    assert score.parser_incompatible == 0
    assert score.parser_extra_bond_count == 1
    assert score.parser_extra_heavy_bond_count == 0
    assert deficit.has_parser_visible_proximity_burden()


def test_cross_residue_hydrogen_attachment_keeps_endpoint_identities() -> None:
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("H1", "H", Vec3(1, 0.3, 0)),
                ),
            ),
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 2),
                is_hetero=True,
                atoms=(atom_payload("S1", "S", Vec3(1.8, 0, 0)),),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(AtomIndex(0), AtomIndex(2)),
                TopologyBond(AtomIndex(1), AtomIndex(2)),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    probe = probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is True
    assert probe.metrics().extra_proximity_bond_count == 1
    assert probe.hydrogen_attachment_witnesses[0].atom_ref_1 == AtomRef(
        ResidueId("L", 1), "H1"
    )
    assert probe.hydrogen_attachment_witnesses[0].atom_ref_2 == AtomRef(
        ResidueId("L", 1), "C1"
    )


def test_polar_donor_acceptor_contact_is_not_a_covalent_h_attachment() -> None:
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=(
                    atom_payload("N1", "N", Vec3(0, 0, 0)),
                    atom_payload("H1", "D", Vec3(1.01, 0, 0)),
                    atom_payload("O1", "O", Vec3(2.9, 0, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(AtomIndex(0), AtomIndex(1)),
                TopologyBond(
                    AtomIndex(1),
                    AtomIndex(2),
                    relationship_type=BondRelationshipType.HYDROGEN_BOND,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    probe = probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is True
    assert probe.metrics().extra_proximity_bond_count == 0
    assert probe.issues() == ()


def test_prepared_attachment_probe_refuses_a_different_library() -> None:
    source = _structure()
    lookup = prepare_rdkit_no_conect_known_bond_lookup(source)
    other_library = type(lookup.component_library)(
        templates=lookup.component_library.templates
    )
    with pytest.raises(ValueError, match="original component library"):
        probe_rdkit_no_conect_parser_readability(
            source, known_bond_lookup=lookup, component_library=other_library
        )


def test_hydrogen_free_probe_is_unobserved_not_attachment_success() -> None:
    source = _structure().without_hydrogens_in_residues((ResidueId("L", 1),))
    probe = probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is None
    assert probe.metrics().sanitize_readable is None
    assert probe.metrics().extra_proximity_bond_count == 0
    assert probe.issues() == ()


def test_unobserved_parser_facts_do_not_plan_attachment_repair() -> None:
    source = _structure()
    facts = StructureParserCompatibilityFacts(
        carrier=source,
        profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
        compatibility_state=ParserCompatibilityState.NOT_OBSERVED,
        extra_proximity_bond_count=1,
    )
    assert not facts.has_parser_visible_proximity_burden()
