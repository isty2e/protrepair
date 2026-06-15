"""Workflow deficit ontology tests."""

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.scope import WholeStructureScope
from protrepair.state import (
    ClashObservationMode,
    HydrogenCoverageState,
    ParserCompatibilityProfile,
    ParserCompatibilityState,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.workflow.contracts import (
    RequestedGoalSet,
    WorkflowLigandContextMode,
    WorkflowPlanningContext,
    WorkflowSpanDonorAvailability,
    WorkflowTargetIntent,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.planning.assessment.deficits import (
    WorkflowDeficitDisposition,
    WorkflowStateDeficit,
)
from protrepair.workflow.planning.planner import plan_workflow_actions


def test_internal_gap_deficit_becomes_required_with_donor_context(
) -> None:

    """Internal blueprint gaps should become required under donor-enabled policy."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(),
                    ),
                    residue_payload(
                        component_id="GLU",
                        residue_id=ResidueId("A", 4),
                        atoms=(),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-internal-gap",
        polymer_blueprint=PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id="A",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="C"),
                        PolymerResidueSlot(sequence_position=3, token="D"),
                        PolymerResidueSlot(sequence_position=4, token="E"),
                    ),
                ),
            )
        ),
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )

    deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        requested_goals=RequestedGoalSet(),
        planning_context=WorkflowPlanningContext(
            span_donor_availability=WorkflowSpanDonorAvailability.AVAILABLE,
        ),
    )

    assert len(deficit.coverage.gap_deficits) == 1
    assert deficit.coverage.gap_deficits[0].gap.absent_sequence_positions == (2, 3)
    assert (
        deficit.coverage.gap_deficits[0].disposition
        is WorkflowDeficitDisposition.REQUIRED
    )


def test_workflow_state_deficit_keeps_prefix_terminal_gap_optional_by_default() -> None:
    """Prefix terminal gaps should remain optional under the default policy."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("A", 3),
                        atoms=(),
                    ),
                    residue_payload(
                        component_id="GLU",
                        residue_id=ResidueId("A", 4),
                        atoms=(),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-prefix-gap",
        polymer_blueprint=PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id="A",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="C"),
                        PolymerResidueSlot(sequence_position=3, token="D"),
                        PolymerResidueSlot(sequence_position=4, token="E"),
                    ),
                ),
            )
        ),
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )

    deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        requested_goals=RequestedGoalSet(),
        planning_context=WorkflowPlanningContext(
            span_donor_availability=WorkflowSpanDonorAvailability.AVAILABLE,
        ),
    )

    assert len(deficit.coverage.gap_deficits) == 1
    assert deficit.coverage.gap_deficits[0].gap.is_prefix_terminal()
    assert (
        deficit.coverage.gap_deficits[0].disposition
        is WorkflowDeficitDisposition.OPTIONAL
    )


def test_md_ready_planning_outcome_carries_atom_and_hydrogen_deficits(
) -> None:

    """Planner outcomes should carry md-ready atom and hydrogen deficits."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-md-ready",
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
        planning_context=WorkflowPlanningContext(
            target_intent=WorkflowTargetIntent.MD_READY,
        ),
    )

    assert outcome.state_deficit is not None
    assert len(outcome.state_deficit.coverage.atom_deficits) == 1
    atom_deficit = outcome.state_deficit.coverage.atom_deficits[0]
    assert atom_deficit.residue_id == ResidueId("A", 1)
    assert atom_deficit.missing_sidechain_atoms is True
    assert atom_deficit.disposition is WorkflowDeficitDisposition.REQUIRED
    assert (
        outcome.state_deficit.chemistry_readiness.hydrogen_prerequisite_residue_ids
        == (ResidueId("A", 1),)
    )
    assert (
        outcome.state_deficit.chemistry_readiness.disposition
        is WorkflowDeficitDisposition.REQUIRED
    )


def test_workflow_state_deficit_marks_unsupported_hydrogen_burden_blocked() -> None:
    """Unsupported component chemistry should block required hydrogen readiness."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNK",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-unsupported-hydrogen",
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )

    deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        requested_goals=RequestedGoalSet(
            goals=(
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
            )
        ),
        planning_context=WorkflowPlanningContext(),
    )

    assert deficit.chemistry_readiness.unsupported_component_residue_ids == (
        residue_id,
    )
    assert deficit.chemistry_readiness.hydrogen_blocked_residue_ids == (
        residue_id,
    )
    assert (
        deficit.chemistry_readiness.disposition
        is WorkflowDeficitDisposition.BLOCKED
    )


def test_parser_visible_proximity_burden_becomes_parser_compatibility_deficit(
) -> None:
    """Parser-visible false proximity should plan as tool-compatibility burden."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-parser-compatibility",
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )
    parser_facts = StructureParserCompatibilityFacts(
        carrier=structure,
        profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
        compatibility_state=ParserCompatibilityState.INCOMPATIBLE,
        extra_proximity_bond_count=3,
        extra_heavy_proximity_bond_count=2,
    )

    deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        requested_goals=RequestedGoalSet(),
        planning_context=WorkflowPlanningContext(),
        parser_compatibility_facts=parser_facts,
    )

    assert deficit.intrinsic_geometry is None
    assert deficit.parser_compatibility is not None
    assert (
        deficit.parser_compatibility.compatibility_state
        is ParserCompatibilityState.INCOMPATIBLE
    )
    assert deficit.parser_compatibility.extra_proximity_bond_count == 3
    assert deficit.parser_compatibility.extra_heavy_proximity_bond_count == 2
    assert deficit.parser_compatibility.has_parser_visible_proximity_burden()
    assert (
        deficit.parser_compatibility.disposition
        is WorkflowDeficitDisposition.OPTIONAL
    )


def test_parser_chemistry_failure_without_extra_proximity_is_not_geometry_deficit(
) -> None:
    """Generic parser chemistry failures should not become geometry repair burden."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-parser-chemistry",
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )
    parser_facts = StructureParserCompatibilityFacts(
        carrier=structure,
        profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
        compatibility_state=ParserCompatibilityState.INCOMPATIBLE,
    )

    deficit = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        requested_goals=RequestedGoalSet(),
        planning_context=WorkflowPlanningContext(),
        parser_compatibility_facts=parser_facts,
    )

    assert deficit.intrinsic_geometry is None
    assert deficit.parser_compatibility is None



def test_deficits_distinguish_optional_intrinsic_from_required_holo_interaction(
) -> None:

    """Intrinsic burden can stay optional while holo interaction burden becomes
    required.
    """

    intrinsic_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("B", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.2, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.2, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.2, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.2, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.2, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-intrinsic",
    )
    holo_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="FAD",
                residue_id=ResidueId("A", 1001),
                atoms=(
                    atom_payload("C1", "C", Vec3(1.2, 1.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="workflow-deficit-holo-interaction",
    )
    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(intrinsic_structure)
    )

    optional_intrinsic = WorkflowStateDeficit.from_facts(
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
        requested_goals=RequestedGoalSet(),
        planning_context=WorkflowPlanningContext(),
        intrinsic_geometry_facts=StructureIntrinsicGeometryFacts.from_structure(
            intrinsic_structure
        ),
    )

    assert optional_intrinsic.intrinsic_geometry is not None
    assert (
        optional_intrinsic.intrinsic_geometry.disposition
        is WorkflowDeficitDisposition.OPTIONAL
    )
    assert optional_intrinsic.intrinsic_geometry.protein_self_clash_count > 0
    assert (
        optional_intrinsic.intrinsic_geometry.protein_self_clash_observation_mode
        is ClashObservationMode.HEAVY_ATOM_LOWER_BOUND
    )

    holo_coverage_facts, holo_chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(holo_structure)
    )
    required_interaction = WorkflowStateDeficit.from_facts(
        coverage_facts=holo_coverage_facts,
        chemistry_readiness_facts=holo_chemistry_readiness_facts,
        requested_goals=RequestedGoalSet(),
        planning_context=WorkflowPlanningContext(
            ligand_context_mode=WorkflowLigandContextMode.CONSIDER_IF_PRESENT,
            target_intent=WorkflowTargetIntent.DOCKING,
        ),
        interaction_facts=StructureInteractionFacts.from_structure(holo_structure),
    )

    assert required_interaction.interaction is not None
    assert required_interaction.interaction.ligand_aware_clash_count > 0
    assert required_interaction.interaction.ligand_aware_worst_overlap_angstrom > 0.0
    assert required_interaction.interaction.ligand_aware_total_overlap_angstrom > 0.0
    assert (
        required_interaction.interaction.ligand_aware_clash_observation_mode
        is ClashObservationMode.HEAVY_ATOM_LOWER_BOUND
    )
    assert (
        required_interaction.interaction.disposition
        is WorkflowDeficitDisposition.REQUIRED
    )
