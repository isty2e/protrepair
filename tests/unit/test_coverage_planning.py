"""Unit tests for split coverage planning phases."""

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.request_builders import (
    transform_requests,
    whole_structure_requested_goals,
)

from protrepair.geometry import Vec3
from protrepair.relation.blueprint import StructureBlueprintCoverageGap
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import ResidueId
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.structure.provenance import FileFormat
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    RequestedGoalSet,
)
from protrepair.workflow.planning.coverage import plan_coverage_transformers
from protrepair.workflow.planning.planner import (
    WorkflowPlannerMemory,
    plan_workflow_actions,
)


def test_plan_coverage_transformers_splits_internal_gap_from_atom_completion() -> None:
    """Coverage planning should keep span and atom completion in separate phases."""

    structure = _gap_and_sidechain_gap_structure()
    reconstruction_spec = _internal_gap_reconstruction_spec()

    outcome = plan_coverage_transformers(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=transform_requests(
            external_span_reconstructions=(reconstruction_spec,)
        ),
        component_library=_component_library(),
        coverage_facts=_coverage_facts(structure),
        chemistry_readiness_facts=_chemistry_readiness_facts(structure),
    )

    assert len(outcome.span_reconstruction_transformers) == 1
    assert isinstance(
        outcome.span_reconstruction_transformers[0],
        ExternalSpanReconstructionTransformer,
    )
    assert len(outcome.atom_completion_transformers) == 1
    assert isinstance(
        outcome.atom_completion_transformers[0],
        HeavyAtomCompletionTransformer,
    )
    assert (
        outcome.current_phase_transformers() == outcome.span_reconstruction_transformers
    )


def test_plan_workflow_actions_emits_span_before_atom_completion() -> None:
    """Workflow planning should expose span reconstruction before atom completion."""

    structure = _gap_and_sidechain_gap_structure()
    reconstruction_spec = _internal_gap_reconstruction_spec()

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=transform_requests(
            external_span_reconstructions=(reconstruction_spec,)
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], ExternalSpanReconstructionTransformer)


def test_plan_workflow_actions_emits_atom_completion_after_span_adoption() -> None:
    """Atom completion should become the next coverage phase after span adoption."""

    structure = _gap_and_sidechain_gap_structure()
    reconstruction_spec = _internal_gap_reconstruction_spec()
    adopted_span_transformer = (
        ExternalSpanReconstructionTransformer.from_reconstruction_spec(
            reconstruction_spec
        )
    )

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=transform_requests(
            external_span_reconstructions=(reconstruction_spec,)
        ),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(adopted_span_transformer,)
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], HeavyAtomCompletionTransformer)


def test_plan_workflow_actions_keeps_existing_residue_gaps_in_atom_completion_phase(
) -> None:
    """Existing-residue atom gaps should not require donor-backed span machinery."""

    structure = _backbone_gap_structure()

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=transform_requests(),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], HeavyAtomCompletionTransformer)




def test_plan_workflow_actions_treats_terminal_gap_opt_in_as_coverage_span_phase() -> (
    None
):
    """Explicit terminal gap reconstruction should remain a span coverage phase."""

    structure = _prefix_gap_structure()
    reconstruction_spec = _prefix_gap_reconstruction_spec()

    outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=transform_requests(
            external_span_reconstructions=(reconstruction_spec,)
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], ExternalSpanReconstructionTransformer)


def test_plan_workflow_actions_reaches_hydrogen_after_coverage_phases() -> None:
    """Hydrogen augmentation should remain after span and atom-completion phases."""

    structure = _sidechain_gap_structure()
    adopted_heavy_transformer = HeavyAtomCompletionTransformer.from_completion_scope(
        scope=_whole_chain_scope()
    )

    first_outcome = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=transform_requests(),
    )
    assert len(first_outcome.transformers) == 1
    assert isinstance(first_outcome.transformers[0], HeavyAtomCompletionTransformer)

    second_outcome = plan_workflow_actions(
        _heavy_complete_structure(),
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        transform_requests=transform_requests(),
        planner_memory=WorkflowPlannerMemory(
            adopted_transformers=(adopted_heavy_transformer,)
        ),
    )

    assert len(second_outcome.transformers) == 1
    assert isinstance(second_outcome.transformers[0], HydrogenCompletionTransformer)


def _gap_and_sidechain_gap_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=_atoms("N", "CA", "C", "O"),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 4),
                        atoms=_atoms("N", "CA", "C", "O"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="gap-and-sidechain-gap",
        polymer_blueprint=PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id="A",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="C"),
                        PolymerResidueSlot(sequence_position=3, token="D"),
                        PolymerResidueSlot(sequence_position=4, token="A"),
                    ),
                ),
            )
        ),
    )


def _prefix_gap_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 3),
                        atoms=_atoms("N", "CA", "C", "O", "CB"),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 4),
                        atoms=_atoms("N", "CA", "C", "O"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="prefix-gap",
        polymer_blueprint=PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id="A",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="M"),
                        PolymerResidueSlot(sequence_position=2, token="A"),
                        PolymerResidueSlot(sequence_position=3, token="A"),
                        PolymerResidueSlot(sequence_position=4, token="G"),
                    ),
                ),
            )
        ),
    )


def _backbone_gap_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=_atoms("N", "CA", "C"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="backbone-gap",
    )


def _sidechain_gap_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=_atoms("N", "CA", "C", "O"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="sidechain-gap",
    )


def _heavy_complete_structure():
    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=_atoms("N", "CA", "C", "O", "CB"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="heavy-complete",
    )


def _internal_gap_reconstruction_spec() -> ExternalSpanReconstructionSpec:
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("X", 2),
                        atoms=_atoms("N", "CA", "C", "O", "CB"),
                    ),
                    residue_payload(
                        component_id="GLU",
                        residue_id=ResidueId("X", 3),
                        atoms=_atoms("N", "CA", "C", "O", "CB", "CG"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="internal-gap-donor",
    )
    return ExternalSpanReconstructionSpec(
        blueprint_coverage_gap=StructureBlueprintCoverageGap(
            structure_chain_id="A",
            blueprint_chain_id="A",
            absent_sequence_positions=(2, 3),
            preceding_residue_id=ResidueId("A", 1),
            following_residue_id=ResidueId("A", 4),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 2), ResidueId("X", 3)),
    )


def _prefix_gap_reconstruction_spec() -> ExternalSpanReconstructionSpec:
    donor_structure = build_structure(
        chains=(
            chain_payload(
                "X",
                (
                    residue_payload(
                        component_id="MET",
                        residue_id=ResidueId("X", 1),
                        atoms=_atoms("N", "CA", "C", "O", "CB"),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("X", 2),
                        atoms=_atoms("N", "CA", "C", "O", "CB"),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="prefix-gap-donor",
    )
    return ExternalSpanReconstructionSpec(
        blueprint_coverage_gap=StructureBlueprintCoverageGap(
            structure_chain_id="A",
            blueprint_chain_id="A",
            absent_sequence_positions=(1, 2),
            preceding_residue_id=None,
            following_residue_id=ResidueId("A", 3),
        ),
        donor_structure=donor_structure,
        donor_residue_ids=(ResidueId("X", 1), ResidueId("X", 2)),
    )


def _atoms(*atom_names: str):
    coordinates = {
        "N": (0.0, 0.0, 0.0),
        "CA": (1.4, 0.0, 0.0),
        "C": (2.8, 0.0, 0.0),
        "O": (3.8, 0.0, 0.0),
        "CB": (1.4, 1.4, 0.0),
        "CG": (2.4, 2.0, 0.0),
    }
    elements = {
        "N": "N",
        "CA": "C",
        "C": "C",
        "O": "O",
        "CB": "C",
        "CG": "C",
    }
    return tuple(
        atom_payload(
            name,
            elements[name],
            Vec3.from_iterable(coordinates[name]),
        )
        for name in atom_names
    )


def _component_library():
    from protrepair.chemistry import build_default_component_library

    return build_default_component_library()


def _coverage_facts(structure):
    from protrepair.state import StructureCoverageFacts

    return StructureCoverageFacts.from_structure(structure)


def _chemistry_readiness_facts(structure):
    from protrepair.state import StructureChemistryReadinessFacts

    return StructureChemistryReadinessFacts.from_structure(structure)


def _whole_chain_scope():
    from protrepair.scope import ResidueSetScope

    return ResidueSetScope(residue_ids=(ResidueId("A", 1),))
