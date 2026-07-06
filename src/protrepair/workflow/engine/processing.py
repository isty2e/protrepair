"""Public workflow processing entrypoints over canonical structures."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from protrepair.analysis.kinds import AnalysisKind
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_heavy_atom_elements,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.io import read_structure
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.source_microstate_adjudication import (
    adjudicate_source_microstate_contradictions,
)
from protrepair.workflow.contracts.planning import (
    WorkflowLigandContextMode,
    WorkflowPlanningContext,
    WorkflowSpanDonorAvailability,
)
from protrepair.workflow.contracts.policies import LigandPolicy
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    StructureIngressOptions,
    WorkflowGoal,
    WorkflowTransformRequests,
)
from protrepair.workflow.contracts.result import ProcessResult
from protrepair.workflow.engine.finalization import finalize_workflow_result
from protrepair.workflow.engine.packing.reference import (
    prepare_workflow_packing_reference,
)
from protrepair.workflow.engine.runtime import execute_iterative_workflow


def process_canonical_structure(
    structure: ProteinStructure,
    *,
    ingress: StructureIngressOptions | None = None,
    requested_goals: Sequence[WorkflowGoal] = (),
    transform_requests: WorkflowTransformRequests | None = None,
    planning_context: WorkflowPlanningContext | None = None,
    analyses: frozenset[AnalysisKind] = frozenset(),
) -> ProcessResult:
    """Process a canonical structure through the current repair workflow."""

    active_ingress = StructureIngressOptions() if ingress is None else ingress
    requested_goal_set = RequestedGoalSet(tuple(requested_goals))
    active_transform_requests = (
        WorkflowTransformRequests()
        if transform_requests is None
        else transform_requests
    )
    active_planning_context = (
        _default_workflow_planning_context(
            ingress=active_ingress,
            transform_requests=active_transform_requests,
        )
        if planning_context is None
        else planning_context
    )
    normalized_analyses = frozenset(analyses)

    component_library = build_default_component_library()
    normalization_policy = active_ingress.structure_normalization_policy()
    normalized_structure = apply_structure_normalization_policy(
        structure,
        policy=normalization_policy,
    )
    adjudicated_structure, initial_microstate_issues = (
        adjudicate_source_microstate_contradictions(
            normalized_structure,
            component_library=component_library,
        )
    )
    validated_retained_non_polymer_chemistry_evidence = (
        _validated_retained_non_polymer_chemistry_evidence(
            adjudicated_structure,
            active_ingress.retained_non_polymer_chemistry_overrides,
        )
    )
    packing_reference = (
        None
        if active_transform_requests.reference_sidechain_packing is None
        else prepare_workflow_packing_reference(
            adjudicated_structure,
            active_transform_requests.reference_sidechain_packing,
        )
    )
    runtime_result = execute_iterative_workflow(
        adjudicated_structure,
        requested_goals=requested_goal_set,
        transform_requests=active_transform_requests,
        planning_context=active_planning_context,
        component_library=component_library,
        reference_structure=(
            None if packing_reference is None else packing_reference.reference_structure
        ),
        orphan_fragment_policy=active_transform_requests.orphan_fragment_policy,
        protonate_histidines=active_transform_requests.protonate_histidines,
        retained_non_polymer_chemistry_evidence=(
            validated_retained_non_polymer_chemistry_evidence
        ),
        initial_issues=initial_microstate_issues,
    )
    packing_reference_issues = (
        ()
        if packing_reference is None
        else packing_reference.packing_result.issues
    )
    return finalize_workflow_result(
        terminal_branches=runtime_result.terminal_branches,
        requested_goals=requested_goal_set,
        planning_context=active_planning_context,
        component_library=component_library,
        initially_satisfied_requested_goals=(
            runtime_result.initially_satisfied_requested_goals
        ),
        requested_analyses=normalized_analyses,
        preliminary_issues=packing_reference_issues,
    )


def process_structure_source(
    source: Path | str | ProteinStructure,
    *,
    ingress: StructureIngressOptions | None = None,
    selected_source_chain_ids: tuple[str, ...] | None = None,
    requested_goals: Sequence[WorkflowGoal] = (),
    transform_requests: WorkflowTransformRequests | None = None,
    planning_context: WorkflowPlanningContext | None = None,
    analyses: frozenset[AnalysisKind] = frozenset(),
) -> ProcessResult:
    """Normalize one supported source and process it through the workflow."""

    active_ingress = StructureIngressOptions() if ingress is None else ingress
    requested_goal_set = RequestedGoalSet(tuple(requested_goals))
    active_transform_requests = (
        WorkflowTransformRequests()
        if transform_requests is None
        else transform_requests
    )
    normalized_analyses = frozenset(analyses)
    structure = normalize_source_structure(
        source,
        ingress=active_ingress,
        selected_source_chain_ids=selected_source_chain_ids,
    )
    return process_canonical_structure(
        structure,
        ingress=active_ingress,
        requested_goals=requested_goal_set.goals,
        transform_requests=active_transform_requests,
        planning_context=planning_context,
        analyses=normalized_analyses,
    )


def normalize_source_structure(
    source: Path | str | ProteinStructure,
    *,
    ingress: StructureIngressOptions,
    selected_source_chain_ids: tuple[str, ...] | None = None,
) -> ProteinStructure:
    """Normalize one supported source into the canonical structure model."""

    normalization_policy = ingress.structure_normalization_policy()
    if selected_source_chain_ids is not None:
        normalization_policy = replace(
            normalization_policy,
            selected_chain_ids=selected_source_chain_ids,
        )

    if isinstance(source, ProteinStructure):
        if selected_source_chain_ids is not None:
            raise ValueError(
                "selected_source_chain_ids applies only to raw source inputs"
            )
        return apply_structure_normalization_policy(
            source,
            policy=normalization_policy,
        )

    path = Path(source)
    return read_structure(
        path,
        policy=normalization_policy,
    )


def _default_workflow_planning_context(
    *,
    ingress: StructureIngressOptions,
    transform_requests: WorkflowTransformRequests,
) -> WorkflowPlanningContext:
    """Build the default explicit planning context from boundary inputs."""

    return WorkflowPlanningContext(
        ligand_context_mode=(
            WorkflowLigandContextMode.CONSIDER_IF_PRESENT
            if ingress.ligand_policy is LigandPolicy.KEEP
            else WorkflowLigandContextMode.IGNORE
        ),
        span_donor_availability=(
            WorkflowSpanDonorAvailability.AVAILABLE
            if transform_requests.requests_external_span_reconstruction()
            else WorkflowSpanDonorAvailability.NONE
        ),
    )


def _validated_retained_non_polymer_chemistry_evidence(
    structure: ProteinStructure,
    overrides: tuple[RetainedNonPolymerChemistryOverride, ...],
) -> tuple[RetainedNonPolymerChemistryEvidence, ...]:
    """Validate ingress retained non-polymer overrides as canonical evidence."""

    if not overrides:
        return ()

    ligands_by_residue_id = {
        ligand.residue_id: ligand for ligand in structure.constitution.ligands
    }
    validated_evidence: list[RetainedNonPolymerChemistryEvidence] = []
    for override in overrides:
        evidence = override.to_evidence()
        ligand = ligands_by_residue_id.get(override.residue_id)
        if ligand is None:
            raise ValueError(
                "retained non-polymer chemistry override must target one kept "
                "retained non-polymer residue after ingress normalization: "
                f"{override.residue_id.display_token()}"
            )

        present_heavy_atom_names = tuple(
            atom_site.name
            for atom_site in ligand.atom_sites
            if atom_site.element != "H"
        )
        if set(present_heavy_atom_names) != set(evidence.heavy_atom_names):
            raise ValueError(
                "retained non-polymer chemistry override heavy_atom_names must "
                "match the kept residue heavy-atom set exactly for "
                f"{override.residue_id.display_token()}"
            )

        try:
            expected_heavy_atom_elements = (
                retained_non_polymer_evidence_heavy_atom_elements(evidence)
            )
        except RdkitUnavailableError as error:
            raise ValueError(
                "retained non-polymer chemistry override validation requires "
                "optional RDKit support for "
                f"{override.residue_id.display_token()}"
            ) from error
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                "invalid retained non-polymer chemistry override for "
                f"{override.residue_id.display_token()}: SMILES evidence could "
                "not be parsed or projected"
            ) from error
        for atom_name, expected_element in zip(
            evidence.heavy_atom_names,
            expected_heavy_atom_elements,
            strict=True,
        ):
            observed_element = ligand.atom_site(atom_name).element
            if observed_element != expected_element:
                raise ValueError(
                    "retained non-polymer chemistry override element mismatch for "
                    f"{override.residue_id.display_token()} atom {atom_name}: "
                    f"observed {observed_element}, expected {expected_element}"
                )

        validated_evidence.append(evidence)

    return tuple(validated_evidence)
