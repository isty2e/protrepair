"""Completion-family workflow action proposal adapters."""

from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.retained_non_polymer_hydrogen_completion import (
    RetainedNonPolymerHydrogenCompletionTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.completion.transformer_candidates import (
    plan_atom_completion_transformers,
    plan_hydrogen_completion_transformers,
    plan_retained_non_polymer_hydrogen_completion_transformers,
)
from protrepair.workflow.planning.default_action_registry.capabilities import (
    HEAVY_ATOM_COMPLETION_CAPABILITY,
    HYDROGEN_COMPLETION_CAPABILITY,
    RETAINED_NON_POLYMER_HYDROGEN_COMPLETION_CAPABILITY,
)

__all__ = [
    "heavy_atom_completion_is_admissible",
    "heavy_atom_completion_proposals",
    "hydrogen_completion_is_admissible",
    "hydrogen_completion_proposals",
    "retained_non_polymer_hydrogen_completion_is_admissible",
    "retained_non_polymer_hydrogen_completion_proposals",
]


def heavy_atom_completion_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether heavy completion is admissible in the active domain."""

    return domain.completion.requires_atom_completion()


def heavy_atom_completion_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return concrete heavy-completion proposals for the active domain."""

    planning_outcome = plan_atom_completion_transformers(
        domain.structure,
        requested_goals=domain.requested_goals,
        component_library=domain.component_library,
        required_residue_ids=domain.explicit_repair.prerequisite_residue_ids(),
        coverage_facts=domain.coverage_facts,
        chemistry_readiness_facts=domain.chemistry_readiness_facts,
    )
    return tuple(
        WorkflowActionProposal(
            transformer=transformer,
            capability=HEAVY_ATOM_COMPLETION_CAPABILITY,
        )
        for transformer in planning_outcome.transformers
        if isinstance(transformer, HeavyAtomCompletionTransformer)
    )


def hydrogen_completion_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether hydrogen completion is admissible in the active domain."""

    return domain.completion.requires_hydrogen_completion()


def hydrogen_completion_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return concrete hydrogen-completion proposals for the active domain."""

    planning_outcome = plan_hydrogen_completion_transformers(
        domain.structure,
        requested_goals=domain.requested_goals,
        component_library=domain.component_library,
        required_residue_ids=domain.explicit_repair.prerequisite_residue_ids(),
        coverage_facts=domain.coverage_facts,
        chemistry_readiness_facts=domain.chemistry_readiness_facts,
    )
    return tuple(
        WorkflowActionProposal(
            transformer=transformer,
            capability=HYDROGEN_COMPLETION_CAPABILITY,
        )
        for transformer in planning_outcome.transformers
        if isinstance(transformer, HydrogenCompletionTransformer)
    )


def retained_non_polymer_hydrogen_completion_is_admissible(
    domain: WorkflowActionDomain,
) -> bool:
    """Return whether retained non-polymer hydrogen completion is admissible."""

    return domain.completion.requires_retained_non_polymer_hydrogen_completion()


def retained_non_polymer_hydrogen_completion_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return concrete retained non-polymer hydrogen proposals."""

    planning_outcome = plan_retained_non_polymer_hydrogen_completion_transformers(
        domain.structure,
        requested_goals=domain.requested_goals,
        component_library=domain.component_library,
        chemistry_readiness_facts=domain.chemistry_readiness_facts,
    )
    return tuple(
        WorkflowActionProposal(
            transformer=transformer,
            capability=RETAINED_NON_POLYMER_HYDROGEN_COMPLETION_CAPABILITY,
        )
        for transformer in planning_outcome.transformers
        if isinstance(transformer, RetainedNonPolymerHydrogenCompletionTransformer)
    )
