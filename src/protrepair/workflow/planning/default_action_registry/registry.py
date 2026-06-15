"""Default workflow action registry assembly."""

from protrepair.workflow.actions.backbone_window_refinement import (
    BackboneWindowRefinementTransformer,
)
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.actions.retained_non_polymer_hydrogen_completion import (
    RetainedNonPolymerHydrogenCompletionTransformer,
)
from protrepair.workflow.actions.stereochemistry import (
    StereochemistryCorrectionTransformer,
)
from protrepair.workflow.actions.terminal_augmentation import (
    TerminalAugmentationTransformer,
)
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.registry import WorkflowActionRegistryEntry
from protrepair.workflow.planning.default_action_registry.capabilities import (
    BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
    COMMITTED_PACKING_CAPABILITY,
    EXTERNAL_SPAN_RECONSTRUCTION_CAPABILITY,
    HEAVY_ATOM_COMPLETION_CAPABILITY,
    HYDROGEN_COMPLETION_CAPABILITY,
    LOCAL_REFINEMENT_CAPABILITY,
    RETAINED_NON_POLYMER_HYDROGEN_COMPLETION_CAPABILITY,
    STEREOCHEMISTRY_CORRECTION_CAPABILITY,
    TERMINAL_AUGMENTATION_CAPABILITY,
)
from protrepair.workflow.planning.default_action_registry.proposals import (
    backbone_window_refinement_is_admissible,
    backbone_window_refinement_proposals,
    committed_packing_is_admissible,
    committed_packing_proposals,
    external_span_reconstruction_is_admissible,
    external_span_reconstruction_proposals,
    heavy_atom_completion_is_admissible,
    heavy_atom_completion_proposals,
    hydrogen_completion_is_admissible,
    hydrogen_completion_proposals,
    local_refinement_is_admissible,
    local_refinement_proposals,
    retained_non_polymer_hydrogen_completion_is_admissible,
    retained_non_polymer_hydrogen_completion_proposals,
    stereochemistry_correction_is_admissible,
    stereochemistry_correction_proposals,
    terminal_augmentation_is_admissible,
    terminal_augmentation_proposals,
)

REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY: tuple[
    WorkflowActionRegistryEntry[WorkflowActionDomain],
    ...,
] = (
    WorkflowActionRegistryEntry(
        action_type=HeavyAtomCompletionTransformer,
        capability=HEAVY_ATOM_COMPLETION_CAPABILITY,
        is_admissible_for_domain=heavy_atom_completion_is_admissible,
        proposals_for_domain=heavy_atom_completion_proposals,
    ),
    WorkflowActionRegistryEntry(
        action_type=HydrogenCompletionTransformer,
        capability=HYDROGEN_COMPLETION_CAPABILITY,
        is_admissible_for_domain=hydrogen_completion_is_admissible,
        proposals_for_domain=hydrogen_completion_proposals,
    ),
    WorkflowActionRegistryEntry(
        action_type=RetainedNonPolymerHydrogenCompletionTransformer,
        capability=RETAINED_NON_POLYMER_HYDROGEN_COMPLETION_CAPABILITY,
        is_admissible_for_domain=(
            retained_non_polymer_hydrogen_completion_is_admissible
        ),
        proposals_for_domain=retained_non_polymer_hydrogen_completion_proposals,
    ),
    WorkflowActionRegistryEntry(
        action_type=TerminalAugmentationTransformer,
        capability=TERMINAL_AUGMENTATION_CAPABILITY,
        is_admissible_for_domain=terminal_augmentation_is_admissible,
        proposals_for_domain=terminal_augmentation_proposals,
    ),
)

EXPLICIT_WORKFLOW_ACTION_REGISTRY: tuple[
    WorkflowActionRegistryEntry[WorkflowActionDomain],
    ...,
] = (
    WorkflowActionRegistryEntry(
        action_type=ExternalSpanReconstructionTransformer,
        capability=EXTERNAL_SPAN_RECONSTRUCTION_CAPABILITY,
        is_admissible_for_domain=external_span_reconstruction_is_admissible,
        proposals_for_domain=external_span_reconstruction_proposals,
    ),
    WorkflowActionRegistryEntry(
        action_type=CommittedPackingTransformer,
        capability=COMMITTED_PACKING_CAPABILITY,
        is_admissible_for_domain=committed_packing_is_admissible,
        proposals_for_domain=committed_packing_proposals,
    ),
    WorkflowActionRegistryEntry(
        action_type=StereochemistryCorrectionTransformer,
        capability=STEREOCHEMISTRY_CORRECTION_CAPABILITY,
        is_admissible_for_domain=stereochemistry_correction_is_admissible,
        proposals_for_domain=stereochemistry_correction_proposals,
    ),
    WorkflowActionRegistryEntry(
        action_type=BackboneWindowRefinementTransformer,
        capability=BACKBONE_WINDOW_REFINEMENT_CAPABILITY,
        is_admissible_for_domain=backbone_window_refinement_is_admissible,
        proposals_for_domain=backbone_window_refinement_proposals,
    ),
    WorkflowActionRegistryEntry(
        action_type=LocalRefinementTransformer,
        capability=LOCAL_REFINEMENT_CAPABILITY,
        is_admissible_for_domain=local_refinement_is_admissible,
        proposals_for_domain=local_refinement_proposals,
    ),
)

WORKFLOW_ACTION_REGISTRY: tuple[
    WorkflowActionRegistryEntry[WorkflowActionDomain],
    ...,
] = (
    *REQUEST_DRIVEN_WORKFLOW_ACTION_REGISTRY,
    *EXPLICIT_WORKFLOW_ACTION_REGISTRY,
)
