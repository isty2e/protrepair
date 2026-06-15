"""Default workflow action proposal adapters grouped by action family."""

from protrepair.workflow.planning.default_action_registry.completion_proposals import (
    heavy_atom_completion_is_admissible,
    heavy_atom_completion_proposals,
    hydrogen_completion_is_admissible,
    hydrogen_completion_proposals,
    retained_non_polymer_hydrogen_completion_is_admissible,
    retained_non_polymer_hydrogen_completion_proposals,
)
from protrepair.workflow.planning.default_action_registry.packing_proposals import (
    committed_packing_is_admissible,
    committed_packing_proposals,
)
from protrepair.workflow.planning.default_action_registry.refinement_proposals import (
    backbone_window_refinement_is_admissible,
    backbone_window_refinement_proposals,
    local_refinement_is_admissible,
    local_refinement_proposals,
)
from protrepair.workflow.planning.default_action_registry.span_proposals import (
    external_span_reconstruction_is_admissible,
    external_span_reconstruction_proposals,
)
from protrepair.workflow.planning.default_action_registry.terminal_proposals import (
    terminal_augmentation_is_admissible,
    terminal_augmentation_proposals,
)

from .stereochemistry_proposals import (
    stereochemistry_correction_is_admissible,
    stereochemistry_correction_proposals,
)

__all__ = [
    "backbone_window_refinement_is_admissible",
    "backbone_window_refinement_proposals",
    "committed_packing_is_admissible",
    "committed_packing_proposals",
    "external_span_reconstruction_is_admissible",
    "external_span_reconstruction_proposals",
    "heavy_atom_completion_is_admissible",
    "heavy_atom_completion_proposals",
    "hydrogen_completion_is_admissible",
    "hydrogen_completion_proposals",
    "local_refinement_is_admissible",
    "local_refinement_proposals",
    "retained_non_polymer_hydrogen_completion_is_admissible",
    "retained_non_polymer_hydrogen_completion_proposals",
    "stereochemistry_correction_is_admissible",
    "stereochemistry_correction_proposals",
    "terminal_augmentation_is_admissible",
    "terminal_augmentation_proposals",
]
