"""Default workflow action capability declarations."""

from protrepair.scope import ResidueBoundarySide
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    HydrogenCoverageState,
    OxtPresenceState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.workflow.planning.capability import (
    WorkflowActionActivationMode,
    WorkflowActionCapability,
    WorkflowActionDonorRequirement,
    WorkflowActionEffectClass,
    WorkflowActionLocality,
    WorkflowCapabilityDeficitFamily,
    WorkflowCapabilityFactFamily,
    residue_boundary_goal_descriptor,
    whole_structure_goal_descriptor,
)

HEAVY_ATOM_COMPLETION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.ATOM_COVERAGE,),
    read_fact_families=(
        WorkflowCapabilityFactFamily.COVERAGE,
        WorkflowCapabilityFactFamily.CHEMISTRY_READINESS,
    ),
    blocking_deficit_families=(
        WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
    ),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.AUGMENTS_ABSENCE,
    supported_localities=(WorkflowActionLocality.RESIDUE_SET,),
    supported_goals=(
        whole_structure_goal_descriptor(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        whole_structure_goal_descriptor(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
    ),
)

DISULFIDE_TOPOLOGY_RESOLUTION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(
        WorkflowCapabilityDeficitFamily.TOPOLOGY_RESOLUTION,
    ),
    read_fact_families=(WorkflowCapabilityFactFamily.TOPOLOGY_EVIDENCE,),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.AUGMENTS_ABSENCE,
    supported_localities=(WorkflowActionLocality.RESIDUE_SET,),
)

DISULFIDE_HYDROGEN_NORMALIZATION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(
        WorkflowCapabilityDeficitFamily.CHEMISTRY_CONTRADICTION,
    ),
    read_fact_families=(
        WorkflowCapabilityFactFamily.CHEMISTRY_CONTRADICTION,
    ),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.REMOVES_PRESENT,
    supported_localities=(WorkflowActionLocality.RESIDUE_SET,),
)

HYDROGEN_COMPLETION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,),
    read_fact_families=(
        WorkflowCapabilityFactFamily.COVERAGE,
        WorkflowCapabilityFactFamily.CHEMISTRY_READINESS,
    ),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.AUGMENTS_ABSENCE,
    supported_localities=(WorkflowActionLocality.RESIDUE_SET,),
    supported_goals=(
        whole_structure_goal_descriptor(HydrogenCoverageState.COMPLETE),
    ),
)

RETAINED_NON_POLYMER_HYDROGEN_COMPLETION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,),
    read_fact_families=(WorkflowCapabilityFactFamily.CHEMISTRY_READINESS,),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.AUGMENTS_ABSENCE,
    supported_localities=(WorkflowActionLocality.RESIDUE_SET,),
    supported_goals=(
        whole_structure_goal_descriptor(HydrogenCoverageState.COMPLETE),
    ),
)

TERMINAL_AUGMENTATION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(),
    read_fact_families=(WorkflowCapabilityFactFamily.BOUNDARY,),
    activation_mode=WorkflowActionActivationMode.GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.AUGMENTS_ABSENCE,
    supported_localities=(WorkflowActionLocality.RESIDUE_SET,),
    supported_goals=(
        residue_boundary_goal_descriptor(
            side=ResidueBoundarySide.C_TERMINUS,
            value=OxtPresenceState.PRESENT,
        ),
    ),
)

EXTERNAL_SPAN_RECONSTRUCTION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.COVERAGE_GAP,),
    read_fact_families=(WorkflowCapabilityFactFamily.COVERAGE,),
    activation_mode=WorkflowActionActivationMode.EXPLICIT_ONLY,
    effect_class=WorkflowActionEffectClass.AUGMENTS_ABSENCE,
    supported_localities=(WorkflowActionLocality.RESIDUE_SPAN,),
    donor_requirement=WorkflowActionDonorRequirement.EXTERNAL_DONOR,
)

COMMITTED_PACKING_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(
        WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY,
        WorkflowCapabilityDeficitFamily.PARSER_COMPATIBILITY,
    ),
    read_fact_families=(
        WorkflowCapabilityFactFamily.INTRINSIC_GEOMETRY,
        WorkflowCapabilityFactFamily.PARSER_COMPATIBILITY,
    ),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    supported_localities=(
        WorkflowActionLocality.WHOLE_STRUCTURE,
        WorkflowActionLocality.RESIDUE_SET,
    ),
)

STEREOCHEMISTRY_CORRECTION_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY,),
    read_fact_families=(WorkflowCapabilityFactFamily.INTRINSIC_GEOMETRY,),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    supported_localities=(WorkflowActionLocality.RESIDUE_SET,),
)

LOCAL_REFINEMENT_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(
        WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY,
        WorkflowCapabilityDeficitFamily.PARSER_COMPATIBILITY,
        WorkflowCapabilityDeficitFamily.INTERACTION,
    ),
    read_fact_families=(
        WorkflowCapabilityFactFamily.INTRINSIC_GEOMETRY,
        WorkflowCapabilityFactFamily.PARSER_COMPATIBILITY,
        WorkflowCapabilityFactFamily.INTERACTION,
    ),
    activation_mode=WorkflowActionActivationMode.AUTOMATIC_OR_GOAL_DRIVEN,
    effect_class=WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    supported_localities=(
        WorkflowActionLocality.RESIDUE_SET,
        WorkflowActionLocality.LOCAL_SCOPE,
    ),
)

BACKBONE_WINDOW_REFINEMENT_CAPABILITY = WorkflowActionCapability(
    reducible_deficit_families=(
        WorkflowCapabilityDeficitFamily.BACKBONE_WINDOW_OPERATOR,
    ),
    read_fact_families=(WorkflowCapabilityFactFamily.INTRINSIC_GEOMETRY,),
    activation_mode=WorkflowActionActivationMode.EXPLICIT_ONLY,
    effect_class=WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    supported_localities=(WorkflowActionLocality.RESIDUE_SPAN,),
)
