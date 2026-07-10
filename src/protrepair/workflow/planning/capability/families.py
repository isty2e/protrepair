"""Planner capability vocabulary axes."""

from enum import Enum


class WorkflowCapabilityDeficitFamily(str, Enum):
    """One planner-facing deficit family reducible by a workflow action."""

    COVERAGE_GAP = "coverage_gap"
    ATOM_COVERAGE = "atom_coverage"
    TOPOLOGY_RESOLUTION = "topology_resolution"
    CHEMISTRY_READINESS = "chemistry_readiness"
    BACKBONE_WINDOW_OPERATOR = "backbone_window_operator"
    INTRINSIC_GEOMETRY = "intrinsic_geometry"
    PARSER_COMPATIBILITY = "parser_compatibility"
    INTERACTION = "interaction"


class WorkflowCapabilityFactFamily(str, Enum):
    """One canonical fact family read by a workflow action."""

    COVERAGE = "coverage"
    TOPOLOGY_EVIDENCE = "topology_evidence"
    CHEMISTRY_READINESS = "chemistry_readiness"
    BOUNDARY = "boundary"
    INTRINSIC_GEOMETRY = "intrinsic_geometry"
    PARSER_COMPATIBILITY = "parser_compatibility"
    INTERACTION = "interaction"


class WorkflowActionActivationMode(str, Enum):
    """How one workflow action family may be activated by the planner."""

    AUTOMATIC_OR_GOAL_DRIVEN = "automatic_or_goal_driven"
    GOAL_DRIVEN = "goal_driven"
    EXPLICIT_ONLY = "explicit_only"


class WorkflowActionEffectClass(str, Enum):
    """What kind of structural effect one action family applies."""

    AUGMENTS_ABSENCE = "augments_absence"
    REVISES_PRESENT_GEOMETRY = "revises_present_geometry"


class WorkflowActionLocality(str, Enum):
    """Primary locality scale over which one action family operates."""

    WHOLE_STRUCTURE = "whole_structure"
    RESIDUE_SET = "residue_set"
    RESIDUE_SPAN = "residue_span"
    LOCAL_SCOPE = "local_scope"


class WorkflowActionDonorRequirement(str, Enum):
    """Whether one action family depends on an external donor structure."""

    NONE = "none"
    EXTERNAL_DONOR = "external_donor"
