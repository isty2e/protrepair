"""Diagnostic event and issue enums."""

from enum import Enum


class RepairEventKind(str, Enum):
    """Kinds of structural repair and normalization events."""

    ABSENT_RESIDUE_SPAN_RECONSTRUCTED = "absent_residue_span_reconstructed"
    HEAVY_ATOMS_ADDED = "heavy_atoms_added"
    HYDROGENS_ADDED = "hydrogens_added"
    LOCAL_REFINEMENT_APPLIED = "local_refinement_applied"
    C_TERMINAL_OXT_ADDED = "c_terminal_oxt_added"
    COMPONENT_NORMALIZED = "component_normalized"
    STEREOCHEMISTRY_CORRECTED = "stereochemistry_corrected"
    DISULFIDE_TOPOLOGY_RESOLVED = "disulfide_topology_resolved"


class ValidationIssueKind(str, Enum):
    """Kinds of structural validation issues."""

    CHEMISTRY_CONTRADICTION = "chemistry_contradiction"
    PARSER_READABILITY = "parser_readability"
    AMBIGUOUS_DISULFIDE = "ambiguous_disulfide"
    CIS_PEPTIDE = "cis_peptide"
    MISSING_EXPECTED_ATOMS = "missing_expected_atoms"
    MISSING_COMPONENT_DEFINITION = "missing_component_definition"
    UNEXPECTED_ATOMS = "unexpected_atoms"
    INVALID_BACKBONE = "invalid_backbone"
    INVALID_GEOMETRY = "invalid_geometry"
    INVALID_STEREOCHEMISTRY = "invalid_stereochemistry"
    REFINEMENT_REJECTED = "refinement_rejected"
    UNSUPPORTED_TEMPLATE_REPAIR = "unsupported_template_repair"
    UNSUPPORTED_HYDROGENATION = "unsupported_hydrogenation"
    PACKING_INVALIDATED_HYDROGENS = "packing_invalidated_hydrogens"
    RETAINED_NON_POLYMER_FALLBACK_USED = "retained_non_polymer_fallback_used"
    RETAINED_NON_POLYMER_FALLBACK_BLOCKED = "retained_non_polymer_fallback_blocked"
    UNSUPPORTED_COMPONENT = "unsupported_component"
    STERIC_CLASH = "steric_clash"


class IssueSeverity(str, Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
