"""Closed state axes for canonical structure observations."""

from enum import Enum

__all__ = [
    "BackboneHeavyAtomCompletenessState",
    "BoundaryAuthenticityState",
    "ClashObservationMode",
    "ClashPresenceState",
    "ComponentSupportState",
    "HydrogenApplicabilityState",
    "HydrogenCoverageState",
    "OrientationCorrectionEligibilityState",
    "OxtPresenceState",
    "ParserCompatibilityProfile",
    "ParserCompatibilityState",
    "SidechainHeavyAtomCompletenessState",
    "StereochemistryState",
]


class ComponentSupportState(str, Enum):
    """Observed support coverage over one canonical residue set."""

    ALL_SUPPORTED = "all_supported"
    UNSUPPORTED_COMPONENTS_PRESENT = "unsupported_components_present"

    def is_fully_supported(self) -> bool:
        """Return whether every observed residue has canonical chemistry support."""

        return self is ComponentSupportState.ALL_SUPPORTED


class BackboneHeavyAtomCompletenessState(str, Enum):
    """Observed backbone heavy-atom completeness over one canonical residue set."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"

    def requires_completion(self) -> bool:
        """Return whether backbone heavy-atom completion is still required."""

        return self is BackboneHeavyAtomCompletenessState.INCOMPLETE


class SidechainHeavyAtomCompletenessState(str, Enum):
    """Observed side-chain heavy-atom completeness over one canonical residue set."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"

    def requires_completion(self) -> bool:
        """Return whether side-chain heavy-atom completion is still required."""

        return self is SidechainHeavyAtomCompletenessState.INCOMPLETE


class HydrogenApplicabilityState(str, Enum):
    """Observed hydrogen applicability over one canonical residue set."""

    NOT_APPLICABLE = "not_applicable"
    APPLICABLE = "applicable"

    def is_applicable(self) -> bool:
        """Return whether hydrogens are applicable for this residue set."""

        return self is HydrogenApplicabilityState.APPLICABLE


class HydrogenCoverageState(str, Enum):
    """Observed hydrogen coverage over hydrogen-applicable residues."""

    NONE = "none"
    PARTIAL = "partial"
    COMPLETE = "complete"

    def needs_hydrogenation(self) -> bool:
        """Return whether hydrogen completion may still add missing hydrogens."""

        return self is not HydrogenCoverageState.COMPLETE


class ClashPresenceState(str, Enum):
    """Observed aggregate clash presence over one structure-level fact owner.

    This is the whole-structure clash axis used by structure facts and
    requested goals. Selected local scopes use `ClashState` instead.
    """

    NONE = "none"
    PRESENT = "present"

    def has_clashes(self) -> bool:
        """Return whether at least one clash is present."""

        return self is ClashPresenceState.PRESENT


class ClashObservationMode(str, Enum):
    """Observation mode behind one canonical clash burden measurement."""

    HEAVY_ATOM_LOWER_BOUND = "heavy_atom_lower_bound"
    ALL_ATOM_COMPLETE = "all_atom_complete"

    def is_complete(self) -> bool:
        """Return whether the clash burden was observed on a complete all-atom set."""

        return self is ClashObservationMode.ALL_ATOM_COMPLETE


class ParserCompatibilityProfile(str, Enum):
    """Parser profile used for an observational compatibility fact."""

    RDKIT_NO_CONECT_SANITIZE = "rdkit_no_conect_sanitize"


class ParserCompatibilityState(str, Enum):
    """Observed compatibility with one parser profile."""

    NOT_OBSERVED = "not_observed"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"

    def is_compatible(self) -> bool:
        """Return whether the observed parser profile accepted the structure."""

        return self is ParserCompatibilityState.COMPATIBLE

    def is_incompatible(self) -> bool:
        """Return whether the observed parser profile rejected the structure."""

        return self is ParserCompatibilityState.INCOMPATIBLE


class OxtPresenceState(str, Enum):
    """Observed OXT presence over one observed residue-boundary scope."""

    ABSENT = "absent"
    PRESENT = "present"

    def is_present(self) -> bool:
        """Return whether OXT is observed on the scoped residue boundary."""

        return self is OxtPresenceState.PRESENT


class StereochemistryState(str, Enum):
    """Observed side-chain stereochemistry state over one canonical residue set."""

    NOT_APPLICABLE = "not_applicable"
    CONSISTENT = "consistent"
    VIOLATED = "violated"


class OrientationCorrectionEligibilityState(str, Enum):
    """Observed discrete side-chain orientation-correction eligibility."""

    NOT_ELIGIBLE = "not_eligible"
    ELIGIBLE = "eligible"


class BoundaryAuthenticityState(str, Enum):
    """Authenticity of one observed residue-boundary scope in context."""

    AUTHENTIC_IN_CONTEXT = "authentic_in_context"
    PROJECTED_FRAGMENT_BOUNDARY = "projected_fragment_boundary"
