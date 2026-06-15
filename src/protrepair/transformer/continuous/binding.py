"""Neutral binding request and decision DTOs for continuous relaxation."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationMethod,
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
    continuous_relaxation_profile_spec,
)


class ContinuousRelaxationBindingReason(str, Enum):
    """Closed explanations for why one continuous-relaxation profile was bound."""

    MANUAL_EXPLICIT_SELECTION = "manual_explicit_selection"
    HYDROGENATED_DOMAIN_POLICY = "hydrogenated_domain_policy"
    BACKEND_PREFERRED_PROFILE = "backend_preferred_profile"
    ONLY_SUPPORTED_PROFILE = "only_supported_profile"


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationBindingDecision:
    """Explainable bound profile choice for one legal execution."""

    settings: ContinuousRelaxationSettings
    reason: ContinuousRelaxationBindingReason

    def __post_init__(self) -> None:
        if not isinstance(self.settings, ContinuousRelaxationSettings):
            raise TypeError(
                "continuous relaxation binding decisions require bound settings"
            )
        if not isinstance(self.reason, ContinuousRelaxationBindingReason):
            raise TypeError(
                "continuous relaxation binding decisions require a binding reason"
            )


@dataclass(frozen=True, slots=True)
class ManualContinuousRelaxationBinding:
    """Explicit method/profile binding chosen before current-state execution."""

    selection: ContinuousRelaxationProfile | ContinuousRelaxationMethod

    def __post_init__(self) -> None:
        if not isinstance(
            self.selection,
            ContinuousRelaxationProfile | ContinuousRelaxationMethod,
        ):
            raise TypeError(
                "manual continuous relaxation binding requires a "
                "ContinuousRelaxationProfile or ContinuousRelaxationMethod value"
            )

    @property
    def profile(self) -> ContinuousRelaxationProfile | None:
        """Return the explicit profile when the manual binding is profile-based."""

        if isinstance(self.selection, ContinuousRelaxationProfile):
            return self.selection
        return None

    @property
    def force_field(self) -> ContinuousRelaxationMethod:
        """Return the legacy method projection for compatibility callers."""

        if isinstance(self.selection, ContinuousRelaxationProfile):
            return continuous_relaxation_profile_spec(self.selection).method
        return self.selection


@dataclass(frozen=True, slots=True)
class RecommendedContinuousRelaxationBinding:
    """Current-state-dependent profile binding selected at execution time."""


ContinuousRelaxationBinding: TypeAlias = (
    ManualContinuousRelaxationBinding | RecommendedContinuousRelaxationBinding
)


__all__ = [
    "ContinuousRelaxationBinding",
    "ContinuousRelaxationBindingDecision",
    "ContinuousRelaxationBindingReason",
    "ManualContinuousRelaxationBinding",
    "RecommendedContinuousRelaxationBinding",
]
