"""Neutral settings and profile identifiers for continuous relaxation."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ContinuousRelaxationMethod(str, Enum):
    """Closed local-relaxation method families used by built-in profiles."""

    UFF = "uff"
    MMFF = "mmff"


ContinuousRelaxationForceField = ContinuousRelaxationMethod


class ContinuousRelaxationOptimizer(str, Enum):
    """Closed optimizer drivers exposed by built-in relaxation profiles."""

    NATIVE = "native"


class ContinuousRelaxationProfile(str, Enum):
    """Validated built-in continuous-relaxation execution profiles."""

    RDKIT_UFF = "rdkit_uff"
    RDKIT_MMFF = "rdkit_mmff"


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationProfileSpec:
    """Describe one validated execution profile over orthogonal runtime axes."""

    profile: ContinuousRelaxationProfile
    backend_name: str
    method: ContinuousRelaxationMethod
    optimizer: ContinuousRelaxationOptimizer

    def __post_init__(self) -> None:
        normalized_backend_name = validated_continuous_relaxation_backend_name(
            self.backend_name
        )
        if not isinstance(self.profile, ContinuousRelaxationProfile):
            raise TypeError(
                "continuous relaxation profile specs require a "
                "ContinuousRelaxationProfile value"
            )
        if not isinstance(self.method, ContinuousRelaxationMethod):
            raise TypeError(
                "continuous relaxation profile specs require a "
                "ContinuousRelaxationMethod value"
            )
        if not isinstance(self.optimizer, ContinuousRelaxationOptimizer):
            raise TypeError(
                "continuous relaxation profile specs require a "
                "ContinuousRelaxationOptimizer value"
            )

        object.__setattr__(self, "backend_name", normalized_backend_name)


def validated_continuous_relaxation_backend_name(backend_name: str) -> str:
    """Normalize and validate one continuous-relaxation backend name."""

    normalized_backend_name = backend_name.strip().lower()
    if not normalized_backend_name:
        raise ValueError("continuous relaxation backend_name must not be blank")

    return normalized_backend_name


BUILTIN_CONTINUOUS_RELAXATION_PROFILE_SPECS: dict[
    ContinuousRelaxationProfile,
    ContinuousRelaxationProfileSpec,
] = {
    ContinuousRelaxationProfile.RDKIT_UFF: ContinuousRelaxationProfileSpec(
        profile=ContinuousRelaxationProfile.RDKIT_UFF,
        backend_name="rdkit",
        method=ContinuousRelaxationMethod.UFF,
        optimizer=ContinuousRelaxationOptimizer.NATIVE,
    ),
    ContinuousRelaxationProfile.RDKIT_MMFF: ContinuousRelaxationProfileSpec(
        profile=ContinuousRelaxationProfile.RDKIT_MMFF,
        backend_name="rdkit",
        method=ContinuousRelaxationMethod.MMFF,
        optimizer=ContinuousRelaxationOptimizer.NATIVE,
    ),
}


def validate_continuous_relaxation_scalars(
    *,
    backend_name: str,
    context_radius_angstrom: float,
    max_iterations: int,
) -> str:
    """Validate profile-independent continuous-relaxation scalar inputs."""

    normalized_backend_name = validated_continuous_relaxation_backend_name(backend_name)
    if isinstance(context_radius_angstrom, bool) or not isinstance(
        context_radius_angstrom,
        int | float,
    ):
        raise TypeError(
            "continuous relaxation context_radius_angstrom must be a finite number"
        )
    if not isfinite(float(context_radius_angstrom)):
        raise ValueError("continuous relaxation context_radius_angstrom must be finite")
    if context_radius_angstrom < 0.0:
        raise ValueError(
            "continuous relaxation context_radius_angstrom must be non-negative"
        )
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError(
            "continuous relaxation max_iterations must be a positive integer"
        )
    if max_iterations <= 0:
        raise ValueError("continuous relaxation max_iterations must be positive")

    return normalized_backend_name


def continuous_relaxation_profile_spec(
    profile: ContinuousRelaxationProfile,
) -> ContinuousRelaxationProfileSpec:
    """Return the canonical built-in metadata for one validated profile."""

    if not isinstance(profile, ContinuousRelaxationProfile):
        raise TypeError(
            "continuous relaxation profile lookups require a "
            "ContinuousRelaxationProfile value"
        )

    return BUILTIN_CONTINUOUS_RELAXATION_PROFILE_SPECS[profile]


def continuous_relaxation_profile_for_backend_and_method(
    backend_name: str,
    method: ContinuousRelaxationMethod,
) -> ContinuousRelaxationProfile:
    """Return the built-in profile that realizes one backend/method pair."""

    normalized_backend_name = validated_continuous_relaxation_backend_name(backend_name)
    if not isinstance(method, ContinuousRelaxationMethod):
        raise TypeError(
            "continuous relaxation profile resolution requires a "
            "ContinuousRelaxationMethod value"
        )

    for profile, profile_spec in BUILTIN_CONTINUOUS_RELAXATION_PROFILE_SPECS.items():
        if (
            profile_spec.backend_name == normalized_backend_name
            and profile_spec.method is method
        ):
            return profile

    raise ValueError(
        "continuous relaxation backend/method pair does not resolve to a built-in "
        f"profile: backend_name={normalized_backend_name!r}, method={method.value!r}"
    )


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationConfig:
    """Profile-free continuous-relaxation configuration with scalar inputs only."""

    backend_name: str = "rdkit"
    context_radius_angstrom: float = 6.0
    max_iterations: int = 200

    def __post_init__(self) -> None:
        normalized_backend_name = validate_continuous_relaxation_scalars(
            backend_name=self.backend_name,
            context_radius_angstrom=self.context_radius_angstrom,
            max_iterations=self.max_iterations,
        )
        object.__setattr__(self, "backend_name", normalized_backend_name)

    def bind(
        self,
        selection: ContinuousRelaxationProfile | ContinuousRelaxationMethod,
    ) -> "ContinuousRelaxationSettings":
        """Bind one explicit built-in method or profile to this canonical config."""

        if isinstance(selection, ContinuousRelaxationProfile):
            profile_spec = continuous_relaxation_profile_spec(selection)
            if profile_spec.backend_name != self.backend_name:
                raise ValueError(
                    "continuous relaxation config backend_name must match the bound "
                    "profile runtime"
                )
        return ContinuousRelaxationSettings(
            profile=(
                selection
                if isinstance(selection, ContinuousRelaxationProfile)
                else continuous_relaxation_profile_for_backend_and_method(
                    self.backend_name,
                    selection,
                )
            ),
            context_radius_angstrom=self.context_radius_angstrom,
            max_iterations=self.max_iterations,
        )


@dataclass(frozen=True, slots=True, init=False)
class ContinuousRelaxationSettings:
    """Bound continuous-relaxation execution settings for already-legal runs."""

    profile: ContinuousRelaxationProfile
    context_radius_angstrom: float
    max_iterations: int

    def __init__(
        self,
        *,
        profile: ContinuousRelaxationProfile | None = None,
        backend_name: str | None = None,
        force_field: ContinuousRelaxationMethod | None = None,
        context_radius_angstrom: float = 6.0,
        max_iterations: int = 200,
    ) -> None:
        if profile is not None:
            if backend_name is not None or force_field is not None:
                raise TypeError(
                    "continuous relaxation settings must not mix explicit profile "
                    "with legacy backend_name/force_field inputs"
                )
            if not isinstance(profile, ContinuousRelaxationProfile):
                raise TypeError(
                    "continuous relaxation settings require a "
                    "ContinuousRelaxationProfile value"
                )
            profile_spec = continuous_relaxation_profile_spec(profile)
            normalized_profile = profile
        else:
            if backend_name is None or force_field is None:
                raise TypeError(
                    "continuous relaxation settings require either one explicit "
                    "profile or both backend_name and force_field"
                )
            normalized_profile = continuous_relaxation_profile_for_backend_and_method(
                backend_name,
                force_field,
            )
            profile_spec = continuous_relaxation_profile_spec(normalized_profile)

        validate_continuous_relaxation_scalars(
            backend_name=profile_spec.backend_name,
            context_radius_angstrom=context_radius_angstrom,
            max_iterations=max_iterations,
        )
        object.__setattr__(self, "profile", normalized_profile)
        object.__setattr__(
            self,
            "context_radius_angstrom",
            context_radius_angstrom,
        )
        object.__setattr__(self, "max_iterations", max_iterations)

    @property
    def backend_name(self) -> str:
        """Return the normalized runtime token implied by this bound profile."""

        return continuous_relaxation_profile_spec(self.profile).backend_name

    @property
    def method(self) -> ContinuousRelaxationMethod:
        """Return the bound continuous-relaxation method for this profile."""

        return continuous_relaxation_profile_spec(self.profile).method

    @property
    def force_field(self) -> ContinuousRelaxationMethod:
        """Return the legacy force-field projection for this profile."""

        return self.method

    @property
    def optimizer(self) -> ContinuousRelaxationOptimizer:
        """Return the optimizer driver implied by this bound profile."""

        return continuous_relaxation_profile_spec(self.profile).optimizer


__all__ = [
    "BUILTIN_CONTINUOUS_RELAXATION_PROFILE_SPECS",
    "ContinuousRelaxationConfig",
    "ContinuousRelaxationForceField",
    "ContinuousRelaxationMethod",
    "ContinuousRelaxationOptimizer",
    "ContinuousRelaxationProfile",
    "ContinuousRelaxationProfileSpec",
    "ContinuousRelaxationSettings",
    "continuous_relaxation_profile_for_backend_and_method",
    "continuous_relaxation_profile_spec",
    "validate_continuous_relaxation_scalars",
    "validated_continuous_relaxation_backend_name",
]
