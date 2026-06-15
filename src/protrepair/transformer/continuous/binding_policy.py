"""Continuous-relaxation binding policy and backend capability semantics."""

from dataclasses import dataclass

from protrepair.errors import RefinementError
from protrepair.state import HydrogenCoverageState
from protrepair.state.domain import AtomScopeStateFacts
from protrepair.transformer.atom_input import AtomInputBasis
from protrepair.transformer.continuous.binding import (
    ContinuousRelaxationBinding,
    ContinuousRelaxationBindingDecision,
    ContinuousRelaxationBindingReason,
    ManualContinuousRelaxationBinding,
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.readiness import (
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationMethod,
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
    continuous_relaxation_profile_spec,
    validated_continuous_relaxation_backend_name,
)


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationBackendCapabilities:
    """Declared backend capability semantics for one continuous backend."""

    backend_name: str
    supported_profiles: frozenset[ContinuousRelaxationProfile]
    preferred_profile: ContinuousRelaxationProfile | None = None

    def __post_init__(self) -> None:
        normalized_backend_name = validated_continuous_relaxation_backend_name(
            self.backend_name
        )
        supported_profiles = frozenset(self.supported_profiles)
        if not supported_profiles:
            raise ValueError(
                "continuous relaxation backend capabilities require at least one "
                "supported profile"
            )
        for profile in supported_profiles:
            if not isinstance(profile, ContinuousRelaxationProfile):
                raise TypeError(
                    "continuous relaxation backend supported profiles must be "
                    "ContinuousRelaxationProfile values"
                )
            profile_spec = continuous_relaxation_profile_spec(profile)
            if profile_spec.backend_name != normalized_backend_name:
                raise ValueError(
                    "continuous relaxation backend capabilities must only declare "
                    "profiles for their own backend"
                )

        preferred_profile = self.preferred_profile
        if preferred_profile is not None:
            if not isinstance(preferred_profile, ContinuousRelaxationProfile):
                raise TypeError(
                    "continuous relaxation backend preferred_profile must be a "
                    "ContinuousRelaxationProfile value"
                )
            if preferred_profile not in supported_profiles:
                raise ValueError(
                    "continuous relaxation backend preferred_profile must be "
                    "supported by the backend"
                )

        object.__setattr__(self, "backend_name", normalized_backend_name)
        object.__setattr__(self, "supported_profiles", supported_profiles)

    @property
    def supported_methods(self) -> frozenset[ContinuousRelaxationMethod]:
        """Return the distinct method families exposed by the supported profiles."""

        return frozenset(
            continuous_relaxation_profile_spec(profile).method
            for profile in self.supported_profiles
        )

    def recommended_binding_decision(
        self,
        config: ContinuousRelaxationConfig,
        *,
        atom_scope_facts: AtomScopeStateFacts,
    ) -> ContinuousRelaxationBindingDecision:
        """Return the explainable recommended binding for one legal domain."""

        require_atom_scope_continuous_relaxation_execution(atom_scope_facts)
        if self.preferred_profile is not None:
            return ContinuousRelaxationBindingDecision(
                settings=config.bind(self.preferred_profile),
                reason=ContinuousRelaxationBindingReason.BACKEND_PREFERRED_PROFILE,
            )

        if len(self.supported_profiles) == 1:
            (profile,) = tuple(self.supported_profiles)
            return ContinuousRelaxationBindingDecision(
                settings=config.bind(profile),
                reason=ContinuousRelaxationBindingReason.ONLY_SUPPORTED_PROFILE,
            )

        supported_tokens = ", ".join(
            sorted(profile.value for profile in self.supported_profiles)
        )
        raise RefinementError(
            "recommended continuous relaxation binding is ambiguous for backend "
            f"{self.backend_name!r}: supported profiles are "
            f"{supported_tokens}, but no preferred profile is declared"
        )


def continuous_relaxation_backend_capabilities(
    backend_name: str,
) -> ContinuousRelaxationBackendCapabilities:
    """Return declared backend capability semantics by canonical name."""

    normalized_backend_name = validated_continuous_relaxation_backend_name(backend_name)
    if normalized_backend_name == "rdkit":
        return ContinuousRelaxationBackendCapabilities(
            backend_name="rdkit",
            supported_profiles=frozenset(
                {
                    ContinuousRelaxationProfile.RDKIT_UFF,
                    ContinuousRelaxationProfile.RDKIT_MMFF,
                }
            ),
            preferred_profile=ContinuousRelaxationProfile.RDKIT_MMFF,
        )
    raise RefinementError(
        f"continuous relaxation backend {backend_name!r} does not declare "
        "capability semantics"
    )


def decide_continuous_relaxation_binding(
    binding: ContinuousRelaxationBinding,
    config: ContinuousRelaxationConfig,
    *,
    atom_scope_facts: AtomScopeStateFacts,
    atom_input_basis: AtomInputBasis,
) -> ContinuousRelaxationBindingDecision:
    """Return one execution binding decision for a public binding request."""

    if isinstance(binding, ManualContinuousRelaxationBinding):
        return _manual_continuous_relaxation_binding_decision(
            binding,
            config,
            atom_scope_facts=atom_scope_facts,
            atom_input_basis=atom_input_basis,
        )
    if isinstance(binding, RecommendedContinuousRelaxationBinding):
        capabilities = continuous_relaxation_backend_capabilities(config.backend_name)
        return _recommended_continuous_relaxation_binding_decision(
            config,
            capabilities=capabilities,
            atom_scope_facts=atom_scope_facts,
            atom_input_basis=atom_input_basis,
        )

    raise TypeError("continuous relaxation binding request has unsupported type")


def _manual_continuous_relaxation_binding_decision(
    binding: ManualContinuousRelaxationBinding,
    config: ContinuousRelaxationConfig,
    *,
    atom_scope_facts: AtomScopeStateFacts,
    atom_input_basis: AtomInputBasis,
) -> ContinuousRelaxationBindingDecision:
    """Return the explicit binding decision for one legal run."""

    del atom_input_basis
    require_atom_scope_continuous_relaxation_execution(atom_scope_facts)
    return ContinuousRelaxationBindingDecision(
        settings=config.bind(binding.selection),
        reason=ContinuousRelaxationBindingReason.MANUAL_EXPLICIT_SELECTION,
    )


def _recommended_continuous_relaxation_binding_decision(
    config: ContinuousRelaxationConfig,
    *,
    capabilities: ContinuousRelaxationBackendCapabilities,
    atom_scope_facts: AtomScopeStateFacts,
    atom_input_basis: AtomInputBasis,
) -> ContinuousRelaxationBindingDecision:
    """Return one current-domain-aware recommended binding decision."""

    require_atom_scope_continuous_relaxation_execution(atom_scope_facts)
    if capabilities.backend_name == "rdkit":
        return _recommended_rdkit_binding_decision(
            config,
            capabilities=capabilities,
            atom_scope_facts=atom_scope_facts,
            atom_input_basis=atom_input_basis,
        )

    return capabilities.recommended_binding_decision(
        config,
        atom_scope_facts=atom_scope_facts,
    )


def _recommended_rdkit_binding_decision(
    config: ContinuousRelaxationConfig,
    *,
    capabilities: ContinuousRelaxationBackendCapabilities,
    atom_scope_facts: AtomScopeStateFacts,
    atom_input_basis: AtomInputBasis,
) -> ContinuousRelaxationBindingDecision:
    """Return the benchmark-backed RDKit recommendation for one hydrogenated domain."""

    del atom_input_basis
    supported_profiles = capabilities.supported_profiles
    if (
        atom_scope_facts.continuous_region_readiness_facts.chemistry_readiness_facts.hydrogen_coverage_state
        is not HydrogenCoverageState.COMPLETE
    ):
        raise RefinementError(
            "recommended RDKit continuous relaxation binding requires explicit "
            "hydrogens to be fully realized in the included local region before "
            "any force field can be bound"
        )

    if ContinuousRelaxationProfile.RDKIT_MMFF in supported_profiles:
        return ContinuousRelaxationBindingDecision(
            settings=config.bind(ContinuousRelaxationProfile.RDKIT_MMFF),
            reason=ContinuousRelaxationBindingReason.HYDROGENATED_DOMAIN_POLICY,
        )

    return capabilities.recommended_binding_decision(
        config,
        atom_scope_facts=atom_scope_facts,
    )


__all__ = [
    "ContinuousRelaxationBackendCapabilities",
    "ContinuousRelaxationBinding",
    "ContinuousRelaxationBindingDecision",
    "ContinuousRelaxationBindingReason",
    "ContinuousRelaxationSettings",
    "ManualContinuousRelaxationBinding",
    "RecommendedContinuousRelaxationBinding",
    "continuous_relaxation_backend_capabilities",
    "decide_continuous_relaxation_binding",
]
