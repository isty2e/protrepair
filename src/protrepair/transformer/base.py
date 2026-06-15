"""Canonical transformer ABCs over projected state carriers and contexts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from typing_extensions import Self

from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.scope import Scope

if TYPE_CHECKING:
    from protrepair.transformer.context import ProteinTransformationContext

OutT = TypeVar("OutT")


class ContextOperation(ABC, Generic[OutT]):
    """Operation over canonical transformation contexts without carrier lift."""

    @abstractmethod
    def is_applicable(
        self,
        context: "ProteinTransformationContext",
    ) -> bool:
        """Return whether this transformer is semantically applicable."""

    @abstractmethod
    def transform(
        self,
        context: "ProteinTransformationContext",
    ) -> OutT:
        """Return the transformed output for one canonical transformation context."""


class DeterministicContextOperation(
    ContextOperation[OutT],
):
    """Deterministic operation over one canonical transformation context."""


PayloadT = TypeVar("PayloadT")
ProvenanceT = TypeVar("ProvenanceT")
ContextT = TypeVar("ContextT")
CarrierT = TypeVar("CarrierT")
DomainT = TypeVar("DomainT")
CodomainT = TypeVar("CodomainT")
StateValueT = TypeVar("StateValueT")


@dataclass(frozen=True, slots=True)
class TransformationCandidate(Generic[PayloadT, ProvenanceT]):
    """One branching/search candidate with payload and typed provenance."""

    payload: PayloadT
    provenance: ProvenanceT
    heuristic_score_hint: float | None = None


class BranchingContextOperation(
    ContextOperation[
        tuple[TransformationCandidate[PayloadT, ProvenanceT], ...]
    ],
    Generic[PayloadT, ProvenanceT],
):
    """Branching operation over one canonical transformation context."""


@dataclass(frozen=True, slots=True)
class ProjectedDomainState(Generic[DomainT]):
    """Projected transformer-local domain over one canonical semantic scope."""

    scope: Scope
    state: DomainT


@dataclass(frozen=True, slots=True)
class ProjectedCodomainState(Generic[CodomainT]):
    """Projected transformer-local codomain plus lifted diagnostics delta."""

    scope: Scope
    state: CodomainT
    repairs: tuple[RepairEvent, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "repairs", tuple(self.repairs))
        object.__setattr__(self, "issues", tuple(self.issues))


class ProteinTransformer(
    ABC,
    Generic[
        ContextT,
        CarrierT,
        DomainT,
        CodomainT,
        StateValueT,
    ],
):
    """Template-method transformer over projected state domain and codomain."""

    @abstractmethod
    def project_domain_state(
        self,
        carrier: CarrierT,
        *,
        context: ContextT,
    ) -> ProjectedDomainState[DomainT]:
        """Project the current carrier into one transformer-local domain state."""

    def accepts_projected_domain(
        self,
        projected_domain: ProjectedDomainState[DomainT],
        *,
        context: ContextT,
    ) -> bool:
        """Return whether this transformer accepts one projected domain state."""

        del projected_domain, context
        return True

    @abstractmethod
    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[DomainT],
        *,
        carrier: CarrierT,
        context: ContextT,
    ) -> ProjectedCodomainState[CodomainT]:
        """Transform one projected domain state into one projected codomain state."""

    @abstractmethod
    def lift_projected_codomain(
        self,
        carrier: CarrierT,
        projected_codomain: ProjectedCodomainState[CodomainT],
        *,
        context: ContextT,
    ) -> CarrierT:
        """Lift one projected codomain state back into the carrier state."""

    def execute(
        self,
        carrier: CarrierT,
        *,
        context: ContextT,
    ) -> CarrierT:
        """Execute this transformer through project-transform-lift."""

        projected_domain = self.project_domain_state(
            carrier,
            context=context,
        )
        if not self.accepts_projected_domain(
            projected_domain,
            context=context,
        ):
            raise ValueError(
                "projected transformer domain was rejected by applicability check"
            )

        projected_codomain = self.transform_projected_domain(
            projected_domain,
            carrier=carrier,
            context=context,
        )
        return self.lift_projected_codomain(
            carrier,
            projected_codomain,
            context=context,
        )

    def proposal_family(self) -> type[Self]:
        """Return the frontier proposal family key for this transformer."""

        return type(self)

    def is_completion_transformer(self) -> bool:
        """Return whether this transformer belongs to the completion family."""

        return False
