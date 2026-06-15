"""Workflow action capability domain model."""

from dataclasses import dataclass
from enum import Enum

from protrepair.scope import Scope
from protrepair.workflow.planning.capability.families import (
    WorkflowActionActivationMode,
    WorkflowActionDonorRequirement,
    WorkflowActionEffectClass,
    WorkflowActionLocality,
    WorkflowCapabilityDeficitFamily,
    WorkflowCapabilityFactFamily,
)
from protrepair.workflow.planning.capability.goals import WorkflowGoalDescriptor


@dataclass(frozen=True, slots=True)
class WorkflowActionCapability:
    """Planner-readable capability contract for one workflow action family."""

    reducible_deficit_families: tuple[WorkflowCapabilityDeficitFamily, ...]
    read_fact_families: tuple[WorkflowCapabilityFactFamily, ...]
    activation_mode: WorkflowActionActivationMode
    effect_class: WorkflowActionEffectClass
    supported_localities: tuple[WorkflowActionLocality, ...]
    blocking_deficit_families: tuple[WorkflowCapabilityDeficitFamily, ...] = ()
    donor_requirement: WorkflowActionDonorRequirement = (
        WorkflowActionDonorRequirement.NONE
    )
    supported_goals: tuple[WorkflowGoalDescriptor, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reducible_deficit_families",
            tuple(self.reducible_deficit_families),
        )
        object.__setattr__(
            self,
            "blocking_deficit_families",
            tuple(self.blocking_deficit_families),
        )
        object.__setattr__(
            self,
            "read_fact_families",
            tuple(self.read_fact_families),
        )
        object.__setattr__(
            self,
            "supported_localities",
            tuple(self.supported_localities),
        )
        object.__setattr__(self, "supported_goals", tuple(self.supported_goals))

    def supports_proposition(
        self,
        *,
        scope: Scope,
        value: Enum,
    ) -> bool:
        """Return whether this capability can advance one state proposition."""

        return any(
            descriptor.matches(scope=scope, value=value)
            for descriptor in self.supported_goals
        )

    def can_reduce_deficit_family(
        self,
        deficit_family: WorkflowCapabilityDeficitFamily,
    ) -> bool:
        """Return whether this capability can reduce one deficit family."""

        return deficit_family in self.reducible_deficit_families

    def can_be_blocked_by_deficit_family(
        self,
        deficit_family: WorkflowCapabilityDeficitFamily,
    ) -> bool:
        """Return whether this capability may be blocked by one deficit family."""

        return deficit_family in self.blocking_deficit_families

    def supports_locality(
        self,
        locality: WorkflowActionLocality,
    ) -> bool:
        """Return whether this capability supports one locality scale."""

        return locality in self.supported_localities

    def reads_fact_family(
        self,
        fact_family: WorkflowCapabilityFactFamily,
    ) -> bool:
        """Return whether this capability reads one canonical fact family."""

        return fact_family in self.read_fact_families
