"""Generic lowering contracts over semantic scopes."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from protrepair.scope.base import Scope

InputScopeT = TypeVar("InputScopeT", bound=Scope)
CarrierT = TypeVar("CarrierT")
LoweringPolicyT = TypeVar("LoweringPolicyT")
OutputT = TypeVar("OutputT")


class ScopeLowering(
    ABC,
    Generic[InputScopeT, CarrierT, LoweringPolicyT, OutputT],
):
    """Carrier- and policy-dependent lowering from semantic scope to one output."""

    @abstractmethod
    def supports(self, scope: Scope) -> bool:
        """Return whether this lowering can realize the given semantic scope."""

    @abstractmethod
    def lower(
        self,
        scope: InputScopeT,
        *,
        carrier: CarrierT,
        policy: LoweringPolicyT | None = None,
    ) -> OutputT:
        """Realize one output from semantic scope, carrier, and policy."""
