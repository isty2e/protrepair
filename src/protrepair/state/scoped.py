"""State assertions bound to semantic scopes, with or without explicit carriers."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from protrepair.scope import Scope

CarrierT = TypeVar("CarrierT")
StateValueT = TypeVar("StateValueT")


@dataclass(frozen=True, slots=True)
class CarrierScopedState(Generic[CarrierT, StateValueT]):
    """One basis-level state fact over an explicit carrier and semantic scope."""

    carrier: CarrierT
    scope: Scope
    value: StateValueT

    def __post_init__(self) -> None:
        if not isinstance(self.scope, Scope):
            raise TypeError(
                "carrier-scoped states require a canonical Scope value"
            )

    def without_carrier(self) -> "ScopedState[StateValueT]":
        """Return the carrier-less scoped state value for this fact."""

        return ScopedState(scope=self.scope, value=self.value)


@dataclass(frozen=True, slots=True)
class ScopedState(Generic[StateValueT]):
    """One carrier-less scoped state value, typically used at request boundaries."""

    scope: Scope
    value: StateValueT

    def __post_init__(self) -> None:
        if not isinstance(self.scope, Scope):
            raise TypeError("scoped states require a canonical Scope value")

    def for_carrier(
        self,
        carrier: CarrierT,
    ) -> CarrierScopedState[CarrierT, StateValueT]:
        """Bind this scoped state value to one explicit carrier."""

        return CarrierScopedState(
            carrier=carrier,
            scope=self.scope,
            value=self.value,
        )
