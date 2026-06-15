"""Protocol types for side-chain packing backend implementations."""

from typing import Protocol

from protrepair.transformer.packing.domain import (
    PackingCapabilities,
    PackingPlan,
    PackingResult,
)


class SidechainPackingBackend(Protocol):
    """Internal protocol implemented by side-chain packing backends."""

    def capabilities(self) -> PackingCapabilities:
        """Return declared backend capabilities."""

        ...

    def pack(self, plan: PackingPlan) -> PackingResult:
        """Execute one canonical side-chain packing plan."""

        ...
