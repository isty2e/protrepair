"""Neutral workflow contracts for donor-backed span selection policy."""

from dataclasses import dataclass

from protrepair.relation.blueprint import StructureBlueprintCoverageGap


@dataclass(frozen=True, slots=True)
class ExternalSpanGapSelectionPolicy:
    """Select which blueprint gaps may become donor-based span reconstructions."""

    include_internal: bool = True
    include_prefix_terminal: bool = False
    include_suffix_terminal: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "include_internal",
            "include_prefix_terminal",
            "include_suffix_terminal",
        ):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, bool):
                raise TypeError(
                    f"external span gap selection policies require boolean {field_name}"
                )

    @classmethod
    def internal_only(cls) -> "ExternalSpanGapSelectionPolicy":
        """Return the default policy that admits only internal gaps."""

        return cls()

    def selects_gap(
        self,
        gap: StructureBlueprintCoverageGap,
    ) -> bool:
        """Return whether one canonical coverage gap should become a spec."""

        if not isinstance(gap, StructureBlueprintCoverageGap):
            raise TypeError(
                "external span gap selection policies require a "
                "StructureBlueprintCoverageGap"
            )
        if gap.is_internal():
            return self.include_internal
        if gap.is_prefix_terminal():
            return self.include_prefix_terminal
        return self.include_suffix_terminal


__all__ = ["ExternalSpanGapSelectionPolicy"]
