"""Parser compatibility facts for canonical structures."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.parser_readability import (
    measure_rdkit_no_conect_sanitize_readability_metrics,
)
from protrepair.state.structure_axes import (
    ParserCompatibilityProfile,
    ParserCompatibilityState,
)
from protrepair.structure.aggregate import ProteinStructure

__all__ = ["StructureParserCompatibilityFacts"]


@dataclass(frozen=True, slots=True)
class StructureParserCompatibilityFacts:
    """Parser-profile compatibility observations over one canonical structure.

    These facts are not chemistry-readiness truth. They record whether a
    specific downstream parser profile accepts the current coordinates without
    inferring invalid proximity bonds.
    """

    carrier: ProteinStructure
    profile: ParserCompatibilityProfile
    compatibility_state: ParserCompatibilityState
    extra_proximity_bond_count: int = 0
    extra_heavy_proximity_bond_count: int = 0

    def __post_init__(self) -> None:
        if self.extra_proximity_bond_count < 0:
            raise ValueError("extra proximity-bond count must be non-negative")
        if self.extra_heavy_proximity_bond_count < 0:
            raise ValueError("extra heavy proximity-bond count must be non-negative")

    def has_parser_visible_proximity_burden(self) -> bool:
        """Return whether parser-visible false proximity remains observed."""

        return (
            self.compatibility_state.is_incompatible()
            and self.extra_proximity_bond_count > 0
        )

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> "StructureParserCompatibilityFacts":
        """Derive parser-profile compatibility facts from one structure."""

        metrics = measure_rdkit_no_conect_sanitize_readability_metrics(
            structure,
            component_library=component_library,
        )
        if metrics.sanitize_readable is True:
            compatibility_state = ParserCompatibilityState.COMPATIBLE
        elif metrics.sanitize_readable is False:
            compatibility_state = ParserCompatibilityState.INCOMPATIBLE
        else:
            compatibility_state = ParserCompatibilityState.NOT_OBSERVED

        return cls(
            carrier=structure,
            profile=ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE,
            compatibility_state=compatibility_state,
            extra_proximity_bond_count=metrics.extra_proximity_bond_count,
            extra_heavy_proximity_bond_count=metrics.extra_heavy_proximity_bond_count,
        )
