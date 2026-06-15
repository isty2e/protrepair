"""Structure ingress normalization policy contracts."""

from dataclasses import dataclass
from enum import Enum


class OccupancyPolicy(str, Enum):
    """Policy for resolving alternate atom occupancies."""

    HIGHEST = "highest"
    LOWEST = "lowest"


class MutationPolicy(str, Enum):
    """Policy for resolving residue-level mutation conflicts."""

    HIGHEST_OCCUPANCY = "highest_occupancy"
    LOWEST_OCCUPANCY = "lowest_occupancy"


class LigandHandling(str, Enum):
    """How one normalization boundary should treat non-water ligands."""

    KEEP = "keep"
    DROP = "drop"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class StructureNormalizationPolicy:
    """Canonical normalization policy for one structure-ingress boundary."""

    occupancy_policy: OccupancyPolicy = OccupancyPolicy.HIGHEST
    mutation_policy: MutationPolicy = MutationPolicy.HIGHEST_OCCUPANCY
    ligand_handling: LigandHandling = LigandHandling.KEEP
    selected_chain_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.occupancy_policy, OccupancyPolicy):
            raise TypeError(
                "structure normalization occupancy_policy must be an "
                "OccupancyPolicy value"
            )

        if not isinstance(self.mutation_policy, MutationPolicy):
            raise TypeError(
                "structure normalization mutation_policy must be a "
                "MutationPolicy value"
            )

        if not isinstance(self.ligand_handling, LigandHandling):
            raise TypeError(
                "structure normalization ligand_handling must be a "
                "LigandHandling value"
            )

        selected_chain_ids = self.selected_chain_ids
        if selected_chain_ids is None:
            return

        normalized_chain_ids: list[str] = []
        seen_chain_ids: set[str] = set()
        for chain_id in selected_chain_ids:
            normalized_chain_id = chain_id.strip()
            if not normalized_chain_id:
                raise ValueError(
                    "structure normalization selected_chain_ids must not contain blanks"
                )

            if normalized_chain_id in seen_chain_ids:
                continue

            normalized_chain_ids.append(normalized_chain_id)
            seen_chain_ids.add(normalized_chain_id)

        object.__setattr__(
            self,
            "selected_chain_ids",
            tuple(normalized_chain_ids) or None,
        )

    def selects_chain(self, chain_id: str) -> bool:
        """Return whether one chain should be normalized by this policy."""

        if self.selected_chain_ids is None:
            return True

        return chain_id.strip() in self.selected_chain_ids

    def keeps_ligands(self) -> bool:
        """Return whether the boundary preserves ligands."""

        return self.ligand_handling is LigandHandling.KEEP

    def drops_ligands(self) -> bool:
        """Return whether the boundary silently drops ligands."""

        return self.ligand_handling is LigandHandling.DROP

    def rejects_ligands(self) -> bool:
        """Return whether the boundary rejects ligand-bearing payloads."""

        return self.ligand_handling is LigandHandling.REJECT


__all__ = [
    "LigandHandling",
    "MutationPolicy",
    "OccupancyPolicy",
    "StructureNormalizationPolicy",
]
