"""Candidate lineage values for local refinement."""

from dataclasses import dataclass
from enum import Enum

from protrepair.structure.slots import AtomIndex


class CandidateConstructionStageKind(str, Enum):
    """Closed internal stages that can construct one refinement candidate."""

    DISCRETE_PREPARATION = "discrete_preparation"
    PARSER_WITNESS_PRE_UNTANGLE = "parser_witness_pre_untangle"
    DISCRETE_SEEDING = "discrete_seeding"


@dataclass(frozen=True, slots=True)
class CandidateConstructionStep:
    """One internal candidate-construction stage plus the atoms it moved."""

    kind: CandidateConstructionStageKind
    moved_atom_indices: tuple[AtomIndex, ...] = ()

    def __post_init__(self) -> None:
        ordered_atom_indices: list[AtomIndex] = []
        seen_atom_indices: set[AtomIndex] = set()
        for atom_index in self.moved_atom_indices:
            if atom_index in seen_atom_indices:
                continue

            ordered_atom_indices.append(atom_index)
            seen_atom_indices.add(atom_index)

        object.__setattr__(self, "moved_atom_indices", tuple(ordered_atom_indices))


@dataclass(frozen=True, slots=True)
class RefinementCandidateLineage:
    """Internal lineage of discrete stages applied before backend execution."""

    steps: tuple[CandidateConstructionStep, ...] = ()

    def with_step(
        self,
        *,
        kind: CandidateConstructionStageKind,
        moved_atom_indices: tuple[AtomIndex, ...],
    ) -> "RefinementCandidateLineage":
        """Return lineage extended by one construction stage when it moved atoms."""

        if not moved_atom_indices:
            return self

        return type(self)(
            steps=(
                *self.steps,
                CandidateConstructionStep(
                    kind=kind,
                    moved_atom_indices=moved_atom_indices,
                ),
            )
        )

    def moved_atom_indices(self) -> tuple[AtomIndex, ...]:
        """Return all pre-backend atom movement in canonical deduplicated order."""

        ordered_atom_indices: list[AtomIndex] = []
        seen_atom_indices: set[AtomIndex] = set()
        for step in self.steps:
            for atom_index in step.moved_atom_indices:
                if atom_index in seen_atom_indices:
                    continue

                ordered_atom_indices.append(atom_index)
                seen_atom_indices.add(atom_index)

        return tuple(ordered_atom_indices)

    def has_step_kind(self, kind: CandidateConstructionStageKind) -> bool:
        """Return whether this lineage contains one construction stage kind."""

        return any(step.kind is kind for step in self.steps)
