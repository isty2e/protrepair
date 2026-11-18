"""Explicit X-H length requests and their workflow results."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from protrepair.analysis.hydrogen_length_reference import (
    HydrogenLengthConvention,
    HydrogenLengthReference,
    index_hydrogen_length_references,
)
from protrepair.analysis.hydrogen_lengths import HydrogenLengthObservation
from protrepair.scope import Scope, WholeStructureScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.structure.labels import AtomRef


@dataclass(frozen=True, slots=True, init=False)
class HydrogenLengthRequest:
    """Check existing H/D/T lengths and optionally adjust them radially.

    Parameters
    ----------
    references : Iterable[HydrogenLengthReference]
        At least one caller-supplied reference, at most one per component.
    scope : Scope or None, default=None
        Observed scope at the normalized workflow input. Only H/D/T sites
        already present there are selected. None selects the whole structure.
    convention : HydrogenLengthConvention, default=UNKNOWN
        Coordinate basis for grading; an unknown basis is read-only.
    repair : bool, default=False
        Authorize a guarded length-only move after branch selection.

    Raises
    ------
    TypeError
        A supplied reference, scope, convention or repair flag is unsupported.
    ValueError
        References are empty or repeated, or repair has no declared convention.
    """

    references: tuple[HydrogenLengthReference, ...]
    scope: Scope
    convention: HydrogenLengthConvention
    repair: bool

    def __init__(
        self,
        references: Iterable[HydrogenLengthReference],
        *,
        scope: Scope | None = None,
        convention: HydrogenLengthConvention = HydrogenLengthConvention.UNKNOWN,
        repair: bool = False,
    ) -> None:
        selected_scope = WholeStructureScope() if scope is None else scope
        if not OBSERVED_ATOM_SCOPE_LOWERING.supports(selected_scope):
            raise TypeError("hydrogen length requests require an observed atom scope")
        if not isinstance(convention, HydrogenLengthConvention):
            raise TypeError("convention must be a HydrogenLengthConvention")
        if not isinstance(repair, bool):
            raise TypeError("repair must be a bool")
        if repair and convention is HydrogenLengthConvention.UNKNOWN:
            raise ValueError("hydrogen length repair requires a declared convention")
        by_component = index_hydrogen_length_references(references)
        if not by_component:
            raise ValueError("supply at least one hydrogen length reference")

        object.__setattr__(self, "references", tuple(by_component.values()))
        object.__setattr__(self, "scope", selected_scope)
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "repair", repair)


class HydrogenLengthRepairStatus(str, Enum):
    """Disposition of one selected site in the explicit repair request."""

    IN_WINDOW = "in_window"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    REPAIRED = "repaired"


@dataclass(frozen=True, slots=True)
class HydrogenLengthRepairOutcome:
    """Repair disposition for one H with an explanation when it cannot move.

    Parameters
    ----------
    hydrogen : AtomRef
        Identity selected on the normalized workflow input.
    status : HydrogenLengthRepairStatus
        Whether no move was needed, the site could not enter the batch,
        the batch was rejected, or the move was retained.
    detail : str or None
        Refusal reason when the site is unavailable or the batch was rejected.
    """

    hydrogen: AtomRef
    status: HydrogenLengthRepairStatus
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class HydrogenLengthReport:
    """Final observations and optional dispositions of the same selected sites.

    Parameters
    ----------
    observations : tuple[HydrogenLengthObservation, ...]
        Grades on the returned structure, in input selection order. An empty
        result means no input H/D/T sites were selected.
    repair_outcomes : tuple[HydrogenLengthRepairOutcome, ...] or None
        Disposition of each selected site in the same order, or None when
        repair was not requested.
    """

    observations: tuple[HydrogenLengthObservation, ...]
    repair_outcomes: tuple[HydrogenLengthRepairOutcome, ...] | None = None

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        object.__setattr__(self, "observations", observations)
        if self.repair_outcomes is not None:
            outcomes = tuple(self.repair_outcomes)
            if tuple(row.hydrogen for row in outcomes) != tuple(
                row.hydrogen for row in observations
            ):
                raise ValueError(
                    "repair outcomes must match the observed sites in order"
                )
            object.__setattr__(self, "repair_outcomes", outcomes)
