"""Final per-site disposition of a source-compatible protonation request."""

from dataclasses import dataclass, field
from enum import Enum

from protrepair.chemistry.microstate.resolution import (
    MicrostateResolutionStatus,
    SiteProtonationReason,
)
from protrepair.chemistry.microstate.selection import SiteProtonation


class SiteProtonationStatus(str, Enum):
    """Outcome on the selected branch, separate from chemical feasibility."""

    ALREADY_SATISFIED = "already_satisfied"
    APPLIED = "applied"
    SATISFIED_BY_OTHER_WORK = "satisfied_by_other_work"
    REFUSED = "refused"
    PLACEMENT_FAILED = "placement_failed"
    NOT_RETAINED = "not_retained"
    NOT_ATTEMPTED = "not_attempted"
    SEARCH_LIMIT = "search_limit"


@dataclass(frozen=True, slots=True)
class SiteProtonationOutcome:
    """Chemical feasibility and execution outcome on the returned structure.

    Parameters
    ----------
    request : SiteProtonation
        Caller selection on the normalized input residue.
    assessment : MicrostateResolutionStatus
        Chemical feasibility against the supplied evidence and final scaffold.
    classes : tuple
        Compatible (net site charge, named parent H counts) pairs; resonance
        drawings with the same proton positions count once.
    status : SiteProtonationStatus
        Whether and how the final structure realizes the selection.
    details : tuple[str, ...]
        Evidence for a refusal, placement failure or boundary assumption.
    reason : SiteProtonationReason or None
        Specific chemical or input refusal cause, when available.
    """

    request: SiteProtonation
    assessment: MicrostateResolutionStatus
    classes: tuple[tuple[int, tuple[tuple[str, int], ...]], ...]
    status: SiteProtonationStatus
    details: tuple[str, ...] = ()
    reason: SiteProtonationReason | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class SiteProtonationReport:
    """One outcome per input request, in caller order."""

    outcomes: tuple[SiteProtonationOutcome, ...]
