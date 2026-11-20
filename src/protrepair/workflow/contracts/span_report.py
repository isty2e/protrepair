"""Returned outcomes of explicit donor-backed span requests."""

from dataclasses import dataclass
from enum import Enum

from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure.labels import ResidueId


class ExternalSpanReconstructionStatus(str, Enum):
    """Disposition of one donor request in the returned workflow result."""

    APPLIED = "applied"
    ALREADY_PRESENT = "already_present"
    PRESENT_WITHOUT_THIS_REQUEST = "present_without_this_request"
    NOT_RETAINED = "not_retained"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True, slots=True)
class ExternalSpanReconstructionOutcome:
    """Returned coverage and trial evidence for one normalized request.

    Parameters
    ----------
    request_index : int
        Zero-based index in the normalized external reconstruction request tuple.
    scope : AbsentResidueSpanScope
        Source anchors and requested target positions.
    status : ExternalSpanReconstructionStatus
        Disposition relative to the returned structure.
    missing_residue_ids : tuple[ResidueId, ...]
        Requested positions absent from the returned structure, in scope order.
    attempt_count, successful_attempt_count : int
        Executed trials and successful grafts across the entire search.
    last_failure : ValidationIssue or None
        Most recent refusal for this request, including unselected trials.

    Raises
    ------
    TypeError
        A scope, status or failure value is not of the expected type.
    ValueError
        Index, counts, missing positions or disposition are inconsistent.
    """

    request_index: int
    scope: AbsentResidueSpanScope
    status: ExternalSpanReconstructionStatus
    missing_residue_ids: tuple[ResidueId, ...]
    attempt_count: int
    successful_attempt_count: int
    last_failure: ValidationIssue | None

    def __post_init__(self) -> None:
        for value in (
            self.request_index,
            self.attempt_count,
            self.successful_attempt_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "span outcome indices and counts must be non-negative integers"
                )
        if self.successful_attempt_count > self.attempt_count:
            raise ValueError("successful span trials cannot exceed executed trials")
        if not isinstance(self.scope, AbsentResidueSpanScope):
            raise TypeError("span outcomes require an absent-residue span scope")
        if not isinstance(self.status, ExternalSpanReconstructionStatus):
            raise TypeError("span outcomes require a canonical status")
        if self.last_failure is not None and (
            not isinstance(self.last_failure, ValidationIssue)
            or self.last_failure.kind
            is not ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
            or self.last_failure.scope.residue_ids != self.scope.absent_residue_ids
        ):
            raise TypeError("span failure evidence must belong to the requested span")
        if (self.last_failure is not None) != (
            self.attempt_count > self.successful_attempt_count
        ):
            raise ValueError("span failure evidence must agree with trial counts")

        missing = tuple(self.missing_residue_ids)
        missing_set = frozenset(missing)
        if missing != tuple(
            residue_id
            for residue_id in self.scope.absent_residue_ids
            if residue_id in missing_set
        ):
            raise ValueError("missing span positions must follow the request scope")
        complete = self.status in (
            ExternalSpanReconstructionStatus.APPLIED,
            ExternalSpanReconstructionStatus.ALREADY_PRESENT,
            ExternalSpanReconstructionStatus.PRESENT_WITHOUT_THIS_REQUEST,
        )
        if complete == bool(missing):
            raise ValueError("span disposition must agree with the missing positions")
        if (
            self.status
            in (
                ExternalSpanReconstructionStatus.APPLIED,
                ExternalSpanReconstructionStatus.NOT_RETAINED,
            )
            and not self.successful_attempt_count
        ):
            raise ValueError(
                "applied and not-retained spans require a successful trial"
            )
        if self.status is ExternalSpanReconstructionStatus.FAILED and (
            not self.attempt_count or self.successful_attempt_count
        ):
            raise ValueError("a failed span requires trials that all refused")
        if (
            self.status is ExternalSpanReconstructionStatus.NOT_ATTEMPTED
            and self.attempt_count
        ):
            raise ValueError("an unattempted span cannot have trial evidence")
        object.__setattr__(self, "missing_residue_ids", missing)


@dataclass(frozen=True, slots=True)
class ExternalSpanReconstructionReport:
    """One outcome per normalized external span request.

    Parameters
    ----------
    outcomes : tuple[ExternalSpanReconstructionOutcome, ...]
        Outcomes in normalized request order, including unexecuted requests.

    Raises
    ------
    TypeError
        An outcome is not a canonical span outcome.
    ValueError
        Outcomes are empty or out of request order.
    """

    outcomes: tuple[ExternalSpanReconstructionOutcome, ...]

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError(
                "a span reconstruction report requires at least one request"
            )
        for index, outcome in enumerate(outcomes):
            if not isinstance(outcome, ExternalSpanReconstructionOutcome):
                raise TypeError(
                    "span reconstruction reports require canonical outcomes"
                )
            if outcome.request_index != index:
                raise ValueError(
                    "span reconstruction outcomes must follow normalized request order"
                )
        object.__setattr__(self, "outcomes", outcomes)
