"""Directed peptide port selection for a cap-placement workflow."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.sequence_boundary import DepositedPeptideBoundary


class PeptideEnd(str, Enum):
    """The amide N or carbonyl C side of an observed peptide residue."""

    N = "N"
    C = "C"


@dataclass(frozen=True, slots=True)
class PeptidePort:
    """A directed end of a biological residue.

    Parameters
    ----------
    residue_id : ResidueId
        Original polymer residue, distinct from the new cap's address.
    end : PeptideEnd
        N for an acetyl cap, C for an N-methylamide cap.
    """

    residue_id: ResidueId
    end: PeptideEnd

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId) or not isinstance(
            self.end, PeptideEnd
        ):
            raise TypeError("a peptide port requires a residue ID and a peptide end")

    @property
    def atom_ref(self) -> AtomRef:
        """Return the attachment atom, even when it still needs heavy repair."""
        return AtomRef(self.residue_id, self.end.value)

    @property
    def expected_component(self) -> str:
        """Component that would close this directed port."""
        return "ACE" if self.end is PeptideEnd.N else "NME"


@dataclass(frozen=True, slots=True, kw_only=True)
class PeptideCappingRequest:
    """Select directed flanks without assuming that they are chemically vacant.

    Parameters
    ----------
    ports : tuple[PeptidePort, ...] or None
        Distinct ports, or None to consider all exposed flanks.

    Raises
    ------
    TypeError
        A selection does not contain canonical peptide ports.
    ValueError
        An explicit selection is empty or includes a duplicate.
    """

    ports: tuple[PeptidePort, ...] | None = None

    def __post_init__(self) -> None:
        if self.ports is None:
            return
        if not isinstance(self.ports, tuple) or any(
            not isinstance(p, PeptidePort) for p in self.ports
        ):
            raise TypeError("a peptide cap selection requires a tuple of peptide ports")
        if not self.ports or len(set(self.ports)) != len(self.ports):
            raise ValueError(
                "an explicit cap selection requires nonempty, distinct ports"
            )

    @classmethod
    def only(cls, ports: Iterable[PeptidePort]) -> "PeptideCappingRequest":
        """Consider exactly these biological ports under the usual evidence checks."""
        return cls(ports=tuple(ports))


class PeptideCapStatus(str, Enum):
    """Closure and attribution on the selected workflow result."""

    APPLIED = "applied"
    ALREADY_CAPPED = "already_capped"
    ATTACHED_UNVERIFIED = "attached_unverified"
    OCCUPIED = "occupied"
    OMITTED = "omitted"
    REFUSED = "refused"
    NOT_RETAINED = "not_retained"
    SEARCH_LIMIT = "search_limit"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True, slots=True)
class PeptideCapOutcome:
    """One input port and its outcome, independently of other ports' success.

    The input boundary and reserved address identify the proposed edit. A named
    omitted partner can have an address but no attachment in the final graph.
    """

    port: PeptidePort
    status: PeptideCapStatus
    reason: str
    input_boundary: DepositedPeptideBoundary | None
    reserved_cap_id: ResidueId | None
    cap_id: ResidueId | None
    attachment: tuple[AtomRef, AtomRef] | None
    detail: str = ""
    pose_refusals: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class PeptideCappingReport:
    """Outcomes in residue and attachment-atom order, including unclassified ends."""

    outcomes: tuple[PeptideCapOutcome, ...]
