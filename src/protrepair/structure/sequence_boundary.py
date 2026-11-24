"""Positive evidence about the deposited sequence at a peptide port."""

from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from protrepair.structure.constitution import ResidueRole, StructureConstitution
from protrepair.structure.labels import AtomRef, ResidueId


@dataclass(frozen=True, slots=True)
class DepositedPeptideSequence:
    """Sequence and original coordinate positions for one proven chain instance.

    Parameters
    ----------
    instance_id, polymer_type : str
        Deposited instance and peptide class (PDB only identifies the chain).
    components : tuple[str, ...]
        The construct's monomers, including positions without coordinates.
    positions : tuple of (ResidueId, int, int)
        Earliest and latest feasible one-based position per raw coordinate slot.
    trailing_monomers : tuple of (ResidueId, str)
        Post-TER slots that might occupy part of a PDB declaration.
    scheme : tuple of (int, str, str, str) or None
        Deposited position, PDB number, author number and insertion-code CIF
        tokens. None denotes a contradictory scheme; an empty tuple denotes no rows.
    unavailable_cap_numbers : frozenset[int] or None
        Coordinate, missing-residue or scheme numbers above the extension range.
        None means that a new cap address cannot be certified from this path.
    """

    instance_id: str
    polymer_type: str
    components: tuple[str, ...]
    positions: tuple[tuple[ResidueId, int, int], ...]
    trailing_monomers: tuple[tuple[ResidueId, str], ...] = ()
    scheme: tuple[tuple[int, str, str, str], ...] | None = ()
    unavailable_cap_numbers: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if not self.instance_id or not self.positions or not self.components:
            raise ValueError(
                "a deposited peptide path needs observed and declared positions"
            )
        if any(
            rid.chain_id != self.chain_id or not 1 <= lo <= hi <= len(self.components)
            for rid, lo, hi in self.positions
        ) or any(
            b_lo <= a_lo or b_hi <= a_hi
            for (_, a_lo, a_hi), (_, b_lo, b_hi) in pairwise(self.positions)
        ):
            raise ValueError(
                "deposited peptide positions must increase within one chain"
            )
        if self.scheme is not None and any(
            not 1 <= pos <= len(self.components) for pos, *_ in self.scheme
        ):
            raise ValueError("address scheme lies outside the deposited sequence")
        numbers = self.unavailable_cap_numbers
        if numbers is not None:
            if not isinstance(numbers, frozenset) or any(
                type(n) is not int for n in numbers
            ):
                raise TypeError(
                    "cap address exclusions require a frozen set of integers"
                )
            first = self.first_cap_number
            object.__setattr__(
                self,
                "unavailable_cap_numbers",
                None
                if self.scheme is None
                else frozenset(n for n in numbers if n >= first),
            )

    @property
    def chain_id(self) -> str:
        """Author chain used by the observed biological slots."""
        return self.positions[0][0].chain_id

    @property
    def first_cap_number(self) -> int:
        """First number beyond ordinary terminal extensions of the original peptide."""
        return (
            max(rid.seq_num for rid, _, _ in self.positions) + len(self.components) + 1
        )


class PeptideSequenceBoundary(str, Enum):
    """What the deposited sequence establishes in the direction of a port."""

    CONSTRUCT_END = "construct_end"
    MISSING_CONTINUATION = "missing_continuation"


@dataclass(frozen=True, slots=True)
class DepositedPeptideBoundary:
    """Bind a directed sequence fact to the original residue, not its chemistry.

    Parameters
    ----------
    port : AtomRef
        N or C of the observed residue; it need not carry a coordinate.
    component_id : str
        Deposited component at that residue.
    instance_id : str
        Label asym ID in mmCIF, or the normalized SEQRES chain ID in PDB.
    sequence_length : int
        Length of the declared construct.
    first_position, last_position : int
        Earliest and latest one-based positions compatible with the source.
        They coincide for an explicitly labelled mmCIF residue.
    kind : PeptideSequenceBoundary
        Construct end or a continuation requiring missing coordinates.

    Raises
    ------
    ValueError
        The port, identity, or interval cannot describe the stated boundary.
    """

    port: AtomRef
    component_id: str
    instance_id: str
    sequence_length: int
    first_position: int
    last_position: int
    kind: PeptideSequenceBoundary

    def __post_init__(self) -> None:
        if (
            self.port.atom_name not in {"N", "C"}
            or not self.component_id.strip()
            or not self.instance_id.strip()
            or not isinstance(self.kind, PeptideSequenceBoundary)
            or not 1
            <= self.first_position
            <= self.last_position
            <= self.sequence_length
        ):
            raise ValueError("invalid deposited peptide boundary")
        end = 1 if self.port.atom_name == "N" else self.sequence_length
        if self.kind is PeptideSequenceBoundary.CONSTRUCT_END:
            if not self.first_position == self.last_position == end:
                raise ValueError("a construct end requires a fixed terminal position")
        elif self.first_position <= end <= self.last_position:
            raise ValueError("a missing continuation cannot include the construct end")

    def identifies_polymer_in(self, constitution: StructureConstitution) -> bool:
        """Check the observed component and role without requiring port atoms.

        Parameters
        ----------
        constitution : StructureConstitution
            Selected residue identities to test.

        Returns
        -------
        bool
            True only for the same component in a polymer chain.
        """
        rid = self.port.residue_id
        site = constitution.residue_or_ligand(rid)
        return (
            site is not None
            and site.component_id == self.component_id
            and constitution.role_at(constitution.residue_index(rid))
            is ResidueRole.POLYMER
        )
