"""Positive evidence about the deposited sequence at a peptide port."""

from dataclasses import dataclass
from enum import Enum

from protrepair.structure.constitution import ResidueRole, StructureConstitution
from protrepair.structure.labels import AtomRef


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
