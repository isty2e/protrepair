"""Semantic structure labels over residues and atoms."""

from collections.abc import Iterable
from dataclasses import dataclass

from typing_extensions import Self


@dataclass(frozen=True, order=True, slots=True)
class ResidueId:
    """Canonical identifier for a residue in a chain."""

    chain_id: str
    seq_num: int
    insertion_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chain_id, str):
            raise TypeError("chain_id must be a string")
        if not isinstance(self.seq_num, int) or isinstance(self.seq_num, bool):
            raise TypeError("seq_num must be an integer")
        if self.insertion_code is not None and not isinstance(
            self.insertion_code,
            str,
        ):
            raise TypeError("insertion_code must be a string or None")

        chain_id = self.chain_id.strip()
        if not chain_id:
            raise ValueError("chain_id must not be blank")

        insertion_code = self.insertion_code
        if insertion_code is not None:
            insertion_code = insertion_code.strip() or None

        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "insertion_code", insertion_code)

    def display_token(self) -> str:
        """Return a compact human-readable residue token."""

        insertion = self.insertion_code or ""
        return f"{self.chain_id}:{self.seq_num}{insertion}"

    def with_chain_id(self, chain_id: str) -> Self:
        """Return a copy with a different chain identifier."""

        return type(self)(
            chain_id=chain_id,
            seq_num=self.seq_num,
            insertion_code=self.insertion_code,
        )


@dataclass(frozen=True, order=True, slots=True)
class AtomRef:
    """Canonical identifier for one atom within one residue."""

    residue_id: ResidueId
    atom_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError("residue_id must be a ResidueId")
        if not isinstance(self.atom_name, str):
            raise TypeError("atom_name must be a string")

        atom_name = self.atom_name.strip().upper()
        if not atom_name:
            raise ValueError("atom_name must not be blank")

        object.__setattr__(self, "atom_name", atom_name)

    def display_token(self) -> str:
        """Return a compact human-readable atom token."""

        return f"{self.residue_id.display_token()}.{self.atom_name}"

    @classmethod
    def deduplicated(cls, atom_refs: Iterable["AtomRef"]) -> tuple["AtomRef", ...]:
        """Return atom references deduplicated in first-seen order."""

        ordered_atom_refs: list[AtomRef] = []
        seen_atom_refs: set[AtomRef] = set()
        for atom_ref in atom_refs:
            if not isinstance(atom_ref, cls):
                raise TypeError("atom_refs must contain AtomRef values")
            if atom_ref in seen_atom_refs:
                continue

            ordered_atom_refs.append(atom_ref)
            seen_atom_refs.add(atom_ref)

        return tuple(ordered_atom_refs)
