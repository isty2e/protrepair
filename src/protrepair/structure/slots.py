"""Canonical slot addresses over one structure constitution."""

from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class ChainIndex:
    """Canonical slot index for one chain inside one structure constitution."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("chain index value must be an integer")
        if self.value < 0:
            raise ValueError("chain index value must be non-negative")

    def __index__(self) -> int:
        """Return the underlying integer slot value."""

        return self.value


@dataclass(frozen=True, order=True, slots=True)
class ResidueIndex:
    """Canonical slot index for one residue inside one structure constitution."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("residue index value must be an integer")
        if self.value < 0:
            raise ValueError("residue index value must be non-negative")

    def __index__(self) -> int:
        """Return the underlying integer slot value."""

        return self.value


@dataclass(frozen=True, order=True, slots=True)
class AtomIndex:
    """Canonical slot index for one atom site inside one structure constitution."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("atom index value must be an integer")
        if self.value < 0:
            raise ValueError("atom index value must be non-negative")

    def __index__(self) -> int:
        """Return the underlying integer slot value."""

        return self.value
