"""Canonical polymer blueprint facets detached from realized structure payload."""

from collections.abc import Collection
from dataclasses import dataclass


def _normalize_optional_chain_id(
    chain_id: str | None,
    *,
    field_name: str,
) -> str | None:
    """Return a normalized assigned or unassigned blueprint chain id."""

    if chain_id is None:
        return None
    if not isinstance(chain_id, str):
        raise TypeError(f"{field_name} must be a string or None")

    return chain_id.strip() or None


@dataclass(frozen=True, slots=True)
class PolymerResidueSlot:
    """One canonical polymer residue slot identified by sequence position."""

    sequence_position: int
    token: str

    def __post_init__(self) -> None:
        sequence_position = self.sequence_position
        if (
            isinstance(sequence_position, bool)
            or not isinstance(sequence_position, int)
        ):
            raise TypeError(
                "polymer residue slots require sequence_position to be an integer"
            )
        if sequence_position <= 0:
            raise ValueError(
                "polymer residue slots require positive sequence positions"
            )

        token = self.token.strip().upper()
        if len(token) != 1 or not token.isalpha():
            raise ValueError(
                "polymer residue slots require one alphabetic residue token"
            )

        object.__setattr__(self, "token", token)


@dataclass(frozen=True, slots=True)
class PolymerChainBlueprint:
    """One canonical polymer chain blueprint ordered by sequence slot.

    A ``None`` chain id denotes an unassigned reference blueprint. Structure-
    attached blueprints must use concrete chain ids matching the constitution.
    """

    chain_id: str | None
    residue_slots: tuple[PolymerResidueSlot, ...]

    def __post_init__(self) -> None:
        chain_id = _normalize_optional_chain_id(
            self.chain_id,
            field_name="polymer chain blueprint chain_id",
        )

        residue_slots = tuple(self.residue_slots)
        if not residue_slots:
            raise ValueError(
                "polymer chain blueprints require at least one residue slot"
            )

        sequence_positions = tuple(
            residue_slot.sequence_position
            for residue_slot in residue_slots
        )
        if len(sequence_positions) != len(set(sequence_positions)):
            raise ValueError(
                "polymer chain blueprints must not repeat sequence positions"
            )
        if sequence_positions != tuple(sorted(sequence_positions)):
            raise ValueError(
                "polymer chain blueprints require strictly increasing sequence "
                "positions"
            )

        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "residue_slots", residue_slots)

    def sequence(self) -> str:
        """Return the canonical sequence token string in slot order."""

        return "".join(
            residue_slot.token
            for residue_slot in self.residue_slots
        )

    def sequence_positions(self) -> tuple[int, ...]:
        """Return canonical sequence positions in slot order."""

        return tuple(
            residue_slot.sequence_position
            for residue_slot in self.residue_slots
        )

    def first_sequence_position(self) -> int:
        """Return the first sequence position covered by this blueprint."""

        return self.residue_slots[0].sequence_position


@dataclass(frozen=True, slots=True)
class PolymerBlueprint:
    """One canonical polymer blueprint over one or more polymer chains."""

    chains: tuple[PolymerChainBlueprint, ...]

    def __post_init__(self) -> None:
        chains = tuple(self.chains)
        if not chains:
            raise ValueError("polymer blueprints require at least one chain blueprint")

        chain_ids = tuple(chain.chain_id for chain in chains)
        if len(chain_ids) != len(set(chain_ids)):
            raise ValueError("polymer blueprints must not repeat chain ids")

        object.__setattr__(self, "chains", chains)

    def chain_ids(self) -> tuple[str | None, ...]:
        """Return chain identifiers in blueprint order."""

        return tuple(chain.chain_id for chain in self.chains)

    def chain(self, chain_id: str | None) -> PolymerChainBlueprint:
        """Return one chain blueprint by id or raise if absent."""

        normalized_chain_id = _normalize_optional_chain_id(
            chain_id,
            field_name="polymer blueprint chain lookup chain_id",
        )
        for chain in self.chains:
            if chain.chain_id == normalized_chain_id:
                return chain

        raise KeyError(f"polymer blueprint has no chain {normalized_chain_id!r}")

    def select_chains(
        self,
        chain_ids: Collection[str | None],
    ) -> "PolymerBlueprint":
        """Return one blueprint restricted to the given chain ids in order."""

        normalized_chain_ids = tuple(
            _normalize_optional_chain_id(
                chain_id,
                field_name="polymer blueprint selected chain_id",
            )
            for chain_id in chain_ids
        )
        return PolymerBlueprint(
            chains=tuple(
                self.chain(chain_id)
                for chain_id in normalized_chain_ids
            )
        )
