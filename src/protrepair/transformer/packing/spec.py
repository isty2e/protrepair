"""Side-chain packing transformer request specifications."""

from dataclasses import dataclass
from enum import Enum

from typing_extensions import Self

from protrepair.structure.labels import ResidueId


class PackingScope(str, Enum):
    """How much of a structure one packing transformation may modify."""

    FULL = "full"
    LOCAL = "local"


class PackingMode(str, Enum):
    """Operational intent for one side-chain packing transformation."""

    PACK = "pack"
    REFINE = "refine"


@dataclass(frozen=True, slots=True)
class PackingSpec:
    """Boundary configuration for one side-chain packing transformation."""

    backend_name: str
    mode: PackingMode = PackingMode.PACK
    scope: PackingScope = PackingScope.FULL
    target_sequence: str | None = None
    mutable_residue_ids: tuple[ResidueId, ...] | None = None
    frozen_residue_ids: tuple[ResidueId, ...] | None = None

    def __post_init__(self) -> None:
        backend_name = self.backend_name.strip().lower()
        if not backend_name:
            raise ValueError("packing backend_name must not be blank")

        target_sequence = self.target_sequence
        if target_sequence is not None:
            target_sequence = "".join(target_sequence.split()).upper()
            if not target_sequence:
                raise ValueError("packing target_sequence must not be blank")
            if not target_sequence.isalpha():
                raise ValueError(
                    "packing target_sequence must contain only alphabetic codes"
                )

        mutable_residue_ids = _normalize_residue_id_tuple(self.mutable_residue_ids)
        frozen_residue_ids = _normalize_residue_id_tuple(self.frozen_residue_ids)
        mutable_residue_id_set = (
            set(mutable_residue_ids) if mutable_residue_ids is not None else set()
        )
        frozen_residue_id_set = (
            set(frozen_residue_ids) if frozen_residue_ids is not None else set()
        )

        if mutable_residue_id_set & frozen_residue_id_set:
            raise ValueError(
                "packing mutable_residue_ids and frozen_residue_ids must not overlap"
            )

        if self.scope is PackingScope.LOCAL and mutable_residue_ids is None:
            raise ValueError("local side-chain packing requires mutable_residue_ids")

        object.__setattr__(self, "backend_name", backend_name)
        object.__setattr__(self, "target_sequence", target_sequence)
        object.__setattr__(self, "mutable_residue_ids", mutable_residue_ids)
        object.__setattr__(self, "frozen_residue_ids", frozen_residue_ids)

    def has_sequence_override(self) -> bool:
        """Return whether the spec includes a sequence override."""

        return self.target_sequence is not None

    def is_local(self) -> bool:
        """Return whether the request targets a local packing region."""

        return self.scope is PackingScope.LOCAL

    def referenced_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return mutable and frozen residue identifiers in first-seen order."""

        ordered_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()

        for residue_ids in (self.mutable_residue_ids, self.frozen_residue_ids):
            if residue_ids is None:
                continue

            for residue_id in residue_ids:
                if residue_id not in seen_residue_ids:
                    ordered_residue_ids.append(residue_id)
                    seen_residue_ids.add(residue_id)

        return tuple(ordered_residue_ids)

    def references_residue(self, residue_id: ResidueId) -> bool:
        """Return whether a residue is named in the packing specification."""

        return residue_id in self.referenced_residue_ids()

    def with_mutable_residue_ids(
        self,
        mutable_residue_ids: tuple[ResidueId, ...] | None,
    ) -> Self:
        """Return a copy with updated mutable-residue targeting."""

        return type(self)(
            backend_name=self.backend_name,
            mode=self.mode,
            scope=self.scope,
            target_sequence=self.target_sequence,
            mutable_residue_ids=mutable_residue_ids,
            frozen_residue_ids=self.frozen_residue_ids,
        )

    def with_scope(self, scope: PackingScope) -> Self:
        """Return a copy with updated packing scope."""

        return type(self)(
            backend_name=self.backend_name,
            mode=self.mode,
            scope=scope,
            target_sequence=self.target_sequence,
            mutable_residue_ids=self.mutable_residue_ids,
            frozen_residue_ids=self.frozen_residue_ids,
        )


def _normalize_residue_id_tuple(
    residue_ids: tuple[ResidueId, ...] | None,
) -> tuple[ResidueId, ...] | None:
    """Normalize one optional residue-identifier tuple."""

    if residue_ids is None:
        return None

    normalized_residue_ids: list[ResidueId] = []
    seen_residue_ids: set[ResidueId] = set()
    for residue_id in residue_ids:
        if residue_id not in seen_residue_ids:
            normalized_residue_ids.append(residue_id)
            seen_residue_ids.add(residue_id)

    return tuple(normalized_residue_ids) or None


__all__ = ["PackingMode", "PackingScope", "PackingSpec"]
