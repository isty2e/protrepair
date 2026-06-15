"""Canonical cross-carrier scoped structure endpoints."""

from dataclasses import dataclass
from enum import Enum

from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    ChainSetScope,
    CompositeScope,
    ResidueBoundaryScope,
    ResidueSetScope,
    Scope,
    WholeStructureScope,
    scope_refines,
)


class StructureCarrierKind(str, Enum):
    """Closed structure-carrier kinds used in cross-carrier endpoints."""

    SOURCE = "source"
    SUPPORTING = "supporting"


@dataclass(frozen=True, slots=True)
class StructureRealizationSelector:
    """Selector for one specific realization of one structure carrier."""

    model_index: int | None = None
    altloc_label: str | None = None

    def __post_init__(self) -> None:
        model_index = self.model_index
        if model_index is not None:
            if isinstance(model_index, bool) or not isinstance(model_index, int):
                raise TypeError(
                    "structure realization selectors require model_index to be "
                    "an integer when provided"
                )
            if model_index < 0:
                raise ValueError(
                    "structure realization selectors require model_index to be "
                    "non-negative"
                )

        altloc_label = self.altloc_label
        if altloc_label is not None:
            altloc_label = altloc_label.strip() or None

        if model_index is None and altloc_label is None:
            raise ValueError(
                "structure realization selectors require at least one of "
                "model_index or altloc_label"
            )

        object.__setattr__(self, "altloc_label", altloc_label)


@dataclass(frozen=True, slots=True)
class StructureCarrierHandle:
    """Opaque handle identifying one structure carrier in a relation or lineage."""

    kind: StructureCarrierKind
    token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StructureCarrierKind):
            raise TypeError(
                "structure carrier handles require a StructureCarrierKind value"
            )

        token = self.token
        if token is not None:
            token = token.strip() or None

        object.__setattr__(self, "token", token)

    @classmethod
    def source(cls, token: str | None = None) -> "StructureCarrierHandle":
        """Return the canonical handle for the active source structure."""

        return cls(kind=StructureCarrierKind.SOURCE, token=token)

    @classmethod
    def supporting(
        cls,
        token: str | None = None,
    ) -> "StructureCarrierHandle":
        """Return the canonical handle for one supporting structure."""

        return cls(kind=StructureCarrierKind.SUPPORTING, token=token)


@dataclass(frozen=True, slots=True)
class StructureEndpoint:
    """One scoped endpoint inside one specific structure carrier."""

    carrier_handle: StructureCarrierHandle
    scope: Scope
    realization_selector: StructureRealizationSelector | None = None

    @classmethod
    def source(
        cls,
        scope: Scope,
        *,
        token: str | None = None,
        realization_selector: StructureRealizationSelector | None = None,
    ) -> "StructureEndpoint":
        """Return one source-carrier endpoint over one semantic scope."""

        return cls(
            carrier_handle=StructureCarrierHandle.source(token),
            scope=scope,
            realization_selector=realization_selector,
        )

    @classmethod
    def supporting(
        cls,
        scope: Scope,
        *,
        token: str | None = None,
        realization_selector: StructureRealizationSelector | None = None,
    ) -> "StructureEndpoint":
        """Return one supporting-carrier endpoint over one semantic scope."""

        return cls(
            carrier_handle=StructureCarrierHandle.supporting(token),
            scope=scope,
            realization_selector=realization_selector,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.carrier_handle, StructureCarrierHandle):
            raise TypeError(
                "structure endpoints require a StructureCarrierHandle value"
            )
        if not isinstance(self.scope, Scope):
            raise TypeError("structure endpoints require a Scope value")
        if self.realization_selector is not None and not isinstance(
            self.realization_selector,
            StructureRealizationSelector,
        ):
            raise TypeError(
                "structure endpoints require realization_selector to be a "
                "StructureRealizationSelector or None"
            )

    def is_source(self) -> bool:
        """Return whether this endpoint points into the active source carrier."""

        return self.carrier_handle.kind is StructureCarrierKind.SOURCE

    def is_supporting(self) -> bool:
        """Return whether this endpoint points into one supporting carrier."""

        return self.carrier_handle.kind is StructureCarrierKind.SUPPORTING

    def cardinality(self) -> int:
        """Return the number of semantic members addressed by this endpoint."""

        if isinstance(self.scope, WholeStructureScope):
            return 1
        if isinstance(self.scope, ChainSetScope):
            return len(self.scope.chain_ids)
        if isinstance(self.scope, ResidueSetScope):
            return len(self.scope.residue_ids)
        if isinstance(self.scope, ResidueBoundaryScope):
            return 1
        if isinstance(self.scope, AtomSetScope):
            return len(self.scope.atom_refs)
        if isinstance(self.scope, AbsentResidueSpanScope):
            return len(
                tuple(
                    dict.fromkeys(
                        (
                            *(
                                ()
                                if self.scope.preceding_residue_id is None
                                else (self.scope.preceding_residue_id,)
                            ),
                            *self.scope.absent_residue_ids,
                            *(
                                ()
                                if self.scope.following_residue_id is None
                                else (self.scope.following_residue_id,)
                            ),
                        )
                    )
                )
            )
        if isinstance(self.scope, AnchorAtomPairScope):
            return 2
        if isinstance(self.scope, CompositeScope):
            return len(self.scope.scopes)

        raise TypeError("unsupported scope for structure-endpoint cardinality")

    def covers(self, other: "StructureEndpoint") -> bool:
        """Return whether this endpoint semantically covers another endpoint."""

        if not isinstance(other, StructureEndpoint):
            raise TypeError(
                "structure-endpoint coverage requires one "
                "StructureEndpoint"
            )

        if self.carrier_handle != other.carrier_handle:
            return False

        if (
            self.realization_selector is not None
            and other.realization_selector != self.realization_selector
        ):
            return False

        return scope_refines(other.scope, self.scope)
