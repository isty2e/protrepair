"""Shared semantic scope domain for state and transformer contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from protrepair.structure.labels import AtomRef, ResidueId


class ScopeKind(str, Enum):
    """Closed semantic scope kinds for protein-state and transformer contracts."""

    WHOLE_STRUCTURE = "whole_structure"
    CHAIN_SET = "chain_set"
    RESIDUE_SET = "residue_set"
    RESIDUE_BOUNDARY = "residue_boundary"
    ATOM_SET = "atom_set"
    ABSENT_RESIDUE_SPAN = "absent_residue_span"
    ANCHOR_ATOM_PAIR = "anchor_atom_pair"
    COMPOSITE = "composite"


class Scope(ABC):
    """Semantic scope shared by state values and transformers."""

    @property
    @abstractmethod
    def kind(self) -> ScopeKind:
        """Return the closed semantic kind for this scope."""

    @abstractmethod
    def display_tokens(self) -> tuple[str, ...]:
        """Return stable human-readable tokens for this scope."""


@dataclass(frozen=True, slots=True)
class WholeStructureScope(Scope):
    """Semantic scope over the whole active structure."""

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.WHOLE_STRUCTURE

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens for this scope."""

        return ("whole-structure",)


@dataclass(frozen=True, slots=True)
class ChainSetScope(Scope):
    """Semantic scope over one explicit set of chains."""

    chain_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ordered_chain_ids: list[str] = []
        seen_chain_ids: set[str] = set()
        for chain_id in self.chain_ids:
            normalized_chain_id = chain_id.strip()
            if not normalized_chain_id:
                raise ValueError("chain-set scopes must not contain blank chain ids")
            if normalized_chain_id in seen_chain_ids:
                continue

            ordered_chain_ids.append(normalized_chain_id)
            seen_chain_ids.add(normalized_chain_id)

        if not ordered_chain_ids:
            raise ValueError("chain-set scopes require at least one chain id")

        object.__setattr__(self, "chain_ids", tuple(ordered_chain_ids))

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.CHAIN_SET

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens for this scope."""

        return self.chain_ids


@dataclass(frozen=True, slots=True)
class ResidueSetScope(Scope):
    """Semantic scope over one explicit residue subset."""

    residue_ids: tuple[ResidueId, ...]

    def __post_init__(self) -> None:
        residue_ids = tuple(dict.fromkeys(self.residue_ids))
        if not residue_ids:
            raise ValueError("residue-set scopes require at least one residue")

        object.__setattr__(self, "residue_ids", residue_ids)

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.RESIDUE_SET

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens for this scope."""

        return tuple(residue_id.display_token() for residue_id in self.residue_ids)


class ResidueBoundarySide(str, Enum):
    """One terminal-boundary side for one residue-boundary scope."""

    N_TERMINUS = "n_terminus"
    C_TERMINUS = "c_terminus"


@dataclass(frozen=True, slots=True)
class ResidueBoundaryScope(Scope):
    """Semantic scope over one residue-boundary pair."""

    residue_id: ResidueId
    side: ResidueBoundarySide

    def __post_init__(self) -> None:
        if not isinstance(self.side, ResidueBoundarySide):
            raise TypeError(
                "residue-boundary scopes require a ResidueBoundarySide value"
            )

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.RESIDUE_BOUNDARY

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens for this scope."""

        return (
            self.residue_id.display_token(),
            self.side.value,
        )


@dataclass(frozen=True, slots=True)
class AtomSetScope(Scope):
    """Semantic scope over one explicit atom subset."""

    atom_refs: tuple[AtomRef, ...]

    def __post_init__(self) -> None:
        atom_refs = AtomRef.deduplicated(self.atom_refs)
        if not atom_refs:
            raise ValueError("atom-set scopes require at least one atom ref")

        object.__setattr__(self, "atom_refs", atom_refs)

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.ATOM_SET

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens for this scope."""

        return tuple(atom_ref.display_token() for atom_ref in self.atom_refs)


@dataclass(frozen=True, slots=True)
class AbsentResidueSpanScope(Scope):
    """Semantic scope over an absent span anchored by present residues."""

    preceding_residue_id: ResidueId | None = None
    following_residue_id: ResidueId | None = None
    absent_residue_ids: tuple[ResidueId, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.preceding_residue_id is None
            and self.following_residue_id is None
        ):
            raise ValueError(
                "absent-residue-span scopes require at least one anchor residue"
            )
        if (
            self.preceding_residue_id is not None
            and self.following_residue_id is not None
            and self.preceding_residue_id == self.following_residue_id
        ):
            raise ValueError(
                "absent-residue-span scopes require distinct anchor residues"
            )

        ordered_absent_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        for residue_id in self.absent_residue_ids:
            if residue_id == self.preceding_residue_id:
                raise ValueError(
                    "absent-residue-span scopes must not list the preceding "
                    "anchor as absent"
                )
            if residue_id == self.following_residue_id:
                raise ValueError(
                    "absent-residue-span scopes must not list the following "
                    "anchor as absent"
                )
            if residue_id in seen_residue_ids:
                continue

            ordered_absent_residue_ids.append(residue_id)
            seen_residue_ids.add(residue_id)

        object.__setattr__(
            self,
            "absent_residue_ids",
            tuple(ordered_absent_residue_ids),
        )

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.ABSENT_RESIDUE_SPAN

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens for this scope."""

        tokens: list[str] = []
        if self.preceding_residue_id is not None:
            tokens.append(f"preceding:{self.preceding_residue_id.display_token()}")
        if self.following_residue_id is not None:
            tokens.append(f"following:{self.following_residue_id.display_token()}")
        tokens.extend(
            f"absent:{residue_id.display_token()}"
            for residue_id in self.absent_residue_ids
        )
        return tuple(tokens)

    def anchor_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return present anchor residues in first-seen order."""

        anchor_residue_ids: list[ResidueId] = []
        if self.preceding_residue_id is not None:
            anchor_residue_ids.append(self.preceding_residue_id)
        if self.following_residue_id is not None:
            anchor_residue_ids.append(self.following_residue_id)

        return tuple(anchor_residue_ids)


@dataclass(frozen=True, slots=True)
class AnchorAtomPairScope(Scope):
    """Semantic scope defined by one relational anchor atom pair."""

    left_anchor_atom_ref: AtomRef
    right_anchor_atom_ref: AtomRef

    def __post_init__(self) -> None:
        if self.left_anchor_atom_ref == self.right_anchor_atom_ref:
            raise ValueError(
                "anchor-atom-pair scopes require two distinct anchor atoms"
            )

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.ANCHOR_ATOM_PAIR

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens for this scope."""

        return (
            self.left_anchor_atom_ref.display_token(),
            self.right_anchor_atom_ref.display_token(),
        )

    def anchor_atom_refs(self) -> tuple[AtomRef, AtomRef]:
        """Return the ordered anchor atoms that define this scope."""

        return (self.left_anchor_atom_ref, self.right_anchor_atom_ref)

    def anchor_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return anchor residues in first-seen order."""

        left_residue_id = self.left_anchor_atom_ref.residue_id
        right_residue_id = self.right_anchor_atom_ref.residue_id
        if left_residue_id == right_residue_id:
            return (left_residue_id,)

        return (left_residue_id, right_residue_id)


@dataclass(frozen=True, slots=True)
class CompositeScope(Scope):
    """Flat composite of multiple non-composite semantic scopes."""

    scopes: tuple[Scope, ...]

    def __post_init__(self) -> None:
        ordered_scopes: list[Scope] = []
        seen_scopes: set[Scope] = set()
        for scope in self.scopes:
            if not isinstance(scope, Scope):
                raise TypeError("composite scopes require Scope values")
            if isinstance(scope, CompositeScope):
                raise ValueError("composite scopes must stay flat")
            if scope in seen_scopes:
                continue

            ordered_scopes.append(scope)
            seen_scopes.add(scope)

        if len(ordered_scopes) < 2:
            raise ValueError(
                "composite scopes require at least two distinct scopes"
            )

        object.__setattr__(self, "scopes", tuple(ordered_scopes))

    @property
    def kind(self) -> ScopeKind:
        """Return the semantic kind for this scope."""

        return ScopeKind.COMPOSITE

    def display_tokens(self) -> tuple[str, ...]:
        """Return stable tokens with child scope roles preserved."""

        return tuple(
            f"{scope.kind.value}:{token}"
            for scope in self.scopes
            for token in scope.display_tokens()
        )
