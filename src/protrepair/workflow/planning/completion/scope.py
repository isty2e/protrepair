"""Workflow completion execution-scope ontology."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    CompositeScope,
    ResidueSetScope,
)
from protrepair.structure.labels import AtomRef, ResidueId


class WorkflowExecutionScopeKind(str, Enum):
    """Closed execution-scope ontology for workflow completion stages."""

    RESIDUE_SET = "residue_set"
    ABSENT_RESIDUE_SPAN = "absent_residue_span"
    ANCHOR_ATOM_PAIR = "anchor_atom_pair"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class WorkflowResidueSetExecutionScope:
    """Executable workflow scope over one existing residue subset."""

    residue_ids: tuple[ResidueId, ...]

    def __post_init__(self) -> None:
        residue_ids = tuple(dict.fromkeys(self.residue_ids))
        if not residue_ids:
            raise ValueError(
                "workflow residue-set execution scopes require at least one residue"
            )

        object.__setattr__(self, "residue_ids", residue_ids)

    @property
    def kind(self) -> WorkflowExecutionScopeKind:
        """Return the canonical execution-scope kind."""

        return WorkflowExecutionScopeKind.RESIDUE_SET

    def display_tokens(self) -> tuple[str, ...]:
        """Return human-readable residue tokens for this scope."""

        return tuple(residue_id.display_token() for residue_id in self.residue_ids)

    def as_scope(self) -> ResidueSetScope:
        """Return this workflow execution scope as one semantic scope."""

        return ResidueSetScope(residue_ids=self.residue_ids)


@dataclass(frozen=True, slots=True)
class WorkflowAbsentResidueSpanExecutionScope:
    """Executable workflow scope over one absent span anchored by residues."""

    preceding_residue_id: ResidueId | None = None
    following_residue_id: ResidueId | None = None
    absent_residue_ids: tuple[ResidueId, ...] = ()

    def __post_init__(self) -> None:
        if self.preceding_residue_id is None and self.following_residue_id is None:
            raise ValueError(
                "workflow absent-span execution scopes require at least one "
                "anchor residue"
            )
        if (
            self.preceding_residue_id is not None
            and self.following_residue_id is not None
            and self.preceding_residue_id == self.following_residue_id
        ):
            raise ValueError(
                "workflow absent-span execution scope anchors must not be identical"
            )

        ordered_absent_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        for residue_id in self.absent_residue_ids:
            if residue_id == self.preceding_residue_id:
                raise ValueError(
                    "workflow absent-span execution scopes must not list the "
                    "preceding anchor as absent"
                )
            if residue_id == self.following_residue_id:
                raise ValueError(
                    "workflow absent-span execution scopes must not list the "
                    "following anchor as absent"
                )
            if residue_id not in seen_residue_ids:
                ordered_absent_residue_ids.append(residue_id)
                seen_residue_ids.add(residue_id)

        object.__setattr__(
            self,
            "absent_residue_ids",
            tuple(ordered_absent_residue_ids),
        )

    @property
    def kind(self) -> WorkflowExecutionScopeKind:
        """Return the canonical execution-scope kind."""

        return WorkflowExecutionScopeKind.ABSENT_RESIDUE_SPAN

    def display_tokens(self) -> tuple[str, ...]:
        """Return human-readable tokens for this absent-span target."""

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

    def as_scope(self) -> AbsentResidueSpanScope:
        """Return this workflow execution scope as one semantic scope."""

        return AbsentResidueSpanScope(
            preceding_residue_id=self.preceding_residue_id,
            following_residue_id=self.following_residue_id,
            absent_residue_ids=self.absent_residue_ids,
        )


@dataclass(frozen=True, slots=True)
class WorkflowAnchorAtomPairExecutionScope:
    """Executable workflow scope defined by one relational anchor atom pair."""

    left_anchor_atom_ref: AtomRef
    right_anchor_atom_ref: AtomRef

    def __post_init__(self) -> None:
        if self.left_anchor_atom_ref == self.right_anchor_atom_ref:
            raise ValueError(
                "workflow anchor-pair execution scopes require distinct anchor atoms"
            )

    @property
    def kind(self) -> WorkflowExecutionScopeKind:
        """Return the canonical execution-scope kind."""

        return WorkflowExecutionScopeKind.ANCHOR_ATOM_PAIR

    def display_tokens(self) -> tuple[str, ...]:
        """Return human-readable atom tokens for this anchor-pair target."""

        return (
            self.left_anchor_atom_ref.display_token(),
            self.right_anchor_atom_ref.display_token(),
        )

    def as_scope(self) -> AnchorAtomPairScope:
        """Return this workflow execution scope as one semantic scope."""

        return AnchorAtomPairScope(
            left_anchor_atom_ref=self.left_anchor_atom_ref,
            right_anchor_atom_ref=self.right_anchor_atom_ref,
        )


WorkflowExecutionScopePrimitive: TypeAlias = (
    WorkflowResidueSetExecutionScope
    | WorkflowAbsentResidueSpanExecutionScope
    | WorkflowAnchorAtomPairExecutionScope
)


@dataclass(frozen=True, slots=True)
class WorkflowCompositeExecutionScope:
    """Composite workflow scope that preserves multiple primitive scope roles."""

    scopes: tuple[WorkflowExecutionScopePrimitive, ...]

    def __post_init__(self) -> None:
        ordered_scopes: list[WorkflowExecutionScopePrimitive] = []
        seen_scopes: set[WorkflowExecutionScopePrimitive] = set()
        for scope in self.scopes:
            if not isinstance(
                scope,
                (
                    WorkflowResidueSetExecutionScope,
                    WorkflowAbsentResidueSpanExecutionScope,
                    WorkflowAnchorAtomPairExecutionScope,
                ),
            ):
                raise TypeError(
                    "workflow composite execution scopes require primitive "
                    "WorkflowExecutionScope values"
                )
            if scope in seen_scopes:
                continue

            ordered_scopes.append(scope)
            seen_scopes.add(scope)

        if len(ordered_scopes) < 2:
            raise ValueError(
                "workflow composite execution scopes require at least two "
                "distinct primitive scopes"
            )

        object.__setattr__(self, "scopes", tuple(ordered_scopes))

    @property
    def kind(self) -> WorkflowExecutionScopeKind:
        """Return the canonical execution-scope kind."""

        return WorkflowExecutionScopeKind.COMPOSITE

    def display_tokens(self) -> tuple[str, ...]:
        """Return human-readable tokens with preserved child scope roles."""

        return tuple(
            f"{scope.kind.value}:{token}"
            for scope in self.scopes
            for token in scope.display_tokens()
        )

    def as_scope(self) -> CompositeScope:
        """Return this workflow execution scope as one semantic scope."""

        return CompositeScope(scopes=tuple(scope.as_scope() for scope in self.scopes))


WorkflowExecutionScope: TypeAlias = (
    WorkflowExecutionScopePrimitive | WorkflowCompositeExecutionScope
)
