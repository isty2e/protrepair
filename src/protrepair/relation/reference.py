"""Boundary structure reference adapters over residue and atom labels."""

from dataclasses import dataclass
from enum import Enum

from protrepair.relation.endpoint import StructureRealizationSelector
from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    CompositeScope,
    ResidueSetScope,
    Scope,
)
from protrepair.structure.labels import AtomRef, ResidueId


class StructureReferenceKind(str, Enum):
    """Closed reference kinds for source or supporting structure subregions."""

    RESIDUE_SET = "residue_set"
    ATOM_SET = "atom_set"
    ABSENT_RESIDUE_SPAN = "absent_residue_span"
    ANCHOR_ATOM_PAIR = "anchor_atom_pair"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class StructureRealizationReference:
    """Boundary realization selector over one specific structure realization."""

    model_index: int | None = None
    altloc_label: str | None = None

    def __post_init__(self) -> None:
        model_index = self.model_index
        if model_index is not None:
            if isinstance(model_index, bool) or not isinstance(model_index, int):
                raise TypeError(
                    "structure realization references require model_index to be "
                    "an integer when provided"
                )
            if model_index < 0:
                raise ValueError(
                    "structure realization references require model_index to be "
                    "non-negative"
                )

        altloc_label = self.altloc_label
        if altloc_label is not None:
            altloc_label = altloc_label.strip() or None

        if model_index is None and altloc_label is None:
            raise ValueError(
                "structure realization references require at least one of "
                "model_index or altloc_label"
            )

        object.__setattr__(self, "altloc_label", altloc_label)


@dataclass(frozen=True, slots=True)
class StructureReference:
    """Boundary structure subregion plus an optional realization selector."""

    kind: StructureReferenceKind
    residue_ids: tuple[ResidueId, ...] = ()
    atom_refs: tuple[AtomRef, ...] = ()
    absent_residue_ids: tuple[ResidueId, ...] = ()
    preceding_residue_id: ResidueId | None = None
    following_residue_id: ResidueId | None = None
    left_anchor_atom_ref: AtomRef | None = None
    right_anchor_atom_ref: AtomRef | None = None
    member_references: tuple["StructureReference", ...] = ()
    realization: StructureRealizationReference | None = None

    def __post_init__(self) -> None:
        residue_ids = tuple(dict.fromkeys(self.residue_ids))
        atom_refs = AtomRef.deduplicated(self.atom_refs)
        absent_residue_ids = tuple(dict.fromkeys(self.absent_residue_ids))
        member_references = tuple(dict.fromkeys(self.member_references))
        realization = self.realization
        if self.kind is StructureReferenceKind.RESIDUE_SET:
            if not residue_ids:
                raise ValueError(
                    "residue-set structure references require at least one residue id"
                )
            if (
                atom_refs
                or absent_residue_ids
                or self.preceding_residue_id is not None
                or self.following_residue_id is not None
                or self.left_anchor_atom_ref is not None
                or self.right_anchor_atom_ref is not None
                or member_references
            ):
                raise ValueError(
                    "residue-set structure references must not carry non-residue-set "
                    "payload"
                )
        elif self.kind is StructureReferenceKind.ATOM_SET:
            if not atom_refs:
                raise ValueError(
                    "atom-set structure references require at least one atom ref"
                )
            if (
                residue_ids
                or absent_residue_ids
                or self.preceding_residue_id is not None
                or self.following_residue_id is not None
                or self.left_anchor_atom_ref is not None
                or self.right_anchor_atom_ref is not None
                or member_references
            ):
                raise ValueError(
                    "atom-set structure references must not carry non-atom-set "
                    "payload"
                )
        elif self.kind is StructureReferenceKind.ABSENT_RESIDUE_SPAN:
            if (
                self.preceding_residue_id is None
                and self.following_residue_id is None
            ):
                raise ValueError(
                    "absent-residue-span structure references require at least one "
                    "anchor residue"
                )
            if (
                self.preceding_residue_id is not None
                and self.following_residue_id is not None
                and self.preceding_residue_id == self.following_residue_id
            ):
                raise ValueError(
                    "absent-residue-span structure references require distinct "
                    "anchor residues"
                )
            if (
                residue_ids
                or atom_refs
                or self.left_anchor_atom_ref is not None
                or self.right_anchor_atom_ref is not None
                or member_references
            ):
                raise ValueError(
                    "absent-residue-span structure references must not carry "
                    "residue-set, atom-set, anchor-pair, or composite payload"
                )
            for residue_id in absent_residue_ids:
                if residue_id == self.preceding_residue_id:
                    raise ValueError(
                        "absent-residue-span structure references must not include "
                        "the preceding anchor in absent_residue_ids"
                    )
                if residue_id == self.following_residue_id:
                    raise ValueError(
                        "absent-residue-span structure references must not include "
                        "the following anchor in absent_residue_ids"
                    )
        elif self.kind is StructureReferenceKind.ANCHOR_ATOM_PAIR:
            if (
                self.left_anchor_atom_ref is None
                or self.right_anchor_atom_ref is None
            ):
                raise ValueError(
                    "anchor-atom-pair structure references require both anchor "
                    "atom refs"
                )
            if self.left_anchor_atom_ref == self.right_anchor_atom_ref:
                raise ValueError(
                    "anchor-atom-pair structure references require distinct "
                    "anchor atom refs"
                )
            if (
                residue_ids
                or atom_refs
                or absent_residue_ids
                or self.preceding_residue_id is not None
                or self.following_residue_id is not None
                or member_references
            ):
                raise ValueError(
                    "anchor-atom-pair structure references must not carry "
                    "residue-set, atom-set, absent-span, or composite payload"
                )
        elif self.kind is StructureReferenceKind.COMPOSITE:
            if len(member_references) < 2:
                raise ValueError(
                    "composite structure references require at least two "
                    "member references"
                )
            if realization is not None:
                raise ValueError(
                    "composite structure references must not carry a top-level "
                    "realization; member references own their realizations"
                )
            if (
                residue_ids
                or atom_refs
                or absent_residue_ids
                or self.preceding_residue_id is not None
                or self.following_residue_id is not None
                or self.left_anchor_atom_ref is not None
                or self.right_anchor_atom_ref is not None
            ):
                raise ValueError(
                    "composite structure references must not carry primitive "
                    "payload directly"
                )
            for member_reference in member_references:
                if not isinstance(member_reference, StructureReference):
                    raise TypeError(
                        "composite structure references require StructureReference "
                        "member references"
                    )
                if member_reference.kind is StructureReferenceKind.COMPOSITE:
                    raise ValueError(
                        "composite structure references must stay flat"
                    )
        else:
            raise TypeError(
                "structure references require a StructureReferenceKind value"
            )
        if realization is not None and not isinstance(
            realization,
            StructureRealizationReference,
        ):
            raise TypeError(
                "structure references require realization to be a "
                "StructureRealizationReference or None"
            )

        object.__setattr__(self, "residue_ids", residue_ids)
        object.__setattr__(self, "atom_refs", atom_refs)
        object.__setattr__(self, "absent_residue_ids", absent_residue_ids)
        object.__setattr__(self, "member_references", member_references)
        object.__setattr__(self, "realization", realization)

    @classmethod
    def from_residues(
        cls,
        residue_ids: tuple[ResidueId, ...],
        *,
        realization: StructureRealizationReference | None = None,
    ) -> "StructureReference":
        """Return one residue-set structure reference."""

        return cls(
            kind=StructureReferenceKind.RESIDUE_SET,
            residue_ids=residue_ids,
            realization=realization,
        )

    @classmethod
    def from_atoms(
        cls,
        atom_refs: tuple[AtomRef, ...],
        *,
        realization: StructureRealizationReference | None = None,
    ) -> "StructureReference":
        """Return one atom-set structure reference."""

        return cls(
            kind=StructureReferenceKind.ATOM_SET,
            atom_refs=atom_refs,
            realization=realization,
        )

    @classmethod
    def from_absent_residue_span(
        cls,
        *,
        preceding_residue_id: ResidueId | None = None,
        following_residue_id: ResidueId | None = None,
        absent_residue_ids: tuple[ResidueId, ...] = (),
        realization: StructureRealizationReference | None = None,
    ) -> "StructureReference":
        """Return one absent-residue-span structure reference."""

        return cls(
            kind=StructureReferenceKind.ABSENT_RESIDUE_SPAN,
            absent_residue_ids=absent_residue_ids,
            preceding_residue_id=preceding_residue_id,
            following_residue_id=following_residue_id,
            realization=realization,
        )

    @classmethod
    def from_anchor_atom_pair(
        cls,
        left_anchor_atom_ref: AtomRef,
        right_anchor_atom_ref: AtomRef,
        *,
        realization: StructureRealizationReference | None = None,
    ) -> "StructureReference":
        """Return one anchor-atom-pair structure reference."""

        return cls(
            kind=StructureReferenceKind.ANCHOR_ATOM_PAIR,
            left_anchor_atom_ref=left_anchor_atom_ref,
            right_anchor_atom_ref=right_anchor_atom_ref,
            realization=realization,
        )

    @classmethod
    def composite(
        cls,
        member_references: tuple["StructureReference", ...],
    ) -> "StructureReference":
        """Return one flat composite structure reference."""

        return cls(
            kind=StructureReferenceKind.COMPOSITE,
            member_references=member_references,
        )

    def referenced_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return referenced residue ids in stable first-seen order."""

        if self.kind is StructureReferenceKind.RESIDUE_SET:
            return self.residue_ids
        if self.kind is StructureReferenceKind.ABSENT_RESIDUE_SPAN:
            span_residue_ids: list[ResidueId] = []
            if self.preceding_residue_id is not None:
                span_residue_ids.append(self.preceding_residue_id)
            span_residue_ids.extend(self.absent_residue_ids)
            if self.following_residue_id is not None:
                span_residue_ids.append(self.following_residue_id)

            return tuple(dict.fromkeys(span_residue_ids))
        if self.kind is StructureReferenceKind.ANCHOR_ATOM_PAIR:
            assert self.left_anchor_atom_ref is not None
            assert self.right_anchor_atom_ref is not None
            left_residue_id = self.left_anchor_atom_ref.residue_id
            right_residue_id = self.right_anchor_atom_ref.residue_id
            if left_residue_id == right_residue_id:
                return (left_residue_id,)

            return (left_residue_id, right_residue_id)
        if self.kind is StructureReferenceKind.COMPOSITE:
            composite_residue_ids: list[ResidueId] = []
            composite_seen_residue_ids: set[ResidueId] = set()
            for member_reference in self.member_references:
                for residue_id in member_reference.referenced_residue_ids():
                    if residue_id in composite_seen_residue_ids:
                        continue

                    composite_residue_ids.append(residue_id)
                    composite_seen_residue_ids.add(residue_id)

            return tuple(composite_residue_ids)

        ordered_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        for atom_ref in self.atom_refs:
            if atom_ref.residue_id not in seen_residue_ids:
                ordered_residue_ids.append(atom_ref.residue_id)
                seen_residue_ids.add(atom_ref.residue_id)

        return tuple(ordered_residue_ids)

    def cardinality(self) -> int:
        """Return the number of canonical members carried by this reference."""

        if self.kind is StructureReferenceKind.RESIDUE_SET:
            return len(self.residue_ids)
        if self.kind is StructureReferenceKind.ATOM_SET:
            return len(self.atom_refs)
        if self.kind is StructureReferenceKind.ABSENT_RESIDUE_SPAN:
            return len(self.referenced_residue_ids())
        if self.kind is StructureReferenceKind.ANCHOR_ATOM_PAIR:
            return 2
        if self.kind is StructureReferenceKind.COMPOSITE:
            return len(self.member_references)

        raise AssertionError("unreachable structure reference kind")

    def covers(self, other: "StructureReference") -> bool:
        """Return whether this reference semantically covers another reference."""

        if self.realization is not None and other.realization != self.realization:
            return False

        if self.kind is StructureReferenceKind.RESIDUE_SET:
            return set(other.referenced_residue_ids()) <= set(self.residue_ids)
        if self.kind is StructureReferenceKind.ABSENT_RESIDUE_SPAN:
            return (
                other.kind is StructureReferenceKind.ABSENT_RESIDUE_SPAN
                and other.preceding_residue_id == self.preceding_residue_id
                and other.following_residue_id == self.following_residue_id
                and set(other.absent_residue_ids) <= set(self.absent_residue_ids)
            )
        if self.kind is StructureReferenceKind.ANCHOR_ATOM_PAIR:
            return (
                other.kind is StructureReferenceKind.ANCHOR_ATOM_PAIR
                and other.left_anchor_atom_ref == self.left_anchor_atom_ref
                and other.right_anchor_atom_ref == self.right_anchor_atom_ref
            )
        if self.kind is StructureReferenceKind.COMPOSITE:
            if other.kind is StructureReferenceKind.COMPOSITE:
                return all(
                    any(
                        member_reference.covers(other_member)
                        for member_reference in self.member_references
                    )
                    for other_member in other.member_references
                )

            return any(
                member_reference.covers(other)
                for member_reference in self.member_references
            )

        if other.kind is not StructureReferenceKind.ATOM_SET:
            return False

        return set(other.atom_refs) <= set(self.atom_refs)


def scope_from_reference(reference: StructureReference) -> Scope:
    """Project one semantic scope from one boundary structure reference."""

    if not isinstance(reference, StructureReference):
        raise TypeError(
            "scope projection requires one StructureReference value"
        )

    if reference.kind is StructureReferenceKind.RESIDUE_SET:
        return ResidueSetScope(residue_ids=reference.residue_ids)
    if reference.kind is StructureReferenceKind.ATOM_SET:
        return AtomSetScope(atom_refs=reference.atom_refs)
    if reference.kind is StructureReferenceKind.ABSENT_RESIDUE_SPAN:
        return AbsentResidueSpanScope(
            preceding_residue_id=reference.preceding_residue_id,
            following_residue_id=reference.following_residue_id,
            absent_residue_ids=reference.absent_residue_ids,
        )
    if reference.kind is StructureReferenceKind.ANCHOR_ATOM_PAIR:
        assert reference.left_anchor_atom_ref is not None
        assert reference.right_anchor_atom_ref is not None
        return AnchorAtomPairScope(
            left_anchor_atom_ref=reference.left_anchor_atom_ref,
            right_anchor_atom_ref=reference.right_anchor_atom_ref,
        )
    if reference.kind is StructureReferenceKind.COMPOSITE:
        return CompositeScope(
            scopes=tuple(
                scope_from_reference(member_reference)
                for member_reference in reference.member_references
            )
        )

    raise AssertionError("unreachable structure reference kind")


def realization_selector_from_reference(
    reference: StructureReference,
) -> StructureRealizationSelector | None:
    """Project one realization selector from one boundary structure reference."""

    if not isinstance(reference, StructureReference):
        raise TypeError(
            "realization-selector projection requires one StructureReference value"
        )

    realization = reference.realization
    if realization is None:
        return None

    return StructureRealizationSelector(
        model_index=realization.model_index,
        altloc_label=realization.altloc_label,
    )


def reference_from_scope(
    scope: Scope,
    *,
    realization_selector: StructureRealizationSelector | None,
) -> StructureReference:
    """Project one boundary structure reference from one semantic scope."""

    realization = None
    if realization_selector is not None:
        realization = StructureRealizationReference(
            model_index=realization_selector.model_index,
            altloc_label=realization_selector.altloc_label,
        )

    if isinstance(scope, ResidueSetScope):
        return StructureReference.from_residues(
            scope.residue_ids,
            realization=realization,
        )
    if isinstance(scope, AtomSetScope):
        return StructureReference.from_atoms(
            scope.atom_refs,
            realization=realization,
        )
    if isinstance(scope, AbsentResidueSpanScope):
        return StructureReference.from_absent_residue_span(
            preceding_residue_id=scope.preceding_residue_id,
            following_residue_id=scope.following_residue_id,
            absent_residue_ids=scope.absent_residue_ids,
            realization=realization,
        )
    if isinstance(scope, AnchorAtomPairScope):
        return StructureReference.from_anchor_atom_pair(
            scope.left_anchor_atom_ref,
            scope.right_anchor_atom_ref,
            realization=realization,
        )
    if isinstance(scope, CompositeScope):
        return StructureReference.composite(
            tuple(
                reference_from_scope(
                    member_scope,
                    realization_selector=None,
                )
                for member_scope in scope.scopes
            )
        )

    raise TypeError(
        "structure reference projection requires residue-set, atom-set, "
        "absent-span, anchor-pair, or composite scope"
    )
