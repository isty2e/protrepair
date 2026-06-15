"""Refinement transformer request specifications."""

from dataclasses import dataclass, field
from enum import Enum

from protrepair.scope import ResidueSetScope
from protrepair.structure.labels import ResidueId
from protrepair.transformer.continuous.binding import (
    ContinuousRelaxationBinding,
    ManualContinuousRelaxationBinding,
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationConfig
from protrepair.transformer.local.models import LocalScopeSpec

BACKBONE_WINDOW_MOVABLE_ATOM_NAMES = ("N", "CA", "C", "O", "OXT")


class RefinementOperatorFamily(str, Enum):
    """Closed refinement transformer operator families."""

    BACKBONE_WINDOW_REFINEMENT = "backbone_window_refinement"


@dataclass(frozen=True, slots=True)
class BackboneWindowRefinementSpec:
    """Contract for correction operators that may move an ordered backbone window.

    This spec is intentionally not a local continuous-relaxation request. It
    defines the operator family needed for low-resolution misthread/window cases
    while keeping residue-local FF refinement honest about its smaller domain.
    """

    residue_ids: tuple[ResidueId, ...]
    movable_atom_names: tuple[str, ...] = BACKBONE_WINDOW_MOVABLE_ATOM_NAMES

    def __post_init__(self) -> None:
        residue_ids = tuple(dict.fromkeys(self.residue_ids))
        if len(residue_ids) < 2:
            raise ValueError(
                "backbone-window refinement requires at least two residues"
            )

        chain_ids = {residue_id.chain_id for residue_id in residue_ids}
        if len(chain_ids) != 1:
            raise ValueError(
                "backbone-window refinement requires one ordered chain window"
            )
        if tuple(sorted(residue_ids)) != residue_ids:
            raise ValueError(
                "backbone-window refinement residue ids must be in chain order"
            )

        movable_atom_names = _normalize_backbone_window_atom_names(
            self.movable_atom_names
        )
        object.__setattr__(self, "residue_ids", residue_ids)
        object.__setattr__(self, "movable_atom_names", movable_atom_names)

    @property
    def operator_family(self) -> RefinementOperatorFamily:
        """Return the operator family represented by this contract."""

        return RefinementOperatorFamily.BACKBONE_WINDOW_REFINEMENT

    def as_residue_scope(self) -> ResidueSetScope:
        """Return the addressed residue window as a semantic residue scope."""

        return ResidueSetScope(residue_ids=self.residue_ids)


def _normalize_backbone_window_atom_names(
    atom_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return validated backbone-window atom names in first-seen order."""

    ordered_atom_names: list[str] = []
    seen_atom_names: set[str] = set()
    allowed_atom_names = set(BACKBONE_WINDOW_MOVABLE_ATOM_NAMES)
    for atom_name in atom_names:
        normalized_atom_name = atom_name.strip().upper()
        if not normalized_atom_name:
            raise ValueError(
                "backbone-window movable atom names must not contain blanks"
            )
        if normalized_atom_name not in allowed_atom_names:
            raise ValueError(
                "backbone-window movable atom names must be polymer backbone atoms"
            )
        if normalized_atom_name in seen_atom_names:
            continue

        ordered_atom_names.append(normalized_atom_name)
        seen_atom_names.add(normalized_atom_name)

    if not ordered_atom_names:
        raise ValueError("backbone-window refinement requires movable atom names")

    return tuple(ordered_atom_names)


@dataclass(frozen=True, slots=True)
class _StageRefinementSpec:
    """Private shared fields for stage refinement specs."""

    scope_spec: LocalScopeSpec
    execution_scope_spec: LocalScopeSpec | None = field(
        default=None,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.scope_spec, LocalScopeSpec):
            raise TypeError(
                "stage refinement scope_spec must be a LocalScopeSpec value"
            )
        if self.scope_spec.includes_attached_hydrogens():
            raise TypeError(
                "stage refinement scope_spec must not request attached-hydrogen closure"
            )
        if self.execution_scope_spec is None:
            return

        if not isinstance(self.execution_scope_spec, LocalScopeSpec):
            raise TypeError(
                "stage refinement execution_scope_spec must be a LocalScopeSpec "
                "value when present"
            )
        if self.execution_scope_spec.includes_attached_hydrogens():
            raise TypeError(
                "stage refinement execution_scope_spec must not request "
                "attached-hydrogen closure"
            )
        if not self.scope_spec.is_residuewise():
            raise TypeError(
                "stage refinement execution_scope_spec is supported only for "
                "residuewise scope specs"
            )
        if not self.execution_scope_spec.is_residuewise():
            raise TypeError(
                "stage refinement execution_scope_spec must stay residuewise"
            )

        semantic_residue_ids = set(self.scope_spec.referenced_residue_ids())
        execution_residue_ids = set(
            self.execution_scope_spec.referenced_residue_ids()
        )
        if not semantic_residue_ids <= execution_residue_ids:
            raise ValueError(
                "stage refinement execution_scope_spec must cover the semantic "
                "scope residue ids"
            )

    def resolved_execution_scope_spec(self) -> LocalScopeSpec:
        """Return the execution-local scope spec to lower against one snapshot."""

        return (
            self.scope_spec
            if self.execution_scope_spec is None
            else self.execution_scope_spec
        )


@dataclass(frozen=True, slots=True)
class RepairRefinementSpec(_StageRefinementSpec):
    """Repair-stage local-refinement spec."""

    binding: ContinuousRelaxationBinding
    config: ContinuousRelaxationConfig = field(
        default_factory=ContinuousRelaxationConfig
    )

    def __post_init__(self) -> None:
        _StageRefinementSpec.__post_init__(self)
        if not isinstance(
            self.binding,
            (
                ManualContinuousRelaxationBinding,
                RecommendedContinuousRelaxationBinding,
            ),
        ):
            raise TypeError(
                "repair refinement binding must be a manual or recommended "
                "continuous-relaxation binding value"
            )
        if not isinstance(self.config, ContinuousRelaxationConfig):
            raise TypeError(
                "repair refinement config must be a ContinuousRelaxationConfig value"
            )
