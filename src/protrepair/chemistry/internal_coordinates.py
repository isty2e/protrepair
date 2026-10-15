"""Internal-coordinate programs used by heavy-atom completion semantics."""

from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Protocol

from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
)


def _normalize_atom_name(atom_name: str) -> str:
    """Return one canonical atom name token."""

    normalized_atom_name = atom_name.strip().upper()
    if not normalized_atom_name:
        raise ValueError("internal-coordinate atom names must not be blank")

    return normalized_atom_name


def _normalize_atom_names(atom_names: Collection[str]) -> tuple[str, ...]:
    """Return canonical atom names while preserving order."""

    normalized_atom_names = tuple(
        _normalize_atom_name(atom_name) for atom_name in atom_names
    )
    if len(normalized_atom_names) != len(set(normalized_atom_names)):
        raise ValueError("internal-coordinate atom names must be unique")

    return normalized_atom_names


class InternalCoordinateExecutionContext(Protocol):
    """Execution substrate for one internal-coordinate program."""

    def has_atom(self, atom_name: str) -> bool:
        """Return whether one named atom is currently present."""

        ...

    def missing_any(self, atom_names: Collection[str]) -> bool:
        """Return whether any named atom was missing at program ingress."""

        ...

    def torsion(self, atom_names: tuple[str, str, str, str]) -> float:
        """Return the current torsion angle for four named atoms."""

        ...

    def ensure_backbone_oxygen(self) -> None:
        """Ensure the residue backbone oxygen is present."""

        ...

    def ensure_beta_carbon(self, *, dihedral_degrees: float) -> None:
        """Ensure the residue beta carbon is present."""

        ...

    def place(self, placement: "InternalCoordinatePlacement") -> None:
        """Place one atom according to one internal-coordinate instruction."""

        ...

    def place_with_bond_closure(
        self,
        placement: "InternalCoordinatePlacement",
        *,
        closing_atom_name: str,
        closing_bond_length: float,
        bond_targets: tuple[BondRestraintTarget, ...],
        angle_targets: tuple[AngleRestraintTarget, ...],
    ) -> None:
        """Place a missing target while closing a second materialized bond."""

        ...


class InternalCoordinateOperation(Protocol):
    """One executable internal-coordinate program operation."""

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Apply this operation to one execution context."""

        ...


class InternalCoordinateDihedral(Protocol):
    """One dihedral expression resolved against a placement context."""

    def resolve(self, context: InternalCoordinateExecutionContext) -> float:
        """Return the dihedral angle in degrees for one execution context."""

        ...


@dataclass(frozen=True, slots=True)
class AbsoluteDihedral:
    """One literal dihedral angle in degrees."""

    degrees: float

    def resolve(self, context: InternalCoordinateExecutionContext) -> float:
        """Return the literal dihedral angle."""

        del context
        return float(self.degrees)


@dataclass(frozen=True, slots=True)
class ReferencedTorsionDihedral:
    """One dihedral angle defined relative to a currently observed torsion."""

    atom_names: tuple[str, str, str, str]
    offset_degrees: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_names", _normalize_atom_names(self.atom_names))

    def resolve(self, context: InternalCoordinateExecutionContext) -> float:
        """Return the referenced torsion plus the configured offset."""

        return context.torsion(self.atom_names) + float(self.offset_degrees)


@dataclass(frozen=True, slots=True)
class InternalCoordinatePlacement:
    """One atom placement defined by anchors and internal coordinates."""

    target_atom_name: str
    anchor_atom_names: tuple[str, str, str]
    bond_length: float
    bond_angle: float
    dihedral: AbsoluteDihedral | ReferencedTorsionDihedral

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_atom_name",
            _normalize_atom_name(self.target_atom_name),
        )
        object.__setattr__(
            self,
            "anchor_atom_names",
            _normalize_atom_names(self.anchor_atom_names),
        )

        if len(self.anchor_atom_names) != 3:
            raise ValueError(
                "internal-coordinate placement requires exactly three anchors"
            )

    @classmethod
    def absolute(
        cls,
        *,
        target_atom_name: str,
        anchor_atom_names: tuple[str, str, str],
        bond_length: float,
        bond_angle: float,
        dihedral_degrees: float,
    ) -> "InternalCoordinatePlacement":
        """Build one placement with a literal dihedral angle."""

        return cls(
            target_atom_name=target_atom_name,
            anchor_atom_names=anchor_atom_names,
            bond_length=bond_length,
            bond_angle=bond_angle,
            dihedral=AbsoluteDihedral(dihedral_degrees),
        )

    @classmethod
    def relative_to_torsion(
        cls,
        *,
        target_atom_name: str,
        anchor_atom_names: tuple[str, str, str],
        bond_length: float,
        bond_angle: float,
        reference_atom_names: tuple[str, str, str, str],
        offset_degrees: float,
    ) -> "InternalCoordinatePlacement":
        """Build one placement relative to a referenced torsion angle."""

        return cls(
            target_atom_name=target_atom_name,
            anchor_atom_names=anchor_atom_names,
            bond_length=bond_length,
            bond_angle=bond_angle,
            dihedral=ReferencedTorsionDihedral(
                atom_names=reference_atom_names,
                offset_degrees=offset_degrees,
            ),
        )

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Place the target atom onto one execution context."""

        context.place(self)


@dataclass(frozen=True, slots=True)
class EnsureBackboneOxygen:
    """Ensure the residue backbone oxygen is present."""

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Apply the backbone-oxygen operation."""

        context.ensure_backbone_oxygen()


@dataclass(frozen=True, slots=True)
class EnsureBetaCarbon:
    """Ensure the residue beta carbon is present."""

    dihedral_degrees: float

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Apply the beta-carbon operation."""

        context.ensure_beta_carbon(dihedral_degrees=float(self.dihedral_degrees))


@dataclass(frozen=True, slots=True)
class PlaceWhenAbsent:
    """Place one atom only when it is currently absent."""

    placement: InternalCoordinatePlacement

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Apply the placement when the target atom is absent."""

        if context.has_atom(self.placement.target_atom_name):
            return

        self.placement.apply(context)


@dataclass(frozen=True, slots=True)
class PlaceWhenAnyMissing:
    """Place a coherent atom group when any trigger atom was missing at ingress."""

    trigger_atom_names: tuple[str, ...]
    placements: tuple[InternalCoordinatePlacement, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trigger_atom_names",
            _normalize_atom_names(self.trigger_atom_names),
        )
        object.__setattr__(self, "placements", tuple(self.placements))
        if not self.placements:
            raise ValueError(
                "conditional internal-coordinate groups require placements"
            )

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Apply all placements when any trigger atom was missing."""

        if not context.missing_any(self.trigger_atom_names):
            return

        for placement in self.placements:
            placement.apply(context)


@dataclass(frozen=True, slots=True)
class CompleteMissingAtomGroup:
    """Fill missing targets without replacing materialized group geometry.

    Parameters
    ----------
    placements : tuple[InternalCoordinatePlacement, ...]
        Ordered fallback placements. An already materialized same-frame sibling
        supplies its torsion with the recipe's relative angular offset. Targets
        generated during this operation do not become observation references.
    bond_targets : tuple[BondRestraintTarget, ...]
        Dictionary targets for non-parent closure bonds. These are used only
        when part of the group was materialized at operation entry.
    angle_targets : tuple[AngleRestraintTarget, ...]
        Dictionary angles used to disambiguate feasible closure solutions.
    """

    placements: tuple[InternalCoordinatePlacement, ...]
    bond_targets: tuple[BondRestraintTarget, ...] = ()
    angle_targets: tuple[AngleRestraintTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        if not self.placements:
            raise ValueError("missing-atom groups require placements")
        _normalize_atom_names(
            tuple(placement.target_atom_name for placement in self.placements)
        )
        object.__setattr__(self, "bond_targets", tuple(self.bond_targets))
        object.__setattr__(self, "angle_targets", tuple(self.angle_targets))

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Fill absent targets using materialized same-frame siblings.

        Parameters
        ----------
        context : InternalCoordinateExecutionContext
            Current residue-local geometry and placement execution substrate.
        """

        materialized = frozenset(
            placement.target_atom_name
            for placement in self.placements
            if context.has_atom(placement.target_atom_name)
        )
        for placement in self.placements:
            if context.has_atom(placement.target_atom_name):
                continue

            if isinstance(placement.dihedral, AbsoluteDihedral):
                for reference in self.placements:
                    if (
                        reference.target_atom_name in materialized
                        and reference.anchor_atom_names == placement.anchor_atom_names
                        and isinstance(reference.dihedral, AbsoluteDihedral)
                    ):
                        placement = replace(
                            placement,
                            dihedral=ReferencedTorsionDihedral(
                                (
                                    *reference.anchor_atom_names,
                                    reference.target_atom_name,
                                ),
                                placement.dihedral.degrees - reference.dihedral.degrees,
                            ),
                        )
                        break

            closing_target = self._closing_target(placement, context)
            if materialized and closing_target is not None:
                closing_name, closing_distance = closing_target
                context.place_with_bond_closure(
                    placement,
                    closing_atom_name=closing_name,
                    closing_bond_length=closing_distance,
                    bond_targets=self.bond_targets,
                    angle_targets=self.angle_targets,
                )
            else:
                placement.apply(context)

    def _closing_target(
        self,
        placement: InternalCoordinatePlacement,
        context: InternalCoordinateExecutionContext,
    ) -> tuple[str, float] | None:
        for bond in self.bond_targets:
            if bond.atom_name_1 == placement.target_atom_name:
                closing_name = bond.atom_name_2
            elif bond.atom_name_2 == placement.target_atom_name:
                closing_name = bond.atom_name_1
            else:
                continue

            if closing_name not in placement.anchor_atom_names and context.has_atom(
                closing_name
            ):
                return closing_name, bond.target_distance_angstrom

        return None


@dataclass(frozen=True, slots=True)
class InternalCoordinateProgram:
    """Ordered internal-coordinate operations for one residue template."""

    operations: tuple[InternalCoordinateOperation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.operations:
            raise ValueError(
                "internal-coordinate programs require at least one operation"
            )

    @classmethod
    def backbone_only(cls) -> "InternalCoordinateProgram":
        """Build one program that only ensures backbone oxygen."""

        return cls(operations=(EnsureBackboneOxygen(),))

    @classmethod
    def with_backbone_and_beta_carbon(
        cls,
        *,
        cb_dihedral_degrees: float,
        operations: tuple[InternalCoordinateOperation, ...] = (),
    ) -> "InternalCoordinateProgram":
        """Build one program that ensures backbone oxygen and beta carbon first."""

        return cls(
            operations=(
                EnsureBackboneOxygen(),
                EnsureBetaCarbon(dihedral_degrees=cb_dihedral_degrees),
                *operations,
            )
        )

    def apply(self, context: InternalCoordinateExecutionContext) -> None:
        """Execute this program over one placement context."""

        for operation in self.operations:
            operation.apply(context)

    def as_missing_atom_completion(
        self,
        *,
        bond_targets: tuple[BondRestraintTarget, ...] = (),
        angle_targets: tuple[AngleRestraintTarget, ...] = (),
    ) -> "InternalCoordinateProgram":
        """Lower replacement groups to materialized-anchor-aware completion.

        Parameters
        ----------
        bond_targets : tuple[BondRestraintTarget, ...]
            Existing dictionary targets available for partial ring closure.
        angle_targets : tuple[AngleRestraintTarget, ...]
            Existing dictionary angles for closure-solution selection.

        Returns
        -------
        InternalCoordinateProgram
            A missing-only program. The original literal/replacement program and
            non-group operations remain unchanged.
        """

        return type(self)(
            operations=tuple(
                CompleteMissingAtomGroup(
                    operation.placements, bond_targets, angle_targets
                )
                if isinstance(operation, PlaceWhenAnyMissing)
                else operation
                for operation in self.operations
            )
        )
