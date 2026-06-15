"""Internal-coordinate programs used by heavy-atom completion semantics."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol


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
