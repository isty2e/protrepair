"""Canonical geometry-restraint templates for residue-level diagnostics."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BondRestraintTarget:
    """Expected bond-length target for one residue-local atom pair."""

    atom_name_1: str
    atom_name_2: str
    target_distance_angstrom: float
    esd_angstrom: float | None = None

    def __post_init__(self) -> None:
        atom_name_1 = self.atom_name_1.strip().upper()
        atom_name_2 = self.atom_name_2.strip().upper()
        if not atom_name_1 or not atom_name_2:
            raise ValueError("bond-restraint atom names must not be blank")

        if self.target_distance_angstrom <= 0.0:
            raise ValueError("bond-restraint distance must be positive")

        if self.esd_angstrom is not None and self.esd_angstrom <= 0.0:
            raise ValueError("bond-restraint esd must be positive when provided")

        object.__setattr__(self, "atom_name_1", atom_name_1)
        object.__setattr__(self, "atom_name_2", atom_name_2)

    def matches(self, atom_name_1: str, atom_name_2: str) -> bool:
        """Return whether this target matches one unordered atom-name pair."""

        left = atom_name_1.strip().upper()
        right = atom_name_2.strip().upper()
        return {left, right} == {self.atom_name_1, self.atom_name_2}


@dataclass(frozen=True, slots=True)
class AngleRestraintTarget:
    """Expected bond-angle target for one residue-local atom triplet."""

    atom_name_1: str
    center_atom_name: str
    atom_name_2: str
    target_angle_degrees: float
    esd_degrees: float | None = None

    def __post_init__(self) -> None:
        atom_name_1 = self.atom_name_1.strip().upper()
        center_atom_name = self.center_atom_name.strip().upper()
        atom_name_2 = self.atom_name_2.strip().upper()
        if not atom_name_1 or not center_atom_name or not atom_name_2:
            raise ValueError("angle-restraint atom names must not be blank")

        if len({atom_name_1, center_atom_name, atom_name_2}) != 3:
            raise ValueError("angle-restraint atom names must be unique")

        if not 0.0 < self.target_angle_degrees <= 180.0:
            raise ValueError("angle-restraint target must be between 0 and 180")

        if self.esd_degrees is not None and self.esd_degrees <= 0.0:
            raise ValueError("angle-restraint esd must be positive when provided")

        object.__setattr__(self, "atom_name_1", atom_name_1)
        object.__setattr__(self, "center_atom_name", center_atom_name)
        object.__setattr__(self, "atom_name_2", atom_name_2)

    def matches(
        self,
        *,
        atom_name_1: str,
        center_atom_name: str,
        atom_name_2: str,
    ) -> bool:
        """Return whether this target matches one residue-local angle."""

        normalized_center = center_atom_name.strip().upper()
        if normalized_center != self.center_atom_name:
            return False

        left = atom_name_1.strip().upper()
        right = atom_name_2.strip().upper()
        return {left, right} == {self.atom_name_1, self.atom_name_2}


@dataclass(frozen=True, slots=True)
class ChiralityRestraintTarget:
    """Expected tetrahedral orientation target for one residue-local center."""

    center_atom_name: str
    ordered_neighbor_atom_names: tuple[str, str, str]
    expected_orientation_sign: int

    def __post_init__(self) -> None:
        center_atom_name = self.center_atom_name.strip().upper()
        ordered_neighbor_atom_names = tuple(
            atom_name.strip().upper() for atom_name in self.ordered_neighbor_atom_names
        )
        if not center_atom_name:
            raise ValueError("chirality-restraint center atom name must not be blank")

        if len(ordered_neighbor_atom_names) != 3:
            raise ValueError(
                "chirality-restraint centers require exactly three neighbors"
            )

        if len(set(ordered_neighbor_atom_names)) != 3:
            raise ValueError("chirality-restraint neighbors must be unique")

        if any(not atom_name for atom_name in ordered_neighbor_atom_names):
            raise ValueError("chirality-restraint neighbors must not be blank")

        if self.expected_orientation_sign not in (-1, 1):
            raise ValueError("chirality-restraint expected sign must be -1 or 1")

        object.__setattr__(self, "center_atom_name", center_atom_name)
        object.__setattr__(
            self,
            "ordered_neighbor_atom_names",
            ordered_neighbor_atom_names,
        )


@dataclass(frozen=True, slots=True)
class PlaneRestraintTarget:
    """Expected planar heavy-atom set for one residue-local conjugated system."""

    atom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        atom_names = tuple(atom_name.strip().upper() for atom_name in self.atom_names)
        if len(atom_names) < 4:
            raise ValueError("plane-restraint targets require at least four atoms")

        if len(atom_names) != len(set(atom_names)):
            raise ValueError("plane-restraint atom names must be unique")

        if any(not atom_name for atom_name in atom_names):
            raise ValueError("plane-restraint atom names must not be blank")

        object.__setattr__(self, "atom_names", atom_names)


@dataclass(frozen=True, slots=True)
class ResidueRestraintTemplate:
    """Canonical geometry-restraint targets for one residue."""

    component_id: str
    bond_targets: tuple[BondRestraintTarget, ...]
    angle_targets: tuple[AngleRestraintTarget, ...]
    chirality_targets: tuple[ChiralityRestraintTarget, ...] = ()
    plane_targets: tuple[PlaneRestraintTarget, ...] = ()

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        if not component_id:
            raise ValueError("restraint template component id must not be blank")

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "bond_targets", tuple(self.bond_targets))
        object.__setattr__(self, "angle_targets", tuple(self.angle_targets))
        object.__setattr__(self, "chirality_targets", tuple(self.chirality_targets))
        object.__setattr__(self, "plane_targets", tuple(self.plane_targets))

    def bond_target(
        self,
        atom_name_1: str,
        atom_name_2: str,
    ) -> BondRestraintTarget | None:
        """Return the target for one unordered bond pair if present."""

        for target in self.bond_targets:
            if target.matches(atom_name_1, atom_name_2):
                return target

        return None

    def angle_targets_for_center(
        self,
        center_atom_name: str,
    ) -> tuple[AngleRestraintTarget, ...]:
        """Return all angle targets centered on one atom name."""

        normalized_center = center_atom_name.strip().upper()
        return tuple(
            target
            for target in self.angle_targets
            if target.center_atom_name == normalized_center
        )
