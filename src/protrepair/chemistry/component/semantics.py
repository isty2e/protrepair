"""Heavy-atom and hydrogen completion semantics over component models."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.chemistry.internal_coordinates import InternalCoordinateProgram
from protrepair.geometry import (
    InternalCoordinateFrame,
    PlanarCenter,
    TetrahedralCenter,
    Vec3,
)


class RotatableHydrogenKind(str, Enum):
    """Rotatable sidechain hydrogen families."""

    CYS = "CYS"
    SER = "SER"
    THR = "THR"
    TYR = "TYR"


ROTATABLE_HYDROGEN_ANCHOR_BY_NAME: Mapping[
    RotatableHydrogenKind,
    Mapping[str, str],
] = MappingProxyType(
    {
        RotatableHydrogenKind.CYS: MappingProxyType(
            {"HA": "CA", "HB1": "CB", "HB2": "CB", "HG": "SG"}
        ),
        RotatableHydrogenKind.SER: MappingProxyType(
            {"HA": "CA", "HB1": "CB", "HB2": "CB", "HG": "OG"}
        ),
        RotatableHydrogenKind.THR: MappingProxyType(
            {
                "HG1": "OG1",
                "HA": "CA",
                "HB": "CB",
                "1HG2": "CG2",
                "2HG2": "CG2",
                "3HG2": "CG2",
            }
        ),
        RotatableHydrogenKind.TYR: MappingProxyType(
            {
                "HA": "CA",
                "HB1": "CB",
                "HB2": "CB",
                "HD1": "CD1",
                "HD2": "CD2",
                "HE1": "CE1",
                "HE2": "CE2",
                "HH": "OH",
            }
        ),
    }
)
ROTATABLE_HYDROGEN_ATOM_NAMES: Mapping[
    RotatableHydrogenKind,
    tuple[str, ...],
] = MappingProxyType(
    {
        rotatable_kind: tuple(anchor_by_name)
        for rotatable_kind, anchor_by_name in (
            ROTATABLE_HYDROGEN_ANCHOR_BY_NAME.items()
        )
    }
)
IDEALIZED_BACKBONE_OR_TERMINAL_HYDROGEN_ANCHORS: frozenset[str] = frozenset(
    {"N", "OXT"}
)


class IdealGeometryHeavyAtomMode(str, Enum):
    """Canonical heavy-atom completion strategies driven by ideal coordinates."""

    LOCAL_FRAME = "local_frame"
    RESIDUE_FRAME = "residue_frame"
    REFERENCE_OR_SALVAGE = "reference_or_salvage"


HydrogenPlanArgument = str | float | int
HydrogenOperation = tuple[
    tuple[str, ...],
    str,
    tuple[HydrogenPlanArgument, ...],
]


@dataclass(frozen=True, slots=True)
class IdealGeometryHeavyAtomSemantics:
    """Heavy-atom completion semantics driven by an idealized component model."""

    component: IdealizedComponent
    mode: IdealGeometryHeavyAtomMode

    def requires_local_refinement(self) -> bool:
        """Return whether this path is gated behind explicit local refinement."""

        return self.mode is IdealGeometryHeavyAtomMode.RESIDUE_FRAME

    def allows_reference_or_salvage(self) -> bool:
        """Return whether this path may salvage a retained reference fragment."""

        return self.mode is IdealGeometryHeavyAtomMode.REFERENCE_OR_SALVAGE


@dataclass(frozen=True, slots=True)
class IdealGeometryHydrogenSemantics:
    """Hydrogen-placement semantics driven by an idealized component model."""

    component: IdealizedComponent

    def expected_hydrogen_atom_names(self) -> tuple[str, ...]:
        """Return hydrogen atom names owned by this idealized hydrogen source."""

        expected_hydrogen_atom_names: list[str] = []
        for atom in self.component.hydrogen_atoms():
            anchor_atom_name = self.component.hydrogen_anchor_atom_name(atom.atom_name)
            if (
                anchor_atom_name is None
                or anchor_atom_name in IDEALIZED_BACKBONE_OR_TERMINAL_HYDROGEN_ANCHORS
            ):
                continue

            expected_hydrogen_atom_names.append(atom.atom_name)

        return tuple(expected_hydrogen_atom_names)

    def template_hydrogen_anchor_by_name(self) -> Mapping[str, str]:
        """Return template-resolved heavy-atom anchors for idealized hydrogens."""

        anchor_by_name: dict[str, str] = {}
        for hydrogen_atom in self.component.hydrogen_atoms():
            anchor_atom_name = self.component.hydrogen_anchor_atom_name(
                hydrogen_atom.atom_name
            )
            if anchor_atom_name is None:
                continue

            anchor_by_name[hydrogen_atom.atom_name] = anchor_atom_name

        return MappingProxyType(anchor_by_name)


@dataclass(frozen=True, slots=True)
class HeavyAtomSemantics:
    """Heavy-atom repair semantics attached to a residue template."""

    program: InternalCoordinateProgram
    atom_order: tuple[str, ...]

    def __post_init__(self) -> None:
        atom_order = tuple(atom_name.strip().upper() for atom_name in self.atom_order)
        if not atom_order:
            raise ValueError("heavy-atom semantics require at least one atom name")

        if any(not atom_name for atom_name in atom_order):
            raise ValueError("heavy-atom semantics atom names must not be blank")

        if len(atom_order) != len(set(atom_order)):
            raise ValueError("heavy-atom semantics atom names must be unique")

        object.__setattr__(self, "atom_order", atom_order)


@dataclass(frozen=True, slots=True)
class HydrogenSemantics:
    """Hydrogen-placement semantics attached to a residue template."""

    plan_with_backbone: tuple[HydrogenOperation, ...] | None = None
    plan_without_backbone: tuple[HydrogenOperation, ...] | None = None
    rotatable_kind: RotatableHydrogenKind | None = None

    def __post_init__(self) -> None:
        uses_static_plan = (
            self.plan_with_backbone is not None
            or self.plan_without_backbone is not None
        )
        uses_rotatable_plan = self.rotatable_kind is not None

        if uses_static_plan and uses_rotatable_plan:
            raise ValueError(
                "hydrogen semantics must use either static plans or a rotatable kind"
            )

        if not uses_static_plan and not uses_rotatable_plan:
            raise ValueError(
                "hydrogen semantics require a static plan or a rotatable kind"
            )

        if self.plan_without_backbone is not None and self.plan_with_backbone is None:
            raise ValueError(
                "hydrogen semantics with_backbone plan is required when "
                "plan_without_backbone is provided"
            )

    def static_plan(
        self,
        *,
        include_backbone_hydrogen: bool,
    ) -> tuple[HydrogenOperation, ...] | None:
        """Return the active static plan for a residue context."""

        if not include_backbone_hydrogen and self.plan_without_backbone is not None:
            return self.plan_without_backbone

        return self.plan_with_backbone

    def template_hydrogen_anchor_by_name(self) -> Mapping[str, str]:
        """Return explicit heavy-atom anchors encoded by static plans."""

        if self.rotatable_kind is not None:
            return ROTATABLE_HYDROGEN_ANCHOR_BY_NAME[self.rotatable_kind]

        anchor_by_name: dict[str, str] = {}
        for plan in (self.plan_with_backbone, self.plan_without_backbone):
            if plan is None:
                continue

            for hydrogen_atom_names, method_name, arguments in plan:
                anchor_atom_name = HydrogenSemantics.anchor_atom_name_for_operation(
                    method_name=method_name,
                    arguments=arguments,
                )
                if anchor_atom_name is None:
                    continue

                for hydrogen_atom_name in hydrogen_atom_names:
                    previous_anchor = anchor_by_name.get(hydrogen_atom_name)
                    if (
                        previous_anchor is not None
                        and previous_anchor != anchor_atom_name
                    ):
                        raise ValueError(
                            "hydrogen semantics encode conflicting anchor atoms for "
                            f"{hydrogen_atom_name!r}"
                        )

                    anchor_by_name[hydrogen_atom_name] = anchor_atom_name

        return MappingProxyType(anchor_by_name)

    @staticmethod
    def anchor_atom_name_for_operation(
        *,
        method_name: str,
        arguments: tuple[HydrogenPlanArgument, ...],
    ) -> str | None:
        """Return the anchor atom name encoded by one hydrogen-plan operation."""

        if method_name in {"class2", "class3", "class4", "calcCoordinate"}:
            anchor_index = 2
        elif method_name == "class5":
            anchor_index = 1
        else:
            return None

        if len(arguments) <= anchor_index:
            return None

        anchor_atom_name = arguments[anchor_index]
        if not isinstance(anchor_atom_name, str):
            return None

        normalized_anchor_atom_name = anchor_atom_name.strip().upper()
        if not normalized_anchor_atom_name:
            return None

        return normalized_anchor_atom_name

    def expected_hydrogen_atom_names(self) -> tuple[str, ...]:
        """Return canonical hydrogen atom names generated by this semantics block."""

        if self.rotatable_kind is not None:
            return ROTATABLE_HYDROGEN_ATOM_NAMES[self.rotatable_kind]

        ordered_hydrogen_atom_names: list[str] = []
        seen_hydrogen_atom_names: set[str] = set()
        for plan in (self.plan_with_backbone, self.plan_without_backbone):
            if plan is None:
                continue

            for hydrogen_atom_names, _method_name, _arguments in plan:
                for hydrogen_atom_name in hydrogen_atom_names:
                    normalized_hydrogen_atom_name = hydrogen_atom_name.strip().upper()
                    if normalized_hydrogen_atom_name in seen_hydrogen_atom_names:
                        continue

                    ordered_hydrogen_atom_names.append(normalized_hydrogen_atom_name)
                    seen_hydrogen_atom_names.add(normalized_hydrogen_atom_name)

        return tuple(ordered_hydrogen_atom_names)

    @staticmethod
    def evaluate_operation(
        method_name: str,
        arguments: tuple[HydrogenPlanArgument, ...],
        *,
        atom_coordinates: Mapping[str, Vec3],
    ) -> tuple[Vec3, ...]:
        """Evaluate one hydrogen-plan operation into one or more coordinates."""

        if method_name == "class2":
            bond_length = (
                HydrogenSemantics.resolve_numeric_argument(arguments[3])
                if len(arguments) == 4
                else 1.09
            )
            first, second = TetrahedralCenter(
                HydrogenSemantics.resolve_coordinate_argument(
                    arguments[0],
                    atom_coordinates,
                ),
                HydrogenSemantics.resolve_coordinate_argument(
                    arguments[1],
                    atom_coordinates,
                ),
                HydrogenSemantics.resolve_coordinate_argument(
                    arguments[2],
                    atom_coordinates,
                ),
            ).pair(bond_length=bond_length)
            return (first, second)

        if method_name == "class3":
            return (
                TetrahedralCenter(
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[0],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[1],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[2],
                        atom_coordinates,
                    ),
                ).remaining(),
            )

        if method_name == "class4":
            return (
                PlanarCenter(
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[0],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[2],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[1],
                        atom_coordinates,
                    ),
                ).bisector(),
            )

        if method_name == "class5":
            return (
                PlanarCenter(
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[0],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[1],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[2],
                        atom_coordinates,
                    ),
                ).projected(
                    bond_length=HydrogenSemantics.resolve_numeric_argument(
                        arguments[3]
                    ),
                ),
            )

        if method_name == "calcCoordinate":
            return (
                InternalCoordinateFrame(
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[0],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[1],
                        atom_coordinates,
                    ),
                    HydrogenSemantics.resolve_coordinate_argument(
                        arguments[2],
                        atom_coordinates,
                    ),
                ).place(
                    bond_length=HydrogenSemantics.resolve_numeric_argument(
                        arguments[3]
                    ),
                    dihedral_degrees=HydrogenSemantics.resolve_numeric_argument(
                        arguments[4]
                    ),
                    bond_angle_degrees=HydrogenSemantics.resolve_numeric_argument(
                        arguments[5]
                    ),
                ),
            )

        raise ValueError(f"unsupported hydrogen geometry method {method_name!r}")

    @staticmethod
    def resolve_coordinate_argument(
        argument: HydrogenPlanArgument,
        atom_coordinates: Mapping[str, Vec3],
    ) -> Vec3:
        """Resolve one named atom argument to a coordinate."""

        if not isinstance(argument, str):
            raise TypeError("coordinate arguments must resolve to atom names")

        return atom_coordinates[argument]

    @staticmethod
    def resolve_numeric_argument(argument: HydrogenPlanArgument) -> float:
        """Resolve one numeric hydrogen-plan argument."""

        if isinstance(argument, str):
            raise TypeError("numeric hydrogen-plan arguments must be floats")

        return float(argument)
