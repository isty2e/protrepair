"""Bundled nonstandard-component registry over packaged canonical assets."""

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import cast

from protrepair.chemistry.component.graph import (
    BondDefinition,
    ChemicalComponentDefinition,
)
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import (
    IdealGeometryHeavyAtomMode,
    IdealGeometryHeavyAtomSemantics,
    IdealGeometryHydrogenSemantics,
)
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.nonstandard.support_policy import (
    supports_bundled_nonstandard_heavy_repair,
    supports_bundled_nonstandard_hydrogenation,
    supports_bundled_nonstandard_reference_or_salvage_repair,
    supports_bundled_nonstandard_template_repair,
)
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ChiralityRestraintTarget,
    PlaneRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.structure.element import ElementIdentity

IdealPosition = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class NonstandardComponentAtom:
    """One packaged nonstandard-component atom record."""

    atom_name: str
    element: str
    formal_charge: int
    stereo_config: str | None = None
    ideal_position: IdealPosition | None = None

    def __post_init__(self) -> None:
        atom_name = self.atom_name.strip().upper()
        element = self.element.strip().upper()
        if not atom_name:
            raise ValueError("nonstandard atom name must not be blank")

        if not element:
            raise ValueError("nonstandard atom element must not be blank")

        object.__setattr__(self, "atom_name", atom_name)
        object.__setattr__(self, "element", element)
        if self.stereo_config is not None:
            normalized_stereo_config = self.stereo_config.strip().upper()
            if normalized_stereo_config not in {"R", "S"}:
                raise ValueError(
                    "nonstandard atom stereo config must be R or S when provided"
                )
            object.__setattr__(self, "stereo_config", normalized_stereo_config)

        if self.ideal_position is not None:
            if len(self.ideal_position) != 3:
                raise ValueError(
                    "nonstandard atom ideal position must contain three coordinates"
                )
            object.__setattr__(
                self,
                "ideal_position",
                tuple(float(value) for value in self.ideal_position),
            )

    def is_hydrogen(self) -> bool:
        """Return whether this atom record represents hydrogen."""

        return ElementIdentity(self.element).is_hydrogen()

    def to_idealized_atom(self) -> IdealizedComponentAtom:
        """Return the canonical idealized atom projected from this source record."""

        return IdealizedComponentAtom(
            atom_name=self.atom_name,
            element=self.element,
            formal_charge=self.formal_charge,
            stereo_config=self.stereo_config,
            ideal_position=self.ideal_position,
        )


@dataclass(frozen=True, slots=True)
class NonstandardComponentRecord:
    """One bundled nonstandard residue normalized into canonical asset form."""

    component_id: str
    parent_standard_id: str | None
    chem_comp_type: str
    name: str
    atoms: tuple[NonstandardComponentAtom, ...]
    bonds: tuple[BondDefinition, ...]
    bond_targets: tuple[BondRestraintTarget, ...] = ()
    angle_targets: tuple[AngleRestraintTarget, ...] = ()
    chirality_targets: tuple[ChiralityRestraintTarget, ...] = ()
    plane_targets: tuple[PlaneRestraintTarget, ...] = ()
    one_letter_code: str | None = None
    pcm_categories: tuple[str, ...] = ()
    pcm_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        chem_comp_type = self.chem_comp_type.strip().upper()
        name = self.name.strip()
        if not component_id:
            raise ValueError("nonstandard component id must not be blank")

        if not chem_comp_type:
            raise ValueError("nonstandard chemical component type must not be blank")

        if not name:
            raise ValueError("nonstandard component name must not be blank")

        if not self.atoms:
            raise ValueError("nonstandard component records require atoms")

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "chem_comp_type", chem_comp_type)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        object.__setattr__(self, "bond_targets", tuple(self.bond_targets))
        object.__setattr__(self, "angle_targets", tuple(self.angle_targets))
        object.__setattr__(self, "chirality_targets", tuple(self.chirality_targets))
        object.__setattr__(self, "plane_targets", tuple(self.plane_targets))
        object.__setattr__(self, "pcm_categories", tuple(self.pcm_categories))
        object.__setattr__(self, "pcm_types", tuple(self.pcm_types))
        if self.parent_standard_id is not None:
            object.__setattr__(
                self,
                "parent_standard_id",
                self.parent_standard_id.strip().upper() or None,
            )
        if self.one_letter_code is not None:
            object.__setattr__(
                self,
                "one_letter_code",
                self.one_letter_code.strip().upper() or None,
            )

    def heavy_atoms(self) -> tuple[NonstandardComponentAtom, ...]:
        """Return only non-hydrogen atom records in packaged order."""

        return tuple(atom for atom in self.atoms if not atom.is_hydrogen())

    def heavy_bonds(self) -> tuple[BondDefinition, ...]:
        """Return only bonds whose endpoints are both non-hydrogen atoms."""

        heavy_atom_names = {atom.atom_name for atom in self.heavy_atoms()}
        return tuple(
            bond
            for bond in self.bonds
            if bond.atom_name_1 in heavy_atom_names
            and bond.atom_name_2 in heavy_atom_names
        )

    def to_template(self) -> ResidueTemplate:
        """Project the packaged record into a canonical residue template."""

        idealized_component = self.to_idealized_component()
        heavy_atom_names = idealized_component.heavy_atom_names()
        formal_charges = {
            atom.atom_name: atom.formal_charge
            for atom in idealized_component.heavy_atoms()
            if atom.formal_charge != 0
        }
        heavy_atom_semantics = None
        hydrogen_semantics = None
        if supports_bundled_nonstandard_heavy_repair(self.component_id):
            repair_mode = (
                IdealGeometryHeavyAtomMode.REFERENCE_OR_SALVAGE
                if supports_bundled_nonstandard_reference_or_salvage_repair(
                    self.component_id
                )
                else (
                    IdealGeometryHeavyAtomMode.LOCAL_FRAME
                    if supports_bundled_nonstandard_template_repair(self.component_id)
                    else IdealGeometryHeavyAtomMode.RESIDUE_FRAME
                )
            )
            heavy_atom_semantics = IdealGeometryHeavyAtomSemantics(
                component=idealized_component,
                mode=repair_mode,
            )

        if supports_bundled_nonstandard_hydrogenation(self.component_id):
            hydrogen_semantics = IdealGeometryHydrogenSemantics(
                component=idealized_component
            )

        return ResidueTemplate(
            definition=ChemicalComponentDefinition(
                component_id=self.component_id,
                atom_names=heavy_atom_names,
                bonds=idealized_component.heavy_bonds(),
                formal_charges=formal_charges,
            ),
            lineage_parent_component_id=self.parent_standard_id,
            backbone_family_component_id=(self.parent_standard_id or self.component_id),
            preferred_atom_order=heavy_atom_names,
            heavy_atom_semantics=heavy_atom_semantics,
            hydrogen_semantics=hydrogen_semantics,
        )

    def to_idealized_component(self) -> IdealizedComponent:
        """Project the packaged record into the canonical ideal-geometry model."""

        return IdealizedComponent(
            component_id=self.component_id,
            lineage_parent_component_id=self.parent_standard_id,
            atoms=tuple(atom.to_idealized_atom() for atom in self.atoms),
            bonds=self.bonds,
        )

    def to_restraint_template(self) -> ResidueRestraintTemplate:
        """Project the packaged record into a canonical restraint template."""

        return ResidueRestraintTemplate(
            component_id=self.component_id,
            bond_targets=self.bond_targets,
            angle_targets=self.angle_targets,
            chirality_targets=self.chirality_targets,
            plane_targets=self.plane_targets,
        )


@dataclass(frozen=True, slots=True)
class BundledNonstandardRegistry:
    """Lazy-loaded bundled registry for curated nonstandard residue records."""

    records: Mapping[str, NonstandardComponentRecord]

    def __post_init__(self) -> None:
        normalized_records = {
            component_id.strip().upper(): record
            for component_id, record in self.records.items()
        }
        object.__setattr__(self, "records", MappingProxyType(normalized_records))

    def get(self, component_id: str) -> NonstandardComponentRecord | None:
        """Return one bundled component record if present."""

        normalized_component_id = component_id.strip().upper()
        return self.records.get(normalized_component_id)

    def component_library(self) -> ComponentLibrary:
        """Project bundled records into a canonical component library."""

        return ComponentLibrary(
            templates={
                component_id: record.to_template()
                for component_id, record in self.records.items()
            }
        )

    def restraint_library(self) -> RestraintLibrary:
        """Project bundled records into canonical restraint templates."""

        return RestraintLibrary(
            templates={
                component_id: record.to_restraint_template()
                for component_id, record in self.records.items()
            }
        )


def bundled_nonstandard_asset_path() -> str:
    """Return the packaged path for the bundled nonstandard asset."""

    return str(
        files("protrepair.chemistry.resources").joinpath("nonstandard_components.json.gz")
    )


@lru_cache(maxsize=1)
def build_bundled_nonstandard_registry() -> BundledNonstandardRegistry:
    """Return the packaged bundled registry for curated nonstandard residues."""

    asset_path = files("protrepair.chemistry.resources").joinpath(
        "nonstandard_components.json.gz"
    )
    with asset_path.open("rb") as handle:
        payload = gzip.decompress(handle.read()).decode("utf-8")

    raw_payload = cast(object, json.loads(payload))
    return parse_bundled_nonstandard_registry(raw_payload)


@lru_cache(maxsize=1)
def build_bundled_nonstandard_component_library() -> ComponentLibrary:
    """Return the canonical component-library projection of the bundled registry."""

    return build_bundled_nonstandard_registry().component_library()


@lru_cache(maxsize=1)
def build_bundled_nonstandard_restraint_library() -> RestraintLibrary:
    """Return the canonical restraint-library projection of the bundled registry."""

    return build_bundled_nonstandard_registry().restraint_library()


def parse_bundled_nonstandard_registry(
    payload: object,
) -> BundledNonstandardRegistry:
    """Parse one packaged nonstandard-registry payload."""

    payload_mapping = expect_mapping(payload, path="root")
    components = expect_sequence(payload_mapping.get("components"), path="components")
    records = {
        record.component_id: record
        for record in (
            parse_nonstandard_component_record(
                item,
                path=f"components[{index}]",
            )
            for index, item in enumerate(components)
        )
    }
    return BundledNonstandardRegistry(records=records)


def parse_nonstandard_component_record(
    payload: object,
    *,
    path: str,
) -> NonstandardComponentRecord:
    """Parse one packaged nonstandard component record."""

    payload_mapping = expect_mapping(payload, path=path)
    atoms = expect_sequence(payload_mapping.get("atoms"), path=f"{path}.atoms")
    bonds = expect_sequence(payload_mapping.get("bonds"), path=f"{path}.bonds")
    pcm_categories = expect_string_sequence(
        payload_mapping.get("pcm_categories", ()),
        path=f"{path}.pcm_categories",
    )
    pcm_types = expect_string_sequence(
        payload_mapping.get("pcm_types", ()),
        path=f"{path}.pcm_types",
    )
    bond_targets = expect_sequence(
        payload_mapping.get("bond_targets", ()),
        path=f"{path}.bond_targets",
    )
    angle_targets = expect_sequence(
        payload_mapping.get("angle_targets", ()),
        path=f"{path}.angle_targets",
    )
    chirality_targets = expect_sequence(
        payload_mapping.get("chirality_targets", ()),
        path=f"{path}.chirality_targets",
    )
    plane_targets = expect_sequence(
        payload_mapping.get("plane_targets", ()),
        path=f"{path}.plane_targets",
    )
    return NonstandardComponentRecord(
        component_id=expect_string(
            payload_mapping.get("component_id"),
            path=f"{path}.component_id",
        ),
        parent_standard_id=expect_optional_string(
            payload_mapping.get("parent_standard_id"),
            path=f"{path}.parent_standard_id",
        ),
        chem_comp_type=expect_string(
            payload_mapping.get("chem_comp_type"),
            path=f"{path}.chem_comp_type",
        ),
        name=expect_string(payload_mapping.get("name"), path=f"{path}.name"),
        one_letter_code=expect_optional_string(
            payload_mapping.get("one_letter_code"),
            path=f"{path}.one_letter_code",
        ),
        atoms=tuple(
            parse_nonstandard_component_atom(
                atom_payload,
                path=f"{path}.atoms[{index}]",
            )
            for index, atom_payload in enumerate(atoms)
        ),
        bonds=tuple(
            parse_nonstandard_component_bond(
                bond_payload,
                path=f"{path}.bonds[{index}]",
            )
            for index, bond_payload in enumerate(bonds)
        ),
        bond_targets=tuple(
            parse_nonstandard_bond_target(
                target_payload,
                path=f"{path}.bond_targets[{index}]",
            )
            for index, target_payload in enumerate(bond_targets)
        ),
        angle_targets=tuple(
            parse_nonstandard_angle_target(
                target_payload,
                path=f"{path}.angle_targets[{index}]",
            )
            for index, target_payload in enumerate(angle_targets)
        ),
        chirality_targets=tuple(
            parse_nonstandard_chirality_target(
                target_payload,
                path=f"{path}.chirality_targets[{index}]",
            )
            for index, target_payload in enumerate(chirality_targets)
        ),
        plane_targets=tuple(
            parse_nonstandard_plane_target(
                target_payload,
                path=f"{path}.plane_targets[{index}]",
            )
            for index, target_payload in enumerate(plane_targets)
        ),
        pcm_categories=pcm_categories,
        pcm_types=pcm_types,
    )


def parse_nonstandard_component_atom(
    payload: object,
    *,
    path: str,
) -> NonstandardComponentAtom:
    """Parse one packaged nonstandard atom record."""

    payload_mapping = expect_mapping(payload, path=path)
    return NonstandardComponentAtom(
        atom_name=expect_string(
            payload_mapping.get("atom_name"),
            path=f"{path}.atom_name",
        ),
        element=expect_string(
            payload_mapping.get("element"),
            path=f"{path}.element",
        ),
        formal_charge=expect_int(
            payload_mapping.get("formal_charge", 0),
            path=f"{path}.formal_charge",
        ),
        stereo_config=expect_optional_string(
            payload_mapping.get("stereo_config"),
            path=f"{path}.stereo_config",
        ),
        ideal_position=expect_optional_position(
            payload_mapping.get("ideal_position"),
            path=f"{path}.ideal_position",
        ),
    )


def parse_nonstandard_component_bond(
    payload: object,
    *,
    path: str,
) -> BondDefinition:
    """Parse one packaged nonstandard bond record."""

    payload_mapping = expect_mapping(payload, path=path)
    return BondDefinition(
        atom_name_1=expect_string(
            payload_mapping.get("atom_name_1"),
            path=f"{path}.atom_name_1",
        ),
        atom_name_2=expect_string(
            payload_mapping.get("atom_name_2"),
            path=f"{path}.atom_name_2",
        ),
        order=expect_int(payload_mapping.get("order", 1), path=f"{path}.order"),
        aromatic=expect_bool(
            payload_mapping.get("aromatic", False),
            path=f"{path}.aromatic",
        ),
    )


def parse_nonstandard_bond_target(
    payload: object,
    *,
    path: str,
) -> BondRestraintTarget:
    """Parse one packaged nonstandard bond-restraint target."""

    payload_mapping = expect_mapping(payload, path=path)
    return BondRestraintTarget(
        atom_name_1=expect_string(
            payload_mapping.get("atom_name_1"),
            path=f"{path}.atom_name_1",
        ),
        atom_name_2=expect_string(
            payload_mapping.get("atom_name_2"),
            path=f"{path}.atom_name_2",
        ),
        target_distance_angstrom=expect_float(
            payload_mapping.get("target_distance_angstrom"),
            path=f"{path}.target_distance_angstrom",
        ),
        esd_angstrom=expect_optional_float(
            payload_mapping.get("esd_angstrom"),
            path=f"{path}.esd_angstrom",
        ),
    )


def parse_nonstandard_angle_target(
    payload: object,
    *,
    path: str,
) -> AngleRestraintTarget:
    """Parse one packaged nonstandard angle-restraint target."""

    payload_mapping = expect_mapping(payload, path=path)
    return AngleRestraintTarget(
        atom_name_1=expect_string(
            payload_mapping.get("atom_name_1"),
            path=f"{path}.atom_name_1",
        ),
        center_atom_name=expect_string(
            payload_mapping.get("center_atom_name"),
            path=f"{path}.center_atom_name",
        ),
        atom_name_2=expect_string(
            payload_mapping.get("atom_name_2"),
            path=f"{path}.atom_name_2",
        ),
        target_angle_degrees=expect_float(
            payload_mapping.get("target_angle_degrees"),
            path=f"{path}.target_angle_degrees",
        ),
        esd_degrees=expect_optional_float(
            payload_mapping.get("esd_degrees"),
            path=f"{path}.esd_degrees",
        ),
    )


def parse_nonstandard_chirality_target(
    payload: object,
    *,
    path: str,
) -> ChiralityRestraintTarget:
    """Parse one packaged nonstandard chirality-restraint target."""

    payload_mapping = expect_mapping(payload, path=path)
    return ChiralityRestraintTarget(
        center_atom_name=expect_string(
            payload_mapping.get("center_atom_name"),
            path=f"{path}.center_atom_name",
        ),
        ordered_neighbor_atom_names=cast(
            tuple[str, str, str],
            expect_string_sequence(
                payload_mapping.get("ordered_neighbor_atom_names"),
                path=f"{path}.ordered_neighbor_atom_names",
            ),
        ),
        expected_orientation_sign=expect_int(
            payload_mapping.get("expected_orientation_sign"),
            path=f"{path}.expected_orientation_sign",
        ),
    )


def parse_nonstandard_plane_target(
    payload: object,
    *,
    path: str,
) -> PlaneRestraintTarget:
    """Parse one packaged nonstandard plane-restraint target."""

    payload_mapping = expect_mapping(payload, path=path)
    return PlaneRestraintTarget(
        atom_names=expect_string_sequence(
            payload_mapping.get("atom_names"),
            path=f"{path}.atom_names",
        )
    )


def expect_mapping(value: object, *, path: str) -> Mapping[str, object]:
    """Return one mapping payload or raise with a precise path."""

    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")

    normalized_mapping: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} keys must be strings")
        normalized_mapping[key] = item

    return normalized_mapping


def expect_sequence(value: object, *, path: str) -> Sequence[object]:
    """Return one sequence payload or raise with a precise path."""

    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")

    return value


def expect_string(value: object, *, path: str) -> str:
    """Return one required string payload or raise."""

    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{path} must not be blank")

    return normalized_value


def expect_optional_string(value: object, *, path: str) -> str | None:
    """Return one optional string payload or ``None``."""

    if value is None:
        return None

    normalized_value = expect_string(value, path=path)
    return normalized_value


def expect_string_sequence(value: object, *, path: str) -> tuple[str, ...]:
    """Return one sequence of strings or raise."""

    sequence = expect_sequence(value, path=path)
    return tuple(
        expect_string(item, path=f"{path}[{index}]")
        for index, item in enumerate(sequence)
    )


def expect_int(value: object, *, path: str) -> int:
    """Return one integer payload or raise."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")

    return value


def expect_float(value: object, *, path: str) -> float:
    """Return one floating-point payload or raise."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a float")

    return float(value)


def expect_optional_float(value: object, *, path: str) -> float | None:
    """Return one optional float payload or ``None``."""

    if value is None:
        return None

    return expect_float(value, path=path)


def expect_bool(value: object, *, path: str) -> bool:
    """Return one boolean payload or raise."""

    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")

    return value


def expect_optional_position(value: object, *, path: str) -> IdealPosition | None:
    """Return one optional Cartesian coordinate triplet or ``None``."""

    if value is None:
        return None

    sequence = expect_sequence(value, path=path)
    if len(sequence) != 3:
        raise ValueError(f"{path} must contain exactly three coordinates")

    return (
        expect_float(sequence[0], path=f"{path}[0]"),
        expect_float(sequence[1], path=f"{path}[1]"),
        expect_float(sequence[2], path=f"{path}[2]"),
    )
