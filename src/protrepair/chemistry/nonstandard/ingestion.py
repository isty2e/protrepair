"""External nonstandard component ingestion from CCD or monomer-CIF assets."""

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from math import acos, degrees, sqrt
from pathlib import Path
from typing import cast

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.nonstandard.registry import (
    NonstandardComponentAtom,
    NonstandardComponentRecord,
)
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ChiralityRestraintTarget,
    PlaneRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.io.gemmi_normalization import gemmi

BOND_ORDER_LABEL_TO_INT: dict[str, int] = {
    "AROM": 1,
    "SING": 1,
    "DOUB": 2,
    "TRIP": 3,
}
IDEAL_PLANE_MAX_DEVIATION_ANGSTROM = 0.15
SIGNED_VOLUME_EPSILON = 1.0e-6
Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _IdealAtomGeometry:
    """Idealized CCD geometry for one atom when present."""

    atom_name: str
    stereo_config: str | None
    position: Vector3 | None


def _build_component_atom_record(
    *,
    atom_name: str,
    element: str,
    charge: str | None,
    ideal_geometry_by_atom_name: Mapping[str, _IdealAtomGeometry],
) -> NonstandardComponentAtom:
    """Build one canonical atom record with optional ideal geometry."""

    normalized_atom_name = atom_name.strip().upper()
    atom_geometry = ideal_geometry_by_atom_name.get(normalized_atom_name)
    return NonstandardComponentAtom(
        atom_name=atom_name,
        element=element,
        formal_charge=int((charge or "0").strip() or "0"),
        stereo_config=None if atom_geometry is None else atom_geometry.stereo_config,
        ideal_position=None if atom_geometry is None else atom_geometry.position,
    )


def ingest_component_library(path: Path) -> ComponentLibrary:
    """Ingest one CCD or monomer-CIF asset into a canonical component library."""

    component_library, _ = _ingest_external_assets(path)
    return component_library


def ingest_restraint_library(path: Path) -> RestraintLibrary:
    """Ingest one CCD or monomer-CIF asset into a canonical restraint library."""

    _, restraint_library = _ingest_external_assets(path)
    return restraint_library


def ingest_component_template(block) -> ResidueTemplate | None:
    """Ingest one CIF block into a canonical residue template."""

    parsed_record = _parse_component_record(block)
    if parsed_record is None:
        return None

    return parsed_record.to_template()


def ingest_restraint_template(block) -> ResidueRestraintTemplate | None:
    """Ingest one CIF block into a canonical restraint template."""

    parsed_record = _parse_component_record(block)
    if parsed_record is None:
        return None

    return parsed_record.to_restraint_template()


def _read_cif_document_blocks(path: Path):
    """Return gemmi CIF blocks from one external component asset."""

    return tuple(gemmi.cif.read_file(str(path)))


def _ingest_external_assets(
    path: Path,
) -> tuple[ComponentLibrary, RestraintLibrary]:
    """Ingest one external asset into canonical component and restraint libraries."""

    parsed_records = tuple(
        parsed_record
        for parsed_record in (
            _parse_component_record(block) for block in _read_cif_document_blocks(path)
        )
        if parsed_record is not None
    )
    component_library = ComponentLibrary(
        templates={
            parsed_record.component_id: parsed_record.to_template()
            for parsed_record in parsed_records
        }
    )
    restraint_library = RestraintLibrary(
        templates={
            parsed_record.component_id: parsed_record.to_restraint_template()
            for parsed_record in parsed_records
        },
        alias_to_component_id=component_library.alias_to_component_id,
    )
    return component_library, restraint_library


def _parse_component_record(block) -> NonstandardComponentRecord | None:
    """Parse one gemmi CIF block into one source-local component record."""

    component_id = clean_value(block.find_value("_chem_comp.id"))
    if component_id is None:
        return None

    atom_rows = list(
        block.find(
            [
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
            ]
        )
    )
    if not atom_rows:
        return None

    ideal_geometry_by_atom_name = parse_ideal_geometry(block)
    bond_rows = list(
        block.find(
            [
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
            ]
        )
    )
    atoms = tuple(
        _build_component_atom_record(
            atom_name=atom_name,
            element=element,
            charge=charge,
            ideal_geometry_by_atom_name=ideal_geometry_by_atom_name,
        )
        for atom_name, element, charge in atom_rows
    )
    atom_order_index = {atom.atom_name: index for index, atom in enumerate(atoms)}
    bonds = tuple(
        parse_bond_row(
            atom_name_1=atom_name_1,
            atom_name_2=atom_name_2,
            order_label=order_label,
            aromatic_flag=aromatic_flag,
            component_id=component_id,
        )
        for atom_name_1, atom_name_2, order_label, aromatic_flag in bond_rows
    )
    bond_targets = derive_bond_targets(
        atoms=atoms,
        bonds=bonds,
        ideal_geometry_by_atom_name=ideal_geometry_by_atom_name,
    )
    angle_targets = derive_angle_targets(
        atoms=atoms,
        bonds=bonds,
        ideal_geometry_by_atom_name=ideal_geometry_by_atom_name,
        atom_order_index=atom_order_index,
    )
    chirality_targets = derive_chirality_targets(
        atoms=atoms,
        bonds=bonds,
        ideal_geometry_by_atom_name=ideal_geometry_by_atom_name,
        atom_order_index=atom_order_index,
    )
    plane_targets = derive_plane_targets(
        atoms=atoms,
        bonds=bonds,
        ideal_geometry_by_atom_name=ideal_geometry_by_atom_name,
        atom_order_index=atom_order_index,
    )
    return NonstandardComponentRecord(
        component_id=component_id,
        parent_standard_id=parse_parent_standard_id(
            block.find_value("_chem_comp.mon_nstd_parent_comp_id")
        ),
        chem_comp_type=clean_value(block.find_value("_chem_comp.type")) or "UNKNOWN",
        name=clean_value(block.find_value("_chem_comp.name")) or component_id,
        atoms=atoms,
        bonds=bonds,
        bond_targets=bond_targets,
        angle_targets=angle_targets,
        chirality_targets=chirality_targets,
        plane_targets=plane_targets,
        one_letter_code=clean_value(block.find_value("_chem_comp.one_letter_code")),
        pcm_categories=tuple(
            cleaned_value
            for cleaned_value in (
                clean_value(row[0])
                for row in block.find(["_pdbx_chem_comp_pcm.category"])
            )
            if cleaned_value is not None
        ),
        pcm_types=tuple(
            cleaned_value
            for cleaned_value in (
                clean_value(row[0]) for row in block.find(["_pdbx_chem_comp_pcm.type"])
            )
            if cleaned_value is not None
        ),
    )


def parse_ideal_geometry(block) -> Mapping[str, _IdealAtomGeometry]:
    """Return idealized atom geometry keyed by atom name."""

    rows = list(
        block.find(
            [
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.pdbx_stereo_config",
                "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
            ]
        )
    )
    ideal_geometry: dict[str, _IdealAtomGeometry] = {}
    for atom_name, stereo_config, x_coord, y_coord, z_coord in rows:
        normalized_atom_name = clean_value(atom_name)
        if normalized_atom_name is None:
            continue

        ideal_geometry[normalized_atom_name] = _IdealAtomGeometry(
            atom_name=normalized_atom_name,
            stereo_config=parse_stereo_config(stereo_config),
            position=parse_ideal_position(x_coord, y_coord, z_coord),
        )

    return ideal_geometry


def derive_bond_targets(
    *,
    atoms: tuple[NonstandardComponentAtom, ...],
    bonds: tuple[BondDefinition, ...],
    ideal_geometry_by_atom_name: Mapping[str, _IdealAtomGeometry],
) -> tuple[BondRestraintTarget, ...]:
    """Derive heavy-atom bond-length targets from ideal CCD coordinates."""

    heavy_atom_names = {atom.atom_name for atom in atoms if not atom.is_hydrogen()}
    targets: list[BondRestraintTarget] = []
    for bond in bonds:
        if (
            bond.atom_name_1 not in heavy_atom_names
            or bond.atom_name_2 not in heavy_atom_names
        ):
            continue

        left_position = atom_position(
            ideal_geometry_by_atom_name,
            atom_name=bond.atom_name_1,
        )
        right_position = atom_position(
            ideal_geometry_by_atom_name,
            atom_name=bond.atom_name_2,
        )
        if left_position is None or right_position is None:
            continue

        targets.append(
            BondRestraintTarget(
                atom_name_1=bond.atom_name_1,
                atom_name_2=bond.atom_name_2,
                target_distance_angstrom=distance(left_position, right_position),
            )
        )

    return tuple(targets)


def derive_angle_targets(
    *,
    atoms: tuple[NonstandardComponentAtom, ...],
    bonds: tuple[BondDefinition, ...],
    ideal_geometry_by_atom_name: Mapping[str, _IdealAtomGeometry],
    atom_order_index: Mapping[str, int],
) -> tuple[AngleRestraintTarget, ...]:
    """Derive heavy-atom bond-angle targets from ideal CCD coordinates."""

    heavy_atom_names = {atom.atom_name for atom in atoms if not atom.is_hydrogen()}
    heavy_neighbor_map = build_heavy_neighbor_map(
        bonds=bonds,
        heavy_atom_names=heavy_atom_names,
    )
    targets: list[AngleRestraintTarget] = []
    for center_atom_name, neighbor_atom_names in heavy_neighbor_map.items():
        ordered_neighbors = tuple(
            sorted(
                neighbor_atom_names,
                key=lambda atom_name: atom_order_index[atom_name],
            )
        )
        for atom_name_1, atom_name_2 in combinations(ordered_neighbors, 2):
            first_position = atom_position(
                ideal_geometry_by_atom_name,
                atom_name=atom_name_1,
            )
            center_position = atom_position(
                ideal_geometry_by_atom_name,
                atom_name=center_atom_name,
            )
            second_position = atom_position(
                ideal_geometry_by_atom_name,
                atom_name=atom_name_2,
            )
            if (
                first_position is None
                or center_position is None
                or second_position is None
            ):
                continue

            angle_degrees = angle_between(
                center_position=center_position,
                first_position=first_position,
                second_position=second_position,
            )
            if angle_degrees is None:
                continue

            targets.append(
                AngleRestraintTarget(
                    atom_name_1=atom_name_1,
                    center_atom_name=center_atom_name,
                    atom_name_2=atom_name_2,
                    target_angle_degrees=angle_degrees,
                )
            )

    return tuple(targets)


def derive_chirality_targets(
    *,
    atoms: tuple[NonstandardComponentAtom, ...],
    bonds: tuple[BondDefinition, ...],
    ideal_geometry_by_atom_name: Mapping[str, _IdealAtomGeometry],
    atom_order_index: Mapping[str, int],
) -> tuple[ChiralityRestraintTarget, ...]:
    """Derive tetrahedral chirality targets from ideal CCD coordinates."""

    heavy_atom_names = {atom.atom_name for atom in atoms if not atom.is_hydrogen()}
    heavy_neighbor_map = build_heavy_neighbor_map(
        bonds=bonds,
        heavy_atom_names=heavy_atom_names,
    )
    targets: list[ChiralityRestraintTarget] = []
    for center_atom_name in sorted(
        heavy_atom_names,
        key=lambda name: atom_order_index[name],
    ):
        atom_geometry = ideal_geometry_by_atom_name.get(center_atom_name)
        if atom_geometry is None or atom_geometry.position is None:
            continue

        if atom_geometry.stereo_config not in {"R", "S"}:
            continue

        neighbor_atom_names = heavy_neighbor_map.get(center_atom_name, frozenset())
        if len(neighbor_atom_names) != 3:
            continue

        ordered_neighbor_atom_names = cast(
            tuple[str, str, str],
            tuple(
                sorted(
                    neighbor_atom_names,
                    key=lambda atom_name: atom_order_index[atom_name],
                )
            ),
        )
        neighbor_positions = tuple(
            atom_position(ideal_geometry_by_atom_name, atom_name=atom_name)
            for atom_name in ordered_neighbor_atom_names
        )
        if any(position is None for position in neighbor_positions):
            continue

        resolved_neighbor_positions = cast(
            tuple[Vector3, Vector3, Vector3],
            tuple(position for position in neighbor_positions if position is not None),
        )
        signed_volume = tetrahedral_signed_volume(
            center_position=atom_geometry.position,
            ordered_neighbor_positions=resolved_neighbor_positions,
        )
        if abs(signed_volume) <= SIGNED_VOLUME_EPSILON:
            continue

        targets.append(
            ChiralityRestraintTarget(
                center_atom_name=center_atom_name,
                ordered_neighbor_atom_names=ordered_neighbor_atom_names,
                expected_orientation_sign=1 if signed_volume > 0.0 else -1,
            )
        )

    return tuple(targets)


def derive_plane_targets(
    *,
    atoms: tuple[NonstandardComponentAtom, ...],
    bonds: tuple[BondDefinition, ...],
    ideal_geometry_by_atom_name: Mapping[str, _IdealAtomGeometry],
    atom_order_index: Mapping[str, int],
) -> tuple[PlaneRestraintTarget, ...]:
    """Derive planar conjugated heavy-atom sets from ideal CCD coordinates."""

    heavy_atom_names = {atom.atom_name for atom in atoms if not atom.is_hydrogen()}
    planar_graph: dict[str, set[str]] = defaultdict(set)
    for bond in bonds:
        if (
            bond.atom_name_1 not in heavy_atom_names
            or bond.atom_name_2 not in heavy_atom_names
        ):
            continue

        if not (bond.aromatic or bond.order >= 2):
            continue

        planar_graph[bond.atom_name_1].add(bond.atom_name_2)
        planar_graph[bond.atom_name_2].add(bond.atom_name_1)

    components = connected_components(planar_graph)
    plane_targets: list[PlaneRestraintTarget] = []
    for component_atom_names in components:
        if len(component_atom_names) < 4:
            continue

        ordered_atom_names = tuple(
            sorted(
                component_atom_names,
                key=lambda atom_name: atom_order_index[atom_name],
            )
        )
        positions = tuple(
            atom_position(ideal_geometry_by_atom_name, atom_name=atom_name)
            for atom_name in ordered_atom_names
        )
        if any(position is None for position in positions):
            continue

        ideal_positions = tuple(
            position for position in positions if position is not None
        )
        normal = first_non_degenerate_plane_normal(ideal_positions)
        if normal is None:
            continue

        anchor_position = ideal_positions[0]
        maximum_distance = max(
            point_plane_distance(
                point_position=position,
                anchor_position=anchor_position,
                plane_normal=normal,
            )
            for position in ideal_positions[1:]
        )
        if maximum_distance > IDEAL_PLANE_MAX_DEVIATION_ANGSTROM:
            continue

        plane_targets.append(PlaneRestraintTarget(atom_names=ordered_atom_names))

    return tuple(plane_targets)


def build_heavy_neighbor_map(
    *,
    bonds: tuple[BondDefinition, ...],
    heavy_atom_names: set[str],
) -> Mapping[str, frozenset[str]]:
    """Return heavy-atom neighbor sets derived from one bond list."""

    neighbor_map: dict[str, set[str]] = defaultdict(set)
    for bond in bonds:
        if (
            bond.atom_name_1 not in heavy_atom_names
            or bond.atom_name_2 not in heavy_atom_names
        ):
            continue

        neighbor_map[bond.atom_name_1].add(bond.atom_name_2)
        neighbor_map[bond.atom_name_2].add(bond.atom_name_1)

    return {
        atom_name: frozenset(neighbor_atom_names)
        for atom_name, neighbor_atom_names in neighbor_map.items()
    }


def connected_components(graph: Mapping[str, set[str]]) -> tuple[frozenset[str], ...]:
    """Return connected components for one undirected atom graph."""

    remaining_atom_names = set(graph)
    components: list[frozenset[str]] = []
    while remaining_atom_names:
        seed_atom_name = remaining_atom_names.pop()
        component_atom_names = {seed_atom_name}
        frontier = deque([seed_atom_name])
        while frontier:
            current_atom_name = frontier.popleft()
            for neighbor_atom_name in graph[current_atom_name]:
                if neighbor_atom_name in component_atom_names:
                    continue

                component_atom_names.add(neighbor_atom_name)
                if neighbor_atom_name in remaining_atom_names:
                    remaining_atom_names.remove(neighbor_atom_name)
                frontier.append(neighbor_atom_name)

        components.append(frozenset(component_atom_names))

    return tuple(components)


def atom_position(
    ideal_geometry_by_atom_name: Mapping[str, _IdealAtomGeometry],
    *,
    atom_name: str,
) -> Vector3 | None:
    """Return one ideal atom position when available."""

    atom_geometry = ideal_geometry_by_atom_name.get(atom_name)
    if atom_geometry is None:
        return None

    return atom_geometry.position


def parse_bond_row(
    *,
    atom_name_1: str,
    atom_name_2: str,
    order_label: str,
    aromatic_flag: str,
    component_id: str,
) -> BondDefinition:
    """Normalize one raw bond row into a canonical bond definition."""

    normalized_order_label = order_label.strip().upper()
    order = BOND_ORDER_LABEL_TO_INT.get(normalized_order_label)
    if order is None:
        raise ValueError(
            f"unsupported bond order {normalized_order_label!r} for {component_id}"
        )

    return BondDefinition(
        atom_name_1=atom_name_1,
        atom_name_2=atom_name_2,
        order=order,
        aromatic=aromatic_flag.strip().upper() == "Y",
    )


def parse_parent_standard_id(raw_parent_standard_id: str | None) -> str | None:
    """Normalize an optional nonstandard-to-standard parent mapping."""

    cleaned_parent_standard_id = clean_value(raw_parent_standard_id)
    if cleaned_parent_standard_id is None:
        return None

    separators_normalized = cleaned_parent_standard_id.replace(";", ",")
    parent_standard_ids = [
        token.strip().upper()
        for token in separators_normalized.split(",")
        if token.strip()
    ]
    if len(parent_standard_ids) != 1:
        return None

    return parent_standard_ids[0]


def parse_stereo_config(raw_stereo_config: str | None) -> str | None:
    """Normalize one optional CCD stereo configuration label."""

    cleaned_stereo_config = clean_value(raw_stereo_config)
    if cleaned_stereo_config is None:
        return None

    normalized_stereo_config = cleaned_stereo_config.upper()
    if normalized_stereo_config in {"R", "S"}:
        return normalized_stereo_config

    return None


def parse_ideal_position(
    x_coord: str | None,
    y_coord: str | None,
    z_coord: str | None,
) -> Vector3 | None:
    """Parse one idealized Cartesian position when all coordinates are present."""

    x_value = clean_value(x_coord)
    y_value = clean_value(y_coord)
    z_value = clean_value(z_coord)
    if x_value is None or y_value is None or z_value is None:
        return None

    return (float(x_value), float(y_value), float(z_value))


def clean_value(value: str | None) -> str | None:
    """Normalize a scalar CIF value into a nullable trimmed string."""

    if value is None:
        return None

    normalized_value = value.strip().strip('"').strip("'").strip()
    if normalized_value in {"", ".", "?"}:
        return None

    return normalized_value


def distance(left_position: Vector3, right_position: Vector3) -> float:
    """Return Euclidean distance between two points."""

    return vector_norm(subtract(right_position, left_position))


def angle_between(
    *,
    center_position: Vector3,
    first_position: Vector3,
    second_position: Vector3,
) -> float | None:
    """Return one angle in degrees for a residue-local atom triplet."""

    first_vector = subtract(first_position, center_position)
    second_vector = subtract(second_position, center_position)
    first_norm = vector_norm(first_vector)
    second_norm = vector_norm(second_vector)
    if first_norm <= 0.0 or second_norm <= 0.0:
        return None

    cosine = dot(first_vector, second_vector) / (first_norm * second_norm)
    clamped_cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(clamped_cosine))


def tetrahedral_signed_volume(
    *,
    center_position: Vector3,
    ordered_neighbor_positions: tuple[Vector3, Vector3, Vector3],
) -> float:
    """Return signed tetrahedral volume for one center and three neighbors."""

    vector_1 = subtract(ordered_neighbor_positions[0], center_position)
    vector_2 = subtract(ordered_neighbor_positions[1], center_position)
    vector_3 = subtract(ordered_neighbor_positions[2], center_position)
    cross_product = cross(vector_1, vector_2)
    return dot(cross_product, vector_3)


def first_non_degenerate_plane_normal(
    positions: tuple[Vector3, ...],
) -> Vector3 | None:
    """Return one normalized plane normal for a non-degenerate point set."""

    anchor_position = positions[0]
    for first_position, second_position in combinations(positions[1:], 2):
        first_vector = subtract(first_position, anchor_position)
        second_vector = subtract(second_position, anchor_position)
        normal = cross(first_vector, second_vector)
        normal_norm = vector_norm(normal)
        if normal_norm <= SIGNED_VOLUME_EPSILON:
            continue

        return scale(normal, 1.0 / normal_norm)

    return None


def point_plane_distance(
    *,
    point_position: Vector3,
    anchor_position: Vector3,
    plane_normal: Vector3,
) -> float:
    """Return absolute point-to-plane distance for one normalized plane."""

    return abs(dot(subtract(point_position, anchor_position), plane_normal))


def subtract(left_vector: Vector3, right_vector: Vector3) -> Vector3:
    """Return one vector subtraction."""

    return (
        left_vector[0] - right_vector[0],
        left_vector[1] - right_vector[1],
        left_vector[2] - right_vector[2],
    )


def dot(left_vector: Vector3, right_vector: Vector3) -> float:
    """Return one vector dot product."""

    return (
        left_vector[0] * right_vector[0]
        + left_vector[1] * right_vector[1]
        + left_vector[2] * right_vector[2]
    )


def cross(left_vector: Vector3, right_vector: Vector3) -> Vector3:
    """Return one vector cross product."""

    return (
        left_vector[1] * right_vector[2] - left_vector[2] * right_vector[1],
        left_vector[2] * right_vector[0] - left_vector[0] * right_vector[2],
        left_vector[0] * right_vector[1] - left_vector[1] * right_vector[0],
    )


def scale(vector: Vector3, multiplier: float) -> Vector3:
    """Return one vector scaled by a scalar multiplier."""

    return (
        vector[0] * multiplier,
        vector[1] * multiplier,
        vector[2] * multiplier,
    )


def vector_norm(vector: Vector3) -> float:
    """Return Euclidean norm of one vector."""

    return sqrt(dot(vector, vector))
