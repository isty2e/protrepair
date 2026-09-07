"""Anchor-constrained reconstruction of missing polymer residue spans."""

from dataclasses import dataclass
from enum import Enum
from math import atan2, isfinite
from numbers import Real

import numpy as np
import numpy.typing as npt

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.peptide_geometry import PeptideJunctionGeometry
from protrepair.geometry import Vec3
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import is_covalent_like_relationship
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]

ROTATION_AXIS_NORM_EPSILON = 1.0e-10


class SpanReconstructionFailureKind(str, Enum):
    """Closed failure families for donor-backed span reconstruction."""

    MISSING_BACKBONE_CONTEXT = "missing_backbone_context"
    INVALID_TARGET_STATE = "invalid_target_state"
    NON_FINITE_COORDINATES = "non_finite_coordinates"
    DEGENERATE_ANCHOR_FRAME = "degenerate_anchor_frame"
    NON_CONVERGENT_CLOSURE = "non_convergent_closure"
    INVALID_PEPTIDE_JUNCTION = "invalid_peptide_junction"
    UNSUPPORTED_COMPONENT_CHEMISTRY = "unsupported_component_chemistry"
    INVALID_INTRINSIC_GEOMETRY = "invalid_intrinsic_geometry"
    INVALID_STEREOCHEMISTRY = "invalid_stereochemistry"


@dataclass(frozen=True, slots=True)
class SpanClosureSettings:
    """Deterministic numerical bounds for donor-seeded CCD closure.

    Parameters
    ----------
    maximum_iterations : int
        Maximum complete CCD sweeps per axis order. Up to four deterministic
        orders are tried from the same seed, each with this full limit.
    endpoint_rmsd_tolerance_angstrom : float
        Maximum RMSD between the moving and source endpoint triads.
    convergence_delta_angstrom : float
        Minimum sweep-to-sweep RMSD change treated as continued progress.

    Raises
    ------
    TypeError
        If an input is boolean or not numeric.
    ValueError
        If a numerical bound is non-finite or non-positive.
    """

    maximum_iterations: int = 500
    endpoint_rmsd_tolerance_angstrom: float = 0.1
    convergence_delta_angstrom: float = 1.0e-12

    def __post_init__(self) -> None:
        if isinstance(self.maximum_iterations, bool) or not isinstance(
            self.maximum_iterations, int
        ):
            raise TypeError("span closure maximum_iterations must be an integer")
        if self.maximum_iterations <= 0:
            raise ValueError("span closure maximum_iterations must be positive")
        for field_name, value in (
            (
                "endpoint_rmsd_tolerance_angstrom",
                self.endpoint_rmsd_tolerance_angstrom,
            ),
            ("convergence_delta_angstrom", self.convergence_delta_angstrom),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"span closure {field_name} must be numeric")
            normalized_value = float(value)
            if not isfinite(normalized_value) or normalized_value <= 0.0:
                raise ValueError(
                    f"span closure {field_name} must be finite and positive"
                )
            object.__setattr__(self, field_name, normalized_value)


@dataclass(frozen=True, slots=True)
class ReconstructedSpanCandidate:
    """One closed heavy-atom span candidate ready for atomic materialization.

    Parameters
    ----------
    residue_payloads : tuple[CompletionResiduePayload, ...]
        Ordered heavy-atom facets for the missing source residues.
    endpoint_rmsd_angstrom : float | None
        Final endpoint-triad RMSD, or ``None`` when only one anchor exists.
    iteration_count : int
        Number of complete CCD sweeps used for closure.

    Raises
    ------
    TypeError
        If a non-null RMSD or iteration value has the wrong type.
    ValueError
        If payloads are empty or a numerical result is invalid.
    """

    residue_payloads: tuple[CompletionResiduePayload, ...]
    endpoint_rmsd_angstrom: float | None
    iteration_count: int

    def __post_init__(self) -> None:
        residue_payloads = tuple(self.residue_payloads)
        if not residue_payloads:
            raise ValueError("reconstructed span candidates require residue payloads")
        if any(
            not isinstance(payload, CompletionResiduePayload)
            for payload in residue_payloads
        ):
            raise TypeError(
                "reconstructed span candidates require CompletionResiduePayload values"
            )
        endpoint_rmsd_angstrom = self.endpoint_rmsd_angstrom
        if endpoint_rmsd_angstrom is not None:
            if isinstance(endpoint_rmsd_angstrom, bool) or not isinstance(
                endpoint_rmsd_angstrom,
                Real,
            ):
                raise TypeError("reconstructed span endpoint RMSD must be numeric")
            endpoint_rmsd_angstrom = float(endpoint_rmsd_angstrom)
            if not isfinite(endpoint_rmsd_angstrom):
                raise ValueError("reconstructed span endpoint RMSD must be finite")
            if endpoint_rmsd_angstrom < 0.0:
                raise ValueError(
                    "reconstructed span endpoint RMSD must be non-negative"
                )
        if isinstance(self.iteration_count, bool) or not isinstance(
            self.iteration_count,
            int,
        ):
            raise TypeError("reconstructed span iteration count must be an integer")
        if self.iteration_count < 0:
            raise ValueError("reconstructed span iteration count must be non-negative")

        object.__setattr__(self, "residue_payloads", residue_payloads)
        object.__setattr__(self, "endpoint_rmsd_angstrom", endpoint_rmsd_angstrom)


@dataclass(frozen=True, slots=True)
class SpanReconstructionFailure:
    """Typed evidence that one requested span could not be reconstructed safely.

    Parameters
    ----------
    kind : SpanReconstructionFailureKind
        Stable failure family for internal workflow decisions.
    message : str
        Non-empty operator-facing explanation.

    Raises
    ------
    TypeError
        If either field has the wrong type.
    ValueError
        If ``message`` is blank.
    """

    kind: SpanReconstructionFailureKind
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SpanReconstructionFailureKind):
            raise TypeError(
                "span reconstruction failures require a SpanReconstructionFailureKind"
            )
        if not isinstance(self.message, str):
            raise TypeError("span reconstruction failure messages must be strings")
        message = self.message.strip()
        if not message:
            raise ValueError("span reconstruction failure messages must not be blank")
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class _RotationAxis:
    start_index: int
    end_index: int
    moved_indices: IndexArray


@dataclass(frozen=True, slots=True)
class _DonorWindow:
    residue_payloads: tuple[CompletionResiduePayload, ...]
    coordinates: FloatArray
    atom_index_by_residue_and_name: dict[tuple[int, str], int]
    atom_indices_by_residue: tuple[IndexArray, ...]
    bonded_indices: tuple[frozenset[int], ...]
    rigid_bond_pairs: frozenset[tuple[int, int]]

    def atom_index(self, residue_offset: int, atom_name: str) -> int:
        return self.atom_index_by_residue_and_name[
            (residue_offset, atom_name.strip().upper())
        ]

    def positions(
        self,
        residue_offset: int,
        atom_names: tuple[str, ...],
    ) -> FloatArray:
        """Return selected atom positions from one donor residue."""

        return np.asarray(
            [
                self.coordinates[self.atom_index(residue_offset, atom_name)]
                for atom_name in atom_names
            ],
            dtype=np.float64,
        )

    def internal_span_rotation_axes(
        self,
        *,
        preceding_offset: int,
        first_inserted_offset: int,
        inserted_residue_count: int,
        following_offset: int,
    ) -> tuple[_RotationAxis, ...]:
        """Cut covalent bridges without moving the fixed preceding anchor."""

        fixed_indices = frozenset(
            int(index) for index in self.atom_indices_by_residue[preceding_offset]
        )
        axes: list[_RotationAxis] = []
        for residue_offset in range(
            first_inserted_offset, first_inserted_offset + inserted_residue_count
        ):
            for start_name, end_name in (("N", "CA"), ("CA", "C")):
                axis = self.bridge_rotation_axis(
                    self.atom_index(residue_offset, start_name),
                    self.atom_index(residue_offset, end_name),
                    fixed_indices=fixed_indices,
                )
                if axis is not None:
                    axes.append(axis)

        following_phi = self.bridge_rotation_axis(
            self.atom_index(following_offset, "N"),
            self.atom_index(following_offset, "CA"),
            fixed_indices=fixed_indices,
        )
        if following_phi is not None:
            axes.append(following_phi)
        return tuple(axes)

    def bridge_rotation_axis(
        self, start_index: int, end_index: int, *, fixed_indices: frozenset[int]
    ) -> _RotationAxis | None:
        """Return the moving bond component, or None for a ring or fixed cut."""

        if (
            end_index not in self.bonded_indices[start_index]
            or (min(start_index, end_index), max(start_index, end_index))
            in self.rigid_bond_pairs
        ):
            return None
        moving: set[int] = set()
        pending = [end_index]
        while pending:
            index = pending.pop()
            if index in moving:
                continue
            if index == start_index or index in fixed_indices:
                return None
            moving.add(index)
            pending.extend(
                neighbor
                for neighbor in self.bonded_indices[index]
                if not (index == end_index and neighbor == start_index)
                and neighbor not in moving
            )
        moving.discard(end_index)
        return _RotationAxis(
            start_index, end_index, np.asarray(sorted(moving), dtype=np.int64)
        )

    def materialize_inserted_payloads(
        self,
        *,
        working_coordinates: FloatArray,
        first_inserted_offset: int,
        target_residue_ids: tuple[ResidueId, ...],
    ) -> tuple[CompletionResiduePayload, ...]:
        """Return inserted residue facets addressed in the source structure."""

        payloads: list[CompletionResiduePayload] = []
        for insertion_index, target_residue_id in enumerate(target_residue_ids):
            residue_offset = first_inserted_offset + insertion_index
            donor_payload = self.residue_payloads[residue_offset]
            atom_geometries = {
                atom_site.name: AtomGeometry(
                    position=Vec3.from_iterable(
                        working_coordinates[
                            self.atom_index(residue_offset, atom_site.name)
                        ]
                    ),
                    occupancy=1.0,
                    b_factor=None,
                    altloc=None,
                )
                for atom_site in donor_payload.atom_sites
            }
            payloads.append(
                CompletionResiduePayload(
                    residue_site=donor_payload.residue_site.with_residue_id(
                        target_residue_id
                    ),
                    residue_geometry=ResidueGeometry(atoms_by_name=atom_geometries),
                    formal_charge_by_atom_name=(
                        donor_payload.formal_charge_by_atom_name
                    ),
                )
            )
        return tuple(payloads)


def reconstruct_donor_span(
    source_structure: ProteinStructure,
    *,
    scope: AbsentResidueSpanScope,
    donor_structure: ProteinStructure,
    donor_residue_ids: tuple[ResidueId, ...],
    donor_preceding_residue_id: ResidueId | None,
    donor_following_residue_id: ResidueId | None,
    settings: SpanClosureSettings | None = None,
    component_library: ComponentLibrary | None = None,
) -> ReconstructedSpanCandidate | SpanReconstructionFailure:
    """Return a closed donor-backed span or a typed atomic failure.

    Parameters
    ----------
    source_structure : ProteinStructure
        Canonical structure whose existing coordinates remain fixed.
    scope : AbsentResidueSpanScope
        Missing source residues and their available anchors.
    donor_structure : ProteinStructure
        Canonical structure supplying the residue conformations.
    donor_residue_ids : tuple[ResidueId, ...]
        Donor residues corresponding one-to-one with ``scope``.
    donor_preceding_residue_id : ResidueId | None
        Donor flank corresponding to the preceding source anchor.
    donor_following_residue_id : ResidueId | None
        Donor flank corresponding to the following source anchor.
    settings : SpanClosureSettings | None
        Optional numerical closure settings.
    component_library : ComponentLibrary | None
        Chemistry defining covalent motion constraints and nitrogen neighbors;
        defaults to the bundled library.

    Returns
    -------
    ReconstructedSpanCandidate | SpanReconstructionFailure
        A screened heavy-atom candidate, or failure evidence without a partial
        mutation.

    Raises
    ------
    TypeError
        If an argument does not satisfy its declared contract.
    ValueError
        If donor residue cardinality or donor flank availability disagrees with
        ``scope``.
    """

    if not isinstance(source_structure, ProteinStructure):
        raise TypeError("span reconstruction source must be a ProteinStructure")
    if not isinstance(scope, AbsentResidueSpanScope):
        raise TypeError("span reconstruction scope must be AbsentResidueSpanScope")
    if not isinstance(donor_structure, ProteinStructure):
        raise TypeError("span reconstruction donor must be a ProteinStructure")
    normalized_donor_residue_ids = tuple(donor_residue_ids)
    if any(
        not isinstance(residue_id, ResidueId)
        for residue_id in normalized_donor_residue_ids
    ):
        raise TypeError("span reconstruction donor ids must be ResidueId values")
    for field_name, residue_id in (
        ("donor_preceding_residue_id", donor_preceding_residue_id),
        ("donor_following_residue_id", donor_following_residue_id),
    ):
        if residue_id is not None and not isinstance(residue_id, ResidueId):
            raise TypeError(
                f"span reconstruction {field_name} must be ResidueId or None"
            )

    active_settings = SpanClosureSettings() if settings is None else settings
    if not isinstance(active_settings, SpanClosureSettings):
        raise TypeError("span reconstruction settings must be SpanClosureSettings")
    if len(normalized_donor_residue_ids) != len(scope.absent_residue_ids):
        raise ValueError(
            "span reconstruction requires one donor residue for each target residue"
        )
    if (scope.preceding_residue_id is None) != (donor_preceding_residue_id is None) or (
        scope.following_residue_id is None
    ) != (donor_following_residue_id is None):
        raise ValueError(
            "span reconstruction donor flanks must correspond to source anchors"
        )

    active_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    if not isinstance(active_library, ComponentLibrary):
        raise TypeError(
            "span reconstruction component_library must be ComponentLibrary"
        )

    donor_window_ids = tuple(
        residue_id
        for residue_id in (
            donor_preceding_residue_id,
            *normalized_donor_residue_ids,
            donor_following_residue_id,
        )
        if residue_id is not None
    )
    preceding_offset = 0 if donor_preceding_residue_id is not None else None
    first_inserted_offset = 1 if preceding_offset is not None else 0
    following_offset = (
        len(donor_window_ids) - 1 if donor_following_residue_id is not None else None
    )
    donor_window = _build_donor_window(
        donor_structure,
        donor_window_ids,
        preceding_offset=preceding_offset,
        following_offset=following_offset,
        component_library=active_library,
    )
    if isinstance(donor_window, SpanReconstructionFailure):
        return donor_window

    frame_result = _place_donor_window_in_source_frame(
        donor_window,
        source_structure=source_structure,
        scope=scope,
        preceding_offset=preceding_offset,
        following_offset=following_offset,
    )
    if isinstance(frame_result, SpanReconstructionFailure):
        return frame_result
    working_coordinates = frame_result

    iteration_count = 0
    endpoint_rmsd: float | None = None
    if preceding_offset is not None and following_offset is not None:
        endpoint_indices = np.asarray(
            [
                donor_window.atom_index(following_offset, atom_name)
                for atom_name in ("N", "CA", "C")
            ],
            dtype=np.int64,
        )
        following_source_residue_id = scope.following_residue_id
        assert following_source_residue_id is not None
        endpoint_targets = _backbone_positions(
            source_structure,
            following_source_residue_id,
            ("N", "CA", "C"),
        )
        if isinstance(endpoint_targets, SpanReconstructionFailure):
            return endpoint_targets

        rotation_axes = donor_window.internal_span_rotation_axes(
            preceding_offset=preceding_offset,
            first_inserted_offset=first_inserted_offset,
            inserted_residue_count=len(normalized_donor_residue_ids),
            following_offset=following_offset,
        )
        closure_result = _close_endpoint_by_cyclic_coordinate_descent(
            working_coordinates,
            rotation_axes=rotation_axes,
            endpoint_indices=endpoint_indices,
            endpoint_targets=endpoint_targets,
            settings=active_settings,
        )
        if isinstance(closure_result, SpanReconstructionFailure):
            return closure_result
        working_coordinates, endpoint_rmsd, iteration_count = closure_result

    inserted_payloads = donor_window.materialize_inserted_payloads(
        working_coordinates=working_coordinates,
        first_inserted_offset=first_inserted_offset,
        target_residue_ids=scope.absent_residue_ids,
    )
    junction_failure = _validate_materialized_peptide_junctions(
        source_structure,
        scope=scope,
        inserted_payloads=inserted_payloads,
        component_library=active_library,
    )
    if junction_failure is not None:
        return junction_failure

    return ReconstructedSpanCandidate(
        residue_payloads=inserted_payloads,
        endpoint_rmsd_angstrom=endpoint_rmsd,
        iteration_count=iteration_count,
    )


def _build_donor_window(
    donor_structure: ProteinStructure,
    residue_ids: tuple[ResidueId, ...],
    *,
    preceding_offset: int | None,
    following_offset: int | None,
    component_library: ComponentLibrary,
) -> _DonorWindow | SpanReconstructionFailure:
    residue_payloads: list[CompletionResiduePayload] = []
    coordinates: list[FloatArray] = []
    atom_index_by_residue_and_name: dict[tuple[int, str], int] = {}
    atom_indices_by_residue: list[IndexArray] = []
    for residue_offset, residue_id in enumerate(residue_ids):
        if donor_structure.constitution.residue_or_ligand(residue_id) is None:
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
                message=f"donor residue {residue_id.display_token()} is absent",
            )
        residue_index = donor_structure.constitution.residue_index(residue_id)
        residue_site = donor_structure.constitution.residue_site_at(residue_index)
        residue_geometry = donor_structure.residue_geometry(residue_index)
        required_backbone_names = (
            ("CA", "C", "O")
            if residue_offset == preceding_offset
            else ("N", "CA", "C")
            if residue_offset == following_offset
            else ("N", "CA", "C", "O")
        )
        missing_backbone_names = tuple(
            atom_name
            for atom_name in required_backbone_names
            if not residue_site.has_atom_site(atom_name)
            or not residue_geometry.has_atom(atom_name)
        )
        if missing_backbone_names:
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
                message=(
                    f"donor residue {residue_id.display_token()} lacks backbone "
                    f"atom(s) {', '.join(missing_backbone_names)}"
                ),
            )

        hydrogen_atom_names = {
            atom_site.name
            for atom_site in residue_site.atom_sites
            if atom_site.is_hydrogen()
        }
        heavy_residue_site = residue_site.without_atom_sites(hydrogen_atom_names)
        heavy_residue_geometry = residue_geometry.without_atoms(hydrogen_atom_names)
        formal_charge_by_atom_name = tuple(
            (atom_name, formal_charge)
            for atom_name, formal_charge in (
                donor_structure.residue_formal_charge_by_atom_name(residue_index)
            )
            if atom_name not in hydrogen_atom_names
        )
        residue_payloads.append(
            CompletionResiduePayload(
                residue_site=heavy_residue_site,
                residue_geometry=heavy_residue_geometry,
                formal_charge_by_atom_name=formal_charge_by_atom_name,
            )
        )

        residue_atom_indices: list[int] = []
        for atom_site in heavy_residue_site.atom_sites:
            atom_index = len(coordinates)
            position = heavy_residue_geometry.position(atom_site.name).to_array()
            coordinates.append(position)
            residue_atom_indices.append(atom_index)
            atom_index_by_residue_and_name[(residue_offset, atom_site.name)] = (
                atom_index
            )
        atom_indices_by_residue.append(np.asarray(residue_atom_indices, dtype=np.int64))

    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
            message="donor span did not provide a three-dimensional atom payload",
        )
    if not np.isfinite(coordinate_array).all():
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
            message="donor span contains non-finite atom coordinates",
        )

    # This is a window-local kinematic projection, not output connectivity.
    neighbors: list[set[int]] = [set() for _ in coordinates]
    rigid_pairs: set[tuple[int, int]] = set()
    for residue_offset, payload in enumerate(residue_payloads):
        template = component_library.get(payload.component_id)
        if template is None:
            return SpanReconstructionFailure(
                SpanReconstructionFailureKind.UNSUPPORTED_COMPONENT_CHEMISTRY,
                f"donor component {payload.component_id} "
                "lacks covalent motion constraints",
            )
        for bond in template.definition.bonds:
            left = atom_index_by_residue_and_name.get(
                (residue_offset, bond.atom_name_1)
            )
            right = atom_index_by_residue_and_name.get(
                (residue_offset, bond.atom_name_2)
            )
            if left is not None and right is not None:
                neighbors[left].add(right)
                neighbors[right].add(left)
                if bond.order != 1 or bond.aromatic:
                    rigid_pairs.add((min(left, right), max(left, right)))
    for offset in range(len(residue_ids) - 1):
        left = atom_index_by_residue_and_name[(offset, "C")]
        right = atom_index_by_residue_and_name[(offset + 1, "N")]
        neighbors[left].add(right)
        neighbors[right].add(left)
    local_indices = {
        donor_structure.constitution.atom_index(
            AtomRef(residue_ids[offset], name)
        ): index
        for (offset, name), index in atom_index_by_residue_and_name.items()
    }
    for bond in donor_structure.topology.bonds:
        if not is_covalent_like_relationship(bond):
            continue
        left = local_indices.get(bond.atom_index_1)
        right = local_indices.get(bond.atom_index_2)
        if left is not None and right is not None:
            neighbors[left].add(right)
            neighbors[right].add(left)
            if bond.order != 1 or bond.aromatic:
                rigid_pairs.add((min(left, right), max(left, right)))

    return _DonorWindow(
        residue_payloads=tuple(residue_payloads),
        coordinates=coordinate_array,
        atom_index_by_residue_and_name=atom_index_by_residue_and_name,
        atom_indices_by_residue=tuple(atom_indices_by_residue),
        bonded_indices=tuple(frozenset(indices) for indices in neighbors),
        rigid_bond_pairs=frozenset(rigid_pairs),
    )


def _place_donor_window_in_source_frame(
    donor_window: _DonorWindow,
    *,
    source_structure: ProteinStructure,
    scope: AbsentResidueSpanScope,
    preceding_offset: int | None,
    following_offset: int | None,
) -> FloatArray | SpanReconstructionFailure:
    if preceding_offset is not None:
        preceding_source_residue_id = scope.preceding_residue_id
        assert preceding_source_residue_id is not None
        moving_frame_points = donor_window.positions(
            preceding_offset,
            ("C", "CA", "O"),
        )
        target_frame_points = _backbone_positions(
            source_structure,
            preceding_source_residue_id,
            ("C", "CA", "O"),
        )
    else:
        following_source_residue_id = scope.following_residue_id
        assert following_source_residue_id is not None
        assert following_offset is not None
        moving_frame_points = donor_window.positions(
            following_offset,
            ("N", "CA", "C"),
        )
        target_frame_points = _backbone_positions(
            source_structure,
            following_source_residue_id,
            ("N", "CA", "C"),
        )
    if isinstance(target_frame_points, SpanReconstructionFailure):
        return target_frame_points

    moving_frame = _orthonormal_frame(moving_frame_points)
    target_frame = _orthonormal_frame(target_frame_points)
    if moving_frame is None or target_frame is None:
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.DEGENERATE_ANCHOR_FRAME,
            message=(
                "span reconstruction requires non-collinear donor and source anchors"
            ),
        )
    moving_origin, moving_basis = moving_frame
    target_origin, target_basis = target_frame
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = (
            donor_window.coordinates - moving_origin
        ) @ moving_basis.T @ target_basis + target_origin
    if not np.isfinite(transformed).all():
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
            message="span anchor-frame projection produced non-finite coordinates",
        )
    return transformed


def _orthonormal_frame(points: FloatArray) -> tuple[FloatArray, FloatArray] | None:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        origin = points[0]
        primary = points[1] - origin
        primary_norm = float(np.linalg.norm(primary))
        if not isfinite(primary_norm) or primary_norm <= ROTATION_AXIS_NORM_EPSILON:
            return None
        unit_primary = primary / primary_norm

        plane_vector = points[2] - origin
        projected_plane_vector = plane_vector - (
            float(np.dot(plane_vector, unit_primary)) * unit_primary
        )
        projected_norm = float(np.linalg.norm(projected_plane_vector))
        if not isfinite(projected_norm) or projected_norm <= ROTATION_AXIS_NORM_EPSILON:
            return None
        unit_plane = projected_plane_vector / projected_norm
        unit_normal = np.cross(unit_primary, unit_plane)
        basis = np.asarray((unit_primary, unit_plane, unit_normal), dtype=np.float64)
    if not np.isfinite(basis).all():
        return None
    return origin, basis


def _close_endpoint_by_cyclic_coordinate_descent(
    coordinates: FloatArray,
    *,
    rotation_axes: tuple[_RotationAxis, ...],
    endpoint_indices: IndexArray,
    endpoint_targets: FloatArray,
    settings: SpanClosureSettings,
) -> tuple[FloatArray, float, int] | SpanReconstructionFailure:
    initial_rmsd = _endpoint_rmsd(
        coordinates,
        endpoint_indices=endpoint_indices,
        endpoint_targets=endpoint_targets,
    )
    if not isfinite(initial_rmsd):
        return SpanReconstructionFailure(
            SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
            "span closure endpoint calculation overflowed",
        )
    if initial_rmsd <= settings.endpoint_rmsd_tolerance_angstrom:
        return coordinates.copy(), initial_rmsd, 0

    total_iterations = 0
    best_rmsd = initial_rmsd
    axis_objectives = tuple(
        (axis, np.isin(endpoint_indices, axis.moved_indices)) for axis in rotation_axes
    )
    seed_offset = _endpoint_descent_seed_offset(
        coordinates,
        axis_objectives=axis_objectives,
        endpoint_indices=endpoint_indices,
        endpoint_targets=endpoint_targets,
    )
    if isinstance(seed_offset, SpanReconstructionFailure):
        return seed_offset
    forward = tuple(range(len(axis_objectives)))
    seeded = forward[seed_offset:] + forward[:seed_offset]
    orders = tuple(dict.fromkeys((seeded, seeded[::-1], forward, forward[::-1])))
    # A different sweep start can avoid distributing one correctable torsion
    # across the whole span. Retries restart from the unchanged donor seed.
    for order in orders:
        working = coordinates.copy()
        endpoint_rmsd = initial_rmsd
        for _ in range(settings.maximum_iterations):
            previous_rmsd = endpoint_rmsd
            for axis_index in order:
                axis, moving_endpoints = axis_objectives[axis_index]
                if not moving_endpoints.any():
                    continue
                axis_start = working[axis.start_index].copy()
                axis_end = working[axis.end_index].copy()
                rotation = _optimal_axis_rotation(
                    working,
                    endpoint_indices=endpoint_indices[moving_endpoints],
                    endpoint_targets=endpoint_targets[moving_endpoints],
                    axis_start=axis_start,
                    axis_end=axis_end,
                )
                if isinstance(rotation, SpanReconstructionFailure):
                    return rotation
                if not _rotate_points_in_place(
                    working,
                    point_indices=axis.moved_indices,
                    axis_start=axis_start,
                    axis_end=axis_end,
                    theta_radians=rotation,
                ):
                    return SpanReconstructionFailure(
                        SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
                        "span closure rotation produced non-finite coordinates",
                    )
            total_iterations += 1
            endpoint_rmsd = _endpoint_rmsd(
                working,
                endpoint_indices=endpoint_indices,
                endpoint_targets=endpoint_targets,
            )
            if not isfinite(endpoint_rmsd):
                return SpanReconstructionFailure(
                    SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
                    "span closure endpoint calculation overflowed",
                )
            best_rmsd = min(best_rmsd, endpoint_rmsd)
            if endpoint_rmsd <= settings.endpoint_rmsd_tolerance_angstrom:
                return working, endpoint_rmsd, total_iterations
            if (
                abs(previous_rmsd - endpoint_rmsd)
                <= settings.convergence_delta_angstrom
            ):
                break

    return SpanReconstructionFailure(
        SpanReconstructionFailureKind.NON_CONVERGENT_CLOSURE,
        f"donor-seeded CCD did not close the span after {total_iterations} sweeps "
        f"across {len(orders)} deterministic axis orders; "
        f"best endpoint RMSD was {best_rmsd:.3f} A",
    )


def _endpoint_descent_seed_offset(
    coordinates: FloatArray,
    *,
    axis_objectives: tuple[tuple[_RotationAxis, npt.NDArray[np.bool_]], ...],
    endpoint_indices: IndexArray,
    endpoint_targets: FloatArray,
) -> int | SpanReconstructionFailure:
    """Choose the sweep start with the smallest one-axis endpoint residual."""

    best_rmsd = float("inf")
    best_offset = 0
    local_endpoint_indices = np.arange(len(endpoint_indices), dtype=np.int64)
    for offset, (axis, moving_endpoints) in enumerate(axis_objectives):
        if not moving_endpoints.any():
            continue
        axis_start = coordinates[axis.start_index]
        axis_end = coordinates[axis.end_index]
        rotation = _optimal_axis_rotation(
            coordinates,
            endpoint_indices=endpoint_indices[moving_endpoints],
            endpoint_targets=endpoint_targets[moving_endpoints],
            axis_start=axis_start,
            axis_end=axis_end,
        )
        if isinstance(rotation, SpanReconstructionFailure):
            return rotation
        trial_endpoints = coordinates[endpoint_indices].copy()
        if not _rotate_points_in_place(
            trial_endpoints,
            point_indices=local_endpoint_indices[moving_endpoints],
            axis_start=axis_start,
            axis_end=axis_end,
            theta_radians=rotation,
        ):
            return SpanReconstructionFailure(
                SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
                "span closure endpoint descent probe produced non-finite coordinates",
            )
        rmsd = _endpoint_rmsd(
            trial_endpoints,
            endpoint_indices=local_endpoint_indices,
            endpoint_targets=endpoint_targets,
        )
        if not isfinite(rmsd):
            return SpanReconstructionFailure(
                SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
                "span closure endpoint descent probe overflowed",
            )
        if rmsd < best_rmsd:
            best_rmsd, best_offset = rmsd, offset
    return best_offset


def _optimal_axis_rotation(
    coordinates: FloatArray,
    *,
    endpoint_indices: IndexArray,
    endpoint_targets: FloatArray,
    axis_start: FloatArray,
    axis_end: FloatArray,
) -> float | SpanReconstructionFailure:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        unit_axis = axis_end - axis_start
        axis_norm = float(np.linalg.norm(unit_axis))
        if not isfinite(axis_norm):
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
                message="span closure axis calculation overflowed",
            )
        if axis_norm <= ROTATION_AXIS_NORM_EPSILON:
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.DEGENERATE_ANCHOR_FRAME,
                message="span closure encountered a degenerate backbone axis",
            )
        unit_axis /= axis_norm

        moving = coordinates[endpoint_indices] - axis_start
        targets = endpoint_targets - axis_start
        moving -= np.outer(moving @ unit_axis, unit_axis)
        targets -= np.outer(targets @ unit_axis, unit_axis)
        cosine_coefficient = float(np.sum(moving * targets))
        sine_coefficient = float(np.sum(np.cross(unit_axis, moving) * targets))
    if not isfinite(cosine_coefficient) or not isfinite(sine_coefficient):
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
            message="span closure rotation calculation overflowed",
        )
    angle = atan2(sine_coefficient, cosine_coefficient)
    if not isfinite(angle):
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
            message="span closure rotation calculation produced a non-finite angle",
        )
    return angle


def _rotate_points_in_place(
    coordinates: FloatArray,
    *,
    point_indices: IndexArray,
    axis_start: FloatArray,
    axis_end: FloatArray,
    theta_radians: float,
) -> bool:
    if not point_indices.size:
        return True
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        unit_axis = axis_end - axis_start
        unit_axis /= np.linalg.norm(unit_axis)
        relative = coordinates[point_indices] - axis_start
        cosine = np.cos(theta_radians)
        sine = np.sin(theta_radians)
        rotated = (
            relative * cosine
            + np.cross(unit_axis, relative) * sine
            + np.outer(relative @ unit_axis, unit_axis) * (1.0 - cosine)
            + axis_start
        )
    if not np.isfinite(rotated).all():
        return False
    coordinates[point_indices] = rotated
    return True


def _endpoint_rmsd(
    coordinates: FloatArray,
    *,
    endpoint_indices: IndexArray,
    endpoint_targets: FloatArray,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        squared_distances = np.sum(
            (coordinates[endpoint_indices] - endpoint_targets) ** 2,
            axis=1,
        )
        return float(np.sqrt(np.mean(squared_distances)))


def _validate_materialized_peptide_junctions(
    source_structure: ProteinStructure,
    *,
    scope: AbsentResidueSpanScope,
    inserted_payloads: tuple[CompletionResiduePayload, ...],
    component_library: ComponentLibrary,
) -> SpanReconstructionFailure | None:
    payloads = list(inserted_payloads)
    for residue_id, names, at_start in (
        (scope.preceding_residue_id, ("CA", "C", "O"), True),
        (scope.following_residue_id, ("N", "CA"), False),
    ):
        if residue_id is None:
            continue
        positions = _backbone_positions(source_structure, residue_id, names)
        if isinstance(positions, SpanReconstructionFailure):
            return positions
        index = source_structure.constitution.residue_index(residue_id)
        payload = CompletionResiduePayload(
            source_structure.constitution.residue_site_at(index),
            source_structure.residue_geometry(index),
        )
        if at_start:
            payloads.insert(0, payload)
        else:
            payloads.append(payload)

    for preceding, following in zip(payloads, payloads[1:], strict=False):
        template = component_library.get(following.component_id)
        if template is None:
            return SpanReconstructionFailure(
                SpanReconstructionFailureKind.UNSUPPORTED_COMPONENT_CHEMISTRY,
                f"component {following.component_id} lacks amide-N chemistry",
            )
        heavy_names = frozenset(template.expected_heavy_atom_names()) | frozenset(
            atom.name for atom in following.atom_sites if not atom.is_hydrogen()
        )
        neighbor_names = template.definition.bonded_atom_names("N") & heavy_names
        missing = sorted(
            name for name in neighbor_names if not following.has_atom(name)
        )
        if missing:
            return SpanReconstructionFailure(
                SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
                f"residue {following.residue_id.display_token()} "
                f"lacks amide-N neighbor(s) {', '.join(missing)}",
            )
        substituents = [
            following.position(name) for name in sorted(neighbor_names - {"CA"})
        ]
        if following.residue_id == scope.following_residue_id:
            # Source N can have explicit substituents beyond its component graph.
            nitrogen_index = source_structure.constitution.atom_index(
                AtomRef(following.residue_id, "N")
            )
            excluded = {AtomRef(following.residue_id, name) for name in neighbor_names}
            excluded.add(AtomRef(preceding.residue_id, "C"))
            for bond in source_structure.topology.bonds:
                if not is_covalent_like_relationship(bond) or not bond.involves(
                    nitrogen_index
                ):
                    continue
                other = (
                    bond.atom_index_2
                    if bond.atom_index_1 == nitrogen_index
                    else bond.atom_index_1
                )
                ref = source_structure.constitution.atom_ref_at(other)
                if (
                    ref not in excluded
                    and not source_structure.constitution.atom_site_at(
                        other
                    ).is_hydrogen()
                ):
                    substituents.append(source_structure.geometry.position(other))
        junction = PeptideJunctionGeometry(
            preceding_ca=preceding.position("CA"),
            carbonyl_c=preceding.position("C"),
            carbonyl_o=preceding.position("O"),
            nitrogen=following.position("N"),
            following_ca=following.position("CA"),
            nitrogen_substituents=tuple(substituents),
        )
        if not junction.is_plausible():
            location = (
                "preceding"
                if preceding.residue_id == scope.preceding_residue_id
                else "following"
                if following.residue_id == scope.following_residue_id
                else "internal"
            )
            return SpanReconstructionFailure(
                SpanReconstructionFailureKind.INVALID_PEPTIDE_JUNCTION,
                f"reconstructed span failed the {location} peptide-junction gate "
                f"between {preceding.residue_id.display_token()} "
                f"and {following.residue_id.display_token()}",
            )
    return None


def _backbone_positions(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_names: tuple[str, ...],
) -> FloatArray | SpanReconstructionFailure:
    residue = structure.constitution.residue_or_ligand(residue_id)
    if residue is None:
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
            message=f"source anchor {residue_id.display_token()} is absent",
        )
    missing_atom_names = tuple(
        atom_name for atom_name in atom_names if not residue.has_atom_site(atom_name)
    )
    if missing_atom_names:
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
            message=(
                f"source anchor {residue_id.display_token()} lacks backbone atom(s) "
                f"{', '.join(missing_atom_names)}"
            ),
        )
    residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    )
    positions = np.asarray(
        [residue_geometry.position(atom_name).to_array() for atom_name in atom_names],
        dtype=np.float64,
    )
    if not np.isfinite(positions).all():
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
            message=f"source anchor {residue_id.display_token()} is non-finite",
        )
    return positions
