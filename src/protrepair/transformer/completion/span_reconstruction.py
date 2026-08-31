"""Anchor-constrained reconstruction of missing polymer residue spans."""

from dataclasses import dataclass
from enum import Enum
from math import acos, atan2, degrees, isfinite
from numbers import Real

import numpy as np
import numpy.typing as npt

from protrepair.geometry import Vec3
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]

PEPTIDE_CN_DISTANCE_MIN_ANGSTROM = 1.1
PEPTIDE_CN_DISTANCE_MAX_ANGSTROM = 1.6
PEPTIDE_BOND_ANGLE_MIN_DEGREES = 85.0
PEPTIDE_BOND_ANGLE_MAX_DEGREES = 145.0
PEPTIDE_PLANARITY_MAX_DEVIATION_DEGREES = 30.0
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
        Maximum complete CCD sweeps before returning non-convergence.
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

    def indices_after_residue(self, residue_offset: int) -> IndexArray:
        """Return atom indices belonging to residues after one offset."""

        later = self.atom_indices_by_residue[residue_offset + 1 :]
        if not later:
            return np.asarray((), dtype=np.int64)
        return np.concatenate(later)

    def internal_span_rotation_axes(
        self,
        *,
        preceding_offset: int,
        first_inserted_offset: int,
        inserted_residue_count: int,
        following_offset: int,
    ) -> tuple[_RotationAxis, ...]:
        """Return donor-window torsion axes used by internal closure."""

        axes: list[_RotationAxis] = [
            _RotationAxis(
                start_index=self.atom_index(preceding_offset, "CA"),
                end_index=self.atom_index(preceding_offset, "C"),
                moved_indices=self.indices_after_residue(preceding_offset),
            )
        ]
        for residue_offset in range(
            first_inserted_offset,
            first_inserted_offset + inserted_residue_count,
        ):
            later_indices = self.indices_after_residue(residue_offset)
            residue_indices = self.atom_indices_by_residue[residue_offset]
            fixed_phi_indices = {
                self.atom_index(residue_offset, "N"),
                self.atom_index(residue_offset, "CA"),
            }
            phi_local_indices = np.asarray(
                [
                    atom_index
                    for atom_index in residue_indices
                    if int(atom_index) not in fixed_phi_indices
                ],
                dtype=np.int64,
            )
            axes.append(
                _RotationAxis(
                    start_index=self.atom_index(residue_offset, "N"),
                    end_index=self.atom_index(residue_offset, "CA"),
                    moved_indices=np.concatenate((phi_local_indices, later_indices)),
                )
            )
            axes.append(
                _RotationAxis(
                    start_index=self.atom_index(residue_offset, "CA"),
                    end_index=self.atom_index(residue_offset, "C"),
                    moved_indices=np.concatenate(
                        (
                            np.asarray(
                                [self.atom_index(residue_offset, "O")],
                                dtype=np.int64,
                            ),
                            later_indices,
                        )
                    ),
                )
            )

        following_indices = self.atom_indices_by_residue[following_offset]
        fixed_following_indices = {
            self.atom_index(following_offset, "N"),
            self.atom_index(following_offset, "CA"),
        }
        axes.append(
            _RotationAxis(
                start_index=self.atom_index(following_offset, "N"),
                end_index=self.atom_index(following_offset, "CA"),
                moved_indices=np.asarray(
                    [
                        atom_index
                        for atom_index in following_indices
                        if int(atom_index) not in fixed_following_indices
                    ],
                    dtype=np.int64,
                ),
            )
        )
        return tuple(axes)

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

    return _DonorWindow(
        residue_payloads=tuple(residue_payloads),
        coordinates=coordinate_array,
        atom_index_by_residue_and_name=atom_index_by_residue_and_name,
        atom_indices_by_residue=tuple(atom_indices_by_residue),
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
    working = coordinates.copy()
    endpoint_rmsd = _endpoint_rmsd(
        working,
        endpoint_indices=endpoint_indices,
        endpoint_targets=endpoint_targets,
    )
    if not isfinite(endpoint_rmsd):
        return SpanReconstructionFailure(
            kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
            message="span closure endpoint calculation overflowed",
        )
    if endpoint_rmsd <= settings.endpoint_rmsd_tolerance_angstrom:
        return working, endpoint_rmsd, 0

    for iteration_count in range(1, settings.maximum_iterations + 1):
        previous_rmsd = endpoint_rmsd
        for axis in rotation_axes:
            axis_start = working[axis.start_index].copy()
            axis_end = working[axis.end_index].copy()
            rotation_outcome = _optimal_axis_rotation(
                working,
                endpoint_indices=endpoint_indices,
                endpoint_targets=endpoint_targets,
                axis_start=axis_start,
                axis_end=axis_end,
            )
            if isinstance(rotation_outcome, SpanReconstructionFailure):
                return rotation_outcome
            if not _rotate_points_in_place(
                working,
                point_indices=axis.moved_indices,
                axis_start=axis_start,
                axis_end=axis_end,
                theta_radians=rotation_outcome,
            ):
                return SpanReconstructionFailure(
                    kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
                    message="span closure rotation produced non-finite coordinates",
                )

        endpoint_rmsd = _endpoint_rmsd(
            working,
            endpoint_indices=endpoint_indices,
            endpoint_targets=endpoint_targets,
        )
        if not isfinite(endpoint_rmsd):
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.NON_FINITE_COORDINATES,
                message="span closure endpoint calculation overflowed",
            )
        if endpoint_rmsd <= settings.endpoint_rmsd_tolerance_angstrom:
            return working, endpoint_rmsd, iteration_count
        if abs(previous_rmsd - endpoint_rmsd) <= settings.convergence_delta_angstrom:
            break

    return SpanReconstructionFailure(
        kind=SpanReconstructionFailureKind.NON_CONVERGENT_CLOSURE,
        message=(
            "donor-seeded CCD did not close the span within "
            f"{settings.maximum_iterations} iterations; endpoint RMSD was "
            f"{endpoint_rmsd:.3f} A"
        ),
    )


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
) -> SpanReconstructionFailure | None:
    first_payload = inserted_payloads[0]
    last_payload = inserted_payloads[-1]
    preceding_residue_id = scope.preceding_residue_id
    if preceding_residue_id is not None:
        preceding_positions = _backbone_positions(
            source_structure,
            preceding_residue_id,
            ("CA", "C", "O"),
        )
        if isinstance(preceding_positions, SpanReconstructionFailure):
            return preceding_positions
        if not _peptide_junction_is_plausible(
            preceding_ca=preceding_positions[0],
            preceding_c=preceding_positions[1],
            preceding_o=preceding_positions[2],
            following_n=first_payload.position("N").to_array(),
            following_ca=first_payload.position("CA").to_array(),
        ):
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.INVALID_PEPTIDE_JUNCTION,
                message="reconstructed span failed the preceding peptide-junction gate",
            )

    for preceding_payload, following_payload in zip(
        inserted_payloads,
        inserted_payloads[1:],
        strict=False,
    ):
        if not _peptide_junction_is_plausible(
            preceding_ca=preceding_payload.position("CA").to_array(),
            preceding_c=preceding_payload.position("C").to_array(),
            preceding_o=preceding_payload.position("O").to_array(),
            following_n=following_payload.position("N").to_array(),
            following_ca=following_payload.position("CA").to_array(),
        ):
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.INVALID_PEPTIDE_JUNCTION,
                message=(
                    "reconstructed span failed the internal peptide-junction "
                    "gate between "
                    f"{preceding_payload.residue_id.display_token()} and "
                    f"{following_payload.residue_id.display_token()}"
                ),
            )

    following_residue_id = scope.following_residue_id
    if following_residue_id is not None:
        following_positions = _backbone_positions(
            source_structure,
            following_residue_id,
            ("N", "CA"),
        )
        if isinstance(following_positions, SpanReconstructionFailure):
            return following_positions
        if not _peptide_junction_is_plausible(
            preceding_ca=last_payload.position("CA").to_array(),
            preceding_c=last_payload.position("C").to_array(),
            preceding_o=last_payload.position("O").to_array(),
            following_n=following_positions[0],
            following_ca=following_positions[1],
        ):
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.INVALID_PEPTIDE_JUNCTION,
                message="reconstructed span failed the following peptide-junction gate",
            )
    return None


def _peptide_junction_is_plausible(
    *,
    preceding_ca: FloatArray,
    preceding_c: FloatArray,
    preceding_o: FloatArray,
    following_n: FloatArray,
    following_ca: FloatArray,
) -> bool:
    distance = float(np.linalg.norm(following_n - preceding_c))
    if not (
        PEPTIDE_CN_DISTANCE_MIN_ANGSTROM <= distance <= PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
    ):
        return False

    angles = (
        _bond_angle_degrees(preceding_ca, preceding_c, following_n),
        _bond_angle_degrees(preceding_o, preceding_c, following_n),
        _bond_angle_degrees(preceding_c, following_n, following_ca),
    )
    if not all(
        angle is not None
        and PEPTIDE_BOND_ANGLE_MIN_DEGREES <= angle <= PEPTIDE_BOND_ANGLE_MAX_DEGREES
        for angle in angles
    ):
        return False

    planar_dihedrals = (
        _dihedral_degrees(preceding_ca, preceding_c, following_n, following_ca),
        _dihedral_degrees(preceding_o, preceding_c, following_n, following_ca),
    )
    return all(
        angle is not None
        and _planarity_deviation_degrees(angle)
        <= PEPTIDE_PLANARITY_MAX_DEVIATION_DEGREES
        for angle in planar_dihedrals
    )


def _bond_angle_degrees(
    first: FloatArray,
    center: FloatArray,
    third: FloatArray,
) -> float | None:
    with np.errstate(over="ignore", invalid="ignore"):
        first_vector = first - center
        third_vector = third - center
        denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(third_vector))
    if not isfinite(denominator) or denominator <= ROTATION_AXIS_NORM_EPSILON:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        cosine = float(np.dot(first_vector, third_vector)) / denominator
    if not isfinite(cosine):
        return None
    return degrees(acos(min(1.0, max(-1.0, cosine))))


def _dihedral_degrees(
    first: FloatArray,
    second: FloatArray,
    third: FloatArray,
    fourth: FloatArray,
) -> float | None:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        first_bond = -(second - first)
        central_bond = third - second
        last_bond = fourth - third
        central_norm = float(np.linalg.norm(central_bond))
        if not isfinite(central_norm) or central_norm <= ROTATION_AXIS_NORM_EPSILON:
            return None
        central_unit = central_bond / central_norm
        first_plane = first_bond - np.dot(first_bond, central_unit) * central_unit
        last_plane = last_bond - np.dot(last_bond, central_unit) * central_unit
        first_plane_norm = float(np.linalg.norm(first_plane))
        last_plane_norm = float(np.linalg.norm(last_plane))
        if (
            not isfinite(first_plane_norm)
            or not isfinite(last_plane_norm)
            or first_plane_norm <= ROTATION_AXIS_NORM_EPSILON
            or last_plane_norm <= ROTATION_AXIS_NORM_EPSILON
        ):
            return None
        cosine_component = float(np.dot(first_plane, last_plane))
        sine_component = float(np.dot(np.cross(central_unit, first_plane), last_plane))
    if not isfinite(cosine_component) or not isfinite(sine_component):
        return None
    angle = degrees(atan2(sine_component, cosine_component))
    return angle if isfinite(angle) else None


def _planarity_deviation_degrees(angle_degrees: float) -> float:
    absolute_angle = abs(angle_degrees)
    return min(absolute_angle, abs(180.0 - absolute_angle))


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
