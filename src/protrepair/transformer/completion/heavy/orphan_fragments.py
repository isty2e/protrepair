"""Orphan-fragment policy decisions for atom completion."""

from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from protrepair.geometry import RigidTransform, Vec3
from protrepair.structure.geometry import AtomGeometry
from protrepair.transformer.completion.fragment_matching import ResidueFragmentMatch
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

FRAGMENT_REPAIR_ANCHOR_ATOM_NAMES: tuple[str, ...] = ("N", "CA", "C", "O")
FloatArray = npt.NDArray[np.float64]
MAX_SALVAGE_ALIGNMENT_RMSD = 0.35


def prepare_residue_for_fragment_repair(
    residue: CompletionResiduePayload,
    *,
    fragment_match: ResidueFragmentMatch,
    orphan_fragment_policy: OrphanFragmentPolicy,
    preferred_anchor_atom_names: Collection[str] = FRAGMENT_REPAIR_ANCHOR_ATOM_NAMES,
) -> CompletionResiduePayload:
    """Drop orphan template atoms so disconnected fragments can rebuild cleanly."""

    if orphan_fragment_policy is OrphanFragmentPolicy.PRESERVE:
        return residue

    if not fragment_match.has_missing_atoms():
        return residue

    orphan_atom_names = fragment_match.orphan_atom_names(
        preferred_anchor_atom_names=preferred_anchor_atom_names,
    )
    if not orphan_atom_names:
        return residue

    orphan_atom_name_set = frozenset(orphan_atom_names)
    return residue.without_atom_sites(orphan_atom_name_set)


def apply_orphan_fragment_policy(
    *,
    original_residue: CompletionResiduePayload,
    repaired_residue: CompletionResiduePayload,
    fragment_match: ResidueFragmentMatch,
    orphan_fragment_policy: OrphanFragmentPolicy,
    preferred_anchor_atom_names: Collection[str] = FRAGMENT_REPAIR_ANCHOR_ATOM_NAMES,
) -> CompletionResiduePayload:
    """Apply the normalized orphan-fragment policy after atom completion."""

    if orphan_fragment_policy is not OrphanFragmentPolicy.SALVAGE_WHEN_SAFE:
        return repaired_residue

    if not fragment_match.has_missing_atoms():
        return repaired_residue

    return salvage_orphan_fragments(
        original_residue,
        repaired_residue,
        fragment_match=fragment_match,
        preferred_anchor_atom_names=preferred_anchor_atom_names,
    )


def salvage_orphan_fragments(
    original_residue: CompletionResiduePayload,
    repaired_residue: CompletionResiduePayload,
    *,
    fragment_match: ResidueFragmentMatch,
    preferred_anchor_atom_names: Collection[str],
) -> CompletionResiduePayload:
    """Rigidly reattach salvageable orphan fragments onto a rebuilt residue."""

    primary_fragment = frozenset(
        fragment_match.primary_repair_fragment(
            preferred_anchor_atom_names=preferred_anchor_atom_names,
        )
    )
    salvaged_residue = repaired_residue
    for fragment in fragment_match.present_fragments:
        if len(fragment) < 2:
            continue

        if any(atom_name in primary_fragment for atom_name in fragment):
            continue

        if not all(
            original_residue.has_atom(atom_name)
            and repaired_residue.has_atom(atom_name)
            for atom_name in fragment
        ):
            continue

        source_points = _coordinate_matrix(original_residue, fragment)
        target_points = _coordinate_matrix(repaired_residue, fragment)
        transform = RigidTransform.fit(
            source_points,
            target_points,
            minimum_point_count=2,
            minimum_rank=1,
        )
        if transform is None:
            continue

        transformed_points = transform.apply_many(source_points)
        deltas = transformed_points - target_points
        squared_distances = np.sum(deltas * deltas, axis=1)
        rmsd = float(np.sqrt(np.mean(squared_distances)))
        if rmsd > MAX_SALVAGE_ALIGNMENT_RMSD:
            continue

        salvaged_residue = CompletionResiduePayload(
            residue_site=salvaged_residue.residue_site,
            residue_geometry=salvaged_residue.residue_geometry.with_atom_geometries(
                (
                    (
                        atom_name,
                        AtomGeometry(
                            position=Vec3.from_iterable(transformed_points[index]),
                            occupancy=original_residue.atom_geometry(
                                atom_name
                            ).occupancy,
                            b_factor=original_residue.atom_geometry(
                                atom_name
                            ).b_factor,
                            altloc=original_residue.atom_geometry(
                                atom_name
                            ).altloc,
                        ),
                    )
                    for index, atom_name in enumerate(fragment)
                )
            ),
            formal_charge_by_atom_name=salvaged_residue.formal_charge_by_atom_name,
        )

    return salvaged_residue


def _coordinate_matrix(
    residue: CompletionResiduePayload,
    atom_names: tuple[str, ...],
) -> FloatArray:
    """Return one coordinate matrix for a tuple of atom names."""

    residue_geometry = residue.residue_geometry
    return np.asarray(
        [
            [
                residue_geometry.position(atom_name).x,
                residue_geometry.position(atom_name).y,
                residue_geometry.position(atom_name).z,
            ]
            for atom_name in atom_names
        ],
        dtype=np.float64,
    )
