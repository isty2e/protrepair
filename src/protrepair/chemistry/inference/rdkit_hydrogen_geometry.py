"""Native H placement that preserves fixed atoms and specified stereo."""

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.inference.rdkit_graph import (
    has_declared_stereochemistry,
    molecule_with_pose_stereochemistry,
    require_compatible_pose_stereochemistry,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.geometry.placement_vector import (
    normalized_vector_or_none,
    scaled_point_from_origin_or_none,
)
from protrepair.structure.geometry import ResidueGeometry

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


def named_pose_hydrogen_bond_lengths(
    heavy: "Mol", names: tuple[str, ...], geometry: ResidueGeometry
) -> tuple[tuple[str, float], ...]:
    """Measure temporary native H bonds on fixed, named heavy coordinates.

    Parameters
    ----------
    heavy : Mol
        Selected heavy-atom graph with its hydrogen inventory.
    names : tuple[str, ...]
        Heavy-atom names in native index order.
    geometry : ResidueGeometry
        Deposited or current pose; existing H positions do not enter the fit.

    Returns
    -------
    tuple[tuple[str, float], ...]
        Heavy-parent name and bond length in angstroms for each generated H.

    Raises
    ------
    ValueError
        The atom mapping, coordinates or native H placement are invalid.
    RdkitUnavailableError
        Native preparation is unavailable.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("reference H correspondence requires RDKit")
    if heavy.GetNumAtoms() != len(names):
        raise ValueError("heavy-atom names must match the native graph")
    pose = chem.Mol(heavy)
    pose.RemoveAllConformers()
    conformer = chem.Conformer(len(names))
    conformer.Set3D(True)
    for index, name in enumerate(names):
        if not geometry.has_atom(name):
            raise ValueError(f"reference correspondence has no coordinate for {name}")
        position = geometry.position(name)
        if not all(isfinite(value) for value in position):
            raise ValueError(
                "reference correspondence requires finite heavy coordinates"
            )
        conformer.SetAtomPosition(index, tuple(position))
    pose.AddConformer(conformer)
    try:
        prepared = chem.AddHs(pose, addCoords=True)
    except RuntimeError as error:
        raise ValueError(f"reference H preparation failed: {error}") from error
    positions = prepared.GetConformer()
    if any(
        tuple(positions.GetAtomPosition(index)) != tuple(geometry.position(name))
        for index, name in enumerate(names)
    ):
        raise ValueError("reference H preparation moved fixed heavy coordinates")

    lengths = []
    for atom in prepared.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbors = tuple(atom.GetNeighbors())
        if len(neighbors) != 1 or neighbors[0].GetIdx() >= len(names):
            raise ValueError("temporary H requires one reference heavy parent")
        parent = names[neighbors[0].GetIdx()]
        point = Vec3.from_iterable(positions.GetAtomPosition(atom.GetIdx()))
        distance = point.distance_to(geometry.position(parent))
        if not isfinite(distance) or distance <= 0:
            raise ValueError("reference H preparation produced an invalid bond length")
        lengths.append((parent, distance))
    return tuple(lengths)


def add_hydrogens_preserving_stereochemistry(molecule: "Mol") -> "Mol":
    """Add H coordinates, correcting only a new tetrahedral H when necessary.

    Parameters
    ----------
    molecule : Mol
        Sanitized selected chemistry with aligned coordinates and declared stereo.

    Returns
    -------
    Mol
        New native graph. Original atom positions and native H lengths are fixed;
        already-compatible placements are unchanged.

    Raises
    ------
    ValueError
        Source stereo conflicts, original coordinates change, or a generated
        stereo mismatch cannot be corrected using one new tetrahedral H.
    RdkitUnavailableError
        Native hydrogen placement is unavailable.

    Notes
    -----
    This is a local placement correction, not an optimized geometry certificate.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("H placement requires operational RDKit")
    require_compatible_pose_stereochemistry(molecule)
    result = chem.AddHs(chem.Mol(molecule), addCoords=True)
    original = molecule.GetConformer()
    conformer = result.GetConformer()
    original_count = molecule.GetNumAtoms()
    if any(
        tuple(original.GetAtomPosition(i)) != tuple(conformer.GetAtomPosition(i))
        for i in range(original_count)
    ):
        raise ValueError("native H placement changed original atom coordinates")
    if not has_declared_stereochemistry(result):
        return result

    observed = molecule_with_pose_stereochemistry(result)
    for atom in result.GetAtoms():
        actual = observed.GetAtomWithIdx(atom.GetIdx())
        if not (
            atom.GetChiralTag()
            and actual.GetChiralTag()
            and atom.GetChiralTag() != actual.GetChiralTag()
        ):
            continue
        new_hydrogens = tuple(
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIdx() >= original_count
        )
        if len(new_hydrogens) != 1 or atom.GetDegree() != 4:
            raise ValueError("stereochemistry mismatch has no single new H to place")
        hydrogen = new_hydrogens[0]
        center = Vec3.from_iterable(conformer.GetAtomPosition(atom.GetIdx()))
        old_h = Vec3.from_iterable(conformer.GetAtomPosition(hydrogen.GetIdx()))
        anchors = tuple(
            Vec3.from_iterable(conformer.GetAtomPosition(neighbor.GetIdx())).to_array()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetIdx() != hydrogen.GetIdx()
        )
        with np.errstate(over="ignore", invalid="ignore"):
            normal = normalized_vector_or_none(
                np.asarray(
                    np.cross(anchors[1] - anchors[0], anchors[2] - anchors[0]),
                    dtype=np.float64,
                )
            )
        if normal is None:
            raise ValueError("stereochemistry-preserving H plane is undefined")

        # An unassigned center's displacement cannot choose the declared side.
        for direction in (normal, -normal):
            point = scaled_point_from_origin_or_none(
                center.to_array(),
                center.to_array() + direction,
                center.distance_to(old_h),
            )
            if point is None:
                continue
            conformer.SetAtomPosition(hydrogen.GetIdx(), tuple(point))
            candidate = molecule_with_pose_stereochemistry(result)
            if (
                candidate.GetAtomWithIdx(atom.GetIdx()).GetChiralTag()
                == atom.GetChiralTag()
            ):
                observed = candidate
                break
        else:
            raise ValueError("new H stereo cannot be resolved with the fixed anchors")
    require_compatible_pose_stereochemistry(result)
    return result
