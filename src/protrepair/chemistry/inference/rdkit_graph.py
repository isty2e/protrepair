"""RDKit graph declarations and independently observed pose stereochemistry."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.errors import BondOrderConstraintError, RdkitUnavailableError

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


def has_declared_stereochemistry(molecule: "Mol") -> bool:
    """Return whether the graph declares atom or double-bond stereo.

    Parameters
    ----------
    molecule : Mol
        Sanitized named or native-indexed chemical graph.

    Returns
    -------
    bool
        True for assigned tags, not an unspecified or ANY stereo constraint.

    Raises
    ------
    RdkitUnavailableError
        Native stereochemical interpretation is unavailable.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("stereochemistry requires operational RDKit")
    return any(
        atom.GetChiralTag() != chem.ChiralType.CHI_UNSPECIFIED
        for atom in molecule.GetAtoms()
    ) or any(
        bond.GetStereo() not in (chem.BondStereo.STEREONONE, chem.BondStereo.STEREOANY)
        for bond in molecule.GetBonds()
    )


def require_compatible_pose_stereochemistry(molecule: "Mol") -> None:
    """Reject confirmed 3D stereo opposite to the same graph's declarations.

    Parameters
    ----------
    molecule : Mol
        Sanitized chemical graph with its aligned 3D conformer.

    Raises
    ------
    ValueError
        A declared atom or double-bond stereo has a confirmed opposite pose.
    RdkitUnavailableError
        Native stereochemical interpretation is unavailable.

    Notes
    -----
    Undefined graph stereo imposes no constraint. An unassigned geometric
    result is not treated as opposite stereo. The molecule is not mutated.
    RDKit can assign tetrahedral configuration from the heavy neighbors without
    using an attached H position; this is not an independent H-geometry check.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("stereochemistry requires operational RDKit")
    if not has_declared_stereochemistry(molecule):
        return
    observed = molecule_with_pose_stereochemistry(molecule)
    for expected, actual in zip(molecule.GetAtoms(), observed.GetAtoms(), strict=True):
        if (
            expected.GetChiralTag()
            and actual.GetChiralTag()
            and expected.GetChiralTag() != actual.GetChiralTag()
        ):
            raise ValueError(
                "declared atom stereochemistry contradicts source geometry"
            )
    for expected, actual in zip(molecule.GetBonds(), observed.GetBonds(), strict=True):
        if (
            expected.GetStereo()
            not in (chem.BondStereo.STEREONONE, chem.BondStereo.STEREOANY)
            and actual.GetStereo()
            not in (chem.BondStereo.STEREONONE, chem.BondStereo.STEREOANY)
            and expected.GetStereo() != actual.GetStereo()
        ):
            raise ValueError(
                "declared bond stereochemistry contradicts source geometry"
            )


def molecule_with_pose_stereochemistry(molecule: "Mol") -> "Mol":
    """Copy a native graph, replacing inherited stereo with its 3D observation.

    Parameters
    ----------
    molecule : Mol
        Chemical graph with an aligned 3D conformer.

    Returns
    -------
    Mol
        Geometry-derived stereo on a separate graph; the input is not mutated.

    Raises
    ------
    RdkitUnavailableError
        Native stereochemical interpretation is unavailable.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("stereochemistry requires operational RDKit")
    observed = chem.Mol(molecule)
    chem.RemoveStereochemistry(observed)
    chem.AssignStereochemistryFrom3D(observed, replaceExistingTags=True)
    return observed


def kekulized_molecule(
    molecule: "Mol",
    *,
    bond_order_constraints: Mapping[tuple[int, int], int] | None = None,
) -> "Mol":
    """Copy a native graph with integral aromatic orders and retained flags.

    Parameters
    ----------
    molecule : Mol
        Resolved native chemistry, including any typed coordination bonds.
    bond_order_constraints : Mapping[tuple[int, int], int] or None
        Reported integral covalent orders keyed by native endpoint pairs.
        Only aromatic single/double assignments may differ from the input.

    Returns
    -------
    Mol
        Defensive copy with a source-compatible Kekule form. Atom identities,
        charges, H, stereochemistry and conformers are not modified. Coordination
        remains typed; this operation does not infer a microstate or sanitize it.

    Raises
    ------
    ValueError
        The graph cannot be kekulized.
    BondOrderConstraintError
        Reported orders cannot preserve the resolved chemistry.
    RdkitUnavailableError
        RDKit is not operational.
    """

    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("graph projection requires operational RDKit")

    projected = chem.Mol(molecule)
    try:
        chem.Kekulize(projected, clearAromaticFlags=False)
    except (ValueError, RuntimeError) as error:
        raise ValueError("native graph has no compatible Kekule form") from error

    if not bond_order_constraints:
        return projected

    constraints: dict[int, int] = {}
    covalent_types = {
        chem.BondType.SINGLE,
        chem.BondType.DOUBLE,
        chem.BondType.TRIPLE,
        chem.BondType.QUADRUPLE,
    }
    for (first, second), order in bond_order_constraints.items():
        if (
            type(first) is not int
            or type(second) is not int
            or not 0 <= first < molecule.GetNumAtoms()
            or not 0 <= second < molecule.GetNumAtoms()
            or first == second
            or type(order) is not int
            or order not in (1, 2, 3, 4)
        ):
            raise BondOrderConstraintError("invalid native bond order constraint")
        bond = projected.GetBondBetweenAtoms(first, second)
        if bond is None or bond.HasQuery() or bond.GetBondType() not in covalent_types:
            raise BondOrderConstraintError(
                "bond order constraint lacks a resolved covalent edge"
            )
        index = bond.GetIdx()
        if index in constraints and constraints[index] != order:
            raise BondOrderConstraintError("conflicting native bond order constraints")
        if bond.GetIsAromatic():
            if order not in (1, 2):
                raise BondOrderConstraintError(
                    "aromatic order constraints must be single or double"
                )
        elif bond.GetBondTypeAsDouble() != order:
            raise BondOrderConstraintError(
                "bond order constraint conflicts with non-aromatic chemistry"
            )
        constraints[index] = order
    if all(
        projected.GetBondWithIdx(index).GetBondTypeAsDouble() == order
        for index, order in constraints.items()
    ):
        return projected

    constrained = chem.Mol(molecule)
    for index, order in constraints.items():
        bond = constrained.GetBondWithIdx(index)
        if bond.GetIsAromatic():
            bond.SetBondType(
                chem.BondType.SINGLE if order == 1 else chem.BondType.DOUBLE
            )
            # RDKit only reassigns aromatic-flagged edges during kekulization.
            bond.SetIsAromatic(False)
    try:
        chem.Kekulize(constrained, clearAromaticFlags=False)
    except (ValueError, RuntimeError) as error:
        raise BondOrderConstraintError(
            "source orders have no compatible Kekule form"
        ) from error
    for before, after in zip(projected.GetBonds(), constrained.GetBonds(), strict=True):
        after.SetIsAromatic(before.GetIsAromatic())
    if any(
        constrained.GetBondWithIdx(index).GetBondTypeAsDouble() != order
        for index, order in constraints.items()
    ) or _aromatic_order_sums(constrained) != _aromatic_order_sums(projected):
        raise BondOrderConstraintError(
            "source orders cannot preserve aromatic atom valence"
        )
    return constrained


def _aromatic_order_sums(molecule: "Mol") -> tuple[float, ...]:
    sums = [0.0] * molecule.GetNumAtoms()
    for bond in molecule.GetBonds():
        if bond.GetIsAromatic():
            order = bond.GetBondTypeAsDouble()
            sums[bond.GetBeginAtomIdx()] += order
            sums[bond.GetEndAtomIdx()] += order
    return tuple(sums)
