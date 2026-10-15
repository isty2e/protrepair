"""RDKit graph declarations and independently observed pose stereochemistry."""

from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.errors import RdkitUnavailableError

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
