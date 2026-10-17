"""RDKit graph projections preserving covalent orders and dative direction."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.component.graph import (
    BondDefinition,
    CoordinationBondDefinition,
)
from protrepair.errors import BondOrderConstraintError, RdkitUnavailableError

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

AROMATIC_N_COORDINATION_METALS = frozenset(
    ("FE", "ZN", "CU", "CO", "NI", "MN", "CD", "HG")
)


@dataclass(frozen=True, slots=True)
class NativeBondProjection:
    """Joint named graph projection with disjoint physical relationship products.

    Parameters
    ----------
    covalent_bonds : tuple[BondDefinition, ...]
        Integral covalent orders with aromatic identity retained.
    coordination_bonds : tuple[CoordinationBondDefinition, ...]
        Oriented dative relationships, not covalent valence contributions.

    Raises
    ------
    ValueError
        A pair is repeated or has competing physical relationships.
    """

    covalent_bonds: tuple[BondDefinition, ...]
    coordination_bonds: tuple[CoordinationBondDefinition, ...]

    def __post_init__(self) -> None:
        pairs = tuple(
            tuple(sorted((bond.atom_name_1, bond.atom_name_2)))
            for bond in self.covalent_bonds
        ) + tuple(bond.endpoint_pair() for bond in self.coordination_bonds)
        if any(first == second for first, second in pairs) or len(set(pairs)) != len(
            pairs
        ):
            raise ValueError("native graph projection has competing endpoint pairs")


def require_aromatic_nitrogen_donor(molecule: "Mol", donor_index: int) -> None:
    """Require a closed-shell aromatic N lone-pair donor in the native graph.

    Parameters
    ----------
    molecule : Mol
        Native chemistry with an initialized atom-property cache.
    donor_index : int
        Native index of the donor, not a constitution atom slot.

    Raises
    ------
    ValueError
        The donor is protonated, substituted, radical, or outside neutral/anionic
        aromatic N chemistry with exactly two covalently bonded carbon neighbors.
    RdkitUnavailableError
        RDKit is not operational.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError(
            "coordination validation requires operational RDKit"
        )
    donor = molecule.GetAtomWithIdx(donor_index)
    neighbors = tuple(
        bond.GetOtherAtom(donor)
        for bond in donor.GetBonds()
        if bond.GetBondType() != chem.BondType.DATIVE
    )
    if (
        donor.GetAtomicNum() != 7
        or donor.GetFormalCharge() not in {0, -1}
        or not donor.GetIsAromatic()
        or donor.GetNumRadicalElectrons() != 0
        or donor.GetTotalNumHs(includeNeighbors=True) != 0
        or len(neighbors) != 2
        or any(atom.GetAtomicNum() != 6 for atom in neighbors)
    ):
        raise ValueError(
            "coordination requires an unprotonated aromatic nitrogen donor"
        )


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


def covalent_bond_definitions_from_molecule(
    molecule: "Mol",
    *,
    atom_names_by_index: Mapping[int, str],
    bond_order_constraints: Mapping[tuple[str, str], int] | None = None,
) -> tuple[BondDefinition, ...]:
    """Project a resolved covalent graph without rounding aromatic bond orders.

    Parameters
    ----------
    molecule : Mol
        Resolved native covalent chemistry; the input is not modified.
    atom_names_by_index : Mapping[int, str]
        Complete native-index to canonical atom-name correspondence.
    bond_order_constraints : Mapping[tuple[str, str], int] or None
        Reported orders in canonical atom-name identity, independent of orientation.

    Returns
    -------
    tuple[BondDefinition, ...]
        Integral covalent orders in native edge order, retaining aromatic identity.
        The chosen resonance form need not be identical across native versions.

    Raises
    ------
    ValueError
        Names are missing, colliding or blank, the graph cannot be kekulized, or
        a query or non-covalent bond cannot be represented by BondDefinition.
    BondOrderConstraintError
        Reported orders cannot preserve the resolved chemistry.
    RdkitUnavailableError
        RDKit is not operational.
    """

    if Chem is None:
        raise RdkitUnavailableError("graph projection requires operational RDKit")
    if any(bond.GetBondType() == Chem.BondType.DATIVE for bond in molecule.GetBonds()):
        raise ValueError("native bond type is not resolved covalent chemistry")
    return native_bond_projection_from_molecule(
        molecule,
        atom_names_by_index=atom_names_by_index,
        bond_order_constraints=bond_order_constraints,
    ).covalent_bonds


def native_bond_projection_from_molecule(
    molecule: "Mol",
    *,
    atom_names_by_index: Mapping[int, str],
    bond_order_constraints: Mapping[tuple[str, str], int] | None = None,
) -> NativeBondProjection:
    """Project covalent and oriented dative relationships without losing either.

    Parameters
    ----------
    molecule : Mol
        Resolved native chemistry; the input is not modified.
    atom_names_by_index : Mapping[int, str]
        Complete native-index to canonical atom-name correspondence.
    bond_order_constraints : Mapping[tuple[str, str], int] or None
        Reported integral covalent orders keyed by named endpoints.

    Returns
    -------
    NativeBondProjection
        Joint covalent and coordination products in their native edge order.
        Dative direction is retained; this projection does not certify a donor
        motif or force-field support.

    Raises
    ------
    ValueError
        Names, queries or native relationship types cannot be represented.
    BondOrderConstraintError
        Reported orders cannot preserve resolved covalent chemistry.
    RdkitUnavailableError
        RDKit is not operational.
    """
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError("graph projection requires operational RDKit")

    if set(atom_names_by_index) != set(range(molecule.GetNumAtoms())):
        raise ValueError("native graph requires a complete atom-name mapping")
    names = tuple(
        atom_names_by_index[index].strip().upper()
        for index in range(molecule.GetNumAtoms())
    )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("native graph atom names must be nonblank and unique")
    if any(atom.HasQuery() for atom in molecule.GetAtoms()):
        raise ValueError("query chemistry is not a resolved covalent graph")

    order_by_type = {
        chem.BondType.SINGLE: 1,
        chem.BondType.DOUBLE: 2,
        chem.BondType.TRIPLE: 3,
        chem.BondType.QUADRUPLE: 4,
    }
    supported_types = frozenset(
        (*order_by_type, chem.BondType.AROMATIC, chem.BondType.DATIVE)
    )
    if any(
        bond.HasQuery() or bond.GetBondType() not in supported_types
        for bond in molecule.GetBonds()
    ):
        raise ValueError("native bond type is not resolved covalent chemistry")

    native_constraints = {}
    indices = {name: index for index, name in enumerate(names)}
    for (first, second), order in (bond_order_constraints or {}).items():
        if first not in indices or second not in indices:
            raise BondOrderConstraintError(
                "bond order constraint refers to an absent atom name"
            )
        native_constraints[(indices[first], indices[second])] = order
    projected = kekulized_molecule(molecule, bond_order_constraints=native_constraints)
    definitions: list[BondDefinition] = []
    coordination: list[CoordinationBondDefinition] = []
    for bond in projected.GetBonds():
        if bond.GetBondType() == chem.BondType.DATIVE:
            coordination.append(
                CoordinationBondDefinition(
                    donor_atom_name=names[bond.GetBeginAtomIdx()],
                    acceptor_atom_name=names[bond.GetEndAtomIdx()],
                )
            )
            continue
        order = order_by_type.get(bond.GetBondType())
        if order is None:
            raise ValueError("native aromatic bond has no integral covalent order")
        definitions.append(
            BondDefinition(
                atom_name_1=names[bond.GetBeginAtomIdx()],
                atom_name_2=names[bond.GetEndAtomIdx()],
                order=order,
                aromatic=bond.GetIsAromatic(),
            )
        )

    return NativeBondProjection(tuple(definitions), tuple(coordination))
