"""Convert explicit monomer CIF chemistry and targets into a state reference."""

import math
from dataclasses import replace
from hashlib import sha256

import gemmi

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.restraint.reference import ComponentRestraintReference
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.structure.element import ElementIdentity


def _positive_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("monomer targets and ESDs must be finite and positive")
    return number


def _validated_graph(
    component: IdealizedComponent, *, normalize_aromatic: bool
) -> IdealizedComponent:
    try:
        from rdkit import Chem
    except ImportError as error:
        raise RdkitUnavailableError(
            "monomer reference validation requires operational RDKit"
        ) from error

    molecule = Chem.RWMol()
    indices: dict[str, int] = {}
    for record in component.atoms:
        atom = Chem.Atom(ElementIdentity(record.element).chemical_symbol.title())
        atom.SetFormalCharge(record.formal_charge)
        atom.SetNoImplicit(True)
        indices[record.atom_name] = molecule.AddAtom(atom)
    orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    seen: set[frozenset[str]] = set()
    for bond in component.bonds:
        pair = frozenset((bond.atom_name_1, bond.atom_name_2))
        if len(pair) != 2 or pair in seen:
            raise ValueError("monomer bonds require distinct, unrepeated endpoints")
        seen.add(pair)
        molecule.AddBond(
            indices[bond.atom_name_1],
            indices[bond.atom_name_2],
            Chem.BondType.AROMATIC if bond.aromatic else orders[bond.order],
        )
    checked = molecule.GetMol()
    Chem.SanitizeMol(checked)
    if any(atom.GetNumRadicalElectrons() for atom in checked.GetAtoms()):
        raise ValueError("monomer references require closed-shell chemical states")
    if not normalize_aromatic:
        return component

    names = tuple(atom.atom_name for atom in component.atoms)
    aromatic_pairs = {
        frozenset((names[bond.GetBeginAtomIdx()], names[bond.GetEndAtomIdx()]))
        for bond in checked.GetBonds()
        if bond.GetIsAromatic()
    }
    return replace(
        component,
        bonds=tuple(
            replace(
                bond,
                aromatic=frozenset((bond.atom_name_1, bond.atom_name_2))
                in aromatic_pairs,
            )
            for bond in component.bonds
        ),
    )


def read_monomer_chemistry(
    text: str,
    *,
    component_id: str,
    source: str,
    normalize_aromatic: bool = False,
) -> tuple[gemmi.cif.Block, IdealizedComponent, str]:
    """Read one explicit monomer graph and bind provenance to the loaded text.

    Parameters
    ----------
    text : str
        Loaded monomer-library CIF contents.
    component_id : str
        Canonical name of the requested component block.
    source : str
        Stable provenance identity supplied by the caller.
    normalize_aromatic : bool, default=False
        Use validated aromaticity rather than the input bond flags.

    Returns
    -------
    tuple[gemmi.cif.Block, IdealizedComponent, str]
        Source block, validated closed-shell graph, and source with text digest.

    Raises
    ------
    ValueError
        The reference has no matching component or a malformed chemical graph.
    RdkitUnavailableError
        The native graph validator cannot be imported.
    """

    source = source.strip()
    if not source:
        raise ValueError("monomer references require a source identity")

    try:
        document = gemmi.cif.read_string(text)
    except RuntimeError as error:
        raise ValueError("malformed monomer CIF") from error
    block = document.find_block("comp_" + component_id)
    if block is None:
        raise ValueError("monomer CIF has no block for the fallback component")
    atoms = tuple(
        IdealizedComponentAtom(row.str(0), row.str(1), int(row.str(2)))
        for row in block.find(
            [
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
            ]
        )
    )
    orders = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
    bonds = []
    for row in block.find(
        [
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.type",
            "_chem_comp_bond.aromatic",
        ]
    ):
        if row.str(2).upper() not in orders or row.str(3).lower() not in {"y", "n"}:
            raise ValueError(
                "monomer references require explicit integral and aromatic bond facts"
            )
        bonds.append(
            BondDefinition(
                row.str(0),
                row.str(1),
                orders[row.str(2).upper()],
                row.str(3).lower() == "y",
            )
        )
    component = _validated_graph(
        IdealizedComponent(component_id, None, atoms, tuple(bonds)),
        normalize_aromatic=normalize_aromatic,
    )
    return (
        block,
        component,
        source + "#text-sha256=" + sha256(text.encode("utf-8")).hexdigest(),
    )


def component_reference_from_monomer(
    text: str,
    *,
    fallback: ResidueRestraintTemplate,
    atom_names: frozenset[str],
    boundary_atom_names: frozenset[str],
    source: str,
) -> ComponentRestraintReference:
    """Read explicit targets and ESDs without deriving them from coordinates.

    Parameters
    ----------
    text : str
        Loaded monomer-library CIF contents, not a path or remote locator.
    fallback : ResidueRestraintTemplate
        Existing dictionary whose target identities must remain unchanged.
    atom_names, boundary_atom_names : frozenset[str]
        Complete matching scope and its non-writable heavy boundary.
    source : str
        Stable provenance identity; a digest of the supplied UTF-8 text is added.
        This is distinct from an upstream raw-file checksum.

    Returns
    -------
    ComponentRestraintReference
        Closed-shell graph with eligible explicit heavy target replacements.

    Raises
    ------
    ValueError
        The supplied reference configuration is malformed or incompatible.
    """
    block, component, provenance = read_monomer_chemistry(
        text, component_id=fallback.component_id, source=source
    )

    scope = frozenset(name.strip().upper() for name in atom_names)
    boundary = frozenset(name.strip().upper() for name in boundary_atom_names)
    writable = scope - boundary
    bond_targets = []
    bond_rows = block.find(
        [
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_dist",
            "_chem_comp_bond.value_dist_esd",
        ]
    )
    if not bond_rows and any(
        {t.atom_name_1, t.atom_name_2} <= writable for t in fallback.bond_targets
    ):
        raise ValueError("monomer reference lacks the required bond-target loop")
    for row in bond_rows:
        names = row.str(0).upper(), row.str(1).upper()
        if set(names) <= writable and fallback.bond_target(*names) is not None:
            bond_targets.append(
                BondRestraintTarget(
                    *names, _positive_number(row.str(2)), _positive_number(row.str(3))
                )
            )
    expected_angles = {
        t.canonical_triplet()
        for t in fallback.angle_targets
        if {t.atom_name_1, t.center_atom_name, t.atom_name_2} <= writable
    }
    reference = ComponentRestraintReference(
        component,
        scope,
        boundary,
        tuple(bond_targets),
        (),
        provenance,
    )
    proposed_angles = []
    angle_rows = block.find(
        [
            "_chem_comp_angle.atom_id_1",
            "_chem_comp_angle.atom_id_2",
            "_chem_comp_angle.atom_id_3",
            "_chem_comp_angle.value_angle",
            "_chem_comp_angle.value_angle_esd",
        ]
    )
    if expected_angles and not angle_rows:
        raise ValueError("monomer reference lacks the required angle-target loop")
    for row in angle_rows:
        names = row.str(0).upper(), row.str(1).upper(), row.str(2).upper()
        key = names[1], *sorted((names[0], names[2]))
        if key in expected_angles and reference.can_replace_angle_center(names[1]):
            proposed_angles.append(
                AngleRestraintTarget(
                    *names, _positive_number(row.str(3)), _positive_number(row.str(4))
                )
            )
    available = {t.canonical_triplet() for t in proposed_angles}
    incomplete_centers = {center for center, left, right in expected_angles - available}
    reference = replace(
        reference,
        angle_targets=tuple(
            t for t in proposed_angles if t.center_atom_name not in incomplete_centers
        ),
    )
    reference.validate_targets(fallback)
    return reference
