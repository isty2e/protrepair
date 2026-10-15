"""Native RDKit emission without coordinate-text chemistry inference."""

from types import MappingProxyType

from rdkit import Chem
from rdkit.Chem import rdqueries
from rdkit.Chem.rdchem import BondType, Mol, RWMol

from protrepair.chemistry.inference.rdkit_graph import kekulized_molecule
from protrepair.errors import RdkitProjectionError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondRelationshipType, TopologyBond

_COVALENT_BOND_TYPES = MappingProxyType(
    {
        1: BondType.SINGLE,
        2: BondType.DOUBLE,
        3: BondType.TRIPLE,
        4: BondType.QUADRUPLE,
    }
)


def to_rdkit_molecule(structure: ProteinStructure) -> Mol:
    """Project selected canonical chemistry into a sanitized, independent molecule.

    Parameters
    ----------
    structure : ProteinStructure
        Structure with resolved chemical bonds and explicit H. Preparation and
        metal charge selection must already have been performed.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule with native indices equal to current canonical atom indices,
        a full-precision 3D conformer, isotope and residue identity, and stereo
        assigned from coordinates. Nonmetal unknown charge is represented as
        zero with ``protrepair_formal_charge_known=False`` on the atom, not as
        new canonical evidence. Omitted hydrogen-bond interactions are counted
        in ``protrepair_omitted_hydrogen_bonds`` on the molecule.

    Raises
    ------
    TypeError
        The input is not a canonical ProteinStructure.
    RdkitProjectionError
        Chemistry is unresolved or unsupported, a metal charge is unknown,
        sanitation changes the selected graph or charges, or the result needs
        additional H or radical-electron assignments.

    Notes
    -----
    Typed coordination is directed from the nonmetal endpoint to the metal
    using RDKit's M-atom grammar. Chemical representation does not establish
    oxidation, spin, pKa, force-field support, or geometric quality. This
    function does not prepare chemistry, optimize coordinates or parse PDB.
    """
    if not isinstance(structure, ProteinStructure):
        raise TypeError("native RDKit projection requires a ProteinStructure")

    try:
        molecule = RWMol()
        _add_atoms(structure, molecule)
        omitted_hydrogen_bonds = 0
        for bond in structure.topology.bonds:
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                omitted_hydrogen_bonds += 1
                continue
            _add_bond(molecule, bond)

        native = molecule.GetMol()
        Chem.SanitizeMol(native)
        declared_doubles = {
            (bond.atom_index_1.value, bond.atom_index_2.value): 2
            for bond in structure.topology.bonds
            if bond.aromatic and bond.order == 2
        }
        if declared_doubles:
            kekulized_molecule(native, bond_order_constraints=declared_doubles)
        Chem.AssignStereochemistryFrom3D(native, confId=0, replaceExistingTags=True)
        Chem.AssignStereochemistry(native, cleanIt=True, force=True)
        _verify_projection(structure, native)
        native.SetIntProp("protrepair_omitted_hydrogen_bonds", omitted_hydrogen_bonds)
        return native
    except (RuntimeError, ValueError, OverflowError) as error:
        raise RdkitProjectionError(
            f"native RDKit projection failed: {error}"
        ) from error


def _add_atoms(structure: ProteinStructure, molecule: RWMol) -> None:
    conformer = Chem.Conformer(structure.geometry.atom_count())
    conformer.Set3D(True)
    metal_query = rdqueries.MAtomQueryAtom()
    for value, site in enumerate(structure.constitution.atom_slots):
        index = AtomIndex(value)
        geometry = structure.geometry.atom_geometry(index)
        ref = structure.constitution.atom_ref_at(index)
        residue = structure.constitution.residue_site_at(
            structure.constitution.residue_index_for_atom_index(index)
        )
        atom = Chem.Atom(site.element_identity.chemical_symbol.title())
        charge = structure.topology.formal_charge(index)
        if charge is None and metal_query.Match(atom):
            raise RdkitProjectionError(
                f"unknown metal charge at {ref.display_token()}; "
                "select chemistry before native projection"
            )
        atom.SetFormalCharge(0 if charge is None else charge)
        atom.SetBoolProp("protrepair_formal_charge_known", charge is not None)
        atom.SetBoolProp("protrepair_b_factor_known", geometry.b_factor is not None)
        atom.SetProp("protrepair_source_element", site.element_identity.source_symbol)
        atom.SetNoImplicit(True)
        isotope = site.element_identity.isotope_mass_number
        if isotope is not None:
            atom.SetIsotope(isotope)

        info = Chem.AtomPDBResidueInfo(site.name.ljust(4))
        info.SetSerialNumber(value + 1)
        info.SetResidueName(residue.component_id)
        info.SetChainId(ref.residue_id.chain_id)
        info.SetResidueNumber(ref.residue_id.seq_num)
        info.SetInsertionCode(ref.residue_id.insertion_code or "")
        info.SetIsHeteroAtom(residue.is_hetero)
        info.SetAltLoc(geometry.altloc or "")
        info.SetOccupancy(geometry.occupancy)
        info.SetTempFactor(0.0 if geometry.b_factor is None else geometry.b_factor)
        atom.SetMonomerInfo(info)
        molecule.AddAtom(atom)
        conformer.SetAtomPosition(value, tuple(geometry.position))

    molecule.AddConformer(conformer, assignId=True)


def _add_bond(molecule: RWMol, bond: TopologyBond) -> None:
    first, second = (index.value for index in bond.endpoint_pair())
    if bond.relationship_type is BondRelationshipType.METAL_COORDINATION:
        if bond.aromatic:
            raise RdkitProjectionError("aromatic coordination is unsupported")
        metal_query = rdqueries.MAtomQueryAtom()
        first_is_metal = metal_query.Match(molecule.GetAtomWithIdx(first))
        second_is_metal = metal_query.Match(molecule.GetAtomWithIdx(second))
        if first_is_metal == second_is_metal:
            raise RdkitProjectionError(
                f"coordination pair {first, second} requires exactly one metal"
            )
        if first_is_metal:
            first, second = second, first
        bond_type = BondType.DATIVE
    elif bond.relationship_type in {
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
    }:
        if bond.order is None:
            raise RdkitProjectionError(f"unresolved bond order at pair {first, second}")
        if bond.aromatic and bond.order not in {1, 2}:
            raise RdkitProjectionError(
                f"aromatic pair {first, second} has incompatible order {bond.order}"
            )
        bond_type = _COVALENT_BOND_TYPES.get(bond.order)
        if bond_type is None:
            raise RdkitProjectionError(
                f"unsupported bond order {bond.order} at pair {first, second}"
            )
        if bond.aromatic:
            bond_type = BondType.AROMATIC
    else:
        raise RdkitProjectionError(f"unresolved relationship at pair {first, second}")

    molecule.AddBond(first, second, bond_type)
    native = molecule.GetBondBetweenAtoms(first, second)
    native.SetProp("protrepair_relationship_type", bond.relationship_type.value)
    native.SetProp("protrepair_bond_provenance", bond.provenance.value)
    native.SetBoolProp("protrepair_bond_order_known", bond.order is not None)
    if bond.order is not None:
        native.SetIntProp("protrepair_bond_order", bond.order)
    native.SetBoolProp("protrepair_aromatic", bond.aromatic)


def _verify_projection(structure: ProteinStructure, molecule: Mol) -> None:
    if molecule.GetNumAtoms() != structure.geometry.atom_count():
        raise RdkitProjectionError("sanitation changed the canonical atom inventory")
    metal_query = rdqueries.MAtomQueryAtom()
    for value in range(molecule.GetNumAtoms()):
        atom = molecule.GetAtomWithIdx(value)
        index = AtomIndex(value)
        charge = structure.topology.formal_charge(index)
        expected_charge = 0 if charge is None else charge
        if atom.GetFormalCharge() != expected_charge:
            ref = structure.constitution.atom_ref_at(index)
            raise RdkitProjectionError(
                f"sanitation changed formal charge at {ref.display_token()}: "
                f"{expected_charge} -> {atom.GetFormalCharge()}"
            )
        if atom.GetNumExplicitHs() or atom.GetNumImplicitHs():
            raise RdkitProjectionError(
                f"sanitation introduced H without canonical atom sites at index {value}"
            )
        if atom.GetNumRadicalElectrons():
            raise RdkitProjectionError(
                "native projection requires unselected radical chemistry "
                f"at index {value}"
            )

    expected_count = 0
    for bond in structure.topology.bonds:
        if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
            continue
        expected_count += 1
        first, second = (index.value for index in bond.endpoint_pair())
        native = molecule.GetBondBetweenAtoms(first, second)
        if native is None:
            raise RdkitProjectionError(
                f"sanitation removed bond at pair {first, second}"
            )
        if bond.relationship_type is BondRelationshipType.METAL_COORDINATION:
            if native.GetBondType() != BondType.DATIVE or not metal_query.Match(
                native.GetEndAtom()
            ):
                raise RdkitProjectionError(
                    f"sanitation changed coordination at pair {first, second}"
                )
        elif native.GetBondType() == BondType.AROMATIC:
            if bond.order not in {1, 2}:
                raise RdkitProjectionError(
                    f"sanitation changed bond order at pair {first, second}"
                )
        elif (
            bond.order is None
            or bond.aromatic
            or native.GetBondType() != _COVALENT_BOND_TYPES[bond.order]
        ):
            raise RdkitProjectionError(
                f"sanitation changed bond chemistry at pair {first, second}"
            )

    if molecule.GetNumBonds() != expected_count:
        raise RdkitProjectionError("sanitation changed the chemical bond inventory")
