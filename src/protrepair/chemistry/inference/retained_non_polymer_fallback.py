"""RDKit-backed inference for retained non-polymer fallback chemistry."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.errors import RdkitUnavailableError
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Atom, Mol


def retained_non_polymer_rdkit_fallback_expected_hydrogen_atom_names(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> tuple[str, ...]:
    """Return preferred expected hydrogen names from RDKit heuristic chemistry."""

    hydrogenated_molecule = retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )
    inferred_hydrogen_count = sum(
        1 for atom in hydrogenated_molecule.GetAtoms() if atom.GetAtomicNum() == 1
    )
    present_hydrogen_atom_names = tuple(
        atom_site.name
        for atom_site in residue_site.atom_sites
        if atom_site.element == "H"
    )
    return preferred_retained_non_polymer_expected_hydrogen_atom_names(
        inferred_hydrogen_count=inferred_hydrogen_count,
        present_hydrogen_atom_names=present_hydrogen_atom_names,
    )


def retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> "Mol":
    """Return one RDKit hydrogenated molecule inferred from the current pose."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer RDKit fallback hydrogenation requires the "
            "optional rdkit dependency"
        )

    heavy_atom_molecule = _retained_non_polymer_rdkit_pose_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )
    return Chem.AddHs(heavy_atom_molecule, addCoords=True)


def retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> tuple[BondDefinition, ...]:
    """Return heavy-atom bond definitions inferred from one retained pose."""

    heavy_atom_molecule = _retained_non_polymer_rdkit_pose_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )
    return tuple(
        BondDefinition(
            atom_name_1=_rdkit_atom_name(
                heavy_atom_molecule.GetAtomWithIdx(bond.GetBeginAtomIdx())
            ),
            atom_name_2=_rdkit_atom_name(
                heavy_atom_molecule.GetAtomWithIdx(bond.GetEndAtomIdx())
            ),
            order=max(1, round(bond.GetBondTypeAsDouble())),
            aromatic=bond.GetIsAromatic(),
        )
        for bond in heavy_atom_molecule.GetBonds()
    )


def retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions(
    hydrogenated_molecule: "Mol",
) -> tuple[BondDefinition, ...]:
    """Return generated H-heavy bond definitions from an RDKit fallback molecule."""

    atom_names_by_index = _rdkit_fallback_atom_names_by_index(
        hydrogenated_molecule
    )
    return tuple(
        BondDefinition(
            atom_name_1=atom_names_by_index[begin_atom.GetIdx()],
            atom_name_2=atom_names_by_index[end_atom.GetIdx()],
            order=max(1, round(bond.GetBondTypeAsDouble())),
            aromatic=bond.GetIsAromatic(),
        )
        for bond in hydrogenated_molecule.GetBonds()
        for begin_atom, end_atom in (
            (
                hydrogenated_molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                hydrogenated_molecule.GetAtomWithIdx(bond.GetEndAtomIdx()),
            ),
        )
        if (begin_atom.GetAtomicNum() == 1) != (end_atom.GetAtomicNum() == 1)
    )


def retained_non_polymer_rdkit_fallback_supports_passive_context(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> bool:
    """Return whether fallback chemistry is coherent enough for passive context."""

    heavy_atom_molecule = _retained_non_polymer_rdkit_pose_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )
    if heavy_atom_molecule.GetNumAtoms() <= 1:
        return True

    if heavy_atom_molecule.GetNumBonds() <= 0:
        return False

    assert Chem is not None
    return len(Chem.GetMolFrags(heavy_atom_molecule)) == 1


def preferred_retained_non_polymer_expected_hydrogen_atom_names(
    *,
    inferred_hydrogen_count: int,
    present_hydrogen_atom_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return preferred expected hydrogen names for one inferred hydrogen count."""

    if inferred_hydrogen_count <= 0:
        return ()

    if (
        present_hydrogen_atom_names
        and len(present_hydrogen_atom_names) == inferred_hydrogen_count
    ):
        return present_hydrogen_atom_names

    return tuple(
        f"H{index:03d}" for index in range(1, inferred_hydrogen_count + 1)
    )


def _retained_non_polymer_rdkit_pose_molecule(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> "Mol":
    """Return one sanitized RDKit heavy-atom pose for one residue payload."""

    assert Chem is not None

    heavy_atom_pdb_block = _retained_non_polymer_heavy_atom_pdb_block(
        residue_site,
        residue_geometry,
    )
    pose_molecule = Chem.MolFromPDBBlock(
        heavy_atom_pdb_block,
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    if pose_molecule is None:
        raise ValueError(
            "RDKit could not parse retained non-polymer fallback heavy-atom pose "
            f"for {residue_site.residue_id.display_token()}"
        )

    normalized_formal_charges = (
        {}
        if formal_charge_by_atom_name is None
        else {
            atom_name.strip().upper(): formal_charge
            for atom_name, formal_charge in formal_charge_by_atom_name.items()
            if formal_charge is not None
        }
    )
    for atom in pose_molecule.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        if residue_info is None:
            continue

        atom_name = residue_info.GetName().strip().upper()
        formal_charge = normalized_formal_charges.get(atom_name)
        if formal_charge is None:
            continue

        atom.SetFormalCharge(formal_charge)

    Chem.SanitizeMol(pose_molecule)
    return pose_molecule


def _retained_non_polymer_heavy_atom_pdb_block(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
) -> str:
    """Return one heavy-atom-only PDB block for RDKit pose inference."""

    record_name = "HETATM" if residue_site.is_hetero else "ATOM  "
    residue_name = residue_site.component_id[:3].rjust(3)
    chain_id = residue_site.residue_id.chain_id[:1]
    insertion_code = (residue_site.residue_id.insertion_code or " ")[:1]
    pdb_lines: list[str] = []
    atom_serial = 1
    for atom_site in residue_site.atom_sites:
        if atom_site.element == "H":
            continue

        atom_geometry = residue_geometry.atom_geometry(atom_site.name)
        position = atom_geometry.position
        altloc = (atom_geometry.altloc or " ")[:1]
        occupancy = atom_geometry.occupancy
        b_factor = 0.0 if atom_geometry.b_factor is None else atom_geometry.b_factor
        pdb_lines.append(
            f"{record_name:<6}{atom_serial:5d} "
            f"{atom_site.name:>4}{altloc}{residue_name} {chain_id}"
            f"{residue_site.residue_id.seq_num:4d}{insertion_code}   "
            f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"{occupancy:6.2f}{b_factor:6.2f}          "
            f"{_pdb_element_field(atom_site.element)}"
        )
        atom_serial += 1

    pdb_lines.append("END")
    return "\n".join(pdb_lines) + "\n"


def _pdb_element_field(element: str) -> str:
    """Return one PDB-aligned element field."""

    normalized_element = element.strip()
    if len(normalized_element) >= 2:
        normalized_element = (
            normalized_element[0].upper() + normalized_element[1].lower()
        )
    else:
        normalized_element = normalized_element.upper()

    return normalized_element.rjust(2)


def _rdkit_atom_name(atom: "Atom") -> str:
    """Return the PDB atom name attached to one RDKit atom."""

    residue_info = atom.GetPDBResidueInfo()
    if residue_info is None:
        raise ValueError("RDKit fallback atom is missing PDB residue metadata")

    return residue_info.GetName().strip()


def _rdkit_fallback_atom_names_by_index(
    hydrogenated_molecule: "Mol",
) -> dict[int, str]:
    """Return fallback atom names matching RDKit-generated H append order."""

    atom_names_by_index: dict[int, str] = {}
    hydrogen_index = 1
    for atom in hydrogenated_molecule.GetAtoms():
        atom_index = atom.GetIdx()
        if atom.GetAtomicNum() == 1:
            atom_names_by_index[atom_index] = f"H{hydrogen_index:03d}"
            hydrogen_index += 1
            continue

        atom_names_by_index[atom_index] = _rdkit_atom_name(atom)

    return atom_names_by_index
