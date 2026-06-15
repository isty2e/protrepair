"""Built-in standard-component definitions for baseline protein repair."""

from functools import lru_cache

from protrepair.chemistry.component.graph import (
    BondDefinition,
    ChemicalComponentDefinition,
    ForceFieldAtomParams,
    TetrahedralCenterSemantics,
)
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import (
    HydrogenSemantics,
    RotatableHydrogenKind,
)
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.hydrogen_plans import (
    STANDARD_HYDROGEN_PLANS,
    TRP_WITHOUT_BACKBONE_HYDROGEN_PLAN,
)
from protrepair.chemistry.standard.forcefield import FORCEFIELD_PARAMETERS
from protrepair.chemistry.standard.heavy_atom_programs import (
    STANDARD_HEAVY_ATOM_SEMANTICS,
)

STANDARD_COMPONENT_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("N", "CA", "C", "O", "CB", "OXT"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "OXT"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "ND2", "OD1", "OXT"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG", "OXT"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE2", "OE1", "OXT"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT"),
    "GLY": ("N", "CA", "C", "O", "OXT"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "CD2", "ND1", "CE1", "NE2", "OXT"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OXT"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD", "OXT"),
    "SER": ("N", "CA", "C", "O", "CB", "OG", "OXT"),
    "THR": ("N", "CA", "C", "O", "CB", "CG2", "OG1", "OXT"),
    "TRP": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "NE1",
        "CZ2",
        "CZ3",
        "CH2",
        "OXT",
    ),
    "TYR": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "OXT",
    ),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT"),
}
STANDARD_COMPONENT_BOND_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "ALA": (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"), ("CA", "CB")),
    "ARG": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "NE"),
        ("NE", "CZ"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
    ),
    "ASN": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "OD1"),
        ("CG", "ND2"),
    ),
    "ASP": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "OD1"),
        ("CG", "OD2"),
    ),
    "CYS": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "SG"),
    ),
    "GLN": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "NE2"),
    ),
    "GLU": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "OE2"),
    ),
    "GLY": (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT")),
    "HIS": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "ND1"),
        ("CG", "CD2"),
        ("ND1", "CE1"),
        ("CE1", "NE2"),
        ("NE2", "CD2"),
    ),
    "ILE": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CG1", "CD1"),
    ),
    "LEU": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
    ),
    "LYS": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "CE"),
        ("CE", "NZ"),
    ),
    "MET": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "SD"),
        ("SD", "CE"),
    ),
    "PHE": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    ),
    "PRO": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "N"),
    ),
    "SER": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "OG"),
    ),
    "THR": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "OG1"),
        ("CB", "CG2"),
    ),
    "TRP": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "NE1"),
        ("NE1", "CE2"),
        ("CE2", "CD2"),
        ("CD2", "CE3"),
        ("CE2", "CZ2"),
        ("CE3", "CZ3"),
        ("CZ2", "CH2"),
        ("CZ3", "CH2"),
    ),
    "TYR": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    ),
    "VAL": (
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
        ("CB", "CG1"),
        ("CB", "CG2"),
    ),
}
HISTIDINE_ALIASES: tuple[str, ...] = ("HSD", "HSE", "HIE", "HSP")

STANDARD_HYDROGEN_SEMANTICS: dict[str, HydrogenSemantics] = {
    component_id: HydrogenSemantics(plan_with_backbone=plan)
    for component_id, plan in STANDARD_HYDROGEN_PLANS.items()
}
STANDARD_HYDROGEN_SEMANTICS["CYS"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.CYS
)
STANDARD_HYDROGEN_SEMANTICS["SER"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.SER
)
STANDARD_HYDROGEN_SEMANTICS["THR"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.THR
)
STANDARD_HYDROGEN_SEMANTICS["TYR"] = HydrogenSemantics(
    rotatable_kind=RotatableHydrogenKind.TYR
)
STANDARD_HYDROGEN_SEMANTICS["TRP"] = HydrogenSemantics(
    plan_with_backbone=STANDARD_HYDROGEN_PLANS["TRP"],
    plan_without_backbone=TRP_WITHOUT_BACKBONE_HYDROGEN_PLAN,
)

STANDARD_TETRAHEDRAL_CENTER_SEMANTICS: dict[
    str, tuple[TetrahedralCenterSemantics, ...]
] = {
    "ILE": (
        TetrahedralCenterSemantics(
            center_atom_name="CB",
            ordered_neighbor_atom_names=("CA", "CG1", "CG2"),
            expected_orientation_sign=1,
        ),
    ),
    "THR": (
        TetrahedralCenterSemantics(
            center_atom_name="CB",
            ordered_neighbor_atom_names=("CA", "OG1", "CG2"),
            expected_orientation_sign=1,
        ),
    ),
}


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def build_standard_component_library() -> ComponentLibrary:
    """Return the built-in library for the 20 standard residues."""

    templates: dict[str, ResidueTemplate] = {}
    for component_id, atom_names in STANDARD_COMPONENT_ATOMS.items():
        aliases = HISTIDINE_ALIASES if component_id == "HIS" else ()
        heavy_atom_semantics = STANDARD_HEAVY_ATOM_SEMANTICS.get(component_id)
        hydrogen_semantics = STANDARD_HYDROGEN_SEMANTICS.get(component_id)
        tetrahedral_centers = STANDARD_TETRAHEDRAL_CENTER_SEMANTICS.get(
            component_id,
            (),
        )

        definition = ChemicalComponentDefinition(
            component_id=component_id,
            atom_names=atom_names,
            bonds=tuple(
                BondDefinition(atom_name_1=atom_name_1, atom_name_2=atom_name_2)
                for atom_name_1, atom_name_2 in STANDARD_COMPONENT_BOND_PAIRS[
                    component_id
                ]
            ),
            aliases=aliases,
        )
        templates[component_id] = ResidueTemplate(
            definition=definition,
            backbone_family_component_id=component_id,
            forcefield_parameters={
                atom_name: ForceFieldAtomParams(
                    charge=params[0],
                    sigma_nm=params[1],
                    epsilon_kj_mol=params[2],
                )
                for atom_name, params in FORCEFIELD_PARAMETERS[component_id].items()
            },
            heavy_atom_semantics=heavy_atom_semantics,
            hydrogen_semantics=hydrogen_semantics,
            tetrahedral_centers=tetrahedral_centers,
        )

    return ComponentLibrary(templates=templates)
