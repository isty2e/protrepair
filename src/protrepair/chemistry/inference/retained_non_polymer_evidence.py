"""RDKit-backed inference for retained non-polymer chemistry evidence."""

from functools import lru_cache
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.inference.rdkit_graph import (
    has_declared_stereochemistry,
    require_compatible_pose_stereochemistry,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.structure.geometry import ResidueGeometry

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

_EVIDENCE_TEMPLATE_CACHE_MAXSIZE = 256


def require_retained_non_polymer_evidence_stereochemistry(
    evidence: RetainedNonPolymerChemistryEvidence,
    geometry: ResidueGeometry,
    *,
    hydrogen_attachments: tuple[tuple[str, str], ...] = (),
) -> None:
    """Compare declared mapped stereo with confirmed source geometry.

    Parameters
    ----------
    evidence : RetainedNonPolymerChemistryEvidence
        Selected chemical graph and its ordered heavy-atom names.
    geometry : ResidueGeometry
        Current named coordinates, without inferred connectivity.
    hydrogen_attachments : tuple[tuple[str, str], ...], optional
        Actual H-name/parent-name pairs. Complete observed parent H sets are
        included in the pose rather than replaced by implicit H geometry.

    Raises
    ------
    ValueError
        Mapping or confirmed geometric stereo contradicts the evidence.
    RdkitUnavailableError
        Native interpretation is unavailable.
    """
    template = template_without_hydrogens(evidence.smiles)
    if not has_declared_stereochemistry(template):
        return
    names = _evidence_heavy_atom_names_by_index(
        template, heavy_atom_names=evidence.heavy_atom_names
    )
    assert Chem is not None
    hydrogens_by_parent: dict[str, list[str]] = {}
    for hydrogen, parent in hydrogen_attachments:
        hydrogens_by_parent.setdefault(parent, []).append(hydrogen)
    complete_parents = tuple(
        index
        for index, name in names.items()
        if name in hydrogens_by_parent
        and len(hydrogens_by_parent[name])
        == template.GetAtomWithIdx(index).GetTotalNumHs()
    )
    if complete_parents:
        template = Chem.AddHs(template, onlyOnAtoms=complete_parents)
        for atom in template.GetAtoms():
            if atom.GetIdx() not in names:
                parent = names[atom.GetNeighbors()[0].GetIdx()]
                names[atom.GetIdx()] = hydrogens_by_parent[parent].pop(0)
    conformer = Chem.Conformer(template.GetNumAtoms())
    conformer.Set3D(True)
    for index, name in names.items():
        if not geometry.has_atom(name):
            raise ValueError(
                "retained non-polymer evidence mapping has no coordinate "
                f"for atom {name}"
            )
        conformer.SetAtomPosition(index, tuple(geometry.position(name)))
    template.RemoveAllConformers()
    template.AddConformer(conformer, assignId=True)
    require_compatible_pose_stereochemistry(template)


def retained_non_polymer_evidence_heavy_atom_elements(
    evidence: RetainedNonPolymerChemistryEvidence,
) -> tuple[str, ...]:
    """Return heavy-atom elements in evidence-template order."""

    heavy_template = template_without_hydrogens(evidence.smiles)
    if heavy_template.GetNumAtoms() != len(evidence.heavy_atom_names):
        raise ValueError(
            "retained non-polymer chemistry evidence heavy_atom_names must match "
            "the number of non-hydrogen atoms implied by the evidence smiles"
        )

    return tuple(atom.GetSymbol().upper() for atom in heavy_template.GetAtoms())


def retained_non_polymer_evidence_expected_hydrogen_atom_names(
    evidence: RetainedNonPolymerChemistryEvidence,
) -> tuple[str, ...]:
    """Return deterministic expected hydrogen names for one evidence item."""

    hydrogenated_template = template_with_hydrogens(evidence.smiles)
    hydrogen_count = sum(
        1 for atom in hydrogenated_template.GetAtoms() if atom.GetAtomicNum() == 1
    )
    return tuple(f"H{index:03d}" for index in range(1, hydrogen_count + 1))


def retained_non_polymer_evidence_formal_charge_requirements(
    evidence: RetainedNonPolymerChemistryEvidence,
) -> tuple[tuple[str, int], ...]:
    """Project the explicit SMILES heavy-atom charge assignment.

    Parameters
    ----------
    evidence : RetainedNonPolymerChemistryEvidence
        SMILES and its ordered canonical heavy atom identities.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Charge requirements paired with the supplied heavy atom names.

    Raises
    ------
    ValueError
        The atom mapping or SMILES cannot be interpreted.
    RdkitUnavailableError
        RDKit is not operational.
    """
    template = template_without_hydrogens(evidence.smiles)
    names = _evidence_heavy_atom_names_by_index(
        template, heavy_atom_names=evidence.heavy_atom_names
    )
    return tuple(
        (names[atom.GetIdx()], atom.GetFormalCharge()) for atom in template.GetAtoms()
    )


def retained_non_polymer_evidence_heavy_bond_definitions(
    evidence: RetainedNonPolymerChemistryEvidence,
) -> tuple[BondDefinition, ...]:
    """Return heavy-atom bond definitions mapped from evidence chemistry."""

    heavy_template = template_without_hydrogens(evidence.smiles)
    heavy_atom_names_by_index = _evidence_heavy_atom_names_by_index(
        heavy_template,
        heavy_atom_names=evidence.heavy_atom_names,
    )
    return tuple(
        BondDefinition(
            atom_name_1=heavy_atom_names_by_index[bond.GetBeginAtomIdx()],
            atom_name_2=heavy_atom_names_by_index[bond.GetEndAtomIdx()],
            order=max(1, round(bond.GetBondTypeAsDouble())),
            aromatic=bond.GetIsAromatic(),
        )
        for bond in heavy_template.GetBonds()
    )


def retained_non_polymer_evidence_hydrogen_bond_definitions(
    evidence: RetainedNonPolymerChemistryEvidence,
) -> tuple[BondDefinition, ...]:
    """Return generated H-heavy bond definitions mapped from evidence chemistry."""

    hydrogenated_template = template_with_hydrogens(evidence.smiles)
    atom_names_by_index = _evidence_atom_names_by_index(
        hydrogenated_template,
        heavy_atom_names=evidence.heavy_atom_names,
    )
    return tuple(
        BondDefinition(
            atom_name_1=atom_names_by_index[begin_atom.GetIdx()],
            atom_name_2=atom_names_by_index[end_atom.GetIdx()],
            order=max(1, round(bond.GetBondTypeAsDouble())),
            aromatic=bond.GetIsAromatic(),
        )
        for bond in hydrogenated_template.GetBonds()
        for begin_atom, end_atom in (
            (
                hydrogenated_template.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                hydrogenated_template.GetAtomWithIdx(bond.GetEndAtomIdx()),
            ),
        )
        if (begin_atom.GetAtomicNum() == 1) != (end_atom.GetAtomicNum() == 1)
    )


def template_without_hydrogens(smiles: str) -> "Mol":
    """Return a sanitized heavy-atom template for an evidence smiles string."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires an operational "
            "RDKit installation"
        )

    return Chem.Mol(_cached_template_without_hydrogens(smiles))


def template_with_hydrogens(smiles: str) -> "Mol":
    """Return a hydrogenated template for an evidence smiles string."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires an operational "
            "RDKit installation"
        )

    return Chem.Mol(_cached_template_with_hydrogens(smiles))


@lru_cache(maxsize=_EVIDENCE_TEMPLATE_CACHE_MAXSIZE)
def _cached_template_without_hydrogens(smiles: str) -> "Mol":
    """Return the cached sanitized heavy-atom template for evidence smiles."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires an operational "
            "RDKit installation"
        )

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(
            "retained non-polymer chemistry evidence smiles could not be parsed "
            "by RDKit"
        )

    return Chem.RemoveHs(template)


@lru_cache(maxsize=_EVIDENCE_TEMPLATE_CACHE_MAXSIZE)
def _cached_template_with_hydrogens(smiles: str) -> "Mol":
    """Return the cached hydrogenated template for evidence smiles."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires an operational "
            "RDKit installation"
        )

    return Chem.AddHs(Chem.Mol(_cached_template_without_hydrogens(smiles)))


def _evidence_heavy_atom_names_by_index(
    template: "Mol",
    *,
    heavy_atom_names: tuple[str, ...],
) -> dict[int, str]:
    """Return evidence heavy atom names keyed by RDKit atom index."""

    if template.GetNumAtoms() != len(heavy_atom_names):
        raise ValueError(
            "retained non-polymer chemistry evidence heavy_atom_names must match "
            "the evidence template heavy-atom count"
        )

    return {
        atom.GetIdx(): heavy_atom_names[atom_index]
        for atom_index, atom in enumerate(template.GetAtoms())
    }


def _evidence_atom_names_by_index(
    hydrogenated_template: "Mol",
    *,
    heavy_atom_names: tuple[str, ...],
) -> dict[int, str]:
    """Return evidence atom names matching RDKit-generated H append order."""

    atom_names_by_index: dict[int, str] = {}
    heavy_atom_index = 0
    hydrogen_index = 1
    for atom in hydrogenated_template.GetAtoms():
        atom_index = atom.GetIdx()
        if atom.GetAtomicNum() == 1:
            atom_names_by_index[atom_index] = f"H{hydrogen_index:03d}"
            hydrogen_index += 1
            continue

        if heavy_atom_index >= len(heavy_atom_names):
            raise ValueError(
                "retained non-polymer chemistry evidence heavy_atom_names must "
                "not be shorter than the hydrogenated evidence template "
                "heavy-atom count"
            )

        atom_names_by_index[atom_index] = heavy_atom_names[heavy_atom_index]
        heavy_atom_index += 1

    if heavy_atom_index != len(heavy_atom_names):
        raise ValueError(
            "retained non-polymer chemistry evidence heavy_atom_names must match "
            "the hydrogenated evidence template heavy-atom count"
        )

    return atom_names_by_index
