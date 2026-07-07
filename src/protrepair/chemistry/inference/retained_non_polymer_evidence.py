"""RDKit-backed inference for retained non-polymer chemistry evidence."""

from functools import lru_cache
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

_EVIDENCE_TEMPLATE_CACHE_MAXSIZE = 256


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
            "retained non-polymer chemistry evidence requires the required "
            "rdkit dependency"
        )

    return Chem.Mol(_cached_template_without_hydrogens(smiles))


def template_with_hydrogens(smiles: str) -> "Mol":
    """Return a hydrogenated template for an evidence smiles string."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires the required "
            "rdkit dependency"
        )

    return Chem.Mol(_cached_template_with_hydrogens(smiles))


@lru_cache(maxsize=_EVIDENCE_TEMPLATE_CACHE_MAXSIZE)
def _cached_template_without_hydrogens(smiles: str) -> "Mol":
    """Return the cached sanitized heavy-atom template for evidence smiles."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires the required "
            "rdkit dependency"
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
            "retained non-polymer chemistry evidence requires the required "
            "rdkit dependency"
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
