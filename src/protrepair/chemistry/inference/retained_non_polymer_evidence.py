"""RDKit-backed inference for retained non-polymer chemistry evidence."""

from functools import cache
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


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


@cache
def template_without_hydrogens(smiles: str) -> "Mol":
    """Return one sanitized heavy-atom template for an evidence smiles string."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires the optional "
            "rdkit dependency"
        )

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(
            "retained non-polymer chemistry evidence smiles could not be parsed "
            "by RDKit"
        )

    return Chem.RemoveHs(template)


@cache
def template_with_hydrogens(smiles: str) -> "Mol":
    """Return one hydrogenated template for an evidence smiles string."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer chemistry evidence requires the optional "
            "rdkit dependency"
        )

    return Chem.AddHs(template_without_hydrogens(smiles))
