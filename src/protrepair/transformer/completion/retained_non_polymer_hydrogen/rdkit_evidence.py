"""RDKit-backed retained non-polymer hydrogen placement for evidence chemistry."""

from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_heavy_atom_elements,
    template_without_hydrogens,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


def hydrogenate_retained_non_polymer_payload_with_evidence(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> CompletionResiduePayload:
    """Hydrogenate one retained non-polymer payload from evidence chemistry."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer evidence hydrogenation requires the optional "
            "rdkit dependency"
        )

    _validate_evidence_alignment(
        payload,
        evidence=evidence,
    )
    hydrogenated_molecule = Chem.AddHs(
        _rdkit_pose_molecule(
            payload,
            evidence=evidence,
        ),
        addCoords=True,
    )
    return payload.apply_patch(
        _hydrogen_append_patch(
            payload,
            hydrogenated_molecule=hydrogenated_molecule,
        )
    )


def _validate_evidence_alignment(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> None:
    """Raise when one evidence cannot be projected onto the current payload."""

    payload_heavy_atom_names = tuple(
        atom_site.name for atom_site in payload.atom_sites if atom_site.element != "H"
    )
    if set(payload_heavy_atom_names) != set(evidence.heavy_atom_names):
        raise ValueError(
            "retained non-polymer evidence heavy_atom_names must match the payload "
            f"heavy-atom set for {payload.residue_id.display_token()}"
        )

    for atom_name, expected_element in zip(
        evidence.heavy_atom_names,
        retained_non_polymer_evidence_heavy_atom_elements(evidence),
        strict=True,
    ):
        if not payload.has_atom(atom_name):
            raise ValueError(
                "retained non-polymer evidence heavy atom is absent from the "
                f"payload for {payload.residue_id.display_token()}: {atom_name}"
            )

        observed_element = payload.atom_site(atom_name).element
        if observed_element != expected_element:
            raise ValueError(
                "retained non-polymer evidence element mismatch for "
                f"{payload.residue_id.display_token()} atom {atom_name}: "
                f"observed {observed_element}, expected {expected_element}"
            )


def _rdkit_pose_molecule(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> "Mol":
    """Return one heavy-atom RDKit pose aligned to the current payload."""

    assert Chem is not None

    pose_molecule = Chem.Mol(template_without_hydrogens(evidence.smiles))
    pose_molecule.RemoveAllConformers()
    conformer = Chem.Conformer(pose_molecule.GetNumAtoms())
    for atom_index, atom_name in enumerate(evidence.heavy_atom_names):
        conformer.SetAtomPosition(atom_index, tuple(payload.position(atom_name)))

    pose_molecule.AddConformer(conformer, assignId=True)
    return pose_molecule


def _hydrogen_append_patch(
    payload: CompletionResiduePayload,
    *,
    hydrogenated_molecule: "Mol",
) -> OrderedAtomPatch:
    """Return one patch that appends generated evidence hydrogens."""

    conformer = hydrogenated_molecule.GetConformer()
    hydrogen_positions: list[Vec3] = []
    for atom in hydrogenated_molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue

        coordinates = conformer.GetAtomPosition(atom.GetIdx())
        hydrogen_positions.append(
            Vec3(
                float(coordinates.x),
                float(coordinates.y),
                float(coordinates.z),
            )
        )

    hydrogen_atom_names = tuple(
        f"H{index:03d}" for index in range(1, len(hydrogen_positions) + 1)
    )
    return OrderedAtomPatch.from_residue_payload(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
    ).append_atoms(
        hydrogen_atom_names,
        hydrogen_positions,
    )

