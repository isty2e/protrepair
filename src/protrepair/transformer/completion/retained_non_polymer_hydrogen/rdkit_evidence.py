"""RDKit-backed retained non-polymer hydrogen placement for evidence chemistry."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_heavy_atom_elements,
    retained_non_polymer_evidence_heavy_bond_definitions,
    retained_non_polymer_evidence_hydrogen_bond_definitions,
    template_without_hydrogens,
)
from protrepair.chemistry.radii import covalent_radius_angstrom
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

EVIDENCE_BOND_DISTANCE_TOLERANCE_ANGSTROM = 0.60


@dataclass(frozen=True, slots=True)
class EvidenceHydrogenationResult:
    """Hydrogenated evidence payload plus evidence-resolved topology bonds."""

    payload: CompletionResiduePayload
    heavy_bond_definitions: tuple[BondDefinition, ...]
    hydrogen_bond_definitions: tuple[BondDefinition, ...]


def hydrogenate_retained_non_polymer_payload_with_evidence(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> CompletionResiduePayload:
    """Hydrogenate one retained non-polymer payload from evidence chemistry."""

    return hydrogenate_retained_non_polymer_payload_with_evidence_result(
        payload,
        evidence=evidence,
    ).payload


def hydrogenate_retained_non_polymer_payload_with_evidence_result(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> EvidenceHydrogenationResult:
    """Hydrogenate one payload and return evidence-resolved H anchors."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer evidence hydrogenation requires the required "
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
    return EvidenceHydrogenationResult(
        payload=payload.apply_patch(
            _hydrogen_append_patch(
                payload,
                hydrogenated_molecule=hydrogenated_molecule,
            )
        ),
        heavy_bond_definitions=retained_non_polymer_evidence_heavy_bond_definitions(
            evidence
        ),
        hydrogen_bond_definitions=(
            retained_non_polymer_evidence_hydrogen_bond_definitions(evidence)
        ),
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

    _validate_evidence_bond_geometry(payload, evidence=evidence)


def _validate_evidence_bond_geometry(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> None:
    """Raise when evidence atom mapping is implausible for payload geometry."""

    for bond_definition in retained_non_polymer_evidence_heavy_bond_definitions(
        evidence
    ):
        atom_name_1 = bond_definition.atom_name_1
        atom_name_2 = bond_definition.atom_name_2
        element_1 = payload.atom_site(atom_name_1).element
        element_2 = payload.atom_site(atom_name_2).element
        observed_distance = payload.position(atom_name_1).distance_to(
            payload.position(atom_name_2)
        )
        expected_distance = covalent_radius_angstrom(
            element_1
        ) + covalent_radius_angstrom(element_2)
        deviation = abs(observed_distance - expected_distance)
        if deviation <= EVIDENCE_BOND_DISTANCE_TOLERANCE_ANGSTROM:
            continue

        raise ValueError(
            "retained non-polymer evidence atom mapping is inconsistent with "
            f"payload geometry for {payload.residue_id.display_token()} "
            f"{atom_name_1}-{atom_name_2}: observed {observed_distance:.2f} A, "
            f"expected about {expected_distance:.2f} A"
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
