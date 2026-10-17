"""RDKit-backed retained non-polymer hydrogen placement for evidence chemistry."""

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
from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.chemistry.inference.coordination_donor import NitrogenCoordinationDonor
from protrepair.chemistry.inference.idealized_stereochemistry import (
    component_reference_molecule,
)
from protrepair.chemistry.inference.rdkit_hydrogen_geometry import (
    add_hydrogens_preserving_stereochemistry,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    require_retained_non_polymer_evidence_stereochemistry,
    retained_non_polymer_evidence_bond_projection,
    retained_non_polymer_evidence_heavy_atom_elements,
    retained_non_polymer_evidence_heavy_bond_definitions,
    retained_non_polymer_evidence_hydrogen_bond_definitions,
    template_without_hydrogens,
)
from protrepair.chemistry.radii import (
    ElementRadiusLookup,
    RadiusKind,
    prepare_radius_lookup,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.transformer.completion.retained_non_polymer_hydrogen import (
    rdkit_patch,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

EVIDENCE_BOND_DISTANCE_TOLERANCE_ANGSTROM = 0.60


@dataclass(frozen=True, slots=True)
class EvidenceHydrogenationResult:
    """Hydrogenated evidence payload plus the coupled graph to publish.

    Parameters
    ----------
    payload : CompletionResiduePayload
        Hydrogenated atom inventory and coordinates before source reconciliation.
    heavy_bond_definitions : tuple[BondDefinition, ...]
        Selected covalent heavy graph with source-compatible integral orders.
    hydrogen_bond_definitions : tuple[BondDefinition, ...]
        Generated covalent H attachments, before source-name reconciliation.
    coordination_bond_definitions : tuple[CoordinationBondDefinition, ...]
        Selected donor-to-acceptor relationships, separate from covalent valence.
    """

    payload: CompletionResiduePayload
    heavy_bond_definitions: tuple[BondDefinition, ...]
    hydrogen_bond_definitions: tuple[BondDefinition, ...]
    coordination_bond_definitions: tuple[CoordinationBondDefinition, ...]


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
    bond_order_constraints: Mapping[tuple[str, str], int] | None = None,
    coordination_partner_positions: Mapping[str, Vec3] | None = None,
) -> EvidenceHydrogenationResult:
    """Hydrogenate one payload with evidence chemistry and reported orders.

    Parameters
    ----------
    payload : CompletionResiduePayload
        Current heavy scaffold and source scalar/charge evidence.
    evidence : RetainedNonPolymerChemistryEvidence
        Selected SMILES chemistry and canonical atom correspondence.
    bond_order_constraints : Mapping[tuple[str, str], int] or None
        Reported local heavy orders for a compatible integral resonance form.
    coordination_partner_positions : Mapping[str, Vec3] or None
        Validated external partner positions keyed by the local donor. Used only
        as geometry anchors for donor H, never as published atoms or chemistry.

    Returns
    -------
    EvidenceHydrogenationResult
        Hydrogenated payload and jointly compatible covalent heavy/H bonds and
        oriented coordination relationships.

    Raises
    ------
    ValueError
        Evidence, atom correspondence or reported orders contradict the chemistry.
    GeometryPlacementError
        The declared partner coincides with the donor.
    RdkitUnavailableError
        RDKit is not operational.
    """

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer evidence hydrogenation requires an operational "
            "RDKit installation"
        )

    _validate_evidence_alignment(
        payload,
        evidence=evidence,
    )
    pose = _rdkit_pose_molecule(payload, evidence=evidence)
    pose, anchors = _pose_with_coordination_anchors(
        pose, evidence.heavy_atom_names, coordination_partner_positions or {}
    )
    hydrogenated_molecule = add_hydrogens_preserving_stereochemistry(pose)
    if anchors:
        projected = Chem.RWMol(hydrogenated_molecule)
        for index in reversed(anchors):
            projected.RemoveAtom(index)
        hydrogenated_molecule = projected.GetMol()
    bond_projection = retained_non_polymer_evidence_bond_projection(
        evidence, bond_order_constraints=bond_order_constraints
    )
    return EvidenceHydrogenationResult(
        payload=payload.apply_patch(
            rdkit_patch.rdkit_hydrogen_append_patch(
                payload,
                hydrogenated_molecule=hydrogenated_molecule,
                hydrogen_atom_names=_evidence_hydrogen_atom_names(
                    hydrogenated_molecule,
                ),
            )
        ),
        heavy_bond_definitions=bond_projection.covalent_bonds,
        hydrogen_bond_definitions=(
            retained_non_polymer_evidence_hydrogen_bond_definitions(evidence)
        ),
        coordination_bond_definitions=bond_projection.coordination_bonds,
    )


def _pose_with_coordination_anchors(
    pose: "Mol",
    atom_names: tuple[str, ...],
    positions: Mapping[str, Vec3],
) -> tuple["Mol", tuple[int, ...]]:
    assert Chem is not None
    if not positions:
        return pose, ()
    molecule = Chem.RWMol(pose)
    indices = {name: i for i, name in enumerate(atom_names)}
    anchors = []
    for name, position in positions.items():
        index = indices[name]
        role = NitrogenCoordinationDonor.from_molecule(pose, index)
        if not role.hydrogen_count:
            continue
        if any(
            b.GetBondType() == Chem.BondType.DATIVE
            for b in pose.GetAtomWithIdx(index).GetBonds()
        ):
            raise ValueError("donor H placement has multiple coordination partners")
        if (
            Vec3.from_iterable(pose.GetConformer().GetAtomPosition(index)).distance_to(
                position
            )
            == 0
        ):
            raise GeometryPlacementError("donor H placement has a coincident partner")
        # A dummy point guides native H geometry without assigning a metal state.
        anchor = Chem.Atom(0)
        anchor.SetNoImplicit(True)
        anchor_index = molecule.AddAtom(anchor)
        molecule.AddBond(index, anchor_index, Chem.BondType.DATIVE)
        molecule.GetConformer().SetAtomPosition(anchor_index, tuple(position))
        anchors.append(anchor_index)
    if anchors:
        Chem.SanitizeMol(molecule)
    return molecule.GetMol(), tuple(anchors)


def place_coordinated_template_hydrogens(
    payload: CompletionResiduePayload,
    component: IdealizedComponent,
    partner_positions: Mapping[str, Vec3],
) -> CompletionResiduePayload:
    """Place only generated donor H against validated external partner geometry.

    Parameters
    ----------
    payload : CompletionResiduePayload
        Complete generated template payload, before source-H reconciliation.
    component : IdealizedComponent
        Active explicit chemical graph, retaining its H names and charges.
    partner_positions : Mapping[str, Vec3]
        Validated external coordinates keyed by the mapped donor.

    Returns
    -------
    CompletionResiduePayload
        Updated donor H positions; all other atoms and graph identity are kept.

    Raises
    ------
    ValueError
        Donor chemistry, graph correspondence or placement is unsupported.
    GeometryPlacementError
        The declared partner coincides with the donor.
    RdkitUnavailableError
        Native geometry is unavailable.
    """
    if Chem is None:
        raise RdkitUnavailableError(
            "coordination H placement requires operational RDKit"
        )
    if not partner_positions:
        return payload
    pose = Chem.RemoveHs(component_reference_molecule(component))
    names = component.heavy_atom_names()
    pose.RemoveAllConformers()
    conformer = Chem.Conformer(pose.GetNumAtoms())
    conformer.Set3D(True)
    for i, name in enumerate(names):
        conformer.SetAtomPosition(i, tuple(payload.position(name)))
    pose.AddConformer(conformer)
    augmented, _ = _pose_with_coordination_anchors(pose, names, partner_positions)
    native = add_hydrogens_preserving_stereochemistry(augmented)
    positions = {}
    for name in partner_positions:
        i = names.index(name)
        if not NitrogenCoordinationDonor.from_molecule(pose, i).hydrogen_count:
            continue
        hydrogen_names = tuple(
            a.atom_name
            for a in component.hydrogen_atoms()
            if component.hydrogen_anchor_atom_name(a.atom_name) == name
        )
        native_hydrogens = tuple(
            a.GetIdx()
            for a in native.GetAtomWithIdx(i).GetNeighbors()
            if a.GetAtomicNum() == 1
        )
        if len(hydrogen_names) != 1 or len(native_hydrogens) != 1:
            raise ValueError("selected donor H correspondence is not unique")
        positions[hydrogen_names[0]] = Vec3.from_iterable(
            native.GetConformer().GetAtomPosition(native_hydrogens[0])
        )
    return payload.apply_patch(
        OrderedAtomPatch.from_atom_coordinates(
            atom_names=list(payload.atom_names()),
            atom_coordinates=[
                positions.get(name, payload.position(name))
                for name in payload.atom_names()
            ],
        )
    )


def _validate_evidence_alignment(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> None:
    """Raise when one evidence cannot be projected onto the current payload."""

    payload_heavy_atom_names = tuple(
        atom_site.name
        for atom_site in payload.atom_sites
        if not atom_site.is_hydrogen()
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

    require_retained_non_polymer_evidence_stereochemistry(
        evidence, payload.residue_geometry
    )
    _validate_evidence_bond_geometry(payload, evidence=evidence)


def _validate_evidence_bond_geometry(
    payload: CompletionResiduePayload,
    *,
    evidence: RetainedNonPolymerChemistryEvidence,
) -> None:
    """Raise when evidence atom mapping is implausible for payload geometry."""

    heavy_bond_definitions = retained_non_polymer_evidence_heavy_bond_definitions(
        evidence
    )
    covalent_radius_lookup = _evidence_bond_covalent_radius_lookup(
        payload,
        heavy_bond_definitions=heavy_bond_definitions,
    )
    for bond_definition in heavy_bond_definitions:
        atom_name_1 = bond_definition.atom_name_1
        atom_name_2 = bond_definition.atom_name_2
        element_1 = payload.atom_site(atom_name_1).element
        element_2 = payload.atom_site(atom_name_2).element
        observed_distance = payload.position(atom_name_1).distance_to(
            payload.position(atom_name_2)
        )
        expected_distance = covalent_radius_lookup.radius_angstrom(
            element_1
        ) + covalent_radius_lookup.radius_angstrom(element_2)
        deviation = abs(observed_distance - expected_distance)
        if deviation <= EVIDENCE_BOND_DISTANCE_TOLERANCE_ANGSTROM:
            continue

        raise ValueError(
            "retained non-polymer evidence atom mapping is inconsistent with "
            f"payload geometry for {payload.residue_id.display_token()} "
            f"{atom_name_1}-{atom_name_2}: observed {observed_distance:.2f} A, "
            f"expected about {expected_distance:.2f} A"
        )


def _evidence_bond_covalent_radius_lookup(
    payload: CompletionResiduePayload,
    *,
    heavy_bond_definitions: tuple[BondDefinition, ...],
) -> ElementRadiusLookup:
    """Return prepared covalent radii needed by evidence-bond geometry checks."""

    radius_lookup = prepare_radius_lookup(
        (
            payload.atom_site(atom_name).element
            for bond_definition in heavy_bond_definitions
            for atom_name in (
                bond_definition.atom_name_1,
                bond_definition.atom_name_2,
            )
        ),
        RadiusKind.COVALENT,
    )
    radius_lookup.require_complete("retained non-polymer evidence bond geometry")
    return radius_lookup


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


def _evidence_hydrogen_atom_names(
    hydrogenated_molecule: "Mol",
) -> tuple[str, ...]:
    """Return generated evidence H names in RDKit atom order."""

    hydrogen_count = sum(
        atom.GetAtomicNum() == 1 for atom in hydrogenated_molecule.GetAtoms()
    )
    return tuple(f"H{index:03d}" for index in range(1, hydrogen_count + 1))
