"""RDKit-backed inference for retained non-polymer chemistry evidence."""

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.inference.coordination_donor import NitrogenCoordinationDonor
from protrepair.chemistry.inference.rdkit_graph import (
    AROMATIC_N_COORDINATION_METALS,
    NativeBondProjection,
    has_declared_stereochemistry,
    native_bond_projection_from_molecule,
    require_compatible_pose_stereochemistry,
)
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import (
    BondOrderConstraintError,
    CoordinationEvidenceConflictError,
    RdkitUnavailableError,
    UnsupportedCoordinationChemistryError,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondRelationshipType,
    is_covalent_like_relationship,
)

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


def retained_non_polymer_evidence_external_bonds(
    evidence: RetainedNonPolymerChemistryEvidence,
    structure: ProteinStructure,
) -> tuple[tuple[AtomRef, AtomRef], ...]:
    """Admit isolated evidence or a terminal thiol bound in a canonical disulfide.

    Parameters
    ----------
    evidence : RetainedNonPolymerChemistryEvidence
        Residue-local precursor SMILES with an ordered heavy-atom mapping.
    structure : ProteinStructure
        Current canonical graph, including source charges and external attachments.

    Returns
    -------
    tuple[tuple[AtomRef, AtomRef], ...]
        Stable endpoint pairs whose binding consumes a generated precursor thiol H.
        Empty for isolated evidence. Other coupled-reaction chemistry is unsupported.

    Raises
    ------
    ValueError
        An attachment, sulfur charge, precursor or mapping is incompatible with the
        single neutral-thiol disulfide grammar.
    RdkitUnavailableError
        RDKit is not operational when the precursor must be inspected.
    """
    context = RetainedCovalentContext(structure, evidence.residue_id)
    if not context.attachments:
        return ()
    pair = context.require_neutral_disulfide_pair()
    _, inside, _ = context.attachments[0]

    if Chem is None:
        raise RdkitUnavailableError(
            "disulfide evidence requires an operational RDKit installation"
        )
    template = template_without_hydrogens(evidence.smiles)
    names = _evidence_heavy_atom_names_by_index(
        template, heavy_atom_names=evidence.heavy_atom_names
    )
    sulfur_index = next(
        (index for index, name in names.items() if name == inside.atom_name), None
    )
    if sulfur_index is None:
        raise ValueError("disulfide evidence must map the attached SG atom")
    sulfur = template.GetAtomWithIdx(sulfur_index)
    neighbors = tuple(sulfur.GetNeighbors())
    if (
        len(Chem.GetMolFrags(template)) != 1
        or sulfur.GetAtomicNum() != 16
        or sulfur.GetFormalCharge() != 0
        or sulfur.GetNumRadicalElectrons() != 0
        or sulfur.GetIsAromatic()
        or sulfur.GetTotalNumHs() != 1
        or len(neighbors) != 1
        or neighbors[0].GetAtomicNum() != 6
        or any(bond.GetBondTypeAsDouble() != 1.0 for bond in sulfur.GetBonds())
    ):
        raise ValueError(
            "disulfide evidence requires a neutral closed-shell terminal thiol"
        )

    admitted = (pair,)
    context.require_admitted(admitted)
    return admitted


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
    *,
    bond_order_constraints: Mapping[tuple[str, str], int] | None = None,
) -> tuple[BondDefinition, ...]:
    """Project evidence chemistry while retaining compatible source orders.

    Parameters
    ----------
    evidence : RetainedNonPolymerChemistryEvidence
        Resolved SMILES chemistry with canonical heavy atom names.
    bond_order_constraints : Mapping[tuple[str, str], int] or None
        Reported local heavy orders; aromatic resonance may change, not microstate.

    Returns
    -------
    tuple[BondDefinition, ...]
        Mapped integral heavy bonds with retained aromatic identity.

    Raises
    ------
    ValueError
        Evidence or source orders cannot describe the same resolved chemistry.
    RdkitUnavailableError
        RDKit is not operational.
    """

    return retained_non_polymer_evidence_bond_projection(
        evidence,
        bond_order_constraints=bond_order_constraints,
    ).covalent_bonds


def retained_non_polymer_evidence_bond_projection(
    evidence: RetainedNonPolymerChemistryEvidence,
    *,
    bond_order_constraints: Mapping[tuple[str, str], int] | None = None,
) -> NativeBondProjection:
    """Project selected heavy chemistry, retaining admitted N-to-metal direction.

    Parameters
    ----------
    evidence : RetainedNonPolymerChemistryEvidence
        Resolved SMILES and ordered canonical heavy atom identities.
    bond_order_constraints : Mapping[tuple[str, str], int] or None
        Reported integral covalent orders, not coordination strengths.

    Returns
    -------
    NativeBondProjection
        Coupled covalent and typed coordination products. Coordination admits
        unprotonated aromatic N donors to Fe, Zn, Cu, Co, Ni, Mn, Cd or Hg, and
        primary sulfonamidate NH(-1) to Zn. This is preparation support, not
        metal force-field support or an oxidation-state prediction.

    Raises
    ------
    ValueError
        Mapping, orders or the declared donor-to-metal motif is incompatible.
    RdkitUnavailableError
        RDKit is not operational.
    """
    template = template_without_hydrogens(evidence.smiles)
    names = _evidence_heavy_atom_names_by_index(
        template, heavy_atom_names=evidence.heavy_atom_names
    )
    assert Chem is not None
    coordinate_pairs = {
        frozenset((names[bond.GetBeginAtomIdx()], names[bond.GetEndAtomIdx()]))
        for bond in template.GetBonds()
        if bond.GetBondType() == Chem.BondType.DATIVE
    }
    covalent_constraints = {}
    for pair, order in (bond_order_constraints or {}).items():
        if frozenset(pair) in coordinate_pairs:
            if type(order) is not int or order != 1:
                raise BondOrderConstraintError(
                    "reported multiple order conflicts with selected dative chemistry"
                )
        else:
            covalent_constraints[pair] = order
    projection = native_bond_projection_from_molecule(
        template, atom_names_by_index=names, bond_order_constraints=covalent_constraints
    )
    indices = {name: index for index, name in names.items()}
    donor_names = []
    for bond in projection.coordination_bonds:
        donor = NitrogenCoordinationDonor.from_molecule(
            template, indices[bond.donor_atom_name]
        )
        acceptor = template.GetAtomWithIdx(indices[bond.acceptor_atom_name])
        if not donor.admits_metal(acceptor.GetSymbol().upper()):
            raise UnsupportedCoordinationChemistryError(
                "selected coordination has an unsupported donor/metal combination"
            )
        donor_names.append(bond.donor_atom_name)
    if len(set(donor_names)) != len(donor_names):
        raise ValueError("selected nitrogen donor has multiple coordination partners")
    return projection


def retained_non_polymer_evidence_coordination_bonds(
    evidence: RetainedNonPolymerChemistryEvidence,
    structure: ProteinStructure,
) -> tuple[tuple[AtomRef, AtomRef], ...]:
    """Bind selected local chemistry to current and original typed partners.

    Parameters
    ----------
    evidence : RetainedNonPolymerChemistryEvidence
        Selected local SMILES with an exact heavy atom correspondence.
    structure : ProteinStructure
        Current graph and, when available, immutable original observations.

    Returns
    -------
    tuple[tuple[AtomRef, AtomRef], ...]
        Required physical coordination pairs in stable atom identity. External
        partners retain their own preparation responsibilities; this does not
        certify a complete site or assign their unknown charges.

    Raises
    ------
    ValueError
        A local relationship is absent from the selection, an original external
        partner is missing, a mapped donor or partner conflicts with the motif,
        or a bound H charge disagrees with the selected neutral H chemistry.
    RdkitUnavailableError
        RDKit is not operational.
    """
    projection = retained_non_polymer_evidence_bond_projection(evidence)
    molecule = template_without_hydrogens(evidence.smiles)
    return selected_graph_coordination_bonds(
        structure,
        residue_id=evidence.residue_id,
        molecule=molecule,
        atom_names_by_index=_evidence_heavy_atom_names_by_index(
            molecule, heavy_atom_names=evidence.heavy_atom_names
        ),
        projection=projection,
    )


def selected_graph_coordination_bonds(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    molecule: "Mol",
    atom_names_by_index: Mapping[int, str],
    projection: NativeBondProjection,
) -> tuple[tuple[AtomRef, AtomRef], ...]:
    """Validate selected native chemistry against current and original partners.

    Parameters
    ----------
    structure : ProteinStructure
        Current canonical graph and original observations.
    residue_id : ResidueId
        Residue whose selected local graph is being bound.
    molecule : Mol
        Sanitized selected graph; explicit native H may be present.
    atom_names_by_index : Mapping[int, str]
        Stable heavy-atom names for native indices. Generated H are not required.
    projection : NativeBondProjection
        Selected named covalent and local dative relationships.

    Returns
    -------
    tuple[tuple[AtomRef, AtomRef], ...]
        Required coordination pairs, without assigning partner chemistry.

    Raises
    ------
    CoordinationEvidenceConflictError
        Selected chemistry conflicts with source inventory or relationships.
    UnsupportedCoordinationChemistryError
        The donor or metal lies outside the admitted preparation grammar.
    """
    pairs = {
        frozenset((first, second)): (first, second)
        for bond in projection.coordination_bonds
        for first, second in (
            (
                AtomRef(residue_id, bond.donor_atom_name),
                AtomRef(residue_id, bond.acceptor_atom_name),
            ),
        )
    }
    for pair in pairs.values():
        for ref in pair:
            if structure.constitution.resolve_atom_index(ref) is None:
                raise CoordinationEvidenceConflictError(
                    "selected coordination has a missing mapped endpoint "
                    f"at {ref.display_token()}"
                )
    template = molecule
    indices = {name: index for index, name in atom_names_by_index.items()}
    partners_by_donor = {
        AtomRef(residue_id, bond.donor_atom_name): {
            AtomRef(residue_id, bond.acceptor_atom_name)
        }
        for bond in projection.coordination_bonds
    }
    snapshots = [
        (
            structure.constitution,
            tuple(
                dict.fromkeys(
                    bond
                    for index in structure.constitution.atom_indices_for_residue_index(
                        structure.constitution.residue_index(residue_id)
                    )
                    for bond in structure.topology.incident_bonds_by_atom().get(
                        index, ()
                    )
                )
            ),
        )
    ]
    observation = structure.provenance.ingress.observation
    charge_snapshots = [(structure.constitution, structure.topology)]
    if observation is not None:
        charge_snapshots.append((observation.constitution, observation.topology))
        snapshots.append(
            (
                observation.constitution,
                observation.bonds_for_residue(residue_id),
            )
        )
    if not projection.coordination_bonds and not any(
        bond.relationship_type is BondRelationshipType.METAL_COORDINATION
        for _, bonds in snapshots
        for bond in bonds
    ):
        return ()
    for constitution, topology in charge_snapshots:
        for atom in template.GetAtoms():
            name = atom_names_by_index.get(atom.GetIdx())
            if name is None:
                continue
            ref = AtomRef(residue_id, name)
            index = constitution.resolve_atom_index(ref)
            if index is not None:
                charge = topology.formal_charge(index)
                if charge is not None and charge != atom.GetFormalCharge():
                    raise CoordinationEvidenceConflictError(
                        "selected coordination conflicts with source formal charge "
                        f"at {ref.display_token()}"
                    )
        residue = constitution.residue_or_ligand(residue_id)
        if residue is None:
            continue
        for atom in residue.atom_sites:
            if atom.is_hydrogen() and topology.formal_charge(
                constitution.atom_index(AtomRef(residue_id, atom.name))
            ) not in {None, 0}:
                raise CoordinationEvidenceConflictError(
                    "selected neutral H chemistry conflicts with source hydrogen charge"
                )
    for constitution, bonds in snapshots:
        for bond in bonds:
            if bond.relationship_type is not BondRelationshipType.METAL_COORDINATION:
                continue
            refs = tuple(
                constitution.atom_ref_at(index) for index in bond.endpoint_pair()
            )
            inside = tuple(ref for ref in refs if ref.residue_id == residue_id)
            if not inside:
                continue
            if len(inside) == 2:
                if frozenset(refs) not in pairs:
                    raise CoordinationEvidenceConflictError(
                        "local coordination relationship is absent from "
                        "selected chemistry"
                    )
                continue
            local = inside[0]
            partner = next(ref for ref in refs if ref != local)
            local_index = indices.get(local.atom_name)
            partner_index = structure.constitution.resolve_atom_index(partner)
            if local_index is None or partner_index is None:
                raise CoordinationEvidenceConflictError(
                    "selected coordination has a missing mapped partner"
                )
            first_index, second_index = (
                structure.constitution.resolve_atom_index(ref) for ref in refs
            )
            if first_index is None or second_index is None:
                raise CoordinationEvidenceConflictError(
                    "original coordination has a missing current endpoint"
                )
            current = structure.topology.bond_between(first_index, second_index)
            if (
                current is None
                or current.relationship_type
                is not BondRelationshipType.METAL_COORDINATION
            ):
                raise CoordinationEvidenceConflictError(
                    "original coordination is missing from current topology"
                )
            local_atom = template.GetAtomWithIdx(local_index)
            partner_element = structure.constitution.atom_site_at(partner_index).element
            if (
                local_atom.GetAtomicNum() == 7
                and partner_element in AROMATIC_N_COORDINATION_METALS
            ):
                role = NitrogenCoordinationDonor.from_molecule(template, local_index)
                if not role.admits_metal(partner_element):
                    raise UnsupportedCoordinationChemistryError(
                        "selected coordination has an unsupported "
                        "donor/metal combination"
                    )
                donor = local
            elif (
                local_atom.GetSymbol().upper() in AROMATIC_N_COORDINATION_METALS
                and partner_element == "N"
            ):
                donor = partner
            else:
                raise UnsupportedCoordinationChemistryError(
                    "selected external coordination has an unsupported "
                    "donor/metal motif"
                )
            partners_by_donor.setdefault(donor, set()).add(
                partner if donor == local else local
            )
            pairs.setdefault(frozenset(refs), (refs[0], refs[1]))

    attached_by_donor = {
        structure.constitution.atom_index(donor): [] for donor in partners_by_donor
    }
    for index in attached_by_donor:
        attached_by_donor[index].extend(
            structure.topology.incident_bonds_by_atom().get(index, ())
        )
    for donor, partners in partners_by_donor.items():
        donor_index = structure.constitution.atom_index(donor)
        local_index = (
            indices.get(donor.atom_name) if donor.residue_id == residue_id else None
        )
        role = (
            None
            if local_index is None
            else NitrogenCoordinationDonor.from_molecule(template, local_index)
        )
        hydrogen_count = None if role is None else role.hydrogen_count
        selected_charge = (
            None
            if local_index is None
            else template.GetAtomWithIdx(local_index).GetFormalCharge()
        )
        for charge in (
            structure.topology.formal_charge(donor_index),
            None if observation is None else observation.formal_charge(donor),
        ):
            if charge not in {None, 0, -1} or (
                charge is not None
                and selected_charge is not None
                and charge != selected_charge
            ):
                raise CoordinationEvidenceConflictError(
                    "selected coordination conflicts with a source donor charge"
                )
        current_hydrogens = set()
        for attached in attached_by_donor[donor_index]:
            other = next(
                index for index in attached.endpoint_pair() if index != donor_index
            )
            if attached.relationship_type is BondRelationshipType.METAL_COORDINATION:
                partners.add(structure.constitution.atom_ref_at(other))
            if (
                is_covalent_like_relationship(attached)
                or attached.relationship_type is BondRelationshipType.UNKNOWN
            ) and structure.constitution.atom_site_at(other).is_hydrogen():
                current_hydrogens.add(other)
        if hydrogen_count is not None and len(current_hydrogens) > hydrogen_count:
            raise CoordinationEvidenceConflictError(
                "selected coordination conflicts with source donor hydrogen"
            )
        if observation is not None:
            original_hydrogens = set()
            for attached in observation.bonds_for_atom(donor):
                if not (
                    is_covalent_like_relationship(attached)
                    or attached.relationship_type is BondRelationshipType.UNKNOWN
                ):
                    continue
                other = next(
                    index
                    for index in attached.endpoint_pair()
                    if observation.constitution.atom_ref_at(index) != donor
                )
                if observation.constitution.atom_site_at(other).is_hydrogen():
                    original_atom = observation.constitution.atom_site_at(other)
                    ref = observation.constitution.atom_ref_at(other)
                    current_atom = structure.constitution.resolve_atom_site(ref)
                    if (
                        current_atom is None
                        or current_atom.element_identity
                        != original_atom.element_identity
                        or structure.constitution.resolve_atom_index(ref)
                        not in current_hydrogens
                    ):
                        raise CoordinationEvidenceConflictError(
                            "selected coordination cannot hide original donor "
                            f"hydrogen identity or attachment at {ref.display_token()}"
                        )
                    original_hydrogens.add(other)
            if hydrogen_count is not None and len(original_hydrogens) > hydrogen_count:
                raise CoordinationEvidenceConflictError(
                    "selected coordination conflicts with original donor hydrogen"
                )
    if any(len(partners) > 1 for partners in partners_by_donor.values()):
        raise CoordinationEvidenceConflictError(
            "selected nitrogen donor has multiple coordination partners"
        )
    return tuple(pairs.values())


def retained_non_polymer_evidence_hydrogen_bond_definitions(
    evidence: RetainedNonPolymerChemistryEvidence,
) -> tuple[BondDefinition, ...]:
    """Return generated H-heavy bond definitions mapped from evidence chemistry."""

    hydrogenated_template = template_with_hydrogens(evidence.smiles)
    atom_names_by_index = _evidence_atom_names_by_index(
        hydrogenated_template,
        heavy_atom_names=evidence.heavy_atom_names,
    )
    hydrogen_names = frozenset(
        atom_names_by_index[atom.GetIdx()]
        for atom in hydrogenated_template.GetAtoms()
        if atom.GetAtomicNum() == 1
    )
    return tuple(
        bond
        for bond in native_bond_projection_from_molecule(
            hydrogenated_template, atom_names_by_index=atom_names_by_index
        ).covalent_bonds
        if (bond.atom_name_1 in hydrogen_names) != (bond.atom_name_2 in hydrogen_names)
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
