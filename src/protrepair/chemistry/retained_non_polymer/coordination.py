"""Source-compatible preparation of known coordinated ligand scaffolds."""

from collections import Counter
from dataclasses import dataclass, replace

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    require_retained_non_polymer_evidence_stereochemistry,
    retained_non_polymer_evidence_bond_projection,
    retained_non_polymer_evidence_coordination_bonds,
    retained_non_polymer_evidence_external_bonds,
    retained_non_polymer_evidence_formal_charge_requirements,
    template_without_hydrogens,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import pras_microstate_preferences
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.chemistry.retained_non_polymer.coordination_catalogue import (
    coordination_microstate_family,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite, StructureConstitution
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
    is_covalent_like_relationship,
)


@dataclass(frozen=True, slots=True, init=False)
class CoordinatedComponentPreparation:
    """Bind a common ligand preparation without resolving ambiguous metal charge.

    Parameters
    ----------
    structure : ProteinStructure
        Current chemistry with independent original observations.
    residue : ResidueSite
        Registered coordination chemistry without an active component template.
    component_library : ComponentLibrary
        Active library, including the standard HIS partner grammar.

    Raises
    ------
    ValueError
        Scaffold, source chemistry, partners or candidate projections disagree.
    RdkitUnavailableError
        The required native chemistry backend is unavailable.

    Notes
    -----
    Neutral bound imidazoles and deprotonated HME propionates are preferences,
    not pKa predictions. Fe(II)/Fe(III) hypotheses are a bounded native preparation
    domain, not an exhaustive redox model or a metal force-field admission.
    """

    hypotheses: tuple[RetainedNonPolymerChemistryEvidence, ...]
    coordination_bonds: tuple[tuple[AtomRef, AtomRef], ...]

    def __init__(
        self,
        structure: ProteinStructure,
        residue: ResidueSite,
        component_library: ComponentLibrary,
    ) -> None:
        if component_library.get(residue.component_id) is not None:
            raise ValueError("active component templates own their preparation")
        if coordination_microstate_family(residue.component_id) is None:
            raise ValueError("no coordinated component preparation profile")

        site_residues = _known_site_residues(structure, residue, component_library)
        selected = {
            member.residue_id: _source_compatible_hypotheses(structure, member)
            for member in site_residues
        }
        hypotheses = selected[residue.residue_id]
        pairs = retained_non_polymer_evidence_coordination_bonds(
            hypotheses[0], structure
        )
        first_projection = retained_non_polymer_evidence_bond_projection(hypotheses[0])
        first_hydrogens = _hydrogen_counts(hypotheses[0])
        for hypothesis in hypotheses[1:]:
            if (
                retained_non_polymer_evidence_bond_projection(hypothesis)
                != first_projection
                or _hydrogen_counts(hypothesis) != first_hydrogens
                or retained_non_polymer_evidence_coordination_bonds(
                    hypothesis, structure
                )
                != pairs
            ):
                raise ValueError("coordination hypotheses lack a common preparation")

        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "coordination_bonds", pairs)

    @property
    def hydrogenation_template(self) -> RetainedNonPolymerChemistryEvidence:
        """Return a native H-placement representative, not a selected metal state.

        Returns
        -------
        RetainedNonPolymerChemistryEvidence
            One mapped native template with the admitted common H and bond graph.
            Its non-common charges must never be published to canonical topology.
        """
        return self.hypotheses[0]

    def formal_charge_requirements(self) -> tuple[tuple[str, int], ...]:
        """Project only per-atom charges shared by every admitted hypothesis.

        Returns
        -------
        tuple[tuple[str, int], ...]
            Assigned ligand charges and, only when constrained, the metal charge.
        """
        candidates = tuple(
            dict(retained_non_polymer_evidence_formal_charge_requirements(evidence))
            for evidence in self.hypotheses
        )
        return tuple(
            (name, charge)
            for name, charge in candidates[0].items()
            if all(candidate[name] == charge for candidate in candidates[1:])
        )

    def unresolved_charge_atom_names(self) -> tuple[str, ...]:
        """Return mapped atoms whose charge is not common to the hypotheses.

        Returns
        -------
        tuple[str, ...]
            Canonical names omitted from the preparation's charge requirements.
        """
        assigned = {name for name, _ in self.formal_charge_requirements()}
        return tuple(
            name
            for name in self.hydrogenation_template.heavy_atom_names
            if name not in assigned
        )


def coordinated_component_preparation(
    structure: ProteinStructure,
    residue: ResidueSite,
    component_library: ComponentLibrary,
) -> CoordinatedComponentPreparation | None:
    """Bind a known contextual profile, leaving other template/fallback owners alone.

    Parameters
    ----------
    structure : ProteinStructure
        Source-bound current snapshot.
    residue : ResidueSite
        Retained component whose preparation is being resolved.
    component_library : ComponentLibrary
        Active template authority; an existing template takes precedence.

    Returns
    -------
    CoordinatedComponentPreparation or None
        Admitted common chemistry, or None when this profile is inapplicable.

    Raises
    ------
    ValueError
        A recognized profile cannot be prepared under its source/site constraints.
    RdkitUnavailableError
        The required native chemistry backend is unavailable.
    """
    if component_library.get(residue.component_id) is not None:
        return None
    family = coordination_microstate_family(residue.component_id)
    if family is None:
        return None
    if not family.has_internal_coordination and not _incident_coordination(
        structure, residue
    ):
        return None
    return CoordinatedComponentPreparation(structure, residue, component_library)


def _incident_coordination(
    structure: ProteinStructure, residue: ResidueSite
) -> tuple[tuple[AtomRef, AtomRef], ...]:
    pairs = {}
    observation = structure.provenance.ingress.observation
    residue_index = structure.constitution.residue_index(residue.residue_id)
    incident = structure.topology.incident_bonds_by_atom()
    current_bonds = tuple(
        dict.fromkeys(
            bond
            for atom in residue.atom_sites
            for bond in incident.get(
                structure.constitution.atom_index_in_residue(residue_index, atom.name),
                (),
            )
        )
    )
    snapshots = ((structure.constitution, current_bonds),)
    if observation is not None:
        snapshots += (
            (
                observation.constitution,
                observation.bonds_for_residue(residue.residue_id),
            ),
        )
    for constitution, bonds in snapshots:
        for bond in bonds:
            if bond.relationship_type is not BondRelationshipType.METAL_COORDINATION:
                continue
            refs = (
                constitution.atom_ref_at(bond.atom_index_1),
                constitution.atom_ref_at(bond.atom_index_2),
            )
            if not any(ref.residue_id == residue.residue_id for ref in refs):
                continue
            bond.with_resolved_chemistry(replace(bond, order=None, aromatic=False))
            indices = tuple(
                structure.constitution.resolve_atom_index(ref) for ref in refs
            )
            first_index, second_index = indices
            if first_index is None or second_index is None:
                raise ValueError("original coordination has a missing current partner")
            current = structure.topology.bond_between(first_index, second_index)
            if (
                current is None
                or current.relationship_type
                is not BondRelationshipType.METAL_COORDINATION
            ):
                raise ValueError(
                    "original coordination is missing from current topology"
                )
            pairs.setdefault(frozenset(refs), refs)
    return tuple(pairs.values())


def _known_site_residues(
    structure: ProteinStructure,
    residue: ResidueSite,
    component_library: ComponentLibrary,
) -> tuple[ResidueSite, ...]:
    members = {residue.residue_id: residue}
    pending = [residue]
    while pending:
        member = pending.pop()
        pairs = _incident_coordination(structure, member)
        family = coordination_microstate_family(member.component_id)
        assert family is not None
        if not family.has_internal_coordination:
            if len(pairs) != 1:
                raise ValueError(
                    f"bound {member.component_id} requires one typed "
                    "coordination partner"
                )
            local = next(ref for ref in pairs[0] if ref.residue_id == member.residue_id)
            partner = next(ref for ref in pairs[0] if ref != local)
            if not family.admits_donor(
                local.atom_name,
                metal_element=structure.constitution.atom_site_at(
                    structure.constitution.atom_index(partner)
                ).element,
            ):
                raise ValueError(
                    f"unsupported {member.component_id} coordination donor/metal motif"
                )
        for pair in pairs:
            outside = tuple(ref for ref in pair if ref.residue_id != member.residue_id)
            for ref in outside:
                other = structure.constitution.residue_or_ligand(ref.residue_id)
                if other is None:
                    raise ValueError("coordination has no current partner residue")
                if coordination_microstate_family(other.component_id) is not None:
                    if component_library.get(other.component_id) is not None:
                        raise ValueError(
                            "automatic coordination cannot replace an active "
                            "partner component template; supply mapped chemistry "
                            "for this mixed site"
                        )
                    if other.residue_id not in members:
                        members[other.residue_id] = other
                        pending.append(other)
                elif other.component_id == "HIS":
                    template = component_library.get("HIS")
                    if template is None:
                        raise ValueError(
                            "coordinated HIS partner has no active chemical template"
                        )
                    context = PolymerMicrostateContext(structure)
                    site = PolymerMicrostateSite(
                        template, PolymerChemicalSite.SIDECHAIN
                    )
                    resolution = context.resolve(
                        other.residue_id,
                        site,
                        preferences=pras_microstate_preferences(site),
                    )
                    if resolution.graph is None:
                        raise ValueError(
                            "coordinated HIS partner chemistry is unresolved "
                            "or contradictory"
                        )
                    context.hydrogen_attachments(other.residue_id, site, resolution)
                elif family.has_internal_coordination:
                    raise ValueError(
                        "unsupported internal metal axial partner preparation"
                    )
    return tuple(members.values())


def _hydrogen_counts(
    evidence: RetainedNonPolymerChemistryEvidence,
) -> tuple[tuple[str, int], ...]:
    template = template_without_hydrogens(evidence.smiles)
    return tuple(
        (name, template.GetAtomWithIdx(index).GetTotalNumHs())
        for index, name in enumerate(evidence.heavy_atom_names)
    )


def _source_compatible_hypotheses(
    structure: ProteinStructure, residue: ResidueSite
) -> tuple[RetainedNonPolymerChemistryEvidence, ...]:
    pairs = _incident_coordination(structure, residue)
    family = coordination_microstate_family(residue.component_id)
    assert family is not None
    donor = (
        None
        if family.has_internal_coordination
        else next(
            ref.atom_name for ref in pairs[0] if ref.residue_id == residue.residue_id
        )
    )
    candidates = family.hypotheses(residue.residue_id, donor_atom_name=donor)
    names = candidates[0].heavy_atom_names
    native = template_without_hydrogens(candidates[0].smiles)
    heavy = {
        atom.name: atom.element for atom in residue.atom_sites if not atom.is_hydrogen()
    }
    expected = {
        name: native.GetAtomWithIdx(index).GetSymbol().upper()
        for index, name in enumerate(names)
    }
    if heavy != expected:
        raise ValueError(
            "coordinated component heavy scaffold is incomplete or changed"
        )

    observation = structure.provenance.ingress.observation
    if observation is not None and not observation.heavy_atoms_are_preserved_in(
        residue
    ):
        raise ValueError(
            "coordinated preparation cannot hide original heavy atom identities"
        )

    require_retained_non_polymer_evidence_stereochemistry(
        candidates[0],
        structure.residue_geometry(
            structure.constitution.residue_index(residue.residue_id)
        ),
        hydrogen_attachments=tuple(
            structure.topology.residue_hydrogen_anchor_by_name(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(residue.residue_id),
            ).items()
        ),
    )
    constraints = _source_constraints(structure, residue, names)
    source_orders = structure.topology.residue_source_heavy_bond_orders(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue.residue_id),
    )
    compatible = []
    for candidate in candidates:
        charges = dict(
            retained_non_polymer_evidence_formal_charge_requirements(candidate)
        )
        hydrogens = dict(_hydrogen_counts(candidate))
        if any(charges[name] != charge for name, charge in constraints.charges) or any(
            hydrogens[name] < count for name, count in constraints.minimum_hydrogens
        ):
            continue
        try:
            retained_non_polymer_evidence_bond_projection(
                candidate, bond_order_constraints=source_orders
            )
        except ValueError:
            continue
        compatible.append(candidate)
    if not compatible:
        raise ValueError(
            "source H/charge/orders conflict with coordinated component preparation"
        )

    # Prefer compatible ligand chemistry without selecting an unresolved metal state.
    first = compatible[0]
    ligand_charges = tuple(
        (name, charge)
        for name, charge in retained_non_polymer_evidence_formal_charge_requirements(
            first
        )
        if name not in family.metal_atom_names
    )
    selected = []
    for candidate in compatible:
        candidate_charges = retained_non_polymer_evidence_formal_charge_requirements(
            candidate
        )
        if (
            _hydrogen_counts(candidate) == _hydrogen_counts(first)
            and tuple(
                (name, charge)
                for name, charge in candidate_charges
                if name not in family.metal_atom_names
            )
            == ligand_charges
        ):
            selected.append(candidate)
    for candidate in selected:
        if retained_non_polymer_evidence_external_bonds(candidate, structure):
            raise ValueError(
                "coordinated component cannot have an external covalent attachment"
            )
        retained_non_polymer_evidence_coordination_bonds(candidate, structure)
    return tuple(selected)


def _source_constraints(
    structure: ProteinStructure, residue: ResidueSite, names: tuple[str, ...]
) -> MicrostateConstraints:
    observation = structure.provenance.ingress.observation
    charges = {}
    minimum_hydrogens = Counter()
    for name in names:
        ref = AtomRef(residue.residue_id, name)
        index = structure.constitution.atom_index(ref)
        current = structure.topology.formal_charge(index)
        original = None if observation is None else observation.formal_charge(ref)
        if current is not None and original is not None and current != original:
            raise ValueError(
                "current charge contradicts original coordinated chemistry"
            )
        charge = original if original is not None else current
        if charge is not None:
            charges[name] = charge

    residue_index = structure.constitution.residue_index(residue.residue_id)
    incident = structure.topology.incident_bonds_by_atom()
    current_bonds = tuple(
        dict.fromkeys(
            bond
            for atom in residue.atom_sites
            for bond in incident.get(
                structure.constitution.atom_index_in_residue(residue_index, atom.name),
                (),
            )
        )
    )
    snapshots = ((structure.constitution, current_bonds),)
    if observation is not None:
        for bond in observation.bonds_for_residue(residue.residue_id):
            refs = tuple(
                observation.constitution.atom_ref_at(index)
                for index in bond.endpoint_pair()
            )
            first, second = (
                structure.constitution.resolve_atom_index(ref) for ref in refs
            )
            if first is None or second is None:
                raise ValueError("original source connectivity has a missing endpoint")
            current = structure.topology.bond_between(first, second)
            if (
                current is None
                or current.provenance is not BondProvenance.SOURCE_EXPLICIT
                or current.source_metadata != bond.source_metadata
            ):
                raise ValueError("original coordinated source connectivity was changed")
        for ref in observation.hydrogen_atoms(residue.residue_id):
            index = structure.constitution.resolve_atom_index(ref)
            original_index = observation.constitution.atom_index(ref)
            if (
                index is None
                or structure.constitution.atom_site_at(index).element_identity
                != observation.constitution.atom_site_at(
                    original_index
                ).element_identity
            ):
                raise ValueError(
                    "coordinated preparation cannot hide original H/isotope identity"
                )
        snapshots += (
            (
                observation.constitution,
                observation.bonds_for_residue(residue.residue_id),
            ),
        )
    for constitution, bonds in snapshots:
        counts = _attached_hydrogen_counts(
            constitution, bonds, residue.residue_id, names
        )
        for name, count in counts.items():
            minimum_hydrogens[name] = max(minimum_hydrogens[name], count)
    return MicrostateConstraints(
        charges=tuple(charges.items()),
        minimum_hydrogens=tuple(minimum_hydrogens.items()),
    )


def _attached_hydrogen_counts(
    constitution: StructureConstitution,
    bonds: tuple[TopologyBond, ...],
    residue_id: ResidueId,
    names: tuple[str, ...],
) -> Counter[str]:
    residue = constitution.residue_or_ligand(residue_id)
    if residue is None:
        return Counter()
    parents = {atom.name: set() for atom in residue.atom_sites if atom.is_hydrogen()}
    for bond in bonds:
        if not (
            is_covalent_like_relationship(bond)
            or bond.relationship_type is BondRelationshipType.UNKNOWN
        ):
            continue
        refs = tuple(constitution.atom_ref_at(index) for index in bond.endpoint_pair())
        for hydrogen, parent in (refs, tuple(reversed(refs))):
            if hydrogen.residue_id == residue_id and hydrogen.atom_name in parents:
                if bond.order not in {None, 1} or bond.aromatic:
                    raise ValueError(
                        "source H requires a non-aromatic single attachment"
                    )
                if bond.source_metadata is not None and (
                    bond.source_metadata.reported_order not in {None, 1}
                    or bond.source_metadata.reported_relationship_type
                    not in {BondRelationshipType.UNKNOWN, BondRelationshipType.COVALENT}
                ):
                    raise ValueError(
                        "source H reports incompatible attachment chemistry"
                    )
                if parent.residue_id != residue_id or parent.atom_name not in names:
                    raise ValueError(
                        "coordinated source H has an unsupported attachment"
                    )
                parents[hydrogen.atom_name].add(parent.atom_name)
    if any(len(anchors) != 1 for anchors in parents.values()):
        raise ValueError(
            "coordinated source H requires an unambiguous heavy attachment"
        )
    return Counter(next(iter(anchors)) for anchors in parents.values())
