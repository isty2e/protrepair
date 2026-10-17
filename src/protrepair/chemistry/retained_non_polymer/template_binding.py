"""Bind selected retained chemistry to covalent and coordination context."""

from dataclasses import dataclass, replace

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.inference.coordination_donor import NitrogenCoordinationDonor
from protrepair.chemistry.inference.idealized_stereochemistry import (
    component_reference_molecule,
)
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.chemistry.retained_non_polymer.glycosidic import (
    GlycosidicTemplateBinding,
)
from protrepair.chemistry.retained_non_polymer.michael_adduct import (
    CysteineMichaelAdductBinding,
)
from protrepair.errors import UnsupportedCoordinationChemistryError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.topology import (
    BondRelationshipType,
    is_covalent_like_relationship,
)


@dataclass(frozen=True, slots=True, init=False)
class RetainedTemplateBinding:
    """Admit a component template in its current chemical context.

    Parameters
    ----------
    structure : ProteinStructure
        Snapshot supplying current links and original source observations.
    residue : ResidueSite
        Retained component on that snapshot.
    template : ResidueTemplate
        Isolated chemistry selected before contextual application.
    component_library : ComponentLibrary
        Polymer partner chemistry used to check the linked backbone site.

    Notes
    -----
    Neutral aldehyde templates support one deposited carbonyl C--polymer N
    substitution, using the canonical backbone atom names C and N.
    Graph-qualified pyranoses use substituted anomeric carbons and occupied
    hydroxyl valences. Terminal acrylamides can bind an explicit polymer CYS SG.
    Remaining templates require an isolated covalent context,
    except canonical CYS SG--SG disulfides. Typed hydrogen bonds and metal
    coordination are not substitutions.
    Coordinated N requires an explicit selected donor graph compatible with
    current and original H, charge and partner observations.
    This binding does not mutate source atoms or resolve a deposited bond.

    Raises
    ------
    ValueError
        An attachment has unsupported, incomplete or contradictory chemistry.
    RdkitUnavailableError
        Native donor validation is unavailable.
    """

    template: ResidueTemplate
    external_bonds: tuple[tuple[AtomRef, AtomRef], ...]
    coordination_bonds: tuple[tuple[AtomRef, AtomRef], ...]
    glycosidic: GlycosidicTemplateBinding
    michael: CysteineMichaelAdductBinding | None

    def __init__(
        self,
        structure: ProteinStructure,
        residue: ResidueSite,
        template: ResidueTemplate,
        component_library: ComponentLibrary,
    ) -> None:
        glycosidic = GlycosidicTemplateBinding(
            structure, residue, template, component_library
        )
        object.__setattr__(self, "glycosidic", glycosidic)
        template = glycosidic.template
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "external_bonds", glycosidic.external_bonds)
        object.__setattr__(self, "michael", None)
        object.__setattr__(self, "coordination_bonds", ())
        context = RetainedCovalentContext(structure, residue.residue_id)
        if not context.attachments:
            object.__setattr__(
                self, "coordination_bonds", self._bind_coordination(structure, residue)
            )
            return
        try:
            michael = CysteineMichaelAdductBinding(
                structure, residue, template, component_library
            )
        except ValueError:
            pass
        else:
            object.__setattr__(self, "michael", michael)
            object.__setattr__(self, "template", michael.template)
            object.__setattr__(self, "external_bonds", michael.external_bonds())
            context.require_admitted(self.external_bonds)
            object.__setattr__(
                self, "coordination_bonds", self._bind_coordination(structure, residue)
            )
            return
        self._bind_acyl(structure, residue, template, component_library, context)
        if residue.component_id == "CYS" and len(context.attachments) == 1:
            pair = context.require_neutral_disulfide_pair()
            object.__setattr__(self, "external_bonds", (pair,))
        context.require_admitted(self.external_bonds)
        object.__setattr__(
            self, "coordination_bonds", self._bind_coordination(structure, residue)
        )

    def _bind_coordination(
        self, structure: ProteinStructure, residue: ResidueSite
    ) -> tuple[tuple[AtomRef, AtomRef], ...]:
        constitution = structure.constitution
        observation = structure.provenance.ingress.observation
        incident = structure.topology.incident_bonds_by_atom()
        current_bonds = tuple(
            dict.fromkeys(
                bond
                for atom in residue.atom_sites
                for bond in incident.get(
                    constitution.atom_index(AtomRef(residue.residue_id, atom.name)), ()
                )
            )
        )
        snapshots = [(constitution, current_bonds)]
        if observation is not None:
            snapshots.append(
                (
                    observation.constitution,
                    observation.bonds_for_residue(residue.residue_id),
                )
            )
        molecule = None
        pairs = {}
        donor_partners: dict[AtomRef, set[AtomRef]] = {}
        for source_constitution, bonds in snapshots:
            for bond in bonds:
                if (
                    bond.relationship_type
                    is not BondRelationshipType.METAL_COORDINATION
                ):
                    continue
                first, second = (
                    source_constitution.atom_ref_at(i) for i in bond.endpoint_pair()
                )
                indices = tuple(
                    constitution.resolve_atom_index(ref) for ref in (first, second)
                )
                if any(index is None for index in indices):
                    raise ValueError(
                        "template coordination has a missing original endpoint"
                    )
                first_index, second_index = indices
                assert first_index is not None and second_index is not None
                current = structure.topology.bond_between(first_index, second_index)
                if (
                    current is None
                    or current.relationship_type
                    is not BondRelationshipType.METAL_COORDINATION
                ):
                    raise ValueError("template cannot hide original coordination")
                pairs.setdefault(frozenset((first, second)), (first, second))
                for local, partner in ((first, second), (second, first)):
                    if local.residue_id != residue.residue_id:
                        continue
                    atom = constitution.resolve_atom_site(local)
                    if atom is None:
                        raise ValueError(
                            "template coordination has a missing original endpoint"
                        )
                    if atom.element != "N":
                        continue
                    semantics = self.template.hydrogen_semantics
                    if not isinstance(semantics, IdealGeometryHydrogenSemantics):
                        raise UnsupportedCoordinationChemistryError(
                            "coordinated template N requires a complete "
                            "explicit selected graph"
                        )
                    if not self.template.has_complete_heavy_scaffold(residue):
                        raise ValueError(
                            "coordinated template N requires a complete heavy scaffold"
                        )
                    if molecule is None:
                        molecule = component_reference_molecule(semantics.component)
                        self._require_coordination_source_inventory(
                            structure, residue, semantics
                        )
                    names = tuple(a.atom_name for a in semantics.component.atoms)
                    if local.atom_name not in names:
                        raise ValueError(
                            "selected template lacks the coordination donor"
                        )
                    role = NitrogenCoordinationDonor.from_molecule(
                        molecule, names.index(local.atom_name)
                    )
                    selected_charge = molecule.GetAtomWithIdx(
                        names.index(local.atom_name)
                    ).GetFormalCharge()
                    for charge in (
                        structure.topology.formal_charge(
                            constitution.atom_index(local)
                        ),
                        None
                        if observation is None
                        else observation.formal_charge(local),
                    ):
                        if charge is not None and charge != selected_charge:
                            raise ValueError(
                                "selected template conflicts with source donor charge"
                            )
                    hydrogen_snapshots = [
                        (
                            constitution,
                            structure.topology,
                            incident.get(constitution.atom_index(local), ()),
                        )
                    ]
                    if observation is not None:
                        hydrogen_snapshots.append(
                            (
                                observation.constitution,
                                observation.topology,
                                observation.bonds_for_atom(local),
                            )
                        )
                    for (
                        bound_constitution,
                        bound_topology,
                        attached,
                    ) in hydrogen_snapshots:
                        hydrogens = set()
                        for edge in attached:
                            if (
                                not is_covalent_like_relationship(edge)
                                and edge.relationship_type
                                is not BondRelationshipType.UNKNOWN
                            ):
                                continue
                            for index in edge.endpoint_pair():
                                site = bound_constitution.atom_site_at(index)
                                ref = bound_constitution.atom_ref_at(index)
                                if site.is_hydrogen():
                                    current_site = constitution.resolve_atom_site(ref)
                                    if (
                                        current_site is None
                                        or current_site.element_identity
                                        != site.element_identity
                                    ):
                                        raise ValueError(
                                            "selected template cannot hide "
                                            "original donor H"
                                        )
                                    if bound_topology.formal_charge(index) not in (
                                        None,
                                        0,
                                    ):
                                        raise ValueError(
                                            "selected template conflicts with "
                                            "source H charge"
                                        )
                                    hydrogens.add(ref)
                                elif (
                                    ref != local
                                    and ref.residue_id == residue.residue_id
                                ):
                                    selected_bond = (
                                        molecule.GetBondBetweenAtoms(
                                            names.index(local.atom_name),
                                            names.index(ref.atom_name),
                                        )
                                        if ref.atom_name in names
                                        else None
                                    )
                                    if selected_bond is None or (
                                        edge.order is not None
                                        and (
                                            edge.order not in (1, 2)
                                            if selected_bond.GetIsAromatic()
                                            else edge.order
                                            != selected_bond.GetBondTypeAsDouble()
                                        )
                                    ):
                                        raise ValueError(
                                            "selected template conflicts with "
                                            "source donor bonds"
                                        )
                        if len(hydrogens) > role.hydrogen_count:
                            raise ValueError(
                                "selected template conflicts with source donor H"
                            )
                    partner_atom = constitution.resolve_atom_site(partner)
                    if partner_atom is None:
                        raise ValueError(
                            "template coordination has a missing original partner"
                        )
                    if not role.admits_metal(partner_atom.element):
                        raise UnsupportedCoordinationChemistryError(
                            "selected template has an unsupported "
                            "donor/metal combination"
                        )
                    donor_partners.setdefault(local, set()).add(partner)
        if any(len(partners) != 1 for partners in donor_partners.values()):
            raise UnsupportedCoordinationChemistryError(
                "coordinated template N requires one typed partner per donor"
            )
        return tuple(pairs.values())

    def _require_coordination_source_inventory(
        self,
        structure: ProteinStructure,
        residue: ResidueSite,
        semantics: IdealGeometryHydrogenSemantics,
    ) -> None:
        constitution = structure.constitution
        observation = structure.provenance.ingress.observation
        for atom in semantics.component.heavy_atoms():
            ref = AtomRef(residue.residue_id, atom.atom_name)
            for charge in (
                structure.topology.formal_charge(constitution.atom_index(ref)),
                None if observation is None else observation.formal_charge(ref),
            ):
                if charge is not None and charge != atom.formal_charge:
                    raise ValueError("selected template conflicts with source charge")
        for atom in residue.atom_sites:
            if atom.is_hydrogen() and structure.topology.formal_charge(
                constitution.atom_index(AtomRef(residue.residue_id, atom.name))
            ) not in (None, 0):
                raise ValueError("selected template conflicts with source H charge")
        if observation is None:
            return
        for ref in observation.hydrogen_atoms(residue.residue_id):
            original_index = observation.constitution.atom_index(ref)
            current = constitution.resolve_atom_site(ref)
            if (
                current is None
                or current.element_identity
                != observation.constitution.atom_site_at(
                    original_index
                ).element_identity
            ):
                raise ValueError("selected template cannot hide original H inventory")
            if observation.formal_charge(ref) not in (None, 0):
                raise ValueError("selected template conflicts with original H charge")

    def _bind_acyl(
        self,
        structure: ProteinStructure,
        residue: ResidueSite,
        template: ResidueTemplate,
        component_library: ComponentLibrary,
        context: RetainedCovalentContext,
    ) -> None:
        semantics = template.hydrogen_semantics
        if not isinstance(semantics, IdealGeometryHydrogenSemantics):
            return
        carbonyl_names = semantics.component.neutral_aldehyde_carbonyl_atom_names()
        if not carbonyl_names:
            return
        constitution = structure.constitution
        polymer_count = len(constitution.residue_slots) - len(constitution.ligands)
        if not any(
            carbonyl.atom_name == "C"
            and carbonyl.atom_name in carbonyl_names
            and nitrogen.atom_name == "N"
            and bond.relationship_type is BondRelationshipType.COVALENT
            and constitution.residue_index(nitrogen.residue_id).value < polymer_count
            and constitution.atom_site_at(constitution.atom_index(nitrogen)).element
            == "N"
            for bond, carbonyl, nitrogen in context.attachments
        ):
            return
        if not template.has_complete_heavy_scaffold(residue):
            raise ValueError("acyl chemistry requires a complete compatible scaffold")
        residue_id = residue.residue_id
        for atom in semantics.component.heavy_atoms():
            index = constitution.atom_index(AtomRef(residue_id, atom.atom_name))
            if structure.topology.formal_charge(index) not in (
                None,
                atom.formal_charge,
            ):
                raise ValueError("source charge disagrees with neutral acyl chemistry")

        external = context.attachments
        if len(external) != 1:
            raise ValueError("acyl substitution requires one external covalent link")
        bond, carbonyl, nitrogen = external[0]
        if bond.order not in (None, 1) or bond.aromatic:
            raise ValueError(
                "unsupported acyl attachment; expected carbonyl C--polymer N"
            )
        if structure.topology.formal_charge(constitution.atom_index(nitrogen)) not in (
            None,
            0,
        ):
            raise ValueError("linked amide N has a conflicting formal charge")
        partner = constitution.residue_site_at(
            constitution.residue_index(nitrogen.residue_id)
        )
        partner_context = RetainedCovalentContext(structure, nitrogen.residue_id)
        if not partner_context.has_exclusive_attachment(nitrogen, carbonyl):
            raise ValueError(
                "linked amide N has another external covalent or "
                "unresolved relationship"
            )
        partner_template = component_library.get(partner.component_id)
        if partner_template is None:
            raise ValueError(
                "acyl attachment has no supported polymer partner chemistry"
            )
        resolution = PolymerMicrostateSite(
            partner_template, PolymerChemicalSite.BACKBONE_N, PeptideLinkage.LINKED
        ).resolve(partner, structure.provenance.ingress.observation)
        if resolution.status is not MicrostateResolutionStatus.RESOLVED:
            raise ValueError(
                "source polymer N is incompatible with linked amide chemistry"
            )

        projected = semantics.component.without_hydrogens_anchored_to(
            (carbonyl.atom_name,)
        )
        object.__setattr__(
            self,
            "template",
            replace(
                template,
                definition=replace(
                    template.definition,
                    formal_charges={
                        atom.atom_name: atom.formal_charge
                        for atom in projected.heavy_atoms()
                    },
                ),
                hydrogen_semantics=IdealGeometryHydrogenSemantics(component=projected),
            ),
        )
        object.__setattr__(self, "external_bonds", ((carbonyl, nitrogen),))


def is_resolved_acyl_amide_link(
    structure: ProteinStructure, carbonyl: AtomRef, nitrogen: AtomRef
) -> bool:
    """Check the realized neutral acyl-amide graph, not a component label.

    Parameters
    ----------
    structure : ProteinStructure
        Current canonical graph.
    carbonyl, nitrogen : AtomRef
        Retained carbonyl C and polymer backbone N endpoints.

    Returns
    -------
    bool
        Whether C has exactly C=O, C--C and C--N bonds, with no H or unresolved
        covalent valence and no formal charge on the carbonyl or nitrogen.
    """
    constitution = structure.constitution
    indices = tuple(
        constitution.resolve_atom_index(ref) for ref in (carbonyl, nitrogen)
    )
    if any(index is None for index in indices):
        return False
    carbonyl_index, nitrogen_index = indices
    assert carbonyl_index is not None and nitrogen_index is not None
    partner_context = RetainedCovalentContext(structure, nitrogen.residue_id)
    if not partner_context.has_exclusive_attachment(nitrogen, carbonyl):
        return False
    amide = structure.topology.bond_between(carbonyl_index, nitrogen_index)
    if (
        amide is None
        or amide.relationship_type is not BondRelationshipType.COVALENT
        or amide.order != 1
        or amide.aromatic
    ):
        return False
    if constitution.atom_site_at(carbonyl_index).element != "C":
        return False
    if constitution.atom_site_at(nitrogen_index).element != "N":
        return False
    if any(
        structure.topology.formal_charge(index) not in (None, 0)
        for index in (carbonyl_index, nitrogen_index)
    ):
        return False
    neighbors = []
    for bond in structure.topology.bonds:
        if carbonyl_index not in bond.endpoint_pair() or bond.relationship_type in (
            BondRelationshipType.HYDROGEN_BOND,
            BondRelationshipType.METAL_COORDINATION,
        ):
            continue
        if (
            bond.relationship_type is not BondRelationshipType.COVALENT
            or bond.aromatic
            or bond.order not in (1, 2)
        ):
            return False
        other = next(index for index in bond.endpoint_pair() if index != carbonyl_index)
        ref = constitution.atom_ref_at(other)
        if other != nitrogen_index and ref.residue_id != carbonyl.residue_id:
            return False
        if structure.topology.formal_charge(other) not in (None, 0):
            return False
        neighbors.append((constitution.atom_site_at(other).element, bond.order))
    return sorted(neighbors) == [("C", 1), ("N", 1), ("O", 2)]
