"""Bind isolated component chemistry to deposited acyl-amide attachments."""

from dataclasses import dataclass, replace

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.topology import BondRelationshipType


@dataclass(frozen=True, slots=True, init=False)
class RetainedTemplateBinding:
    """Admit a component template in its current covalent context.

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
    substitution, using the canonical backbone atom names C and N. Other
    templates require an isolated covalent context, except canonical CYS SG--SG
    disulfides. Typed hydrogen bonds and metal coordination are not substitutions.
    This binding does not mutate source atoms or resolve a deposited bond.

    Raises
    ------
    ValueError
        An attachment has unsupported, incomplete or contradictory chemistry.
    """

    template: ResidueTemplate
    external_bonds: tuple[tuple[AtomRef, AtomRef], ...]

    def __init__(
        self,
        structure: ProteinStructure,
        residue: ResidueSite,
        template: ResidueTemplate,
        component_library: ComponentLibrary,
    ) -> None:
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "external_bonds", ())
        context = RetainedCovalentContext(structure, residue.residue_id)
        if not context.attachments:
            return
        self._bind_acyl(structure, residue, template, component_library, context)
        if residue.component_id == "CYS" and len(context.attachments) == 1:
            pair = context.require_neutral_disulfide_pair()
            object.__setattr__(self, "external_bonds", (pair,))
        context.require_admitted(self.external_bonds)

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
