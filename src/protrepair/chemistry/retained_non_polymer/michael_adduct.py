"""Source-linked terminal acrylamide addition to polymer cysteine sulfur."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite

import numpy as np

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import (
    HydrogenSemantics,
    IdealGeometryHydrogenSemantics,
    RotatableHydrogenKind,
)
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.hydrogen_plans import DISULFIDE_CYSTEINE_PLAN
from protrepair.chemistry.inference.idealized_stereochemistry import (
    require_compatible_component_reference,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.restraint.template import ResidueRestraintTemplate
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.geometry.placement_vector import normalized_vector_or_none
from protrepair.geometry.polyhedra import TetrahedralCenter
from protrepair.geometry.vector import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite, StructureConstitution
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


@dataclass(frozen=True, slots=True, init=False)
class _TerminalAcrylamide:
    """Graph-qualified neutral N-C(=O)-CH=CH2 and its product H inventory."""

    carbonyl: str
    alpha: str
    beta: str
    alpha_hydrogens: tuple[str, str]
    beta_hydrogens: tuple[str, str]

    def __init__(self, component: IdealizedComponent, beta: str) -> None:
        atoms = {atom.atom_name: atom for atom in component.atoms}
        neighbors: dict[str, list[tuple[str, int]]] = {name: [] for name in atoms}
        for bond in component.bonds:
            if bond.aromatic:
                continue
            neighbors[bond.atom_name_1].append((bond.atom_name_2, bond.order))
            neighbors[bond.atom_name_2].append((bond.atom_name_1, bond.order))

        def require_neighborhood(name: str, expected: list[tuple[str, int]]) -> None:
            atom = atoms.get(name)
            if (
                atom is None
                or atom.formal_charge != 0
                or atom.stereo_config is not None
                or any(
                    bond.aromatic
                    for bond in component.bonds
                    if name in (bond.atom_name_1, bond.atom_name_2)
                )
                or sorted(
                    ("H" if atoms[other].is_hydrogen() else atoms[other].element, order)
                    for other, order in neighbors[name]
                )
                != sorted(expected)
            ):
                raise ValueError("unsupported terminal acrylamide neighborhood")

        require_neighborhood(beta, [("C", 2), ("H", 1), ("H", 1)])
        if atoms[beta].element != "C":
            raise ValueError("Michael attachment requires a terminal beta carbon")
        alpha = next(name for name, order in neighbors[beta] if order == 2)
        require_neighborhood(alpha, [("C", 1), ("C", 2), ("H", 1)])
        carbonyl = next(
            name
            for name, order in neighbors[alpha]
            if order == 1 and atoms[name].element == "C"
        )
        require_neighborhood(carbonyl, [("C", 1), ("N", 1), ("O", 2)])
        oxygen = next(name for name, order in neighbors[carbonyl] if order == 2)
        require_neighborhood(oxygen, [("C", 2)])
        nitrogen = next(
            name for name, _ in neighbors[carbonyl] if atoms[name].element == "N"
        )
        if (
            atoms[nitrogen].formal_charge != 0
            or any(order != 1 for _, order in neighbors[nitrogen])
            or any(
                bond.aromatic
                for bond in component.bonds
                if nitrogen in (bond.atom_name_1, bond.atom_name_2)
            )
        ):
            raise ValueError("terminal acrylamide requires a neutral amide nitrogen")

        hydrogen_names = {
            name: tuple(
                other for other, _ in neighbors[name] if atoms[other].is_hydrogen()
            )
            for name in (alpha, beta)
        }
        if any(
            atoms[name].formal_charge != 0 or neighbors[name] != [(parent, 1)]
            for parent, names in hydrogen_names.items()
            for name in names
        ):
            raise ValueError("terminal acrylamide has incompatible hydrogen chemistry")
        suffix = 1
        while f"H{suffix}" in atoms:
            suffix += 1
        object.__setattr__(self, "carbonyl", carbonyl)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(
            self, "alpha_hydrogens", (hydrogen_names[alpha][0], f"H{suffix}")
        )
        object.__setattr__(self, "beta_hydrogens", hydrogen_names[beta])

    def product(self, component: IdealizedComponent) -> IdealizedComponent:
        added = self.alpha_hydrogens[1]
        return replace(
            component,
            atoms=(*component.atoms, IdealizedComponentAtom(added, "H", 0)),
            bonds=tuple(
                replace(bond, order=1)
                if {bond.atom_name_1, bond.atom_name_2} == {self.alpha, self.beta}
                else bond
                for bond in component.bonds
            )
            + (BondDefinition(self.alpha, added),),
        )


@dataclass(frozen=True, slots=True, init=False)
class CysteineMichaelAdductBinding:
    """Admit a graph-qualified acrylamide--CYS product without mutation.

    Parameters
    ----------
    structure : ProteinStructure
        Current snapshot and immutable original chemical observations.
    residue : ResidueSite
        Retained component with one covalent beta-carbon--polymer CYS SG link.
    template : ResidueTemplate
        Complete isolated precursor graph and declared stereochemical reference.
    component_library : ComponentLibrary
        Chemistry for both participants.

    Notes
    -----
    The terminal acrylamide becomes C(=O)--CH2--CH2--SG.
    Both carbons are methylene groups; no new non-isotopic stereocenter is assigned.
    This is a product application, not a reaction predictor or pKa model.
    Other electrophiles, nucleophiles and residue-local coupled overrides are
    unsupported until their graph and stereo semantics are explicitly modeled.

    Raises
    ------
    ValueError
        Link, scaffold, charge, original H, graph or mapped stereo is incompatible.
    """

    template: ResidueTemplate
    cysteine_template: ResidueTemplate
    sulfur: AtomRef
    ligand_id: ResidueId
    _warhead: _TerminalAcrylamide = field(repr=False)
    _structure: ProteinStructure = field(repr=False)

    def __init__(
        self,
        structure: ProteinStructure,
        residue: ResidueSite,
        template: ResidueTemplate,
        component_library: ComponentLibrary,
    ) -> None:
        semantics = template.hydrogen_semantics
        if (
            residue.component_id != template.component_id
            or not isinstance(semantics, IdealGeometryHydrogenSemantics)
            or not template.has_complete_heavy_scaffold(residue)
        ):
            raise ValueError("Michael adduct requires its complete compatible scaffold")
        attachments = RetainedCovalentContext(structure, residue.residue_id).attachments
        if len(attachments) != 1:
            raise ValueError(
                "Michael adduct requires exactly one external covalent link"
            )
        bond, carbon, sulfur = attachments[0]
        constitution = structure.constitution
        partner = constitution.residue_or_ligand(sulfur.residue_id)
        polymer_count = len(constitution.residue_slots) - len(constitution.ligands)
        if (
            sulfur.atom_name != "SG"
            or partner is None
            or partner.component_id != "CYS"
            or constitution.residue_index(sulfur.residue_id).value >= polymer_count
            or bond.relationship_type is not BondRelationshipType.COVALENT
            or bond.order not in (None, 1)
            or bond.aromatic
        ):
            raise ValueError(
                "Michael adduct requires a covalent single "
                "beta-carbon--polymer CYS SG link"
            )
        partner_template = component_library.get("CYS")
        if partner_template is None or any(
            not partner.has_atom_site(name)
            or partner.atom_site(name).element != element
            for name, element in (
                ("N", "N"),
                ("CA", "C"),
                ("C", "C"),
                ("O", "O"),
                ("CB", "C"),
                ("SG", "S"),
            )
        ):
            raise ValueError("Michael adduct requires a complete polymer CYS scaffold")
        partner_semantics = partner_template.hydrogen_semantics
        if (
            not isinstance(partner_semantics, HydrogenSemantics)
            or partner_semantics.rotatable_kind is not RotatableHydrogenKind.CYS
        ):
            raise ValueError(
                "Michael adduct requires supported cysteine carbon-H chemistry"
            )
        if partner_template.definition.bonded_atom_names("SG") != frozenset(
            {"CB"}
        ) or any(
            b.order != 1 or b.aromatic
            for b in partner_template.definition.bonds
            if {"CB", "SG"}.intersection((b.atom_name_1, b.atom_name_2))
        ):
            raise ValueError(
                "cysteine template does not describe the supported saturated skeleton"
            )
        if partner_template.definition.bonded_atom_names("CB") != frozenset(
            {"CA", "SG"}
        ) or any(
            partner_template.definition.formal_charges.get(name, 0) != 0
            for name in ("CB", "SG")
        ):
            raise ValueError(
                "cysteine template has unsupported thioether boundary chemistry"
            )

        component = semantics.component
        if (
            {
                (
                    frozenset((bond.atom_name_1, bond.atom_name_2)),
                    bond.order,
                    bond.aromatic,
                )
                for bond in template.definition.bonds
            }
            != {
                (
                    frozenset((bond.atom_name_1, bond.atom_name_2)),
                    bond.order,
                    bond.aromatic,
                )
                for bond in component.heavy_bonds()
            }
            or set(template.expected_heavy_atom_names())
            != set(component.heavy_atom_names())
            or any(
                residue.atom_site(atom.atom_name).element != atom.element
                or template.definition.formal_charges.get(atom.atom_name, 0)
                != atom.formal_charge
                for atom in component.heavy_atoms()
            )
        ):
            raise ValueError(
                "Michael template and H semantics disagree on the heavy graph"
            )
        warhead = _TerminalAcrylamide(component, carbon.atom_name)
        require_compatible_component_reference(component)
        product = warhead.product(component)
        object.__setattr__(self, "_warhead", warhead)
        object.__setattr__(self, "_structure", structure)
        object.__setattr__(self, "ligand_id", residue.residue_id)
        object.__setattr__(self, "sulfur", sulfur)
        object.__setattr__(
            self,
            "template",
            replace(
                template,
                definition=replace(template.definition, bonds=product.heavy_bonds()),
                hydrogen_semantics=IdealGeometryHydrogenSemantics(product),
            ),
        )
        # Carbon-H geometry is identical to the no-thiol disulfide inventory;
        # this reuse does not classify the product as a disulfide.
        object.__setattr__(
            self,
            "cysteine_template",
            replace(
                partner_template,
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=DISULFIDE_CYSTEINE_PLAN
                ),
            ),
        )
        self._require_graph_and_charge()
        self._require_original_hydrogens(partner, partner_template)
        self.excess_generated_thiol_hydrogen_names()
        self.require_ligand_stereochemistry(
            structure.residue_geometry(constitution.residue_index(residue.residue_id))
        )

    def external_bonds(self) -> tuple[tuple[AtomRef, AtomRef], ...]:
        """Return the admitted cross-residue product bond.

        Returns
        -------
        tuple[tuple[AtomRef, AtomRef], ...]
            Ligand beta carbon and cysteine SG under stable canonical identities.
        """
        return ((AtomRef(self.ligand_id, self._warhead.beta), self.sulfur),)

    def reaction_hydrogen_names(self) -> frozenset[str]:
        """Return H identities requiring product-aware methylene placement.

        Returns
        -------
        frozenset[str]
            Two alpha H names and two beta H names from the selected graph.
        """
        return frozenset(
            (*self._warhead.alpha_hydrogens, *self._warhead.beta_hydrogens)
        )

    def reaction_hydrogen_positions(
        self, geometry: ResidueGeometry
    ) -> tuple[tuple[str, Vec3], ...]:
        """Place product methylene H against current carbon and partner geometry.

        Parameters
        ----------
        geometry : ResidueGeometry
            Current ligand coordinates. No original atom is moved.

        Returns
        -------
        tuple[tuple[str, Vec3], ...]
            Named alpha and beta H positions for transformer publication.

        Raises
        ------
        GeometryPlacementError
            Current anchors cannot define methylene placement.
        KeyError
            A required carbon anchor is absent.
        """
        warhead = self._warhead
        alpha = TetrahedralCenter(
            geometry.position(warhead.carbonyl),
            geometry.position(warhead.beta),
            geometry.position(warhead.alpha),
        ).pair()
        beta = TetrahedralCenter(
            geometry.position(warhead.alpha),
            self.sulfur_position(),
            geometry.position(warhead.beta),
        ).pair()
        return tuple(
            zip(
                (*warhead.alpha_hydrogens, *warhead.beta_hydrogens),
                (*alpha, *beta),
                strict=True,
            )
        )

    def sulfur_position(self) -> Vec3:
        """Return the partner sulfur coordinate used for product H placement.

        Returns
        -------
        Vec3
            Current SG position; neither participant is moved.
        """
        return self._structure.geometry.position(
            self._structure.constitution.atom_index(self.sulfur)
        )

    def context_bonds(self) -> tuple[tuple[AtomRef, AtomRef], ...]:
        """Return the partner-local single bond needed by the thioether graph.

        Returns
        -------
        tuple[tuple[AtomRef, AtomRef], ...]
            CYS CB--SG, independently of the cross-residue product link.
            Publication must preserve any compatible explicit source metadata.
        """
        return ((AtomRef(self.sulfur.residue_id, "CB"), self.sulfur),)

    def applicable_restraints(
        self, restraints: ResidueRestraintTemplate
    ) -> ResidueRestraintTemplate:
        """Exclude isolated alkene targets after product graph realization.

        Parameters
        ----------
        restraints : ResidueRestraintTemplate
            Isolated precursor dictionary selected by the snapshot-bound owner.

        Returns
        -------
        ResidueRestraintTemplate
            Unchanged targets before realization; otherwise targets unaffected
            by alpha/beta hybridization. Native FF supplies reaction-site geometry.

        Raises
        ------
        ValueError
            The dictionary describes a different component.
        """
        if restraints.component_id != self.template.component_id:
            raise ValueError("Michael adduct requires its own restraint dictionary")
        if not self.cysteine_is_realized():
            return restraints

        centers = {self._warhead.alpha, self._warhead.beta}
        return replace(
            restraints,
            bond_targets=tuple(
                target
                for target in restraints.bond_targets
                if not centers.intersection((target.atom_name_1, target.atom_name_2))
            ),
            angle_targets=tuple(
                target
                for target in restraints.angle_targets
                if target.center_atom_name not in centers
            ),
            plane_targets=tuple(
                target
                for target in restraints.plane_targets
                if not centers.intersection(target.atom_names)
            ),
        )

    def _require_graph_and_charge(self) -> None:
        structure = self._structure
        constitution = structure.constitution
        beta_carbon = AtomRef(self.sulfur.residue_id, "CB")
        for first, second in self.context_bonds():
            first_index = constitution.atom_index(first)
            second_index = constitution.atom_index(second)
            existing = structure.topology.bond_between(first_index, second_index)
            if existing is not None:
                existing.with_resolved_chemistry(
                    TopologyBond(first_index, second_index)
                )

        definitions = {
            frozenset((b.atom_name_1, b.atom_name_2)): b
            for b in self.template.definition.bonds
        }
        semantics = self.template.hydrogen_semantics
        assert isinstance(semantics, IdealGeometryHydrogenSemantics)
        charges = {
            AtomRef(self.ligand_id, atom.atom_name): atom.formal_charge
            for atom in semantics.component.heavy_atoms()
        }
        charges.update({self.sulfur: 0, beta_carbon: 0})
        for ref in (
            *(
                AtomRef(self.ligand_id, name)
                for name in self.template.expected_heavy_atom_names()
            ),
            self.sulfur,
            beta_carbon,
        ):
            index = constitution.atom_index(ref)
            if structure.topology.formal_charge(index) not in (None, charges[ref]):
                raise ValueError(
                    "source charge disagrees with selected Michael product chemistry"
                )
            observation = structure.provenance.ingress.observation
            original_index = (
                None
                if observation is None
                else observation.constitution.resolve_atom_index(ref)
            )
            if (
                original_index is not None
                and observation is not None
                and observation.topology.formal_charge(original_index)
                not in (None, charges[ref])
            ):
                raise ValueError(
                    "original charge disagrees with selected Michael product chemistry"
                )
        for bond in structure.topology.bonds:
            if bond.relationship_type in (
                BondRelationshipType.HYDROGEN_BOND,
                BondRelationshipType.METAL_COORDINATION,
            ):
                continue
            refs = tuple(constitution.atom_ref_at(i) for i in bond.endpoint_pair())
            if self.sulfur in refs:
                other = next(ref for ref in refs if ref != self.sulfur)
                if constitution.atom_site_at(
                    constitution.atom_index(other)
                ).is_hydrogen():
                    if other.residue_id != self.sulfur.residue_id:
                        raise ValueError(
                            "cysteine sulfur has an unsupported external H attachment"
                        )
                    continue
                allowed = {
                    beta_carbon,
                    AtomRef(self.ligand_id, self._warhead.beta),
                }
                if other not in allowed:
                    raise ValueError(
                        "cysteine sulfur has an unsupported additional attachment"
                    )
                expected = TopologyBond(*bond.endpoint_pair(), order=1)
            elif beta_carbon in refs:
                other = next(ref for ref in refs if ref != beta_carbon)
                if constitution.atom_site_at(
                    constitution.atom_index(other)
                ).is_hydrogen():
                    continue
                if (
                    other != AtomRef(self.sulfur.residue_id, "CA")
                    or bond.order not in (None, 1)
                    or bond.aromatic
                ):
                    raise ValueError(
                        "cysteine beta carbon has unsupported saturated chemistry"
                    )
                expected = TopologyBond(*bond.endpoint_pair(), order=1)
            elif all(ref.residue_id == self.ligand_id for ref in refs):
                if any(
                    constitution.atom_site_at(i).is_hydrogen()
                    for i in bond.endpoint_pair()
                ):
                    continue
                definition = definitions.get(frozenset(ref.atom_name for ref in refs))
                if definition is None:
                    raise ValueError(
                        "current Michael graph changes the supported skeleton"
                    )
                expected = TopologyBond(
                    *bond.endpoint_pair(),
                    order=definition.order,
                    aromatic=definition.aromatic,
                )
            else:
                continue
            if bond.provenance is BondProvenance.SOURCE_EXPLICIT:
                bond.with_resolved_chemistry(expected)
            elif bond.relationship_type not in (
                BondRelationshipType.COVALENT,
                BondRelationshipType.UNKNOWN,
            ):
                raise ValueError("unsupported Michael product relationship")

    def _require_original_hydrogens(
        self, partner: ResidueSite, isolated: ResidueTemplate
    ) -> None:
        observation = self._structure.provenance.ingress.observation
        if observation is None:
            return
        if self._thiol_hydrogen_names(observation.constitution, observation.topology):
            raise ValueError(
                "original sulfur-bound H conflicts with thioether chemistry"
            )
        for residue_id, template in (
            (self.ligand_id, self.template),
            (partner.residue_id, isolated),
        ):
            original = observation.constitution.residue_or_ligand(residue_id)
            if original is None:
                continue
            anchors = dict(
                template.template_hydrogen_anchor_by_name(
                    template.expected_hydrogen_atom_names()
                )
            )
            if residue_id == partner.residue_id:
                anchors.update(
                    PolymerMicrostateSite(
                        isolated, PolymerChemicalSite.SIDECHAIN
                    ).named_hydrogen_parents(original)
                )
            anchors.update(
                observation.topology.residue_hydrogen_anchor_by_name(
                    constitution=observation.constitution,
                    residue_index=observation.constitution.residue_index(residue_id),
                )
            )
            counts: Counter[str] = Counter()
            allowed_counts = Counter(
                template.template_hydrogen_anchor_by_name(
                    template.expected_hydrogen_atom_names()
                ).values()
            )
            for atom in original.atom_sites:
                if not atom.is_hydrogen():
                    continue
                parent = anchors.get(atom.name)
                if residue_id == partner.residue_id and parent != "SG":
                    continue
                if parent is None:
                    raise ValueError(
                        "original Michael-product H has no supported parent"
                    )
                if parent == "SG" or observation.topology.formal_charge(
                    observation.constitution.atom_index(AtomRef(residue_id, atom.name))
                ) not in (None, 0):
                    raise ValueError(
                        "original H conflicts with Michael thioether chemistry"
                    )
                counts[parent] += 1
            if any(count > allowed_counts[parent] for parent, count in counts.items()):
                raise ValueError(
                    "original Michael-product H exceed the product inventory"
                )

    def _thiol_hydrogen_names(
        self, constitution: StructureConstitution, topology: StructureTopology
    ) -> frozenset[str]:
        sulfur_index = constitution.resolve_atom_index(self.sulfur)
        if sulfur_index is None:
            return frozenset()
        return frozenset(
            constitution.atom_ref_at(other).atom_name
            for bond in topology.bonds
            if sulfur_index in bond.endpoint_pair()
            and bond.relationship_type
            not in (
                BondRelationshipType.HYDROGEN_BOND,
                BondRelationshipType.METAL_COORDINATION,
            )
            for other in bond.endpoint_pair()
            if other != sulfur_index and constitution.atom_site_at(other).is_hydrogen()
        )

    def excess_generated_thiol_hydrogen_names(self) -> tuple[str, ...]:
        """Identify removable generated H on sulfur, never original H.

        Returns
        -------
        tuple[str, ...]
            Current cysteine H identities anchored to SG. Source incompatibility
            is rejected at binding construction, before any mutation.

        Raises
        ------
        ValueError
            A charged or multiply attached current thiol H cannot be removed.
        """
        structure = self._structure
        residue = structure.constitution.residue_or_ligand(self.sulfur.residue_id)
        assert residue is not None
        anchors = PolymerMicrostateSite(
            self.cysteine_template, PolymerChemicalSite.SIDECHAIN
        ).named_hydrogen_parents(residue)
        anchors.update(
            structure.topology.residue_hydrogen_anchor_by_name(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(
                    self.sulfur.residue_id
                ),
            )
        )
        connected_names = self._thiol_hydrogen_names(
            structure.constitution, structure.topology
        )
        names = tuple(
            atom.name
            for atom in residue.atom_sites
            if atom.is_hydrogen()
            and (
                anchors.get(atom.name) == "SG"
                or atom.name == "HG"
                or atom.name in connected_names
            )
        )
        observation = structure.provenance.ingress.observation
        original = (
            None
            if observation is None
            else observation.constitution.residue_or_ligand(residue.residue_id)
        )
        if names and (
            observation is None
            or original is None
            or any(original.has_atom_site(name) for name in names)
        ):
            raise ValueError("original or unproven cysteine thiol H cannot be removed")
        for name in names:
            index = structure.constitution.atom_index(AtomRef(residue.residue_id, name))
            if structure.topology.formal_charge(index) not in (None, 0):
                raise ValueError("current cysteine thiol H has a formal charge")
            for bond in structure.topology.bonds:
                if (
                    index not in bond.endpoint_pair()
                    or bond.relationship_type is BondRelationshipType.HYDROGEN_BOND
                ):
                    continue
                other = next(i for i in bond.endpoint_pair() if i != index)
                if (
                    structure.constitution.atom_ref_at(other) != self.sulfur
                    or bond.order not in (None, 1)
                    or bond.relationship_type
                    not in (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
                ):
                    raise ValueError(
                        "current cysteine thiol H has contradictory attachments"
                    )
        return names

    def cysteine_is_realized(self) -> bool:
        """Check the product boundary before replacing standard thiol preparation.

        Returns
        -------
        bool
            Whether neutral SG has two single heavy bonds, no thiol H, and the
            ligand alkene is reduced. Selection alone is not conformance.
        """
        if self.excess_generated_thiol_hydrogen_names():
            return False
        constitution = self._structure.constitution
        pairs = (
            (self.sulfur, AtomRef(self.sulfur.residue_id, "CB")),
            self.external_bonds()[0],
            (
                AtomRef(self.ligand_id, self._warhead.alpha),
                AtomRef(self.ligand_id, self._warhead.beta),
            ),
        )
        return all(
            (
                bond := self._structure.topology.bond_between(
                    *(constitution.atom_index(ref) for ref in pair)
                )
            )
            is not None
            and bond.relationship_type is BondRelationshipType.COVALENT
            and bond.order == 1
            and not bond.aromatic
            for pair in pairs
        )

    def require_ligand_stereochemistry(
        self,
        geometry: ResidueGeometry,
        hydrogen_anchor_by_name: Mapping[str, str] | None = None,
    ) -> None:
        """Compare current mapped centers with CCD geometry, including present H.

        Parameters
        ----------
        geometry : ResidueGeometry
            Current or proposed product coordinates; no atom is moved by this check.
        hydrogen_anchor_by_name : Mapping[str, str] or None
            Parent mapping after source H/isotope reconciliation. Without an
            explicit mapping, current topology and template naming are used.

        Raises
        ------
        ValueError
            A named tetrahedral center is degenerate or reverses its CCD reference.
        """
        semantics = self.template.hydrogen_semantics
        assert isinstance(semantics, IdealGeometryHydrogenSemantics)
        component = semantics.component
        anchors = dict(
            self.template.template_hydrogen_anchor_by_name(
                self.template.expected_hydrogen_atom_names()
            )
        )
        if hydrogen_anchor_by_name is None:
            anchors.update(
                self._structure.topology.residue_hydrogen_anchor_by_name(
                    constitution=self._structure.constitution,
                    residue_index=self._structure.constitution.residue_index(
                        self.ligand_id
                    ),
                )
            )
        else:
            anchors.update(hydrogen_anchor_by_name)
        for declared_center in component.heavy_atoms():
            if declared_center.stereo_config is None:
                continue
            name = declared_center.atom_name
            center = component.atom(name)
            assert center is not None and center.ideal_position is not None
            neighbors = sorted(component.bonded_atom_names(name))
            heavy = [
                n
                for n in neighbors
                if (atom := component.atom(n)) is not None and not atom.is_hydrogen()
            ]
            if (
                len(heavy) not in (3, 4)
                or len(neighbors) != 4
                or any(
                    b.order != 1 or b.aromatic
                    for b in component.bonds
                    if name in (b.atom_name_1, b.atom_name_2)
                )
            ):
                raise ValueError("unsupported Michael tetrahedral neighborhood")
            ideal = []
            actual = []
            for neighbor_name in heavy:
                atom = component.atom(neighbor_name)
                assert atom is not None and atom.ideal_position is not None
                ideal.append(np.asarray(atom.ideal_position) - center.ideal_position)
                actual.append(
                    geometry.position(neighbor_name).to_array()
                    - geometry.position(name).to_array()
                )
            actual_hydrogens = [
                n
                for n, parent in anchors.items()
                if parent == name and geometry.has_atom(n)
            ]
            ideal_hydrogen = next(
                (component.atom(n) for n in neighbors if n not in heavy), None
            )
            if len(heavy) == 4:
                ideal = [vector - ideal[-1] for vector in ideal[:3]]
                actual = [vector - actual[-1] for vector in actual[:3]]
                if actual_hydrogens:
                    raise ValueError(
                        "observed H conflicts with a four-heavy stereocenter"
                    )
            for hydrogen in (None, *actual_hydrogens):
                expected = ideal
                observed = actual
                if hydrogen is not None:
                    atom = ideal_hydrogen
                    assert atom is not None and atom.ideal_position is not None
                    ideal_h = np.asarray(atom.ideal_position) - center.ideal_position
                    actual_h = (
                        geometry.position(hydrogen).to_array()
                        - geometry.position(name).to_array()
                    )
                    expected = [vector - ideal_h for vector in ideal]
                    observed = [vector - actual_h for vector in actual]
                signs = []
                for vectors in (expected, observed):
                    units = [normalized_vector_or_none(vector) for vector in vectors]
                    if any(unit is None for unit in units):
                        raise ValueError("degenerate Michael stereochemistry")
                    first, second, third = units
                    assert (
                        first is not None and second is not None and third is not None
                    )
                    volume = float(np.dot(first, np.cross(second, third)))
                    if not isfinite(volume) or volume == 0:
                        raise ValueError("degenerate Michael stereochemistry")
                    signs.append(volume > 0)
                if signs[0] != signs[1]:
                    raise ValueError(f"coordinate Michael stereo disagrees at {name}")
