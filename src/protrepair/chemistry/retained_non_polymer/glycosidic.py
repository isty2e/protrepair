"""Contextual pyranose inventory and stereo at deposited glycosidic links."""

from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite

import numpy as np

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.pyranose import PyranoseScaffold
from protrepair.chemistry.component.semantics import (
    HeavyAtomSemantics,
    HydrogenSemantics,
    IdealGeometryHeavyAtomSemantics,
    IdealGeometryHydrogenSemantics,
)
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.inference.idealized_stereochemistry import (
    require_compatible_component_reference,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.geometry.placement_vector import (
    PlacementVector,
    normalized_vector_or_none,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType, TopologyBond

_NONCOVALENT = frozenset(
    {BondRelationshipType.HYDROGEN_BOND, BondRelationshipType.METAL_COORDINATION}
)
_ExternalLink = tuple[TopologyBond, AtomRef, AtomRef]


@dataclass(frozen=True, slots=True, init=False)
class GlycosidicTemplateBinding:
    """Interpret a supported covalent glycan without changing deposited atoms.

    Parameters
    ----------
    structure : ProteinStructure
        Current graph, coordinates and immutable original chemical observations.
    residue : ResidueSite
        Graph-qualified pyranose or a linked polymer ASN.
    template : ResidueTemplate
        Isolated component chemistry, never modified in place.
    component_library : ComponentLibrary
        Chemistry for both ends of every link in the connected glycan.

    Notes
    -----
    A donor's leaving O/H inventory is replaced by the actual N/O partner. Acceptor
    hydroxyl H or one ASN ND2 H is excluded from the application, not deleted
    from source. A conflicting connected partner prevents admission.
    Other component families and unlinked templates retain their semantics.
    This binding supports graph-qualified pyranose alcohol acceptors and polymer
    ASN ND2; other glycoconjugates require their own chemical interpretation.
    An acceptor-only methyl glycoside retains its anomeric O--methyl group;
    only an anomeric donor requires a leaving hydroxyl in its isolated template.
    Sugar geometry must come from an idealized component; static heavy-atom
    placement programs are not adapted to substituted anchors.

    Raises
    ------
    ValueError
        Link type, scaffold, original H, charge or mapped 3D stereo is unsupported
        or contradictory. Placement and topology publication belong to transformers.
    """

    template: ResidueTemplate
    external_bonds: tuple[tuple[AtomRef, AtomRef], ...]
    amide_partner: AtomRef | None
    _structure: ProteinStructure = field(repr=False)
    _residue_id: ResidueId
    _pyranose: PyranoseScaffold | None = field(repr=False)
    _donor_partner: AtomRef | None

    def __init__(
        self,
        structure: ProteinStructure,
        residue: ResidueSite,
        template: ResidueTemplate,
        component_library: ComponentLibrary,
    ) -> None:
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "external_bonds", ())
        object.__setattr__(self, "amide_partner", None)
        object.__setattr__(self, "_structure", structure)
        object.__setattr__(self, "_residue_id", residue.residue_id)
        object.__setattr__(self, "_pyranose", None)
        object.__setattr__(self, "_donor_partner", None)
        constitution = structure.constitution
        initial_links = RetainedCovalentContext(
            structure, residue.residue_id
        ).attachments
        if residue.component_id == "ASN":
            initial_links = tuple(
                link for link in initial_links if link[1].atom_name == "ND2"
            )
        if not initial_links:
            return
        roles_by_residue: dict[ResidueId, PyranoseScaffold | None] = {}

        def roles_for(
            site: ResidueSite, isolated: ResidueTemplate
        ) -> PyranoseScaffold | None:
            if site.residue_id not in roles_by_residue:
                semantics = isolated.hydrogen_semantics
                roles = None
                if isinstance(semantics, IdealGeometryHydrogenSemantics):
                    try:
                        roles = PyranoseScaffold(semantics.component)
                    except ValueError:
                        pass
                roles_by_residue[site.residue_id] = roles
            return roles_by_residue[site.residue_id]

        if residue.component_id != "ASN" and roles_for(residue, template) is None:
            return

        queue = deque((residue.residue_id,))
        visited: set[ResidueId] = set()
        while queue:
            residue_id = queue.popleft()
            if residue_id in visited:
                continue
            visited.add(residue_id)
            site = constitution.residue_or_ligand(residue_id)
            assert site is not None
            isolated = (
                template
                if residue_id == residue.residue_id
                else component_library.get(site.component_id)
            )
            if isolated is None:
                raise ValueError("glycosidic partner has no supported template")
            links = RetainedCovalentContext(structure, residue_id).attachments
            if site.component_id == "ASN":
                links = tuple(link for link in links if link[1].atom_name == "ND2")
            if not links or len({inside.atom_name for _, inside, _ in links}) != len(
                links
            ):
                raise ValueError("glycosidic anchor requires exactly one external bond")
            roles = roles_for(site, isolated)
            for bond, inside, outside in links:
                partner = constitution.residue_or_ligand(outside.residue_id)
                assert partner is not None
                partner_template = (
                    template
                    if partner.residue_id == residue.residue_id
                    else component_library.get(partner.component_id)
                )
                if partner_template is None:
                    raise ValueError("glycosidic partner has no supported template")
                partner_roles = roles_for(partner, partner_template)
                self._require_link(inside, outside, bond, roles, partner_roles)
                queue.append(outside.residue_id)

            if roles is not None:
                require_compatible_component_reference(roles.component)
                projected, donor = self._bind_sugar(site, isolated, links, roles)
                anchors = PolymerMicrostateSite(
                    isolated, PolymerChemicalSite.SIDECHAIN
                ).named_hydrogen_parents(site)
                anchors.update(
                    structure.topology.residue_hydrogen_anchor_by_name(
                        constitution=constitution,
                        residue_index=constitution.residue_index(site.residue_id),
                    )
                )
                self._require_sugar_stereo(
                    site.residue_id,
                    roles,
                    donor,
                    structure.residue_geometry(
                        constitution.residue_index(site.residue_id)
                    ),
                    anchors,
                )
            else:
                projected = self._bind_asparagine(site, isolated)
                donor = None
            self._require_hydrogen_compatibility(site, isolated, projected)
            if residue_id == residue.residue_id:
                object.__setattr__(self, "template", projected)
                object.__setattr__(
                    self,
                    "external_bonds",
                    tuple((inside, outside) for _, inside, outside in links),
                )
                object.__setattr__(self, "_pyranose", roles)
                object.__setattr__(self, "_donor_partner", donor)
                if site.component_id == "ASN":
                    object.__setattr__(self, "amide_partner", links[0][2])

    def _require_link(
        self,
        inside: AtomRef,
        outside: AtomRef,
        bond: TopologyBond,
        left_roles: PyranoseScaffold | None,
        right_roles: PyranoseScaffold | None,
    ) -> None:
        if (
            bond.relationship_type is not BondRelationshipType.COVALENT
            or bond.order not in (None, 1)
            or bond.aromatic
        ):
            raise ValueError("glycosidic attachment must be a covalent single bond")
        constitution = self._structure.constitution
        left = constitution.residue_or_ligand(inside.residue_id)
        right = constitution.residue_or_ligand(outside.residue_id)
        assert left is not None and right is not None
        donors = tuple(
            (endpoint, partner, partner_site, donor_roles, acceptor_roles)
            for endpoint, partner, partner_site, donor_roles, acceptor_roles in (
                (inside, outside, right, left_roles, right_roles),
                (outside, inside, left, right_roles, left_roles),
            )
            if donor_roles is not None
            and endpoint.atom_name == donor_roles.anomeric_carbon
        )
        if len(donors) != 1:
            raise ValueError(
                "glycosidic attachment requires one unambiguous anomeric donor"
            )
        donor, acceptor, acceptor_site, donor_roles, acceptor_roles = donors[0]
        assert donor_roles is not None
        donor_roles.require_donor(donor.atom_name)

        if acceptor_site.component_id == "ASN":
            polymer_count = len(constitution.residue_slots) - len(constitution.ligands)
            if (
                acceptor.atom_name != "ND2"
                or constitution.residue_index(acceptor.residue_id).value
                >= polymer_count
            ):
                raise ValueError("N-glycosylation requires polymer ASN ND2")
        elif acceptor_roles is not None:
            acceptor_roles.require_acceptor(acceptor.atom_name)
        else:
            raise ValueError("unsupported glycosidic acceptor component")
        for ref, element in (
            (donor, "C"),
            (acceptor, "N" if acceptor_site.component_id == "ASN" else "O"),
        ):
            atom = constitution.resolve_atom_site(ref)
            if atom is None or atom.element != element:
                raise ValueError(
                    "glycosidic endpoint element disagrees with its template"
                )
            if self._structure.topology.formal_charge(
                constitution.atom_index(ref)
            ) not in (None, 0):
                raise ValueError("glycosidic endpoint has a conflicting formal charge")

    def _bind_sugar(
        self,
        residue: ResidueSite,
        template: ResidueTemplate,
        links: tuple[_ExternalLink, ...],
        roles: PyranoseScaffold,
    ) -> tuple[ResidueTemplate, AtomRef | None]:
        semantics = template.hydrogen_semantics
        if not isinstance(semantics, IdealGeometryHydrogenSemantics):
            raise ValueError("glycosidic donor requires idealized pyranose chemistry")
        component = roles.component
        if (
            set(template.definition.bonds) != set(component.heavy_bonds())
            or set(template.expected_heavy_atom_names())
            != set(component.heavy_atom_names())
            or any(
                template.definition.formal_charges.get(atom.atom_name, 0)
                != atom.formal_charge
                for atom in component.heavy_atoms()
            )
        ):
            raise ValueError(
                "pyranose template and H chemistry disagree on the precursor graph"
            )
        donor = next(
            (
                outside
                for _, inside, outside in links
                if inside.atom_name == roles.anomeric_carbon
            ),
            None,
        )
        projected = roles.project(tuple(inside.atom_name for _, inside, _ in links))
        heavy_semantics = template.heavy_atom_semantics
        if isinstance(heavy_semantics, HeavyAtomSemantics):
            raise ValueError(
                "glycosidic binding does not adapt static heavy-atom programs"
            )
        if isinstance(heavy_semantics, IdealGeometryHeavyAtomSemantics):
            heavy_semantics = replace(heavy_semantics, component=projected)
        projected_template = replace(
            template,
            definition=replace(
                template.definition,
                atom_names=projected.heavy_atom_names(),
                bonds=projected.heavy_bonds(),
                formal_charges={
                    atom.atom_name: atom.formal_charge
                    for atom in projected.heavy_atoms()
                },
            ),
            preferred_atom_order=projected.heavy_atom_names(),
            heavy_atom_semantics=heavy_semantics,
            hydrogen_semantics=IdealGeometryHydrogenSemantics(projected),
        )
        if not projected_template.has_complete_heavy_scaffold(residue):
            raise ValueError(
                "glycosidic chemistry requires a complete compatible heavy scaffold"
            )
        for atom in projected.heavy_atoms():
            index = self._structure.constitution.atom_index(
                AtomRef(residue.residue_id, atom.atom_name)
            )
            if self._structure.topology.formal_charge(index) not in (
                None,
                atom.formal_charge,
            ):
                raise ValueError(
                    "current charge disagrees with selected pyranose chemistry"
                )
        observation = self._structure.provenance.ingress.observation
        if observation is not None:
            original = observation.constitution.residue_or_ligand(residue.residue_id)
            excluded = set(component.heavy_atom_names()) - set(
                projected.heavy_atom_names()
            )
            if original is not None and excluded.intersection(
                original.atom_site_names()
            ):
                raise ValueError(
                    "original leaving atoms cannot be discarded by glycosidic binding"
                )
            for atom in projected.heavy_atoms():
                ref = AtomRef(residue.residue_id, atom.atom_name)
                original_index = observation.constitution.resolve_atom_index(ref)
                if original_index is not None and observation.topology.formal_charge(
                    original_index
                ) not in (None, atom.formal_charge):
                    raise ValueError(
                        "original charge disagrees with selected pyranose chemistry"
                    )
        return projected_template, donor

    def _bind_asparagine(
        self, residue: ResidueSite, template: ResidueTemplate
    ) -> ResidueTemplate:
        if not set(template.expected_polymer_heavy_atom_names()).issubset(
            residue.atom_site_names()
        ):
            raise ValueError("glycosylated ASN requires a complete heavy scaffold")
        amide_names = {"CG", "OD1", "ND2"}
        expected_bonds = {
            frozenset(("CB", "CG")): 1,
            frozenset(("CG", "OD1")): 2,
            frozenset(("CG", "ND2")): 1,
        }
        amide_bonds = tuple(
            bond
            for bond in template.definition.bonds
            if amide_names.intersection((bond.atom_name_1, bond.atom_name_2))
        )
        if {
            frozenset((bond.atom_name_1, bond.atom_name_2)) for bond in amide_bonds
        } != expected_bonds.keys() or any(
            bond.aromatic
            or bond.order
            != expected_bonds[frozenset((bond.atom_name_1, bond.atom_name_2))]
            for bond in amide_bonds
        ):
            raise ValueError("glycosylated ASN requires neutral carboxamide chemistry")

        constitution = self._structure.constitution
        for bond in self._structure.topology.bonds:
            if bond.relationship_type in _NONCOVALENT:
                continue
            refs = tuple(constitution.atom_ref_at(i) for i in bond.endpoint_pair())
            if not any(
                ref.residue_id == residue.residue_id and ref.atom_name in amide_names
                for ref in refs
            ):
                continue
            if any(
                constitution.atom_site_at(i).is_hydrogen() for i in bond.endpoint_pair()
            ):
                continue
            if refs[0].residue_id != refs[1].residue_id:
                inside = next(
                    ref for ref in refs if ref.residue_id == residue.residue_id
                )
                if inside.atom_name != "ND2":
                    raise ValueError("unsupported external ASN carbonyl attachment")
                continue
            expected = expected_bonds.get(frozenset(ref.atom_name for ref in refs))
            if (
                expected is None
                or bond.relationship_type
                not in (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
                or bond.order not in (None, expected)
                or bond.aromatic
            ):
                raise ValueError("current ASN graph contradicts carboxamide chemistry")
        for name, element in (("CG", "C"), ("OD1", "O"), ("ND2", "N")):
            atom = residue.atom_site(name)
            index = self._structure.constitution.atom_index(
                AtomRef(residue.residue_id, name)
            )
            if (
                atom.element != element
                or self._structure.topology.formal_charge(index) not in (None, 0)
                or template.definition.formal_charges.get(name, 0) != 0
            ):
                raise ValueError(
                    "source ASN amide disagrees with neutral glycosylation chemistry"
                )
        semantics = template.hydrogen_semantics
        if not isinstance(semantics, HydrogenSemantics):
            raise ValueError("glycosylated ASN requires supported fixed-H semantics")
        names = tuple(
            name
            for name, parent in semantics.template_hydrogen_anchor_by_name().items()
            if parent == "ND2"
        )
        if len(names) != 2:
            raise ValueError("isolated ASN amide must have two template H")
        excluded = names[1]
        return replace(
            template,
            hydrogen_semantics=replace(
                semantics,
                plan_with_backbone=tuple(
                    operation
                    for operation in semantics.plan_with_backbone or ()
                    if excluded not in operation[0]
                ),
                plan_without_backbone=None
                if semantics.plan_without_backbone is None
                else tuple(
                    operation
                    for operation in semantics.plan_without_backbone
                    if excluded not in operation[0]
                ),
            ),
        )

    def _require_hydrogen_compatibility(
        self,
        residue: ResidueSite,
        isolated: ResidueTemplate,
        projected: ResidueTemplate,
    ) -> None:
        current_anchors = PolymerMicrostateSite(
            isolated, PolymerChemicalSite.SIDECHAIN
        ).named_hydrogen_parents(residue)
        current_anchors.update(
            self._structure.topology.residue_hydrogen_anchor_by_name(
                constitution=self._structure.constitution,
                residue_index=self._structure.constitution.residue_index(
                    residue.residue_id
                ),
            )
        )
        for atom in residue.atom_sites:
            if not atom.is_hydrogen() or (
                residue.component_id == "ASN"
                and current_anchors.get(atom.name) != "ND2"
            ):
                continue
            index = self._structure.constitution.atom_index(
                AtomRef(residue.residue_id, atom.name)
            )
            if self._structure.topology.formal_charge(index) not in (None, 0):
                raise ValueError("current glycosidic H carries a formal charge")

        observation = self._structure.provenance.ingress.observation
        if observation is None:
            return
        original = observation.constitution.residue_or_ligand(residue.residue_id)
        if original is None:
            return
        anchors = dict(
            isolated.template_hydrogen_anchor_by_name(
                isolated.expected_hydrogen_atom_names()
            )
        )
        if residue.component_id == "ASN":
            anchors.update(
                PolymerMicrostateSite(
                    isolated, PolymerChemicalSite.SIDECHAIN
                ).named_hydrogen_parents(original)
            )
        anchors.update(
            observation.topology.residue_hydrogen_anchor_by_name(
                constitution=observation.constitution,
                residue_index=observation.constitution.residue_index(
                    residue.residue_id
                ),
            )
        )
        expected = Counter(
            projected.template_hydrogen_anchor_by_name(
                projected.expected_hydrogen_atom_names()
            ).values()
        )
        observed: Counter[str] = Counter()
        for atom in original.atom_sites:
            if not atom.is_hydrogen():
                continue
            parent = anchors.get(atom.name)
            if parent is None:
                raise ValueError(
                    "original glycosidic H has no supported parent identity"
                )
            if residue.component_id == "ASN" and parent != "ND2":
                continue
            index = observation.constitution.atom_index(
                AtomRef(residue.residue_id, atom.name)
            )
            if observation.topology.formal_charge(index) not in (None, 0):
                raise ValueError("original glycosidic H carries a formal charge")
            observed[parent] += 1
        if any(count > expected[parent] for parent, count in observed.items()):
            raise ValueError("original H exceed the linked glycosidic inventory")

    def _require_sugar_stereo(
        self,
        residue_id: ResidueId,
        roles: PyranoseScaffold,
        donor: AtomRef | None,
        geometry: ResidueGeometry | None = None,
        hydrogen_anchor_by_name: Mapping[str, str] | None = None,
    ) -> None:
        structure = self._structure
        component = roles.component
        for atom in component.heavy_atoms():
            if atom.stereo_config is None:
                continue
            if atom.element != "C" or any(
                bond.order != 1 or bond.aromatic
                for bond in component.bonds
                if atom.atom_name in (bond.atom_name_1, bond.atom_name_2)
            ):
                raise ValueError("unsupported pyranose tetrahedral bond chemistry")
            if atom.ideal_position is None:
                raise ValueError("pyranose stereo requires ideal coordinates")
            names = sorted(component.bonded_atom_names(atom.atom_name))
            neighbors = tuple(component.atom(name) for name in names)
            heavy_names = [
                neighbor.atom_name
                for neighbor in neighbors
                if neighbor is not None and not neighbor.is_hydrogen()
            ]
            if len(heavy_names) not in (3, 4) or len(names) != 4:
                raise ValueError("unsupported pyranose tetrahedral center")
            center = structure.geometry.position(
                structure.constitution.atom_index(AtomRef(residue_id, atom.atom_name))
            ).to_array()
            ideal: list[PlacementVector] = []
            actual: list[PlacementVector] = []
            for name in heavy_names:
                neighbor = component.atom_with_ideal_position(name)
                if neighbor is None:
                    raise ValueError(
                        "pyranose stereo neighbor has no ideal coordinates"
                    )
                ideal.append(np.asarray(neighbor.ideal_position) - atom.ideal_position)
                ref = (
                    donor
                    if atom.atom_name == roles.anomeric_carbon
                    and name == roles.exocyclic_oxygen
                    and donor is not None
                    else AtomRef(residue_id, name)
                )
                position = structure.geometry.position(
                    structure.constitution.atom_index(ref)
                ).to_array()
                actual.append(position - center)
            if len(heavy_names) == 4:
                ideal = [vector - ideal[-1] for vector in ideal[:3]]
                actual = [vector - actual[-1] for vector in actual[:3]]
            self._require_matching_orientation(ideal, actual, atom.atom_name)
            if geometry is None or hydrogen_anchor_by_name is None:
                continue
            h_names = tuple(
                name
                for name, parent in hydrogen_anchor_by_name.items()
                if parent == atom.atom_name and geometry.has_atom(name)
            )
            if not h_names:
                continue
            ideal_h_atoms = tuple(
                neighbor
                for neighbor in neighbors
                if neighbor is not None and neighbor.is_hydrogen()
            )
            if (
                len(h_names) != 1
                or len(ideal_h_atoms) != 1
                or ideal_h_atoms[0].ideal_position is None
            ):
                raise ValueError("unsupported pyranose stereocenter H inventory")
            h = geometry.position(h_names[0]).to_array() - center
            ideal_h = np.asarray(ideal_h_atoms[0].ideal_position) - atom.ideal_position
            self._require_matching_orientation(
                [vector - ideal_h for vector in ideal],
                [vector - h for vector in actual],
                atom.atom_name,
            )

    @staticmethod
    def _require_matching_orientation(
        ideal: list[PlacementVector], actual: list[PlacementVector], center_name: str
    ) -> None:
        volumes = []
        for vectors in (ideal, actual):
            units = []
            for vector in vectors:
                unit = normalized_vector_or_none(vector)
                if unit is None:
                    raise ValueError("degenerate pyranose stereo geometry")
                units.append(unit)
            volume = float(np.dot(units[0], np.cross(units[1], units[2])))
            if not isfinite(volume) or volume == 0:
                raise ValueError("degenerate pyranose stereo geometry")
            volumes.append(volume)
        if (volumes[0] > 0) != (volumes[1] > 0):
            raise ValueError(f"coordinate pyranose stereo disagrees at {center_name}")

    def excess_generated_hydrogen_names(self) -> tuple[str, ...]:
        """Identify non-source H on hydroxyl valences occupied by current links.

        Returns
        -------
        tuple[str, ...]
            H names a transformer may remove before regeneration. Original atom
            identities and charged H are never discarded by this projection.

        """
        observation = self._structure.provenance.ingress.observation
        roles = self._pyranose
        if observation is None or roles is None:
            return ()
        component = roles.component
        isolated_anchors = IdealGeometryHydrogenSemantics(
            component
        ).template_hydrogen_anchor_by_name()
        projected_anchors = self.template.template_hydrogen_anchor_by_name(
            self.template.expected_hydrogen_atom_names()
        )
        excluded_parents = set(isolated_anchors.values()) - set(
            projected_anchors.values()
        )
        original = observation.constitution.residue_or_ligand(self._residue_id)
        original_names = set() if original is None else set(original.atom_site_names())
        constitution = self._structure.constitution
        residue = constitution.residue_or_ligand(self._residue_id)
        assert residue is not None
        anchors = dict(isolated_anchors)
        anchors.update(
            self._structure.topology.residue_hydrogen_anchor_by_name(
                constitution=constitution,
                residue_index=constitution.residue_index(self._residue_id),
            )
        )
        removable = []
        for atom in residue.atom_sites:
            if (
                not atom.is_hydrogen()
                or atom.name in original_names
                or anchors.get(atom.name) not in excluded_parents
            ):
                continue
            removable.append(atom.name)
        return tuple(removable)

    def require_hydrogen_stereochemistry(
        self, geometry: ResidueGeometry, hydrogen_anchor_by_name: Mapping[str, str]
    ) -> None:
        """Check a proposed H geometry before a transformer publishes it.

        Parameters
        ----------
        geometry : ResidueGeometry
            Candidate local H coordinates on the unchanged heavy scaffold.
        hydrogen_anchor_by_name : Mapping[str, str]
            Admitted candidate H/D/T identities and their actual heavy parents.

        Raises
        ------
        ValueError
            A present template stereocenter H reverses the mapped tetrahedron.
        """
        if self._pyranose is not None:
            self._require_sugar_stereo(
                self._residue_id,
                self._pyranose,
                self._donor_partner,
                geometry,
                hydrogen_anchor_by_name,
            )
