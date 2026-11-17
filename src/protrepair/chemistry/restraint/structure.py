"""Snapshot-bound selection of chemically applicable residue restraints."""

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.restraint.template import ResidueRestraintTemplate
from protrepair.chemistry.retained_non_polymer.michael_adduct import (
    CysteineMichaelAdductBinding,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_restraint_library,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.topology import BondRelationshipType


class StructureRestraints:
    """Bind dictionary applicability to one immutable chemical snapshot.

    Parameters
    ----------
    structure : ProteinStructure
        Current atoms, topology and original/applied chemistry evidence.
    component_library : ComponentLibrary
        Active chemical definitions, including caller overrides.
    restraint_library : RestraintLibrary
        Static and conditional targets. No built-in replacement is injected.

    Notes
    -----
    Selection does not prepare H or infer protonation from coordinates. Unknown
    chemistry retains static targets only. Consumers own tolerances/FF strength.
    """

    def __init__(
        self,
        structure: ProteinStructure,
        component_library: ComponentLibrary,
        restraint_library: RestraintLibrary,
    ) -> None:
        self._structure = structure
        self._components = component_library
        self._restraints = restraint_library
        self._context: PolymerMicrostateContext | None = None
        self._resolved: dict[
            ResidueId, tuple[ResidueRestraintTemplate | None, frozenset[str]]
        ] = {}

    def for_residue(self, residue_id: ResidueId) -> ResidueRestraintTemplate | None:
        """Return flat applicable targets for this residue and snapshot.

        Parameters
        ----------
        residue_id : ResidueId
            Canonical residue identity.

        Returns
        -------
        ResidueRestraintTemplate or None
            Static targets plus a chemically supported variant, or no dictionary.

        Raises
        ------
        ResidueNotFoundError
            The residue is absent from the bound snapshot.
        """
        return self.bound_targets(residue_id)[0]

    def bound_targets(
        self, residue_id: ResidueId
    ) -> tuple[ResidueRestraintTemplate | None, frozenset[str]]:
        """Return usable targets and atoms with unresolved conditional evidence."""

        if residue_id in self._resolved:
            return self._resolved[residue_id]
        constitution = self._structure.constitution
        residue = constitution.residue_site_at(constitution.residue_index(residue_id))
        template = self._restraints.get(residue.component_id)
        static_template = template
        unresolved: frozenset[str] = frozenset()
        references = self._restraints.component_references.get(
            self._restraints.normalize_component_id(residue.component_id), ()
        )
        if template is not None and references:
            scope = frozenset(
                name for reference in references for name in reference.atom_names
            )
            observed = self._component_state(residue_id, scope)
            if observed is not None:
                reference = next(
                    (entry for entry in references if entry.matches(observed)), None
                )
                if reference is not None:
                    template = reference.apply_to(template)
                else:
                    unresolved = scope
            else:
                unresolved = scope
        if template is not None and template.sidechain_variants:
            component = self._components.get(residue.component_id)
            graph = None
            if component is not None:
                if self._context is None:
                    self._context = PolymerMicrostateContext(self._structure)
                site = PolymerMicrostateSite(component, PolymerChemicalSite.SIDECHAIN)
                resolution = self._context.resolve(residue_id, site)
                graph = self._context.realized_heavy_graph(residue_id, site, resolution)
            template, unresolved_variant = template.resolve_sidechain_graph(graph)
            unresolved |= unresolved_variant
        if (
            template is not None
            and static_template
            == build_bundled_retained_non_polymer_restraint_library().get(
                residue.component_id
            )
            and self._structure.topology.inter_residue_bonds_for_residue(
                constitution=constitution,
                residue_index=constitution.residue_index(residue_id),
            )
        ):
            component = self._components.get(residue.component_id)
            if component is not None:
                try:
                    binding = CysteineMichaelAdductBinding(
                        self._structure, residue, component, self._components
                    )
                except ValueError:
                    pass
                else:
                    template = binding.applicable_restraints(template)

        self._resolved[residue_id] = (template, unresolved)
        return template, unresolved

    def _component_state(
        self, residue_id: ResidueId, scope: frozenset[str]
    ) -> IdealizedComponent | None:
        constitution = self._structure.constitution
        residue_index = constitution.residue_index(residue_id)
        residue = constitution.residue_site_at(residue_index)
        component = self._components.get(residue.component_id)
        if component is None:
            return None

        indices = constitution.atom_indices_for_residue_index(residue_index)
        local = frozenset(indices)
        parents = self._structure.topology.residue_hydrogen_anchor_by_name(
            constitution=constitution, residue_index=residue_index
        )
        if any(
            atom.is_hydrogen() and atom.name not in parents
            for atom in residue.atom_sites
        ):
            return None
        atoms: list[IdealizedComponentAtom] = []
        for index in indices:
            atom = constitution.atom_site_at(index)
            charge = self._structure.topology.formal_charge(index)
            atoms.append(
                IdealizedComponentAtom(
                    atom.name,
                    atom.element,
                    component.definition.formal_charges.get(atom.name, 0)
                    if charge is None
                    else charge,
                )
            )

        bonds = [BondDefinition(parent, child) for child, parent in parents.items()]
        incident = self._structure.topology.incident_bonds_by_atom()
        for index in indices:
            name = constitution.atom_site_at(index).name
            for bond in incident.get(index, ()):
                first, second = bond.endpoint_pair()
                if first not in local or second not in local:
                    if name in scope:
                        return None
                    continue
                if index != first:
                    continue
                left, right = (
                    constitution.atom_site_at(first),
                    constitution.atom_site_at(second),
                )
                if left.is_hydrogen() or right.is_hydrogen():
                    continue
                if left.name not in scope and right.name not in scope:
                    continue
                if (
                    bond.relationship_type is not BondRelationshipType.COVALENT
                    or bond.order is None
                ):
                    return None
                bonds.append(
                    BondDefinition(left.name, right.name, bond.order, bond.aromatic)
                )

        return IdealizedComponent(
            component.component_id,
            component.lineage_parent_component_id,
            tuple(atoms),
            tuple(bonds),
        )
