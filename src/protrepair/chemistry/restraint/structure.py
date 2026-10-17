"""Snapshot-bound selection of chemically applicable residue restraints."""

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
        self._resolved: dict[ResidueId, ResidueRestraintTemplate | None] = {}

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
        if residue_id in self._resolved:
            return self._resolved[residue_id]
        constitution = self._structure.constitution
        residue = constitution.residue_site_at(constitution.residue_index(residue_id))
        template = self._restraints.get(residue.component_id)
        if template is not None and template.sidechain_variants:
            component = self._components.get(residue.component_id)
            graph = None
            if component is not None:
                if self._context is None:
                    self._context = PolymerMicrostateContext(self._structure)
                site = PolymerMicrostateSite(component, PolymerChemicalSite.SIDECHAIN)
                resolution = self._context.resolve(residue_id, site)
                graph = self._context.realized_heavy_graph(residue_id, site, resolution)
            template = template.for_sidechain_graph(graph)
        if (
            template is not None
            and template
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

        self._resolved[residue_id] = template
        return template
