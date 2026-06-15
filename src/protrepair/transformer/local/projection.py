"""Current-state projections derived from local transformer scope requests."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.domain import ContinuousRelaxationRegion
from protrepair.transformer.local.models import LocalScopeSpec


@dataclass(frozen=True, slots=True)
class LocalContinuousExecutionResidueProjection:
    """Actual included polymer residues for one local continuous execution scope."""

    included_polymer_residue_ids: tuple[ResidueId, ...]

    @classmethod
    def from_scope_spec(
        cls,
        snapshot: ProteinStructureSnapshot,
        scope_spec: LocalScopeSpec,
        *,
        context_radius_angstrom: float,
        component_library: ComponentLibrary | None = None,
    ) -> "LocalContinuousExecutionResidueProjection":
        """Project one local scope request onto its actual included polymer region."""

        active_component_library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        atom_input = scope_spec.lower_to_atom_input(
            snapshot,
            component_library=active_component_library,
        )
        continuous_region = ContinuousRelaxationRegion.from_inputs(
            snapshot,
            atom_input,
            context_radius_angstrom=context_radius_angstrom,
        )
        ordered_polymer_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        for residue_index in continuous_region.included_residue_indices:
            residue_site = continuous_region.residue_site(residue_index)
            if residue_site.is_hetero or residue_site.residue_id in seen_residue_ids:
                continue

            ordered_polymer_residue_ids.append(residue_site.residue_id)
            seen_residue_ids.add(residue_site.residue_id)

        return cls(included_polymer_residue_ids=tuple(ordered_polymer_residue_ids))
