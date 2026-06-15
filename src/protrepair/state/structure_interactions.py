"""Ligand-aware interaction facts for canonical structures."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics import ClashPolicy
from protrepair.diagnostics.clashes import detect_clashes
from protrepair.state.structure_axes import (
    ClashObservationMode,
    ClashPresenceState,
    HydrogenCoverageState,
)
from protrepair.state.structure_clash_observation import (
    active_chemistry_readiness_facts,
    clash_state_from_count,
    polymer_ligand_clashes,
    total_clash_overlap_angstrom,
    worst_clash_overlap_angstrom,
)
from protrepair.state.structure_readiness import StructureChemistryReadinessFacts
from protrepair.structure.aggregate import ProteinStructure

__all__ = ["StructureInteractionFacts"]


@dataclass(frozen=True, slots=True)
class StructureInteractionFacts:
    """Whole-structure ligand-aware interaction truth without intrinsic axes."""

    carrier: ProteinStructure
    ligand_aware_clash_state: ClashPresenceState
    ligand_aware_clash_count: int
    ligand_aware_worst_overlap_angstrom: float
    ligand_aware_total_overlap_angstrom: float
    ligand_aware_clash_observation_mode: ClashObservationMode
    observed_heavy_atom_ligand_clash_count: int
    observed_heavy_atom_ligand_worst_overlap_angstrom: float
    observed_heavy_atom_ligand_total_overlap_angstrom: float
    observed_hydrogen_inclusive_ligand_clash_count: int
    observed_hydrogen_inclusive_ligand_worst_overlap_angstrom: float
    observed_hydrogen_inclusive_ligand_total_overlap_angstrom: float

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
        chemistry_readiness_facts: "StructureChemistryReadinessFacts | None" = None,
    ) -> "StructureInteractionFacts":
        """Derive ligand-aware interaction truth from one canonical structure."""

        library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        active_chemistry_facts = active_chemistry_readiness_facts(
            structure,
            component_library=library,
            chemistry_readiness_facts=chemistry_readiness_facts,
        )
        ligand_aware_clashes = polymer_ligand_clashes(
            detect_clashes(
                structure,
                component_library=library,
                policy=ClashPolicy(
                    include_hydrogens=False,
                    include_ligands=True,
                ),
            ).clashes
        )
        hydrogen_inclusive_ligand_clashes = polymer_ligand_clashes(
            detect_clashes(
                structure,
                component_library=library,
                policy=ClashPolicy(
                    include_hydrogens=True,
                    include_ligands=True,
                ),
            ).clashes
        )
        canonical_ligand_clashes = ligand_aware_clashes
        clash_observation_mode = ClashObservationMode.HEAVY_ATOM_LOWER_BOUND
        if (
            active_chemistry_facts.hydrogen_coverage_state
            is HydrogenCoverageState.COMPLETE
        ):
            canonical_ligand_clashes = hydrogen_inclusive_ligand_clashes
            clash_observation_mode = ClashObservationMode.ALL_ATOM_COMPLETE
        return cls(
            carrier=structure,
            ligand_aware_clash_state=clash_state_from_count(
                len(canonical_ligand_clashes)
            ),
            ligand_aware_clash_count=len(canonical_ligand_clashes),
            ligand_aware_worst_overlap_angstrom=worst_clash_overlap_angstrom(
                canonical_ligand_clashes
            ),
            ligand_aware_total_overlap_angstrom=total_clash_overlap_angstrom(
                canonical_ligand_clashes
            ),
            ligand_aware_clash_observation_mode=clash_observation_mode,
            observed_heavy_atom_ligand_clash_count=len(ligand_aware_clashes),
            observed_heavy_atom_ligand_worst_overlap_angstrom=(
                worst_clash_overlap_angstrom(ligand_aware_clashes)
            ),
            observed_heavy_atom_ligand_total_overlap_angstrom=(
                total_clash_overlap_angstrom(ligand_aware_clashes)
            ),
            observed_hydrogen_inclusive_ligand_clash_count=(
                len(hydrogen_inclusive_ligand_clashes)
            ),
            observed_hydrogen_inclusive_ligand_worst_overlap_angstrom=(
                worst_clash_overlap_angstrom(hydrogen_inclusive_ligand_clashes)
            ),
            observed_hydrogen_inclusive_ligand_total_overlap_angstrom=(
                total_clash_overlap_angstrom(hydrogen_inclusive_ligand_clashes)
            ),
        )
