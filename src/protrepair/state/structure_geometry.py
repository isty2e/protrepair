"""Intrinsic geometry facts for canonical structures."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics import ClashPolicy
from protrepair.diagnostics.clashes import StericClash, detect_clashes
from protrepair.state.structure_axes import (
    ClashObservationMode,
    ClashPresenceState,
    HydrogenCoverageState,
    OrientationCorrectionEligibilityState,
    StereochemistryState,
)
from protrepair.state.structure_clash_observation import (
    active_chemistry_readiness_facts,
    clash_state_from_count,
    heavy_heavy_clashes,
    protein_self_clashes,
)
from protrepair.state.structure_observation_rules import stereochemistry_state
from protrepair.state.structure_readiness import (
    StructureChemistryReadinessFacts,
)
from protrepair.structure.aggregate import ProteinStructure

__all__ = ["StructureIntrinsicGeometryFacts"]


@dataclass(frozen=True, slots=True)
class StructureIntrinsicGeometryFacts:
    """Whole-structure intrinsic geometry truth without ligand-aware axes."""

    carrier: ProteinStructure
    protein_self_clash_state: ClashPresenceState
    protein_self_clash_count: int
    protein_self_clash_observation_mode: ClashObservationMode
    observed_heavy_atom_self_clash_count: int
    observed_hydrogen_inclusive_self_clash_count: int | None
    orientation_correction_eligibility_state: OrientationCorrectionEligibilityState
    stereochemistry_state: StereochemistryState

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
        chemistry_readiness_facts: "StructureChemistryReadinessFacts | None" = None,
        orientation_correction_eligibility_state: (
            OrientationCorrectionEligibilityState
        ) = OrientationCorrectionEligibilityState.NOT_ELIGIBLE,
    ) -> "StructureIntrinsicGeometryFacts":
        """Derive intrinsic geometry truth from one canonical structure."""

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
        hydrogen_inclusive_protein_self_clashes: tuple[StericClash, ...] | None = (
            None
        )
        if (
            active_chemistry_facts.hydrogen_coverage_state
            is HydrogenCoverageState.COMPLETE
        ):
            hydrogen_inclusive_protein_self_clashes = protein_self_clashes(
                detect_clashes(
                    structure,
                    component_library=library,
                    policy=ClashPolicy(
                        include_hydrogens=True,
                        include_ligands=False,
                    ),
                ).clashes
            )
            heavy_atom_protein_self_clashes = heavy_heavy_clashes(
                structure,
                hydrogen_inclusive_protein_self_clashes,
            )
            protein_clashes = hydrogen_inclusive_protein_self_clashes
            protein_self_clash_observation_mode = (
                ClashObservationMode.ALL_ATOM_COMPLETE
            )
        else:
            heavy_atom_protein_self_clashes = protein_self_clashes(
                detect_clashes(
                    structure,
                    component_library=library,
                    policy=ClashPolicy(
                        include_hydrogens=False,
                        include_ligands=False,
                    ),
                ).clashes
            )
            protein_clashes = heavy_atom_protein_self_clashes
            protein_self_clash_observation_mode = (
                ClashObservationMode.HEAVY_ATOM_LOWER_BOUND
            )
        return cls(
            carrier=structure,
            protein_self_clash_state=clash_state_from_count(
                len(protein_clashes),
            ),
            protein_self_clash_count=len(protein_clashes),
            protein_self_clash_observation_mode=protein_self_clash_observation_mode,
            observed_heavy_atom_self_clash_count=len(
                heavy_atom_protein_self_clashes
            ),
            observed_hydrogen_inclusive_self_clash_count=(
                None
                if hydrogen_inclusive_protein_self_clashes is None
                else len(hydrogen_inclusive_protein_self_clashes)
            ),
            orientation_correction_eligibility_state=(
                orientation_correction_eligibility_state
            ),
            stereochemistry_state=stereochemistry_state(
                structure,
                residues=tuple(
                    structure.constitution.iter_residues(include_ligands=False)
                ),
                component_library=library,
            ),
        )
