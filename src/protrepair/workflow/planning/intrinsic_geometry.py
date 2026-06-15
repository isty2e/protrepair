"""Decision-owned intrinsic geometry facts with action eligibility overlays."""

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.scope import ResidueSetScope
from protrepair.state import (
    OrientationCorrectionEligibilityState,
    StructureChemistryReadinessFacts,
    StructureIntrinsicGeometryFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.discrete.axis_rotation import (
    select_best_discrete_correction_candidate,
)


def derive_structure_intrinsic_geometry_facts(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None,
) -> StructureIntrinsicGeometryFacts:
    """Derive intrinsic geometry facts plus correction-action eligibility."""

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    return StructureIntrinsicGeometryFacts.from_structure(
        structure,
        component_library=active_component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
        orientation_correction_eligibility_state=(
            derive_orientation_correction_eligibility_state(
                structure,
                component_library=active_component_library,
            )
        ),
    )


def derive_orientation_correction_eligibility_state(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
) -> OrientationCorrectionEligibilityState:
    """Return whether one structure admits a discrete orientation correction."""

    snapshot = ProteinStructureSnapshot.from_structure(structure)
    for residue in structure.constitution.iter_residues(include_ligands=False):
        atom_input = AtomInput(
            atom_indices=structure.constitution.atom_indices_for_residue(
                residue.residue_id
            ),
            basis=AtomInputBasis.RESIDUEWISE,
            selected_scope=ResidueSetScope(residue_ids=(residue.residue_id,)),
        )
        if (
            select_best_discrete_correction_candidate(
                snapshot=snapshot,
                atom_input=atom_input,
                structure=structure,
                residue_id=residue.residue_id,
                component_library=component_library,
            )
            is not None
        ):
            return OrientationCorrectionEligibilityState.ELIGIBLE

    return OrientationCorrectionEligibilityState.NOT_ELIGIBLE
