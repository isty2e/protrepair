"""Realizability predicates for continuous-relaxation bond planning."""

from collections.abc import Mapping

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RefinementError
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryResolution,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.continuous.bonds import plan_continuous_region_bonds
from protrepair.transformer.continuous.domain import ContinuousRelaxationRegion


def continuous_bond_realizability_error(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    *,
    component_library: ComponentLibrary,
    context_radius_angstrom: float = 6.0,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    retained_non_polymer_chemistry_resolution_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryResolution,
    ]
    | None = None,
) -> str | None:
    """Return the selected-scope bond-graph blocker under continuous semantics."""

    try:
        region = ContinuousRelaxationRegion.from_inputs(
            snapshot,
            atom_input,
            context_radius_angstrom=context_radius_angstrom,
        )
        return continuous_region_bond_realizability_error(
            region,
            component_library=component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            retained_non_polymer_chemistry_resolution_by_residue_id=(
                retained_non_polymer_chemistry_resolution_by_residue_id
            ),
        )
    except RefinementError as error:
        return str(error)


def continuous_region_bond_realizability_error(
    region: ContinuousRelaxationRegion,
    *,
    component_library: ComponentLibrary,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    retained_non_polymer_chemistry_resolution_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryResolution,
    ]
    | None = None,
) -> str | None:
    """Return the bond-graph blocker for one prebuilt continuous region."""

    try:
        support_by_residue_index = region.require_local_bond_planning_support(
            component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            retained_non_polymer_chemistry_resolution_by_residue_id=(
                retained_non_polymer_chemistry_resolution_by_residue_id
            ),
        )
        plan_continuous_region_bonds(
            region,
            component_library,
            support_by_residue_index=support_by_residue_index,
        )
    except RefinementError as error:
        return str(error)

    return None


def continuous_bond_graph_is_realizable(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    *,
    component_library: ComponentLibrary,
    context_radius_angstrom: float = 6.0,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
    retained_non_polymer_chemistry_resolution_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryResolution,
    ]
    | None = None,
) -> bool:
    """Return whether continuous bond-graph planning can realize this scope."""

    return (
        continuous_bond_realizability_error(
            snapshot,
            atom_input,
            component_library=component_library,
            context_radius_angstrom=context_radius_angstrom,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            retained_non_polymer_chemistry_resolution_by_residue_id=(
                retained_non_polymer_chemistry_resolution_by_residue_id
            ),
        )
        is None
    )
