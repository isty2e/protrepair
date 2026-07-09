"""Primitive observation rules over canonical structure residues."""

from collections.abc import Iterable, Mapping

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.stereochemistry import (
    detect_residue_stereochemistry_violations,
)
from protrepair.state.hydrogen_expectation import (
    derive_structure_hydrogen_expectation_model,
)
from protrepair.state.structure_axes import (
    BackboneHeavyAtomCompletenessState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId

REQUIRED_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})

__all__ = [
    "REQUIRED_BACKBONE_ATOM_NAMES",
    "backbone_heavy_atom_completeness_state",
    "component_support_state",
    "hydrogen_applicability_state",
    "hydrogen_coverage_state",
    "projection_chain_count",
    "sidechain_heavy_atom_completeness_state",
    "stereochemistry_state",
]


def projection_chain_count(
    residues: tuple[ResidueSite, ...],
    *,
    ligands: tuple[ResidueSite, ...],
) -> int:
    """Return the number of chains represented in one projection."""

    return len({residue.residue_id.chain_id for residue in (*residues, *ligands)})


def component_support_state(
    residues: Iterable[ResidueSite],
    *,
    component_library: ComponentLibrary,
) -> ComponentSupportState:
    """Return support coverage over one residue collection."""

    for residue in residues:
        if component_library.get(residue.component_id) is None:
            return ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT

    return ComponentSupportState.ALL_SUPPORTED


def backbone_heavy_atom_completeness_state(
    residues: Iterable[ResidueSite],
    *,
    component_library: ComponentLibrary,
) -> BackboneHeavyAtomCompletenessState:
    """Return backbone heavy-atom completeness over one residue collection."""

    for residue in residues:
        template = component_library.get(residue.component_id)
        if template is None:
            continue

        present_heavy_atom_names = {
            atom_site.name
            for atom_site in residue.atom_sites
            if not atom_site.is_hydrogen()
        }
        missing_backbone_atom_names = tuple(
            atom_name
            for atom_name in REQUIRED_BACKBONE_ATOM_NAMES
            if atom_name in template.expected_heavy_atom_names()
            and atom_name not in present_heavy_atom_names
        )
        if missing_backbone_atom_names:
            return BackboneHeavyAtomCompletenessState.INCOMPLETE

    return BackboneHeavyAtomCompletenessState.COMPLETE


def sidechain_heavy_atom_completeness_state(
    residues: Iterable[ResidueSite],
    *,
    component_library: ComponentLibrary,
) -> SidechainHeavyAtomCompletenessState:
    """Return side-chain heavy-atom completeness over one residue collection."""

    for residue in residues:
        template = component_library.get(residue.component_id)
        if template is None:
            continue

        present_heavy_atom_names = {
            atom_site.name
            for atom_site in residue.atom_sites
            if not atom_site.is_hydrogen()
        }
        missing_sidechain_heavy_atom_names = tuple(
            atom_name
            for atom_name in template.expected_heavy_atom_names()
            if atom_name not in present_heavy_atom_names
            and atom_name not in REQUIRED_BACKBONE_ATOM_NAMES
        )
        if missing_sidechain_heavy_atom_names:
            return SidechainHeavyAtomCompletenessState.INCOMPLETE

    return SidechainHeavyAtomCompletenessState.COMPLETE


def hydrogen_applicability_state(
    residues: Iterable[ResidueSite],
    *,
    context_structure: ProteinStructure,
    component_library: ComponentLibrary,
    expected_hydrogen_atom_names_by_residue: Mapping[ResidueId, tuple[str, ...]]
    | None = None,
) -> HydrogenApplicabilityState:
    """Return whether hydrogen coverage applies to one residue collection."""

    active_expected_hydrogen_atom_names_by_residue = (
        derive_structure_hydrogen_expectation_model(
            context_structure,
            component_library=component_library,
        ).expected_hydrogen_atom_names_by_residue
        if expected_hydrogen_atom_names_by_residue is None
        else expected_hydrogen_atom_names_by_residue
    )
    if any(
        active_expected_hydrogen_atom_names_by_residue.get(residue.residue_id)
        for residue in residues
    ):
        return HydrogenApplicabilityState.APPLICABLE

    return HydrogenApplicabilityState.NOT_APPLICABLE


def hydrogen_coverage_state(
    context_structure: ProteinStructure,
    *,
    residues: Iterable[ResidueSite],
    component_library: ComponentLibrary,
    expected_hydrogen_atom_names_by_residue: Mapping[ResidueId, tuple[str, ...]]
    | None = None,
) -> HydrogenCoverageState:
    """Return hydrogen coverage over applicable residues in one collection."""

    active_expected_hydrogen_atom_names_by_residue = (
        derive_structure_hydrogen_expectation_model(
            context_structure,
            component_library=component_library,
        ).expected_hydrogen_atom_names_by_residue
        if expected_hydrogen_atom_names_by_residue is None
        else expected_hydrogen_atom_names_by_residue
    )
    hydrogen_capable_residue_count = 0
    fully_hydrogenated_residue_count = 0
    partially_hydrogenated_residue_count = 0
    for residue in residues:
        expected_hydrogen_atom_names = (
            active_expected_hydrogen_atom_names_by_residue.get(residue.residue_id)
        )
        if expected_hydrogen_atom_names is None:
            continue

        hydrogen_capable_residue_count += 1
        present_hydrogen_atom_names = {
            atom_site.name
            for atom_site in residue.atom_sites
            if atom_site.is_hydrogen()
        }
        if not present_hydrogen_atom_names:
            continue

        missing_hydrogen_atom_names = tuple(
            atom_name
            for atom_name in expected_hydrogen_atom_names
            if atom_name not in present_hydrogen_atom_names
        )
        if not missing_hydrogen_atom_names:
            fully_hydrogenated_residue_count += 1
            continue

        partially_hydrogenated_residue_count += 1

    if hydrogen_capable_residue_count == 0:
        return HydrogenCoverageState.COMPLETE

    if (
        fully_hydrogenated_residue_count == 0
        and partially_hydrogenated_residue_count == 0
    ):
        return HydrogenCoverageState.NONE

    if (
        fully_hydrogenated_residue_count == hydrogen_capable_residue_count
        and partially_hydrogenated_residue_count == 0
    ):
        return HydrogenCoverageState.COMPLETE

    return HydrogenCoverageState.PARTIAL


def stereochemistry_state(
    structure: ProteinStructure,
    *,
    residues: Iterable[ResidueSite],
    component_library: ComponentLibrary,
) -> StereochemistryState:
    """Return side-chain stereochemistry state over one residue collection."""

    has_supported_tetrahedral_center = False
    for residue in residues:
        template = component_library.get(residue.component_id)
        if template is None or not template.has_tetrahedral_stereochemistry():
            continue

        has_supported_tetrahedral_center = True
        residue_geometry = structure.residue_geometry(
            structure.constitution.residue_index(residue.residue_id)
        )
        if detect_residue_stereochemistry_violations(
            residue,
            residue_geometry=residue_geometry,
            template=template,
        ):
            return StereochemistryState.VIOLATED

    if has_supported_tetrahedral_center:
        return StereochemistryState.CONSISTENT

    return StereochemistryState.NOT_APPLICABLE
