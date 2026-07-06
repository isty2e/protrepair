"""Local bond-planning support resolution for continuous execution regions."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
    evidence_by_residue_id,
)
from protrepair.chemistry.single_atom_inorganic import is_single_atom_inorganic_residue
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    RetainedNonPolymerChemistryResolution,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot


class LocalBondPlanningSupportMode(Enum):
    """Continuous local bond-planning support mode for one included residue."""

    TEMPLATE = "template"
    TEMPLATE_LESS_PASSIVE_CONTEXT = "template_less_passive_context"
    NONBONDING_PASSIVE_CONTEXT = "nonbonding_passive_context"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class LocalBondPlanningSupportResolution:
    """Resolved local bond-planning support for one included residue."""

    mode: LocalBondPlanningSupportMode
    fallback_bond_definitions: tuple[BondDefinition, ...] = ()
    blocker_message: str = ""

    def __post_init__(self) -> None:
        if (
            self.mode
            is not LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
            and self.fallback_bond_definitions
        ):
            raise ValueError(
                "fallback bond definitions require template-less passive-context "
                "support"
            )
        if (
            self.mode is LocalBondPlanningSupportMode.UNSUPPORTED
            and not self.blocker_message
        ):
            return

        if (
            self.mode is not LocalBondPlanningSupportMode.UNSUPPORTED
            and self.blocker_message
        ):
            raise ValueError("only unsupported local bond planning may carry blockers")

    def supports_local_bond_planning(self) -> bool:
        """Return whether this residue admits continuous local bond planning."""

        return self.mode is not LocalBondPlanningSupportMode.UNSUPPORTED


def resolve_local_bond_planning_support(
    snapshot: ProteinStructureSnapshot,
    residue_index: ResidueIndex,
    residue_site: ResidueSite,
    *,
    movable_atom_indices: tuple[AtomIndex, ...],
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
) -> LocalBondPlanningSupportResolution:
    """Resolve local bond-planning support for one included residue."""

    if component_library.get(residue_site.component_id) is not None:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.TEMPLATE
        )

    constitution = snapshot.structure.constitution
    movable_atom_index_set = set(movable_atom_indices)
    residue_has_movable_atom = any(
        constitution.atom_index_in_residue(residue_index, atom_site.name)
        in movable_atom_index_set
        for atom_site in residue_site.atom_sites
    )
    if is_single_atom_inorganic_residue(residue_site):
        if residue_has_movable_atom:
            return LocalBondPlanningSupportResolution(
                mode=LocalBondPlanningSupportMode.UNSUPPORTED,
                blocker_message=(
                    "continuous relaxation treats single-atom metal/ion residues "
                    "as nonbonding fixed context, not movable chemistry: "
                    f"{residue_site.component_id} "
                    f"({residue_site.residue_id.display_token()})"
                ),
            )
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
        )

    if residue_has_movable_atom:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.UNSUPPORTED
        )

    heavy_atom_sites = tuple(
        atom_site for atom_site in residue_site.atom_sites if atom_site.element != "H"
    )
    if len(heavy_atom_sites) <= 1:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
        )

    if not residue_site.is_hetero:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.UNSUPPORTED
        )

    explicit_evidence = evidence_by_residue_id(
        retained_non_polymer_chemistry_evidence
    ).get(residue_site.residue_id)
    if explicit_evidence is not None:
        chemistry_resolution = _retained_non_polymer_chemistry_resolution(
            snapshot.structure,
            residue_site,
            component_library=component_library,
            evidence=explicit_evidence,
            resolution_by_residue_id=(
                retained_non_polymer_chemistry_resolution_by_residue_id
            ),
        )
        passive_bond_definitions = _passive_context_heavy_bond_definitions(
            residue_site,
            chemistry_resolution.heavy_bond_definitions,
        )
        if (
            chemistry_resolution.source
            is not RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
            or passive_bond_definitions is None
        ):
            return LocalBondPlanningSupportResolution(
                mode=LocalBondPlanningSupportMode.UNSUPPORTED,
                blocker_message=_retained_non_polymer_support_blocker_message(
                    residue_site,
                    reason=chemistry_resolution.failure_reason,
                    prefix=(
                        "continuous relaxation could not derive passive retained "
                        "non-polymer bond support from explicit chemistry evidence"
                    ),
                ),
            )
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT,
            fallback_bond_definitions=passive_bond_definitions,
        )

    if not allow_retained_non_polymer_rdkit_fallback:
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.UNSUPPORTED,
            blocker_message=(
                "continuous relaxation requires retained non-polymer passive "
                "context bond support, but RDKit fallback is disabled: "
                f"{residue_site.component_id} "
                f"({residue_site.residue_id.display_token()})"
            ),
        )

    chemistry_resolution = _retained_non_polymer_chemistry_resolution(
        snapshot.structure,
        residue_site,
        component_library=component_library,
        evidence=None,
        resolution_by_residue_id=(
            retained_non_polymer_chemistry_resolution_by_residue_id
        ),
    )
    passive_bond_definitions = _passive_context_heavy_bond_definitions(
        residue_site,
        chemistry_resolution.heavy_bond_definitions,
    )
    if (
        chemistry_resolution.source
        is not RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
        or passive_bond_definitions is None
    ):
        return LocalBondPlanningSupportResolution(
            mode=LocalBondPlanningSupportMode.UNSUPPORTED,
            blocker_message=_retained_non_polymer_support_blocker_message(
                residue_site,
                reason=chemistry_resolution.failure_reason,
                prefix=(
                    "continuous relaxation could not derive RDKit fallback "
                    "passive retained non-polymer bond support"
                ),
            ),
        )

    return LocalBondPlanningSupportResolution(
        mode=LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT,
        fallback_bond_definitions=passive_bond_definitions,
    )


def _retained_non_polymer_chemistry_resolution(
    structure: ProteinStructure,
    residue_site: ResidueSite,
    *,
    component_library: ComponentLibrary,
    evidence: RetainedNonPolymerChemistryEvidence | None,
    resolution_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryResolution,
    ]
    | None,
) -> RetainedNonPolymerChemistryResolution:
    """Return a precomputed retained chemistry resolution when available."""

    if resolution_by_residue_id is not None:
        resolution = resolution_by_residue_id.get(residue_site.residue_id)
        if resolution is not None:
            return resolution

    return resolve_retained_non_polymer_chemistry(
        structure,
        residue_site,
        component_library=component_library,
        evidence=evidence,
    )


def _retained_non_polymer_support_blocker_message(
    residue_site: ResidueSite,
    *,
    reason: str,
    prefix: str,
) -> str:
    """Return a stable local-bond support blocker with optional failure detail."""

    base_message = (
        f"{prefix}: {residue_site.component_id} "
        f"({residue_site.residue_id.display_token()})"
    )
    if not reason:
        return base_message

    return f"{base_message}: {reason}"


def _passive_context_heavy_bond_definitions(
    residue_site: ResidueSite,
    bond_definitions: tuple[BondDefinition, ...],
) -> tuple[BondDefinition, ...] | None:
    """Return connected heavy-bond support for passive template-less context."""

    heavy_atom_names = frozenset(
        atom_site.name
        for atom_site in residue_site.atom_sites
        if atom_site.element != "H"
    )
    heavy_bond_definitions = tuple(
        bond_definition
        for bond_definition in bond_definitions
        if bond_definition.atom_name_1 in heavy_atom_names
        and bond_definition.atom_name_2 in heavy_atom_names
    )
    if not heavy_bond_definitions:
        return None

    neighbors_by_atom_name = {atom_name: set[str]() for atom_name in heavy_atom_names}
    for bond_definition in heavy_bond_definitions:
        neighbors_by_atom_name[bond_definition.atom_name_1].add(
            bond_definition.atom_name_2
        )
        neighbors_by_atom_name[bond_definition.atom_name_2].add(
            bond_definition.atom_name_1
        )

    seed_atom_name = next(iter(heavy_atom_names))
    visited_atom_names: set[str] = set()
    pending_atom_names = [seed_atom_name]
    while pending_atom_names:
        atom_name = pending_atom_names.pop()
        if atom_name in visited_atom_names:
            continue

        visited_atom_names.add(atom_name)
        pending_atom_names.extend(neighbors_by_atom_name[atom_name])

    if visited_atom_names != heavy_atom_names:
        return None

    return heavy_bond_definitions
