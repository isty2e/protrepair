"""Direct completion-transformer planning over requested workflow state."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.scope import ResidueSetScope
from protrepair.state import (
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.workflow.actions.heavy_completion import (
    HeavyAtomCompletionTransformer,
)
from protrepair.workflow.actions.hydrogen_completion import (
    HydrogenCompletionTransformer,
)
from protrepair.workflow.actions.retained_non_polymer_hydrogen_completion import (
    RetainedNonPolymerHydrogenCompletionTransformer,
)
from protrepair.workflow.contracts.request import RequestedGoalSet


@dataclass(frozen=True, slots=True)
class CompletionTransformerPlanningOutcome:
    """Completion-family candidate assessment over requested goals."""

    transformers: tuple[
        HeavyAtomCompletionTransformer
        | HydrogenCompletionTransformer
        | RetainedNonPolymerHydrogenCompletionTransformer,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transformers", tuple(self.transformers))


@dataclass(frozen=True, slots=True)
class _ResidueCompletionPlanningFacts:
    """Residue-local completion truth reused across one planning pass."""

    residue_id: ResidueId
    component_supported: bool
    requires_backbone_completion: bool
    requires_sidechain_completion: bool
    needs_hydrogenation: bool


def plan_completion_transformers(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    coverage_facts: StructureCoverageFacts | None = None,
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None,
) -> CompletionTransformerPlanningOutcome:
    """Return hydrogen-completion transformer candidates and blocker facts."""

    return plan_hydrogen_completion_transformers(
        structure,
        requested_goals=requested_goals,
        component_library=component_library,
        coverage_facts=coverage_facts,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )


def plan_atom_completion_transformers(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    required_residue_ids: tuple[ResidueId, ...] = (),
    coverage_facts: StructureCoverageFacts | None = None,
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None,
) -> CompletionTransformerPlanningOutcome:
    """Return atom-completion transformer candidates and blocker facts."""

    requests_backbone_heavy_atom_completion = (
        requested_goals.requests_whole_structure_backbone_heavy_atom_completion()
    )
    requests_sidechain_heavy_atom_completion = (
        requested_goals.requests_whole_structure_sidechain_heavy_atom_completion()
    )
    requests_heavy_atom_completion = (
        requested_goals.requests_whole_structure_heavy_atom_completion()
    )
    requests_hydrogen_population = (
        requested_goals.requests_whole_structure_hydrogen_population()
    )
    if (
        not requests_heavy_atom_completion
        and not requests_hydrogen_population
        and not required_residue_ids
    ):
        return CompletionTransformerPlanningOutcome()

    active_coverage_facts = coverage_facts
    active_chemistry_readiness_facts = chemistry_readiness_facts
    if active_coverage_facts is None or active_chemistry_readiness_facts is None:
        (
            active_coverage_facts,
            active_chemistry_readiness_facts,
        ) = derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=component_library,
        )

    residue_completion_facts = _residue_completion_planning_facts(
        coverage_facts=active_coverage_facts,
        chemistry_readiness_facts=active_chemistry_readiness_facts,
    )
    blocked_residue_id_set = set(
        _blocked_atom_completion_residue_ids(
            residue_completion_facts,
            requests_heavy_atom_completion=requests_heavy_atom_completion,
            requests_hydrogen_population=requests_hydrogen_population,
            required_residue_id_set=set(required_residue_ids),
        )
    )
    heavy_target_residue_ids = _heavy_completion_target_residue_ids(
        residue_completion_facts,
        blocked_residue_id_set=blocked_residue_id_set,
        requests_backbone_heavy_atom_completion=(
            requests_backbone_heavy_atom_completion
        ),
        requests_sidechain_heavy_atom_completion=(
            requests_sidechain_heavy_atom_completion
        ),
        requests_hydrogen_population=requests_hydrogen_population,
        required_residue_id_set=set(required_residue_ids),
    )
    return CompletionTransformerPlanningOutcome(
        transformers=tuple(
            transformer
            for transformer in (
                _heavy_completion_transformer(heavy_target_residue_ids),
            )
            if transformer is not None
        ),
    )


def plan_hydrogen_completion_transformers(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    required_residue_ids: tuple[ResidueId, ...] = (),
    coverage_facts: StructureCoverageFacts | None = None,
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None,
) -> CompletionTransformerPlanningOutcome:
    """Return hydrogen-completion transformer candidates and blocker facts."""

    if (
        not requested_goals.requests_whole_structure_hydrogen_population()
        and not required_residue_ids
    ):
        return CompletionTransformerPlanningOutcome()

    active_coverage_facts = coverage_facts
    active_chemistry_readiness_facts = chemistry_readiness_facts
    if active_coverage_facts is None or active_chemistry_readiness_facts is None:
        (
            active_coverage_facts,
            active_chemistry_readiness_facts,
        ) = derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=component_library,
        )

    residue_completion_facts = _residue_completion_planning_facts(
        coverage_facts=active_coverage_facts,
        chemistry_readiness_facts=active_chemistry_readiness_facts,
    )
    blocked_residue_id_set = set(
        _blocked_hydrogen_completion_residue_ids(
            residue_completion_facts,
            required_residue_id_set=set(required_residue_ids),
        )
    )
    hydrogen_target_residue_ids = _hydrogen_completion_target_residue_ids(
        residue_completion_facts,
        blocked_residue_id_set=blocked_residue_id_set,
        requests_hydrogen_population=(
            requested_goals.requests_whole_structure_hydrogen_population()
        ),
        required_residue_id_set=set(required_residue_ids),
    )
    return CompletionTransformerPlanningOutcome(
        transformers=tuple(
            transformer
            for transformer in (
                _hydrogen_completion_transformer(hydrogen_target_residue_ids),
            )
            if transformer is not None
        ),
    )


def plan_retained_non_polymer_hydrogen_completion_transformers(
    structure: ProteinStructure,
    *,
    requested_goals: RequestedGoalSet,
    component_library: ComponentLibrary,
    chemistry_readiness_facts: StructureChemistryReadinessFacts | None = None,
) -> CompletionTransformerPlanningOutcome:
    """Return retained non-polymer hydrogen completion candidates."""

    if not requested_goals.requests_whole_structure_hydrogen_population():
        return CompletionTransformerPlanningOutcome()

    active_chemistry_readiness_facts = chemistry_readiness_facts
    if active_chemistry_readiness_facts is None:
        _, active_chemistry_readiness_facts = (
            derive_structure_coverage_and_chemistry_readiness_facts(
                structure,
                component_library=component_library,
            )
        )

    retained_non_polymer_target_residue_ids = tuple(
        retained_fact.residue_id
        for retained_fact in (
            active_chemistry_readiness_facts.retained_non_polymer_facts
        )
        if retained_fact.requires_hydrogen_completion()
    )
    return CompletionTransformerPlanningOutcome(
        transformers=tuple(
            transformer
            for transformer in (
                _retained_non_polymer_hydrogen_completion_transformer(
                    retained_non_polymer_target_residue_ids
                ),
            )
            if transformer is not None
        ),
    )

def _residue_completion_planning_facts(
    *,
    coverage_facts: StructureCoverageFacts,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
) -> tuple[_ResidueCompletionPlanningFacts, ...]:
    """Return one reusable residue-local completion fact bundle per residue."""

    chemistry_readiness_by_residue_id = {
        residue_fact.residue_id: residue_fact
        for residue_fact in chemistry_readiness_facts.residue_facts
    }
    return tuple(
        _ResidueCompletionPlanningFacts(
            residue_id=coverage_residue_fact.residue_id,
            component_supported=chemistry_readiness_by_residue_id[
                coverage_residue_fact.residue_id
            ].component_support_state.is_fully_supported(),
            requires_backbone_completion=(
                coverage_residue_fact.requires_backbone_completion()
            ),
            requires_sidechain_completion=(
                coverage_residue_fact.requires_sidechain_completion()
            ),
            needs_hydrogenation=chemistry_readiness_by_residue_id[
                coverage_residue_fact.residue_id
            ].needs_hydrogenation(),
        )
        for coverage_residue_fact in coverage_facts.residue_facts
    )


def _blocked_atom_completion_residue_ids(
    residue_completion_facts: tuple[_ResidueCompletionPlanningFacts, ...],
    *,
    requests_heavy_atom_completion: bool,
    requests_hydrogen_population: bool,
    required_residue_id_set: set[ResidueId],
) -> tuple[ResidueId, ...]:
    """Return residues that block completion because support is absent."""

    if (
        not requests_heavy_atom_completion
        and not requests_hydrogen_population
        and not required_residue_id_set
    ):
        return ()

    return tuple(
        residue_facts.residue_id
        for residue_facts in residue_completion_facts
        if (
            not residue_facts.component_supported
            and (
                (
                    requests_heavy_atom_completion
                    and (
                        residue_facts.requires_backbone_completion
                        or residue_facts.requires_sidechain_completion
                    )
                )
                or (
                    requests_hydrogen_population
                    and (
                        residue_facts.requires_backbone_completion
                        or residue_facts.requires_sidechain_completion
                    )
                )
                or (
                    residue_facts.residue_id in required_residue_id_set
                    and (
                        residue_facts.requires_backbone_completion
                        or residue_facts.requires_sidechain_completion
                    )
                )
            )
        )
    )


def _blocked_hydrogen_completion_residue_ids(
    residue_completion_facts: tuple[_ResidueCompletionPlanningFacts, ...],
    *,
    required_residue_id_set: set[ResidueId],
) -> tuple[ResidueId, ...]:
    """Return residues that block hydrogen completion because support is absent."""

    return tuple(
        residue_facts.residue_id
        for residue_facts in residue_completion_facts
        if (
            not residue_facts.component_supported
            and (
                residue_facts.needs_hydrogenation
                or residue_facts.residue_id in required_residue_id_set
            )
        )
    )


def _heavy_completion_target_residue_ids(
    residue_completion_facts: tuple[_ResidueCompletionPlanningFacts, ...],
    *,
    blocked_residue_id_set: set[ResidueId],
    requests_backbone_heavy_atom_completion: bool,
    requests_sidechain_heavy_atom_completion: bool,
    requests_hydrogen_population: bool,
    required_residue_id_set: set[ResidueId],
) -> tuple[ResidueId, ...]:
    """Return residues that require heavy completion under the active goals."""

    target_residue_ids: list[ResidueId] = []
    for residue_facts in residue_completion_facts:
        if residue_facts.residue_id in blocked_residue_id_set:
            continue
        if (
            not residue_facts.requires_backbone_completion
            and not residue_facts.requires_sidechain_completion
        ):
            continue
        if (
            requests_backbone_heavy_atom_completion
            and residue_facts.requires_backbone_completion
        ):
            target_residue_ids.append(residue_facts.residue_id)
            continue
        if (
            requests_sidechain_heavy_atom_completion
            and residue_facts.requires_sidechain_completion
        ):
            target_residue_ids.append(residue_facts.residue_id)
            continue
        if requests_hydrogen_population and residue_facts.needs_hydrogenation:
            target_residue_ids.append(residue_facts.residue_id)
            continue
        if residue_facts.residue_id in required_residue_id_set:
            target_residue_ids.append(residue_facts.residue_id)

    return tuple(target_residue_ids)


def _hydrogen_completion_target_residue_ids(
    residue_completion_facts: tuple[_ResidueCompletionPlanningFacts, ...],
    *,
    blocked_residue_id_set: set[ResidueId],
    requests_hydrogen_population: bool,
    required_residue_id_set: set[ResidueId],
) -> tuple[ResidueId, ...]:
    """Return residues that require hydrogen completion under the active goals."""

    if not requests_hydrogen_population and not required_residue_id_set:
        return ()

    return tuple(
        residue_facts.residue_id
        for residue_facts in residue_completion_facts
        if (
            residue_facts.residue_id not in blocked_residue_id_set
            and residue_facts.needs_hydrogenation
            and (
                requests_hydrogen_population
                or residue_facts.residue_id in required_residue_id_set
            )
        )
    )


def _heavy_completion_transformer(
    target_residue_ids: tuple[ResidueId, ...],
) -> HeavyAtomCompletionTransformer | None:
    """Return one heavy-completion transformer when any residues need it."""

    if not target_residue_ids:
        return None

    return HeavyAtomCompletionTransformer.from_completion_scope(
        ResidueSetScope(residue_ids=target_residue_ids)
    )


def _hydrogen_completion_transformer(
    target_residue_ids: tuple[ResidueId, ...],
) -> HydrogenCompletionTransformer | None:
    """Return one hydrogen-completion transformer when any residues need it."""

    if not target_residue_ids:
        return None

    return HydrogenCompletionTransformer.from_completion_scope(
        ResidueSetScope(residue_ids=target_residue_ids)
    )


def _retained_non_polymer_hydrogen_completion_transformer(
    target_residue_ids: tuple[ResidueId, ...],
) -> RetainedNonPolymerHydrogenCompletionTransformer | None:
    """Return one retained non-polymer hydrogen transformer when any targets need it."""

    if not target_residue_ids:
        return None

    return RetainedNonPolymerHydrogenCompletionTransformer.from_completion_scope(
        ResidueSetScope(residue_ids=target_residue_ids)
    )
