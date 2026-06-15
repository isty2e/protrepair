"""Canonical legality and termination decisions for local transformation planning."""

from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.scope import (
    Scope,
)
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import StructureProjectionStateFacts
from protrepair.state.domain import AtomScopeStateFacts, SelectedAtomScopeFacts
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.readiness import (
    atom_scope_facts_supports_continuous_relaxation,
    continuous_bond_realizability_facts_support_continuous_relaxation,
    derive_atom_scope_continuous_relaxation_facts,
)
from protrepair.workflow.planning.transformation.runtime import (
    LocalTransformationFamily,
    TransformationFamilyAttemptSignature,
    TransformationPlanningMemory,
    TransformationPlanningSignature,
    TransformationTerminationReason,
)


class LocalTransformationStratum(str, Enum):
    """Closed legality strata for selected-region local transformer families."""

    PREPARATION = "preparation"
    CANDIDATE_CONSTRUCTION = "candidate_construction"
    RELAXATION = "relaxation"


class LocalPreparationReason(str, Enum):
    """Closed reasons for why one preparation-stage family is legal."""

    TOPOLOGY_PRECONDITION = "topology_precondition"
    CHEMISTRY_PREPARATION = "chemistry_preparation"
    LOCAL_GEOMETRY = "local_geometry"


@dataclass(frozen=True, slots=True)
class LegalTransformationFamily:
    """One legal transformer family under one canonical planning signature."""

    stratum: LocalTransformationStratum
    family: LocalTransformationFamily
    signature: TransformationFamilyAttemptSignature
    preparation_reason: LocalPreparationReason | None = None

    def __post_init__(self) -> None:
        if (
            self.stratum is LocalTransformationStratum.PREPARATION
            and self.preparation_reason is None
        ):
            raise ValueError(
                "preparation-stage legal families require one preparation reason"
            )

        if (
            self.stratum is not LocalTransformationStratum.PREPARATION
            and self.preparation_reason is not None
        ):
            raise ValueError(
                "only preparation-stage legal families may carry a "
                "preparation reason"
            )


@dataclass(frozen=True, slots=True)
class LegalTransformationFamilySet:
    """Canonical legal-family set under one canonical planner signature."""

    planning_signature: TransformationPlanningSignature
    families: tuple[LegalTransformationFamily, ...] = ()
    history_blocked_family_signatures: tuple[
        TransformationFamilyAttemptSignature,
        ...,
    ] = ()

    @classmethod
    def from_applicable_families(
        cls,
        *,
        planning_signature: TransformationPlanningSignature,
        planning_memory: TransformationPlanningMemory,
        applicable_families: tuple[
            tuple[
                LocalTransformationStratum,
                LocalTransformationFamily,
                LocalPreparationReason | None,
            ],
            ...,
        ],
    ) -> "LegalTransformationFamilySet":
        """Build one legal set from applicable families under memory guards."""

        families: list[LegalTransformationFamily] = []
        history_blocked_family_signatures: list[
            TransformationFamilyAttemptSignature
        ] = []
        for stratum, family, preparation_reason in applicable_families:
            family_signature = TransformationFamilyAttemptSignature(
                family=family,
                planning_signature=planning_signature,
            )
            if planning_memory.history.has_attempted_family(family_signature):
                history_blocked_family_signatures.append(family_signature)
                continue

            families.append(
                LegalTransformationFamily(
                    stratum=stratum,
                    family=family,
                    signature=family_signature,
                    preparation_reason=preparation_reason,
                )
            )

        return cls(
            planning_signature=planning_signature,
            families=tuple(families),
            history_blocked_family_signatures=tuple(history_blocked_family_signatures),
        )

    def families_in_stratum(
        self,
        stratum: LocalTransformationStratum,
    ) -> tuple[LegalTransformationFamily, ...]:
        """Return the legal families that belong to one legality stratum."""

        return tuple(family for family in self.families if family.stratum is stratum)

    def contains_stratum(
        self,
        stratum: LocalTransformationStratum,
    ) -> bool:
        """Return whether one legality stratum currently contains any action."""

        return bool(self.families_in_stratum(stratum))

    def is_empty(self) -> bool:
        """Return whether no transformation family is currently legal."""

        return not self.families

    def is_exhausted_by_run_history(self) -> bool:
        """Return whether run memory blocked every otherwise-applicable action."""

        return not self.families and bool(self.history_blocked_family_signatures)

    def contains_family(
        self,
        family: LocalTransformationFamily,
    ) -> bool:
        """Return whether one transformation family is currently legal."""

        return any(legal_family.family is family for legal_family in self.families)

    def signature_for_family(
        self,
        family: LocalTransformationFamily,
    ) -> TransformationFamilyAttemptSignature:
        """Return the canonical signature for one legal transformation family."""

        for legal_family in self.families:
            if legal_family.family is family:
                return legal_family.signature

        raise KeyError(f"{family.value} is not present in the current legal set")

    def family_record_for(
        self,
        family: LocalTransformationFamily,
    ) -> LegalTransformationFamily:
        """Return one legal-family record for one transformer family."""

        for legal_family in self.families:
            if legal_family.family is family:
                return legal_family

        raise KeyError(f"{family.value} is not present in the current legal set")


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    """Canonical termination decision over one canonical planning state."""

    reason: TransformationTerminationReason | None = None

    def is_terminal(self) -> bool:
        """Return whether the planner should terminate immediately."""

        return self.reason is not None


def selected_region_legal_transformations(
    *,
    structure_facts: StructureProjectionStateFacts,
    selected_scope: Scope,
    atom_scope_facts: AtomScopeStateFacts,
    planning_memory: TransformationPlanningMemory,
    discrete_preparation_applicable: bool,
    discrete_seeding_applicable: bool,
) -> LegalTransformationFamilySet:
    """Return the legal transformation families for one selected-region state."""

    planning_signature = TransformationPlanningSignature.from_state_facts(
        structure_facts=structure_facts,
        selected_scope=selected_scope,
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )
    applicable_families: list[
        tuple[
            LocalTransformationStratum,
            LocalTransformationFamily,
            LocalPreparationReason | None,
        ]
    ] = []
    for family, applicable in (
        (
            LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION,
            discrete_preparation_applicable,
        ),
        (
            LocalTransformationFamily.BRANCHED_SIDECHAIN_SEED,
            discrete_seeding_applicable,
        ),
    ):
        if not applicable:
            continue

        if family is LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION:
            applicable_families.append(
                (
                    LocalTransformationStratum.PREPARATION,
                    family,
                    _discrete_preparation_reason(
                        atom_scope_facts.selected_scope_facts
                    ),
                )
            )
            continue

        applicable_families.append(
            (LocalTransformationStratum.CANDIDATE_CONSTRUCTION, family, None)
        )

    if atom_scope_facts_supports_continuous_relaxation(atom_scope_facts):
        applicable_families.append(
            (
                LocalTransformationStratum.RELAXATION,
                LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION,
                None,
            )
        )

    return LegalTransformationFamilySet.from_applicable_families(
        planning_signature=planning_signature,
        planning_memory=planning_memory,
        applicable_families=tuple(applicable_families),
    )


def _discrete_preparation_reason(
    selected_scope_facts: SelectedAtomScopeFacts,
) -> LocalPreparationReason:
    """Return why discrete state repair occupies the preparation stage."""

    if not continuous_bond_realizability_facts_support_continuous_relaxation(
        selected_scope_facts.continuous_bond_realizability_facts
    ):
        return LocalPreparationReason.TOPOLOGY_PRECONDITION

    if _requires_chemistry_preparation(selected_scope_facts):
        return LocalPreparationReason.CHEMISTRY_PREPARATION

    return LocalPreparationReason.LOCAL_GEOMETRY


def _requires_chemistry_preparation(
    selected_scope_facts: SelectedAtomScopeFacts,
) -> bool:
    """Return whether local chemistry detail still needs non-topology repair."""

    return (
        selected_scope_facts.hydrogen_attachment_resolution_facts.any_coordinate_inferred()
    )


def selected_region_termination_decision(
    *,
    planning_memory: TransformationPlanningMemory,
    legal_transformations: LegalTransformationFamilySet,
) -> TerminationDecision:
    """Return the current termination decision for one selected-region run."""

    if planning_memory.termination.is_terminal():
        return TerminationDecision(planning_memory.termination.reason)

    if planning_memory.budget.is_exhausted():
        return TerminationDecision(TransformationTerminationReason.STEP_LIMIT_REACHED)

    if legal_transformations.is_empty():
        if (
            planning_memory.history.has_observed_signature(
                legal_transformations.planning_signature
            )
            and legal_transformations.is_exhausted_by_run_history()
        ):
            return TerminationDecision(TransformationTerminationReason.CYCLE_DETECTED)

        return TerminationDecision(
            TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
        )

    return TerminationDecision()


def selected_region_state(
    snapshot: ProteinStructureSnapshot,
    selected_scope: Scope,
    *,
    component_library: ComponentLibrary,
) -> tuple[StructureProjectionStateFacts, AtomScopeStateFacts]:
    """Return primitive whole-structure and local selected-region state facts."""

    structure_facts = StructureProjectionStateFacts.from_structure(
        snapshot.structure,
        component_library=component_library,
    )
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        selected_scope,
        carrier=snapshot,
    )
    atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=component_library,
    )
    return structure_facts, atom_scope_facts
