"""Soft ranking policy over already-admissible workflow action families."""

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from protrepair.scope import ResidueBoundaryScope
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.action.registry import WorkflowStateActionType
from protrepair.workflow.planning.assessment.deficits import WorkflowDeficitDisposition
from protrepair.workflow.planning.capability import (
    WorkflowActionCapability,
    WorkflowActionDonorRequirement,
    WorkflowActionEffectClass,
    WorkflowActionLocality,
    WorkflowCapabilityDeficitFamily,
)


class WorkflowPolicyFamily(str, Enum):
    """Ranking-template families over admissible workflow actions."""

    COVERAGE_GAP = "coverage_gap"
    ATOM_COVERAGE = "atom_coverage"
    TOPOLOGY_RESOLUTION = "topology_resolution"
    CHEMISTRY_CONTRADICTION = "chemistry_contradiction"
    CHEMISTRY_READINESS = "chemistry_readiness"
    BOUNDARY_GOAL_ONLY = "boundary_goal_only"
    BACKBONE_WINDOW_OPERATOR = "backbone_window_operator"
    INTRINSIC_GEOMETRY = "intrinsic_geometry"
    PARSER_COMPATIBILITY = "parser_compatibility"
    INTERACTION = "interaction"


@dataclass(frozen=True, slots=True)
class _WorkflowCandidateGroup:
    """One proposal-family group plus ranking metadata."""

    action_family: WorkflowStateActionType
    proposals: tuple[WorkflowActionProposal, ...]
    capability: WorkflowActionCapability
    policy_family: WorkflowPolicyFamily
    original_index: int


@dataclass(frozen=True, slots=True)
class WorkflowPlanningPolicy:
    """Soft lexicographic ranking policy over admissible workflow proposals."""

    def rank_candidates(
        self,
        candidates: tuple[WorkflowActionProposal, ...] | list[WorkflowActionProposal],
        *,
        domain: WorkflowActionDomain,
    ) -> tuple[WorkflowActionProposal, ...]:
        """Return admissible proposals ordered by family-specific policy."""

        candidate_groups = self._group_candidates(tuple(candidates), domain=domain)
        ranked_groups = sorted(
            candidate_groups,
            key=lambda group: self._group_sort_key(group, domain=domain),
        )
        return tuple(
            proposal
            for group in ranked_groups
            for proposal in group.proposals
        )

    def _group_candidates(
        self,
        candidates: tuple[WorkflowActionProposal, ...],
        *,
        domain: WorkflowActionDomain,
    ) -> tuple[_WorkflowCandidateGroup, ...]:
        """Return proposal-family groups over candidate workflow proposals."""

        grouped: OrderedDict[
            WorkflowStateActionType, list[WorkflowActionProposal]
        ] = OrderedDict()
        for candidate in candidates:
            grouped.setdefault(candidate.transformer.proposal_family(), []).append(
                candidate
            )

        groups: list[_WorkflowCandidateGroup] = []
        for original_index, (action_family, proposals) in enumerate(grouped.items()):
            capability = proposals[0].capability
            groups.append(
                _WorkflowCandidateGroup(
                    action_family=action_family,
                    proposals=tuple(proposals),
                    capability=capability,
                    policy_family=self._policy_family_for(
                        capability,
                        domain=domain,
                    ),
                    original_index=original_index,
                )
            )

        return tuple(groups)

    def _group_sort_key(
        self,
        group: _WorkflowCandidateGroup,
        *,
        domain: WorkflowActionDomain,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        """Return one explicit lexicographic key for a proposal group."""

        capability = group.capability
        return (
            self._family_priority(group.policy_family, domain=domain),
            self._disposition_priority(group.policy_family, domain=domain),
            self._explicit_request_priority(group),
            self._exact_goal_support_priority(capability, domain=domain),
            self._effect_class_priority(capability),
            self._locality_priority(capability),
            self._donor_priority(capability),
            group.original_index,
        )

    def _policy_family_for(
        self,
        capability: WorkflowActionCapability,
        *,
        domain: WorkflowActionDomain,
    ) -> WorkflowPolicyFamily:
        """Return the ranking-template family for one workflow capability."""

        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.COVERAGE_GAP
        ):
            return WorkflowPolicyFamily.COVERAGE_GAP
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.ATOM_COVERAGE
        ):
            return WorkflowPolicyFamily.ATOM_COVERAGE
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.TOPOLOGY_RESOLUTION
        ):
            return WorkflowPolicyFamily.TOPOLOGY_RESOLUTION
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.CHEMISTRY_CONTRADICTION
        ):
            return WorkflowPolicyFamily.CHEMISTRY_CONTRADICTION
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS
        ):
            return WorkflowPolicyFamily.CHEMISTRY_READINESS
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.INTERACTION
        ) and (
            domain.burden.is_holo_context()
            and domain.burden.has_interaction_burden()
        ):
            return WorkflowPolicyFamily.INTERACTION
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.PARSER_COMPATIBILITY
        ) and domain.burden.has_parser_compatibility_burden():
            return WorkflowPolicyFamily.PARSER_COMPATIBILITY
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY
        ):
            return WorkflowPolicyFamily.INTRINSIC_GEOMETRY
        if not capability.reducible_deficit_families and any(
            issubclass(descriptor.scope_type, ResidueBoundaryScope)
            for descriptor in capability.supported_goals
        ):
            return WorkflowPolicyFamily.BOUNDARY_GOAL_ONLY
        if capability.can_reduce_deficit_family(
            WorkflowCapabilityDeficitFamily.BACKBONE_WINDOW_OPERATOR
        ):
            return WorkflowPolicyFamily.BACKBONE_WINDOW_OPERATOR

        raise NotImplementedError(
            "workflow policy family is not defined for the active capability"
        )

    def _family_priority(
        self,
        policy_family: WorkflowPolicyFamily,
        *,
        domain: WorkflowActionDomain,
    ) -> int:
        """Return the cross-family preference rank for one policy family."""

        family_order = {
            WorkflowPolicyFamily.COVERAGE_GAP: 0,
            WorkflowPolicyFamily.ATOM_COVERAGE: 1,
            WorkflowPolicyFamily.TOPOLOGY_RESOLUTION: 2,
            WorkflowPolicyFamily.CHEMISTRY_CONTRADICTION: 3,
            WorkflowPolicyFamily.CHEMISTRY_READINESS: 4,
            WorkflowPolicyFamily.BOUNDARY_GOAL_ONLY: 5,
            WorkflowPolicyFamily.BACKBONE_WINDOW_OPERATOR: 6,
            WorkflowPolicyFamily.PARSER_COMPATIBILITY: 7,
            WorkflowPolicyFamily.INTRINSIC_GEOMETRY: 8,
            WorkflowPolicyFamily.INTERACTION: 9,
        }
        if domain.burden.is_holo_context() and domain.burden.has_interaction_burden():
            family_order = {
                **family_order,
                WorkflowPolicyFamily.INTERACTION: 6,
                WorkflowPolicyFamily.BACKBONE_WINDOW_OPERATOR: 7,
                WorkflowPolicyFamily.PARSER_COMPATIBILITY: 8,
                WorkflowPolicyFamily.INTRINSIC_GEOMETRY: 9,
            }
        return family_order[policy_family]

    def _disposition_priority(
        self,
        policy_family: WorkflowPolicyFamily,
        *,
        domain: WorkflowActionDomain,
    ) -> int:
        """Return required/optional/blocked pressure rank for one family."""

        disposition = self._family_disposition(policy_family, domain=domain)
        if disposition is WorkflowDeficitDisposition.REQUIRED:
            return 0
        if disposition is WorkflowDeficitDisposition.OPTIONAL:
            return 1
        if disposition is WorkflowDeficitDisposition.BLOCKED:
            return 2
        return 3

    def _family_disposition(
        self,
        policy_family: WorkflowPolicyFamily,
        *,
        domain: WorkflowActionDomain,
    ) -> WorkflowDeficitDisposition | None:
        """Return the strongest active deficit disposition for one family."""

        state_deficit = domain.state_deficit
        if policy_family is WorkflowPolicyFamily.COVERAGE_GAP:
            return _strongest_disposition(
                gap_deficit.disposition
                for gap_deficit in state_deficit.coverage.gap_deficits
            )
        if policy_family is WorkflowPolicyFamily.ATOM_COVERAGE:
            return _strongest_disposition(
                atom_deficit.disposition
                for atom_deficit in state_deficit.coverage.atom_deficits
            )
        if policy_family is WorkflowPolicyFamily.TOPOLOGY_RESOLUTION:
            topology_resolution = state_deficit.topology_resolution
            return (
                topology_resolution.disposition
                if topology_resolution is not None
                else None
            )
        if policy_family is WorkflowPolicyFamily.CHEMISTRY_CONTRADICTION:
            disulfide_hydrogen = state_deficit.disulfide_hydrogen
            return (
                disulfide_hydrogen.disposition
                if disulfide_hydrogen is not None
                else None
            )
        if policy_family is WorkflowPolicyFamily.CHEMISTRY_READINESS:
            chemistry_deficit = state_deficit.chemistry_readiness
            return (
                chemistry_deficit.disposition
                if chemistry_deficit.has_burden()
                else None
            )
        if policy_family is WorkflowPolicyFamily.BOUNDARY_GOAL_ONLY:
            return (
                WorkflowDeficitDisposition.REQUIRED
                if any(
                    isinstance(goal.scope, ResidueBoundaryScope)
                    for goal in domain.requested_goals
                )
                else None
            )
        if policy_family is WorkflowPolicyFamily.BACKBONE_WINDOW_OPERATOR:
            return _strongest_disposition(
                operator_deficit.disposition
                for operator_deficit in state_deficit.backbone_window_operator
            )
        if policy_family is WorkflowPolicyFamily.INTRINSIC_GEOMETRY:
            intrinsic_geometry = state_deficit.intrinsic_geometry
            return (
                intrinsic_geometry.disposition
                if intrinsic_geometry is not None
                else None
            )
        if policy_family is WorkflowPolicyFamily.PARSER_COMPATIBILITY:
            parser_compatibility = state_deficit.parser_compatibility
            return (
                parser_compatibility.disposition
                if parser_compatibility is not None
                else None
            )
        if policy_family is WorkflowPolicyFamily.INTERACTION:
            interaction = state_deficit.interaction
            return interaction.disposition if interaction is not None else None
        return None

    def _explicit_request_priority(
        self,
        group: _WorkflowCandidateGroup,
    ) -> int:
        """Return same-family explicit-request preference rank."""

        return 0 if any(
            proposal.explicitly_requested for proposal in group.proposals
        ) else 1

    def _exact_goal_support_priority(
        self,
        capability: WorkflowActionCapability,
        *,
        domain: WorkflowActionDomain,
    ) -> int:
        """Return a tie-break rank based on exact requested-goal support."""

        exact_goal_support_count = sum(
            1
            for goal in domain.requested_goals
            if capability.supports_proposition(
                scope=goal.scope,
                value=goal.value,
            )
        )
        return -exact_goal_support_count

    def _effect_class_priority(
        self,
        capability: WorkflowActionCapability,
    ) -> int:
        """Return the effect-class tie-break rank for one capability."""

        effect_class_order = {
            WorkflowActionEffectClass.AUGMENTS_ABSENCE: 0,
            WorkflowActionEffectClass.REMOVES_PRESENT: 1,
            WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY: 2,
        }
        return effect_class_order[capability.effect_class]

    def _locality_priority(
        self,
        capability: WorkflowActionCapability,
    ) -> int:
        """Return the best supported locality rank for one capability."""

        locality_order = {
            WorkflowActionLocality.LOCAL_SCOPE: 0,
            WorkflowActionLocality.RESIDUE_SET: 1,
            WorkflowActionLocality.RESIDUE_SPAN: 2,
            WorkflowActionLocality.WHOLE_STRUCTURE: 3,
        }
        return min(
            locality_order[locality]
            for locality in capability.supported_localities
        )

    def _donor_priority(
        self,
        capability: WorkflowActionCapability,
    ) -> int:
        """Return donor-aversion rank for one capability."""

        donor_order = {
            WorkflowActionDonorRequirement.NONE: 0,
            WorkflowActionDonorRequirement.EXTERNAL_DONOR: 1,
        }
        return donor_order[capability.donor_requirement]



def _strongest_disposition(
    dispositions: Iterable[WorkflowDeficitDisposition],
) -> WorkflowDeficitDisposition | None:
    """Return the strongest planner disposition from one iterable."""

    strongest: WorkflowDeficitDisposition | None = None
    disposition_order = {
        WorkflowDeficitDisposition.REQUIRED: 0,
        WorkflowDeficitDisposition.OPTIONAL: 1,
        WorkflowDeficitDisposition.BLOCKED: 2,
    }
    for disposition in dispositions:
        if strongest is None or disposition_order[disposition] < disposition_order[
            strongest
        ]:
            strongest = disposition
    return strongest
