"""Plan a bounded next attempt from qualified terms and observed coupling."""

from collections import defaultdict

from protrepair.scope import AtomSetScope
from protrepair.state.bonded_geometry import BondedTerm
from protrepair.structure.labels import AtomRef
from protrepair.workflow.actions.bonded_geometry import BondedGeometryRepairTransformer
from protrepair.workflow.actions.bonded_geometry_motion import BondedGeometryMotion
from protrepair.workflow.planning.action.attempt import WorkflowActionAttempt
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import (
    WorkflowActionProposal,
    WorkflowProposalSelectionMode,
)
from protrepair.workflow.planning.default_action_registry.capabilities import (
    BONDED_GEOMETRY_CAPABILITY,
)


def bonded_geometry_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Do not activate motion from report-only findings."""

    return domain.requested_goals.requests_bonded_geometry_repair()


def bonded_geometry_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Offer one scope per planning step; the workflow meters each attempt."""

    geometry = domain.chemistry_observation.bonded_geometry
    motion = BondedGeometryMotion(domain.structure, geometry.qualified_residue_ids)
    history: dict[BondedTerm, list[WorkflowActionAttempt]] = defaultdict(list)
    followups: set[BondedTerm] = set()
    for attempt in domain.attempts:
        if isinstance(attempt.transformer, BondedGeometryRepairTransformer):
            focus = attempt.transformer.focus
            history[focus].append(attempt)
            feedback = attempt.bonded_geometry_feedback
            if (
                attempt.input_structure is domain.structure
                and feedback is not None
                and (feedback.coupled_atoms or feedback.residual_focus)
            ):
                followups.add(focus)

    for focus, _ in sorted(
        geometry.residual_excesses,
        key=lambda item: (
            item[0] not in followups,
            len(history[item[0]]),
            -item[1],
            item[0],
        ),
    ):
        mover = next_bonded_geometry_transformer(focus, tuple(history[focus]), motion)
        if mover is not None:
            return (
                WorkflowActionProposal(
                    transformer=mover,
                    capability=BONDED_GEOMETRY_CAPABILITY,
                    selection_mode=WorkflowProposalSelectionMode.QUALITY_ALTERNATIVE,
                ),
            )
    return ()


def next_bonded_geometry_transformer(
    focus: BondedTerm,
    history: tuple[WorkflowActionAttempt, ...],
    motion: BondedGeometryMotion,
) -> BondedGeometryRepairTransformer | None:
    """Choose a still-untried scope for a focus and its observed attempts."""

    attempted = {
        attempt.transformer
        for attempt in history
        if attempt.blocks(attempt.transformer, motion.structure)
    }
    seed = frozenset(atom for ref in focus for atom in motion.group_for(ref))
    if not seed:
        return None
    narrow = BondedGeometryRepairTransformer(focus, AtomSetScope(tuple(sorted(seed))))
    if narrow not in attempted:
        return narrow

    for attempt in reversed(history):
        previous = attempt.transformer
        feedback = attempt.bonded_geometry_feedback
        if (
            not isinstance(previous, BondedGeometryRepairTransformer)
            or feedback is None
        ):
            continue
        same_parent = attempt.input_structure is motion.structure
        if not same_parent and feedback.failure is not None:
            continue
        if not motion.allows(previous.scope):
            continue
        refs = frozenset(previous.scope.atom_refs)
        candidates: list[AtomRef] = []
        if same_parent:
            for ref in feedback.coupled_atoms:
                partner = motion.heavy_contact_partner(ref)
                if partner is not None:
                    candidates.append(partner)
        if feedback.residual_focus:
            candidates.extend(motion.covalent_frontier(refs))
        for candidate in candidates:
            expanded = refs | motion.group_for(candidate)
            if expanded == refs:
                continue
            transformer = BondedGeometryRepairTransformer(
                focus, AtomSetScope(tuple(sorted(expanded)))
            )
            if transformer not in attempted:
                return transformer
    return None
