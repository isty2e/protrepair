"""Generic speculative-planning contracts shared by proposal-based transforms."""

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from typing_extensions import Self

ProposalT = TypeVar("ProposalT")
ExecutionOutcomeT = TypeVar("ExecutionOutcomeT")
EvaluationT = TypeVar("EvaluationT")
ExecutionErrorT = TypeVar("ExecutionErrorT", bound=Exception)
NodeStateT = TypeVar("NodeStateT")


@dataclass(frozen=True, slots=True)
class SpeculativeExecution(
    Generic[ProposalT, ExecutionOutcomeT]
):
    """Executed speculative proposal before downstream evaluation."""

    proposal: ProposalT
    outcome: ExecutionOutcomeT


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionBatch(
    Generic[ProposalT, ExecutionOutcomeT, ExecutionErrorT]
):
    """Batch execution output over one speculative proposal family."""

    executions: tuple[SpeculativeExecution[ProposalT, ExecutionOutcomeT], ...]
    errors: tuple[ExecutionErrorT, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluatedSpeculativeProposal(
    Generic[ProposalT, ExecutionOutcomeT, EvaluationT]
):
    """Executed speculative proposal plus evaluation outcome."""

    execution: SpeculativeExecution[ProposalT, ExecutionOutcomeT]
    evaluation: EvaluationT


@dataclass(frozen=True, slots=True)
class SpeculativeEvaluationBatch(
    Generic[ProposalT, ExecutionOutcomeT, EvaluationT, ExecutionErrorT]
):
    """Evaluation output over a speculative execution batch."""

    evaluated_proposals: tuple[
        EvaluatedSpeculativeProposal[
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
        ...,
    ]
    execution_errors: tuple[ExecutionErrorT, ...] = ()


class SpeculativeAdoptionDecisionKind(str, Enum):
    """Closed adoption outcomes for one evaluated speculative proposal."""

    ADOPT = "adopt"
    REJECT = "reject"


class SpeculativeStopReason(str, Enum):
    """Closed stop reasons for one speculative planning trace."""

    NO_PROPOSALS_AVAILABLE = "no_proposals_available"
    ITERATION_LIMIT_REACHED = "iteration_limit_reached"


@dataclass(frozen=True, order=True, slots=True)
class SpeculativePlanningNodeId:
    """Stable node identifier within one speculative planning trace."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(
                "speculative planning node ids must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class SpeculativeAdoptedChild(
    Generic[NodeStateT, ProposalT, ExecutionOutcomeT, EvaluationT]
):
    """One adopted child state produced while expanding an active node."""

    state: NodeStateT
    adopted_decision: (
        "SpeculativeAdoptionDecision[ProposalT, ExecutionOutcomeT, EvaluationT]"
    )


@dataclass(frozen=True, slots=True)
class SpeculativeAdoptionDecision(
    Generic[ProposalT, ExecutionOutcomeT, EvaluationT]
):
    """Adoption decision over one evaluated speculative proposal."""

    kind: SpeculativeAdoptionDecisionKind
    candidate: EvaluatedSpeculativeProposal[
        ProposalT,
        ExecutionOutcomeT,
        EvaluationT,
    ] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SpeculativeAdoptionDecisionKind):
            raise TypeError(
                "speculative adoption decisions require a "
                "SpeculativeAdoptionDecisionKind value"
            )
        if (
            self.kind is SpeculativeAdoptionDecisionKind.ADOPT
            and self.candidate is None
        ):
            raise ValueError(
                "adopted speculative decisions require an evaluated candidate"
            )
        if (
            self.kind is SpeculativeAdoptionDecisionKind.REJECT
            and self.candidate is not None
        ):
            raise ValueError(
                "rejected speculative decisions must not carry a candidate"
            )

    @classmethod
    def adopt(
        cls,
        candidate: EvaluatedSpeculativeProposal[
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
    ) -> Self:
        """Return one adopted speculative proposal decision."""

        return cls(
            kind=SpeculativeAdoptionDecisionKind.ADOPT,
            candidate=candidate,
        )

    @classmethod
    def reject(
        cls,
        *,
        reason: str | None = None,
    ) -> Self:
        """Return one rejected speculative proposal decision."""

        return cls(
            kind=SpeculativeAdoptionDecisionKind.REJECT,
            reason=reason,
        )

    def require_candidate(
        self,
    ) -> EvaluatedSpeculativeProposal[
        ProposalT,
        ExecutionOutcomeT,
        EvaluationT,
    ]:
        """Return the adopted candidate or raise when no adoption occurred."""

        if self.candidate is None:
            raise ValueError("speculative adoption decision has no candidate")

        return self.candidate


@dataclass(frozen=True, slots=True)
class SpeculativePlanningNode(
    Generic[NodeStateT, ProposalT, ExecutionOutcomeT, EvaluationT]
):
    """One node in a speculative planning trace."""

    node_id: SpeculativePlanningNodeId
    state: NodeStateT
    parent_node_id: SpeculativePlanningNodeId | None = None
    adopted_decision: SpeculativeAdoptionDecision[
        ProposalT,
        ExecutionOutcomeT,
        EvaluationT,
    ] | None = None

    def __post_init__(self) -> None:
        if self.parent_node_id is None and self.adopted_decision is not None:
            raise ValueError(
                "root speculative planning nodes must not carry an adopted decision"
            )
        if self.parent_node_id is not None and self.adopted_decision is None:
            raise ValueError(
                "non-root speculative planning nodes require an adopted decision"
            )

    @classmethod
    def root(
        cls,
        *,
        state: NodeStateT,
        node_id: SpeculativePlanningNodeId | None = None,
    ) -> Self:
        """Return one root planning node."""

        return cls(
            node_id=(
                SpeculativePlanningNodeId(0)
                if node_id is None
                else node_id
            ),
            state=state,
        )

    @classmethod
    def adopted_child(
        cls,
        *,
        node_id: SpeculativePlanningNodeId,
        state: NodeStateT,
        parent_node_id: SpeculativePlanningNodeId,
        adopted_decision: SpeculativeAdoptionDecision[
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
    ) -> Self:
        """Return one child planning node reached by adoption."""

        return cls(
            node_id=node_id,
            state=state,
            parent_node_id=parent_node_id,
            adopted_decision=adopted_decision,
        )


@dataclass(frozen=True, slots=True)
class SpeculativePlanningFrontier(
    Generic[NodeStateT, ProposalT, ExecutionOutcomeT, EvaluationT]
):
    """Active planning nodes awaiting further proposal generation."""

    active_nodes: tuple[
        SpeculativePlanningNode[
            NodeStateT,
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
        ...,
    ]

    def __post_init__(self) -> None:
        active_nodes = tuple(self.active_nodes)
        node_ids = tuple(node.node_id for node in active_nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(
                "speculative planning frontiers must not repeat active node ids"
            )
        object.__setattr__(self, "active_nodes", active_nodes)

    @classmethod
    def empty(cls) -> Self:
        """Return one empty planning frontier."""

        return cls(active_nodes=())

    @classmethod
    def single(
        cls,
        node: SpeculativePlanningNode[
            NodeStateT,
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
    ) -> Self:
        """Return one single-node planning frontier."""

        return cls(active_nodes=(node,))

    def select_next_active_node(
        self,
    ) -> SpeculativePlanningNode[
        NodeStateT,
        ProposalT,
        ExecutionOutcomeT,
        EvaluationT,
    ]:
        """Return the next active node or raise when the frontier is empty."""

        if not self.active_nodes:
            raise ValueError(
                "speculative planning frontier has no active nodes"
            )

        return self.active_nodes[0]

    def require_single_active_node(
        self,
    ) -> SpeculativePlanningNode[
        NodeStateT,
        ProposalT,
        ExecutionOutcomeT,
        EvaluationT,
    ]:
        """Return the single active node or raise when frontier fan-out differs."""

        active_node = self.select_next_active_node()
        if len(self.active_nodes) != 1:
            raise ValueError(
                "speculative planning frontier requires exactly one active node"
            )

        return active_node


@dataclass(frozen=True, slots=True)
class SpeculativePlanningTrace(
    Generic[NodeStateT, ProposalT, ExecutionOutcomeT, EvaluationT]
):
    """Trace of nodes explored by one speculative planning run."""

    nodes: tuple[
        SpeculativePlanningNode[
            NodeStateT,
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
        ...,
    ]
    frontier: SpeculativePlanningFrontier[
        NodeStateT,
        ProposalT,
        ExecutionOutcomeT,
        EvaluationT,
    ]
    stop_reason: SpeculativeStopReason | None = None

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        node_ids = tuple(node.node_id for node in nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(
                "speculative planning traces must not repeat node ids"
            )
        known_node_ids = set(node_ids)
        active_node_ids = {
            active_node.node_id for active_node in self.frontier.active_nodes
        }
        if not active_node_ids.issubset(known_node_ids):
            raise ValueError(
                "speculative planning frontier nodes must belong to the trace"
            )
        object.__setattr__(self, "nodes", nodes)

    @classmethod
    def root(
        cls,
        *,
        initial_state: NodeStateT,
    ) -> Self:
        """Return one root trace with a single active node."""

        root_node = SpeculativePlanningNode.root(state=initial_state)
        return cls(
            nodes=(root_node,),
            frontier=SpeculativePlanningFrontier.single(root_node),
        )

    def expand_active_node(
        self,
        *,
        parent_node: SpeculativePlanningNode[
            NodeStateT,
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
        adopted_children: tuple[
            SpeculativeAdoptedChild[
                NodeStateT,
                ProposalT,
                ExecutionOutcomeT,
                EvaluationT,
            ],
            ...,
        ] = (),
    ) -> Self:
        """Return a new trace after consuming one active node and adding children."""

        if parent_node not in self.frontier.active_nodes:
            raise ValueError(
                "speculative planning expansion requires one active frontier node"
            )

        next_node_id = len(self.nodes)
        child_nodes = tuple(
            SpeculativePlanningNode.adopted_child(
                node_id=SpeculativePlanningNodeId(next_node_id + index),
                state=child.state,
                parent_node_id=parent_node.node_id,
                adopted_decision=child.adopted_decision,
            )
            for index, child in enumerate(adopted_children)
        )
        retained_active_nodes = tuple(
            active_node
            for active_node in self.frontier.active_nodes
            if active_node.node_id != parent_node.node_id
        )
        return type(self)(
            nodes=self.nodes + child_nodes,
            frontier=SpeculativePlanningFrontier(
                active_nodes=retained_active_nodes + child_nodes
            ),
        )

    def append_adopted_child(
        self,
        *,
        parent_node: SpeculativePlanningNode[
            NodeStateT,
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
        child_state: NodeStateT,
        adopted_decision: SpeculativeAdoptionDecision[
            ProposalT,
            ExecutionOutcomeT,
            EvaluationT,
        ],
    ) -> Self:
        """Return a new trace with one adopted child node made active."""

        return self.expand_active_node(
            parent_node=parent_node,
            adopted_children=(
                SpeculativeAdoptedChild(
                    state=child_state,
                    adopted_decision=adopted_decision,
                ),
            ),
        )

    def stop(
        self,
        *,
        reason: SpeculativeStopReason,
    ) -> Self:
        """Return one stopped trace with an empty frontier."""

        return type(self)(
            nodes=self.nodes,
            frontier=SpeculativePlanningFrontier.empty(),
            stop_reason=reason,
        )
