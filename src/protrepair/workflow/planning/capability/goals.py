"""Goal proposition shapes supported by workflow capabilities."""

from dataclasses import dataclass
from enum import Enum

from protrepair.scope import (
    ResidueBoundaryScope,
    ResidueBoundarySide,
    Scope,
    WholeStructureScope,
)


@dataclass(frozen=True, slots=True)
class WorkflowGoalDescriptor:
    """One goal-shape descriptor supported by a workflow action family."""

    scope_type: type[Scope]
    state_type: type[Enum]
    allowed_values: tuple[Enum, ...] = ()
    boundary_side: ResidueBoundarySide | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_values", tuple(self.allowed_values))
        if self.boundary_side is not None and not issubclass(
            self.scope_type,
            ResidueBoundaryScope,
        ):
            raise ValueError(
                "boundary_side requires a residue-boundary scoped goal descriptor"
            )

    def matches(
        self,
        *,
        scope: Scope,
        value: Enum,
    ) -> bool:
        """Return whether one scoped state proposition matches this descriptor."""

        if not isinstance(scope, self.scope_type):
            return False
        if self.boundary_side is not None:
            if not isinstance(scope, ResidueBoundaryScope):
                return False
            if scope.side is not self.boundary_side:
                return False
        if not isinstance(value, self.state_type):
            return False
        return not self.allowed_values or value in self.allowed_values


def whole_structure_goal_descriptor(
    value: Enum,
) -> WorkflowGoalDescriptor:
    """Return one whole-structure goal descriptor for a specific goal value."""

    return WorkflowGoalDescriptor(
        scope_type=WholeStructureScope,
        state_type=type(value),
        allowed_values=(value,),
    )


def residue_boundary_goal_descriptor(
    *,
    side: ResidueBoundarySide,
    value: Enum,
) -> WorkflowGoalDescriptor:
    """Return one residue-boundary goal descriptor for a specific goal value."""

    return WorkflowGoalDescriptor(
        scope_type=ResidueBoundaryScope,
        state_type=type(value),
        allowed_values=(value,),
        boundary_side=side,
    )
