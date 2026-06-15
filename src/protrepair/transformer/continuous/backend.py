"""Execution protocol for continuous local-relaxation realizations."""

from typing import Protocol

from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem


class ContinuousRelaxationBackend(Protocol):
    """Realization seam for one canonical continuous-relaxation problem."""

    def relax(
        self,
        problem: ContinuousRelaxationProblem,
        *,
        restraint_library: RestraintLibrary,
    ) -> RegionTransformationResult:
        """Execute one canonical continuous-relaxation problem."""

        ...
