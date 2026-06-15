"""Continuous local-relaxation transformer over canonical relaxation problems."""

from dataclasses import dataclass

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.errors import RefinementError
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.backend import ContinuousRelaxationBackend
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem
from protrepair.transformer.continuous.settings import ContinuousRelaxationSettings


@dataclass(frozen=True, slots=True)
class ContinuousLocalRelaxationTransformer(
    DeterministicContextOperation[RegionTransformationResult]
):
    """Deterministic transformer for continuous coordinate relaxation."""

    spec: ContinuousRelaxationSettings
    component_library: ComponentLibrary
    restraint_library: RestraintLibrary
    backend: ContinuousRelaxationBackend

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether a canonical continuous-relaxation problem can be built."""

        try:
            ContinuousRelaxationProblem.from_inputs(
                context.source_snapshot,
                context.atom_input,
                spec=self.spec,
                component_library=self.component_library,
            )
        except RefinementError:
            return False

        return True

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> RegionTransformationResult:
        """Run one continuous local-relaxation step over the selected atom domain."""

        problem = ContinuousRelaxationProblem.from_inputs(
            context.source_snapshot,
            context.atom_input,
            spec=self.spec,
            component_library=self.component_library,
        )
        return self.backend.relax(
            problem,
            restraint_library=self.restraint_library,
        )
