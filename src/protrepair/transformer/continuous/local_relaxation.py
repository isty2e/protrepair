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
from protrepair.transformer.continuous.peptide_boundary import PeptideBoundaryPlan
from protrepair.transformer.continuous.peptide_caps import PeptideCapTransformer
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
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
            require_atom_scope_continuous_relaxation_execution(
                derive_atom_scope_continuous_relaxation_facts(
                    context.source_snapshot,
                    context.atom_input.observed_atom_scope(context.source_snapshot),
                    component_library=self.component_library,
                    context_radius_angstrom=self.spec.context_radius_angstrom,
                )
            )
        except RefinementError:
            return False

        return True

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> RegionTransformationResult:
        """Run one continuous local-relaxation step over the selected atom domain."""

        require_atom_scope_continuous_relaxation_execution(
            derive_atom_scope_continuous_relaxation_facts(
                context.source_snapshot,
                context.atom_input.observed_atom_scope(context.source_snapshot),
                component_library=self.component_library,
                context_radius_angstrom=self.spec.context_radius_angstrom,
            )
        )
        problem = ContinuousRelaxationProblem.from_inputs(
            context.source_snapshot,
            context.atom_input,
            spec=self.spec,
            component_library=self.component_library,
        )
        return self.relax_problem(problem)

    def relax_problem(
        self, problem: ContinuousRelaxationProblem
    ) -> RegionTransformationResult:
        """Complete calculation boundaries and project backend coordinates to source.

        Parameters
        ----------
        problem : ContinuousRelaxationProblem
            Source problem after workflow or direct readiness assessment.

        Returns
        -------
        RegionTransformationResult
            Only original source atoms, with explicit cap-use diagnostics.

        Raises
        ------
        RefinementError
            Boundary chemistry or backend output violates the calculation contract.
        """
        plan = PeptideBoundaryPlan.from_region(
            problem.region, problem.bonds, self.component_library
        )
        capped = PeptideCapTransformer().transform(problem, plan)
        result = self.backend.relax(
            capped.problem,
            restraint_library=self.restraint_library,
        )
        return capped.project_result(result)
