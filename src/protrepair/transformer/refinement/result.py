"""Selected local outcomes and their observed continuous invocations."""

from dataclasses import dataclass, field, replace

from typing_extensions import Self

from protrepair.transformer.artifacts.result import RegionTransformationResult
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution
from protrepair.transformer.continuous.result import ContinuousRelaxationResult
from protrepair.transformer.result import TransformationResult


@dataclass(frozen=True, slots=True)
class RefinementTransformationResult(TransformationResult):
    """A structure transition carrying observed native refinement returns.

    Parameters
    ----------
    relaxation_executions : tuple[ContinuousRelaxationExecution, ...]
        Invocation facts, independent of coordinate adoption. Empty facts after
        an exception do not prove that no native work ran.
    """

    relaxation_executions: tuple[ContinuousRelaxationExecution, ...] = field(
        default=(), compare=False, repr=False, kw_only=True
    )

    def __post_init__(self) -> None:
        TransformationResult.__post_init__(self)
        object.__setattr__(
            self,
            "relaxation_executions",
            ContinuousRelaxationExecution.deduplicated(self.relaxation_executions),
        )

    @staticmethod
    def observed_executions(
        result: TransformationResult,
    ) -> tuple[ContinuousRelaxationExecution, ...]:
        """Return transported native facts without inferring them from repairs.

        Parameters
        ----------
        result : TransformationResult
            Structural carrier, possibly with refinement-owned evidence.

        Returns
        -------
        tuple[ContinuousRelaxationExecution, ...]
            Known returns, or no evidence for an ordinary structural carrier.
        """

        return (
            result.relaxation_executions
            if isinstance(result, RefinementTransformationResult)
            else ()
        )

    @classmethod
    def with_observed_executions(
        cls,
        result: TransformationResult,
        executions: tuple[ContinuousRelaxationExecution, ...],
    ) -> "RefinementTransformationResult":
        """Preserve selected structural fields and merge native invocation facts.

        Parameters
        ----------
        result : TransformationResult
            Selected structure and diagnostics, including any prior native facts.
        executions : tuple[ContinuousRelaxationExecution, ...]
            Additional observed invocations in execution order.

        Returns
        -------
        RefinementTransformationResult
            Carrier with identity-deduplicated evidence and unchanged diagnostics.

        Raises
        ------
        TypeError
            A supplied record is not native execution evidence.
        """

        combined = ContinuousRelaxationExecution.deduplicated(
            (*cls.observed_executions(result), *executions)
        )
        if isinstance(result, cls):
            return replace(result, relaxation_executions=combined)

        return cls(
            structure=result.structure,
            repairs=result.repairs,
            issues=result.issues,
            relaxation_executions=combined,
        )


@dataclass(frozen=True, slots=True)
class LocalRefinementResult(RegionTransformationResult):
    """A local selection with native facts independent of the selected geometry.

    Parameters
    ----------
    relaxation_executions : tuple[ContinuousRelaxationExecution, ...]
        All observed native returns, including rejected or unselected candidates.
        Discrete-only success has no native returns. Repeated transport preserves
        identity without adding duplicate invocations.

    Notes
    -----
    An empty record after an exception does not prove that no native work ran.
    These facts are not resumable solver state or accumulated workflow history.
    """

    relaxation_executions: tuple[ContinuousRelaxationExecution, ...] = field(
        default=(), compare=False, repr=False, kw_only=True
    )

    def __post_init__(self) -> None:
        RegionTransformationResult.__post_init__(self)
        object.__setattr__(
            self,
            "relaxation_executions",
            ContinuousRelaxationExecution.deduplicated(self.relaxation_executions),
        )

    @staticmethod
    def observed_executions(
        result: RegionTransformationResult,
    ) -> tuple[ContinuousRelaxationExecution, ...]:
        """Project known invocation facts without inferring unobserved execution.

        Parameters
        ----------
        result : RegionTransformationResult
            A structural, continuous or previously selected local outcome.

        Returns
        -------
        tuple[ContinuousRelaxationExecution, ...]
            The known native facts; an ordinary structural result supplies none.
        """

        if isinstance(result, ContinuousRelaxationResult):
            return (result.execution,)
        if isinstance(result, LocalRefinementResult):
            return result.relaxation_executions
        return ()

    @classmethod
    def from_selection(
        cls,
        result: RegionTransformationResult,
        *,
        relaxation_executions: tuple[ContinuousRelaxationExecution, ...],
    ) -> Self:
        """Project selected geometry while retaining all observed native returns.

        Parameters
        ----------
        result : RegionTransformationResult
            Selected structural outcome, possibly a discrete candidate or parent.
        relaxation_executions : tuple[ContinuousRelaxationExecution, ...]
            Observed invocations before selection.

        Returns
        -------
        Self
            Selected structural fields and identity-deduplicated native facts.

        Raises
        ------
        TypeError
            A supplied invocation fact has the wrong type.
        ValueError
            Native facts from the selected continuous or local result are missing.
        """

        selected = cls(
            refined_structure=result.refined_structure,
            delta=result.delta,
            issues=result.issues,
            backend_name=result.backend_name,
            backend_version=result.backend_version,
            relaxation_executions=relaxation_executions,
        )
        required_executions = cls.observed_executions(result)
        if required_executions and not frozenset(required_executions).issubset(
            selected.relaxation_executions
        ):
            raise ValueError("local selection must retain its selected native return")
        return selected
