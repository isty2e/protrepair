"""Structural outcomes of a returned continuous invocation."""

from dataclasses import dataclass, field

from typing_extensions import Self

from protrepair.transformer.artifacts.result import RegionTransformationResult
from protrepair.transformer.continuous.execution import ContinuousRelaxationExecution


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationResult(RegionTransformationResult):
    """One continuous return, including a rejected endpoint's invocation fact.

    Parameters
    ----------
    execution : ContinuousRelaxationExecution
        The invocation that returned this endpoint. Convergence does not imply
        geometric acceptance; rejected endpoints may retain the original structure.
    """

    execution: ContinuousRelaxationExecution = field(
        compare=False, repr=False, kw_only=True
    )

    def __post_init__(self) -> None:
        RegionTransformationResult.__post_init__(self)
        if not isinstance(self.execution, ContinuousRelaxationExecution):
            raise TypeError("continuous results require one native return fact")

    @classmethod
    def from_transformation(
        cls,
        result: RegionTransformationResult,
        *,
        execution: ContinuousRelaxationExecution,
    ) -> Self:
        """Associate a structural endpoint with its returned native invocation.

        Parameters
        ----------
        result : RegionTransformationResult
            Accepted or rejected endpoint after native guards.
        execution : ContinuousRelaxationExecution
            The invocation that produced the endpoint.

        Returns
        -------
        Self
            The same structural fields with one validated invocation fact.

        Raises
        ------
        TypeError
            The invocation fact has the wrong type.
        """

        return cls(
            refined_structure=result.refined_structure,
            delta=result.delta,
            issues=result.issues,
            backend_name=result.backend_name,
            backend_version=result.backend_version,
            execution=execution,
        )
