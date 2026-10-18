"""Invocation-local evidence of returned continuous minimizations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from protrepair.scope import AtomSetScope
from protrepair.structure.labels import AtomRef
from protrepair.transformer.continuous.settings import ContinuousRelaxationSettings

if TYPE_CHECKING:
    from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem


@dataclass(frozen=True, slots=True, eq=False)
class ContinuousRelaxationExecution:
    """One native return, independent of candidate acceptance.

    Parameters
    ----------
    settings : ContinuousRelaxationSettings
        Actual bound profile and configuration used by the minimizer.
    movable_scope : AtomSetScope
        Realized source atoms permitted to move in that invocation.
    minimizer_status : int
        Native return code; zero denotes native convergence, not candidate adoption.
    backend_version : str
        Version of the runtime that returned the minimization.

    Notes
    -----
    Identity distinguishes invocations even when their inputs are equal. This is
    in-memory return evidence, not a persistent identifier or exception telemetry.
    Absence of evidence after an exception does not prove absence of execution.
    """

    settings: ContinuousRelaxationSettings
    movable_scope: AtomSetScope
    minimizer_status: int
    backend_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.settings, ContinuousRelaxationSettings):
            raise TypeError("relaxation execution requires bound settings")
        if not isinstance(self.movable_scope, AtomSetScope):
            raise TypeError("relaxation execution requires an atom-set motion scope")
        if isinstance(self.minimizer_status, bool) or not isinstance(
            self.minimizer_status, int
        ):
            raise TypeError("native minimizer status must be an integer")
        if self.minimizer_status < 0:
            raise ValueError("native minimizer status must be non-negative")
        if not isinstance(self.backend_version, str):
            raise TypeError("relaxation execution requires a runtime version string")
        version = self.backend_version.strip()
        if not version:
            raise ValueError("relaxation execution runtime version must not be blank")

        object.__setattr__(self, "backend_version", version)

    @classmethod
    def from_native_return(
        cls,
        problem: "ContinuousRelaxationProblem",
        *,
        minimizer_status: int,
        backend_version: str,
    ) -> "ContinuousRelaxationExecution":
        """Capture the realized problem after its native minimizer returns.

        Parameters
        ----------
        problem : ContinuousRelaxationProblem
            Native calculation, including its actual motion authority.
        minimizer_status : int
            Return code from the native minimizer.
        backend_version : str
            Runtime version associated with this return.

        Returns
        -------
        ContinuousRelaxationExecution
            One invocation identity with stable source atom references.
        """

        region = problem.region
        constitution = region.snapshot.structure.constitution
        return cls(
            settings=problem.spec,
            movable_scope=AtomSetScope(
                AtomRef(
                    region.residue_site(
                        constitution.residue_index_for_atom_index(index)
                    ).residue_id,
                    region.atom_site(index).name,
                )
                for index in region.movable_atom_indices
            ),
            minimizer_status=minimizer_status,
            backend_version=backend_version,
        )

    @staticmethod
    def deduplicated(
        executions: tuple["ContinuousRelaxationExecution", ...],
    ) -> tuple["ContinuousRelaxationExecution", ...]:
        """Retain native invocations once, in their original return order.

        Parameters
        ----------
        executions : tuple[ContinuousRelaxationExecution, ...]
            Evidence that may repeat one invocation through result projection.

        Returns
        -------
        tuple[ContinuousRelaxationExecution, ...]
            Identity-deduplicated evidence; equal-input invocations remain distinct.

        Raises
        ------
        TypeError
            An entry is not continuous native-return evidence.
        """

        for execution in executions:
            if not isinstance(execution, ContinuousRelaxationExecution):
                raise TypeError(
                    "relaxation traces require continuous execution evidence"
                )

        return tuple(dict.fromkeys(executions))

    def has_motion_scope(self, scope: AtomSetScope) -> bool:
        """Compare realized movement authority independently of atom ordering.

        Parameters
        ----------
        scope : AtomSetScope
            Source atoms permitted to move.

        Returns
        -------
        bool
            Whether both scopes permit exactly the same atoms.
        """

        return frozenset(self.movable_scope.atom_refs) == frozenset(scope.atom_refs)
