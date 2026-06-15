"""Backend realization for local refinement requests."""

from protrepair.errors import RefinementError
from protrepair.transformer.continuous.backend import ContinuousRelaxationBackend
from protrepair.transformer.continuous.rdkit import RdkitContinuousRelaxationBackend


def resolve_continuous_relaxation_backend(
    backend_name: str,
) -> ContinuousRelaxationBackend:
    """Return one continuous-relaxation backend realization by canonical name."""

    normalized_backend_name = backend_name.strip().lower()
    if normalized_backend_name == "rdkit":
        return RdkitContinuousRelaxationBackend()

    raise RefinementError(
        f"continuous relaxation backend {backend_name!r} is not implemented"
    )
