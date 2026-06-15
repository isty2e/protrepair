"""Canonical side-chain packing execution runtime."""

from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.packing.backend import SidechainPackingBackend
from protrepair.transformer.packing.domain import PackingPlan, PackingResult
from protrepair.transformer.packing.faspr.backend import FasprPackingBackend
from protrepair.transformer.packing.spec import PackingSpec


def execute_sidechain_packing(
    structure: ProteinStructure,
    spec: PackingSpec,
) -> PackingResult:
    """Execute one canonical side-chain packing transformation."""

    packing_plan = PackingPlan.from_inputs(structure, spec)
    return resolve_sidechain_packing_backend(spec.backend_name).pack(packing_plan)


def resolve_sidechain_packing_backend(
    backend_name: str,
) -> SidechainPackingBackend:
    """Return the internal backend implementation for one backend name."""

    if backend_name == "faspr":
        return FasprPackingBackend()

    raise NotImplementedError(
        f"side-chain packing backend {backend_name!r} is not implemented"
    )
