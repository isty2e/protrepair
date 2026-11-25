"""Canonical side-chain packing execution runtime."""

from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.packing.backbone_regions import (
    pack_usable_backbone_regions,
    unresolved_backbone_residue_ids,
)
from protrepair.transformer.packing.backend import SidechainPackingBackend
from protrepair.transformer.packing.domain import PackingPlan, PackingResult
from protrepair.transformer.packing.faspr.backend import FasprPackingBackend
from protrepair.transformer.packing.spec import PackingSpec


def execute_sidechain_packing(
    structure: ProteinStructure,
    spec: PackingSpec,
) -> PackingResult:
    """Execute one canonical side-chain packing transformation.

    When the active run marks backbone atoms unresolved, FASPR packs only the
    usable regions; otherwise the backend receives the ordinary single call.
    """

    packing_plan = PackingPlan.from_inputs(structure, spec)
    backend = resolve_sidechain_packing_backend(spec.backend_name)
    unresolved = unresolved_backbone_residue_ids(structure)
    if unresolved and isinstance(backend, FasprPackingBackend):
        return pack_usable_backbone_regions(
            packing_plan,
            backend=backend,
            unresolved_residue_ids=unresolved,
        )

    return backend.pack(packing_plan)


def resolve_sidechain_packing_backend(
    backend_name: str,
) -> SidechainPackingBackend:
    """Return the internal backend implementation for one backend name."""

    if backend_name == "faspr":
        return FasprPackingBackend()

    raise NotImplementedError(
        f"side-chain packing backend {backend_name!r} is not implemented"
    )
