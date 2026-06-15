"""Typed transformer output artifacts and slot-space delta witnesses."""

from protrepair.transformer.artifacts.patch import (
    AtomElementRewrite,
    AtomFormalChargeRewrite,
    BondTopologyRewrite,
    GraphBondState,
    MovedAtomDelta,
    ResidueIdentityRewrite,
    ResidueTopologyRewrite,
    StructureDelta,
)
from protrepair.transformer.artifacts.result import RegionTransformationResult

__all__ = [
    "AtomElementRewrite",
    "AtomFormalChargeRewrite",
    "BondTopologyRewrite",
    "GraphBondState",
    "MovedAtomDelta",
    "ResidueIdentityRewrite",
    "ResidueTopologyRewrite",
    "RegionTransformationResult",
    "StructureDelta",
]
