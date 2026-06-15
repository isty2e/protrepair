"""Shared semantic scope domain for state and transformer contracts."""

from protrepair.scope.base import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    ChainSetScope,
    CompositeScope,
    ResidueBoundaryScope,
    ResidueBoundarySide,
    ResidueSetScope,
    Scope,
    ScopeKind,
    WholeStructureScope,
)
from protrepair.scope.coarsening import (
    AtomToChainSetScopeCoarsening,
    AtomToResidueSetScopeCoarsening,
    ResidueToChainSetScopeCoarsening,
    ScopeCoarsening,
    ScopeToWholeStructureCoarsening,
    coarsen_atom_scope_to_chain_scope,
    coarsen_atom_scope_to_residue_scope,
    coarsen_residue_scope_to_chain_scope,
    coarsen_scope_to_whole_structure,
)
from protrepair.scope.lowering import ScopeLowering
from protrepair.scope.relations import scope_contains, scope_refines, scopes_overlap

__all__ = [
    "AbsentResidueSpanScope",
    "AtomToChainSetScopeCoarsening",
    "AtomToResidueSetScopeCoarsening",
    "AnchorAtomPairScope",
    "AtomSetScope",
    "ChainSetScope",
    "CompositeScope",
    "ResidueBoundaryScope",
    "ResidueBoundarySide",
    "ResidueSetScope",
    "ResidueToChainSetScopeCoarsening",
    "Scope",
    "ScopeCoarsening",
    "ScopeKind",
    "ScopeLowering",
    "ScopeToWholeStructureCoarsening",
    "WholeStructureScope",
    "coarsen_atom_scope_to_chain_scope",
    "coarsen_atom_scope_to_residue_scope",
    "coarsen_residue_scope_to_chain_scope",
    "coarsen_scope_to_whole_structure",
    "scope_contains",
    "scope_refines",
    "scopes_overlap",
]
