"""Pure semantic coarsening contracts and canonical scope coarsenings."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from protrepair.scope.base import (
    AtomSetScope,
    ChainSetScope,
    ResidueSetScope,
    Scope,
    WholeStructureScope,
)

SourceScopeT = TypeVar("SourceScopeT", bound=Scope)
TargetScopeT = TypeVar("TargetScopeT", bound=Scope)


class ScopeCoarsening(ABC, Generic[SourceScopeT, TargetScopeT]):
    """Pure state-independent map from one semantic scope to a coarser scope."""

    @abstractmethod
    def coarsen(self, scope: SourceScopeT) -> TargetScopeT:
        """Return the canonical coarsened scope for one semantic source scope."""


class AtomToResidueSetScopeCoarsening(
    ScopeCoarsening[AtomSetScope, ResidueSetScope]
):
    """Coarsen one atom-set scope into the residue set that covers it."""

    def coarsen(self, scope: AtomSetScope) -> ResidueSetScope:
        """Return the covering residue-set scope for one atom-set scope."""

        ordered_residue_ids = tuple(
            dict.fromkeys(atom_ref.residue_id for atom_ref in scope.atom_refs)
        )
        return ResidueSetScope(residue_ids=ordered_residue_ids)


class ResidueToChainSetScopeCoarsening(
    ScopeCoarsening[ResidueSetScope, ChainSetScope]
):
    """Coarsen one residue-set scope into the chain set that covers it."""

    def coarsen(self, scope: ResidueSetScope) -> ChainSetScope:
        """Return the covering chain-set scope for one residue-set scope."""

        ordered_chain_ids = tuple(
            dict.fromkeys(residue_id.chain_id for residue_id in scope.residue_ids)
        )
        return ChainSetScope(chain_ids=ordered_chain_ids)


class AtomToChainSetScopeCoarsening(ScopeCoarsening[AtomSetScope, ChainSetScope]):
    """Coarsen one atom-set scope directly into the covering chain set."""

    def __init__(self) -> None:
        self._atom_to_residue = AtomToResidueSetScopeCoarsening()
        self._residue_to_chain = ResidueToChainSetScopeCoarsening()

    def coarsen(self, scope: AtomSetScope) -> ChainSetScope:
        """Return the covering chain-set scope for one atom-set scope."""

        residue_scope = self._atom_to_residue.coarsen(scope)
        return self._residue_to_chain.coarsen(residue_scope)


class ScopeToWholeStructureCoarsening(ScopeCoarsening[Scope, WholeStructureScope]):
    """Coarsen any semantic scope into the whole-structure scope."""

    def coarsen(self, scope: Scope) -> WholeStructureScope:
        """Return the whole-structure scope for one semantic scope."""

        del scope
        return WholeStructureScope()


def coarsen_atom_scope_to_residue_scope(scope: AtomSetScope) -> ResidueSetScope:
    """Return the covering residue-set scope for one atom-set scope."""

    return AtomToResidueSetScopeCoarsening().coarsen(scope)


def coarsen_residue_scope_to_chain_scope(scope: ResidueSetScope) -> ChainSetScope:
    """Return the covering chain-set scope for one residue-set scope."""

    return ResidueToChainSetScopeCoarsening().coarsen(scope)


def coarsen_atom_scope_to_chain_scope(scope: AtomSetScope) -> ChainSetScope:
    """Return the covering chain-set scope for one atom-set scope."""

    return AtomToChainSetScopeCoarsening().coarsen(scope)


def coarsen_scope_to_whole_structure(scope: Scope) -> WholeStructureScope:
    """Return the whole-structure scope for one semantic scope."""

    return ScopeToWholeStructureCoarsening().coarsen(scope)
