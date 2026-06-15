"""Terminal-boundary state facts for structure projections."""

from dataclasses import dataclass, field
from typing import cast

from protrepair.scope import ResidueBoundaryScope, ResidueBoundarySide
from protrepair.state.scoped import CarrierScopedState
from protrepair.state.structure_axes import BoundaryAuthenticityState, OxtPresenceState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId

__all__ = [
    "StructureBoundaryStateFacts",
    "TerminalBoundaryObservation",
    "terminal_boundary_observation",
]


@dataclass(frozen=True, slots=True)
class TerminalBoundaryObservation:
    """Observed terminal-boundary facts over residue-boundary scopes."""

    authenticity_facts: tuple[
        CarrierScopedState[ProteinStructure, BoundaryAuthenticityState],
        ...,
    ] = ()
    oxt_presence_facts: tuple[
        CarrierScopedState[ProteinStructure, OxtPresenceState],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        ordered_authenticity_facts = tuple(
            sorted(
                self.authenticity_facts,
                key=lambda fact: (
                    cast(ResidueBoundaryScope, fact.scope).residue_id,
                    0
                    if cast(ResidueBoundaryScope, fact.scope).side
                    is ResidueBoundarySide.N_TERMINUS
                    else 1,
                ),
            )
        )
        authenticity_fact_keys = tuple(
            boundary_fact.scope for boundary_fact in ordered_authenticity_facts
        )
        if len(set(authenticity_fact_keys)) != len(authenticity_fact_keys):
            raise ValueError(
                "terminal boundary observation must not repeat authenticity facts"
            )
        ordered_oxt_presence_facts = tuple(
            sorted(
                self.oxt_presence_facts,
                key=lambda fact: cast(ResidueBoundaryScope, fact.scope).residue_id,
            )
        )
        oxt_presence_fact_keys = tuple(
            oxt_presence_fact.scope for oxt_presence_fact in ordered_oxt_presence_facts
        )
        if len(set(oxt_presence_fact_keys)) != len(oxt_presence_fact_keys):
            raise ValueError(
                "terminal boundary observation must not repeat OXT presence facts"
            )
        if any(
            cast(ResidueBoundaryScope, authenticity_fact.scope).side
            is not cast(ResidueBoundaryScope, oxt_presence_fact.scope).side
            for authenticity_fact in ordered_authenticity_facts
            for oxt_presence_fact in ordered_oxt_presence_facts
            if authenticity_fact.scope == oxt_presence_fact.scope
        ):
            raise ValueError(
                "terminal boundary observation scope equality must preserve side"
            )
        if any(
            cast(ResidueBoundaryScope, oxt_presence_fact.scope).side
            is not ResidueBoundarySide.C_TERMINUS
            for oxt_presence_fact in ordered_oxt_presence_facts
        ):
            raise ValueError(
                "OXT presence facts require C-terminal residue-boundary scopes"
            )

        object.__setattr__(self, "authenticity_facts", ordered_authenticity_facts)
        object.__setattr__(self, "oxt_presence_facts", ordered_oxt_presence_facts)

    @classmethod
    def empty(cls) -> "TerminalBoundaryObservation":
        """Return one empty observation for projections without protein boundaries."""

        return cls()

    def authenticity_facts_for_side(
        self,
        side: ResidueBoundarySide,
    ) -> tuple[
        CarrierScopedState[ProteinStructure, BoundaryAuthenticityState],
        ...,
    ]:
        """Return boundary-authenticity facts for one specific side."""

        return tuple(
            boundary_fact
            for boundary_fact in self.authenticity_facts
            if cast(ResidueBoundaryScope, boundary_fact.scope).side is side
        )

    def observed_boundary_scopes(
        self,
        side: ResidueBoundarySide,
    ) -> tuple[ResidueBoundaryScope, ...]:
        """Return observed residue-boundary scopes for one side."""

        return tuple(
            cast(ResidueBoundaryScope, boundary_fact.scope)
            for boundary_fact in self.authenticity_facts_for_side(side)
        )

    def authentic_residue_ids(
        self,
        side: ResidueBoundarySide,
    ) -> tuple[ResidueId, ...]:
        """Return authentic residue ids for one observed boundary side."""

        return tuple(
            cast(ResidueBoundaryScope, boundary_fact.scope).residue_id
            for boundary_fact in self.authenticity_facts_for_side(side)
            if boundary_fact.value is BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT
        )

    def has_authentic_boundary(
        self,
        side: ResidueBoundarySide,
    ) -> bool:
        """Return whether one side is authentic for any projected residue."""

        return any(
            cast(ResidueBoundaryScope, boundary_fact.scope).side is side
            and boundary_fact.value is BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT
            for boundary_fact in self.authenticity_facts
        )

    def authenticity_for_scope(
        self,
        scope: ResidueBoundaryScope,
    ) -> BoundaryAuthenticityState | None:
        """Return boundary authenticity for one observed residue-boundary scope."""

        for boundary_fact in self.authenticity_facts:
            if boundary_fact.scope == scope:
                return boundary_fact.value

        return None

    def oxt_presence_for_scope(
        self,
        scope: ResidueBoundaryScope,
    ) -> OxtPresenceState | None:
        """Return OXT presence for one observed residue-boundary scope."""

        for oxt_presence_fact in self.oxt_presence_facts:
            if oxt_presence_fact.scope == scope:
                return oxt_presence_fact.value

        return None


@dataclass(frozen=True, slots=True)
class StructureBoundaryStateFacts:
    """Whole-structure boundary truth without unrelated completion axes."""

    terminal_boundary_observation: TerminalBoundaryObservation = field(
        default_factory=TerminalBoundaryObservation.empty
    )

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
    ) -> "StructureBoundaryStateFacts":
        """Derive whole-structure boundary truth from one canonical structure."""

        return cls(
            terminal_boundary_observation=terminal_boundary_observation(
                structure,
                residues=tuple(
                    structure.constitution.iter_residues(include_ligands=False)
                ),
            )
        )


def terminal_boundary_observation(
    context_structure: ProteinStructure,
    *,
    residues: tuple[ResidueSite, ...],
) -> TerminalBoundaryObservation:
    """Return terminal-boundary observation for one residue projection."""

    if not residues:
        return TerminalBoundaryObservation.empty()

    projected_residue_ids = {residue.residue_id for residue in residues}
    authenticity_facts: list[
        CarrierScopedState[ProteinStructure, BoundaryAuthenticityState]
    ] = []
    oxt_presence_facts: list[CarrierScopedState[ProteinStructure, OxtPresenceState]] = (
        []
    )
    for chain in context_structure.constitution.chains:
        projected_chain_residues = tuple(
            residue
            for residue in chain.residues
            if residue.residue_id in projected_residue_ids
        )
        if not projected_chain_residues:
            continue

        first_projected_residue = projected_chain_residues[0]
        last_projected_residue = projected_chain_residues[-1]
        first_context_residue = chain.residues[0]
        last_context_residue = chain.residues[-1]
        first_boundary_scope = ResidueBoundaryScope(
            residue_id=first_projected_residue.residue_id,
            side=ResidueBoundarySide.N_TERMINUS,
        )
        last_boundary_scope = ResidueBoundaryScope(
            residue_id=last_projected_residue.residue_id,
            side=ResidueBoundarySide.C_TERMINUS,
        )
        authenticity_facts.append(
            CarrierScopedState(
                carrier=context_structure,
                scope=first_boundary_scope,
                value=(
                    BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT
                    if first_projected_residue.residue_id
                    == first_context_residue.residue_id
                    else BoundaryAuthenticityState.PROJECTED_FRAGMENT_BOUNDARY
                ),
            )
        )
        authenticity_facts.append(
            CarrierScopedState(
                carrier=context_structure,
                scope=last_boundary_scope,
                value=(
                    BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT
                    if last_projected_residue.residue_id
                    == last_context_residue.residue_id
                    else BoundaryAuthenticityState.PROJECTED_FRAGMENT_BOUNDARY
                ),
            )
        )
        oxt_presence_facts.append(
            CarrierScopedState(
                carrier=context_structure,
                scope=last_boundary_scope,
                value=(
                    OxtPresenceState.PRESENT
                    if last_projected_residue.has_atom_site("OXT")
                    else OxtPresenceState.ABSENT
                ),
            )
        )

    return TerminalBoundaryObservation(
        authenticity_facts=tuple(authenticity_facts),
        oxt_presence_facts=tuple(oxt_presence_facts),
    )
