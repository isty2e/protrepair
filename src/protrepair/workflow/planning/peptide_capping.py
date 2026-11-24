"""Admission to fresh cap assessment on a single planning branch."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.bound_peptide_ports import BoundPeptidePorts
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType
from protrepair.workflow.actions.external_span_reconstruction import (
    ExternalSpanReconstructionTransformer,
)
from protrepair.workflow.actions.heavy_completion import HeavyAtomCompletionTransformer
from protrepair.workflow.actions.peptide_capping import PeptideCappingTransformer
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal


class PeptidePortReadiness(str, Enum):
    """Whether a new pose may be assessed on this parent."""

    ASSESSABLE = "assessable"
    PENDING_GRAFT = "pending_graft"
    PENDING_HEAVY_REPAIR = "pending_heavy_repair"
    OCCUPIED_PORT = "occupied_port"
    OCCUPIED_ADDRESS = "occupied_address"
    CHANGED_IDENTITY = "changed_identity"


@dataclass(frozen=True, slots=True)
class PeptideCapSchedule:
    """Precedence and occupancy before any new cap geometry is considered."""

    parent: ProteinStructure
    entries: tuple[tuple[AtomRef, ResidueId, PeptidePortReadiness], ...]

    @classmethod
    def on_branch(
        cls,
        binding: BoundPeptidePorts,
        parent: ProteinStructure,
        proposals: Sequence[WorkflowActionProposal],
    ) -> "PeptideCapSchedule":
        """Account for all still-executable prerequisites, before family batching.

        Parameters
        ----------
        binding : BoundPeptidePorts
            Addresses allocated on this run's original input.
        parent : ProteinStructure
            Current branch, including any restored peptide or moved flank.
        proposals : Sequence[WorkflowActionProposal]
            Admitted proposals after attempt-memory filtering and before ranking.

        Returns
        -------
        PeptideCapSchedule
            Stable per-port eligibility for a fresh placement assessment. Occupancy,
            identity and scheduling are distinct from successful chemical placement.
        """
        rechecks = {
            action.port: action.cap_id
            for p in proposals
            if isinstance(action := p.transformer, PeptideCappingTransformer)
            and action.port in binding.attached
        }
        if not binding.cap_ids and not rechecks:
            return cls(parent, ())

        grafts: set[AtomRef] = set()
        heavy: set[ResidueId] = set()
        for proposal in proposals:
            action = proposal.transformer
            if isinstance(action, ExternalSpanReconstructionTransformer):
                gap = action.workflow_scope
                if gap.preceding_residue_id is not None:
                    grafts.add(AtomRef(gap.preceding_residue_id, "C"))
                if gap.following_residue_id is not None:
                    grafts.add(AtomRef(gap.following_residue_id, "N"))
            elif isinstance(action, HeavyAtomCompletionTransformer):
                heavy.update(action.scope.residue_ids)

        constitution = parent.constitution
        incident = parent.topology.incident_bonds_by_atom()
        entries = []
        for port, address in sorted((dict(binding.cap_ids) | rechecks).items()):
            index = constitution.resolve_atom_index(port)
            if not binding.reserves(parent, port):
                status = PeptidePortReadiness.CHANGED_IDENTITY
            elif port in grafts:
                status = PeptidePortReadiness.PENDING_GRAFT
            elif port.residue_id in heavy:
                status = PeptidePortReadiness.PENDING_HEAVY_REPAIR
            elif port in rechecks:
                status = PeptidePortReadiness.ASSESSABLE
            elif index is not None and any(
                bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
                and any(
                    constitution.atom_ref_at(i).residue_id != port.residue_id
                    for i in bond.endpoint_pair()
                )
                for bond in incident.get(index, ())
            ):
                status = PeptidePortReadiness.OCCUPIED_PORT
            elif constitution.residue_or_ligand(address) is not None:
                status = PeptidePortReadiness.OCCUPIED_ADDRESS
            else:
                status = PeptidePortReadiness.ASSESSABLE
            entries.append((port, address, status))
        return cls(parent, tuple(entries))

    def ordered_proposals(
        self, proposals: list[WorkflowActionProposal]
    ) -> list[WorkflowActionProposal]:
        """Keep at most the first outstanding cap, without dropping other action families."""
        available = {
            p.transformer.port: p
            for p in proposals
            if isinstance(p.transformer, PeptideCappingTransformer)
        }
        first = next(
            (
                available[port]
                for port, _ in self.assessable_ports()
                if port in available
            ),
            None,
        )
        return [
            p
            for p in proposals
            if not isinstance(p.transformer, PeptideCappingTransformer) or p is first
        ]

    def assessable_ports(self) -> Iterator[tuple[AtomRef, ResidueId]]:
        """Yield the ready prefix; an earlier pending port cannot be overtaken."""
        for port, address, status in self.entries:
            if status in (
                PeptidePortReadiness.PENDING_GRAFT,
                PeptidePortReadiness.PENDING_HEAVY_REPAIR,
            ):
                return
            if status is PeptidePortReadiness.ASSESSABLE:
                yield port, address
