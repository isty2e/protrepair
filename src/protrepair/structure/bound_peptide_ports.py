"""Run-local reservations for selected, source-identified peptide termini."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from protrepair.structure.cap_attachment import PeptideCapAttachment
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.sequence_boundary import (
    DepositedPeptideBoundary,
    PeptideSequenceBoundary,
)
from protrepair.structure.topology import BondRelationshipType

if TYPE_CHECKING:
    from protrepair.structure.aggregate import ProteinStructure


@dataclass(frozen=True, slots=True, eq=False)
class BoundPeptidePorts:
    """Original polymer occurrences whose free-terminal writes are reserved."""

    source: ProteinStructure
    components: Mapping[AtomRef, str]
    cap_ids: Mapping[AtomRef, ResidueId]
    selected: tuple[AtomRef, ...]
    facts: Mapping[AtomRef, DepositedPeptideBoundary]
    attached: Mapping[AtomRef, ResidueId]
    qualified_gap_ports: frozenset[AtomRef]

    @classmethod
    def bind(
        cls,
        source: ProteinStructure,
        *,
        only: tuple[AtomRef, ...] | None = None,
        graft_ids: frozenset[ResidueId] = frozenset(),
        qualified_gap_ports: frozenset[AtomRef] = frozenset(),
    ) -> BoundPeptidePorts:
        """Reserve selected gaps and known cap partners before any repair can run.

        Parameters
        ----------
        source : ProteinStructure
            Normalized workflow input with original sequence and connection evidence.
        only : tuple[AtomRef, ...] or None
            Explicit N/C ports, or all evidenced ports. A construct end by itself
            does not authorize withholding free-terminal chemistry.
        graft_ids : frozenset[ResidueId]
            Addresses reserved for independently requested span reconstructions.
        qualified_gap_ports : frozenset[AtomRef]
            Gaps whose possible deposited neighbors all have linking chemistry.
            Reservations alone do not license construction.

        Returns
        -------
        BoundPeptidePorts
            Source-bound reservations and distinct addresses for candidate gaps.
            Actual bond occupancy and placement are checked on each candidate branch.

        Raises
        ------
        ValueError
            An explicit selection is empty, repeated, or includes a non-port atom.
        """
        selected = None if only is None else frozenset(only)
        if only is not None and (
            not only
            or len(selected or ()) != len(only)
            or any(port.atom_name not in ("N", "C") for port in only)
        ):
            raise ValueError("select distinct, nonempty peptide N/C ports")
        ingress = source.provenance.ingress
        observation = ingress.observation
        constitution = source.constitution
        facts = {
            fact.port: fact
            for fact in (
                () if observation is None else observation.deposited_peptide_boundaries
            )
            if (selected is None or fact.port in selected)
            and fact.identifies_polymer_in(constitution)
        }
        gaps = {
            port: fact.component_id
            for port, fact in facts.items()
            if fact.kind is PeptideSequenceBoundary.MISSING_CONTINUATION
        }
        attached = {
            attachment
            for attachment, _ in PeptideCapAttachment.bonded_ports(
                constitution,
                source.geometry,
                source.topology,
                adjacent=ingress.pdb_adjacent_cap_links,
                uncertain=ingress.uncertain_peptide_link_ports or frozenset(),
                observation=observation,
            )
        }
        known = attached.union(source.provenance.omitted_peptide_caps)
        occupied = {
            a.port: a.component_id
            for a in known
            if (selected is None or a.port in selected)
            and a.identifies_port_in(constitution)
        }
        pending: dict[str, list[AtomRef]] = defaultdict(list)
        for port in sorted(gaps.keys() - occupied.keys()):
            pending[port.residue_id.chain_id].append(port)
        used: dict[str, set[int]] = defaultdict(set)
        for rid in graft_ids:
            used[rid.chain_id].add(rid.seq_num)
        for row in constitution.residue_slots:
            used[row.residue_id.chain_id].add(row.residue_id.seq_num)
        for a in known:
            used[a.cap_id.chain_id].add(a.cap_id.seq_num)

        addresses: dict[AtomRef, ResidueId] = {}
        for path in ingress.deposited_peptide_sequences:
            if path.chain_id not in pending or path.unavailable_cap_numbers is None:
                continue
            excluded = used[path.chain_id] | path.unavailable_cap_numbers
            number = path.first_cap_number
            for port in pending[path.chain_id]:
                while number in excluded:
                    number += 1
                addresses[port] = ResidueId(path.chain_id, number)
                number += 1
        ports = (
            set(selected)
            if selected is not None
            else (set(facts) | set(occupied) | _exposed_ports(source))
        )
        return cls(
            source,
            MappingProxyType(gaps | occupied),
            MappingProxyType(addresses),
            tuple(sorted(ports)),
            MappingProxyType(facts),
            MappingProxyType({a.port: a.cap_id for a in attached if a.port in ports}),
            frozenset(gaps.keys() & qualified_gap_ports),
        )

    def reserves(self, structure: ProteinStructure, port: AtomRef) -> bool:
        """Rebind this end by source, polymer role and component on the current branch."""
        name = self.components.get(port)
        if name is None:
            return False
        origin = self.source.provenance.ingress
        current = structure.provenance.ingress
        if current is not origin and (
            origin.observation is None or current.observation is not origin.observation
        ):
            return False
        constitution = structure.constitution
        site = constitution.residue_or_ligand(port.residue_id)
        return (
            site is not None
            and site.component_id == name
            and constitution.role_at(constitution.residue_index(port.residue_id))
            is ResidueRole.POLYMER
        )


def _exposed_ports(source: ProteinStructure) -> set[AtomRef]:
    c = source.constitution
    incident = source.topology.incident_bonds_by_atom()
    result = set()
    for chain in c.chains:
        for site in chain.residues:
            for name, opposite in (("N", "C"), ("C", "N")):
                port = AtomRef(site.residue_id, name)
                index = c.resolve_atom_index(port)
                if index is not None and any(
                    edge.relationship_type is BondRelationshipType.COVALENT
                    and edge.order in (None, 1)
                    and not edge.aromatic
                    and other.residue_id != site.residue_id
                    and other.atom_name == opposite
                    and c.role_at(c.residue_index(other.residue_id))
                    is ResidueRole.POLYMER
                    for edge in incident.get(index, ())
                    for other in (
                        c.atom_ref_at(j) for j in edge.endpoint_pair() if j != index
                    )
                ):
                    continue
                result.add(port)
    return result


_ACTIVE: ContextVar[BoundPeptidePorts | None] = ContextVar(
    "bound_peptide_ports", default=None
)


def active_peptide_ports() -> BoundPeptidePorts | None:
    """Return the reservations of the current execution, if any."""
    return _ACTIVE.get()


@contextmanager
def peptide_port_run(binding: BoundPeptidePorts | None) -> Iterator[None]:
    """Install input-bound reservations for one workflow or deferred assessment."""
    token = _ACTIVE.set(binding)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
