"""Project source polymer succession into canonical peptide connectivity."""

import itertools
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import gemmi

from protrepair.chemistry.microstate.graph import MicrostateAtom
from protrepair.errors import StructureNormalizationError
from protrepair.io.source_identity import normalize_chain_id, normalize_insertion_code
from protrepair.relation.sequence_alignment import sequence_requires_observed_end
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.peptide import (
    PEPTIDE_CN_DISTANCE_MAX_ANGSTROM,
    peptide_ports_allow_link,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
    is_covalent_like_relationship,
)


@dataclass(frozen=True, slots=True)
class SourcePolymerLabels:
    """Polymer labels declared by a source atom row.

    Parameters
    ----------
    subchain_id, entity_id, position_token : str
        Source label-asym ID, label-entity ID and label-seq token, respectively.
        An empty field carries no claim.
    """

    subchain_id: str
    entity_id: str
    position_token: str

    def matches(self, residue: gemmi.Residue, position: int | None) -> bool:
        """Check the row's available labels against the resolved source slot."""

        if self.subchain_id and self.subchain_id != residue.subchain:
            return False
        if self.entity_id and self.entity_id != residue.entity_id:
            return False
        if not self.position_token:
            return True
        try:
            return int(self.position_token) == position
        except ValueError:
            return False


@dataclass(frozen=True, slots=True)
class SourcePolymerSuccession:
    """An ordered polymer pair before lowering to materialized C/N positions.

    Parameters
    ----------
    preceding_residue_id, following_residue_id : ResidueId
        Selected identities supported by source polymer order.
    consecutive_source_slots : bool
        The two raw slots are immediate neighbors, without a skipped slot.
    verified_source_path : bool
        For coordinate omission, the consecutive path is unique and any
        available declared sequence agrees with the two component identities.
    """

    preceding_residue_id: ResidueId
    following_residue_id: ResidueId
    consecutive_source_slots: bool
    verified_source_path: bool


def source_polymer_paths(
    source: gemmi.Structure,
    *,
    file_format: FileFormat,
    constitution: StructureConstitution,
    peptide_break_after: frozenset[ResidueId],
    verify_source_paths: bool = False,
    source_labels_by_residue: Mapping[ResidueId, frozenset[SourcePolymerLabels]]
    | None = None,
) -> tuple[tuple[SourcePolymerSuccession, ...], frozenset[AtomRef]]:
    """Resolve sequence neighbors independently of their coordinate inventory.

    Parameters
    ----------
    source : gemmi.Structure
        Original parsed structure, before inventing entities or sequence labels.
    file_format : FileFormat
        PDB SEQRES entities use chain names when no subchain was assigned.
    constitution : StructureConstitution
        Selected canonical residue and atom identities.
    peptide_break_after : frozenset[ResidueId]
        Explicit source TER boundaries.
    verify_source_paths : bool, default=False
        Check for unique, directly observed succession and validate any
        declared sequence before the neighbors can authorize omission.
    source_labels_by_residue : Mapping or None, default=None
        Labels from original atom rows, checked for declared ends and omission paths.

    Returns
    -------
    tuple
        Candidate successions and outer ports not certified by a declared sequence.

    Raises
    ------
    StructureNormalizationError
        A source residue lacks an author sequence number.
    """

    successions: list[SourcePolymerSuccession] = []
    unverified_ports: set[AtomRef] = set()
    chains_by_id = {chain.chain_id: chain for chain in constitution.chains}
    source_labels = {} if source_labels_by_residue is None else source_labels_by_residue
    entity_sequences = {
        entity.name: tuple(entity.full_sequence)
        for entity in source.entities
        if entity.entity_type is gemmi.EntityType.Polymer
    }
    sequences = {
        subchain: entity_sequences[entity.name]
        for entity in source.entities
        if entity.name in entity_sequences
        for subchain in entity.subchains
    }
    chain_sequences = (
        {
            normalize_chain_id(entity.name): tuple(entity.full_sequence)
            for entity in source.entities
            if entity.entity_type is gemmi.EntityType.Polymer
        }
        if file_format is FileFormat.PDB
        else {}
    )
    for raw_chain in source[0]:
        chain_id = normalize_chain_id(raw_chain.name)
        chain = chains_by_id.get(chain_id)
        if chain is None:
            continue
        raw_slots: dict[ResidueId, list[gemmi.Residue]] = defaultdict(list)
        for raw in raw_chain:
            raw_slots[_residue_id(chain_id, raw)].append(raw)

        selected: dict[ResidueId, gemmi.Residue] = {}
        for residue_id, variants in raw_slots.items():
            site = constitution.residue_or_ligand(residue_id)
            if site is None or site.is_hetero:
                continue
            matching = [
                raw for raw in variants if raw.name.strip().upper() == site.component_id
            ]
            if len(matching) == 1:
                selected[residue_id] = matching[0]

        positions = _complete_sequence_positions(
            sequences,
            chain_id,
            raw_slots,
            unassigned_sequence=chain_sequences.get(chain_id),
        )
        for residue, port in (
            ((chain.residues[0], "N"), (chain.residues[-1], "C"))
            if chain.residues
            else ()
        ):
            rid = residue.residue_id
            variants = raw_slots.get(rid, ())
            for raw in variants:
                sequence = (
                    sequences.get(raw.subchain)
                    or entity_sequences.get(raw.entity_id)
                    or chain_sequences.get(chain_id)
                )
                position = (
                    raw.label_seq if raw.label_seq is not None else positions.get(rid)
                )
                labels = source_labels.get(rid, ())
                if not sequence:
                    if file_format is FileFormat.MMCIF and (
                        not all(label.matches(raw, position) for label in labels)
                        or (
                            position is not None
                            and (position < 1 or (port == "N" and position != 1))
                        )
                    ):
                        unverified_ports.add(AtomRef(rid, port))
                    continue
                end = 1 if port == "N" else len(sequence)
                if position is None and file_format is FileFormat.PDB:
                    ids = tuple(selected)
                    if (
                        ids == tuple(s.residue_id for s in chain.residues)
                        and all(len(raw_slots[i]) == 1 for i in ids)
                        and sequence_requires_observed_end(
                            tuple(selected[i].name.strip().upper() for i in ids),
                            sequence,
                            n_terminal=port == "N",
                        )
                    ):
                        position = end
                if (
                    len(variants) != 1
                    or position != end
                    or not _sequence_accepts_source_component(
                        sequence, position, residue.component_id
                    )
                    or not all(label.matches(raw, position) for label in labels)
                ):
                    unverified_ports.add(AtomRef(rid, port))
        source_offsets = {
            residue_id: offset for offset, residue_id in enumerate(raw_slots)
        }
        source_slots = tuple(selected)
        pairs = dict.fromkeys(itertools.pairwise(source_slots))
        atom_slots = tuple(
            residue_id for residue_id, raw in selected.items() if raw.het_flag != "H"
        )
        # Reconcile relocated HET records only when ordinary source order supports
        # author ordering. Canonical sorting must not reverse an insertion run.
        if all(first < second for first, second in itertools.pairwise(atom_slots)):
            slots = tuple(site.residue_id for site in chain.residues)
            for first_id, second_id in itertools.pairwise(slots):
                if any(
                    selected.get(residue_id) is not None
                    and selected[residue_id].het_flag == "H"
                    for residue_id in (first_id, second_id)
                ):
                    pairs.setdefault((first_id, second_id), None)

        for first_id, second_id in pairs:
            if first_id in peptide_break_after:
                continue
            first = selected.get(first_id)
            second = selected.get(second_id)
            if first is None or second is None:
                continue
            if first.entity_id != second.entity_id or first.segment != second.segment:
                continue
            # PDB modified residues can be stored as HETATM after the polymer.
            # Their parser-assigned non-polymer subchain is not an entity fact.
            if first.subchain != second.subchain and not (
                not first.entity_id and "H" in {first.het_flag, second.het_flag}
            ):
                continue

            first_position = first.label_seq
            second_position = second.label_seq
            if first_position is None and second_position is None:
                first_position = positions.get(first_id)
                second_position = positions.get(second_id)
            if first_position is not None or second_position is not None:
                if (
                    first_position is None
                    or first_position <= 0
                    or second_position != first_position + 1
                ):
                    continue
            elif not first_id.immediately_precedes(second_id):
                continue

            consecutive = source_offsets[second_id] == source_offsets[first_id] + 1
            verified = False
            if verify_source_paths and consecutive:
                verified = all(
                    len(raw_slots[residue_id]) == 1
                    and all(
                        labels.matches(raw, position)
                        for labels in source_labels.get(residue_id, ())
                    )
                    and _sequence_accepts_source_component(
                        sequences.get(raw.subchain) or chain_sequences.get(chain_id),
                        position,
                        raw.name,
                    )
                    for residue_id, raw, position in (
                        (first_id, first, first_position),
                        (second_id, second, second_position),
                    )
                )
            successions.append(
                SourcePolymerSuccession(first_id, second_id, consecutive, verified)
            )

    return tuple(successions), frozenset(unverified_ports)


def _sequence_accepts_source_component(
    sequence: tuple[str, ...] | None, position: int | None, component_id: str
) -> bool:
    if not sequence:
        return True
    return (
        position is not None
        and 0 < position <= len(sequence)
        and sequence[position - 1].strip().upper() == component_id.strip().upper()
    )


def source_sequence_inferred_polymer_topology_bonds(
    successions: tuple[SourcePolymerSuccession, ...],
    *,
    constitution: StructureConstitution,
    geometry: StructureGeometry,
) -> tuple[TopologyBond, ...]:
    """Lower source sequence suggestions to materialized C-N bond candidates.

    Parameters
    ----------
    successions : tuple of SourcePolymerSuccession
        Ordered source-supported neighbors.
    constitution : StructureConstitution
        The selected residue and atom identities addressed by the neighbors.
    geometry : StructureGeometry
        Selected coordinates used to corroborate unusual author numbering.

    Returns
    -------
    tuple of TopologyBond
        Candidates still subject to checks against explicit source chemistry.
    """

    bonds: list[TopologyBond] = []
    for succession in successions:
        first_id = succession.preceding_residue_id
        second_id = succession.following_residue_id
        first_index = constitution.residue_index(first_id)
        second_index = constitution.residue_index(second_id)
        first_site = constitution.residue_site_at(first_index)
        second_site = constitution.residue_site_at(second_index)
        if (
            not peptide_ports_allow_link(
                first_site.component_id, second_site.component_id
            )
            or not first_site.has_atom_site("C")
            or first_site.has_atom_site("OXT")
            or not second_site.has_atom_site("N")
        ):
            continue
        carbon = constitution.atom_index_in_residue(first_index, "C")
        nitrogen = constitution.atom_index_in_residue(second_index, "N")
        if not first_id.immediately_precedes(second_id):
            if not succession.consecutive_source_slots:
                continue
            separation = geometry.atom_geometry(carbon).position.distance_to(
                geometry.atom_geometry(nitrogen).position
            )
            if not 0 < separation <= PEPTIDE_CN_DISTANCE_MAX_ANGSTROM:
                continue
        bonds.append(
            TopologyBond(carbon, nitrogen, provenance=BondProvenance.SEQUENCE_INFERRED)
        )
    return tuple(bonds)


def source_compatible_peptide_bonds(
    candidates: tuple[TopologyBond, ...],
    *,
    source_bonds: tuple[TopologyBond, ...],
    template_bonds: tuple[TopologyBond, ...],
    constitution: StructureConstitution,
    atom_topologies: tuple[AtomTopology | None, ...],
) -> tuple[TopologyBond, ...]:
    """Remove inferred peptide edges conflicting with surviving source chemistry.

    Parameters
    ----------
    candidates : tuple[TopologyBond, ...]
        Source-sequence-supported peptide candidates.
    source_bonds : tuple[TopologyBond, ...]
        Explicit connections after component and alternate-location selection.
    template_bonds : tuple[TopologyBond, ...]
        Materialized component bonds, before peptide inference.
    constitution : StructureConstitution
        Shared endpoint address space.
    atom_topologies : tuple[AtomTopology or None, ...]
        Slot-aligned original charges; missing charge is not rewritten.

    Returns
    -------
    tuple[TopologyBond, ...]
        Candidates not superseded by an explicit edge or competing attachment.
        Non-covalent contacts do not occupy covalent valence. Known attachments
        bound available valence; this does not assign missing H or a microstate.
    """

    explicit_pairs = {bond.endpoint_pair() for bond in source_bonds}
    template_pairs = {bond.endpoint_pair() for bond in template_bonds}
    endpoints = {index for bond in candidates for index in bond.endpoint_pair()}
    neighbors: dict[AtomIndex, dict[AtomIndex, TopologyBond]] = defaultdict(dict)
    for bond in template_bonds:
        if not is_covalent_like_relationship(bond):
            continue
        first, second = bond.endpoint_pair()
        for index, neighbor in ((first, second), (second, first)):
            if index in endpoints:
                neighbors[index][neighbor] = bond

    blocking_neighbors: dict[AtomIndex, set[AtomIndex]] = defaultdict(set)
    for bond in source_bonds:
        if not (
            is_covalent_like_relationship(bond)
            or bond.relationship_type is BondRelationshipType.UNKNOWN
        ):
            continue
        first, second = bond.endpoint_pair()
        for index, neighbor in ((first, second), (second, first)):
            if index not in endpoints:
                continue
            neighbors[index][neighbor] = bond
            ref = constitution.atom_ref_at(index)
            neighbor_ref = constitution.atom_ref_at(neighbor)
            if (
                ref.residue_id != neighbor_ref.residue_id
                or (ref.atom_name == "C" and neighbor_ref.atom_name not in {"CA", "O"})
                or (
                    ref.atom_name == "N"
                    and not constitution.atom_site_at(neighbor).is_hydrogen()
                    and neighbor_ref.atom_name != "CA"
                    and bond.endpoint_pair() not in template_pairs
                )
            ):
                blocking_neighbors[index].add(neighbor)

    saturated: set[AtomIndex] = set()
    for index in endpoints:
        site = constitution.atom_site_at(index)
        topology = atom_topologies[index.value]
        charge = topology.formal_charge if topology is not None else None
        try:
            capacity = MicrostateAtom(site.name, site.element, charge or 0, 0, 0)
        except ValueError:
            saturated.add(index)
            continue
        # An unspecified source order occupies at least one valence slot. Source
        # declarations have replaced any template order for the same pair.
        bonds = tuple(neighbors[index].values())
        if any(bond.aromatic for bond in bonds) or (
            1 + sum(bond.order or 1 for bond in bonds) > capacity.closed_shell_valence()
        ):
            saturated.add(index)

    return tuple(
        bond
        for bond in candidates
        if bond.endpoint_pair() not in explicit_pairs
        and not saturated.intersection(bond.endpoint_pair())
        and not blocking_neighbors[bond.atom_index_1] - {bond.atom_index_2}
        and not blocking_neighbors[bond.atom_index_2] - {bond.atom_index_1}
    )


def _complete_sequence_positions(
    sequences: Mapping[str, tuple[str, ...]],
    chain_id: str,
    slots: dict[ResidueId, list[gemmi.Residue]],
    *,
    unassigned_sequence: tuple[str, ...] | None,
) -> dict[ResidueId, int]:
    segments: dict[tuple[str, str, str], list[gemmi.Residue]] = defaultdict(list)
    for variants in slots.values():
        for raw in variants:
            segments[(raw.subchain, raw.entity_id, raw.segment)].append(raw)

    positions: dict[ResidueId, int] = {}
    for (subchain, _, _), residues in segments.items():
        sequence = sequences.get(subchain) if subchain else unassigned_sequence
        # Exact full-sequence coverage avoids ambiguous alignments of repeated
        # residues across a genuine missing segment. No gap placement is guessed.
        if not sequence or tuple(raw.name.strip().upper() for raw in residues) != tuple(
            name.strip().upper() for name in sequence
        ):
            continue
        positions.update(
            (_residue_id(chain_id, raw), offset)
            for offset, raw in enumerate(residues, start=1)
        )
    return positions


def _residue_id(chain_id: str, raw: gemmi.Residue) -> ResidueId:
    number = raw.seqid.num
    if number is None:
        raise StructureNormalizationError(
            "source peptide residue lacks a sequence number"
        )
    return ResidueId(chain_id, number, normalize_insertion_code(raw.seqid.icode))
