"""Project source polymer succession into canonical peptide connectivity."""

from collections import defaultdict
from collections.abc import Mapping

import gemmi

from protrepair.chemistry.microstate.graph import MicrostateAtom
from protrepair.errors import StructureNormalizationError
from protrepair.io.source_identity import normalize_chain_id, normalize_insertion_code
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.peptide import PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
    is_covalent_like_relationship,
)


def source_sequence_inferred_polymer_topology_bonds(
    source: gemmi.Structure,
    *,
    file_format: FileFormat,
    constitution: StructureConstitution,
    geometry: StructureGeometry,
    peptide_break_after: frozenset[ResidueId],
) -> tuple[TopologyBond, ...]:
    """Resolve source-supported peptide succession without renumbering residues.

    Parameters
    ----------
    source : gemmi.Structure
        Original parsed structure, before inventing entities or sequence labels.
    file_format : FileFormat
        PDB SEQRES entities use chain names when no subchain was assigned.
    constitution : StructureConstitution
        Selected canonical residue and atom identities.
    geometry : StructureGeometry
        Selected coordinates used to corroborate unusual author numbering.
    peptide_break_after : frozenset[ResidueId]
        Explicit source TER boundaries.

    Returns
    -------
    tuple[TopologyBond, ...]
        Sequence-inferred candidates; surviving explicit connections must still
        be checked with ``source_compatible_peptide_bonds``.

    Raises
    ------
    StructureNormalizationError
        A source residue lacks an author sequence number.
    """

    bonds: list[TopologyBond] = []
    chains_by_id = {chain.chain_id: chain for chain in constitution.chains}
    sequences = {
        subchain: tuple(entity.full_sequence)
        for entity in source.entities
        if entity.entity_type is gemmi.EntityType.Polymer
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
        source_offsets = {
            residue_id: offset for offset, residue_id in enumerate(raw_slots)
        }
        source_slots = tuple(selected)
        pairs = dict.fromkeys(zip(source_slots, source_slots[1:], strict=False))
        atom_slots = tuple(
            residue_id for residue_id, raw in selected.items() if raw.het_flag != "H"
        )
        # Reconcile relocated HET records only when ordinary source order supports
        # author ordering. Canonical sorting must not reverse an insertion run.
        if all(
            first < second
            for first, second in zip(atom_slots, atom_slots[1:], strict=False)
        ):
            slots = tuple(site.residue_id for site in chain.residues)
            for first_id, second_id in zip(slots, slots[1:], strict=False):
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

            first_index = constitution.residue_index(first_id)
            second_index = constitution.residue_index(second_id)
            first_site = constitution.residue_site_at(first_index)
            second_site = constitution.residue_site_at(second_index)
            if (
                not first_site.has_atom_site("C")
                or first_site.has_atom_site("OXT")
                or not second_site.has_atom_site("N")
            ):
                continue
            carbon = constitution.atom_index_in_residue(first_index, "C")
            nitrogen = constitution.atom_index_in_residue(second_index, "N")
            if not first_id.immediately_precedes(second_id):
                if source_offsets[second_id] != source_offsets[first_id] + 1:
                    continue
                separation = geometry.atom_geometry(carbon).position.distance_to(
                    geometry.atom_geometry(nitrogen).position
                )
                if not 0 < separation <= PEPTIDE_CN_DISTANCE_MAX_ANGSTROM:
                    continue

            bonds.append(
                TopologyBond(
                    carbon, nitrogen, provenance=BondProvenance.SEQUENCE_INFERRED
                )
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
