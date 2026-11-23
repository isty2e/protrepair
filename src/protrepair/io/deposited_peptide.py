"""Read a construct's peptide boundaries before coordinate or role selection."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import accumulate, pairwise

import gemmi

from protrepair.io.source_identity import normalize_chain_id
from protrepair.io.source_peptide import SourcePolymerLabels
from protrepair.relation.sequence_embedding import SequenceEmbedding
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.peptide import PEPTIDE_CAP_COMPONENT_IDS
from protrepair.structure.sequence_boundary import (
    DepositedPeptideBoundary,
    PeptideSequenceBoundary,
)

_AMBIGUOUS = frozenset(("", "UNK", "ASX", "GLX", "XLE", "XAA", ".", "?"))
_PEPTIDES = frozenset(("polypeptide(L)", "polypeptide(D)"))


@dataclass(slots=True)
class PdbSequenceSlot:
    """One run of PDB coordinate rows before applying any retention policy."""

    residue_id: ResidueId
    component_id: str
    segment: str
    hetero: bool
    backbone_names: set[str] = field(default_factory=set, init=False, compare=False)


@dataclass(frozen=True, slots=True)
class MmcifSequenceSlot:
    """One first-model residue identity with its deposited sequence labels."""

    residue_id: ResidueId
    component_id: str
    label_component_id: str
    labels: SourcePolymerLabels


def pdb_seqres_sequences(contents: str) -> dict[str, tuple[str, ...]]:
    """Read complete SEQRES chains without trusting a partial continuation.

    Parameters
    ----------
    contents : str
        Original PDB records.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Chains with consecutive SEQRES rows and the stated number of monomers.
    """
    if "SEQRES" not in contents:
        return {}
    chains: dict[str, list[str]] = defaultdict(list)
    headers: dict[str, tuple[int, int]] = {}
    invalid: set[str] = set()
    for line in contents.splitlines():
        if not line.startswith("SEQRES"):
            continue
        chain = normalize_chain_id(line[10:12])
        names = line[19:70].split()
        try:
            number, length = int(line[7:10]), int(line[13:17])
        except ValueError:
            invalid.add(chain)
            continue
        prior, expected = headers.get(chain, (0, length))
        if number != prior + 1 or length != expected or not 1 <= len(names) <= 13:
            invalid.add(chain)
        headers[chain] = number, length
        chains[chain].extend(names)

    return {
        chain: tuple(sequence)
        for chain, sequence in chains.items()
        if chain not in invalid
        and len(sequence) == headers[chain][1]
        and _AMBIGUOUS.isdisjoint(sequence)
        and _peptide_components(frozenset(sequence), declared_polymer_type=False)
    }


def _peptide_components(
    components: frozenset[str], *, declared_polymer_type: bool
) -> bool:
    if not components.isdisjoint(PEPTIDE_CAP_COMPONENT_IDS):
        return False
    known = tuple(gemmi.find_tabulated_residue(name) for name in components)
    return (
        declared_polymer_type or any(info.is_amino_acid() for info in known)
    ) and all(not info.found() or info.is_amino_acid() for info in known)


def pdb_peptide_boundaries(
    sequences: dict[str, tuple[str, ...]],
    slots: Iterable[PdbSequenceSlot],
    break_after: frozenset[ResidueId],
) -> tuple[DepositedPeptideBoundary, ...]:
    """Find ends and compulsory gaps using all plausible first-model monomers.

    Parameters
    ----------
    sequences : dict[str, tuple[str, ...]]
        Complete SEQRES sequences by author chain.
    slots : Iterable[PdbSequenceSlot]
        Original record runs, including non-polymer and unsupported components.
    break_after : frozenset[ResidueId]
        Original TER boundaries.

    Returns
    -------
    tuple[DepositedPeptideBoundary, ...]
        Positive sequence facts, independent of current occupancy or bonds.
    """
    chains: dict[str, list[PdbSequenceSlot]] = defaultdict(list)
    for slot in slots:
        if slot.residue_id.chain_id in sequences:
            chains[slot.residue_id.chain_id].append(slot)
    result: list[DepositedPeptideBoundary] = []
    for chain, rows in chains.items():
        declared = sequences[chain]
        components = frozenset(declared)
        observed: list[tuple[ResidueId, str]] = []
        terminated = False
        segments: set[str] = set()
        trailing_monomers: set[str] = set()
        seen: set[ResidueId] = set()
        for slot in rows:
            name = slot.component_id
            info = gemmi.find_tabulated_residue(name)
            if not info.is_water():
                if slot.residue_id in seen:
                    break
                seen.add(slot.residue_id)
                if terminated:
                    if not slot.hetero:
                        break
                    if name in components:
                        trailing_monomers.add(name)
                    elif info.is_amino_acid() or slot.backbone_names == {
                        "N",
                        "CA",
                        "C",
                    }:
                        break
                else:
                    observed.append((slot.residue_id, name))
                    segments.add(slot.segment)
            terminated |= slot.residue_id in break_after
        else:
            if len(segments) != 1:
                continue
            embedding = SequenceEmbedding.match(tuple(n for _, n in observed), declared)
            if embedding is not None:
                # A matching HETATM after TER may occupy a declared slot. Ignore
                # it only for intervals in which its component cannot occur.
                possible = (
                    tuple(
                        accumulate(
                            (name in trailing_monomers for name in declared), initial=0
                        )
                    )
                    if trailing_monomers
                    else ()
                )
                result.extend(
                    _boundaries(chain, len(declared), observed, embedding, possible)
                )
    return tuple(result)


def _boundaries(
    instance: str,
    length: int,
    observed: list[tuple[ResidueId, str]],
    embedding: SequenceEmbedding,
    possible_occupants_prefix: tuple[int, ...] = (),
) -> Iterator[DepositedPeptideBoundary]:
    def clear_between(left: int, right: int) -> bool:
        return (
            not possible_occupants_prefix
            or possible_occupants_prefix[left] == possible_occupants_prefix[right - 1]
        )

    def fact(
        i: int, side: str, kind: PeptideSequenceBoundary
    ) -> DepositedPeptideBoundary:
        rid, component = observed[i]
        return DepositedPeptideBoundary(
            AtomRef(rid, side),
            component,
            instance,
            length,
            embedding.first[i],
            embedding.last[i],
            kind,
        )

    end = PeptideSequenceBoundary.CONSTRUCT_END
    gap = PeptideSequenceBoundary.MISSING_CONTINUATION
    if embedding.first[0] > 1 and clear_between(0, embedding.last[0]):
        yield fact(0, "N", gap)
    elif embedding.last[0] == 1:
        yield fact(0, "N", end)
    for i in embedding.gaps_after:
        if clear_between(embedding.first[i], embedding.last[i + 1]):
            yield fact(i, "C", gap)
            yield fact(i + 1, "N", gap)
    if embedding.last[-1] < length and clear_between(embedding.first[-1], length + 1):
        yield fact(-1, "C", gap)
    elif embedding.first[-1] == length:
        yield fact(-1, "C", end)


def _rows(
    block: gemmi.cif.Block, category: str, columns: list[str]
) -> Iterator[tuple[str, ...]]:
    for row in block.find(category, columns):
        yield tuple(gemmi.cif.as_string(value).strip() for value in row)


def _unique_column(rows: Iterable[tuple[str, ...]]) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for key, value in rows:
        by_id[key] = value if key not in by_id else ""
    return by_id


def _mmcif_sequences(
    block: gemmi.cif.Block,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]], set[str]]:
    instances = _unique_column(_rows(block, "_struct_asym.", ["id", "entity_id"]))
    entities = _unique_column(_rows(block, "_entity.", ["id", "type"]))
    polymer_types = _unique_column(_rows(block, "_entity_poly.", ["entity_id", "type"]))
    peptides = {
        eid
        for eid, kind in polymer_types.items()
        if kind in _PEPTIDES and entities.get(eid) == "polymer"
    }
    slots: dict[str, dict[int, str]] = defaultdict(dict)
    invalid: set[str] = set()
    for eid, token, component in _rows(
        block, "_entity_poly_seq.", ["entity_id", "num", "mon_id"]
    ):
        if eid not in peptides:
            continue
        try:
            position = int(token)
        except ValueError:
            invalid.add(eid)
            continue
        if position in slots[eid] or component in _AMBIGUOUS:
            invalid.add(eid)
        slots[eid][position] = component
    sequences = {
        eid: tuple(positions[i] for i in range(1, len(positions) + 1))
        for eid, positions in slots.items()
        if eid not in invalid
        and min(positions) == 1
        and max(positions) == len(positions)
        and _peptide_components(
            frozenset(positions.values()), declared_polymer_type=True
        )
    }
    nonpeptide = {
        asym
        for asym, eid in instances.items()
        if entities.get(eid) in {"non-polymer", "water", "branched"}
        or (
            entities.get(eid) == "polymer"
            and polymer_types.get(eid)
            and eid not in peptides
        )
    }
    return instances, sequences, nonpeptide


def mmcif_peptide_boundaries(
    block: gemmi.cif.Block,
    slots: Iterable[MmcifSequenceSlot],
    *,
    unaddressable_chains: frozenset[str],
) -> tuple[DepositedPeptideBoundary, ...]:
    """Bind original atom labels to contiguous declared polypeptide sequences.

    Parameters
    ----------
    block : gemmi.cif.Block
        Original coordinate block, before entity setup and filtering.
    slots : Iterable[MmcifSequenceSlot]
        Residue runs from the original first-model atom annotation pass.
    unaddressable_chains : frozenset[str]
        Author chains with an uninterpretable atom address or model declaration.

    Returns
    -------
    tuple[DepositedPeptideBoundary, ...]
        Positive facts for instances with consistent row labels and addresses.
    """
    instances, sequences, nonpeptide = _mmcif_sequences(block)
    if not sequences:
        return ()
    sites: dict[str, dict[ResidueId, tuple[int, str]]] = defaultdict(dict)
    owners: dict[ResidueId, str] = {}
    invalid: set[str] = set()
    uncertain_chains = set(unaddressable_chains)
    polymer_instance_by_chain: dict[str, str] = {}
    last_residue: dict[str, ResidueId] = {}
    for row in slots:
        rid = row.residue_id
        chain = rid.chain_id
        asym, eid = row.labels.subchain_id, row.labels.entity_id
        if (
            asym not in nonpeptide
            and polymer_instance_by_chain.setdefault(chain, asym) != asym
        ):
            uncertain_chains.add(chain)
        previous_asym = owners.setdefault(rid, asym)
        if previous_asym != asym:
            invalid.update((previous_asym, asym))
        if instances.get(asym) != eid or not (asym and eid):
            invalid.add(asym)
            uncertain_chains.add(chain)
            continue
        if asym in nonpeptide:
            continue
        sequence = sequences.get(eid)
        if sequence is None:
            continue
        try:
            position = int(row.labels.position_token)
        except ValueError:
            invalid.add(asym)
            continue
        name = row.label_component_id
        slot = (position, name)
        if (
            not 1 <= position <= len(sequence)
            or sequence[position - 1] != name
            or row.component_id != name
            or (rid in sites[asym] and last_residue.get(asym) != rid)
            or sites[asym].setdefault(rid, slot) != slot
        ):
            invalid.add(asym)
        last_residue[asym] = rid
    result: list[DepositedPeptideBoundary] = []
    for asym, residues in sites.items():
        chains = {rid.chain_id for rid in residues}
        if (
            asym in invalid
            or len(chains) != 1
            or not chains.isdisjoint(uncertain_chains)
        ):
            continue
        positions = tuple(pos for pos, _ in residues.values())
        if any(b <= a for a, b in pairwise(positions)):
            continue
        result.extend(
            _boundaries(
                asym,
                len(sequences[instances[asym]]),
                [(rid, name) for rid, (_, name) in residues.items()],
                SequenceEmbedding.labelled(positions),
            )
        )
    return tuple(result)
