"""Restore verified biological sequence for an exported, attached peptide cap."""

from collections.abc import Iterator
from dataclasses import dataclass

import gemmi

from protrepair.errors import StructureSerializationError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.cap_attachment import PeptideCapAttachment
from protrepair.structure.constitution import ChainSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.sequence_boundary import DepositedPeptideSequence
from protrepair.structure.topology import BondProvenance


def _exportable_positions(
    path: DepositedPeptideSequence,
    chain: ChainSite,
    structure: ProteinStructure,
    fmt: FileFormat,
) -> tuple[tuple[int, int], ...]:
    original = {rid: (lo, hi) for rid, lo, hi in path.positions}
    current = {r.residue_id: r.component_id for r in chain.residues}
    if any(
        current.get(rid) != path.components[lo - 1] for rid, (lo, _) in original.items()
    ):
        raise StructureSerializationError(
            "cannot preserve deposited sequence after biological slots were removed or replaced"
        )
    n = len(path.components)

    def images(reverse: bool) -> Iterator[int]:
        cursor = n if reverse else 1
        for residue in reversed(chain.residues) if reverse else chain.residues:
            lo, hi = original.get(residue.residue_id, (1, n))
            cursor = min(hi, cursor) if reverse else max(lo, cursor)
            step = -1 if reverse else 1
            while (
                lo <= cursor <= hi
                and path.components[cursor - 1] != residue.component_id
            ):
                cursor += step
            if not lo <= cursor <= hi:
                raise StructureSerializationError(
                    "polymer coordinates contradict the deposited cap-flank sequence"
                )
            yield cursor
            cursor += step

    bounds = tuple(zip(images(False), reversed(tuple(images(True))), strict=True))
    if fmt is FileFormat.MMCIF and path.scheme is None:
        raise StructureSerializationError(
            "the deposited residue address scheme contradicts its peptide sequence"
        )
    if fmt is FileFormat.MMCIF and (
        path.trailing_monomers or any(lo != hi for lo, hi in bounds)
    ):
        raise StructureSerializationError(
            "cannot assign unique mmCIF sequence labels for this capped PDB chain"
        )
    if fmt is FileFormat.PDB and (
        len(path.components) > 9999
        or any(len(name) > 3 or not name.isascii() for name in path.components)
    ):
        raise StructureSerializationError(
            "deposited peptide sequence cannot fit PDB SEQRES; use mmCIF"
        )
    for rid, name in path.trailing_monomers:
        site = structure.constitution.residue_or_ligand(rid)
        if site is None or site.component_id != name:
            raise StructureSerializationError(
                "a possible deposited sequence occupant is absent from the capped output"
            )
    return bounds


@dataclass(frozen=True, slots=True)
class CapSequenceProjection:
    """One exported polymer subchain, its entity, and its original address map."""

    asym: str
    entity: str
    path: DepositedPeptideSequence

    @classmethod
    def apply(
        cls, raw: gemmi.Structure, structure: ProteinStructure, fmt: FileFormat
    ) -> tuple["CapSequenceProjection", ...]:
        """Keep verified biological sequences when exporting an attached cap."""
        ingress = structure.provenance.ingress
        if not ingress.deposited_peptide_sequences:
            return ()
        attachments = dict(
            PeptideCapAttachment.bonded_ports(
                structure.constitution,
                structure.geometry,
                structure.topology,
                adjacent=ingress.pdb_adjacent_cap_links,
                uncertain=ingress.uncertain_peptide_link_ports or frozenset(),
                observation=ingress.observation,
            )
        )
        if not attachments:
            return ()
        generated = {
            a.port.residue_id.chain_id
            for a, b in attachments.items()
            if b.provenance is BondProvenance.REPAIR_INFERRED
        }
        paths = {p.chain_id: p for p in ingress.deposited_peptide_sequences}
        result = []
        for chain in raw[0]:
            path = paths.get(chain.name)
            if path is None or not structure.constitution.has_chain(chain.name):
                continue
            current = structure.constitution.chain(path.chain_id)
            try:
                bounds = _exportable_positions(path, current, structure, fmt)
            except StructureSerializationError:
                if chain.name in generated:
                    raise
                # A deposited cap alone cannot require new sequence assertions.
                # Leave the ordinary writer's entity unchanged for this chain.
                continue

            span = chain.get_polymer()
            if len(span) != len(current.residues):
                raise StructureSerializationError(
                    "cap projection changed the biological polymer slots"
                )
            entity = raw.get_entity_of(span)
            if entity is None:
                raise StructureSerializationError(
                    "capped peptide has no polymer entity"
                )
            asym = span.subchain_id()
            if len(entity.subchains) != 1:
                # setup_entities may share a coordinate-derived entity between
                # chains whose deposited constructs actually differ.
                ids = {e.name for e in raw.entities}
                name = asym
                while name in ids:
                    name += "_"
                entity.subchains = [s for s in entity.subchains if s != asym]
                new = gemmi.Entity(name)
                new.entity_type = gemmi.EntityType.Polymer
                new.polymer_type = entity.polymer_type
                new.subchains = [asym]
                raw.entities.append(new)
                entity = raw.get_entity_of(span)
            if path.polymer_type:
                entity.polymer_type = (
                    gemmi.PolymerType.PeptideD
                    if path.polymer_type == "polypeptide(D)"
                    else gemmi.PolymerType.PeptideL
                )
            entity.full_sequence = list(path.components)
            for atom_slot, expected, (lo, hi) in zip(
                span, current.residues, bounds, strict=True
            ):
                number = atom_slot.seqid.num
                if number is None:
                    raise StructureSerializationError(
                        "capped peptide lost its author residue number"
                    )
                rid = ResidueId(
                    chain.name,
                    number,
                    atom_slot.seqid.icode.strip() or None,
                )
                if (
                    rid != expected.residue_id
                    or atom_slot.name != expected.component_id
                ):
                    raise StructureSerializationError(
                        "cap export reordered the biological sequence"
                    )
                atom_slot.label_seq = lo if lo == hi else None
                atom_slot.entity_id = entity.name
            result.append(cls(asym, entity.name, path))
        return tuple(result)

    @staticmethod
    def write_scheme(
        block: gemmi.cif.Block, projections: tuple["CapSequenceProjection", ...]
    ) -> None:
        """Keep deposited author/PDB numbers, including unmodelled positions."""
        rows = [(p, row) for p in projections for row in (p.path.scheme or ())]
        if not rows:
            return
        loop = block.init_loop(
            "_pdbx_poly_seq_scheme.",
            [
                "asym_id",
                "entity_id",
                "seq_id",
                "mon_id",
                "pdb_seq_num",
                "auth_seq_num",
                "pdb_ins_code",
                "pdb_strand_id",
            ],
        )
        for p, (pos, pdb, author, ins) in rows:
            loop.add_row(
                [
                    gemmi.cif.quote(p.asym),
                    gemmi.cif.quote(p.entity),
                    str(pos),
                    gemmi.cif.quote(p.path.components[pos - 1]),
                    *(
                        token
                        if gemmi.cif.is_null(token)
                        else gemmi.cif.quote(gemmi.cif.as_string(token))
                        for token in (pdb, author, ins)
                    ),
                    gemmi.cif.quote(p.path.chain_id),
                ]
            )
