"""Retain port-local ambiguities that native connection decoding cannot express."""

import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import gemmi

from protrepair.io.source_identity import normalize_chain_id
from protrepair.io.source_peptide import SourcePolymerLabels
from protrepair.structure.labels import AtomRef, ResidueId


@dataclass(frozen=True, slots=True)
class _Partner:
    ids: frozenset[ResidueId]
    atom: str
    component: str
    image: str

    def implicated_atoms(self, *, include_sidechains: bool) -> Iterator[AtomRef]:
        for rid in self.ids:
            for name in ("N", "C") if not self.atom else (self.atom,):
                if include_sidechains or name in ("N", "C"):
                    yield AtomRef(rid, name)

    def uncertain_with(self, other: "_Partner") -> bool:
        if any(len(p.ids) != 1 or not p.atom or not p.component for p in (self, other)):
            return True
        if self.ids == other.ids and self.atom == other.atom:
            return True
        images = self.image or "1555", other.image or "1555"
        return any(
            re.fullmatch(r"[1-9][0-9]*_?[0-9]{3}", image) is None for image in images
        ) or images[0].replace("_", "") != images[1].replace("_", "")


def _rid(chain: str, seq: str, ins: str) -> ResidueId | None:
    try:
        return ResidueId(normalize_chain_id(chain), int(seq), ins.strip() or None)
    except ValueError:
        return None


def pdb_uncertain_peptide_ports(
    contents: str, *, include_sidechains: bool = False
) -> frozenset[AtomRef]:
    """Find ambiguous LINK endpoints, restricting them to N/C unless whole slots matter."""
    ports: set[AtomRef] = set()
    for line in contents.splitlines():
        if line[:6] not in ("LINK  ", "LINKR "):
            continue
        ends = []
        for offset, sym in ((12, 59), (42, 66)):
            rid = _rid(
                line[offset + 9 : offset + 10],
                line[offset + 10 : offset + 14],
                line[offset + 14 : offset + 15],
            )
            ends.append(
                _Partner(
                    frozenset(() if rid is None else (rid,)),
                    line[offset : offset + 4].strip(),
                    line[offset + 5 : offset + 8].strip(),
                    line[sym : sym + 6].strip(),
                )
            )
        if ends[0].uncertain_with(ends[1]):
            ports.update(
                ref
                for end in ends
                for ref in end.implicated_atoms(include_sidechains=include_sidechains)
            )
    return frozenset(ports)


def cif_uncertain_peptide_ports(
    block: gemmi.cif.Block,
    labels: Mapping[ResidueId, frozenset[SourcePolymerLabels]],
    *,
    include_sidechains: bool = False,
) -> frozenset[AtomRef]:
    """Keep identifiable endpoints of ambiguous struct_conn rows; normally only N/C."""
    table = block.find_mmcif_category("_struct_conn.")
    if not table:
        return frozenset()
    by_label: dict[tuple[str, str], set[ResidueId]] = defaultdict(set)
    for rid, names in labels.items():
        for item in names:
            by_label[item.subchain_id, item.position_token].add(rid)
    keys = tuple(tag.partition(".")[2] for tag in table.tags)
    ports: set[AtomRef] = set()
    for raw in table:
        row = {
            key: (
                "?"
                if token == "?" and key.endswith("_symmetry")
                else gemmi.cif.as_string(token)
            )
            for key, token in zip(keys, raw, strict=True)
        }
        if row.get("conn_type_id", "").lower() == "hydrog":
            continue
        ends = []
        for p in ("ptnr1_", "ptnr2_"):
            src = (
                _rid(
                    row[p + "auth_asym_id"],
                    row[p + "auth_seq_id"],
                    row.get("pdbx_" + p + "PDB_ins_code", ""),
                )
                if row.get(p + "auth_asym_id") and row.get(p + "auth_seq_id")
                else None
            )
            key = (row.get(p + "label_asym_id", ""), row.get(p + "label_seq_id", ""))
            ids = set(by_label.get(key, ())) if key[1] or src is None else set()
            if src is not None:
                ids.add(src)
            atom = row.get(p + "auth_atom_id") or row.get(p + "label_atom_id", "")
            component = row.get(p + "auth_comp_id") or row.get(p + "label_comp_id", "")
            ends.append(
                _Partner(frozenset(ids), atom, component, row.get(p + "symmetry", ""))
            )
        if ends[0].uncertain_with(ends[1]):
            ports.update(
                ref
                for end in ends
                for ref in end.implicated_atoms(include_sidechains=include_sidechains)
            )
    return frozenset(ports)
