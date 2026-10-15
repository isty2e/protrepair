"""Bond lookup parity across canonical graph construction and remapping."""

import pickle
from dataclasses import replace
from itertools import combinations, product
from random import Random

import pytest

from protrepair.structure.constitution import (
    AtomSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)


def _constitution(atom_names: tuple[str, ...]) -> StructureConstitution:
    return StructureConstitution(
        chains=(),
        ligands=(
            ResidueSite(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atom_sites=tuple(AtomSite(name, "C") for name in atom_names),
                is_hetero=True,
            ),
        ),
    )


def _linear_lookup(
    bonds: tuple[TopologyBond, ...], left: AtomIndex, right: AtomIndex
) -> TopologyBond | None:
    return next(
        (
            bond
            for bond in bonds
            if (bond.atom_index_1 == left and bond.atom_index_2 == right)
            or (bond.atom_index_1 == right and bond.atom_index_2 == left)
        ),
        None,
    )


@pytest.mark.parametrize("seed", (0, 7, 91))
@pytest.mark.parametrize("bond_count", (0, 1, 2, 31, 100))
def test_all_pair_queries_match_linear_reference(seed: int, bond_count: int) -> None:
    constitution = _constitution(tuple(f"C{i}" for i in range(24)))
    pairs = list(combinations(range(24), 2))
    Random(seed).shuffle(pairs)
    relationships = tuple(BondRelationshipType)
    provenances = tuple(BondProvenance)
    bonds = tuple(
        TopologyBond(
            AtomIndex(right),
            AtomIndex(left),
            order=(None, 1, 2, 3)[i % 4],
            aromatic=i % 7 == 0,
            relationship_type=relationships[i % len(relationships)],
            provenance=provenances[(i // len(relationships)) % len(provenances)],
            source_metadata=(
                SourceBondMetadata(
                    record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                    source_id=f"source-{i}",
                    reported_distance_angstrom=1.5,
                    reported_order=2,
                    reported_relationship_type=BondRelationshipType.UNKNOWN,
                )
                if provenances[(i // len(relationships)) % len(provenances)]
                is BondProvenance.SOURCE_EXPLICIT
                else None
            ),
        )
        for i, (left, right) in enumerate(pairs[:bond_count])
    )
    supplied_bonds = (*bonds, replace(bonds[0])) if bonds else bonds
    topology = StructureTopology(
        constitution=constitution,
        atom_topologies=(None,) * len(constitution.atom_slots),
        bonds=supplied_bonds,
    )

    indices = (*(AtomIndex(i) for i in range(27)), AtomIndex(10**20))
    for left, right in product(indices, repeat=2):
        assert topology.bond_between(left, right) is _linear_lookup(bonds, left, right)


@pytest.mark.parametrize(
    "target_names",
    (
        ("E", "C", "A", "D", "B"),
        ("E", "A", "C"),
        ("C", "E"),
        ("Z", "E", "A", "C"),
    ),
)
def test_lookup_after_atom_reordering_and_removal(
    target_names: tuple[str, ...],
) -> None:
    source = _constitution(("A", "B", "C", "D", "E"))
    topology = StructureTopology(
        constitution=source,
        atom_topologies=(None,) * len(source.atom_slots),
        bonds=tuple(
            TopologyBond(
                AtomIndex(left),
                AtomIndex(right),
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_CONECT,
                    source_id=f"{left}-{right}",
                ),
            )
            for left, right in ((0, 1), (0, 4), (1, 3), (2, 4))
        ),
    )
    target = _constitution(target_names)
    remapped = StructureTopology(
        constitution=target,
        atom_topologies=(None,) * len(target.atom_slots),
        bonds=topology.bonds_for_constitution(
            source_constitution=source, target_constitution=target
        ),
    )

    for left_name, right_name in product(target_names, repeat=2):
        left_ref = AtomRef(ResidueId("L", 1), left_name)
        right_ref = AtomRef(ResidueId("L", 1), right_name)
        source_left = source.resolve_atom_index(left_ref)
        source_right = source.resolve_atom_index(right_ref)
        previous = (
            None
            if source_left is None or source_right is None
            else _linear_lookup(topology.bonds, source_left, source_right)
        )
        left = target.atom_index(left_ref)
        right = target.atom_index(right_ref)
        expected = (
            None
            if previous is None
            else replace(previous, atom_index_1=left, atom_index_2=right)
        )
        assert remapped.bond_between(left, right) == expected


def test_lookup_after_pickle_roundtrip() -> None:
    constitution = _constitution(("A", "B", "C", "D", "E"))
    topology = StructureTopology(
        constitution=constitution,
        atom_topologies=(None,) * len(constitution.atom_slots),
        bonds=(
            TopologyBond(AtomIndex(3), AtomIndex(4)),
            TopologyBond(
                AtomIndex(0),
                AtomIndex(4),
                order=None,
                relationship_type=BondRelationshipType.UNKNOWN,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_CONECT,
                    source_id="unknown",
                ),
            ),
            TopologyBond(AtomIndex(0), AtomIndex(2)),
        ),
    )
    restored = pickle.loads(pickle.dumps(topology))

    indices = tuple(AtomIndex(i) for i in range(7))
    for left, right in product(indices, repeat=2):
        assert restored.bond_between(left, right) == _linear_lookup(
            topology.bonds, left, right
        )
