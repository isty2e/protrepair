"""FASPR preparation preserves atom custody while avoiding repeated membership work."""

from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.unit.test_faspr_backend import copy_input_faspr_executable

from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.packing.domain import PackingPlan
from protrepair.transformer.packing.faspr import backend
from protrepair.transformer.packing.spec import PackingScope, PackingSpec


def _source(*, hydrogens: bool, seed: int) -> ProteinStructure:
    random = Random(seed)
    chains = []
    for chain_id, insertions in (("A", ("A", "B")), ("B", ("",))):
        residues = []
        for insertion in insertions:
            names = [
                ("N", "N"),
                ("CA", "C"),
                ("C", "C"),
                ("O", "O"),
                ("CB", "C"),
                ("SG", "S"),
            ]
            if hydrogens:
                names.extend((("H", "H"), ("D1", "D"), ("T1", "T")))
            random.shuffle(names)
            residues.append(
                residue_payload(
                    component_id="CYS",
                    residue_id=ResidueId(chain_id, 10, insertion),
                    atoms=tuple(
                        atom_payload(
                            name,
                            element,
                            Vec3(index, len(residues), len(chains)),
                            occupancy=0.5,
                            b_factor=10.0 + index,
                            altloc="A",
                            formal_charge=1 if name == "N" else None,
                        )
                        for index, (name, element) in enumerate(names)
                    ),
                )
            )
        chains.append(chain_payload(chain_id, tuple(residues)))
    source = build_structure(
        chains=tuple(chains),
        ligands=(
            residue_payload(
                component_id="OBS",
                residue_id=ResidueId("Z", 10),
                atoms=(
                    atom_payload("FE", "Fe", Vec3(9, 0, 0), formal_charge=2),
                    atom_payload("H1", "H", Vec3(10, 0, 0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="faspr-preparation-custody",
        polymer_blueprint=PolymerBlueprint(
            chains=tuple(
                PolymerChainBlueprint(
                    chain_id=chain.chain_id,
                    residue_slots=tuple(
                        PolymerResidueSlot(index + 1, "C")
                        for index in range(len(chain.residues))
                    ),
                )
                for chain, _, _ in chains
            )
        ),
    )
    a = ResidueId("A", 10, "A")
    b = ResidueId("A", 10, "B")
    ligand = ResidueId("Z", 10)
    bonds = []
    for left, right, relationship, provenance in (
        (
            AtomRef(a, "N"),
            AtomRef(a, "CA"),
            BondRelationshipType.COVALENT,
            BondProvenance.TEMPLATE_RESOLVED,
        ),
        (
            AtomRef(a, "C"),
            AtomRef(b, "N"),
            BondRelationshipType.COVALENT,
            BondProvenance.SOURCE_EXPLICIT,
        ),
        (
            AtomRef(a, "SG"),
            AtomRef(b, "SG"),
            BondRelationshipType.DISULFIDE,
            BondProvenance.SEQUENCE_INFERRED,
        ),
        (
            AtomRef(a, "CB"),
            AtomRef(ligand, "FE"),
            BondRelationshipType.METAL_COORDINATION,
            BondProvenance.SOURCE_EXPLICIT,
        ),
        (
            AtomRef(b, "O"),
            AtomRef(ligand, "H1"),
            BondRelationshipType.HYDROGEN_BOND,
            BondProvenance.SOURCE_EXPLICIT,
        ),
        (
            AtomRef(b, "CB"),
            AtomRef(ligand, "FE"),
            BondRelationshipType.UNKNOWN,
            BondProvenance.SOURCE_EXPLICIT,
        ),
        (
            AtomRef(ligand, "FE"),
            AtomRef(ligand, "H1"),
            BondRelationshipType.COVALENT,
            BondProvenance.REPAIR_INFERRED,
        ),
    ):
        bonds.append(
            TopologyBond(
                source.constitution.atom_index(left),
                source.constitution.atom_index(right),
                order=None,
                relationship_type=relationship,
                provenance=provenance,
                source_metadata=(
                    SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_LINK,
                        source_id="observed",
                        reported_distance_angstrom=2.1,
                        reported_relationship_type=relationship,
                    )
                    if provenance is BondProvenance.SOURCE_EXPLICIT
                    else None
                ),
            )
        )
    if hydrogens:
        for residue in source.constitution.chains[0].residues:
            for hydrogen in ("H", "D1", "T1"):
                bonds.append(
                    TopologyBond(
                        source.constitution.atom_index(
                            AtomRef(residue.residue_id, "N")
                        ),
                        source.constitution.atom_index(
                            AtomRef(residue.residue_id, hydrogen)
                        ),
                    )
                )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(bonds),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def _expected(source: ProteinStructure) -> ProteinStructure:
    constitution = StructureConstitution(
        chains=tuple(
            chain.with_residues(
                replace(
                    residue,
                    atom_sites=tuple(
                        atom for atom in residue.atom_sites if not atom.is_hydrogen()
                    ),
                )
                for residue in chain.residues
            )
            for chain in source.constitution.chains
        ),
        ligands=source.constitution.ligands,
    )
    indices_by_ref = {
        constitution.atom_ref_at(AtomIndex(index)): AtomIndex(index)
        for index in range(len(constitution.atom_slots))
    }
    source_indices = tuple(
        source.constitution.atom_index(ref) for ref in indices_by_ref
    )
    bonds = []
    for bond in source.topology.bonds:
        left = source.constitution.atom_ref_at(bond.atom_index_1)
        right = source.constitution.atom_ref_at(bond.atom_index_2)
        if left in indices_by_ref and right in indices_by_ref:
            bonds.append(
                replace(
                    bond,
                    atom_index_1=indices_by_ref[left],
                    atom_index_2=indices_by_ref[right],
                )
            )
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=StructureGeometry(
            constitution=constitution,
            atom_geometries=tuple(
                source.geometry.atom_geometries[index.value] for index in source_indices
            ),
        ),
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=tuple(
                source.topology.atom_topologies[index.value] for index in source_indices
            ),
            bonds=tuple(bonds),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


@pytest.mark.parametrize("seed", (0, 7, 91))
@pytest.mark.parametrize("hydrogens", (False, True))
@pytest.mark.parametrize("scope", tuple(PackingScope))
def test_preparation_preserves_every_surviving_facet(
    seed: int, hydrogens: bool, scope: PackingScope
) -> None:
    source = _source(hydrogens=hydrogens, seed=seed)
    before = repr(source)
    plan = PackingPlan.from_inputs(
        source,
        PackingSpec(
            backend_name="faspr",
            scope=scope,
            mutable_residue_ids=(ResidueId("A", 10, "A"),)
            if scope is PackingScope.LOCAL
            else None,
        ),
    )
    prepared = backend.prepare_structure_for_faspr(plan)
    expected = _expected(source)
    assert prepared.constitution == expected.constitution
    assert prepared.geometry == expected.geometry
    assert prepared.topology == expected.topology
    assert prepared.polymer_blueprint == expected.polymer_blueprint
    assert prepared.provenance == expected.provenance
    assert repr(source) == before


@pytest.mark.parametrize("policy", ("frozen", "sequence", "local_sequence"))
@pytest.mark.parametrize("hydrogens", (False, True))
def test_preparation_keeps_full_polymer_context_for_packing_overrides(
    policy: str, hydrogens: bool
) -> None:
    source = _source(hydrogens=hydrogens, seed=91)
    local = policy == "local_sequence"
    plan = PackingPlan.from_inputs(
        source,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL if local else PackingScope.FULL,
            mutable_residue_ids=(ResidueId("A", 10, "A"),) if local else None,
            frozen_residue_ids=(ResidueId("B", 10),),
            target_sequence=("A" if local else "AAA") if policy != "frozen" else None,
        ),
    )
    execution = backend.FasprExecutionInput.from_plan(plan)
    expected = _expected(source)

    assert execution.structure.constitution == expected.constitution
    assert execution.structure.geometry == expected.geometry
    assert execution.structure.topology == expected.topology
    assert execution.structure.polymer_blueprint is source.polymer_blueprint
    assert execution.structure.provenance is source.provenance
    assert execution.plan.structure is source


class _ObservedIndices(frozenset[AtomIndex]):
    traversals: int = 0

    def __iter__(self) -> Iterator[AtomIndex]:
        self.traversals += 1
        return super().__iter__()


@pytest.mark.parametrize("hydrogens", (False, True))
def test_removal_membership_is_not_rebuilt_for_each_atom(
    monkeypatch: pytest.MonkeyPatch, hydrogens: bool
) -> None:
    source = _source(hydrogens=hydrogens, seed=7)
    plan = PackingPlan.from_inputs(source, PackingSpec(backend_name="faspr"))
    observed = _ObservedIndices()
    calls = 0

    def observe(
        values: Iterable[AtomRef | AtomIndex],
    ) -> frozenset[AtomRef | AtomIndex]:
        nonlocal calls, observed
        calls += 1
        result = frozenset(values)
        if calls == 2:
            assert all(isinstance(value, AtomIndex) for value in result)
            observed = _ObservedIndices(
                value for value in result if isinstance(value, AtomIndex)
            )
            return observed
        return result

    monkeypatch.setattr(backend, "frozenset", observe, raising=False)
    prepared = backend.prepare_structure_for_faspr(plan)
    assert prepared.topology == _expected(source).topology
    assert observed.traversals == 2


def test_packing_preserves_mixed_chain_insertion_isotope_and_retained_chemistry(
    tmp_path: Path,
) -> None:
    source = _source(hydrogens=True, seed=42)
    executable = copy_input_faspr_executable(tmp_path)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    plan = PackingPlan.from_inputs(
        source,
        PackingSpec(
            backend_name="faspr", mutable_residue_ids=(ResidueId("A", 10, "A"),)
        ),
    )
    result = backend.FasprPackingBackend(executable_path=executable).pack(plan)

    assert result.packed_structure.constitution == source.constitution
    assert result.packed_structure.geometry == source.geometry
    assert result.packed_structure.topology == source.topology
    assert result.packed_structure.provenance is source.provenance
    assert result.changed_residue_ids == ()
    assert result.issues == ()
