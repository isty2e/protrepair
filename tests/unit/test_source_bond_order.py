"""Connectivity records preserve order evidence without inventing chemistry."""

from dataclasses import replace
from typing import cast

import gemmi
import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.errors import (
    ModelInvariantError,
    RefinementError,
    StructureNormalizationError,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.source_connection import SourceConnection
from protrepair.io.source_identity import SourceAtomIdentity
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
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
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.continuous.bonds import planned_bond_from_topology_bond
from protrepair.transformer.continuous.rdkit import (
    build_rdkit_molecule,
    rdkit_bond_type,
)
from protrepair.transformer.local import LocalScopeSpec


def _two_atoms(component_id: str = "LIG") -> ProteinStructure:
    residue = residue_payload(
        component_id=component_id,
        residue_id=ResidueId("A", 1),
        atoms=(
            atom_payload("C", "C", Vec3(0.0, 0.0, 0.0)),
            atom_payload("O", "O", Vec3(1.23, 0.0, 0.0)),
        ),
        is_hetero=component_id == "LIG",
    )
    return build_structure(
        chains=() if component_id == "LIG" else (chain_payload("A", (residue,)),),
        ligands=(residue,) if component_id == "LIG" else (),
        source_format=FileFormat.PDB,
    )


def _with_bond(structure: ProteinStructure, bond: TopologyBond) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(bond,),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _conect_input(records: tuple[str, ...], component_id: str = "LIG") -> str:
    return write_structure_string(_two_atoms(component_id), FileFormat.PDB).replace(
        "\nEND", "\n" + "\n".join(records) + "\nEND", 1
    )


@pytest.mark.parametrize("component_id, expected", [("LIG", None), ("ALA", 2)])
def test_connectivity_only_is_not_an_explicit_single_order(
    component_id: str, expected: int | None
) -> None:
    structure = read_structure_string(
        _conect_input(("CONECT    1    2", "CONECT    2    1"), component_id),
        FileFormat.PDB,
    )
    (bond,) = structure.topology.bonds
    assert bond.order == expected
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert bond.source_metadata == SourceBondMetadata(
        SourceBondRecordType.PDB_CONECT, "CONECT"
    )
    assert bond.source_metadata is not None
    assert bond.source_metadata.reported_order is None
    assert bond.relationship_type is (
        BondRelationshipType.UNKNOWN
        if expected is None
        else BondRelationshipType.COVALENT
    )


@pytest.mark.parametrize("order", (2, 3, 4))
def test_conect_multiplicity_survives_reciprocal_and_duplicate_rows(order: int) -> None:
    forward = "CONECT    1" + "    2" * order
    reverse = "CONECT    2" + "    1" * order
    structure = read_structure_string(
        _conect_input((forward, reverse, forward, reverse)), FileFormat.PDB
    )
    assert len(structure.topology.bonds) == 1
    assert structure.topology.bonds[0].order == order
    assert structure.topology.bonds[0].source_metadata == SourceBondMetadata(
        SourceBondRecordType.PDB_CONECT, "CONECT", reported_order=order
    )
    assert (
        read_structure_string(
            write_structure_string(structure, FileFormat.PDB), FileFormat.PDB
        )
        .topology.bonds[0]
        .order
        == order
    )


def test_conflicting_conect_multiplicities_do_not_silently_choose_one() -> None:
    with pytest.raises(ModelInvariantError, match="order"):
        read_structure_string(
            _conect_input(("CONECT    1    2    2", "CONECT    2    1    1    1")),
            FileFormat.PDB,
        )


@pytest.mark.parametrize(
    "order,token", [(1, "sing"), (2, "doub"), (3, "trip"), (4, "quad"), (None, "?")]
)
def test_mmcif_intra_residue_order_roundtrip(order: int | None, token: str) -> None:
    original = _with_bond(
        _two_atoms(),
        TopologyBond(
            AtomIndex(0),
            AtomIndex(1),
            order=order,
            provenance=BondProvenance.REPAIR_INFERRED,
        ),
    )
    text = write_structure_string(original, FileFormat.MMCIF)
    block = gemmi.cif.read_string(text).sole_block()
    assert list(block.find_values("_struct_conn.pdbx_value_order")) == [token]
    restored = read_structure_string(text, FileFormat.MMCIF)
    (bond,) = restored.topology.bonds
    assert bond.order == order
    assert bond.relationship_type is BondRelationshipType.COVALENT
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert bond.source_metadata is not None
    assert bond.source_metadata.reported_order == order


@pytest.mark.parametrize("order", (1, 3))
def test_explicit_mmcif_order_is_not_overwritten_by_default_carbonyl(
    order: int,
) -> None:
    original = _with_bond(
        _two_atoms("ALA"), TopologyBond(AtomIndex(0), AtomIndex(1), order=order)
    )
    restored = read_structure_string(
        write_structure_string(original, FileFormat.MMCIF), FileFormat.MMCIF
    )
    assert restored.topology.bonds[0].order == order


@pytest.mark.parametrize("relationship", tuple(BondRelationshipType))
def test_unresolved_order_cannot_be_projected_as_a_single_ff_bond(
    relationship: BondRelationshipType,
) -> None:
    bond = TopologyBond(
        AtomIndex(0), AtomIndex(1), order=None, relationship_type=relationship
    )
    if relationship in {BondRelationshipType.COVALENT, BondRelationshipType.DISULFIDE}:
        planned = planned_bond_from_topology_bond(bond)
        assert planned is not None
        assert planned.order is None
        with pytest.raises(RefinementError, match="bond order None"):
            rdkit_bond_type(planned)
    else:
        assert planned_bond_from_topology_bond(bond) is None


def test_non_covalent_source_relationship_is_not_promoted_by_template() -> None:
    original = _with_bond(
        _two_atoms("ALA"),
        TopologyBond(
            AtomIndex(0),
            AtomIndex(1),
            order=None,
            relationship_type=BondRelationshipType.METAL_COORDINATION,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                SourceBondRecordType.MMCIF_STRUCT_CONN, "coordination"
            ),
        ),
    )
    restored = read_structure_string(
        write_structure_string(original, FileFormat.MMCIF), FileFormat.MMCIF
    )
    (bond,) = restored.topology.bonds
    assert bond.relationship_type is BondRelationshipType.METAL_COORDINATION
    assert bond.order is None


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_known_carbonyl_graph_survives_repeated_roundtrips(
    file_format: FileFormat,
) -> None:
    structure = read_structure_string(
        write_structure_string(_two_atoms("ALA"), FileFormat.PDB), FileFormat.PDB
    )
    for _ in range(3):
        structure = read_structure_string(
            write_structure_string(structure, file_format), file_format
        )
        (bond,) = structure.topology.bonds
        assert bond.order == 2
        assert bond.relationship_type is BondRelationshipType.COVALENT
        assert not bond.aromatic


def _connection(
    record_type: SourceBondRecordType, order: int | None
) -> SourceConnection:
    return SourceConnection(
        SourceAtomIdentity(AtomRef(ResidueId("A", 1), "C"), "ALA", None),
        SourceAtomIdentity(AtomRef(ResidueId("A", 2), "N"), "ALA", None),
        source_metadata=SourceBondMetadata(
            record_type,
            "connection",
            1.33,
            reported_order=order,
            reported_relationship_type=BondRelationshipType.UNKNOWN
            if record_type is SourceBondRecordType.PDB_CONECT
            else BondRelationshipType.COVALENT,
        ),
    )


@pytest.mark.parametrize(
    "record_type",
    (
        SourceBondRecordType.PDB_LINK,
        SourceBondRecordType.PDB_SSBOND,
        SourceBondRecordType.MMCIF_STRUCT_CONN,
    ),
)
def test_typed_metadata_survives_supplementary_conect_order(
    record_type: SourceBondRecordType,
) -> None:
    typed = _connection(record_type, None)
    fallback = _connection(SourceBondRecordType.PDB_CONECT, 2)
    merged = typed.merge(fallback)
    assert merged == fallback.merge(typed)
    assert merged.source_metadata == replace(typed.source_metadata, reported_order=2)
    assert (
        merged.source_metadata.reported_relationship_type
        == typed.source_metadata.reported_relationship_type
    )
    assert merged.source_metadata.reported_order == 2


@pytest.mark.parametrize("order", (None, 1, 2, 3, 4))
def test_compatible_typed_declarations_merge_order_evidence(order: int | None) -> None:
    unspecified = _connection(SourceBondRecordType.MMCIF_STRUCT_CONN, None)
    explicit = _connection(SourceBondRecordType.MMCIF_STRUCT_CONN, order)
    assert unspecified.merge(explicit) == explicit
    assert explicit.merge(unspecified) == explicit
    assert explicit.merge(explicit) == explicit


def test_compatible_orders_do_not_hide_conflicting_typed_metadata() -> None:
    first = _connection(SourceBondRecordType.MMCIF_STRUCT_CONN, None)
    second = replace(
        first,
        source_metadata=replace(
            first.source_metadata, reported_order=2, reported_distance_angstrom=1.5
        ),
    )
    for left, right in ((first, second), (second, first)):
        with pytest.raises(ModelInvariantError, match="conflicting bonds.*typed"):
            left.merge(right)


def test_explicit_orders_conflict_even_when_one_record_has_lower_type_priority() -> (
    None
):
    typed = _connection(SourceBondRecordType.MMCIF_STRUCT_CONN, 1)
    fallback = _connection(SourceBondRecordType.PDB_CONECT, 2)
    for left, right in ((typed, fallback), (fallback, typed)):
        with pytest.raises(ModelInvariantError, match="conflicting.*orders"):
            left.merge(right)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_both_order_boundaries_reject_invalid_values(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SourceBondMetadata(
            SourceBondRecordType.PDB_CONECT, reported_order=cast(int, invalid)
        )
    with pytest.raises((TypeError, ValueError)):
        TopologyBond(AtomIndex(0), AtomIndex(1), order=cast(int, invalid))


@pytest.mark.parametrize(
    "token,expected", [(None, 2), ("?", 2), (".", 2), ("'DoUb'", 2), ("sing", 1)]
)
def test_mmcif_absent_and_explicit_order_have_different_precedence(
    token: str | None, expected: int
) -> None:
    original = _with_bond(
        _two_atoms("ALA"), TopologyBond(AtomIndex(0), AtomIndex(1), order=2)
    )
    document = gemmi.cif.read_string(write_structure_string(original, FileFormat.MMCIF))
    if token is None:
        loop = (
            document.sole_block().find_loop("_struct_conn.pdbx_value_order").get_loop()
        )
        assert loop is not None
        loop.remove_column("_struct_conn.pdbx_value_order")
    else:
        document.sole_block().find_values("_struct_conn.pdbx_value_order")[0] = token
    restored = read_structure_string(document.as_string(), FileFormat.MMCIF)
    (bond,) = restored.topology.bonds
    assert bond.order == expected
    assert bond.source_metadata is not None
    assert bond.source_metadata.reported_order == (
        None if token in {None, "?", "."} else expected
    )


def test_mmcif_unsupported_explicit_order_is_not_silently_defaulted() -> None:
    original = _with_bond(
        _two_atoms(), TopologyBond(AtomIndex(0), AtomIndex(1), order=2)
    )
    document = gemmi.cif.read_string(write_structure_string(original, FileFormat.MMCIF))
    document.sole_block().find_values("_struct_conn.pdbx_value_order")[0] = "invalid"
    with pytest.raises(StructureNormalizationError, match="unsupported.*order"):
        read_structure_string(document.as_string(), FileFormat.MMCIF)


def test_bad_conect_source_serial_does_not_reassign_first_target_as_source() -> None:
    structure = read_structure_string(
        _conect_input(("CONECTxxxxx    1    2",)), FileFormat.PDB
    )
    assert structure.topology.bonds == ()


@pytest.mark.parametrize("order", (1, 2, 3, None))
def test_canonical_order_overrides_default_template_at_native_binding(
    order: int | None,
) -> None:
    structure = _with_bond(
        _two_atoms("ALA"), TopologyBond(AtomIndex(0), AtomIndex(1), order=order)
    )
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_residues((ResidueId("A", 1),)),
        component_library=build_standard_component_library(),
    )
    if order is None:
        with pytest.raises(RefinementError, match="bond order None"):
            build_rdkit_molecule(problem)
    else:
        molecule, _ = build_rdkit_molecule(problem)
        assert [bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()] == [
            float(order)
        ]


def test_unknown_order_outside_selected_region_does_not_block_binding() -> None:
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", number),
                        atoms=(
                            atom_payload("C", "C", Vec3(x, 0, 0)),
                            atom_payload("O", "O", Vec3(x + 1.23, 0, 0)),
                        ),
                    )
                    for number, x in ((1, 0.0), (10, 100.0), (20, 102.0))
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    structure = _with_bond(
        structure, TopologyBond(AtomIndex(2), AtomIndex(4), order=None)
    )
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_residues((ResidueId("A", 1),)),
        component_library=build_standard_component_library(),
    )
    molecule, _ = build_rdkit_molecule(problem)
    assert molecule.GetNumAtoms() == 2
    assert molecule.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == 2.0


def test_conect_packing_keeps_multiplicity_together_across_rows() -> None:
    residue = residue_payload(
        component_id="LIG",
        residue_id=ResidueId("L", 1),
        is_hetero=True,
        atoms=tuple(atom_payload(f"C{i}", "C", Vec3(float(i), 0, 0)) for i in range(4)),
    )
    structure = build_structure(
        chains=(), ligands=(residue,), source_format=FileFormat.PDB
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                TopologyBond(AtomIndex(0), AtomIndex(i), order=i + 1)
                for i in range(1, 4)
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    text = write_structure_string(structure, FileFormat.PDB)
    lines = tuple(line for line in text.splitlines() if line.startswith("CONECT    1"))
    assert lines == (
        "CONECT    1    2    2",
        "CONECT    1    3    3    3",
        "CONECT    1    4    4    4    4",
    )
    restored = read_structure_string(text, FileFormat.PDB)
    assert [bond.order for bond in restored.topology.bonds] == [2, 3, 4]


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_source_hydrogen_charge_and_coordinates_survive_order_projection(
    file_format: FileFormat,
) -> None:
    residue = residue_payload(
        component_id="LIG",
        residue_id=ResidueId("L", 1),
        is_hetero=True,
        atoms=(
            atom_payload("C", "C", Vec3(0, 0, 0)),
            atom_payload("O", "O", Vec3(1.23, 0, 0), formal_charge=1),
            atom_payload("H", "H", Vec3(1.5, 0.9, 0)),
        ),
    )
    structure = build_structure(
        chains=(), ligands=(residue,), source_format=FileFormat.PDB
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(AtomIndex(0), AtomIndex(1), order=2),
                TopologyBond(AtomIndex(1), AtomIndex(2)),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    restored = read_structure_string(
        write_structure_string(structure, file_format), file_format
    )
    assert restored.constitution == structure.constitution
    assert tuple(atom.position for atom in restored.geometry.atom_geometries) == tuple(
        atom.position for atom in structure.geometry.atom_geometries
    )
    assert restored.topology.atom_topologies == structure.topology.atom_topologies
    assert restored.topology.bonds[0].order == 2
    assert {bond.endpoint_pair() for bond in restored.topology.bonds} == {
        bond.endpoint_pair() for bond in structure.topology.bonds
    }


def test_disulfide_source_type_cannot_be_combined_with_multiple_order() -> None:
    original = _connection(SourceBondRecordType.MMCIF_STRUCT_CONN, 2)
    connection = replace(
        original,
        source_metadata=replace(
            original.source_metadata,
            reported_relationship_type=BondRelationshipType.DISULFIDE,
        ),
    )
    with pytest.raises(ModelInvariantError, match="disulfide.*multiple"):
        connection.to_topology_bond(AtomIndex(0), AtomIndex(1), expected_bond=None)


@pytest.mark.parametrize("reported_order", (None, 1))
def test_disulfide_resolution_does_not_invent_explicit_order_evidence(
    reported_order: int | None,
) -> None:
    original = _connection(SourceBondRecordType.PDB_SSBOND, reported_order)
    connection = replace(
        original,
        source_metadata=replace(
            original.source_metadata,
            reported_relationship_type=BondRelationshipType.DISULFIDE,
        ),
    )
    bond = connection.to_topology_bond(AtomIndex(0), AtomIndex(1), expected_bond=None)
    assert bond.order == 1
    assert bond.source_metadata is not None
    assert bond.source_metadata.reported_order == reported_order


@pytest.mark.parametrize("layout", ("rdkit", "six_fields", "split_order"))
def test_rdkit_phosphate_connectivity_and_multiplicity(layout: str) -> None:
    molecule = Chem.MolFromSmiles("P(=O)(O)(O)O")
    text = Chem.MolToPDBBlock(molecule)
    atom_lines = [
        line for line in text.splitlines() if not line.startswith(("CONECT", "END"))
    ]
    if layout == "six_fields":
        text = "\n".join((*atom_lines, "CONECT    1    2    2    3    4    5", "END"))
    elif layout == "split_order":
        text = "\n".join(
            (*atom_lines, "CONECT    1    2    3    4    5", "CONECT    1    2", "END")
        )
    restored = read_structure_string(text, FileFormat.PDB)
    assert len(restored.topology.bonds) == molecule.GetNumBonds()
    assert [bond.order for bond in restored.topology.bonds] == [2, None, None, None]
    output = write_structure_string(restored, FileFormat.PDB)
    native = Chem.MolFromPDBBlock(
        output, sanitize=False, removeHs=False, proximityBonding=False
    )
    assert native is not None
    assert [bond.GetBondTypeAsDouble() for bond in native.GetBonds()] == [
        2.0,
        1.0,
        1.0,
        1.0,
    ]


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_known_cofactor_order_uses_component_chemistry(file_format: FileFormat) -> None:
    residue = residue_payload(
        component_id="FAD",
        residue_id=ResidueId("L", 1),
        is_hetero=True,
        atoms=(
            atom_payload("PA", "P", Vec3(0, 0, 0)),
            atom_payload("O1A", "O", Vec3(1.5, 0, 0)),
        ),
    )
    structure = build_structure(
        chains=(), ligands=(residue,), source_format=FileFormat.PDB
    )
    source = write_structure_string(structure, FileFormat.PDB).replace(
        "\nEND", "\nCONECT    1    2\nEND", 1
    )
    restored = read_structure_string(source, FileFormat.PDB)
    for _ in range(2):
        assert restored.topology.bonds[0].order == 2
        assert (
            restored.topology.bonds[0].relationship_type
            is BondRelationshipType.COVALENT
        )
        restored = read_structure_string(
            write_structure_string(restored, file_format), file_format
        )


def test_pdb_link_keeps_typed_metadata_while_conect_supplies_order() -> None:
    structure = build_structure(
        chains=(),
        ligands=tuple(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", i + 1),
                is_hetero=True,
                atoms=(atom_payload(name, name, Vec3(1.23 * i, 0, 0)),),
            )
            for i, name in enumerate(("C", "O"))
        ),
        source_format=FileFormat.PDB,
    )
    structure = _with_bond(
        structure,
        TopologyBond(
            AtomIndex(0),
            AtomIndex(1),
            order=2,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                SourceBondRecordType.PDB_LINK, "link", 1.23
            ),
        ),
    )
    text = write_structure_string(structure, FileFormat.PDB)
    typed_only = read_structure_string(
        "\n".join(line for line in text.splitlines() if not line.startswith("CONECT")),
        FileFormat.PDB,
    )
    restored = read_structure_string(text, FileFormat.PDB)
    (bond,) = restored.topology.bonds
    assert typed_only.topology.bonds[0].order is None
    assert bond.order == 2
    assert bond.relationship_type is BondRelationshipType.COVALENT
    metadata = typed_only.topology.bonds[0].source_metadata
    assert metadata is not None
    assert metadata.reported_order is None
    assert bond.source_metadata == replace(metadata, reported_order=2)


@pytest.mark.parametrize(
    "order,reported_order", [(None, None), (1, None), (2, None), (2, 2), (1, 2)]
)
def test_order_survives_constitution_remapping(
    order: int | None, reported_order: int | None
) -> None:
    metadata = SourceBondMetadata(
        SourceBondRecordType.MMCIF_STRUCT_CONN, reported_order=reported_order
    )
    source = _with_bond(
        _two_atoms(),
        TopologyBond(
            AtomIndex(0),
            AtomIndex(1),
            order=order,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=metadata,
        ),
    )
    target = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("A", 1),
                is_hetero=True,
                atoms=(
                    atom_payload("H", "H", Vec3(0, 1, 0)),
                    atom_payload("O", "O", Vec3(1.23, 0, 0)),
                    atom_payload("C", "C", Vec3(0, 0, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    (remapped,) = source.topology.bonds_for_constitution(
        source_constitution=source.constitution, target_constitution=target.constitution
    )
    assert remapped.endpoint_pair() == (AtomIndex(1), AtomIndex(2))
    assert remapped.order == order
    assert remapped.source_metadata == metadata


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("effective_order,reported_order", [(2, 1), (3, 2)])
def test_writer_and_native_binding_use_effective_not_reported_order(
    file_format: FileFormat, effective_order: int, reported_order: int
) -> None:
    metadata = SourceBondMetadata(
        SourceBondRecordType.MMCIF_STRUCT_CONN, reported_order=reported_order
    )
    original = _with_bond(
        _two_atoms("ALA"),
        TopologyBond(
            AtomIndex(0),
            AtomIndex(1),
            order=effective_order,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=metadata,
        ),
    )
    problem = build_continuous_relaxation_problem(
        original,
        LocalScopeSpec.from_residues((ResidueId("A", 1),)),
        component_library=build_standard_component_library(),
    )
    molecule, _ = build_rdkit_molecule(problem)
    assert [bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()] == [
        float(effective_order)
    ]

    restored = read_structure_string(
        write_structure_string(original, file_format), file_format
    )
    (bond,) = restored.topology.bonds
    assert bond.order == effective_order
    assert bond.source_metadata is not None
    assert bond.source_metadata.reported_order == effective_order
    assert original.topology.bonds[0].source_metadata == metadata


@pytest.mark.parametrize("reported_order", (None, 1, 2))
@pytest.mark.parametrize("targeted", (False, True))
def test_source_order_evidence_survives_heavy_and_hydrogen_completion(
    reported_order: int | None, targeted: bool
) -> None:
    residue_id = ResidueId("A", 1)
    residue = residue_payload(
        component_id="ALA",
        residue_id=residue_id,
        atoms=(
            atom_payload("N", "N", Vec3(-1.2, 0.8, 0)),
            atom_payload("CA", "C", Vec3(0, 0, 0)),
            atom_payload("C", "C", Vec3(1.3, 0.5, 0)),
            atom_payload("O", "O", Vec3(2.2, -0.3, 0)),
        ),
    )
    original = _with_bond(
        build_structure(
            chains=(chain_payload("A", (residue,)),), source_format=FileFormat.MMCIF
        ),
        TopologyBond(AtomIndex(2), AtomIndex(3), order=reported_order),
    )
    original = read_structure_string(
        write_structure_string(original, FileFormat.MMCIF), FileFormat.MMCIF
    )
    source_bond = original.topology.bond_between(AtomIndex(2), AtomIndex(3))
    assert source_bond is not None and source_bond.source_metadata is not None
    assert source_bond.source_metadata.reported_order == reported_order
    target_residue_ids = frozenset((residue_id,)) if targeted else None
    repaired = repair_heavy_atoms_core(
        original, target_residue_ids=target_residue_ids
    ).structure
    assert (
        repaired.constitution.resolve_atom_index(AtomRef(residue_id, "CB")) is not None
    )
    hydrogenated = materialize_hydrogens_core(
        repaired, target_residue_ids=target_residue_ids
    ).structure
    assert any(
        atom_site.is_hydrogen() for atom_site in hydrogenated.constitution.atom_slots
    )

    for structure in (repaired, hydrogenated, hydrogenated.without_hydrogens()):
        carbon_index = structure.constitution.resolve_atom_index(
            AtomRef(residue_id, "C")
        )
        oxygen_index = structure.constitution.resolve_atom_index(
            AtomRef(residue_id, "O")
        )
        assert carbon_index is not None and oxygen_index is not None
        bond = structure.topology.bond_between(carbon_index, oxygen_index)
        assert bond is not None
        assert bond.order == source_bond.order
        assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
        assert bond.source_metadata == source_bond.source_metadata


def test_mmcif_shared_source_id_does_not_mix_orders_between_connections() -> None:
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=(
                    atom_payload("C", "C", Vec3(0, 0, 0)),
                    atom_payload("O", "O", Vec3(1.23, 0, 0)),
                    atom_payload("N", "N", Vec3(-1.33, 0, 0)),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    metadata = SourceBondMetadata(SourceBondRecordType.MMCIF_STRUCT_CONN, "shared")
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                TopologyBond(
                    AtomIndex(0),
                    AtomIndex(i),
                    order=order,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=metadata,
                )
                for i, order in ((1, 2), (2, 1))
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    text = write_structure_string(structure, FileFormat.MMCIF)
    assert (
        len(
            set(gemmi.cif.read_string(text).sole_block().find_values("_struct_conn.id"))
        )
        == 2
    )
    restored = read_structure_string(text, FileFormat.MMCIF)
    assert [bond.order for bond in restored.topology.bonds] == [2, 1]
    assert all(
        bond.source_metadata is not None and bond.source_metadata.source_id == "shared"
        for bond in restored.topology.bonds
    )
