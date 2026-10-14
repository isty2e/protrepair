"""Resolved sulfur chemistry survives coordinate-format boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.clashes import detect_clashes_involving_residues
from protrepair.geometry import Vec3
from protrepair.io import (
    FileFormat,
    read_structure,
    read_structure_string,
    write_structure_string,
)
from protrepair.io.pdb_projection import (
    pdb_without_conect,
    pdb_without_conect_for_parser_probe,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.state.structure_topology import StructureDisulfideHydrogenFacts
from protrepair.structure import ProteinStructure, StructureTopology
from protrepair.structure.disulfide import disulfide_atom_ref_pairs
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    TopologyBond,
)
from protrepair.workflow.contracts import requested_process_goal

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/pdb/span-reconstruction/1crn.pdb"
)


def _with_sulfur_relationships(
    relationship_type: BondRelationshipType,
    provenance: BondProvenance,
    source_metadata: SourceBondMetadata | None = None,
) -> ProteinStructure:
    source = read_structure(FIXTURE)
    bonds = tuple(
        replace(
            bond,
            relationship_type=relationship_type,
            provenance=provenance,
            source_metadata=source_metadata,
        )
        if bond.relationship_type is BondRelationshipType.DISULFIDE
        else bond
        for bond in source.topology.bonds
    )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


@pytest.mark.parametrize("provenance", tuple(BondProvenance))
@pytest.mark.parametrize(
    "relationship_type", [BondRelationshipType.DISULFIDE, BondRelationshipType.COVALENT]
)
@pytest.mark.parametrize("file_format", [FileFormat.PDB, FileFormat.MMCIF])
def test_disulfide_roundtrip_preserves_chemistry_independently_of_provenance(
    provenance: BondProvenance,
    relationship_type: BondRelationshipType,
    file_format: FileFormat,
) -> None:
    structure = _with_sulfur_relationships(relationship_type, provenance)
    pairs = disulfide_atom_ref_pairs(structure)
    assert len(pairs) == 3
    text = write_structure_string(structure, file_format)
    roundtripped = read_structure_string(text, file_format)

    assert disulfide_atom_ref_pairs(roundtripped) == pairs
    for left, right in pairs:
        bond = roundtripped.topology.bond_between(
            roundtripped.constitution.atom_index(left),
            roundtripped.constitution.atom_index(right),
        )
        assert bond is not None
        assert bond.relationship_type is (
            BondRelationshipType.DISULFIDE
            if file_format is FileFormat.PDB
            else relationship_type
        )
        assert bond.order == 1
        assert bond.source_metadata is not None
    for index in range(structure.geometry.atom_count()):
        atom_ref = structure.constitution.atom_ref_at(AtomIndex(index))
        before = structure.geometry.position(
            structure.constitution.atom_index(atom_ref)
        )
        after = roundtripped.geometry.position(
            roundtripped.constitution.atom_index(atom_ref)
        )
        assert tuple(after) == pytest.approx(tuple(before), abs=0.0005)
    assert write_structure_string(structure, file_format) == text
    assert (
        disulfide_atom_ref_pairs(
            read_structure_string(
                write_structure_string(roundtripped, file_format), file_format
            )
        )
        == pairs
    )


@pytest.mark.parametrize(
    "relationship_type", [BondRelationshipType.DISULFIDE, BondRelationshipType.COVALENT]
)
def test_resolved_conect_disulfide_keeps_its_resolved_chemistry(
    relationship_type: BondRelationshipType,
) -> None:
    structure = _with_sulfur_relationships(
        relationship_type,
        BondProvenance.SOURCE_EXPLICIT,
        SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_CONECT, source_id="CONECT"
        ),
    )
    for text in (
        write_structure_string(structure, FileFormat.PDB),
        pdb_without_conect(structure),
    ):
        roundtripped = read_structure_string(text, FileFormat.PDB)
        assert disulfide_atom_ref_pairs(roundtripped) == disulfide_atom_ref_pairs(
            structure
        )


@pytest.mark.parametrize("order,aromatic", [(None, False), (2, False), (1, True)])
def test_generic_sulfur_bond_is_not_silently_narrowed_to_single_disulfide(
    order: int | None, aromatic: bool
) -> None:
    source = _with_sulfur_relationships(
        BondRelationshipType.COVALENT, BondProvenance.EVIDENCE_RESOLVED
    )
    pairs = disulfide_atom_ref_pairs(source)
    endpoint_pairs = {
        tuple(sorted(source.constitution.atom_index(ref) for ref in pair))
        for pair in pairs
    }
    structure = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(
                replace(bond, order=order, aromatic=aromatic)
                if bond.endpoint_pair() in endpoint_pairs
                else bond
                for bond in source.topology.bonds
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    roundtripped = read_structure_string(
        write_structure_string(structure, FileFormat.PDB), FileFormat.PDB
    )
    assert disulfide_atom_ref_pairs(roundtripped) == pairs
    for left, right in pairs:
        bond = roundtripped.topology.bond_between(
            roundtripped.constitution.atom_index(left),
            roundtripped.constitution.atom_index(right),
        )
        assert bond is not None
        assert bond.relationship_type is BondRelationshipType.COVALENT
        assert bond.order == (order if order == 2 else None)


@pytest.mark.parametrize("file_format", [FileFormat.PDB, FileFormat.MMCIF])
@pytest.mark.parametrize(
    "relationship_type", [BondRelationshipType.DISULFIDE, BondRelationshipType.COVALENT]
)
def test_resolved_disulfide_retains_chain_insertion_and_selected_altloc_identity(
    file_format: FileFormat, relationship_type: BondRelationshipType
) -> None:
    left = AtomRef(ResidueId("A", 10, "B"), "SG")
    right = AtomRef(ResidueId("Z", 10, "A"), "SG")
    source = build_structure(
        source_format=file_format,
        chains=tuple(
            chain_payload(
                ref.residue_id.chain_id,
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=ref.residue_id,
                        atoms=(
                            atom_payload("SG", "S", Vec3(x, 0.0, 0.0), altloc=altloc),
                        ),
                    ),
                ),
            )
            for ref, x, altloc in ((left, 0.0, "B"), (right, 4.0, "A"))
        ),
    )
    structure = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    source.constitution.atom_index(left),
                    source.constitution.atom_index(right),
                    relationship_type=relationship_type,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
            ),
        ),
        provenance=source.provenance,
    )
    roundtripped = read_structure_string(
        write_structure_string(structure, file_format), file_format
    )
    assert disulfide_atom_ref_pairs(roundtripped) == frozenset({(left, right)})
    for ref, altloc in ((left, "B"), (right, "A")):
        geometry = roundtripped.residue_geometry(
            roundtripped.constitution.residue_index(ref.residue_id)
        )
        assert geometry.atom_geometry(ref.atom_name).altloc == altloc


@pytest.mark.parametrize(
    "relationship_type",
    [
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.METAL_COORDINATION,
        BondRelationshipType.HYDROGEN_BOND,
    ],
)
def test_sulfur_endpoint_names_do_not_promote_unresolved_or_noncovalent_records(
    relationship_type: BondRelationshipType,
) -> None:
    structure = _with_sulfur_relationships(
        relationship_type,
        BondProvenance.SOURCE_EXPLICIT,
        SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_CONECT, source_id="CONECT"
        ),
    )
    roundtripped = read_structure_string(
        write_structure_string(structure, FileFormat.PDB), FileFormat.PDB
    )
    assert not disulfide_atom_ref_pairs(roundtripped)
    sulfur_indices = {
        roundtripped.constitution.atom_index(AtomRef(residue.residue_id, "SG"))
        for residue in roundtripped.constitution.iter_residues()
        if residue.component_id == "CYS"
    }
    sulfur_bonds = tuple(
        bond
        for bond in roundtripped.topology.bonds
        if bond.atom_index_1 in sulfur_indices and bond.atom_index_2 in sulfur_indices
    )
    assert len(sulfur_bonds) == 3
    assert all(
        bond.relationship_type is BondRelationshipType.UNKNOWN for bond in sulfur_bonds
    )
    assert not disulfide_atom_ref_pairs(
        read_structure_string(pdb_without_conect(structure), FileFormat.PDB)
    )


@pytest.mark.parametrize(
    "relationship_type", [BondRelationshipType.DISULFIDE, BondRelationshipType.COVALENT]
)
def test_model_resolved_disulfide_invalidates_coordinate_only_pdb_projection(
    relationship_type: BondRelationshipType,
) -> None:
    structure = _with_sulfur_relationships(
        relationship_type, BondProvenance.EVIDENCE_RESOLVED
    )
    left, right = next(iter(disulfide_atom_ref_pairs(structure)))
    geometry = structure.residue_geometry(
        structure.constitution.residue_index(left.residue_id)
    )
    atom = geometry.atom_geometry(left.atom_name)
    position = atom.position
    moved = structure.with_updated_residue_geometries(
        (
            (
                left.residue_id,
                geometry.with_atom_geometry(
                    left.atom_name,
                    atom.with_position(Vec3(position.x + 2.0, position.y, position.z)),
                ),
            ),
        )
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    assert projector is not None
    assert not projector.can_render(moved)
    text = pdb_without_conect_for_parser_probe(moved, projector)
    assert text == pdb_without_conect(moved)
    roundtripped = read_structure_string(text, FileFormat.PDB)
    assert disulfide_atom_ref_pairs(roundtripped) == disulfide_atom_ref_pairs(structure)
    bond = roundtripped.topology.bond_between(
        roundtripped.constitution.atom_index(left),
        roundtripped.constitution.atom_index(right),
    )
    assert bond is not None and bond.source_metadata is not None
    distance = moved.geometry.position(moved.constitution.atom_index(left)).distance_to(
        moved.geometry.position(moved.constitution.atom_index(right))
    )
    assert bond.source_metadata.reported_distance_angstrom == pytest.approx(
        distance, abs=0.0051
    )


@pytest.mark.parametrize("file_format", [FileFormat.PDB, FileFormat.MMCIF])
def test_hydrogen_completion_after_roundtrip_respects_resolved_disulfides(
    file_format: FileFormat,
) -> None:
    structure = _with_sulfur_relationships(
        BondRelationshipType.DISULFIDE, BondProvenance.EVIDENCE_RESOLVED
    )
    roundtripped = read_structure_string(
        write_structure_string(structure, file_format), file_format
    )
    result = process_structure(
        roundtripped,
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert not result.has_errors()
    assert result.requested_goals_fully_satisfied()
    pairs = disulfide_atom_ref_pairs(structure)
    assert disulfide_atom_ref_pairs(result.structure) == pairs
    assert not StructureDisulfideHydrogenFacts.from_structure(
        result.structure
    ).has_contradictions()
    for pair in pairs:
        for atom_ref in pair:
            residue = result.structure.constitution.residue_or_ligand(
                atom_ref.residue_id
            )
            assert residue is not None and not residue.has_atom_site("HG")
    contacts = detect_clashes_involving_residues(
        result.structure,
        residue_ids=frozenset(
            atom_ref.residue_id for pair in pairs for atom_ref in pair
        ),
        component_library=build_default_component_library(),
    )
    assert not any(
        clash.left_atom_name == clash.right_atom_name == "SG"
        for clash in contacts.clashes
    )
