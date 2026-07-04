from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.parser_readability import (
    prepare_rdkit_no_conect_known_bond_lookup,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.io.pdb_projection import (
    pdb_without_conect,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.structure import StructureTopology
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    TopologyBond,
)


def test_known_bond_lookup_compiles_intra_residue_template_pairs() -> None:
    """Parser-count probes should reuse residue-local known-bond topology."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("H", "H", Vec3(-1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    lookup = prepare_rdkit_no_conect_known_bond_lookup(
        structure,
        component_library=build_default_component_library(),
    )

    assert lookup.is_known_bond(
        AtomRef(residue_id, "N"),
        AtomRef(residue_id, "CA"),
    )
    assert lookup.is_known_bond(
        AtomRef(residue_id, "CA"),
        AtomRef(residue_id, "C"),
    )
    assert not lookup.is_known_bond(
        AtomRef(residue_id, "H"),
        AtomRef(residue_id, "CA"),
    )


def test_known_bond_lookup_uses_topology_bond_projection_before_fallback() -> None:
    """Parser probes should treat topology-owned covalent bonds as known facts."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNK",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("O1", "O", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("N1", "N", Vec3(2.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    structure_with_topology_bonds = type(structure).from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=AtomIndex(0),
                    atom_index_2=AtomIndex(1),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.EVIDENCE_RESOLVED,
                ),
                TopologyBond(
                    atom_index_1=AtomIndex(1),
                    atom_index_2=AtomIndex(2),
                    relationship_type=BondRelationshipType.UNKNOWN,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )

    lookup = prepare_rdkit_no_conect_known_bond_lookup(
        structure_with_topology_bonds,
        component_library=build_default_component_library(),
    )

    assert lookup.is_known_bond(
        AtomRef(residue_id, "C1"),
        AtomRef(residue_id, "O1"),
    )
    assert not lookup.is_known_bond(
        AtomRef(residue_id, "O1"),
        AtomRef(residue_id, "N1"),
    )


def test_known_bond_lookup_treats_source_explicit_unknown_bonds_as_known() -> None:
    """Parser probes should not report source-explicit CONECT as extra topology."""

    moving_residue_id = ResidueId("A", 1)
    context_residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="UNK",
                        residue_id=moving_residue_id,
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("O1", "O", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="OBS",
                residue_id=context_residue_id,
                atoms=(atom_payload("N1", "N", Vec3(2.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    structure_with_conect_bonds = type(structure).from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=AtomIndex(0),
                    atom_index_2=AtomIndex(1),
                    relationship_type=BondRelationshipType.UNKNOWN,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                        source_id="CONECT",
                    ),
                ),
                TopologyBond(
                    atom_index_1=AtomIndex(1),
                    atom_index_2=AtomIndex(2),
                    relationship_type=BondRelationshipType.UNKNOWN,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                        source_id="CONECT",
                    ),
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )

    lookup = prepare_rdkit_no_conect_known_bond_lookup(
        structure_with_conect_bonds,
        component_library=build_default_component_library(),
    )

    assert lookup.is_known_bond(
        AtomRef(moving_residue_id, "C1"),
        AtomRef(moving_residue_id, "O1"),
    )
    assert lookup.is_known_bond(
        AtomRef(moving_residue_id, "O1"),
        AtomRef(context_residue_id, "N1"),
    )


def test_no_conect_pdb_block_projector_patches_coordinate_only_updates() -> None:
    """Coordinate projection should preserve gemmi PDB output semantics."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("H", "H", Vec3(-1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    assert projector is not None

    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    updated_structure = structure.with_updated_residue_geometries(
        (
            (
                residue_id,
                residue_geometry.with_atom_geometry(
                    "CA",
                    residue_geometry.atom_geometry("CA").with_position(
                        Vec3(7.0, 8.0, 9.0)
                    ),
                ),
            ),
        )
    )

    assert projector.render(structure) == pdb_without_conect(structure)
    assert projector.render(updated_structure) == pdb_without_conect(updated_structure)
