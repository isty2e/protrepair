"""Contract tests for the canonical topology bond graph."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.errors import ModelInvariantError
from protrepair.geometry import Vec3
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
    is_covalent_like_relationship,
    is_model_resolved_provenance,
    is_source_provenance,
)


def _residue_id(chain: str, seq: int) -> ResidueId:
    return ResidueId(chain_id=chain, seq_num=seq, insertion_code=None)


def _two_residue_constitution():
    """Build a minimal structure with two residues (4 atom slots total)."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=_residue_id("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0, 0, 0)),
                            atom_payload("CA", "C", Vec3(1, 0, 0)),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=_residue_id("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(2, 0, 0)),
                            atom_payload("CA", "C", Vec3(3, 0, 0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _bonded_rewrite_structure():
    """Build a structure with bonds crossing chains, ligands, and hydrogens."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=_residue_id("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0, 0, 0)),
                            atom_payload("H", "H", Vec3(0, 0, 1)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=_residue_id("B", 2),
                        atoms=(atom_payload("C", "C", Vec3(2, 0, 0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=_residue_id("L", 9),
                atoms=(
                    atom_payload("O", "O", Vec3(3, 0, 0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return type(structure).from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=AtomIndex(0),
                    atom_index_2=AtomIndex(1),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                        source_id="N-H",
                    ),
                ),
                TopologyBond(
                    atom_index_1=AtomIndex(0),
                    atom_index_2=AtomIndex(2),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                        source_id="N-C",
                    ),
                ),
                TopologyBond(
                    atom_index_1=AtomIndex(2),
                    atom_index_2=AtomIndex(3),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                        source_id="C-O",
                    ),
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _bond_source_ids(topology: StructureTopology) -> tuple[str | None, ...]:
    """Return source ids for topology bonds in canonical storage order."""

    return tuple(
        None if bond.source_metadata is None else bond.source_metadata.source_id
        for bond in topology.bonds
    )


# --- BondProvenance ---


class TestBondProvenance:
    def test_enum_values(self):
        assert BondProvenance.SOURCE_EXPLICIT == "source_explicit"
        assert BondProvenance.TEMPLATE_RESOLVED == "template_resolved"
        assert BondProvenance.SEQUENCE_INFERRED == "sequence_inferred"
        assert BondProvenance.EVIDENCE_RESOLVED == "evidence_resolved"
        assert BondProvenance.REPAIR_INFERRED == "repair_inferred"

    def test_is_str_enum(self):
        assert isinstance(BondProvenance.SOURCE_EXPLICIT, str)

    def test_closed_support_mode_axis(self):
        """Provenance records support mode, not lifecycle or serialization."""

        assert set(BondProvenance) == {
            BondProvenance.SOURCE_EXPLICIT,
            BondProvenance.TEMPLATE_RESOLVED,
            BondProvenance.SEQUENCE_INFERRED,
            BondProvenance.EVIDENCE_RESOLVED,
            BondProvenance.REPAIR_INFERRED,
        }


# --- BondRelationshipType ---


class TestBondRelationshipType:
    def test_enum_values(self):
        assert BondRelationshipType.COVALENT == "covalent"
        assert BondRelationshipType.DISULFIDE == "disulfide"
        assert BondRelationshipType.HYDROGEN_BOND == "hydrogen_bond"
        assert BondRelationshipType.METAL_COORDINATION == "metal_coordination"
        assert BondRelationshipType.UNKNOWN == "unknown"


# --- SourceBondRecordType ---


class TestSourceBondRecordType:
    def test_enum_values(self):
        assert SourceBondRecordType.PDB_LINK == "pdb_link"
        assert SourceBondRecordType.PDB_CONECT == "pdb_conect"
        assert SourceBondRecordType.MMCIF_STRUCT_CONN == "mmcif_struct_conn"


# --- SourceBondMetadata ---


class TestSourceBondMetadata:
    def test_construction_with_required_record_type(self):
        meta = SourceBondMetadata(record_type=SourceBondRecordType.PDB_CONECT)
        assert meta.record_type is SourceBondRecordType.PDB_CONECT
        assert meta.source_id is None
        assert meta.reported_distance_angstrom is None

    def test_full_metadata(self):
        meta = SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_LINK,
            source_id="disulf1",
            reported_distance_angstrom=2.05,
        )
        assert meta.source_id == "disulf1"
        assert meta.reported_distance_angstrom == 2.05

    def test_source_id_stripped_to_none(self):
        meta = SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_CONECT,
            source_id="   ",
        )
        assert meta.source_id is None

    def test_negative_distance_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_LINK,
                reported_distance_angstrom=-1.0,
            )

    def test_zero_distance_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_LINK,
                reported_distance_angstrom=0.0,
            )

    @pytest.mark.parametrize("reported_distance", (float("nan"), float("inf")))
    def test_non_finite_distance_rejected(self, reported_distance: float):
        with pytest.raises(ValueError, match="finite"):
            SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_LINK,
                reported_distance_angstrom=reported_distance,
            )


# --- TopologyBond ---


class TestTopologyBond:
    def test_construction_roundtrip(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(2),
            order=1,
            relationship_type=BondRelationshipType.COVALENT,
            provenance=BondProvenance.SEQUENCE_INFERRED,
        )
        assert bond.atom_index_1 == AtomIndex(0)
        assert bond.atom_index_2 == AtomIndex(2)
        assert bond.order == 1
        assert bond.aromatic is False
        assert bond.relationship_type is BondRelationshipType.COVALENT
        assert bond.provenance is BondProvenance.SEQUENCE_INFERRED
        assert bond.source_metadata is None

    def test_endpoint_auto_normalization(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(5),
            atom_index_2=AtomIndex(2),
        )
        assert bond.atom_index_1 == AtomIndex(2)
        assert bond.atom_index_2 == AtomIndex(5)

    def test_endpoint_pair(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(3),
            atom_index_2=AtomIndex(1),
        )
        assert bond.endpoint_pair() == (AtomIndex(1), AtomIndex(3))

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="distinct"):
            TopologyBond(atom_index_1=AtomIndex(0), atom_index_2=AtomIndex(0))

    def test_zero_order_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            TopologyBond(
                atom_index_1=AtomIndex(0),
                atom_index_2=AtomIndex(1),
                order=0,
            )

    def test_negative_order_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            TopologyBond(
                atom_index_1=AtomIndex(0),
                atom_index_2=AtomIndex(1),
                order=-1,
            )

    def test_bool_order_rejected(self):
        with pytest.raises(TypeError, match="integer"):
            TopologyBond(
                atom_index_1=AtomIndex(0),
                atom_index_2=AtomIndex(1),
                order=True,
            )

    @pytest.mark.parametrize(
        "provenance",
        [
            BondProvenance.TEMPLATE_RESOLVED,
            BondProvenance.SEQUENCE_INFERRED,
            BondProvenance.EVIDENCE_RESOLVED,
            BondProvenance.REPAIR_INFERRED,
        ],
    )
    def test_source_metadata_requires_source_explicit_provenance(self, provenance):
        with pytest.raises(ValueError, match="SOURCE_EXPLICIT"):
            TopologyBond(
                atom_index_1=AtomIndex(0),
                atom_index_2=AtomIndex(1),
                provenance=provenance,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_LINK,
                ),
            )

    def test_source_metadata_allowed_with_source_explicit(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
            ),
        )
        assert bond.source_metadata is not None
        assert bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT

    def test_source_metadata_none_allowed_with_source_explicit(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            provenance=BondProvenance.SOURCE_EXPLICIT,
        )
        assert bond.source_metadata is None

    def test_involves(self):
        bond = TopologyBond(atom_index_1=AtomIndex(0), atom_index_2=AtomIndex(3))
        assert bond.involves(AtomIndex(0)) is True
        assert bond.involves(AtomIndex(3)) is True
        assert bond.involves(AtomIndex(1)) is False

    def test_double_bond(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            order=2,
            aromatic=False,
        )
        assert bond.order == 2

    def test_aromatic_bond(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            aromatic=True,
        )
        assert bond.aromatic is True


# --- is_covalent_like_relationship ---


class TestIsCovalentLikeRelationship:
    @pytest.mark.parametrize(
        "relationship_type",
        [BondRelationshipType.COVALENT, BondRelationshipType.DISULFIDE],
    )
    @pytest.mark.parametrize("provenance", list(BondProvenance))
    def test_covalent_like_true(self, relationship_type, provenance):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            relationship_type=relationship_type,
            provenance=provenance,
            source_metadata=(
                SourceBondMetadata(record_type=SourceBondRecordType.PDB_LINK)
                if provenance is BondProvenance.SOURCE_EXPLICIT
                else None
            ),
        )
        assert is_covalent_like_relationship(bond) is True

    @pytest.mark.parametrize(
        "relationship_type",
        [
            BondRelationshipType.HYDROGEN_BOND,
            BondRelationshipType.METAL_COORDINATION,
            BondRelationshipType.UNKNOWN,
        ],
    )
    @pytest.mark.parametrize("provenance", list(BondProvenance))
    def test_non_covalent_false(self, relationship_type, provenance):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            relationship_type=relationship_type,
            provenance=provenance,
            source_metadata=(
                SourceBondMetadata(record_type=SourceBondRecordType.PDB_LINK)
                if provenance is BondProvenance.SOURCE_EXPLICIT
                else None
            ),
        )
        assert is_covalent_like_relationship(bond) is False

    def test_metal_coordination_source_explicit_not_covalent_like(self):
        """Constraint #1: SOURCE_EXPLICIT metal coordination is topology truth
        but is NOT covalent-like."""

        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            relationship_type=BondRelationshipType.METAL_COORDINATION,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                source_id="metalc1",
                reported_distance_angstrom=2.1,
            ),
        )
        assert is_covalent_like_relationship(bond) is False


# --- is_source_provenance ---


class TestIsSourceProvenance:
    def test_source_explicit_true(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
            ),
        )
        assert is_source_provenance(bond) is True

    @pytest.mark.parametrize(
        "provenance",
        [
            BondProvenance.TEMPLATE_RESOLVED,
            BondProvenance.SEQUENCE_INFERRED,
            BondProvenance.EVIDENCE_RESOLVED,
            BondProvenance.REPAIR_INFERRED,
        ],
    )
    def test_non_source_false(self, provenance):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            provenance=provenance,
        )
        assert is_source_provenance(bond) is False


# --- is_model_resolved_provenance ---


class TestIsModelResolvedProvenance:
    @pytest.mark.parametrize(
        "provenance",
        [
            BondProvenance.TEMPLATE_RESOLVED,
            BondProvenance.SEQUENCE_INFERRED,
            BondProvenance.EVIDENCE_RESOLVED,
            BondProvenance.REPAIR_INFERRED,
        ],
    )
    def test_model_resolved_true(self, provenance):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            provenance=provenance,
        )
        assert is_model_resolved_provenance(bond) is True

    def test_source_explicit_false(self):
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
            ),
        )
        assert is_model_resolved_provenance(bond) is False

    def test_repaired_evidence_covalent_is_not_source(self):
        """Evidence-resolved repaired bonds are model truth, not source replay."""

        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(1),
            relationship_type=BondRelationshipType.COVALENT,
            provenance=BondProvenance.EVIDENCE_RESOLVED,
        )

        assert is_covalent_like_relationship(bond) is True
        assert is_model_resolved_provenance(bond) is True
        assert is_source_provenance(bond) is False


# --- StructureTopology bond integration ---


class TestStructureTopologyBonds:
    def test_empty_topology_has_empty_bonds(self):
        structure = _two_residue_constitution()
        empty = StructureTopology.empty(constitution=structure.constitution)
        assert empty.bonds == ()

    def test_default_bonds_parameter(self):
        structure = _two_residue_constitution()
        topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=(None,) * len(structure.constitution.atom_slots),
        )
        assert topology.bonds == ()

    def test_bonds_stored_in_canonical_order(self):
        structure = _two_residue_constitution()
        bond_b = TopologyBond(atom_index_1=AtomIndex(1), atom_index_2=AtomIndex(3))
        bond_a = TopologyBond(atom_index_1=AtomIndex(0), atom_index_2=AtomIndex(2))
        topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=(None,) * len(structure.constitution.atom_slots),
            bonds=(bond_b, bond_a),
        )
        assert topology.bonds[0].endpoint_pair() == (AtomIndex(0), AtomIndex(2))
        assert topology.bonds[1].endpoint_pair() == (AtomIndex(1), AtomIndex(3))

    def test_exact_duplicate_bonds_deduplicated(self):
        structure = _two_residue_constitution()
        bond = TopologyBond(atom_index_1=AtomIndex(0), atom_index_2=AtomIndex(2))
        topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=(None,) * len(structure.constitution.atom_slots),
            bonds=(bond, bond),
        )
        assert len(topology.bonds) == 1

    def test_conflicting_same_endpoint_bonds_rejected(self):
        structure = _two_residue_constitution()
        bond_a = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(2),
            relationship_type=BondRelationshipType.COVALENT,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        bond_b = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(2),
            relationship_type=BondRelationshipType.DISULFIDE,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_LINK,
            ),
        )
        with pytest.raises(ModelInvariantError, match="conflicting bonds"):
            StructureTopology(
                constitution=structure.constitution,
                atom_topologies=(None,) * len(structure.constitution.atom_slots),
                bonds=(bond_a, bond_b),
            )

    def test_out_of_range_endpoint_rejected(self):
        structure = _two_residue_constitution()
        bond = TopologyBond(atom_index_1=AtomIndex(0), atom_index_2=AtomIndex(99))
        with pytest.raises(ModelInvariantError, match="valid atom slots"):
            StructureTopology(
                constitution=structure.constitution,
                atom_topologies=(None,) * len(structure.constitution.atom_slots),
                bonds=(bond,),
            )

    def test_bonds_preserved_alongside_existing_fields(self):
        structure = _two_residue_constitution()
        bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(2),
            provenance=BondProvenance.SEQUENCE_INFERRED,
        )
        topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=(
                AtomTopology(formal_charge=0),
                None,
                None,
                None,
            ),
            bonds=(bond,),
        )
        assert len(topology.bonds) == 1
        assert topology.atom_topologies[0] is not None
        assert topology.atom_topologies[0].formal_charge == 0

    def test_bond_between_returns_canonical_endpoint_bond(self):
        structure = _two_residue_constitution()
        bond = TopologyBond(atom_index_1=AtomIndex(0), atom_index_2=AtomIndex(2))
        topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=(None,) * len(structure.constitution.atom_slots),
            bonds=(bond,),
        )

        assert topology.bond_between(AtomIndex(2), AtomIndex(0)) == bond
        assert topology.bond_between(AtomIndex(0), AtomIndex(0)) is None
        assert topology.bond_between(AtomIndex(1), AtomIndex(3)) is None

    def test_covalent_like_endpoint_pairs_exclude_non_covalent_bonds(self):
        structure = _two_residue_constitution()
        covalent_bond = TopologyBond(
            atom_index_1=AtomIndex(0),
            atom_index_2=AtomIndex(2),
            relationship_type=BondRelationshipType.COVALENT,
        )
        metal_bond = TopologyBond(
            atom_index_1=AtomIndex(1),
            atom_index_2=AtomIndex(3),
            relationship_type=BondRelationshipType.METAL_COORDINATION,
        )
        topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=(None,) * len(structure.constitution.atom_slots),
            bonds=(covalent_bond, metal_bond),
        )

        assert topology.covalent_like_endpoint_pairs() == frozenset(
            {covalent_bond.endpoint_pair()}
        )

    def test_bonds_for_constitution_remaps_surviving_endpoints(self):
        structure = _bonded_rewrite_structure()
        target_constitution = structure.constitution.with_chains(
            (structure.constitution.chain("A"),)
        ).with_ligands(())

        remapped_bonds = structure.topology.bonds_for_constitution(
            source_constitution=structure.constitution,
            target_constitution=target_constitution,
        )

        assert len(remapped_bonds) == 1
        assert remapped_bonds[0].endpoint_pair() == (AtomIndex(0), AtomIndex(1))
        assert remapped_bonds[0].source_metadata is not None
        assert remapped_bonds[0].source_metadata.source_id == "N-H"

    def test_bonds_for_constitution_rejects_mismatched_source(self):
        structure = _bonded_rewrite_structure()
        selected = structure.select_chains(("A",))

        with pytest.raises(ModelInvariantError, match="matching source"):
            structure.topology.bonds_for_constitution(
                source_constitution=selected.constitution,
                target_constitution=selected.constitution,
            )


class TestProteinStructureTopologyBondRewrites:
    def test_select_chains_filters_topology_bonds(self):
        structure = _bonded_rewrite_structure()

        selected = structure.select_chains(("A",))

        assert _bond_source_ids(selected.topology) == ("N-H",)
        assert selected.topology.bonds[0].endpoint_pair() == (
            AtomIndex(0),
            AtomIndex(1),
        )

    def test_ligand_facet_replacement_filters_topology_bonds(self):
        structure = _bonded_rewrite_structure()

        without_ligands = structure.with_ligand_facets(
            ligand_sites=(),
            ligand_geometries=(),
        )

        assert _bond_source_ids(without_ligands.topology) == ("N-H", "N-C")
        assert tuple(
            bond.endpoint_pair() for bond in without_ligands.topology.bonds
        ) == (
            (AtomIndex(0), AtomIndex(1)),
            (AtomIndex(0), AtomIndex(2)),
        )

    def test_without_hydrogens_filters_incident_topology_bonds(self):
        structure = _bonded_rewrite_structure()

        without_hydrogens = structure.without_hydrogens()

        assert _bond_source_ids(without_hydrogens.topology) == ("N-C", "C-O")
        assert tuple(
            bond.endpoint_pair() for bond in without_hydrogens.topology.bonds
        ) == (
            (AtomIndex(0), AtomIndex(1)),
            (AtomIndex(1), AtomIndex(2)),
        )

    def test_residue_facet_rewrite_filters_missing_atom_bonds(self):
        structure = _bonded_rewrite_structure()
        residue_index = ResidueIndex(0)
        residue_site = structure.constitution.residue_site_at(
            residue_index
        ).without_atom_sites(("H",))

        updated = structure.with_updated_residue_facets(
            residue_site,
            residue_geometry=structure.residue_geometry(residue_index).without_atoms(
                ("H",)
            ),
        )

        assert _bond_source_ids(updated.topology) == ("N-C", "C-O")
        assert tuple(bond.endpoint_pair() for bond in updated.topology.bonds) == (
            (AtomIndex(0), AtomIndex(1)),
            (AtomIndex(1), AtomIndex(2)),
        )
