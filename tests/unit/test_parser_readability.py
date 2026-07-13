from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

import protrepair.io.pdb_projection as pdb_projection_module
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics import parser_readability as parser_readability_module
from protrepair.diagnostics.parser_readability import (
    _atom_ref_from_pdb_atom_line,
    prepare_rdkit_no_conect_known_bond_lookup,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.io.gemmi_ingress import read_structure
from protrepair.io.pdb_projection import (
    pdb_without_conect,
    pdb_without_conect_for_parser_probe,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.structure import AtomGeometry, ProteinStructure, StructureTopology
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    TopologyBond,
)


def test_parser_readability_raises_when_required_rdkit_backend_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required RDKit diagnostics should not silently report an empty probe."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H", "H", Vec3(0.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    monkeypatch.setattr(parser_readability_module, "Chem", None)
    monkeypatch.setattr(parser_readability_module, "rdBase", None)

    with pytest.raises(RdkitUnavailableError, match="operational RDKit installation"):
        parser_readability_module.probe_rdkit_no_conect_parser_readability(structure)
    with pytest.raises(RdkitUnavailableError, match="operational RDKit installation"):
        parser_readability_module.measure_rdkit_no_conect_sanitize_readability(
            structure
        )
    extra_heavy_bond_count = (
        parser_readability_module
        .measure_rdkit_no_conect_extra_heavy_proximity_bond_count
    )
    with pytest.raises(RdkitUnavailableError, match="operational RDKit installation"):
        extra_heavy_bond_count(structure)


def test_pdb_atom_line_parser_normalizes_blank_chain_id() -> None:
    """Diagnostics PDB parsing should share ingress blank-chain normalization."""

    atom_ref = _atom_ref_from_pdb_atom_line(
        "ATOM      1  N   GLY     1       1.000   2.000   3.000  "
        "1.00 20.00           N  "
    )

    assert atom_ref == AtomRef(ResidueId("_", 1), "N")


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


def test_no_conect_pdb_block_projector_matches_gemmi_coordinate_formatting() -> (
    None
):
    """Projection should preserve Gemmi's half ties and field truncation."""

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
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    assert projector is not None

    for position in (
        Vec3(7.6255, -1.2355, -0.0005),
        Vec3(-999.9995, 9999.9995, -99.9995),
    ):
        updated_structure = structure.with_updated_residue_geometries(
            (
                (
                    residue_id,
                    residue_geometry.with_atom_geometry(
                        "CA",
                        residue_geometry.atom_geometry("CA").with_position(position),
                    ),
                ),
            )
        )

        assert projector.render(updated_structure) == pdb_without_conect(
            updated_structure
        )


def test_no_conect_pdb_block_projector_avoids_repeated_gemmi_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated coordinate projections should reuse one canonical base write."""

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
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    writer_structure_ids: list[int] = []
    canonical_writer = (
        pdb_projection_module.write_pdb_structure_string_without_conect
    )

    def _recording_writer(candidate_structure: ProteinStructure) -> str:
        writer_structure_ids.append(id(candidate_structure))
        return canonical_writer(candidate_structure)

    monkeypatch.setattr(
        pdb_projection_module,
        "write_pdb_structure_string_without_conect",
        _recording_writer,
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    assert projector is not None

    projector.render(structure)
    projector.render(structure)

    assert writer_structure_ids == [id(structure)]


def test_no_conect_pdb_block_projector_formats_only_changed_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate projection should format only atom slots that moved."""

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
    same_coordinate_structure = structure.with_updated_residue_geometries(
        ((residue_id, residue_geometry),)
    )
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
    formatted_coordinate_triples: list[tuple[float, float, float]] = []
    canonical_formatter = pdb_projection_module._format_pdb_coordinates

    def _recording_formatter(x: float, y: float, z: float) -> str:
        formatted_coordinate_triples.append((x, y, z))
        return canonical_formatter(x, y, z)

    monkeypatch.setattr(
        pdb_projection_module,
        "_format_pdb_coordinates",
        _recording_formatter,
    )

    assert projector.render(structure) == pdb_without_conect(structure)
    assert projector.render(same_coordinate_structure) == pdb_without_conect(
        same_coordinate_structure
    )
    assert formatted_coordinate_triples == []

    assert projector.render(updated_structure) == pdb_without_conect(
        updated_structure
    )
    assert formatted_coordinate_triples == [(7.0, 8.0, 9.0)]


def test_no_conect_pdb_block_projector_falls_back_on_address_space_mismatch() -> (
    None
):
    """Equal atom counts must not permit projection across atom identities."""

    residue_id = ResidueId("A", 1)
    source_structure = build_structure(
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
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    same_size_different_address_space = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(7.0, 8.0, 9.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(source_structure)
    assert projector is not None
    assert not projector.can_render(same_size_different_address_space)

    assert pdb_without_conect_for_parser_probe(
        same_size_different_address_space,
        projector,
    ) == pdb_without_conect(same_size_different_address_space)


def test_no_conect_pdb_block_projector_rejects_non_coordinate_metadata_drift() -> (
    None
):
    """Same-address candidates must preserve parser-visible metadata."""

    residue_id = ResidueId("A", 1)
    source_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("CA", "C", Vec3(1.0, 2.0, 3.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(source_structure)
    assert projector is not None

    changed_component = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("CA", "C", Vec3(1.0, 2.0, 3.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    changed_element = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("CA", "N", Vec3(1.0, 2.0, 3.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    residue_index = source_structure.constitution.residue_index(residue_id)
    residue_site = source_structure.constitution.residue_site_at(residue_index)
    residue_geometry = source_structure.residue_geometry(residue_index)
    base_atom_geometry = residue_geometry.atom_geometry("CA")
    changed_charge = source_structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=(("CA", 1),),
    )
    changed_geometry_metadata = tuple(
        source_structure.with_updated_residue_geometries(
            (
                (
                    residue_id,
                    residue_geometry.with_atom_geometry(
                        "CA",
                        AtomGeometry(
                            position=Vec3(1.0, 2.0, 3.0),
                            occupancy=occupancy,
                            b_factor=b_factor,
                            altloc=altloc,
                        ),
                    ),
                ),
            )
        )
        for occupancy, b_factor, altloc in (
            (0.5, base_atom_geometry.b_factor, base_atom_geometry.altloc),
            (base_atom_geometry.occupancy, 12.0, base_atom_geometry.altloc),
            (base_atom_geometry.occupancy, base_atom_geometry.b_factor, "A"),
        )
    )

    for candidate in (
        changed_component,
        changed_element,
        changed_charge,
        *changed_geometry_metadata,
    ):
        assert not projector.can_render(candidate)
        with pytest.raises(ValueError, match="projection-compatible serialization"):
            projector.render(candidate)
        assert pdb_without_conect_for_parser_probe(
            candidate,
            projector,
        ) == pdb_without_conect(candidate)


@pytest.mark.parametrize(
    ("fixture_path", "record_type", "expected_fast_path"),
    (
        (
            Path("tests/fixtures/pdb/1aho.pdb"),
            SourceBondRecordType.PDB_SSBOND,
            False,
        ),
        (
            Path("tests/fixtures/corpus/pdb1vbo.ent"),
            SourceBondRecordType.PDB_LINK,
            False,
        ),
        (
            Path("tests/fixtures/corpus/6d83.cif"),
            SourceBondRecordType.MMCIF_STRUCT_CONN,
            False,
        ),
        (
            Path("tests/fixtures/corpus/pdb2dri.ent"),
            SourceBondRecordType.PDB_CONECT,
            True,
        ),
    ),
)
def test_no_conect_pdb_block_projector_applies_connection_emission_policy(
    fixture_path: Path,
    record_type: SourceBondRecordType,
    expected_fast_path: bool,
) -> None:
    """Only endpoints serialized outside CONECT require canonical writing."""

    structure = read_structure(fixture_path)
    connection_bond = next(
        bond
        for bond in structure.topology.bonds
        if bond.source_metadata is not None
        and bond.source_metadata.record_type is record_type
    )
    atom_ref = structure.constitution.atom_ref_at(connection_bond.atom_index_1)
    residue_index = structure.constitution.residue_index(atom_ref.residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    atom_geometry = residue_geometry.atom_geometry(atom_ref.atom_name)
    position = atom_geometry.position
    moved_structure = structure.with_updated_residue_geometries(
        (
            (
                atom_ref.residue_id,
                residue_geometry.with_atom_geometry(
                    atom_ref.atom_name,
                    atom_geometry.with_position(
                        Vec3(position.x + 1.0, position.y, position.z)
                    ),
                ),
            ),
        )
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    assert projector is not None

    assert projector.can_render(moved_structure) is expected_fast_path
    if expected_fast_path:
        assert projector.render(moved_structure) == pdb_without_conect(moved_structure)
    else:
        with pytest.raises(ValueError, match="projection-compatible serialization"):
            projector.render(moved_structure)
    assert pdb_without_conect_for_parser_probe(
        moved_structure,
        projector,
    ) == pdb_without_conect(moved_structure)


@pytest.mark.parametrize(
    ("relationship_type", "expected_record_prefix", "expected_fast_path"),
    (
        (BondRelationshipType.HYDROGEN_BOND, None, True),
        (BondRelationshipType.UNKNOWN, "LINK", False),
    ),
)
def test_no_conect_pdb_block_projector_applies_typed_connection_policy(
    relationship_type: BondRelationshipType,
    expected_record_prefix: str | None,
    expected_fast_path: bool,
) -> None:
    """Projection sensitivity must match typed PDB record emission."""

    residue_id_1 = ResidueId("A", 1)
    residue_id_2 = ResidueId("A", 2)
    base_structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=residue_id_1,
                        atoms=(atom_payload("OG", "O", Vec3(0.0, 0.0, 0.0)),),
                    ),
                    residue_payload(
                        component_id="ASN",
                        residue_id=residue_id_2,
                        atoms=(atom_payload("ND2", "N", Vec3(2.8, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    structure = ProteinStructure.from_payload(
        constitution=base_structure.constitution,
        geometry=base_structure.geometry,
        topology=StructureTopology(
            constitution=base_structure.constitution,
            atom_topologies=base_structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=AtomIndex(0),
                    atom_index_2=AtomIndex(1),
                    relationship_type=relationship_type,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                        source_id="hydrog1",
                    ),
                ),
            ),
        ),
        provenance=base_structure.provenance,
    )
    residue_index = structure.constitution.residue_index(residue_id_1)
    residue_geometry = structure.residue_geometry(residue_index)
    moved_structure = structure.with_updated_residue_geometries(
        (
            (
                residue_id_1,
                residue_geometry.with_atom_geometry(
                    "OG",
                    residue_geometry.atom_geometry("OG").with_position(
                        Vec3(0.5, 0.0, 0.0)
                    ),
                ),
            ),
        )
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    assert projector is not None
    base_pdb = pdb_without_conect(structure)

    assert (
        next(
            (
                line[:6].strip()
                for line in base_pdb.splitlines()
                if line.startswith(("LINK", "SSBOND"))
            ),
            None,
        )
        == expected_record_prefix
    )
    assert projector.can_render(moved_structure) is expected_fast_path
    if expected_fast_path:
        assert projector.render(moved_structure) == pdb_without_conect(moved_structure)
    else:
        with pytest.raises(ValueError, match="projection-compatible serialization"):
            projector.render(moved_structure)
    assert pdb_without_conect_for_parser_probe(
        moved_structure,
        projector,
    ) == pdb_without_conect(moved_structure)
