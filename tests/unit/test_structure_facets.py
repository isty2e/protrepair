"""ProteinStructure facet replacement tests."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.errors import AtomNotFoundError, ModelInvariantError
from protrepair.geometry import Vec3
from protrepair.structure import ResidueFacetPayload
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


def test_residue_facet_payload_without_atoms_updates_all_facets() -> None:
    """Residue facet payload atom removal should preserve facet alignment."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H", "H", Vec3(0.1, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="residue-facet-payload",
    )
    residue_index = structure.constitution.residue_index(
        ResidueId(chain_id="A", seq_num=1)
    )
    payload = ResidueFacetPayload(
        residue_site=structure.constitution.residue_site_at(residue_index),
        residue_geometry=structure.residue_geometry(residue_index),
        formal_charge_by_atom_name=(("N", 1), ("H", 0)),
    )

    stripped = payload.without_atoms({"H"})

    assert not stripped.residue_site.has_atom_site("H")
    assert not stripped.residue_geometry.has_atom("H")
    assert stripped.formal_charge_by_atom_name == (("N", 1),)


def test_without_atom_refs_remaps_geometry_charge_and_topology_atomically() -> None:
    """Selective removal must preserve alignment across every structure facet."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload(
                                "N", "N", Vec3(0.0, 0.0, 0.0), formal_charge=1
                            ),
                            atom_payload(
                                "H", "H", Vec3(0.1, 0.0, 0.0), formal_charge=0
                            ),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="selective-atom-removal",
    )
    n_index = structure.constitution.atom_index(AtomRef(residue_id, "N"))
    h_index = structure.constitution.atom_index(AtomRef(residue_id, "H"))
    ca_index = structure.constitution.atom_index(AtomRef(residue_id, "CA"))
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    n_index,
                    h_index,
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
                TopologyBond(
                    n_index,
                    ca_index,
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )

    stripped = structure.without_atom_refs((AtomRef(residue_id, "H"),))

    residue = stripped.constitution.residue_site_at(
        stripped.constitution.residue_index(residue_id)
    )
    assert residue.atom_site_names() == ("N", "CA")
    assert stripped.residue_geometry(
        stripped.constitution.residue_index(residue_id)
    ).atom_names() == ("N", "CA")
    assert stripped.topology.residue_formal_charge_by_atom_name(
        constitution=stripped.constitution,
        residue_index=stripped.constitution.residue_index(residue_id),
    ) == (("N", 1),)
    assert len(stripped.topology.bonds) == 1
    remaining_bond = stripped.topology.bonds[0]
    assert {
        stripped.constitution.atom_ref_at(remaining_bond.atom_index_1),
        stripped.constitution.atom_ref_at(remaining_bond.atom_index_2),
    } == {AtomRef(residue_id, "N"), AtomRef(residue_id, "CA")}


def test_without_atom_refs_rejects_missing_ref_before_returning_partial_state() -> None:
    """Selective atom removal should validate the full request atomically."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=residue_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H", "H", Vec3(0.1, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="selective-atom-removal-missing-ref",
    )

    with pytest.raises(AtomNotFoundError):
        structure.without_atom_refs(
            (
                AtomRef(residue_id, "H"),
                AtomRef(residue_id, "MISSING"),
            )
        )

    assert structure.constitution.residue_site_at(
        structure.constitution.residue_index(residue_id)
    ).has_atom_site("H")


def test_without_atom_refs_empty_request_preserves_aggregate_identity() -> None:
    """An empty selective-removal request should be a true no-op."""

    structure = build_structure(
        chains=(),
        source_format=FileFormat.PDB,
        source_name="empty-selective-atom-removal",
    )

    assert structure.without_atom_refs(()) is structure


def test_with_updated_residue_facets_batch_matches_sequential_updates() -> None:
    """Batch replacement should preserve single-update semantics."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="ZN",
                residue_id=ResidueId(chain_id="Z", seq_num=1),
                atoms=(atom_payload("ZN", "ZN", Vec3(3.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="batch-facet-test",
    )
    polymer_site = structure.constitution.residue_or_ligand(
        ResidueId(chain_id="A", seq_num=1)
    )
    ligand_site = structure.constitution.residue_or_ligand(
        ResidueId(chain_id="Z", seq_num=1)
    )
    assert polymer_site is not None
    assert ligand_site is not None

    polymer_index = structure.constitution.residue_index(polymer_site.residue_id)
    ligand_index = structure.constitution.residue_index(ligand_site.residue_id)
    polymer_geometry = structure.residue_geometry(polymer_index).with_atom_geometry(
        "CA",
        structure.residue_geometry(polymer_index)
        .atom_geometry("CA")
        .with_position(Vec3(1.5, 0.0, 0.0)),
    )
    ligand_geometry = structure.residue_geometry(ligand_index).with_atom_geometry(
        "ZN",
        structure.residue_geometry(ligand_index)
        .atom_geometry("ZN")
        .with_position(Vec3(3.5, 0.0, 0.0)),
    )

    sequential = structure.with_updated_residue_facets(
        polymer_site,
        residue_geometry=polymer_geometry,
        formal_charge_by_atom_name=(("N", 1),),
    ).with_updated_residue_facets(
        ligand_site,
        residue_geometry=ligand_geometry,
        formal_charge_by_atom_name=(("ZN", 2),),
    )
    batched = structure.with_updated_residue_facets_batch(
        (
            (polymer_site, polymer_geometry, (("N", 1),)),
            (ligand_site, ligand_geometry, (("ZN", 2),)),
        )
    )

    assert batched.constitution == sequential.constitution
    assert batched.geometry == sequential.geometry
    assert batched.topology == sequential.topology
    assert batched.polymer_blueprint == sequential.polymer_blueprint
    assert batched.provenance == sequential.provenance


def test_with_updated_residue_facets_batch_rejects_duplicate_residues() -> None:
    """Batch replacement should reject ambiguous duplicate residue ids."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="batch-facet-duplicates",
    )
    residue_site = structure.constitution.residue_site_at(
        structure.constitution.residue_index(ResidueId(chain_id="A", seq_num=1))
    )
    residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_site.residue_id)
    )

    with pytest.raises(ModelInvariantError):
        structure.with_updated_residue_facets_batch(
            (
                (residue_site, residue_geometry, ()),
                (residue_site, residue_geometry, ()),
            )
        )


def test_with_updated_residue_geometries_preserves_non_geometry_facets() -> None:
    """Coordinate-only replacement should not rebuild non-geometry facets."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="geometry-only-update",
    )
    residue_id = ResidueId(chain_id="A", seq_num=1)
    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None

    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.residue_geometry(residue_index).with_atom_geometry(
        "CA",
        structure.residue_geometry(residue_index)
        .atom_geometry("CA")
        .with_position(Vec3(1.5, 0.0, 0.0)),
    )
    full_update = structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=(
            structure.residue_formal_charge_by_atom_name(residue_index)
        ),
    )
    geometry_only = structure.with_updated_residue_geometries(
        ((residue_id, residue_geometry),)
    )

    assert geometry_only.constitution is structure.constitution
    assert geometry_only.topology is structure.topology
    assert geometry_only.polymer_blueprint == structure.polymer_blueprint
    assert geometry_only.provenance == structure.provenance
    assert geometry_only.geometry == full_update.geometry


def test_with_updated_residue_geometries_rejects_atom_name_mismatch() -> None:
    """Coordinate-only replacement should preserve the constitution atom set."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="geometry-only-mismatch",
    )
    residue_id = ResidueId(chain_id="A", seq_num=1)
    residue_index = structure.constitution.residue_index(residue_id)

    with pytest.raises(ModelInvariantError):
        structure.with_updated_residue_geometries(
            (
                (
                    residue_id,
                    structure.residue_geometry(residue_index).without_atoms({"CA"}),
                ),
            )
        )
