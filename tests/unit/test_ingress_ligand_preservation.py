"""Canonical ingress must not confuse polymer chains with source chain selection."""

from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.errors import StructureNormalizationError
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.ingress_policy import LigandHandling, StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    StructureTopology,
    TopologyBond,
)
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions


@pytest.fixture
def ligand_chain_structure() -> ProteinStructure:
    polymer = residue_payload(
        component_id="GLY",
        residue_id=ResidueId("A", 1),
        atoms=(atom_payload("CA", "C", Vec3(0, 0, 0), b_factor=0.0),),
    )
    ligand = residue_payload(
        component_id="LIG",
        residue_id=ResidueId("L", 8),
        is_hetero=True,
        atoms=(
            atom_payload("C1", "C", Vec3(3, 0, 0), b_factor=0.0),
            atom_payload("O1", "O", Vec3(4.4, 0, 0), b_factor=0.0),
        ),
    )
    structure = build_structure(
        chains=(chain_payload(chain_id="A", residues=(polymer,)),),
        ligands=(ligand,),
        source_format=FileFormat.PDB,
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    AtomIndex(1),
                    AtomIndex(2),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
                TopologyBond(
                    AtomIndex(0),
                    AtomIndex(1),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        ),
        provenance=structure.provenance,
    )


def test_keep_without_chain_selection_is_identity(
    ligand_chain_structure: ProteinStructure,
) -> None:
    policy = StructureNormalizationPolicy(ligand_handling=LigandHandling.KEEP)
    first = apply_structure_normalization_policy(ligand_chain_structure, policy=policy)
    second = apply_structure_normalization_policy(first, policy=policy)

    assert first is ligand_chain_structure
    assert second is first
    assert first.topology == ligand_chain_structure.topology


@pytest.mark.parametrize("canonical_input", [False, True])
def test_public_processing_keeps_ligand_only_chain(
    ligand_chain_structure: ProteinStructure,
    tmp_path: Path,
    canonical_input: bool,
) -> None:
    source = tmp_path / "ligand-chain.cif"
    source.write_text(write_structure_string(ligand_chain_structure, FileFormat.MMCIF))
    result = process_structure(
        ligand_chain_structure if canonical_input else source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
    )

    assert (
        result.structure.constitution.ligands
        == ligand_chain_structure.constitution.ligands
    )
    assert result.structure.geometry == ligand_chain_structure.geometry
    assert {bond.endpoint_pair() for bond in result.structure.topology.bonds} == {
        (AtomIndex(0), AtomIndex(1)),
        (AtomIndex(1), AtomIndex(2)),
    }


def test_reject_without_selection_sees_ligand_only_chain(
    ligand_chain_structure: ProteinStructure,
) -> None:
    with pytest.raises(StructureNormalizationError, match="unexpected ligand"):
        apply_structure_normalization_policy(
            ligand_chain_structure,
            policy=StructureNormalizationPolicy(ligand_handling=LigandHandling.REJECT),
        )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_public_source_selection_keeps_a_ligand_only_chain(
    ligand_chain_structure: ProteinStructure,
    tmp_path: Path,
    file_format: FileFormat,
) -> None:
    source = tmp_path / f"selected-ligand.{file_format.value}"
    source.write_text(write_structure_string(ligand_chain_structure, file_format))
    result = process_structure(
        source,
        selected_source_chain_ids=("L",),
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
    )

    assert result.structure.constitution.chains == ()
    assert result.structure.constitution.ligands == (
        ligand_chain_structure.constitution.ligands[0],
    )
    assert (
        result.structure.geometry.atom_geometries
        == (ligand_chain_structure.geometry.atom_geometries[1:])
    )
    assert tuple(bond.endpoint_pair() for bond in result.structure.topology.bonds) == (
        (AtomIndex(0), AtomIndex(1)),
    )


@pytest.mark.parametrize("handling", list(LigandHandling))
def test_explicit_polymer_chain_selection_excludes_other_ligand_chains(
    ligand_chain_structure: ProteinStructure,
    handling: LigandHandling,
) -> None:
    selected = apply_structure_normalization_policy(
        ligand_chain_structure,
        policy=StructureNormalizationPolicy(
            selected_chain_ids=("A",),
            ligand_handling=handling,
        ),
    )

    assert selected.constitution.chain_ids() == ("A",)
    assert selected.constitution.ligands == ()
    assert selected.topology.bonds == ()
    assert len(selected.geometry.atom_geometries) == 1


def test_drop_removes_orphan_ligand_atoms_and_bonds(
    ligand_chain_structure: ProteinStructure,
) -> None:
    dropped = apply_structure_normalization_policy(
        ligand_chain_structure,
        policy=StructureNormalizationPolicy(ligand_handling=LigandHandling.DROP),
    )

    assert dropped.constitution.ligands == ()
    assert dropped.topology.bonds == ()
    assert (
        dropped.geometry.atom_geometries
        == ligand_chain_structure.geometry.atom_geometries[:1]
    )


def test_keep_preserves_structure_with_no_polymer_chains(
    ligand_chain_structure: ProteinStructure,
) -> None:
    text = write_structure_string(ligand_chain_structure, FileFormat.MMCIF)
    only_ligand = read_structure_string(
        text,
        FileFormat.MMCIF,
        policy=StructureNormalizationPolicy(selected_chain_ids=("L",)),
    )
    assert only_ligand.constitution.chain_ids() == ()
    assert len(only_ligand.constitution.ligands) == 1

    normalized = apply_structure_normalization_policy(
        only_ligand,
        policy=StructureNormalizationPolicy(),
    )
    assert normalized is only_ligand
    assert normalized.topology.bonds[0].endpoint_pair() == (AtomIndex(0), AtomIndex(1))
