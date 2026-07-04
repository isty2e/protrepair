"""Tests for chemistry-derived topology projections."""

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry.component.topology import (
    template_resolved_topology_bonds_for_new_atoms,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import BondProvenance, BondRelationshipType


def test_template_topology_for_new_atoms_only_emits_new_endpoint_bonds() -> None:
    """Delta projection should not reseed every present template bond."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    source = _single_ala_structure(
        residue_id=residue_id,
        atom_names=("N", "CA", "C", "O"),
    )
    target = _single_ala_structure(
        residue_id=residue_id,
        atom_names=("N", "CA", "C", "O", "CB"),
    )

    bonds = template_resolved_topology_bonds_for_new_atoms(
        source_constitution=source.constitution,
        target_constitution=target.constitution,
        component_library=build_standard_component_library(),
    )

    assert len(bonds) == 1
    bond = bonds[0]
    assert {
        target.constitution.atom_ref_at(bond.atom_index_1),
        target.constitution.atom_ref_at(bond.atom_index_2),
    } == {AtomRef(residue_id, "CA"), AtomRef(residue_id, "CB")}
    assert bond.relationship_type is BondRelationshipType.COVALENT
    assert bond.provenance is BondProvenance.TEMPLATE_RESOLVED


def _single_ala_structure(
    *,
    residue_id: ResidueId,
    atom_names: tuple[str, ...],
) -> ProteinStructure:
    return build_structure(
        chains=(
            chain_payload(
                residue_id.chain_id,
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=residue_id,
                        atoms=tuple(
                            atom_payload(atom_name, _element(atom_name), Vec3(0, 0, 0))
                            for atom_name in atom_names
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _element(atom_name: str) -> str:
    if atom_name == "N":
        return "N"
    if atom_name.startswith("O"):
        return "O"
    return "C"
