"""Tests for chemistry-derived topology projections."""

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry.component.topology import (
    template_resolved_hydrogen_topology_bonds_for_new_atoms,
    template_resolved_topology_bonds_for_new_atoms,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
)


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


def test_template_hydrogen_topology_for_new_atoms_uses_static_anchors() -> None:
    """Template hydrogen anchors should become topology only for new H atoms."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    source = _single_ala_structure(
        residue_id=residue_id,
        atom_names=("N", "CA", "C", "O", "CB"),
    )
    target = _single_ala_structure(
        residue_id=residue_id,
        atom_names=("N", "CA", "C", "O", "CB", "HA", "HB1"),
    )

    bonds = template_resolved_hydrogen_topology_bonds_for_new_atoms(
        source_constitution=source.constitution,
        target_constitution=target.constitution,
        component_library=build_standard_component_library(),
    )

    assert _has_template_bond(target, bonds, residue_id, "CA", "HA")
    assert _has_template_bond(target, bonds, residue_id, "CB", "HB1")
    assert not _has_template_bond(target, bonds, residue_id, "N", "CA")


def test_template_hydrogen_topology_for_new_atoms_uses_rotatable_anchors() -> None:
    """Rotatable-H semantics still have explicit donor heavy-atom anchors."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    source = _single_ser_structure(
        residue_id=residue_id,
        atom_names=("N", "CA", "C", "O", "CB", "OG"),
    )
    target = _single_ser_structure(
        residue_id=residue_id,
        atom_names=("N", "CA", "C", "O", "CB", "OG", "HG"),
    )

    bonds = template_resolved_hydrogen_topology_bonds_for_new_atoms(
        source_constitution=source.constitution,
        target_constitution=target.constitution,
        component_library=build_standard_component_library(),
    )

    assert _has_template_bond(target, bonds, residue_id, "OG", "HG")


def _has_template_bond(
    structure: ProteinStructure,
    bonds: tuple[TopologyBond, ...],
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> bool:
    atom_index_1 = structure.constitution.atom_index(
        AtomRef(residue_id, atom_name_1)
    )
    atom_index_2 = structure.constitution.atom_index(
        AtomRef(residue_id, atom_name_2)
    )
    return any(
        {bond.atom_index_1, bond.atom_index_2} == {atom_index_1, atom_index_2}
        and bond.relationship_type is BondRelationshipType.COVALENT
        and bond.provenance is BondProvenance.TEMPLATE_RESOLVED
        for bond in bonds
    )


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


def _single_ser_structure(
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
                        component_id="SER",
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
    if atom_name.startswith("H") or "H" in atom_name:
        return "H"
    if atom_name == "N":
        return "N"
    if atom_name.startswith("O"):
        return "O"
    return "C"
