from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.geometry import Vec3
from protrepair.structure import (
    AtomGeometry,
    ResidueGeometry,
)
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import (
    AtomIndex,
)


def test_atom_geometry_projects_coordinate_payload() -> None:
    _atom_site, atom_geometry, _formal_charge = atom_payload(
        name="CA",
        element="C",
        position=Vec3(1.0, 2.0, 3.0),
        occupancy=0.5,
        b_factor=12.0,
        altloc="A",
    )

    assert atom_geometry == AtomGeometry(
        position=Vec3(1.0, 2.0, 3.0),
        occupancy=0.5,
        b_factor=12.0,
        altloc="A",
    )


def test_structure_geometry_uses_constitution_slot_order() -> None:
    residue = residue_payload(
        component_id="SER",
        residue_id=ResidueId(chain_id="A", seq_num=10, insertion_code=None),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
        ),
    )
    structure = build_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
    )

    geometry = structure.geometry

    assert tuple(
        structure.constitution.atom_ref_at(AtomIndex(atom_index))
        for atom_index in range(len(structure.constitution.atom_slots))
    ) == (
        AtomRef(residue_id=residue[0].residue_id, atom_name="N"),
        AtomRef(residue_id=residue[0].residue_id, atom_name="CA"),
    )
    assert geometry.position(
        structure.constitution.atom_index(
            AtomRef(residue_id=residue[0].residue_id, atom_name="CA")
        )
    ) == Vec3(1.0, 0.0, 0.0)


def test_residue_geometry_keys_payload_by_atom_name() -> None:
    residue = residue_payload(
        component_id="SER",
        residue_id=ResidueId(chain_id="A", seq_num=10, insertion_code=None),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
        ),
    )

    assert residue[1] == ResidueGeometry(
        atoms_by_name={
            "N": AtomGeometry(position=Vec3(0.0, 0.0, 0.0)),
            "CA": AtomGeometry(position=Vec3(1.0, 0.0, 0.0)),
        },
    )


def test_residue_geometry_batches_normalized_atom_updates() -> None:
    """Batch updates should preserve order and use the last normalized name."""

    original_ca = AtomGeometry(position=Vec3(1.0, 0.0, 0.0))
    geometry = ResidueGeometry(
        atoms_by_name={
            "N": AtomGeometry(position=Vec3(0.0, 0.0, 0.0)),
            "CA": original_ca,
        }
    )
    updated_ca = AtomGeometry(position=Vec3(2.0, 0.0, 0.0))
    first_cb = AtomGeometry(position=Vec3(3.0, 0.0, 0.0))
    final_cb = AtomGeometry(position=Vec3(4.0, 0.0, 0.0))

    updated = geometry.with_atom_geometries(
        ((" ca ", updated_ca), ("cb", first_cb), ("CB", final_cb))
    )

    assert updated.atom_names() == ("N", "CA", "CB")
    assert updated.atom_geometry("CA") is updated_ca
    assert updated.atom_geometry("CB") is final_cb
    assert geometry.atom_geometry("CA") is original_ca
    assert not geometry.has_atom("CB")


def test_residue_geometry_empty_batch_preserves_identity() -> None:
    """An empty batch should retain the existing immutable geometry value."""

    geometry = ResidueGeometry(
        atoms_by_name={"N": AtomGeometry(position=Vec3(0.0, 0.0, 0.0))}
    )

    assert geometry.with_atom_geometries(()) is geometry


def test_structure_resolves_atom_geometry_from_geometry_payload() -> None:
    residue = residue_payload(
        component_id="SER",
        residue_id=ResidueId(chain_id="A", seq_num=10, insertion_code=None),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0), occupancy=0.5, b_factor=12.0),
        ),
    )
    structure = build_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
    )

    rebuilt_atom_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(
            AtomRef(residue_id=residue[0].residue_id, atom_name="CA")
        )
    )

    assert rebuilt_atom_geometry == AtomGeometry(
        position=Vec3(1.0, 0.0, 0.0),
        occupancy=0.5,
        b_factor=12.0,
    )
