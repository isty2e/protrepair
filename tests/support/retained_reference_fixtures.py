"""Named nonstandard references as retained source-like fixtures."""

from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure, ResidueId
from protrepair.structure.provenance import FileFormat
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    residue_payload,
)

LIGAND = ResidueId("L", 600)


def reference_structure(
    component_id: str, *, with_h: bool, isotope: str = "H"
) -> ProteinStructure:
    record = build_bundled_nonstandard_registry().get(component_id)
    assert record is not None
    atoms = []
    for atom in record.atoms:
        if atom.is_hydrogen() and not with_h:
            continue
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(
                atom.atom_name,
                isotope if atom.is_hydrogen() else atom.element,
                Vec3(*atom.ideal_position),
                formal_charge=atom.formal_charge,
                b_factor=23.0,
            )
        )
    return build_bonded_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id=component_id, residue_id=LIGAND, atoms=tuple(atoms)
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
