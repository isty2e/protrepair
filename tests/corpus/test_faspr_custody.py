"""Canonical chemistry and motion boundaries through the real packing backend."""

from pathlib import Path

import pytest

from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.io import read_structure
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.packing import (
    PackingScope,
    PackingSpec,
    execute_sidechain_packing,
)

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def prepared_1aho() -> ProteinStructure:
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    prepared = add_hydrogens(source).structure
    assert sum(atom.is_hydrogen() for atom in prepared.constitution.atom_slots) > 400
    assert any(
        atom is not None and atom.formal_charge != 0
        for atom in prepared.topology.atom_topologies
    )
    return prepared


@pytest.mark.parametrize("selection", ("frozen", "full_subset", "local_subset"))
def test_native_packing_preserves_current_chemistry_and_protected_geometry(
    prepared_1aho: ProteinStructure, selection: str
) -> None:
    source = prepared_1aho
    residues = tuple(source.constitution.iter_residues())
    selected = next(r.residue_id for r in residues if r.component_id == "LYS")
    spec = PackingSpec(
        backend_name="faspr",
        scope=PackingScope.LOCAL if selection == "local_subset" else PackingScope.FULL,
        mutable_residue_ids=None if selection == "frozen" else (selected,),
        frozen_residue_ids=(
            tuple(r.residue_id for r in residues) if selection == "frozen" else None
        ),
    )

    result = execute_sidechain_packing(source, spec)
    packed = result.packed_structure
    assert packed.provenance is source.provenance
    assert packed.polymer_blueprint == source.polymer_blueprint
    assert packed.topology.bonds == source.topology.bonds_for_constitution(
        source_constitution=source.constitution,
        target_constitution=packed.constitution,
    )
    for offset, atom in enumerate(packed.constitution.atom_slots):
        after_index = AtomIndex(offset)
        ref = packed.constitution.atom_ref_at(after_index)
        before_index = source.constitution.atom_index(ref)
        assert atom == source.constitution.atom_site_at(before_index)
        assert (
            packed.topology.atom_topologies[offset]
            == (source.topology.atom_topologies[before_index.value])
        )
        before = source.geometry.atom_geometry(before_index)
        after = packed.geometry.atom_geometry(after_index)
        assert (after.occupancy, after.b_factor, after.altloc) == (
            before.occupancy,
            before.b_factor,
            before.altloc,
        )
        if (
            selection == "frozen"
            or ref.residue_id != selected
            or atom.name in {"N", "CA", "C", "O", "OXT"}
            or atom.is_hydrogen()
        ):
            assert after == before

    for residue in residues:
        packed_residue = packed.constitution.residue_or_ligand(residue.residue_id)
        assert packed_residue is not None
        moved = any(
            source.geometry.atom_geometry(
                source.constitution.atom_index(AtomRef(residue.residue_id, atom.name))
            ).distance_to(
                packed.geometry.atom_geometry(
                    packed.constitution.atom_index(
                        AtomRef(residue.residue_id, atom.name)
                    )
                )
            )
            > 0.05
            for atom in residue.atom_sites
            if not atom.is_hydrogen()
        )
        assert tuple(a for a in packed_residue.atom_sites if a.is_hydrogen()) == (
            () if moved else tuple(a for a in residue.atom_sites if a.is_hydrogen())
        )
        invalidated = any(
            issue.kind is ValidationIssueKind.PACKING_INVALIDATED_HYDROGENS
            and issue.scope.targets_residue(residue.residue_id)
            for issue in result.issues
        )
        assert invalidated == moved

    if selection == "frozen":
        assert packed.constitution == source.constitution
        assert packed.geometry == source.geometry
        assert packed.topology == source.topology
        assert result.changed_residue_ids == ()
        assert result.issues == ()
