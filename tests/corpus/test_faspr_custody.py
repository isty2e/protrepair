"""Canonical chemistry and motion boundaries through the real packing backend."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.request_builders import (
    transform_requests,
    whole_structure_requested_goals,
)

from protrepair.api import process_structure
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.io import read_structure
from protrepair.state import HydrogenCoverageState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.packing import (
    PackingPlan,
    PackingScope,
    PackingSpec,
    execute_sidechain_packing,
)
from protrepair.transformer.packing.faspr.backend import (
    FasprExecutionInput,
    FasprPackingBackend,
    FasprRuntimeAssets,
    PackingBackendError,
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


@pytest.mark.parametrize("altloc", ["B", "b"])
def test_native_packing_observes_selected_non_a_atoms_without_losing_metadata(
    prepared_1aho: ProteinStructure, altloc: str
) -> None:
    base = prepared_1aho
    source = ProteinStructure.from_payload(
        constitution=base.constitution,
        geometry=StructureGeometry(
            constitution=base.constitution,
            atom_geometries=tuple(
                replace(geometry, altloc=altloc, occupancy=0.3)
                for geometry in base.geometry.atom_geometries
            ),
        ),
        topology=base.topology,
        polymer_blueprint=base.polymer_blueprint,
        provenance=base.provenance,
    )
    spec = PackingSpec(
        backend_name="faspr",
        frozen_residue_ids=tuple(
            residue.residue_id for residue in source.constitution.iter_residues()
        ),
    )
    result = execute_sidechain_packing(source, spec)
    assert result.packed_structure.constitution == source.constitution
    assert result.packed_structure.geometry == source.geometry
    assert result.packed_structure.topology == source.topology
    assert result.packed_structure.provenance is source.provenance
    assert result.changed_residue_ids == ()
    assert result.issues == ()


@pytest.mark.parametrize("scope", (PackingScope.LOCAL, PackingScope.FULL))
@pytest.mark.parametrize(
    ("fixed_id", "missing_names"),
    ((9, ("OD2",)), (30, ("CD", "CE", "NZ")), (50, ("NZ",))),
)
def test_raw_1aho_incomplete_fixed_residue_is_rejected_before_runtime_access(
    monkeypatch: pytest.MonkeyPatch,
    scope: PackingScope,
    fixed_id: int,
    missing_names: tuple[str, ...],
) -> None:
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    mutable_ids = tuple(
        ResidueId("A", index) for index in (1, 9, 30, 50) if index != fixed_id
    )
    plan = PackingPlan.from_inputs(
        source,
        PackingSpec(backend_name="faspr", scope=scope, mutable_residue_ids=mutable_ids),
    )

    def unexpected_runtime_access(self: FasprPackingBackend) -> FasprRuntimeAssets:
        pytest.fail("Incomplete fixed residues must be rejected before runtime access")

    monkeypatch.setattr(
        FasprPackingBackend, "runtime_assets", unexpected_runtime_access
    )
    with pytest.raises(PackingBackendError) as error:
        execute_sidechain_packing(source, plan.spec)

    assert type(error.value) is PackingBackendError
    message = str(error.value)
    assert f"fixed residue A:{fixed_id}" in message
    assert all(name in message for name in missing_names)
    assert "mutable_residue_ids" in message
    assert "complete heavy atoms" in message


@pytest.mark.parametrize("scope", (PackingScope.LOCAL, PackingScope.FULL))
def test_raw_1aho_incomplete_mutable_residues_are_admitted(
    scope: PackingScope,
) -> None:
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    plan = PackingPlan.from_inputs(
        source,
        PackingSpec(
            backend_name="faspr",
            scope=scope,
            mutable_residue_ids=tuple(ResidueId("A", i) for i in (9, 30, 50)),
        ),
    )
    execution_input = FasprExecutionInput.from_plan(plan)
    assert execution_input.structure.constitution.atom_slots == tuple(
        atom for atom in source.constitution.atom_slots if not atom.is_hydrogen()
    )
    assert execution_input.plan is plan


@pytest.mark.parametrize("scope", (PackingScope.LOCAL, PackingScope.FULL))
def test_raw_1aho_preflight_lists_all_incomplete_fixed_residues(
    scope: PackingScope,
) -> None:
    source = read_structure(Path("tests/fixtures/pdb/1aho.pdb"))
    spec = PackingSpec(
        backend_name="faspr", scope=scope, mutable_residue_ids=(ResidueId("A", 1),)
    )
    with pytest.raises(PackingBackendError) as error:
        execute_sidechain_packing(source, spec)

    message = str(error.value)
    assert all(f"fixed residue {token}" in message for token in ("A:9", "A:30", "A:50"))
    assert all(name in message for name in ("OD2", "CD", "CE", "NZ"))


@pytest.mark.parametrize("scope", (PackingScope.LOCAL, PackingScope.FULL))
@pytest.mark.parametrize("include_incomplete", (False, True))
def test_reference_packing_checks_original_fixed_scaffolds_before_completion(
    scope: PackingScope, include_incomplete: bool
) -> None:
    mutable_ids = tuple(
        ResidueId("A", index)
        for index in ((1, 9, 30, 50) if include_incomplete else (1,))
    )
    requests = transform_requests(
        reference_sidechain_packing=PackingSpec(
            backend_name="faspr", scope=scope, mutable_residue_ids=mutable_ids
        )
    )
    goals = whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
    source = Path("tests/fixtures/pdb/1aho.pdb")

    if not include_incomplete:
        with pytest.raises(PackingBackendError, match="fixed residue A:9"):
            process_structure(
                source, requested_goals=goals, transform_requests=requests
            )
        return

    result = process_structure(
        source, requested_goals=goals, transform_requests=requests
    )
    assert result.requested_goals_fully_satisfied() is True
    assert not result.has_errors()


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
