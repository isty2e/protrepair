"""Movement and reporting for linked sidechains through packing execution."""

from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)
from tests.support.faspr_output import relabel_faspr_output
from tests.unit.test_faspr_backend import copy_template_faspr_executable

from protrepair.errors import PackingError
from protrepair.geometry import Vec3
from protrepair.io.pdb_projection import pdb_without_conect
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.packing import (
    PackingPlan,
    PackingScope,
    PackingSpec,
    execute_sidechain_packing,
)
from protrepair.transformer.packing.domain import PackingResult
from protrepair.transformer.packing.faspr.backend import (
    FasprPackingBackend,
    FasprRuntimeAssets,
    PackingBackendExecutionError,
)


def linked_structure(
    relationship: BondRelationshipType | None = BondRelationshipType.DISULFIDE,
    provenance: BondProvenance = BondProvenance.SOURCE_EXPLICIT,
    *,
    missing_cb: bool = False,
    left_atom: str = "SG",
    right_atom: str = "SG",
    partner: str = "polymer",
    third_component: str = "ALA",
) -> ProteinStructure:
    residues = []
    for number, component in ((1, "CYS"), (2, "CYS"), (3, third_component)):
        names = (
            ("N", "CA", "C", "O", "CB", "SG")
            if component == "CYS"
            else ("N", "CA", "C", "O", "CB")
        )
        if component == "GLY":
            names = ("N", "CA", "C", "O")
        if missing_cb and number == 1:
            names = tuple(name for name in names if name != "CB")
        residues.append(
            residue_payload(
                component_id=component,
                residue_id=ResidueId("A", number),
                atoms=tuple(
                    atom_payload(
                        name,
                        "S" if name == "SG" else name[0],
                        Vec3(5.0 * number + index, 0.5 * index, -0.2 * index),
                    )
                    for index, name in enumerate(names)
                ),
            )
        )
    ligands = ()
    if partner != "polymer":
        atom = "ZN" if partner == "metal" else "C1"
        ligands = (
            residue_payload(
                component_id="ZN" if partner == "metal" else "LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload(
                        atom, "ZN" if partner == "metal" else "C", Vec3(12, 2, 3)
                    ),
                ),
                is_hetero=True,
            ),
        )
    source = build_bonded_structure(
        chains=(chain_payload("A", tuple(residues)),),
        ligands=ligands,
        source_format=FileFormat.PDB,
    )
    if relationship is None:
        return source
    partner_ref = (
        AtomRef(ResidueId("A", 2), right_atom)
        if partner == "polymer"
        else AtomRef(ResidueId("L", 1), "ZN" if partner == "metal" else "C1")
    )
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(ResidueId("A", 1), left_atom)),
        source.constitution.atom_index(partner_ref),
        order=None if relationship is BondRelationshipType.UNKNOWN else 1,
        relationship_type=relationship,
        provenance=provenance,
        source_metadata=SourceBondMetadata(SourceBondRecordType.PDB_CONECT)
        if provenance is BondProvenance.SOURCE_EXPLICIT
        else None,
    )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, bond),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def pack_stub(
    tmp_path: Path,
    source: ProteinStructure,
    native: ProteinStructure,
    spec: PackingSpec,
) -> PackingResult:
    executable = copy_template_faspr_executable(tmp_path)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    (tmp_path / "packed-output.pdb").write_text(
        relabel_faspr_output(pdb_without_conect(native), source), encoding="utf-8"
    )
    return FasprPackingBackend(executable_path=executable).pack(
        PackingPlan.from_inputs(source, spec)
    )


def test_all_protected_request_returns_original_before_backend_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = linked_structure()

    def forbidden(self: FasprPackingBackend) -> FasprRuntimeAssets:
        raise AssertionError("all-protected packing must not resolve native assets")

    def forbidden_directory(*, prefix: str) -> None:
        raise AssertionError("all-protected packing must not create temporary files")

    monkeypatch.setattr(FasprPackingBackend, "runtime_assets", forbidden)
    monkeypatch.setattr(
        "protrepair.transformer.packing.faspr.backend.TemporaryDirectory",
        forbidden_directory,
    )
    result = execute_sidechain_packing(
        source,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(ResidueId("A", 1), ResidueId("A", 2)),
        ),
    )
    assert result.packed_structure is source
    assert not result.changed_residue_ids
    assert [(issue.kind.value, issue.residue_id) for issue in result.issues] == [
        ("packing_preserved_linked_sidechain", ResidueId("A", 1)),
        ("packing_preserved_linked_sidechain", ResidueId("A", 2)),
    ]


def test_mixed_request_preserves_links_and_executes_other_residues(
    tmp_path: Path,
) -> None:
    source = linked_structure()
    target = ResidueId("A", 3)
    geometry = source.geometry.residue_geometry(
        constitution=source.constitution,
        residue_index=source.constitution.residue_index(target),
    )
    native = source.with_updated_residue_geometries(
        (
            (
                target,
                geometry.with_atom_geometry(
                    "CB", geometry.atom_geometry("CB").with_position(Vec3(25, 2, 3))
                ),
            ),
        )
    )
    result = pack_stub(tmp_path, source, native, PackingSpec(backend_name="faspr"))
    assert result.changed_residue_ids == (target,)
    assert [
        issue.residue_id
        for issue in result.issues
        if issue.kind.value == "packing_preserved_linked_sidechain"
    ] == [ResidueId("A", 1), ResidueId("A", 2)]
    for rid in (ResidueId("A", 1), ResidueId("A", 2)):
        index = source.constitution.residue_index(rid)
        assert result.packed_structure.geometry.residue_geometry(
            constitution=result.packed_structure.constitution, residue_index=index
        ) == source.geometry.residue_geometry(
            constitution=source.constitution, residue_index=index
        )


@pytest.mark.parametrize(
    "relationship,partner",
    (
        (BondRelationshipType.COVALENT, "ligand"),
        (BondRelationshipType.DISULFIDE, "polymer"),
        (BondRelationshipType.METAL_COORDINATION, "metal"),
        (BondRelationshipType.UNKNOWN, "metal"),
    ),
)
@pytest.mark.parametrize("provenance", tuple(BondProvenance))
def test_link_protection_covers_relationships_and_both_evidence_families(
    relationship: BondRelationshipType,
    partner: str,
    provenance: BondProvenance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = linked_structure(relationship, provenance, partner=partner)

    def forbidden(self: FasprPackingBackend) -> FasprRuntimeAssets:
        raise AssertionError("protected request reached native setup")

    monkeypatch.setattr(FasprPackingBackend, "runtime_assets", forbidden)
    result = execute_sidechain_packing(
        source,
        PackingSpec(backend_name="faspr", mutable_residue_ids=(ResidueId("A", 1),)),
    )
    assert result.packed_structure is source
    assert [(issue.kind.value, issue.residue_id) for issue in result.issues] == [
        ("packing_preserved_linked_sidechain", ResidueId("A", 1))
    ]


@pytest.mark.parametrize("relationship", (None, BondRelationshipType.HYDROGEN_BOND))
def test_unlinked_and_hydrogen_bonded_sidechains_still_pack(
    relationship: BondRelationshipType | None, tmp_path: Path
) -> None:
    source = linked_structure(relationship)
    result = pack_stub(tmp_path, source, source, PackingSpec(backend_name="faspr"))
    assert result.issues == ()


def test_only_the_nonbackbone_endpoint_is_protected(tmp_path: Path) -> None:
    source = linked_structure(BondRelationshipType.COVALENT, left_atom="N")
    result = pack_stub(tmp_path, source, source, PackingSpec(backend_name="faspr"))
    assert [(issue.kind.value, issue.residue_id) for issue in result.issues] == [
        ("packing_preserved_linked_sidechain", ResidueId("A", 2))
    ]


def test_backbone_only_connections_do_not_exclude_sidechains(tmp_path: Path) -> None:
    source = linked_structure(
        BondRelationshipType.COVALENT, left_atom="N", right_atom="C"
    )
    result = pack_stub(tmp_path, source, source, PackingSpec(backend_name="faspr"))
    assert result.issues == ()


def test_proximity_without_topology_does_not_protect_a_sidechain(
    tmp_path: Path,
) -> None:
    source = linked_structure(None)
    first, second = ResidueId("A", 1), ResidueId("A", 2)
    left = source.geometry.residue_geometry(
        constitution=source.constitution,
        residue_index=source.constitution.residue_index(first),
    )
    right = source.geometry.residue_geometry(
        constitution=source.constitution,
        residue_index=source.constitution.residue_index(second),
    )
    right = right.with_atom_geometry(
        "SG",
        right.atom_geometry("SG").with_position(
            left.position("SG").with_offset(2.03, 0, 0)
        ),
    )
    source = source.with_updated_residue_geometries(((second, right),))
    moved = left.with_atom_geometry(
        "SG",
        left.atom_geometry("SG").with_position(
            left.position("SG").with_offset(0, 1, 0)
        ),
    )
    native = source.with_updated_residue_geometries(((first, moved),))
    result = pack_stub(tmp_path, source, native, PackingSpec(backend_name="faspr"))
    assert first in result.changed_residue_ids
    assert result.issues == ()


def test_native_cannot_move_an_automatically_protected_sidechain(
    tmp_path: Path,
) -> None:
    source = linked_structure()
    rid = ResidueId("A", 1)
    geometry = source.geometry.residue_geometry(
        constitution=source.constitution,
        residue_index=source.constitution.residue_index(rid),
    )
    native = source.with_updated_residue_geometries(
        (
            (
                rid,
                geometry.with_atom_geometry(
                    "SG", geometry.atom_geometry("SG").with_position(Vec3(50, 50, 50))
                ),
            ),
        )
    )
    with pytest.raises(PackingBackendExecutionError, match="protected atom"):
        pack_stub(tmp_path, source, native, PackingSpec(backend_name="faspr"))


def test_incomplete_linked_sidechain_is_rebuilt_with_a_scoped_warning(
    tmp_path: Path,
) -> None:
    source = linked_structure(missing_cb=True)
    native = linked_structure()
    result = pack_stub(tmp_path, source, native, PackingSpec(backend_name="faspr"))
    assert (
        result.packed_structure.constitution.resolve_atom_index(
            AtomRef(ResidueId("A", 1), "CB")
        )
        is not None
    )
    assert [(issue.kind.value, issue.residue_id) for issue in result.issues] == [
        ("packing_preserved_linked_sidechain", ResidueId("A", 2)),
        ("packing_rebuilt_linked_sidechain", ResidueId("A", 1)),
    ]
    refs = (AtomRef(ResidueId("A", 1), "SG"), AtomRef(ResidueId("A", 2), "SG"))
    before = source.topology.bond_between(
        *(source.constitution.atom_index(ref) for ref in refs)
    )
    after = result.packed_structure.topology.bond_between(
        *(result.packed_structure.constitution.atom_index(ref) for ref in refs)
    )
    assert before is not None and after is not None
    assert (
        after.relationship_type,
        after.order,
        after.provenance,
        after.source_metadata,
    ) == (
        before.relationship_type,
        before.order,
        before.provenance,
        before.source_metadata,
    )


@pytest.mark.parametrize("subset", (False, True))
def test_sequence_mapping_survives_removing_protected_residues(
    tmp_path: Path, subset: bool
) -> None:
    source = linked_structure()
    native = linked_structure(third_component="GLY")
    result = pack_stub(
        tmp_path,
        source,
        native,
        PackingSpec(
            backend_name="faspr",
            mutable_residue_ids=(
                ResidueId("A", 3),
                ResidueId("A", 1),
                ResidueId("A", 2),
            )
            if subset
            else None,
            target_sequence="GCC" if subset else "CCG",
        ),
    )
    assert (
        result.packed_structure.constitution.chain("A")
        .residue(ResidueId("A", 3))
        .component_id
        == "GLY"
    )
    assert result.changed_residue_ids == (ResidueId("A", 3),)
    assert len(result.issues) == 2


@pytest.mark.parametrize(
    "sequence,error", (("A", PackingBackendExecutionError), ("X", PackingError))
)
def test_protection_does_not_silently_cancel_mutations_or_invalid_sequences(
    sequence: str, error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(self: FasprPackingBackend) -> FasprRuntimeAssets:
        raise AssertionError("invalid request reached native setup")

    monkeypatch.setattr(FasprPackingBackend, "runtime_assets", forbidden)
    with pytest.raises(error):
        execute_sidechain_packing(
            linked_structure(),
            PackingSpec(
                backend_name="faspr",
                mutable_residue_ids=(ResidueId("A", 1),),
                target_sequence=sequence,
            ),
        )


def test_mutation_of_an_explicitly_frozen_residue_is_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(self: FasprPackingBackend) -> FasprRuntimeAssets:
        raise AssertionError("frozen mutation reached native setup")

    monkeypatch.setattr(FasprPackingBackend, "runtime_assets", forbidden)
    with pytest.raises(PackingError, match="frozen residues.*A:3"):
        execute_sidechain_packing(
            linked_structure(),
            PackingSpec(
                backend_name="faspr",
                target_sequence="CCG",
                frozen_residue_ids=(ResidueId("A", 3),),
            ),
        )
