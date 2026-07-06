"""Unit and smoke tests for the FASPR side-chain packing backend."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

import protrepair.transformer.packing.faspr.backend as faspr_backend
import protrepair.transformer.packing.faspr.paths as faspr_paths
from protrepair.diagnostics import IssueSeverity, ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure, write_structure_string
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.packing import (
    PackingMode,
    PackingPlan,
    PackingScope,
    PackingSpec,
)
from protrepair.transformer.packing.faspr.backend import (
    FasprPackingBackend,
    PackingBackendExecutionError,
)
from protrepair.transformer.packing.faspr.paths import faspr_executable_path
from protrepair.workflow.contracts import StructureIngressOptions

FASPR_FIXTURE_PATH = Path("tests/fixtures/pdb/1aho_faspr_input.pdb")


def test_faspr_backend_declares_expected_capabilities() -> None:
    """FASPR should advertise the expected fixed-backbone packing surface."""

    capabilities = FasprPackingBackend().capabilities()

    assert capabilities.supports_full_structure_packing
    assert capabilities.supports_local_packing
    assert capabilities.supports_partial_sequence
    assert not capabilities.supports_refinement
    assert not capabilities.supports_noncanonical_components
    assert capabilities.deterministic_given_same_inputs


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, math.inf, math.nan])
def test_faspr_backend_rejects_invalid_timeout(timeout_seconds: float) -> None:
    """FASPR backend timeouts must be positive finite values."""

    with pytest.raises(ValueError, match="timeout_seconds"):
        FasprPackingBackend(timeout_seconds=timeout_seconds)


@pytest.mark.parametrize(
    "timeout_seconds",
    [cast(float, True), cast(float, "5")],
)
def test_faspr_backend_rejects_non_numeric_timeout(
    timeout_seconds: float,
) -> None:
    """Runtime timeout validation should reject bools and numeric strings."""

    with pytest.raises(TypeError, match="timeout_seconds"):
        FasprPackingBackend(timeout_seconds=timeout_seconds)


def test_faspr_missing_asset_directory_error_points_to_install_or_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing bundled assets should explain installed-package availability."""

    missing_asset_dir = tmp_path / "missing-faspr-assets"
    monkeypatch.setattr(
        faspr_paths,
        "candidate_binary_directories",
        lambda: (missing_asset_dir,),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        faspr_paths.faspr_binary_directory()

    message = str(exc_info.value)
    assert "current protrepair import environment" in message
    assert "built protrepair package or wheel" in message
    assert "explicit executable_path" in message


def test_faspr_missing_executable_error_points_to_install_or_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An asset dir without FASPR should still point to the executable override."""

    asset_dir = tmp_path / "faspr-assets"
    asset_dir.mkdir()
    (asset_dir / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        faspr_paths,
        "candidate_binary_directories",
        lambda: (asset_dir,),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        faspr_paths.faspr_executable_path()

    message = str(exc_info.value)
    assert str(asset_dir) in message
    assert "built protrepair package or wheel" in message
    assert "explicit executable_path" in message


def test_faspr_backend_builds_local_sequence_override_with_fake_executable(
    tmp_path: Path,
) -> None:
    """Local packing should translate mutable/fixed residues into FASPR casing."""

    log_path = tmp_path / "sequence.log"
    executable_path = tmp_path / "FASPR"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "set -eu",
                'input_path=""',
                'output_path=""',
                'sequence_path=""',
                'while [ "$#" -gt 0 ]; do',
                '  case "$1" in',
                '    -i) input_path="$2"; shift 2 ;;',
                '    -o) output_path="$2"; shift 2 ;;',
                '    -s) sequence_path="$2"; shift 2 ;;',
                "    *) shift ;;",
                "  esac",
                "done",
                'cp "$input_path" "$output_path"',
                'if [ -n "$sequence_path" ]; then',
                f'  cat "$sequence_path" > "{log_path}"',
                "else",
                f'  : > "{log_path}"',
                "fi",
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")

    structure = build_test_structure()
    mutable_residue_id = structure.constitution.chain("A").residues[1].residue_id
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(
            backend_name="faspr",
            mode=PackingMode.PACK,
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(mutable_residue_id,),
            target_sequence="V",
        ),
    )

    result = FasprPackingBackend(executable_path=executable_path).pack(plan)

    assert log_path.read_text(encoding="utf-8").strip() == "aVy"
    assert result.backend_name == "faspr"
    assert result.changed_residue_ids == ()
    assert result.packed_structure.chain_ids() == ("A",)


def test_faspr_backend_launches_relative_executable_path_with_sibling_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relative executable overrides should resolve before cwd-based launch."""

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    executable_path = copy_input_faspr_executable(asset_dir)
    (asset_dir / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend(
        executable_path=Path("assets") / executable_path.name
    ).pack(plan)

    assert result.backend_name == "faspr"
    assert result.packed_structure.chain_ids() == ("A",)


def test_faspr_backend_uses_symlink_path_for_sibling_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symlink executable overrides should validate assets beside the symlink."""

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_executable_path = copy_input_faspr_executable(target_dir)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    symlink_path = asset_dir / "FASPR"
    try:
        symlink_path.symlink_to(target_executable_path)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    (asset_dir / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend(executable_path=Path("assets") / "FASPR").pack(plan)

    assert result.backend_name == "faspr"
    assert result.packed_structure.chain_ids() == ("A",)


def test_faspr_backend_wraps_missing_executable_override(
    tmp_path: Path,
) -> None:
    """Missing executable overrides should not leak raw FileNotFoundError."""

    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(PackingBackendExecutionError) as exc_info:
        FasprPackingBackend(executable_path=tmp_path / "missing-FASPR").pack(plan)

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)
    assert "runtime assets are unavailable" in str(exc_info.value)
    assert "missing-FASPR" in str(exc_info.value)


def test_faspr_backend_wraps_missing_sibling_rotamer_library(
    tmp_path: Path,
) -> None:
    """Missing sibling rotamer assets should not leak raw FileNotFoundError."""

    executable_path = copy_input_faspr_executable(tmp_path)
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(PackingBackendExecutionError) as exc_info:
        FasprPackingBackend(executable_path=executable_path).pack(plan)

    assert isinstance(exc_info.value.__cause__, faspr_backend.PackingBackendError)
    assert "runtime assets are unavailable" in str(exc_info.value)
    assert "dun2010bbdep.bin" in str(exc_info.value)


def test_faspr_backend_caches_runtime_asset_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated calls on one backend should not repeat asset discovery."""

    executable_path = copy_input_faspr_executable(tmp_path)
    rotamer_library_path = tmp_path / "dun2010bbdep.bin"
    rotamer_library_path.write_text("stub", encoding="utf-8")
    resolve_calls = 0
    validate_calls = 0

    def fake_resolve_faspr_executable_path(
        executable_path_override: Path | None,
    ) -> Path:
        nonlocal resolve_calls
        resolve_calls += 1
        assert executable_path_override == executable_path
        return executable_path.absolute()

    def fake_validate_rotamer_library_near(resolved_path: Path) -> Path:
        nonlocal validate_calls
        validate_calls += 1
        assert resolved_path == executable_path.absolute()
        return rotamer_library_path

    monkeypatch.setattr(
        faspr_backend,
        "resolve_faspr_executable_path",
        fake_resolve_faspr_executable_path,
    )
    monkeypatch.setattr(
        faspr_backend,
        "validate_rotamer_library_near",
        fake_validate_rotamer_library_near,
    )
    backend = FasprPackingBackend(executable_path=executable_path)
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    first_result = backend.pack(plan)
    second_result = backend.pack(plan)

    assert first_result.backend_name == "faspr"
    assert second_result.backend_name == "faspr"
    assert resolve_calls == 1
    assert validate_calls == 1


def test_faspr_backend_cached_runtime_assets_still_detect_missing_rotamer(
    tmp_path: Path,
) -> None:
    """Cached paths should not hide assets removed after first use."""

    executable_path = copy_input_faspr_executable(tmp_path)
    rotamer_library_path = tmp_path / "dun2010bbdep.bin"
    rotamer_library_path.write_text("stub", encoding="utf-8")
    backend = FasprPackingBackend(executable_path=executable_path)
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    backend.pack(plan)
    rotamer_library_path.unlink()

    with pytest.raises(PackingBackendExecutionError) as exc_info:
        backend.pack(plan)

    assert "runtime assets are unavailable" in str(exc_info.value)
    assert "dun2010bbdep.bin" in str(exc_info.value)


def test_faspr_backend_converts_subprocess_timeout(
    tmp_path: Path,
) -> None:
    """A hanging FASPR subprocess should become a typed backend failure."""

    executable_path = tmp_path / "FASPR"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "sleep 5",
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(PackingBackendExecutionError) as exc_info:
        FasprPackingBackend(
            executable_path=executable_path,
            timeout_seconds=0.05,
        ).pack(plan)

    error_message = str(exc_info.value)
    assert "timed out" in error_message
    assert "0.05" in error_message
    assert str(executable_path) in error_message


def test_faspr_backend_wraps_launch_permission_error(
    tmp_path: Path,
) -> None:
    """Launch permission failures should stay inside backend error taxonomy."""

    executable_path = tmp_path / "FASPR"
    executable_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable_path.chmod(0o644)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(PackingBackendExecutionError) as exc_info:
        FasprPackingBackend(executable_path=executable_path).pack(plan)

    assert isinstance(exc_info.value.__cause__, PermissionError)
    error_message = str(exc_info.value)
    assert "could not start" in error_message
    assert str(executable_path) in error_message


def test_faspr_backend_wraps_launch_os_error_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executable races such as ENOEXEC/ENOENT should not leak raw OSError."""

    executable_path = copy_input_faspr_executable(tmp_path)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    def raise_launch_error(*args: object, **kwargs: object) -> object:
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(faspr_backend.subprocess, "run", raise_launch_error)

    with pytest.raises(PackingBackendExecutionError) as exc_info:
        FasprPackingBackend(executable_path=executable_path).pack(plan)

    assert isinstance(exc_info.value.__cause__, OSError)
    error_message = str(exc_info.value)
    assert "could not start" in error_message
    assert "Exec format error" in error_message
    assert str(executable_path) in error_message


def test_faspr_backend_bounds_subprocess_failure_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FASPR failure diagnostics should not capture unbounded stderr/stdout."""

    executable_path = tmp_path / "FASPR"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "set -eu",
                "i=0",
                'while [ "$i" -lt 64 ]; do',
                '  printf "x" >&2',
                "  i=$((i + 1))",
                "done",
                "exit 7",
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    monkeypatch.setattr(faspr_backend, "MAX_FASPR_CAPTURED_OUTPUT_BYTES", 16)
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(PackingBackendExecutionError) as exc_info:
        FasprPackingBackend(executable_path=executable_path).pack(plan)

    error_message = str(exc_info.value)
    assert "exit code 7" in error_message
    assert "x" * 16 in error_message
    assert "x" * 17 not in error_message
    assert "truncated after 16 bytes" in error_message


def test_faspr_backend_preserves_surviving_topology_bonds_with_fake_executable(
    tmp_path: Path,
) -> None:
    """Packing merge paths should remap topology bonds instead of dropping them."""

    executable_path = tmp_path / "FASPR"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "set -eu",
                'input_path=""',
                'output_path=""',
                'while [ "$#" -gt 0 ]; do',
                '  case "$1" in',
                '    -i) input_path="$2"; shift 2 ;;',
                '    -o) output_path="$2"; shift 2 ;;',
                "    *) shift ;;",
                "  esac",
                "done",
                'cp "$input_path" "$output_path"',
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")

    structure = build_test_structure()
    bonded_structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=AtomIndex(0),
                    atom_index_2=AtomIndex(1),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                        source_id="backbone",
                    ),
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    plan = PackingPlan.from_inputs(
        bonded_structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend(executable_path=executable_path).pack(plan)

    assert len(result.packed_structure.topology.bonds) == 1
    bond = result.packed_structure.topology.bonds[0]
    assert bond.endpoint_pair() == (AtomIndex(0), AtomIndex(1))
    assert bond.source_metadata is not None
    assert bond.source_metadata.source_id == "backbone"


def test_faspr_backend_restores_hydrogens_when_heavy_atoms_are_unchanged(
    tmp_path: Path,
) -> None:
    """Unchanged packed residues should keep original H coordinates and bonds."""

    executable_path = copy_input_faspr_executable(tmp_path)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    structure = build_hydrogenated_test_structure()
    residue_id = structure.constitution.chain("A").residues[0].residue_id
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend(executable_path=executable_path).pack(plan)
    packed_residue = result.packed_structure.constitution.chain("A").residues[0]

    assert packed_residue.atom_site_names() == (
        "N",
        "H",
        "CA",
        "HA",
        "C",
        "O",
        "CB",
        "HB1",
    )
    assert result.issues == ()
    original_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    packed_geometry = result.packed_structure.geometry.residue_geometry(
        constitution=result.packed_structure.constitution,
        residue_index=result.packed_structure.constitution.residue_index(residue_id),
    )
    assert (
        original_geometry.atom_geometry("HA").distance_to(
            packed_geometry.atom_geometry("HA")
        )
        == 0.0
    )
    n_index = result.packed_structure.constitution.resolve_atom_index(
        AtomRef(residue_id=residue_id, atom_name="N")
    )
    h_index = result.packed_structure.constitution.resolve_atom_index(
        AtomRef(residue_id=residue_id, atom_name="H")
    )
    assert n_index is not None
    assert h_index is not None
    assert result.packed_structure.topology.bond_between(n_index, h_index)


def test_faspr_backend_drops_hydrogens_when_fixed_residue_heavy_atoms_move(
    tmp_path: Path,
) -> None:
    """Fixed labels must not preserve stale H when FASPR moves heavy atoms."""

    executable_path = copy_template_faspr_executable(tmp_path)
    output_template_path = tmp_path / "packed-output.pdb"
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    structure = build_hydrogenated_test_structure()
    residue_id = structure.constitution.chain("A").residues[0].residue_id
    output_template_path.write_text(
        write_structure_string(
            build_heavy_only_packed_structure(
                first_residue_atom_positions={"CB": Vec3(9.0, 9.0, 9.0)}
            ),
            FileFormat.PDB,
        ),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.FULL,
            frozen_residue_ids=(residue_id,),
        ),
    )

    result = FasprPackingBackend(executable_path=executable_path).pack(plan)
    packed_residue = result.packed_structure.constitution.chain("A").residues[0]

    assert "H" not in packed_residue.atom_site_names()
    assert "HA" not in packed_residue.atom_site_names()
    assert "HB1" not in packed_residue.atom_site_names()
    assert (
        result.packed_structure.constitution.resolve_atom_index(
            AtomRef(residue_id=residue_id, atom_name="H")
        )
        is None
    )
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.kind is ValidationIssueKind.PACKING_INVALIDATED_HYDROGENS
    assert issue.severity is IssueSeverity.WARNING
    assert issue.scope.targets_residue(residue_id)
    assert "FASPR invalidated polymer hydrogens" in issue.message


def test_faspr_backend_preserves_and_drops_hydrogens_per_residue(
    tmp_path: Path,
) -> None:
    """H merge decisions are residue-local, not all-or-nothing."""

    executable_path = copy_template_faspr_executable(tmp_path)
    output_template_path = tmp_path / "packed-output.pdb"
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    structure = build_hydrogenated_test_structure(residue_count=2)
    first_residue_id = structure.constitution.chain("A").residues[0].residue_id
    second_residue_id = structure.constitution.chain("A").residues[1].residue_id
    output_template_path.write_text(
        write_structure_string(
            build_heavy_only_packed_structure(
                residue_count=2,
                atom_positions_by_residue_seq_num={2: {"CB": Vec3(9.0, 9.0, 9.0)}},
            ),
            FileFormat.PDB,
        ),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend(executable_path=executable_path).pack(plan)

    assert (
        result.packed_structure.constitution.resolve_atom_index(
            AtomRef(residue_id=first_residue_id, atom_name="H")
        )
        is not None
    )
    assert (
        result.packed_structure.constitution.resolve_atom_index(
            AtomRef(residue_id=second_residue_id, atom_name="H")
        )
        is None
    )
    assert len(result.issues) == 1
    assert result.issues[0].kind is ValidationIssueKind.PACKING_INVALIDATED_HYDROGENS
    assert result.issues[0].scope.targets_residue(second_residue_id)
    assert not result.issues[0].scope.targets_residue(first_residue_id)


def test_faspr_backend_drops_hydrogens_when_heavy_atom_charges_change(
    tmp_path: Path,
) -> None:
    """Heavy-atom charge changes should invalidate preserved hydrogens."""

    executable_path = copy_template_faspr_executable(tmp_path)
    output_template_path = tmp_path / "packed-output.pdb"
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    structure = build_charged_hydrogenated_test_structure()
    residue_id = structure.constitution.chain("A").residues[0].residue_id
    output_template_path.write_text(
        write_structure_string(build_heavy_only_packed_structure(), FileFormat.PDB),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend(executable_path=executable_path).pack(plan)

    assert (
        result.packed_structure.constitution.resolve_atom_index(
            AtomRef(residue_id=residue_id, atom_name="H")
        )
        is None
    )
    assert result.changed_residue_ids == (residue_id,)
    assert len(result.issues) == 1
    assert result.issues[0].kind is ValidationIssueKind.PACKING_INVALIDATED_HYDROGENS


def test_faspr_backend_smoke_runs_packaged_binary() -> None:
    """The packaged FASPR executable should pack a representative fixture."""

    if not FASPR_FIXTURE_PATH.exists():
        pytest.skip("FASPR input fixture is unavailable")

    try:
        faspr_executable_path()
    except FileNotFoundError:
        pytest.skip("packaged FASPR executable is unavailable")

    structure = read_structure(
        FASPR_FIXTURE_PATH,
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    result = FasprPackingBackend().pack(plan)

    assert result.backend_name == "faspr"
    assert result.packed_structure.chain_ids() == structure.chain_ids()
    assert sum(
        len(chain_site.residues)
        for chain_site in result.packed_structure.constitution.chains
    ) == sum(len(chain_site.residues) for chain_site in structure.constitution.chains)
    assert (
        result.packed_structure.constitution.ligands == structure.constitution.ligands
    )


def test_faspr_backend_rejects_unexpected_ligand_output(
    tmp_path: Path,
) -> None:
    """FASPR output normalization should reject unexpected hetero residues."""

    executable_path = tmp_path / "FASPR"
    output_template_path = tmp_path / "unexpected-output.pdb"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "set -eu",
                'output_path=""',
                'while [ "$#" -gt 0 ]; do',
                '  case "$1" in',
                '    -o) output_path="$2"; shift 2 ;;',
                "    *) shift ;;",
                "  esac",
                "done",
                f'cp "{output_template_path}" "$output_path"',
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")

    input_structure = build_test_structure()
    output_template_path.write_text(
        write_structure_string(
            build_structure(
                chains=tuple(
                    chain_payload(
                        chain_site.chain_id,
                        tuple(
                            (
                                residue_site,
                                residue_geometry,
                                input_structure.topology.residue_formal_charge_by_atom_name(
                                    constitution=input_structure.constitution,
                                    residue_index=input_structure.constitution.residue_index(
                                        residue_site.residue_id
                                    ),
                                ),
                            )
                            for residue_site in chain_site.residues
                            for residue_geometry in (
                                input_structure.geometry.residue_geometry(
                                    constitution=input_structure.constitution,
                                    residue_index=input_structure.constitution.residue_index(
                                        residue_site.residue_id
                                    ),
                                ),
                            )
                        ),
                    )
                    for chain_site in input_structure.constitution.chains
                ),
                ligands=(
                    build_residue(
                        "FAD",
                        "A",
                        99,
                        ("C1", "N1", "O1"),
                        is_hetero=True,
                    ),
                ),
                source_format=FileFormat.PDB,
                source_name="faspr-unexpected-ligand-output",
            ),
            FileFormat.PDB,
        ),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        input_structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(
        PackingBackendExecutionError,
        match="polymer-only structure",
    ):
        FasprPackingBackend(executable_path=executable_path).pack(plan)


def test_faspr_backend_wraps_unknown_polymer_output_residue_id(
    tmp_path: Path,
) -> None:
    """Unexpected FASPR polymer residue ids should remain backend errors."""

    executable_path = copy_template_faspr_executable(tmp_path)
    output_template_path = tmp_path / "packed-output.pdb"
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    output_template_path.write_text(
        write_structure_string(
            build_structure(
                chains=(
                    chain_payload(
                        "A",
                        (
                            build_residue(
                                "ALA",
                                "A",
                                999,
                                ("N", "CA", "C", "O", "CB"),
                            ),
                        ),
                    ),
                ),
                source_format=FileFormat.PDB,
                source_name="faspr-unknown-polymer-output",
            ),
            FileFormat.PDB,
        ),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(
        PackingBackendExecutionError,
        match="unknown residue identifier",
    ):
        FasprPackingBackend(executable_path=executable_path).pack(plan)


def test_faspr_backend_rejects_dropped_polymer_output_residue(
    tmp_path: Path,
) -> None:
    """Dropped FASPR output residues should be backend output-shape errors."""

    executable_path = copy_template_faspr_executable(tmp_path)
    output_template_path = tmp_path / "packed-output.pdb"
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    output_template_path.write_text(
        write_structure_string(
            build_structure(
                chains=(
                    chain_payload(
                        "A",
                        (
                            build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                            build_residue(
                                "LEU",
                                "A",
                                2,
                                ("N", "CA", "C", "O", "CB", "CG"),
                            ),
                        ),
                    ),
                ),
                source_format=FileFormat.PDB,
                source_name="faspr-dropped-polymer-output",
            ),
            FileFormat.PDB,
        ),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(
        PackingBackendExecutionError,
        match="changed the number of polymer residues",
    ):
        FasprPackingBackend(executable_path=executable_path).pack(plan)


def test_faspr_backend_rejects_reordered_polymer_output_residues(
    tmp_path: Path,
) -> None:
    """Reordered FASPR output residues should be backend output-shape errors."""

    executable_path = copy_template_faspr_executable(tmp_path)
    output_template_path = tmp_path / "packed-output.pdb"
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    output_template_path.write_text(
        write_structure_string(
            build_structure(
                chains=(
                    chain_payload(
                        "A",
                        (
                            build_residue(
                                "LEU",
                                "A",
                                2,
                                ("N", "CA", "C", "O", "CB", "CG"),
                            ),
                            build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                            build_residue(
                                "TYR",
                                "A",
                                3,
                                ("N", "CA", "C", "O", "CB", "CG"),
                            ),
                        ),
                    ),
                ),
                source_format=FileFormat.PDB,
                source_name="faspr-reordered-polymer-output",
            ),
            FileFormat.PDB,
        ),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        build_test_structure(),
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )

    with pytest.raises(
        PackingBackendExecutionError,
        match="changed residue identifiers or order",
    ):
        FasprPackingBackend(executable_path=executable_path).pack(plan)


def build_test_structure() -> ProteinStructure:
    """Build one small canonical structure for backend tests."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                    build_residue("LEU", "A", 2, ("N", "CA", "C", "O", "CB", "CG")),
                    build_residue("TYR", "A", 3, ("N", "CA", "C", "O", "CB", "CG")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="faspr-backend-test",
    )


def build_hydrogenated_test_structure(*, residue_count: int = 1) -> ProteinStructure:
    """Build one small canonical structure with polymer H topology."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    build_residue(
                        "ALA",
                        "A",
                        seq_num,
                        ("N", "H", "CA", "HA", "C", "O", "CB", "HB1"),
                    )
                    for seq_num in range(1, residue_count + 1)
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="faspr-hydrogenated-backend-test",
    )
    h_bonds: list[TopologyBond] = []
    for residue_site in structure.constitution.iter_residues():
        for heavy_atom_name, hydrogen_atom_name in (
            ("N", "H"),
            ("CA", "HA"),
            ("CB", "HB1"),
        ):
            heavy_index = structure.constitution.resolve_atom_index(
                AtomRef(residue_id=residue_site.residue_id, atom_name=heavy_atom_name)
            )
            hydrogen_index = structure.constitution.resolve_atom_index(
                AtomRef(
                    residue_id=residue_site.residue_id,
                    atom_name=hydrogen_atom_name,
                )
            )
            assert heavy_index is not None
            assert hydrogen_index is not None
            h_bonds.append(
                TopologyBond(
                    atom_index_1=heavy_index,
                    atom_index_2=hydrogen_index,
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                )
            )

    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(h_bonds),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def build_charged_hydrogenated_test_structure() -> ProteinStructure:
    """Build one hydrogenated structure with a heavy-atom formal charge."""

    structure = build_hydrogenated_test_structure()
    residue_id = structure.constitution.chain("A").residues[0].residue_id
    charged_atom_index = structure.constitution.resolve_atom_index(
        AtomRef(residue_id=residue_id, atom_name="CA")
    )
    assert charged_atom_index is not None
    atom_topologies = list(structure.topology.atom_topologies)
    atom_topologies[charged_atom_index.value] = AtomTopology(formal_charge=1)
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=tuple(atom_topologies),
            bonds=structure.topology.bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def build_heavy_only_packed_structure(
    *,
    first_residue_atom_positions: Mapping[str, Vec3] | None = None,
    atom_positions_by_residue_seq_num: Mapping[int, Mapping[str, Vec3]] | None = None,
    residue_count: int = 1,
) -> ProteinStructure:
    """Build one fake FASPR output structure without polymer hydrogens."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    build_residue(
                        "ALA",
                        "A",
                        seq_num,
                        ("N", "CA", "C", "O", "CB"),
                        atom_positions_by_name=(
                            _hydrogenated_ala_heavy_atom_positions(
                                first_residue_atom_positions
                            )
                            if seq_num == 1
                            else _hydrogenated_ala_heavy_atom_positions(
                                None
                                if atom_positions_by_residue_seq_num is None
                                else atom_positions_by_residue_seq_num.get(seq_num)
                            )
                        ),
                    )
                    for seq_num in range(1, residue_count + 1)
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="faspr-packed-heavy-only-backend-test",
    )


def _hydrogenated_ala_heavy_atom_positions(
    overrides: Mapping[str, Vec3] | None = None,
) -> Mapping[str, Vec3]:
    """Return heavy atom positions after stripping H from test ALA."""

    positions = {
        "N": Vec3(0.000, 0.000, 0.000),
        "CA": Vec3(2.028, 1.417, 0.000),
        "C": Vec3(1.145, -0.842, 1.074),
        "O": Vec3(2.318, -1.152, 1.556),
        "CB": Vec3(0.000, 0.000, 0.000),
    }
    if overrides is not None:
        positions.update(overrides)

    return positions


def copy_input_faspr_executable(tmp_path: Path) -> Path:
    """Create one fake FASPR executable that copies input to output."""

    executable_path = tmp_path / "FASPR"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "set -eu",
                'input_path=""',
                'output_path=""',
                'while [ "$#" -gt 0 ]; do',
                '  case "$1" in',
                '    -i) input_path="$2"; shift 2 ;;',
                '    -o) output_path="$2"; shift 2 ;;',
                "    *) shift ;;",
                "  esac",
                "done",
                'cp "$input_path" "$output_path"',
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    return executable_path


def copy_template_faspr_executable(tmp_path: Path) -> Path:
    """Create one fake FASPR executable that returns a prepared output PDB."""

    executable_path = tmp_path / "FASPR"
    output_template_path = tmp_path / "packed-output.pdb"
    executable_path.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "set -eu",
                'output_path=""',
                'while [ "$#" -gt 0 ]; do',
                '  case "$1" in',
                '    -o) output_path="$2"; shift 2 ;;',
                "    *) shift ;;",
                "  esac",
                "done",
                f'cp "{output_template_path}" "$output_path"',
            )
        ),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    return executable_path


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
    *,
    is_hetero: bool = False,
    atom_positions_by_name: Mapping[str, Vec3] | None = None,
) -> CanonicalResiduePayload:
    """Build one canonical residue for backend tests."""

    atoms = tuple(
        build_atom(
            atom_name,
            atom_index,
            position=(
                None
                if atom_positions_by_name is None
                else atom_positions_by_name.get(atom_name)
            ),
        )
        for atom_index, atom_name in enumerate(atom_names, start=1)
    )
    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=is_hetero,
    )


def build_atom(
    atom_name: str,
    atom_index: int,
    *,
    position: Vec3 | None = None,
) -> CanonicalAtomPayload:
    """Build one deterministic canonical atom for backend tests."""

    preset_positions = (
        Vec3(0.000, 0.000, 0.000),
        Vec3(1.458, 0.000, 0.000),
        Vec3(2.028, 1.417, 0.000),
        Vec3(3.235, 1.593, 0.248),
        Vec3(1.145, -0.842, 1.074),
        Vec3(2.318, -1.152, 1.556),
    )
    return atom_payload(
        name=atom_name,
        element=infer_element(atom_name),
        position=(
            preset_positions[(atom_index - 1) % len(preset_positions)]
            if position is None
            else position
        ),
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple element token from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
