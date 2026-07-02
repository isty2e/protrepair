"""Unit and smoke tests for the FASPR side-chain packing backend."""

import math
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

import protrepair.transformer.packing.faspr.paths as faspr_paths
from protrepair.geometry import Vec3
from protrepair.io import read_structure, write_structure_string
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
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
    ) == sum(
        len(chain_site.residues)
        for chain_site in structure.constitution.chains
    )
    assert (
        result.packed_structure.constitution.ligands
        == structure.constitution.ligands
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


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
    *,
    is_hetero: bool = False,
) -> CanonicalResiduePayload:
    """Build one canonical residue for backend tests."""

    atoms = tuple(
        build_atom(atom_name, atom_index)
        for atom_index, atom_name in enumerate(atom_names, start=1)
    )
    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=is_hetero,
    )


def build_atom(atom_name: str, atom_index: int) -> CanonicalAtomPayload:
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
        position=preset_positions[(atom_index - 1) % len(preset_positions)],
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple element token from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
