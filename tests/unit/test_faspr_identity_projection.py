"""FASPR residue correspondence and malformed native-output rejection."""

from dataclasses import replace
from pathlib import Path
from random import Random

import gemmi
import pytest
from tests.support.canonical_builders import build_structure, chain_payload
from tests.unit.test_faspr_backend import build_residue, copy_input_faspr_executable

from protrepair.io import read_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite, ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.packing import PackingPlan, PackingSpec
from protrepair.transformer.packing.faspr import backend
from protrepair.transformer.packing.faspr import projection as projection_module
from protrepair.transformer.packing.faspr.backend import (
    FasprPackingBackend,
    PackingBackendError,
    PackingBackendExecutionError,
)
from protrepair.transformer.packing.faspr.projection import (
    FASPR_RESIDUE_CAPACITY,
    FasprPDBProjection,
    faspr_residue_id,
)


@pytest.fixture
def source() -> ProteinStructure:
    return build_structure(
        chains=tuple(
            chain_payload(
                chain,
                (build_residue("ALA", chain, number, ("N", "CA", "C", "O", "CB")),),
            )
            for chain, number in (("LONG-CHAIN", -1000), ("AA", 1223056))
        ),
        source_format=FileFormat.MMCIF,
    )


@pytest.mark.parametrize(
    ("ordinal", "number", "insertion"),
    ((0, -999, None), (10998, 9999, None), (10999, -999, "A"), (692936, 9999, "9")),
)
def test_allocator_boundaries_match_native_decimal_fields(
    ordinal: int, number: int, insertion: str | None
) -> None:
    native_id = faspr_residue_id(ordinal)
    assert native_id == ResidueId("A", number, insertion)
    raw = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "ALA"
    residue.seqid = gemmi.SeqId(number, insertion or " ")
    atom = gemmi.Atom()
    atom.name = "CA"
    atom.element = gemmi.Element("C")
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    raw.add_model(model)
    line = next(
        line for line in raw.make_pdb_string().splitlines() if line.startswith("ATOM")
    )
    assert line[22:27] == f"{number:4d}{insertion or ' '}"
    reread = gemmi.read_pdb_string(line + "\n")[0][0][0]
    assert (reread.seqid.num, reread.seqid.icode.strip() or None) == (number, insertion)


def test_allocator_is_injective_across_insertion_blocks() -> None:
    rng = Random(231)
    ordinals = set(rng.sample(range(FASPR_RESIDUE_CAPACITY), 1000))
    ids = {faspr_residue_id(ordinal) for ordinal in ordinals}
    assert len(ids) == len(ordinals)
    assert all(-999 <= residue.seq_num <= 9999 for residue in ids)
    assert {residue.insertion_code for residue in ids} >= {None, "A", "a", "9"}


@pytest.mark.parametrize("ordinal", (-1, 692937))
def test_allocator_rejects_exhaustion_instead_of_wrapping(ordinal: int) -> None:
    with pytest.raises(ValueError, match="at most 692937 residues"):
        faspr_residue_id(ordinal)


def test_projection_restores_unrepresentable_source_labels(
    source: ProteinStructure,
) -> None:
    projection = FasprPDBProjection(source.constitution.chains)
    text = projection.render(source)
    native = read_structure_string(text, FileFormat.PDB)
    assert tuple(residue.residue_id for residue in native.iter_residue_sites()) == (
        ResidueId("A", -999),
        ResidueId("A", -998),
    )
    restored = projection.decode(text, expected_components=("ALA", "ALA"))
    assert restored.constitution == source.constitution
    assert restored.chain_ids() == ("LONG-CHAIN", "AA")
    assert tuple(g.position for g in restored.geometry.atom_geometries) == tuple(
        g.position for g in source.geometry.atom_geometries
    )
    assert source.chain_ids() == ("LONG-CHAIN", "AA")


@pytest.mark.parametrize(
    "damage",
    (
        "missing",
        "duplicate",
        "reordered",
        "foreign",
        "foreign_extended_chain",
        "component",
        "atom",
        "model",
        "altloc",
    ),
)
def test_projection_rejects_malformed_native_inventory_before_normalization(
    source: ProteinStructure, damage: str
) -> None:
    projection = FasprPDBProjection(source.constitution.chains)
    records = [
        line
        for line in projection.render(source).splitlines()
        if line.startswith("ATOM")
    ]
    if damage == "missing":
        records = records[:5]
    elif damage == "duplicate":
        records += records[:5]
    elif damage == "reordered":
        records = records[5:] + records[:5]
    elif damage == "foreign":
        records[0] = records[0][:21] + "B" + records[0][22:]
    elif damage == "foreign_extended_chain":
        records[0] = records[0][:20] + "B" + records[0][21:]
    elif damage == "component":
        records[0] = records[0][:17] + "LEU" + records[0][20:]
    elif damage == "atom":
        records.insert(1, records[0])
    elif damage == "model":
        records = ["MODEL        1", *records, "ENDMDL", "MODEL        2", "ENDMDL"]
    else:
        records[0] = records[0][:16] + "B" + records[0][17:]
    with pytest.raises(ValueError, match="FASPR"):
        projection.decode("\n".join(records) + "\n", expected_components=("ALA", "ALA"))


def test_native_log_translation_requires_a_unique_component_and_number(
    source: ProteinStructure,
) -> None:
    projection = FasprPDBProjection(source.constitution.chains)
    assert projection.describe_native_error(
        "error! backbone is incomplete at residue: ALA-999, please check!"
    ) == (
        "native FASPR output: error! backbone is incomplete at residue: "
        "ALA LONG-CHAIN:-1000, please check!"
    )
    assert projection.describe_native_error("#residues: 123, index -999") == (
        "native FASPR output: #residues: 123, index -999"
    )


def test_projection_rejects_a_different_source_snapshot(
    source: ProteinStructure,
) -> None:
    projection = FasprPDBProjection(
        (replace(source.constitution.chains[0], residues=()),)
    )
    with pytest.raises(ValueError, match="does not match"):
        projection.render(source)


def test_native_log_omitting_insertion_does_not_invent_a_source_location() -> None:
    # Empty residue metadata suffices; no giant coordinate structure is needed.
    residues = tuple(
        ResidueSite("ALA", ResidueId("LONG", ordinal), (), False)
        for ordinal in range(11000)
    )
    projection = FasprPDBProjection((ChainSite("LONG", residues),))
    assert projection.describe_native_error("warning! incomplete residue ALA-999") == (
        "native FASPR output: warning! incomplete residue ALA-999"
    )
    assert projection.describe_native_error("warning! incomplete residue ALA-998") == (
        "native FASPR output: warning! incomplete residue ALA LONG:1"
    )


def test_projection_preserves_empty_source_chains(source: ProteinStructure) -> None:
    chains = (
        ChainSite("EMPTY", ()),
        *source.constitution.chains,
        ChainSite("LAST", ()),
    )
    namespace = FasprPDBProjection(chains)
    restored = namespace.decode(
        namespace.render(source), expected_components=("ALA", "ALA")
    )
    assert restored.constitution.chains == chains


def test_canonical_duplicate_atom_names_cannot_hide_behind_case(
    source: ProteinStructure,
) -> None:
    projection = FasprPDBProjection(source.constitution.chains)
    lines = projection.render(source).splitlines()
    index = next(
        index for index, line in enumerate(lines) if line[12:16].strip() == "CA"
    )
    line = lines[index]
    lines.insert(index, line[:12] + " ca " + line[16:])
    with pytest.raises(ValueError, match="duplicate"):
        projection.decode("\n".join(lines), expected_components=("ALA", "ALA"))


def test_projection_accepts_mixed_line_endings_without_identity_changes(
    source: ProteinStructure,
) -> None:
    projection = FasprPDBProjection(source.constitution.chains)
    lines = projection.render(source).splitlines()
    text = "".join(
        line + ("\n" if index % 2 else "\r\n") for index, line in enumerate(lines)
    )
    restored = projection.decode(text, expected_components=("ALA", "ALA"))
    assert restored.constitution == source.constitution


def test_capacity_refusal_happens_before_temporary_files(
    source: ProteinStructure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = copy_input_faspr_executable(tmp_path)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    plan = PackingPlan.from_inputs(source, PackingSpec(backend_name="faspr"))
    monkeypatch.setattr(projection_module, "FASPR_RESIDUE_CAPACITY", 1)

    def unexpected_temporary_files(*, prefix: str) -> None:
        pytest.fail("Namespace exhaustion must be refused before creating files")

    monkeypatch.setattr(backend, "TemporaryDirectory", unexpected_temporary_files)
    with pytest.raises(PackingBackendError, match="namespace supports at most"):
        FasprPackingBackend(executable_path=executable).pack(plan)


def test_native_failure_uses_source_location_and_removes_temporary_files(
    source: ProteinStructure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "FASPR"
    captured_path = tmp_path / "native-input-path.txt"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s' \"$2\" > '{captured_path}'\n"
        "printf 'error! backbone is incomplete at residue: ALA-999, please check!' >&2\n"
        "exit 7\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    plan = PackingPlan.from_inputs(source, PackingSpec(backend_name="faspr"))
    with pytest.raises(PackingBackendExecutionError, match="ALA LONG-CHAIN:-1000"):
        FasprPackingBackend(executable_path=executable).pack(plan)
    assert not Path(captured_path.read_text(encoding="utf-8")).exists()


def test_expanded_native_locations_keep_error_excerpt_bounded(
    source: ProteinStructure,
) -> None:
    projection = FasprPDBProjection(source.constitution.chains)
    result = projection.describe_native_error("residue ALA-999 " * 100, max_bytes=64)
    assert "LONG-CHAIN:-1000" in result
    assert result.endswith("[truncated after 64 bytes]")
    assert len(result) <= 64 + len(
        "native FASPR output: ... [truncated after 64 bytes]"
    )
