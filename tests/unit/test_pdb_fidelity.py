"""Identity and numeric fidelity at the public PDB boundary."""

import stat
from pathlib import Path

import gemmi
import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.errors import (
    ProtrepairError,
    StructureNormalizationError,
    StructureSerializationError,
)
from protrepair.geometry import Vec3
from protrepair.io import (
    gemmi_writer,
    read_structure_string,
    write_structure,
    write_structure_string,
)
from protrepair.io.gemmi_writer import (
    pdb_atom_serial_by_atom_ref,
    write_pdb_structure_string_without_conect,
)
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import SourceBondRecordType


def _one_atom(
    *,
    chain: str = "A",
    seq: int = 1,
    component: str = "ALA",
    name: str = "CA",
    element: str = "C",
    charge: int | None = None,
    altloc: str | None = None,
    insertion: str | None = None,
    x: float = 0.0,
    b: float | None = 10.0,
    occupancy: float = 1.0,
) -> ProteinStructure:
    residue = residue_payload(
        component_id=component,
        residue_id=ResidueId(chain, seq, insertion),
        atoms=(
            atom_payload(
                name,
                element,
                Vec3(x, 0.0, 0.0),
                formal_charge=charge,
                altloc=altloc,
                b_factor=b,
                occupancy=occupancy,
            ),
        ),
    )
    return build_structure(
        chains=(chain_payload(chain, (residue,)),), source_format=FileFormat.MMCIF
    )


def _linked(chains: tuple[str, ...] = ("A",)) -> ProteinStructure:
    return build_bonded_structure(
        chains=tuple(
            chain_payload(
                chain,
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId(chain, 1),
                        atoms=(
                            atom_payload("CA", "C", Vec3(index * 10.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(index * 10.0 + 1.5, 0.0, 0.0)),
                        ),
                    ),
                ),
            )
            for index, chain in enumerate(chains)
        ),
        source_format=FileFormat.MMCIF,
    )


@pytest.mark.parametrize("chain", ["A", "AA", "AB", "_"])
@pytest.mark.parametrize("seq", [-999, 1, 9999, 10000, 1223055])
def test_pdb_roundtrip_preserves_admitted_residue_identity(
    chain: str, seq: int
) -> None:
    original = _one_atom(chain=chain, seq=seq, insertion="B", charge=0)
    reread = read_structure_string(
        write_structure_string(original, FileFormat.PDB), FileFormat.PDB
    )

    assert reread.constitution.atom_ref_at(AtomIndex(0)) == AtomRef(
        ResidueId(chain, seq, "B"), "CA"
    )
    assert reread.topology.formal_charge(AtomIndex(0)) == 0


def test_pdb_conect_preserves_distinct_two_character_chains() -> None:
    original = _linked(("A", "AA", "AB"))
    text = write_structure_string(original, FileFormat.PDB)
    reread = read_structure_string(text, FileFormat.PDB)

    assert len([line for line in text.splitlines() if line.startswith("CONECT")]) == 6
    assert {
        frozenset(
            (
                reread.constitution.atom_ref_at(b.atom_index_1),
                reread.constitution.atom_ref_at(b.atom_index_2),
            )
        )
        for b in reread.topology.bonds
    } == {
        frozenset((AtomRef(ResidueId(c, 1), "CA"), AtomRef(ResidueId(c, 1), "CB")))
        for c in ("A", "AA", "AB")
    }


@pytest.mark.parametrize("serial", [99999, 100000, 43770014])
def test_pdb_ingress_resolves_gemmi_hybrid_serial_conect(serial: int) -> None:
    raw = gemmi.read_pdb_string(write_structure_string(_linked(), FileFormat.PDB))
    raw[0][0][0][0].serial = serial
    raw[0][0][0][1].serial = serial + 1
    options = gemmi.PdbWriteOptions()
    options.preserve_serial = True
    options.conect_records = False
    rows = raw.make_pdb_string(options).splitlines()
    atoms = [line for line in rows if line.startswith("ATOM")]
    rows = [line for line in rows if line != "END" and not line.startswith("CONECT")]
    text = "\n".join((*rows, "CONECT" + atoms[0][6:11] + atoms[1][6:11], "END")) + "\n"

    reread = read_structure_string(text, FileFormat.PDB)
    assert reread.topology.bond_between(AtomIndex(0), AtomIndex(1)) is not None


@pytest.mark.parametrize("token", ["*****", "a0002"])
def test_pdb_bad_atom_serial_reports_the_field_and_token(token: str) -> None:
    text = write_structure_string(_one_atom(), FileFormat.PDB)
    text = "\n".join(
        line[:6] + token + line[11:] if line.startswith("ATOM") else line
        for line in text.splitlines()
    )
    with pytest.raises(
        StructureNormalizationError, match="serial.*" + token.replace("*", r"\*")
    ):
        read_structure_string(text, FileFormat.PDB)


@pytest.mark.parametrize("token", ["*****", "a0002"])
def test_pdb_bad_conect_serial_is_not_silently_dropped(token: str) -> None:
    text = write_structure_string(_linked(), FileFormat.PDB)
    text = text.replace("END", "CONECT    1" + token + "\nEND")
    with pytest.raises(StructureNormalizationError, match="CONECT.*serial"):
        read_structure_string(text, FileFormat.PDB)


@pytest.mark.parametrize(
    "x, expected", [(-1000.006, -1000.01), (10000.006, 10000.01), (99999.994, 99999.99)]
)
def test_pdb_overflow_coordinates_round_to_two_decimals(
    x: float, expected: float
) -> None:
    text = write_structure_string(_one_atom(x=x), FileFormat.PDB)
    reread = read_structure_string(text, FileFormat.PDB)
    assert reread.geometry.atom_geometry(AtomIndex(0)).position.x == expected
    row = next(line for line in text.splitlines() if line.startswith("ATOM"))
    assert float(row[30:38]) == expected
    assert float(row[38:46]) == 0.0


@pytest.mark.parametrize(
    "b, expected",
    [(1000.0, 1000.0), (1200.06, 1200.1), (12345.67, 12346.0), (99999.49, 99999.0)],
)
def test_pdb_overflow_b_factor_rounds_without_clamping(
    b: float, expected: float
) -> None:
    text = write_structure_string(_one_atom(b=b), FileFormat.PDB)
    reread = read_structure_string(text, FileFormat.PDB)
    assert reread.geometry.atom_geometry(AtomIndex(0)).b_factor == pytest.approx(
        expected
    )
    row = next(line for line in text.splitlines() if line.startswith("ATOM"))
    assert float(row[60:66]) == expected
    assert "." in row[60:66]


@pytest.mark.parametrize("x", [100000.0, 99999.9951, -10000.0, -9999.9951])
def test_pdb_refuses_coordinates_that_need_fewer_than_two_decimals(x: float) -> None:
    with pytest.raises(ProtrepairError, match="PDB.*coordinate.*A:1.CA.*mmCIF"):
        write_structure_string(_one_atom(x=x), FileFormat.PDB)


@pytest.mark.parametrize("b", [99999.5, 100000.0])
def test_pdb_refuses_b_factor_overflow_after_rounding(b: float) -> None:
    with pytest.raises(ProtrepairError, match="PDB.*B.factor.*A:1.CA.*mmCIF"):
        write_structure_string(_one_atom(b=b), FileFormat.PDB)


@pytest.mark.parametrize(
    "field", ["component", "name", "chain", "seq", "charge", "altloc", "insertion"]
)
def test_pdb_rejects_unrepresentable_identity_without_changing_mmcif(
    field: str,
) -> None:
    if field == "component":
        structure = _one_atom(component="A1H5U")
    elif field == "name":
        structure = _one_atom(name="ABCDE")
    elif field == "chain":
        structure = _one_atom(chain="ABCDE")
    elif field == "seq":
        structure = _one_atom(seq=1223056)
    elif field == "charge":
        structure = _one_atom(charge=10)
    elif field == "altloc":
        structure = _one_atom(altloc="AB")
    else:
        structure = _one_atom(insertion="AB")
    field_name = {
        "component": "component ID",
        "name": "atom name",
        "chain": "chain ID",
        "seq": "residue number",
        "charge": "formal charge",
        "altloc": "altloc",
        "insertion": "insertion code",
    }[field]
    with pytest.raises(StructureSerializationError, match="PDB " + field_name):
        write_structure_string(structure, FileFormat.PDB)
    if field not in {"altloc", "insertion"}:
        reread = read_structure_string(
            write_structure_string(structure, FileFormat.MMCIF), FileFormat.MMCIF
        )
        assert reread.constitution.atom_ref_at(
            AtomIndex(0)
        ) == structure.constitution.atom_ref_at(AtomIndex(0))
        assert (
            reread.constitution.residue_slots[0].component_id
            == structure.constitution.residue_slots[0].component_id
        )
        assert reread.topology.formal_charge(
            AtomIndex(0)
        ) == structure.topology.formal_charge(AtomIndex(0))


@pytest.mark.parametrize("altloc", [None, "A", "B", "C", "1", "2", "a", "b"])
@pytest.mark.parametrize("element", ["H", "D", "T"])
@pytest.mark.parametrize("occupancy", [0.0, 0.3])
def test_pdb_normalizes_selected_altloc_without_mutating_source_metadata(
    altloc: str | None, element: str, occupancy: float
) -> None:
    structure = _one_atom(
        altloc=altloc, charge=0, element=element, name="HD1", occupancy=occupancy
    )
    geometry = structure.geometry
    provenance = structure.provenance
    mmcif = write_structure_string(structure, FileFormat.MMCIF)
    text = write_structure_string(structure, FileFormat.PDB)
    reread = read_structure_string(text, FileFormat.PDB)
    after = reread.geometry.atom_geometry(AtomIndex(0))
    assert after.altloc is None
    assert after.position == geometry.atom_geometry(AtomIndex(0)).position
    assert after.occupancy == pytest.approx(occupancy)
    assert after.b_factor == 10.0
    assert reread.constitution.atom_site_at(AtomIndex(0)).element == element
    assert reread.topology.formal_charge(AtomIndex(0)) == 0
    native = write_pdb_structure_string_without_conect(structure)
    assert next(row for row in native.splitlines() if row.startswith("ATOM"))[16] == " "
    assert structure.geometry is geometry
    assert structure.provenance is provenance
    assert write_structure_string(structure, FileFormat.MMCIF) == mmcif
    restored = read_structure_string(mmcif, FileFormat.MMCIF)
    assert restored.geometry.atom_geometry(AtomIndex(0)).altloc == altloc


def test_pdb_admission_failure_preserves_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "structure.pdb"
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)
    with pytest.raises(ProtrepairError):
        write_structure(_one_atom(x=100000.0), target)
    assert target.read_text(encoding="utf-8") == "original\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("fault", ["missing", "extra", "reordered"])
def test_public_writer_checks_correspondence_without_isotope_or_charge_repairs(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _one_atom() if fault == "extra" else _linked()
    raw = gemmi_writer.build_gemmi_structure(
        _one_atom() if fault == "missing" else _linked()
    )
    if fault == "reordered":
        raw[0][0][0][0].name = "CB"
        raw[0][0][0][1].name = "CA"

    def changed_projection(structure: ProteinStructure) -> gemmi.Structure:
        return raw

    monkeypatch.setattr(gemmi_writer, "build_gemmi_structure", changed_projection)
    with pytest.raises(ProtrepairError, match="PDB.*(records|identity)"):
        write_structure_string(original, FileFormat.PDB)


def test_serial_mapping_uses_canonical_keys_and_accounts_for_numbered_ter() -> None:
    structure = _linked(("A", "AA"))
    text = write_structure_string(structure, FileFormat.PDB)
    mapping = pdb_atom_serial_by_atom_ref(text, structure)
    assert len(mapping) == 4
    assert set(mapping) == {
        structure.constitution.atom_ref_at(AtomIndex(i)) for i in range(4)
    }
    lines = text.splitlines()
    first = next(row for row in lines if row.startswith("ATOM"))[6:11]
    malformed = "\n".join(
        row[:6] + first + row[11:] if row.startswith("TER   ") else row for row in lines
    )
    with pytest.raises(ProtrepairError, match="serial collision"):
        pdb_atom_serial_by_atom_ref(malformed, structure)


def test_duplicate_first_model_serials_only_exclude_ambiguous_connections() -> None:
    text = write_structure_string(_linked(("A", "B")), FileFormat.PDB)
    lines = text.splitlines()
    atom_rows = [i for i, line in enumerate(lines) if line.startswith("ATOM")]
    first = lines[atom_rows[0]][6:11]
    second_row = atom_rows[1]
    lines[second_row] = lines[second_row][:6] + first + lines[second_row][11:]
    reread = read_structure_string("\n".join(lines) + "\n", FileFormat.PDB)

    assert len(reread.constitution.atom_slots) == 4
    source_bonds = [
        b
        for b in reread.topology.bonds
        if b.source_metadata is not None
        and b.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
    ]
    assert len(source_bonds) == 1
    assert {
        reread.constitution.atom_ref_at(i).residue_id.chain_id
        for i in (source_bonds[0].atom_index_1, source_bonds[0].atom_index_2)
    } == {"B"}


def test_conect_supports_six_neighbors_without_double_counting_duplicate_rows() -> None:
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=tuple(
                            atom_payload(f"C{i}", "C", Vec3(i * 2.0, 0.0, 0.0))
                            for i in range(7)
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    text = write_structure_string(structure, FileFormat.PDB)
    row = "CONECT    1    2    3    4    5    6    7\n"
    text = text.replace("END", row + row + "CONECT    1 99998\nEND")
    reread = read_structure_string(text, FileFormat.PDB)
    source_bonds = [
        b
        for b in reread.topology.bonds
        if b.source_metadata is not None
        and b.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
    ]
    assert len(source_bonds) == 6
    assert all(
        b.source_metadata is not None and b.source_metadata.reported_order is None
        for b in source_bonds
    )


def test_invalid_serial_in_later_model_does_not_reject_the_first_model() -> None:
    original = write_structure_string(_one_atom(charge=0), FileFormat.PDB)
    atom = next(row for row in original.splitlines() if row.startswith("ATOM"))
    text = (
        "\n".join(
            (
                "MODEL        1",
                atom,
                "ENDMDL",
                "MODEL        2",
                atom[:6] + "*****" + atom[11:],
                "ENDMDL",
                "END",
            )
        )
        + "\n"
    )
    reread = read_structure_string(text, FileFormat.PDB)
    assert len(reread.constitution.atom_slots) == 1
    assert reread.topology.formal_charge(AtomIndex(0)) == 0


@pytest.mark.parametrize(
    "x", [-999.999, -999.9989999, -0.0005, -0.0001, 0.0, 9999.9989999, 9999.999]
)
@pytest.mark.parametrize("b", [0.0, 999.99, 999.994])
def test_normal_numeric_fields_still_match_the_native_projection(
    x: float, b: float
) -> None:
    structure = _one_atom(x=x, b=b)
    public = write_structure_string(structure, FileFormat.PDB)
    native = write_pdb_structure_string_without_conect(structure)
    public_row = next(r for r in public.splitlines() if r.startswith("ATOM"))
    native_row = next(r for r in native.splitlines() if r.startswith("ATOM"))
    assert public_row[30:66] == native_row[30:66]
