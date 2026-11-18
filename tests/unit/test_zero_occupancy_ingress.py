"""Zero-occupancy ingress applies only to declared source positions."""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import gemmi
import pytest

from protrepair import process_structure
from protrepair.errors import StructureNormalizationError
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.ingress_policy import (
    LigandHandling,
    MutationPolicy,
    OccupancyPolicy,
    StructureNormalizationPolicy,
)
from protrepair.io.structure_ingress import (
    SourceAtomAnnotation,
    apply_structure_normalization_policy,
    normalize_raw_structure,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import SidechainHeavyAtomCompletenessState
from protrepair.structure import AtomRef, FileFormat, ResidueId, StructureIngress
from protrepair.structure.source_connection import SourceAtomIdentity
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.workflow.contracts import (
    StructureIngressOptions,
    requested_process_goal,
)

LYS = ResidueId("A", 1)
NZ = AtomRef(LYS, "NZ")
OMIT_ZEROS = StructureNormalizationPolicy(
    omit_zero_occupancy_sidechain_heavy_atoms=True
)
LYS_POSITIONS = MappingProxyType(
    {
        "N": (0.0, 0.0, 0.0),
        "CA": (1.458, 0.0, 0.0),
        "C": (2.0, 1.4, 0.0),
        "O": (3.2, 1.6, 0.0),
        "CB": (1.95, -0.8, -1.2),
        "CG": (3.4, -1.0, -1.25),
        "CD": (3.9, -2.0, -2.3),
        "CE": (5.4, -2.0, -2.4),
        "NZ": (35.0, 18.0, -7.0),
    }
)


def _pdb_atom(
    name: str,
    position: tuple[float, float, float],
    *,
    serial: int,
    occupancy: str = "1.00",
    component: str = "LYS",
    chain: str = "A",
    residue_number: int = 1,
    altloc: str = " ",
    element: str | None = None,
    charge: str = "",
    record: str = "ATOM",
) -> str:
    x, y, z = position
    symbol = name[0] if element is None else element
    return (
        f"{record:<6}{serial:5d} {name:^4}{altloc}{component:>3} "
        f"{chain}{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:>6}{24.0:6.2f}          {symbol:>2}{charge:>2}"
    )


def _lys_atoms(occupancies: dict[str, str]) -> list[str]:
    return [
        _pdb_atom(
            name, position, serial=serial, occupancy=occupancies.get(name, "1.00")
        )
        for serial, (name, position) in enumerate(LYS_POSITIONS.items(), start=1)
    ]


def _pdb(lines: list[str]) -> str:
    return "\n".join((*lines, "END", ""))


def _lys_mmcif(occupancy: str | None) -> str:
    raw = gemmi.read_pdb_string(_pdb(_lys_atoms({})))
    raw.setup_entities()
    document = raw.make_mmcif_document()
    block = document.sole_block()
    values = block.find_values("_atom_site.occupancy")
    if occupancy is None:
        values.erase()
    else:
        values[len(LYS_POSITIONS) - 1] = occupancy
    return document.as_string()


def _with_omissions(text: str, file_format: FileFormat = FileFormat.PDB):
    return read_structure_string(
        text,
        file_format,
        policy=OMIT_ZEROS,
    )


@pytest.mark.parametrize(
    ("token", "omitted"),
    [("0.00", True), ("-0.00", True), ("", False), ("0.01", False)],
)
def test_pdb_original_token_without_an_alternate(token, omitted):
    text = _pdb(_lys_atoms({"NZ": token}))
    default = read_structure_string(text, FileFormat.PDB)
    assert default.constitution.resolve_atom_index(NZ) is not None
    assert default.provenance.ingress.zero_occupancy_omissions is None

    projected = _with_omissions(text)
    ingress = projected.provenance.ingress
    assert ingress.observation is not None
    assert ingress.zero_occupancy_omissions is not None
    assert (projected.constitution.resolve_atom_index(NZ) is None) is omitted
    assert (ingress.observation.constitution.resolve_atom_index(NZ) is None) is omitted
    assert len(ingress.zero_occupancy_omissions) == int(omitted)
    if omitted:
        (row,) = ingress.zero_occupancy_omissions
        assert row.source_identity.atom_ref == NZ
        assert row.source_identity.component_id == "LYS"
        assert row.source_identity.altloc is None
        assert row.element.chemical_symbol == "N"
        assert row.geometry == default.geometry.atom_geometry(
            default.constitution.atom_index(NZ)
        )
        expected = default.without_atom_refs((NZ,))
        assert projected.constitution == expected.constitution
        assert projected.geometry == expected.geometry
        assert projected.topology == expected.topology
    else:
        assert projected.geometry == default.geometry


@pytest.mark.parametrize(
    ("token", "omitted"),
    [
        ("0", True),
        ("-0", True),
        ("0.00(3)", True),
        ("?", False),
        (".", False),
        (None, False),
        ("1e-60", False),
    ],
)
def test_mmcif_original_token_not_the_native_float(token, omitted):
    projected = _with_omissions(_lys_mmcif(token), FileFormat.MMCIF)
    assert (projected.constitution.resolve_atom_index(NZ) is None) is omitted
    if token == "1e-60":
        assert (
            projected.geometry.atom_geometry(
                projected.constitution.atom_index(NZ)
            ).occupancy
            == 0.0
        )


def test_pdb_malformed_occupancy_is_not_a_zero_declaration():
    text = _pdb(_lys_atoms({"NZ": "0.bad"}))
    with pytest.raises(StructureNormalizationError, match="invalid source occupancy"):
        _with_omissions(text)


def test_a_short_pdb_coordinate_record_is_not_a_zero_declaration():
    lines = _lys_atoms({})
    lines[-1] = lines[-1][:54]
    result = _with_omissions(_pdb(lines))
    assert result.constitution.resolve_atom_index(NZ) is not None
    assert result.provenance.ingress.zero_occupancy_omissions == ()


def test_raw_numeric_zero_does_not_prove_a_zero_declaration():
    raw = gemmi.read_pdb_string(_pdb(_lys_atoms({"NZ": ""})))
    with pytest.raises(StructureNormalizationError, match="source coordinate text"):
        normalize_raw_structure(
            raw, file_format=FileFormat.PDB, source_connections=(), policy=OMIT_ZEROS
        )


def test_zero_proof_must_match_the_parsed_scalar():
    raw = gemmi.read_pdb_string(_pdb(_lys_atoms({})))
    annotations = {
        SourceAtomIdentity(AtomRef(LYS, name), "LYS"): (
            SourceAtomAnnotation(explicitly_zero_occupancy=name == "NZ"),
        )
        for name in LYS_POSITIONS
    }
    with pytest.raises(StructureNormalizationError, match="disagrees with parsed atom"):
        normalize_raw_structure(
            raw,
            file_format=FileFormat.PDB,
            source_connections=(),
            policy=OMIT_ZEROS,
            source_annotations_by_atom_identity=annotations,
        )


@pytest.mark.parametrize("file_format", tuple(FileFormat))
def test_duplicate_occurrences_with_conflicting_custody_refuse(file_format):
    lines = _lys_atoms({"NZ": "0.00"})
    lines.insert(6, _pdb_atom("NZ", LYS_POSITIONS["NZ"], serial=10, occupancy=""))
    text = _pdb(lines)
    if file_format is FileFormat.MMCIF:
        raw = gemmi.read_pdb_string(text)
        raw.setup_entities()
        document = raw.make_mmcif_document()
        block = document.sole_block()
        names = block.find_values("_atom_site.label_atom_id")
        occupancies = block.find_values("_atom_site.occupancy")
        first, second = (index for index, name in enumerate(names) if name == "NZ")
        occupancies[first] = "1e-60"
        occupancies[second] = "0.00"
        text = document.as_string()

    assert (
        read_structure_string(text, file_format).constitution.resolve_atom_index(NZ)
        is not None
    )
    with pytest.raises(
        StructureNormalizationError, match="ambiguous source atom annotations"
    ):
        _with_omissions(text, file_format)


@pytest.mark.parametrize("file_format", tuple(FileFormat))
def test_omit_after_altloc_selection_without_fallback(file_format):
    lines = _lys_atoms({})[:-1]
    lines.extend(
        (
            _pdb_atom(
                "NZ", LYS_POSITIONS["NZ"], serial=9, occupancy="0.00", altloc="A"
            ),
            _pdb_atom("NZ", (5.0, -3.0, -3.0), serial=10, occupancy="0.80", altloc="B"),
        )
    )
    text = _coordinate_text(lines, file_format)
    highest = _with_omissions(text, file_format)
    assert (
        highest.geometry.atom_geometry(highest.constitution.atom_index(NZ)).altloc
        == "B"
    )
    assert highest.provenance.ingress.zero_occupancy_omissions == ()

    lowest = read_structure_string(
        text,
        file_format,
        policy=StructureNormalizationPolicy(
            occupancy_policy=OccupancyPolicy.LOWEST,
            omit_zero_occupancy_sidechain_heavy_atoms=True,
        ),
    )
    assert lowest.constitution.resolve_atom_index(NZ) is None
    assert lowest.provenance.ingress.zero_occupancy_omissions is not None
    (omitted,) = lowest.provenance.ingress.zero_occupancy_omissions
    assert omitted.source_identity.altloc == "A"


def _coordinate_text(lines: list[str], file_format: FileFormat) -> str:
    text = _pdb(lines)
    if file_format is FileFormat.PDB:
        return text
    raw = gemmi.read_pdb_string(text)
    raw.setup_entities()
    return raw.make_mmcif_document().as_string()


def test_residue_variant_is_selected_before_omission():
    lines = _lys_atoms({"NZ": "0.00"})
    lines.extend(
        _pdb_atom(name, position, serial=serial, component="ALA", occupancy="1.00")
        for serial, (name, position) in enumerate(
            tuple(LYS_POSITIONS.items())[:5], start=10
        )
    )
    text = _pdb(lines)
    preferred = _with_omissions(text)
    assert preferred.constitution.chains[0].residues[0].component_id == "ALA"
    assert preferred.provenance.ingress.zero_occupancy_omissions == ()

    selected = read_structure_string(
        text,
        FileFormat.PDB,
        policy=StructureNormalizationPolicy(
            mutation_policy=MutationPolicy.LOWEST_OCCUPANCY,
            omit_zero_occupancy_sidechain_heavy_atoms=True,
        ),
    )
    assert selected.constitution.chains[0].residues[0].component_id == "LYS"
    assert selected.constitution.resolve_atom_index(NZ) is None
    assert selected.provenance.ingress.zero_occupancy_omissions is not None
    assert len(selected.provenance.ingress.zero_occupancy_omissions) == 1


@pytest.mark.parametrize("file_format", tuple(FileFormat))
@pytest.mark.parametrize(("isotope", "mass"), [("H", None), ("D", 2), ("T", 3)])
def test_dependent_zero_hydrogen_keeps_isotope_history(file_format, isotope, mass):
    name = isotope + "Z1"
    lines = _lys_atoms({"NZ": "0.00"})
    lines.extend(
        (
            _pdb_atom(
                name, (35.1, 18.0, -7.0), serial=10, occupancy="0.00", element=isotope
            ),
            _pdb_atom(
                "HB2", (1.95, -1.8, -1.2), serial=11, occupancy="0.00", element="H"
            ),
        )
    )
    text = _coordinate_text(lines, file_format)
    if file_format is FileFormat.MMCIF and isotope == "T":
        document = gemmi.cif.read_string(text)
        element_types = document.sole_block().find_values("_atom_site.type_symbol")
        element_types[9] = "T"
        text = document.as_string()
    result = _with_omissions(text, file_format)
    ingress = result.provenance.ingress
    assert (
        ingress.observation is not None and ingress.zero_occupancy_omissions is not None
    )
    heavy, hydrogen = ingress.zero_occupancy_omissions
    assert heavy.source_identity.atom_ref == NZ
    href = AtomRef(LYS, name)
    assert hydrogen.source_identity.atom_ref == href
    assert hydrogen.element.isotope_mass_number == mass
    assert ingress.observation.constitution.resolve_atom_index(href) is None
    assert result.constitution.resolve_atom_index(href) is None
    assert result.constitution.resolve_atom_index(AtomRef(LYS, "HB2")) is not None


@pytest.mark.parametrize("occupancy", ("1.00", ""))
def test_retained_hydrogen_cannot_lose_its_parent(occupancy):
    lines = _lys_atoms({"NZ": "0.00"})
    lines.append(_pdb_atom("HZ1", (36.0, 18.0, -7.0), serial=10, occupancy=occupancy))
    with pytest.raises(StructureNormalizationError, match="parent of retained H/D/T"):
        _with_omissions(_pdb(lines))


def test_unresolved_hydrogen_in_affected_residue_refuses():
    lines = _lys_atoms({"NZ": "0.00"})
    lines.append(_pdb_atom("HHH", (36.0, 18.0, -7.0), serial=10, occupancy="0.00"))
    with pytest.raises(StructureNormalizationError, match="unresolved H attachment"):
        _with_omissions(_pdb(lines))


def test_omitted_numerical_zero_charge_and_retained_nonzero_charge_are_protected():
    removed_charge = _lys_atoms({"NZ": "0.00"})
    removed_charge[-1] = removed_charge[-1][:78] + "0+"
    with pytest.raises(StructureNormalizationError, match=r"formal charge at .*NZ"):
        _with_omissions(_pdb(removed_charge))

    retained_charge = _lys_atoms({"NZ": "0.00"})
    retained_charge[-2] = retained_charge[-2][:78] + "1+"
    with pytest.raises(StructureNormalizationError, match=r"formal charge at .*CE"):
        _with_omissions(_pdb(retained_charge))

    retained_charge[-2] = retained_charge[-2][:78] + "0+"
    retained_charge[0] = retained_charge[0][:78] + "1+"
    kept = _with_omissions(_pdb(retained_charge))
    assert kept.constitution.resolve_atom_index(NZ) is None
    assert dict(
        kept.residue_formal_charge_by_atom_name(kept.constitution.residue_index(LYS))
    ) == {
        "CE": 0,
        "N": 1,
    }


def test_unexpected_charged_sidechain_atom_is_protected():
    lines = _lys_atoms({"NZ": "0.00"})
    lines.append(
        _pdb_atom("Q1", (6.0, -2.7, -2.8), serial=10, element="C", charge="1+")
    )
    with pytest.raises(StructureNormalizationError, match=r"formal charge at .*Q1"):
        _with_omissions(_pdb([*lines, "CONECT    8   10"]))


def test_an_interior_cut_cannot_disconnect_observed_heavy_atoms():
    with pytest.raises(
        StructureNormalizationError, match="disconnect retained atoms: CE, NZ"
    ):
        _with_omissions(_pdb(_lys_atoms({"CD": "0.00"})))
    safe = _with_omissions(_pdb(_lys_atoms({"CE": "0.00", "NZ": "0.00"})))
    assert (
        safe.constitution.chains[0].residues[0].atom_site_names()
        == tuple(LYS_POSITIONS)[:-2]
    )


def test_ring_cut_retains_the_connected_ring_atoms():
    ring = ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    names = ("N", "CA", "C", "O", *ring)
    lines = [
        _pdb_atom(
            name,
            LYS_POSITIONS.get(name, (float(serial), 2.0, 1.0)),
            serial=serial,
            component="PHE",
            occupancy="0.00" if name == "CZ" else "1.00",
        )
        for serial, name in enumerate(names, start=1)
    ]
    text = _pdb(lines)
    source = read_structure_string(text, FileFormat.PDB)
    projected = _with_omissions(text)
    expected = source.without_atom_refs((AtomRef(LYS, "CZ"),))
    assert projected.constitution == expected.constitution
    assert projected.geometry == expected.geometry
    assert projected.topology == expected.topology
    repaired = repair_heavy_atoms_core(projected, augment_c_terminal_oxt=False)
    for name in names[:-1]:
        ref = AtomRef(LYS, name)
        assert repaired.structure.geometry.atom_geometry(
            repaired.structure.constitution.atom_index(ref)
        ) == projected.geometry.atom_geometry(projected.constitution.atom_index(ref))
    assert (
        repaired.structure.constitution.resolve_atom_index(AtomRef(LYS, "CZ"))
        is not None
        or repaired.issues
    )


def test_whole_zero_proline_terminal_and_ligand_stay_as_observed():
    lines = _lys_atoms(dict.fromkeys(LYS_POSITIONS, "0.00"))
    for serial, name in enumerate(
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OXT"), start=10
    ):
        lines.append(
            _pdb_atom(
                name,
                LYS_POSITIONS.get(name, (2.0, 2.4, 0.0)),
                serial=serial,
                component="PRO",
                residue_number=2,
                occupancy="0.00" if name in {"CD", "OXT"} else "1.00",
            )
        )
    lines.extend(
        (
            _pdb_atom(
                "C1",
                (55.1, 25.0, -7.0),
                serial=18,
                component="LIG",
                chain="B",
                occupancy="0.00",
                record="HETATM",
            ),
            _pdb_atom(
                "O1",
                (56.4, 25.0, -7.0),
                serial=19,
                component="LIG",
                chain="B",
                occupancy="0.00",
                record="HETATM",
            ),
        )
    )
    text = _pdb(lines)
    source = read_structure_string(text, FileFormat.PDB)
    projected = _with_omissions(text)
    assert projected.provenance.ingress.zero_occupancy_omissions == ()
    assert projected.constitution == source.constitution
    assert projected.geometry == source.geometry
    assert projected.topology == source.topology


def test_unexpected_element_is_not_omitted():
    lines = _lys_atoms({})[:-1]
    lines.append(
        _pdb_atom("NZ", LYS_POSITIONS["NZ"], serial=9, occupancy="0.00", element="O")
    )
    result = _with_omissions(_pdb(lines))
    assert result.constitution.resolve_atom_index(NZ) is not None
    assert result.provenance.ingress.zero_occupancy_omissions == ()


def test_incident_conect_is_protected_but_an_unaffected_conect_survives():
    lines = _lys_atoms({"NZ": "0.00"})
    with pytest.raises(StructureNormalizationError, match=r"pdb_conect .*NZ"):
        _with_omissions(_pdb([*lines, "CONECT    8    9"]))
    safe = _with_omissions(_pdb([*lines, "CONECT    7    8"]))
    observation = safe.provenance.ingress.observation
    assert observation is not None
    (bond,) = observation.topology.bonds
    assert {
        observation.constitution.atom_ref_at(index).atom_name
        for index in bond.endpoint_pair()
    } == {"CD", "CE"}


@pytest.mark.parametrize("file_format", tuple(FileFormat))
@pytest.mark.parametrize("altloc", ("A", "B"))
@pytest.mark.parametrize("linked_name", ("NZ", "CE"))
def test_connection_to_excluded_partner_protects_the_selected_sidechain(
    file_format, altloc, linked_name
):
    lines = _lys_atoms({})[:-1]
    if linked_name == "CE":
        lines[-1] = lines[-1][:16] + altloc + lines[-1][17:]
    lines.extend(
        (
            _pdb_atom(
                "NZ", LYS_POSITIONS["NZ"], serial=9, occupancy="0.00", altloc="A"
            ),
            _pdb_atom("NZ", (9.0, 8.0, 7.0), serial=10, occupancy="0.80", altloc="B"),
            _pdb_atom(
                "C1",
                (36.4, 18.0, -7.0),
                serial=11,
                component="LIG",
                chain="B",
                record="HETATM",
            ),
            _pdb_atom(
                "O1",
                (37.5, 18.0, -7.0),
                serial=12,
                component="LIG",
                chain="B",
                record="HETATM",
            ),
        )
    )
    raw = gemmi.read_pdb_string(_pdb(lines))
    raw.setup_entities()
    connection = gemmi.Connection()
    connection.name = "covalent"
    connection.type = gemmi.ConnectionType.Covale
    connection.reported_distance = 1.4
    connection.partner1 = gemmi.AtomAddress(
        "A", gemmi.SeqId(1, " "), "LYS", linked_name, altloc
    )
    connection.partner2 = gemmi.AtomAddress("B", gemmi.SeqId(1, " "), "LIG", "C1")
    raw.connections.append(connection)
    text = (
        raw.make_pdb_string()
        if file_format is FileFormat.PDB
        else raw.make_mmcif_document().as_string()
    )
    drop_partner = StructureNormalizationPolicy(
        occupancy_policy=OccupancyPolicy.LOWEST,
        ligand_handling=LigandHandling.DROP,
        selected_chain_ids=("A",),
        omit_zero_occupancy_sidechain_heavy_atoms=True,
    )
    keep_partner = read_structure_string(
        text,
        file_format,
        policy=StructureNormalizationPolicy(
            occupancy_policy=OccupancyPolicy.LOWEST
            if altloc == "A"
            else OccupancyPolicy.HIGHEST,
        ),
    )
    assert keep_partner.provenance.ingress.observation is not None
    assert len(keep_partner.provenance.ingress.observation.topology.bonds) == 1
    if altloc == "A":
        type_name = "pdb_link" if file_format is FileFormat.PDB else "mmcif_struct_conn"
        with pytest.raises(
            StructureNormalizationError, match=rf"{type_name} .*{linked_name}"
        ):
            read_structure_string(text, file_format, policy=drop_partner)
    else:
        result = read_structure_string(text, file_format, policy=drop_partner)
        assert result.constitution.resolve_atom_index(NZ) is None
        assert not result.constitution.ligands


def test_canonical_opt_in_requires_the_source_receipt_even_on_a_noop():
    text = _pdb(_lys_atoms({}))
    unchecked = read_structure_string(text, FileFormat.PDB)
    assert unchecked.provenance.ingress.observation is not None
    with pytest.raises(StructureNormalizationError, match="re-reading the original"):
        apply_structure_normalization_policy(unchecked, policy=OMIT_ZEROS)
    with pytest.raises(ValueError, match="requires a source observation"):
        StructureIngress(FileFormat.PDB, zero_occupancy_omissions=())

    checked = _with_omissions(text)
    assert checked.provenance.ingress.zero_occupancy_omissions == ()
    assert apply_structure_normalization_policy(checked, policy=OMIT_ZEROS) is checked


def test_source_history_keeps_its_scope_after_later_chain_selection(tmp_path: Path):
    lines = _lys_atoms({"NZ": "0.00"})
    partner = [
        _pdb_atom(
            name,
            position,
            serial=serial,
            chain="B",
            occupancy="0.00" if name == "NZ" else "1.00",
        )
        for serial, (name, position) in enumerate(LYS_POSITIONS.items(), start=10)
    ]
    path = tmp_path / "chains.pdb"
    path.write_text(_pdb([*lines, "TER", *partner]))
    ingress = StructureIngressOptions(omit_zero_occupancy_sidechain_heavy_atoms=True)
    result = process_structure(path, ingress=ingress, selected_source_chain_ids=("A",))
    assert result.structure.constitution.chain_ids() == ("A",)
    history = result.structure.provenance.ingress.zero_occupancy_omissions
    assert history is not None
    assert [
        record.source_identity.atom_ref.residue_id.chain_id for record in history
    ] == ["A"]
    assert not result.repairs

    full = _with_omissions(path.read_text())
    narrowed = apply_structure_normalization_policy(
        full, policy=replace(OMIT_ZEROS, selected_chain_ids=("A",))
    )
    assert narrowed.constitution.chain_ids() == ("A",)
    assert narrowed.provenance.ingress is full.provenance.ingress
    original = narrowed.provenance.ingress.zero_occupancy_omissions
    assert original is not None
    assert [
        record.source_identity.atom_ref.residue_id.chain_id for record in original
    ] == ["A", "B"]


@pytest.mark.parametrize("source_format", tuple(FileFormat))
def test_ingress_alone_does_not_repair_and_the_existing_goal_can_rebuild(
    tmp_path: Path,
    source_format: FileFormat,
):
    path = tmp_path / (
        "sidechain.pdb" if source_format is FileFormat.PDB else "sidechain.cif"
    )
    path.write_text(_coordinate_text(_lys_atoms({"NZ": "0.00"}), source_format))
    ingress = StructureIngressOptions(omit_zero_occupancy_sidechain_heavy_atoms=True)

    observed = process_structure(path, ingress=ingress)
    assert observed.structure.constitution.resolve_atom_index(NZ) is None
    assert not observed.repairs
    without_reapplication = process_structure(observed.structure)
    assert without_reapplication.structure is observed.structure
    assert without_reapplication.branch_report == observed.branch_report

    repaired = process_structure(
        path,
        ingress=ingress,
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(),
                value=SidechainHeavyAtomCompletenessState.COMPLETE,
            ),
        ),
    )
    restored = repaired.structure
    index = restored.constitution.atom_index(NZ)
    source = restored.provenance.ingress
    assert (
        source.observation is not None and source.zero_occupancy_omissions is not None
    )
    assert source.observation.constitution.resolve_atom_index(NZ) is None
    assert (
        restored.geometry.atom_geometry(index).position
        != source.zero_occupancy_omissions[0].geometry.position
    )
    assert repaired.repairs
    assert process_structure(restored, ingress=ingress).structure is restored
    assert process_structure(restored).structure.provenance.ingress is source
    for file_format in FileFormat:
        reread = read_structure_string(
            write_structure_string(restored, file_format), file_format
        )
        assert reread.constitution.resolve_atom_index(NZ) is not None
