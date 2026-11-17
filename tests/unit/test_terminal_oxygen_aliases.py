"""Terminal naming reconciliation preserves selected source evidence."""

from dataclasses import replace
from pathlib import Path

import gemmi
import pytest

from protrepair.api import process_structure
from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.diagnostics.kinds import IssueSeverity
from protrepair.errors import StructureNormalizationError
from protrepair.io import (
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.io.ingress_policy import (
    MutationPolicy,
    OccupancyPolicy,
    StructureNormalizationPolicy,
)
from protrepair.io.structure_ingress import (
    apply_structure_normalization_policy,
    normalize_raw_structure,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure import ProteinStructure
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.source_connection import SourceAtomIdentity, SourceConnection
from protrepair.structure.topology import (
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
)
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.workflow.contracts import requested_process_goal

RESIDUE = ResidueId("A", 1)
FAMILIES = (("OT1", "OT2"), ("OC2", "OC1"))


def _raw(
    names: tuple[str, ...] = ("OT1", "OT2"),
    *,
    hydrogen: str | None = None,
    isotope: str = "H",
    component: str = "ALA",
) -> gemmi.Structure:
    raw = gemmi.Structure()
    raw.name = "terminal"
    raw.add_model(gemmi.Model(1))
    raw[0].add_chain(gemmi.Chain("A"))
    residue = gemmi.Residue()
    residue.name = component
    residue.seqid = gemmi.SeqId(1, " ")
    residue.het_flag = "A"
    atoms = [
        ("N", "N", (0.0, 0.0, 0.0)),
        ("CA", "C", (1.458, 0.0, 0.0)),
        ("C", "C", (2.0, 1.4, 0.0)),
        ("CB", "C", (1.9, -0.8, -1.2)),
        *((name, "O", (2.8 - i, 2.0 + i, 0.0)) for i, name in enumerate(names)),
    ]
    if hydrogen is not None:
        atoms.append((hydrogen, isotope, (1.1, 3.5, 0.0)))
    for serial, (name, element, position) in enumerate(atoms, start=1):
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element("H" if element == "T" else element)
        atom.pos = gemmi.Position(*position)
        atom.occ = 0.75
        atom.b_iso = 21.5
        atom.serial = serial
        residue.add_atom(atom)
    raw[0][0].add_residue(residue)
    return raw


def _text(raw: gemmi.Structure, file_format: FileFormat) -> str:
    if file_format is FileFormat.PDB:
        return raw.make_pdb_string()
    raw.setup_entities()
    return raw.make_mmcif_document().as_string()


def _charged_text(
    raw: gemmi.Structure,
    file_format: FileFormat,
    charges: dict[str, int | None],
) -> str:
    if file_format is FileFormat.MMCIF:
        document = gemmi.cif.read_string(_text(raw, file_format))
        values = document.sole_block().find_values("_atom_site.pdbx_formal_charge")
        for index, atom in enumerate(raw[0][0][0]):
            charge = charges.get(atom.name)
            values[index] = "?" if charge is None else str(charge)
        return document.as_string()

    lines = []
    for line in _text(raw, file_format).splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            charge = charges.get(line[12:16].strip())
            token = (
                "  " if charge is None else f"{abs(charge)}{'-' if charge < 0 else '+'}"
            )
            line = line[:78] + token + line[80:]
        lines.append(line)
    return "\n".join(lines) + "\n"


def _normalize(
    raw: gemmi.Structure,
    connections: tuple[SourceConnection, ...] = (),
    *,
    policy: StructureNormalizationPolicy | None = None,
    isotope: str | None = None,
) -> ProteinStructure:
    return normalize_raw_structure(
        raw,
        file_format=FileFormat.PDB,
        policy=StructureNormalizationPolicy() if policy is None else policy,
        source_connections=connections,
        source_element_by_atom_identity=(
            {
                SourceAtomIdentity(
                    AtomRef(RESIDUE, raw[0][0][0][-1].name), "ALA"
                ): isotope
            }
            if isotope is not None
            else None
        ),
    )


def _connection(
    first: str,
    second: str,
    *,
    record: SourceBondRecordType = SourceBondRecordType.PDB_LINK,
    altloc: str | None = None,
    component: str = "ALA",
) -> SourceConnection:
    return SourceConnection(
        SourceAtomIdentity(AtomRef(RESIDUE, first), component, altloc),
        SourceAtomIdentity(AtomRef(RESIDUE, second), component, altloc),
        SourceBondMetadata(
            record_type=record,
            reported_relationship_type=BondRelationshipType.COVALENT,
            reported_order=1,
        ),
    )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("names", FAMILIES)
def test_complete_pair_normalizes_before_completion_and_source_capture(
    file_format: FileFormat, names: tuple[str, str]
) -> None:
    raw = _raw(names)
    canonical = read_structure_string(
        _text(_raw(("O", "OXT")), file_format), file_format
    )
    original_names = [atom.name for atom in raw[0][0][0]]
    source = read_structure_string(_text(raw, file_format), file_format)
    assert source.constitution == canonical.constitution
    assert source.geometry == canonical.geometry
    assert source.topology == canonical.topology
    assert [atom.name for atom in raw[0][0][0]] == original_names

    observation = source.provenance.ingress.observation
    assert observation is not None
    assert observation.constitution == source.constitution
    assert observation.geometry == source.geometry
    completed = repair_heavy_atoms_core(source).structure
    assert completed.constitution == source.constitution
    assert completed.geometry == source.geometry
    assert (
        apply_structure_normalization_policy(
            source, policy=StructureNormalizationPolicy()
        )
        is source
    )

    restored = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    assert restored.constitution == source.constitution
    assert restored.geometry == source.geometry


@pytest.mark.parametrize("names", FAMILIES)
@pytest.mark.parametrize("charge", (None, 0, -1))
def test_mmcif_charge_declarations_follow_the_physical_oxygen(
    names: tuple[str, str], charge: int | None
) -> None:
    raw = _raw(names)
    raw.setup_entities()
    document = raw.make_mmcif_document()
    charges = document.sole_block().find_values("_atom_site.pdbx_formal_charge")
    for i in range(len(charges)):
        charges[i] = "?"
    charges[4] = "?" if charge is None else str(charge)
    source = read_structure_string(document.as_string(), FileFormat.MMCIF)
    ref = AtomRef(RESIDUE, "OXT" if charge == -1 else "O")
    assert source.topology.formal_charge(source.constitution.atom_index(ref)) == charge
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert observation.formal_charge(ref) == charge
    assert (
        observation.formal_charge(AtomRef(RESIDUE, "O" if charge == -1 else "OXT"))
        is None
    )


@pytest.mark.parametrize(
    "names",
    (
        ("OT1",),
        ("OT2",),
        ("OC1",),
        ("OC2",),
        ("OT1", "OC2"),
        ("OC1", "OT2"),
        ("OT1", "OT2", "O"),
        ("OT1", "OT2", "OXT"),
        ("OC1", "OC2", "OT1", "OT2"),
    ),
)
def test_incomplete_or_mixed_names_are_not_a_terminal_state_claim(
    names: tuple[str, ...],
) -> None:
    with pytest.raises(StructureNormalizationError, match="terminal oxygen"):
        _normalize(_raw(names))


def test_nonoxygen_alias_is_refused() -> None:
    raw = _raw()
    raw[0][0][0]["OT1"][0].element = gemmi.Element("N")
    with pytest.raises(StructureNormalizationError, match="terminal oxygen"):
        _normalize(raw)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("parent_name", ("O", "OXT"))
@pytest.mark.parametrize("declared", (False, True))
@pytest.mark.parametrize("charged", (False, True))
def test_name_dependent_terminal_hydrogens_are_refused_even_with_a_bond(
    isotope: str, parent_name: str, declared: bool, charged: bool
) -> None:
    hydrogen = isotope + ("O" if parent_name == "O" else "XT")
    raw = _raw(hydrogen=hydrogen, isotope=isotope)
    if charged:
        raw[0][0][0]["OT1"][0].charge = -1
    connections = (_connection("OT1", hydrogen),) if declared else ()
    with pytest.raises(StructureNormalizationError, match="terminal oxygen"):
        _normalize(raw, connections, isotope=isotope)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("names", FAMILIES)
@pytest.mark.parametrize("oxygen", (0, 1))
def test_explicit_non_grammar_h_parent_survives_the_bijection(
    isotope: str, names: tuple[str, str], oxygen: int
) -> None:
    source = _normalize(
        _raw(names, hydrogen="QH", isotope=isotope),
        (_connection(names[oxygen], "QH"),),
        isotope=isotope,
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    for constitution, topology in (
        (source.constitution, source.topology),
        (observation.constitution, observation.topology),
    ):
        parent = constitution.atom_index(AtomRef(RESIDUE, ("O", "OXT")[oxygen]))
        hydrogen = constitution.atom_index(AtomRef(RESIDUE, "QH"))
        bond = topology.bond_between(parent, hydrogen)
        assert bond is not None
        assert bond.relationship_type is BondRelationshipType.COVALENT
        assert bond.order == 1
        assert constitution.atom_slots[hydrogen.value].element == isotope


@pytest.mark.parametrize("record", tuple(SourceBondRecordType)[1:])
@pytest.mark.parametrize("names", FAMILIES)
def test_projected_connection_keeps_reported_bond_order(
    record: SourceBondRecordType, names: tuple[str, str]
) -> None:
    connection = _connection("C", names[0], record=record)
    connection = replace(
        connection,
        source_metadata=replace(connection.source_metadata, reported_order=2),
    )
    source = _normalize(_raw(names), (connection,))
    observation = source.provenance.ingress.observation
    assert observation is not None
    for constitution, topology in (
        (source.constitution, source.topology),
        (observation.constitution, observation.topology),
    ):
        bond = topology.bond_between(
            constitution.atom_index(AtomRef(RESIDUE, "C")),
            constitution.atom_index(AtomRef(RESIDUE, "O")),
        )
        assert bond is not None
        assert bond.order == 2
        assert bond.source_metadata == connection.source_metadata


@pytest.mark.parametrize("selected", (None, ("A",), ("B",)))
def test_aliases_in_canonical_objects_are_refused_before_both_selection_paths(
    selected: tuple[str, ...] | None,
) -> None:
    raw = _raw(("O", "OXT"))
    raw[0].add_chain(gemmi.Chain("B"))
    raw[0]["B"].add_residue(_raw(("O", "OXT"))[0][0][0])
    source = read_structure_string(_text(raw, FileFormat.PDB), FileFormat.PDB)
    residue = source.constitution.chains[0].residues[0]
    atoms = tuple(
        replace(atom, name={"O": "OT1", "OXT": "OT2"}.get(atom.name, atom.name))
        for atom in residue.atom_sites
    )
    constitution = replace(
        source.constitution,
        chains=(
            replace(
                source.constitution.chains[0],
                residues=(replace(residue, atom_sites=atoms),),
            ),
            source.constitution.chains[1],
        ),
    )
    aliased = ProteinStructure.from_payload(
        constitution=constitution,
        geometry=StructureGeometry(
            constitution=constitution, atom_geometries=source.geometry.atom_geometries
        ),
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=source.topology.bonds,
        ),
        provenance=source.provenance,
    )
    if selected == ("B",):
        result = apply_structure_normalization_policy(
            aliased, policy=StructureNormalizationPolicy(selected_chain_ids=selected)
        )
        assert result.constitution.chain_ids() == ("B",)
    else:
        with pytest.raises(StructureNormalizationError, match="canonical"):
            apply_structure_normalization_policy(
                aliased,
                policy=StructureNormalizationPolicy(selected_chain_ids=selected),
            )
    with pytest.raises(StructureNormalizationError, match="canonical"):
        process_structure(aliased)


def test_public_path_processing_uses_the_same_normalization(tmp_path: Path) -> None:
    path = tmp_path / "terminal.pdb"
    path.write_text(_text(_raw(), FileFormat.PDB))
    result = process_structure(path)
    residue = result.structure.constitution.chains[0].residues[0]
    assert set(residue.atom_site_names()) == {"N", "CA", "C", "CB", "O", "OXT"}


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_serialized_link_endpoints_are_not_silently_dropped(
    file_format: FileFormat,
) -> None:
    raw = _raw()
    second = _raw(("O", "OXT"))[0][0][0]
    second.seqid = gemmi.SeqId(2, " ")
    raw[0][0].add_residue(second)
    connection = gemmi.Connection()
    connection.name = "oxygen"
    connection.type = gemmi.ConnectionType.Covale
    connection.partner1 = gemmi.AtomAddress("A", gemmi.SeqId(2, " "), "ALA", "CA")
    connection.partner2 = gemmi.AtomAddress("A", gemmi.SeqId(1, " "), "ALA", "OT1")
    raw.connections.append(connection)
    source = read_structure_string(_text(raw, file_format), file_format)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert len(observation.topology.bonds) == 1
    bond = observation.topology.bonds[0]
    assert {
        observation.constitution.atom_ref_at(index).atom_name
        for index in bond.endpoint_pair()
    } == {"CA", "O"}
    assert bond.relationship_type is BondRelationshipType.COVALENT
    assert bond.source_metadata is not None
    assert bond.source_metadata.record_type is (
        SourceBondRecordType.PDB_LINK
        if file_format is FileFormat.PDB
        else SourceBondRecordType.MMCIF_STRUCT_CONN
    )


def test_serialized_conect_retains_explicit_order_and_selected_names() -> None:
    text = _text(_raw(), FileFormat.PDB)
    text = (
        "\n".join(line for line in text.splitlines() if not line.startswith("END"))
        + "\nCONECT    3    5    5\nEND\n"
    )
    source = read_structure_string(text, FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert len(observation.topology.bonds) == 1
    bond = observation.topology.bonds[0]
    assert bond.order == 2
    assert {
        observation.constitution.atom_ref_at(index).atom_name
        for index in bond.endpoint_pair()
    } == {"C", "O"}
    assert bond.source_metadata is not None
    assert bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT


@pytest.mark.parametrize(
    "record",
    (
        SourceBondRecordType.PDB_LINK,
        SourceBondRecordType.PDB_CONECT,
        SourceBondRecordType.MMCIF_STRUCT_CONN,
        SourceBondRecordType.PDB_SSBOND,
    ),
)
@pytest.mark.parametrize("altloc", (None, "A", "B"))
def test_endpoint_projection_preserves_exact_and_wildcard_altloc_rules(
    record: SourceBondRecordType, altloc: str | None
) -> None:
    raw = _raw()
    for atom in raw[0][0][0]:
        atom.altloc = "A"
    alternate = raw[0][0][0]["OT1"][0].clone()
    alternate.altloc = "B"
    alternate.occ = 0.1
    alternate.pos.x = 100
    raw[0][0][0].add_atom(alternate)
    source = _normalize(raw, (_connection("C", "OT1", record=record, altloc=altloc),))
    observation = source.provenance.ingress.observation
    assert observation is not None
    expected = altloc == "A" or (record is SourceBondRecordType.PDB_SSBOND)
    assert len(observation.topology.bonds) == int(expected)
    geometry = source.geometry.atom_geometry(
        source.constitution.atom_index(AtomRef(RESIDUE, "O"))
    )
    assert geometry.position.x == 2.8
    assert geometry.altloc == "A"


def test_discarded_component_connection_is_not_rebound_to_the_selected_variant() -> (
    None
):
    raw = _raw()
    discarded = _raw(("OT1",), component="GLY")[0][0][0]
    for atom in discarded:
        atom.occ = 0.1
    raw[0][0].add_residue(discarded)
    source = _normalize(raw, (_connection("C", "OT1", component="GLY"),))
    assert source.constitution.chains[0].residues[0].component_id == "ALA"
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert not observation.topology.bonds
    with pytest.raises(StructureNormalizationError, match="terminal oxygen"):
        _normalize(
            raw,
            policy=StructureNormalizationPolicy(
                mutation_policy=MutationPolicy.LOWEST_OCCUPANCY
            ),
        )


def test_pair_admission_runs_after_coherent_altloc_selection() -> None:
    raw = _raw()
    for atom in raw[0][0][0]:
        atom.altloc = "A"
    for atom in _raw(("OC1", "OC2"))[0][0][0]:
        if atom.name.startswith("OC"):
            atom.altloc = "B"
            atom.occ = 0.1
            raw[0][0][0].add_atom(atom)
    source = _normalize(raw)
    assert set(source.constitution.chains[0].residues[0].atom_site_names()) == {
        "N",
        "CA",
        "C",
        "CB",
        "O",
        "OXT",
    }


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_hetatm_spelling_does_not_override_the_selected_polymer_role(
    file_format: FileFormat,
) -> None:
    raw = _raw(component="MSE")
    raw[0][0][0].het_flag = "H"
    neighbor = _raw(("O", "OXT"))[0][0][0]
    neighbor.seqid = gemmi.SeqId(0, " ")
    raw[0][0].add_residue(neighbor)
    source = read_structure_string(_text(raw, file_format), file_format)
    residue = source.constitution.residue_or_ligand(RESIDUE)
    assert residue is not None
    assert not residue.is_hetero
    assert "OXT" in residue.atom_site_names()


def test_retained_ligand_and_unrecognized_numbered_oxygens_are_not_reinterpreted() -> (
    None
):
    raw = _raw(component="LIG")
    raw[0][0][0].het_flag = "H"
    source = _normalize(raw)
    assert set(source.constitution.ligands[0].atom_site_names()) >= {"OT1", "OT2"}
    numbered = _normalize(_raw(("O1", "O2")))
    assert set(numbered.constitution.chains[0].residues[0].atom_site_names()) >= {
        "O1",
        "O2",
    }


def test_names_owned_by_the_component_template_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = build_default_component_library()
    template = library.require("ALA")
    custom = replace(
        template,
        definition=replace(
            template.definition,
            atom_names=(*template.definition.atom_names, "OT1", "OT2"),
        ),
    )
    monkeypatch.setattr(
        "protrepair.io.structure_ingress.build_default_component_library",
        lambda: library.with_template(custom),
    )
    source = _normalize(_raw())
    assert set(source.constitution.chains[0].residues[0].atom_site_names()) >= {
        "OT1",
        "OT2",
    }


def test_ordinary_h_and_unknown_h_names_do_not_select_a_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("naming reconciliation must not resolve a microstate")

    monkeypatch.setattr(PolymerMicrostateSite, "resolve", forbidden)
    for name in ("H", "HA", "HB1", "HT2"):
        source = _normalize(_raw(hydrogen=name))
        assert name in source.constitution.chains[0].residues[0].atom_site_names()


def test_source_hydrogen_isotope_survives_the_public_reader() -> None:
    text = _text(_raw(hydrogen="QH"), FileFormat.PDB)
    lines = text.splitlines()
    lines = [
        line[:76] + " T" + line[78:]
        if line.startswith("ATOM") and line[12:16].strip() == "QH"
        else line
        for line in lines
    ]
    source = read_structure_string("\n".join(lines) + "\n", FileFormat.PDB)
    assert source.constitution.atom_slots[-1].element == "T"


def test_oxt_is_normalized_before_sequence_link_inference() -> None:
    raw = _raw()
    second = _raw(("O",))[0][0][0]
    second.seqid = gemmi.SeqId(2, " ")
    second["N"][0].pos = gemmi.Position(3.3, 1.4, 0)
    raw[0][0].add_residue(second)
    canonical = raw.clone()
    canonical[0][0][0]["OT1"][0].name = "O"
    canonical[0][0][0]["OT2"][0].name = "OXT"
    normalized = _normalize(raw)
    assert normalized.topology == _normalize(canonical).topology
    left = normalized.constitution.atom_index(AtomRef(RESIDUE, "C"))
    right = normalized.constitution.atom_index(AtomRef(ResidueId("A", 2), "N"))
    assert normalized.topology.bond_between(left, right) is None

    canonical[0][0][0].remove_atom("OXT", "\x00")
    continuing = _normalize(canonical)
    assert (
        continuing.topology.bond_between(
            continuing.constitution.atom_index(AtomRef(RESIDUE, "C")),
            continuing.constitution.atom_index(AtomRef(ResidueId("A", 2), "N")),
        )
        is not None
    )


def test_excluded_coordination_keeps_raw_names_but_surviving_endpoints_are_projected() -> (
    None
):
    raw = _raw()
    raw[0].add_chain(gemmi.Chain("Z"))
    metal = gemmi.Residue()
    metal.name = "ZN"
    metal.seqid = gemmi.SeqId(1, " ")
    metal.het_flag = "H"
    atom = gemmi.Atom()
    atom.name = "ZN"
    atom.element = gemmi.Element("Zn")
    metal.add_atom(atom)
    raw[0]["Z"].add_residue(metal)
    connection = SourceConnection(
        SourceAtomIdentity(AtomRef(RESIDUE, "OT1"), "ALA"),
        SourceAtomIdentity(AtomRef(ResidueId("Z", 1), "ZN"), "ZN"),
        SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_LINK,
            reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
        ),
    )
    selected = _normalize(
        raw,
        (connection,),
        policy=StructureNormalizationPolicy(selected_chain_ids=("Z",)),
    )
    observation = selected.provenance.ingress.observation
    assert observation is not None
    assert observation.excluded_coordination_connections == (connection,)

    complete = _normalize(raw, (connection,))
    observation = complete.provenance.ingress.observation
    assert observation is not None
    assert not observation.excluded_coordination_connections
    assert len(observation.topology.bonds) == 1
    assert {
        observation.constitution.atom_ref_at(index).atom_name
        for index in observation.topology.bonds[0].endpoint_pair()
    } == {"O", "ZN"}


def test_normalization_does_not_construct_a_force_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("ingress must not build a force field")

    monkeypatch.setattr(
        "protrepair.transformer.continuous.rdkit.build_rdkit_force_field", forbidden
    )
    source = _normalize(_raw())
    assert "OXT" in source.constitution.chains[0].residues[0].atom_site_names()


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("names", FAMILIES)
@pytest.mark.parametrize(
    ("charges", "expected"),
    (
        ((-1, None), ("OXT", "O")),
        ((None, -1), ("O", "OXT")),
        ((-1, 0), ("OXT", "O")),
        ((0, -1), ("O", "OXT")),
        ((None, None), ("O", "OXT")),
        ((0, 0), ("O", "OXT")),
        ((-1, -1), ("O", "OXT")),
        ((-2, 0), ("O", "OXT")),
        ((-1, 1), ("O", "OXT")),
        ((1, -1), ("O", "OXT")),
    ),
)
def test_declared_charge_selects_labels_without_changing_physical_facts(
    file_format: FileFormat,
    names: tuple[str, str],
    charges: tuple[int | None, int | None],
    expected: tuple[str, str],
) -> None:
    raw = _raw(names)
    text = _charged_text(raw, file_format, dict(zip(names, charges, strict=True)))
    wire = (
        gemmi.read_pdb_string(text)
        if file_format is FileFormat.PDB
        else gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block())
    )
    source = read_structure_string(text, file_format)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert source.constitution == observation.constitution
    assert source.constitution.chains[0].residues[0].atom_site_names() == (
        "N",
        "CA",
        "C",
        "CB",
        *expected,
    )
    for original, name, charge in zip(
        tuple(wire[0][0][0])[-2:], expected, charges, strict=True
    ):
        ref = AtomRef(RESIDUE, name)
        index = source.constitution.atom_index(ref)
        geometry = source.geometry.atom_geometry(index)
        assert tuple(geometry.position) == (
            original.pos.x,
            original.pos.y,
            original.pos.z,
        )
        assert geometry.occupancy == original.occ
        assert geometry.b_factor == original.b_iso
        assert source.topology.formal_charge(index) == charge
        assert observation.formal_charge(ref) == charge


@pytest.mark.parametrize("names", FAMILIES)
@pytest.mark.parametrize("order", (1, 2))
def test_charge_mapping_preserves_source_orders_and_their_conflicts(
    names: tuple[str, str], order: int
) -> None:
    raw = _raw(names)
    raw[0][0][0][names[0]][0].charge = -1
    connection = _connection("C", names[0])
    connection = replace(
        connection,
        source_metadata=replace(connection.source_metadata, reported_order=order),
    )
    source = _normalize(raw, (connection,))
    observation = source.provenance.ingress.observation
    assert observation is not None
    for constitution, topology in (
        (source.constitution, source.topology),
        (observation.constitution, observation.topology),
    ):
        bond = topology.bond_between(
            constitution.atom_index(AtomRef(RESIDUE, "C")),
            constitution.atom_index(AtomRef(RESIDUE, "OXT")),
        )
        assert bond is not None
        assert bond.order == order
        assert bond.source_metadata == connection.source_metadata
    preparation = AminoAcidPreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    target = next(
        t
        for t in preparation.targets_for(RESIDUE)
        if t.site.kind is PolymerChemicalSite.BACKBONE_C
    )
    expected = (
        MicrostateResolutionStatus.RESOLVED
        if order == 1
        else MicrostateResolutionStatus.CONFLICT
    )
    assert target.resolution.status is expected


@pytest.mark.parametrize("names", FAMILIES)
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_charge_mapping_keeps_explicit_h_parent_even_when_chemistry_conflicts(
    names: tuple[str, str], isotope: str
) -> None:
    raw = _raw(names, hydrogen="QH", isotope=isotope)
    raw[0][0][0][names[0]][0].charge = -1
    connection = _connection(names[0], "QH")
    source = _normalize(raw, (connection,), isotope=isotope)
    observation = source.provenance.ingress.observation
    assert observation is not None
    for constitution, topology in (
        (source.constitution, source.topology),
        (observation.constitution, observation.topology),
    ):
        bond = topology.bond_between(
            constitution.atom_index(AtomRef(RESIDUE, "OXT")),
            constitution.atom_index(AtomRef(RESIDUE, "QH")),
        )
        assert bond is not None
        assert bond.source_metadata == connection.source_metadata
    preparation = AminoAcidPreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    target = next(
        t
        for t in preparation.targets_for(RESIDUE)
        if t.site.kind is PolymerChemicalSite.BACKBONE_C
    )
    assert target.resolution.status is MicrostateResolutionStatus.CONFLICT


@pytest.mark.parametrize("selection", (OccupancyPolicy.HIGHEST, OccupancyPolicy.LOWEST))
def test_charge_mapping_uses_only_the_selected_altloc(
    selection: OccupancyPolicy,
) -> None:
    raw = _raw()
    for atom in raw[0][0][0]:
        atom.altloc = "A"
    raw[0][0][0]["OT1"][0].charge = -1
    alternative = _raw()
    alternative[0][0][0]["OT2"][0].charge = -1
    for atom in alternative[0][0][0]:
        atom.altloc = "B"
        atom.occ = 0.1
        atom.pos.x += 10
        raw[0][0][0].add_atom(atom)
    source = _normalize(
        raw, policy=StructureNormalizationPolicy(occupancy_policy=selection)
    )
    index = source.constitution.atom_index(AtomRef(RESIDUE, "OXT"))
    assert source.topology.formal_charge(index) == -1
    geometry = source.geometry.atom_geometry(index)
    expected = (
        ("A", (2.8, 2.0, 0.0))
        if selection is OccupancyPolicy.HIGHEST
        else ("B", (11.8, 3.0, 0.0))
    )
    assert (geometry.altloc, tuple(geometry.position)) == expected


@pytest.mark.parametrize("component", ("ALA", "GLY"))
@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("names", FAMILIES)
@pytest.mark.parametrize("charged", (0, 1))
def test_charge_disambiguated_peptide_has_coherent_hydrogens_and_native_graph(
    component: str, file_format: FileFormat, names: tuple[str, str], charged: int
) -> None:
    raw = _raw(names, component=component)
    residue = raw[0][0][0]
    if component == "GLY":
        residue.remove_atom("CB", "\x00")
    residue[names[0]][0].pos = gemmi.Position(3.227, 1.618, 0)
    residue[names[1]][0].pos = gemmi.Position(1.152, 2.321, 0)
    source = read_structure_string(
        _charged_text(raw, file_format, {names[charged]: -1, names[1 - charged]: 0}),
        file_format,
    )
    result = process_structure(
        source,
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].status.value == "satisfied"
    assert not [
        issue for issue in result.issues if issue.severity is IssueSeverity.ERROR
    ]
    native = to_rdkit_molecule(result.structure)
    for name, charge, order in (("O", 0, 2), ("OXT", -1, 1)):
        ref = AtomRef(RESIDUE, name)
        index = result.structure.constitution.atom_index(ref)
        carbon = result.structure.constitution.atom_index(AtomRef(RESIDUE, "C"))
        assert native.GetAtomWithIdx(index.value).GetFormalCharge() == charge
        bond = native.GetBondBetweenAtoms(carbon.value, index.value)
        assert bond is not None
        assert bond.GetBondTypeAsDouble() == order
        assert result.structure.geometry.atom_geometry(
            index
        ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))
