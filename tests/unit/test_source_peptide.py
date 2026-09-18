"""Source sequence evidence must not be confused with author numbering."""

import gemmi
import pytest

from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import BondProvenance


def _source(
    labels: tuple[str, ...] = ("1B", "1A", "1"),
    *,
    sequence: tuple[str, ...] = ("ALA", "GLY", "SER"),
    positions: tuple[int, ...] | None = None,
) -> gemmi.Structure:
    source = gemmi.Structure()
    chain = gemmi.Chain("A")
    for offset, (label, name) in enumerate(
        zip(labels, ("ALA", "GLY", "SER"), strict=True)
    ):
        residue = gemmi.Residue()
        residue.name = name
        number = label.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        residue.seqid = gemmi.SeqId(int(number), label[len(number) :] or " ")
        residue.het_flag = "A"
        residue.subchain = "Axp"
        residue.entity_id = "1"
        residue.entity_type = gemmi.EntityType.Polymer
        if positions is not None:
            residue.label_seq = positions[offset]
        for atom_name, element, dx in (("N", "N", 0), ("CA", "C", 1), ("C", "C", 2)):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(offset * 3.33 + dx, 0, 0)
            residue.add_atom(atom)
        chain.add_residue(residue)
    model = gemmi.Model(1)
    model.add_chain(chain)
    source.add_model(model)
    if sequence:
        entity = gemmi.Entity("1")
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.subchains = ["Axp"]
        entity.full_sequence = list(sequence)
        source.entities.append(entity)
    return source


def _read(source: gemmi.Structure, file_format: FileFormat):
    text = (
        source.make_pdb_string()
        if file_format is FileFormat.PDB
        else source.make_mmcif_document().as_string()
    )
    return read_structure_string(text, file_format)


def _peptide_pairs(structure):
    return {
        frozenset(
            structure.constitution.atom_ref_at(index) for index in bond.endpoint_pair()
        )
        for bond in structure.topology.bonds
        if structure.constitution.atom_ref_at(bond.atom_index_1).residue_id
        != structure.constitution.atom_ref_at(bond.atom_index_2).residue_id
    }


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize(
    "labels", (("1B", "1A", "1"), ("183", "184A", "184"), ("217", "219", "220"))
)
def test_complete_source_sequence_resolves_nonmonotonic_author_labels(
    file_format, labels
):
    source = _source(labels, positions=(1, 2, 3))
    structure = _read(source, file_format)
    ids = tuple(
        residue.residue_id for residue in structure.constitution.chains[0].residues
    )
    assert tuple(residue.display_token() for residue in ids) == tuple(
        f"A:{label}" for label in labels
    )
    expected = {
        frozenset((AtomRef(first, "C"), AtomRef(second, "N")))
        for first, second in zip(ids, ids[1:], strict=False)
    }
    assert _peptide_pairs(structure) == expected
    assert all(
        bond.provenance is BondProvenance.SEQUENCE_INFERRED
        for bond in structure.topology.bonds
        if frozenset(
            structure.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
        )
        in expected
    )
    observation = structure.provenance.ingress.observation
    assert observation is not None
    assert not observation.topology.bonds
    reread = read_structure_string(
        write_structure_string(structure, file_format), file_format
    )
    assert _peptide_pairs(reread) == expected


@pytest.mark.parametrize(
    "sequence", ((), ("ALA", "GLY", "GLY", "SER"), ("ALA", "ASN", "SER"))
)
def test_incomplete_or_mismatching_sequence_does_not_authorize_numbering_recovery(
    sequence,
):
    structure = _read(_source(sequence=sequence), FileFormat.PDB)
    assert not _peptide_pairs(structure)


def test_complete_seqres_without_ter_still_supplies_sequence_positions():
    text = (
        "\n".join(
            line
            for line in _source().make_pdb_string().splitlines()
            if not line.startswith("TER")
        )
        + "\n"
    )
    structure = read_structure_string(text, FileFormat.PDB)
    assert len(_peptide_pairs(structure)) == 2


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_source_component_casing_uses_canonical_identity(file_format):
    source = _source(positions=(1, 2, 3))
    for residue in source[0][0]:
        residue.name = residue.name.lower()
    structure = _read(source, file_format)
    assert len(_peptide_pairs(structure)) == 2
    assert tuple(
        site.component_id for site in structure.constitution.chains[0].residues
    ) == ("ALA", "GLY", "SER")


def test_declared_sequence_gap_overrides_consecutive_author_numbers():
    structure = _read(
        _source(
            ("1", "2", "3"), sequence=("ALA", "ASN", "GLY", "SER"), positions=(1, 3, 4)
        ),
        FileFormat.MMCIF,
    )
    assert _peptide_pairs(structure) == {
        frozenset((AtomRef(ResidueId("A", 2), "C"), AtomRef(ResidueId("A", 3), "N")))
    }


@pytest.mark.parametrize("distance", (0.0, 1.801, 20.0))
def test_numbering_recovery_needs_peptide_geometry(distance):
    source = _source()
    source[0][0][1]["N"][0].pos.x = 2 + distance
    structure = _read(source, FileFormat.PDB)
    assert frozenset(
        (AtomRef(ResidueId("A", 1, "B"), "C"), AtomRef(ResidueId("A", 1, "A"), "N"))
    ) not in _peptide_pairs(structure)


def test_spatially_close_non_neighbors_do_not_become_peptide_partners():
    source = _source()
    source[0][0][2]["N"][0].pos.x = 3.33
    structure = _read(source, FileFormat.PDB)
    assert frozenset(
        (AtomRef(ResidueId("A", 1, "B"), "C"), AtomRef(ResidueId("A", 1), "N"))
    ) not in _peptide_pairs(structure)


@pytest.mark.parametrize("field", ("entity_id", "subchain"))
def test_explicit_entity_or_asym_boundary_prevents_inference(field):
    source = _source(("1", "2", "3"), positions=(1, 2, 3))
    setattr(source[0][0][1], field, "other")
    document = source.make_mmcif_document()
    if field == "entity_id":
        column = document.sole_block().find_values("_atom_site.label_entity_id")
        for index in range(3, 6):
            column[index] = "other"
    structure = read_structure_string(document.as_string(), FileFormat.MMCIF)
    assert not _peptide_pairs(structure)


def test_pdb_ter_boundary_is_not_erased_by_consecutive_author_labels():
    source = _source(("1", "2", "3"))
    text = source.make_pdb_string()
    lines = []
    for line in text.splitlines():
        if (
            line.startswith("ATOM")
            and line[22:26].strip() == "2"
            and line[12:16].strip() == "N"
        ):
            lines.append("TER")
        lines.append(line)
    structure = read_structure_string("\n".join(lines) + "\n", FileFormat.PDB)
    assert frozenset(
        (AtomRef(ResidueId("A", 1), "C"), AtomRef(ResidueId("A", 2), "N"))
    ) not in _peptide_pairs(structure)


@pytest.mark.parametrize("labels", (("1B", "1A", "1"), ("1", "2", "3")))
def test_terminal_oxygen_blocks_inferred_peptide_attachment(labels):
    source = _source(labels)
    oxygen = gemmi.Atom()
    oxygen.name = "OXT"
    oxygen.element = gemmi.Element("O")
    oxygen.pos = gemmi.Position(2, 1.2, 0)
    source[0][0][0].add_atom(oxygen)
    structure = _read(source, FileFormat.PDB)
    assert len(_peptide_pairs(structure)) == 1


def _crosslink(
    source: gemmi.Structure, kind: gemmi.ConnectionType, *, altloc: str = "\0"
) -> None:
    connection = gemmi.Connection()
    connection.name = "external"
    connection.type = kind
    connection.partner1.chain_name = "A"
    connection.partner1.res_id = source[0][0][0]
    connection.partner1.atom_name = "C"
    connection.partner1.altloc = altloc
    connection.partner2.chain_name = "A"
    connection.partner2.res_id = source[0][0][2]
    connection.partner2.atom_name = "N"
    source.connections.append(connection)


@pytest.mark.parametrize(
    "kind,expected_count",
    ((gemmi.ConnectionType.Covale, 1), (gemmi.ConnectionType.Hydrog, 3)),
)
def test_source_crosslink_occupies_valence_but_hydrogen_bond_does_not(
    kind, expected_count
):
    source = _source(positions=(1, 2, 3))
    _crosslink(source, kind)
    structure = _read(source, FileFormat.MMCIF)
    assert len(_peptide_pairs(structure)) == expected_count
    explicit = [
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    ]
    assert len(explicit) == 1


def test_discarded_altloc_crosslink_does_not_block_selected_peptide():
    source = _source(positions=(1, 2, 3))
    carbon = source[0][0][0]["C"][0]
    carbon.altloc = "A"
    carbon.occ = 0.8
    other = carbon.clone()
    other.altloc = "B"
    other.occ = 0.2
    source[0][0][0].add_atom(other)
    _crosslink(source, gemmi.ConnectionType.Covale, altloc="B")
    structure = _read(source, FileFormat.MMCIF)
    assert len(_peptide_pairs(structure)) == 2
    assert all(
        bond.provenance is not BondProvenance.SOURCE_EXPLICIT
        for bond in structure.topology.bonds
    )


def test_explicit_peptide_order_is_not_overridden_by_sequence_inference():
    source = _source(positions=(1, 2, 3))
    _crosslink(source, gemmi.ConnectionType.Covale)
    source.connections[0].partner2.res_id = source[0][0][1]
    document = source.make_mmcif_document()
    block = document.sole_block()
    category = block.get_mmcif_category("_struct_conn.")
    category["pdbx_value_order"] = ["doub"]
    block.set_mmcif_category("_struct_conn.", category)
    structure = read_structure_string(document.as_string(), FileFormat.MMCIF)
    explicit = [
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    ]
    assert len(explicit) == 1
    assert explicit[0].order == 2
    assert explicit[0].source_metadata is not None
    assert explicit[0].source_metadata.reported_order == 2


def test_source_carbonyl_hydrogen_cap_blocks_inferred_peptide():
    source = _source(positions=(1, 2, 3))
    hydrogen = gemmi.Atom()
    hydrogen.name = "HC"
    hydrogen.element = gemmi.Element("H")
    hydrogen.pos = gemmi.Position(2, 1.1, 0)
    source[0][0][0].add_atom(hydrogen)
    _crosslink(source, gemmi.ConnectionType.Covale)
    source.connections[0].partner2.res_id = source[0][0][0]
    source.connections[0].partner2.atom_name = "HC"
    structure = _read(source, FileFormat.MMCIF)
    assert len(_peptide_pairs(structure)) == 1
    assert any(
        bond.provenance is BondProvenance.SOURCE_EXPLICIT
        for bond in structure.topology.bonds
    )


def _source_with_n_attachments(
    names: tuple[str, ...],
    *,
    component: str = "GLY",
    charge: int = 0,
    kind: gemmi.ConnectionType = gemmi.ConnectionType.Covale,
) -> gemmi.Structure:
    source = _source(positions=(1, 2, 3))
    residue = source[0][0][1]
    residue.name = component
    residue["N"][0].charge = charge
    source.entities[0].full_sequence = ["ALA", component, "SER"]
    for name in names:
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element("D" if name.startswith("D") else name[0])
        atom.pos = gemmi.Position(3.33, 1.0, 0.0)
        residue.add_atom(atom)

        connection = gemmi.Connection()
        connection.name = f"local-{name}"
        connection.type = kind
        for partner, atom_name in (
            (connection.partner1, "N"),
            (connection.partner2, name),
        ):
            partner.chain_name = "A"
            partner.res_id = residue
            partner.atom_name = atom_name
        source.connections.append(connection)
    return source


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize(
    "component,names,charge,accept",
    (
        ("GLY", ("H",), 0, True),
        ("GLY", ("H", "CB"), 0, False),
        ("GLY", ("CB",), 0, False),
        ("GLY", ("H1", "H2"), 0, False),
        ("GLY", ("H1", "H2"), 1, True),
        ("GLY", ("H1", "H2", "H3"), 1, False),
        ("GLY", ("D1",), 0, True),
        ("GLY", ("D1", "D2"), 0, False),
        ("PRO", ("CD",), 0, True),
        ("PRO", ("CD", "H"), 0, False),
        ("HYP", ("CD",), 0, True),
    ),
)
def test_source_n_attachments_constrain_inferred_peptide(
    file_format, component, names, charge, accept
):
    source = _source_with_n_attachments(names, component=component, charge=charge)
    if file_format is FileFormat.PDB:
        text = source.make_pdb_string()
        serials = {
            line[12:16].strip(): int(line[6:11])
            for line in text.splitlines()
            if line.startswith("ATOM") and line[22:27].strip() == "1A"
        }
        records = "".join(
            f"CONECT{serials['N']:5d}{serials[name]:5d}\n" for name in names
        )
        structure = read_structure_string(
            text.replace("END\n", "") + records + "END\n", file_format
        )
    else:
        structure = _read(source, file_format)
    nitrogen = AtomRef(ResidueId("A", 1, "A"), "N")
    candidate = frozenset((AtomRef(ResidueId("A", 1, "B"), "C"), nitrogen))
    assert (candidate in _peptide_pairs(structure)) is accept
    assert structure.topology.formal_charge(
        structure.constitution.atom_index(nitrogen)
    ) == (charge or None)
    explicit_pairs = {
        frozenset(structure.constitution.atom_ref_at(i) for i in b.endpoint_pair())
        for b in structure.topology.bonds
        if b.provenance is BondProvenance.SOURCE_EXPLICIT
    }
    assert explicit_pairs == {
        frozenset((nitrogen, AtomRef(nitrogen.residue_id, name))) for name in names
    }


def test_non_covalent_n_contact_does_not_block_inferred_peptide():
    source = _source_with_n_attachments(("CB",), kind=gemmi.ConnectionType.Hydrog)
    structure = _read(source, FileFormat.MMCIF)
    assert len(_peptide_pairs(structure)) == 2


@pytest.mark.parametrize("atom_name,order", (("N", "trip"), ("C", "doub")))
def test_source_backbone_bond_order_limits_remaining_peptide_valence(atom_name, order):
    source = _source(("1", "2", "3"), positions=(1, 2, 3))
    residue = source[0][0][1]
    oxygen = gemmi.Atom()
    oxygen.name = "O"
    oxygen.element = gemmi.Element("O")
    oxygen.pos = gemmi.Position(5.33, 1.2, 0)
    residue.add_atom(oxygen)
    _crosslink(source, gemmi.ConnectionType.Covale)
    connection = source.connections[0]
    connection.partner1.res_id = residue
    connection.partner1.atom_name = atom_name
    connection.partner2.res_id = residue
    connection.partner2.atom_name = "CA"
    document = source.make_mmcif_document()
    block = document.sole_block()
    category = block.get_mmcif_category("_struct_conn.")
    category["pdbx_value_order"] = [order]
    block.set_mmcif_category("_struct_conn.", category)

    structure = read_structure_string(document.as_string(), FileFormat.MMCIF)
    endpoint = AtomRef(ResidueId("A", 2), atom_name)
    partner = (
        AtomRef(ResidueId("A", 1), "C")
        if atom_name == "N"
        else AtomRef(ResidueId("A", 3), "N")
    )
    assert frozenset((endpoint, partner)) not in _peptide_pairs(structure)
    explicit = [
        b
        for b in structure.topology.bonds
        if b.provenance is BondProvenance.SOURCE_EXPLICIT
    ]
    assert len(explicit) == 1
    assert explicit[0].order == (3 if order == "trip" else 2)
