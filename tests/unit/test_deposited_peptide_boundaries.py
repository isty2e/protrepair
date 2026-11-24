"""Source sequence evidence is independent of retention and repair chemistry."""

from itertools import combinations, product
from pathlib import Path

import pytest

from protrepair.io import FileFormat, read_structure, read_structure_string
from protrepair.io.ingress_policy import LigandHandling, StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.relation.sequence_embedding import SequenceEmbedding
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.bound_peptide_ports import BoundPeptidePorts
from protrepair.structure.sequence_boundary import PeptideSequenceBoundary

END = PeptideSequenceBoundary.CONSTRUCT_END
MISSING = PeptideSequenceBoundary.MISSING_CONTINUATION
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _facts(structure: ProteinStructure) -> dict[str, PeptideSequenceBoundary]:
    observation = structure.provenance.ingress.observation
    assert observation is not None
    return {
        f.port.display_token(): f.kind for f in observation.deposited_peptide_boundaries
    }


def _pdb_atom(
    serial: int,
    name: str,
    number: int,
    *,
    het: bool = False,
    ins: str = " ",
    atom: str = "CA",
) -> str:
    return (
        f"{'HETATM' if het else 'ATOM  '}{serial:5d} {atom:^4} {name:>3} A{number:4d}{ins}"
        f"      1.000   2.000   3.000  1.00 20.00          {atom[0]:>2}"
    )


def _seqres(*tokens: str) -> str:
    return f"SEQRES   1 A {len(tokens):4d}  " + " ".join(tokens)


def _pdb(*rows: str, drop: bool = False) -> ProteinStructure:
    return read_structure_string(
        "\n".join((*rows, "END", "")),
        FileFormat.PDB,
        policy=StructureNormalizationPolicy(
            ligand_handling=LigandHandling.DROP if drop else LigandHandling.KEEP
        ),
    )


def test_embedding_bounds_and_compulsory_gaps_match_all_small_embeddings():
    for size in range(1, 7):
        for declared in product(("ALA", "GLY"), repeat=size):
            embeddings: dict[tuple[str, ...], list[tuple[int, ...]]] = {}
            for n in range(1, size + 1):
                for indices in combinations(range(size), n):
                    tokens = tuple(declared[i] for i in indices)
                    embeddings.setdefault(tokens, []).append(
                        tuple(i + 1 for i in indices)
                    )
            for observed, candidates in embeddings.items():
                proof = SequenceEmbedding.match(observed, declared)
                assert proof is not None
                assert proof.first == tuple(map(min, zip(*candidates)))
                assert proof.last == tuple(map(max, zip(*candidates)))
                assert proof.gaps_after == tuple(
                    i
                    for i in range(len(observed) - 1)
                    if all(c[i + 1] > c[i] + 1 for c in candidates)
                )
            assert SequenceEmbedding.match(("SER",), declared) is None


def test_pdb_gaps_use_seqres_not_author_arithmetic_or_a_preferred_alignment():
    structure = _pdb(
        _seqres("SER", "ALA", "VAL", "GLY", "THR"),
        _pdb_atom(1, "ALA", -3, ins="B"),
        _pdb_atom(2, "GLY", 907),
    )
    assert _facts(structure) == {
        "A:-3B.N": MISSING,
        "A:-3B.C": MISSING,
        "A:907.N": MISSING,
        "A:907.C": MISSING,
    }
    assert not _facts(
        _pdb(
            _seqres("GLY", "GLY", "GLY"),
            _pdb_atom(1, "GLY", 1),
            _pdb_atom(2, "GLY", 300),
        )
    )


@pytest.mark.parametrize("drop", [False, True])
def test_unknown_het_monomer_counts_before_keep_or_drop(drop):
    residues = (
        _pdb_atom(1, "ALA", 10),
        _pdb_atom(2, "XYZ", 11, het=True),
        _pdb_atom(3, "GLY", 12),
    )
    header = _seqres("ALA", "XYZ", "GLY")
    assert _facts(_pdb(header, *residues, drop=drop)) == {"A:10.N": END, "A:12.C": END}
    assert not _facts(_pdb(header, residues[0], "TER", *residues[1:], drop=drop))
    assert _facts(
        _pdb(_seqres("ALA", "XYZ"), *residues[:1], "TER", residues[1], drop=drop)
    ) == {"A:10.N": END}
    assert _facts(
        _pdb(header, *residues, "TER", _pdb_atom(4, "ZN", 900, het=True), drop=drop)
    ) == {"A:10.N": END, "A:12.C": END}


@pytest.mark.parametrize(
    "header",
    [
        "",
        "SEQRES   1 A    4  ALA GLY SER",
        "SEQRES   2 A    3  ALA GLY SER",
        _seqres("DA", "DC"),
    ],
)
def test_incomplete_or_non_peptide_seqres_does_not_certify_ends(header):
    assert not _facts(_pdb(header, _pdb_atom(1, "ALA", 1), _pdb_atom(2, "GLY", 2)))


@pytest.mark.parametrize("component", ["HOH", "NAG", "DC", "SO4", "NME"])
def test_a_known_nonpeptide_cannot_supply_the_missing_biological_monomer(component):
    assert not _facts(_pdb(_seqres("ALA", component), _pdb_atom(1, "ALA", 10)))
    assert not _facts(
        read_structure_string(
            _CIF.replace("1 2 SER", f"1 2 {component}"), FileFormat.MMCIF
        )
    )


@pytest.mark.parametrize(
    ("name", "atoms"), [("PTR", ("CA",)), ("XYZ", ("N", "CA", "C"))]
)
def test_possible_modified_monomer_after_ter_withholds_the_chain(name, atoms):
    assert not _facts(
        _pdb(
            _seqres("ALA", "TYR"),
            _pdb_atom(1, "ALA", 1),
            "TER",
            *(
                _pdb_atom(i, name, 2, het=True, atom=atom)
                for i, atom in enumerate(atoms, 2)
            ),
            drop=True,
        )
    )


@pytest.mark.parametrize("drop", [True, False])
def test_a_post_ter_monomer_only_blocks_gaps_it_could_occupy(drop):
    rows = (
        _pdb_atom(1, "ALA", 10),
        _pdb_atom(2, "GLY", 12),
        "TER",
        _pdb_atom(3, "GLU", 900, het=True),
    )
    assert _facts(_pdb(_seqres("GLU", "ALA", "SER", "GLY"), *rows, drop=drop)) == {
        "A:10.C": MISSING,
        "A:12.N": MISSING,
        "A:12.C": END,
    }
    assert _facts(_pdb(_seqres("ALA", "GLU", "GLY"), *rows, drop=drop)) == {
        "A:10.N": END,
        "A:12.C": END,
    }


def test_full_pdb_with_trailing_solvent_preserves_the_deposited_ends():
    source = read_structure(FIXTURES / "pdb" / "1aho.pdb")
    assert _facts(source) == {"A:1.N": END, "A:64.C": END}


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize(
    ("conect", "serial", "veto"),
    [
        ("CONECT    1   99", 3, True),
        ("CONECT    1", 3, True),
        ("CONECT    1    1", 3, True),
        ("CONECT         1", 3, True),
        ("CONECT    1    3", 1, True),
        ("CONECT    3   99", 3, False),
    ],
)
def test_unbound_conect_marks_only_the_implicated_port_without_a_cap(
    conect, serial, veto, drop
):
    structure = _pdb(
        _seqres("ALA", "SER", "GLY"),
        _pdb_atom(1, "ALA", 10, atom="C"),
        _pdb_atom(2, "GLY", 12, atom="N"),
        _pdb_atom(serial, "GLY", 12),
        conect,
        drop=drop,
    )
    obs = structure.provenance.ingress.observation
    assert obs is not None
    assert _facts(structure)["A:10.C"] is MISSING
    assert obs.unresolved_conect_ports == (
        {AtomRef(ResidueId("A", 10), "C")} if veto else set()
    )


def test_conect_within_a_later_model_does_not_veto_the_first_model_port():
    rows = (_pdb_atom(1, "ALA", 10, atom="C"), _pdb_atom(2, "GLY", 12, atom="N"))
    structure = _pdb(
        _seqres("ALA", "SER", "GLY"),
        "MODEL        1",
        *rows,
        "ENDMDL",
        "MODEL        2",
        *rows,
        "CONECT    1   99",
        "ENDMDL",
    )
    obs = structure.provenance.ingress.observation
    assert obs is not None
    assert _facts(structure)["A:10.C"] is MISSING
    assert not obs.unresolved_conect_ports


def test_decoded_link_to_filtered_partner_is_not_lost_or_mistaken_for_a_bad_serial():
    structure = _pdb(
        _seqres("ALA", "SER", "GLY"),
        _pdb_atom(1, "ALA", 10, atom="C"),
        _pdb_atom(2, "GLY", 12, atom="N"),
        "TER",
        _pdb_atom(3, "LIG", 900, het=True),
        "CONECT    1    3",
        drop=True,
    )
    obs = structure.provenance.ingress.observation
    assert obs is not None
    assert _facts(structure)["A:10.C"] is MISSING
    assert AtomRef(ResidueId("A", 10), "C") in obs.unresolved_polymer_endpoints
    assert not obs.unresolved_conect_ports


_CIF = """data_cut
loop_
_entity.id
_entity.type
1 polymer
loop_
_struct_asym.id
_struct_asym.entity_id
X 1
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 SER
1 3 GLY
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA X 1 1 A 10 1 2 3 1 20 1
ATOM 2 C CA . GLY X 1 3 A 11 3 4 5 1 20 1
"""


def test_mmcif_only_considers_the_original_model_and_its_entity_positions():
    source = read_structure_string(
        _CIF + "ATOM 3 C CA . SER X 1 2 A 50 2 3 4 1 20 2\n", FileFormat.MMCIF
    )
    assert _facts(source) == {
        "A:10.N": END,
        "A:10.C": MISSING,
        "A:11.N": MISSING,
        "A:11.C": END,
    }
    assert source.constitution.chain("A").residue_ids() == (
        ResidueId("A", 10),
        ResidueId("A", 11),
    )


@pytest.mark.parametrize("contradicts", [False, True])
def test_mmcif_cap_addresses_avoid_both_number_columns_even_without_coordinates(
    contradicts,
):
    scheme = """loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
X 1 10 10 .
X 2 '15' ? A
X 3 11 16 .
"""
    missing = """loop_
_pdbx_unobs_or_zero_occ_residues.auth_asym_id
_pdbx_unobs_or_zero_occ_residues.auth_seq_id
A 17
B 18
"""
    source = read_structure_string(
        _CIF + scheme + ("X 3 11 17 .\n" if contradicts else "") + missing,
        FileFormat.MMCIF,
    )
    bound = BoundPeptidePorts.bind(source)
    assert len(bound.components) == 2
    assert sorted(r.seq_num for r in bound.cap_ids.values()) == (
        [] if contradicts else [18, 19]
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("1 2 SER", "1 9 SER"),
        ("1 2 SER", "1 2 SER\n1 2 THR"),
        ("GLY X 1 3", "GLY X 1 ?"),
        ("GLY X 1 3", "GLY X 1 2"),
        ("GLY X 1 3 A 11", "GLY X 1 3 B 11"),
        ("GLY X 1 3", "GLY X 2 3"),
        ("X 1\nloop_", "X 2\nloop_"),
        ("1 'polypeptide(L)'", "1 polyribonucleotide"),
    ],
)
def test_inconsistent_entity_or_position_is_not_a_declaration(old, new):
    assert not _facts(read_structure_string(_CIF.replace(old, new), FileFormat.MMCIF))


def test_a_labelled_hetero_residue_is_not_a_missing_monomer():
    cif = (
        _CIF.replace("A 11 ", "A 41 ")
        .replace("1 2 SER", "1 2 XYZ")
        .replace("ATOM 2", "HETATM 3 C CA . XYZ X 1 2 A 40 6 7 8 1 20 1\nATOM 2")
    )
    for handling in (LigandHandling.KEEP, LigandHandling.DROP):
        source = read_structure_string(
            cif,
            FileFormat.MMCIF,
            policy=StructureNormalizationPolicy(ligand_handling=handling),
        )
        assert _facts(source) == {"A:10.N": END, "A:41.C": END}


@pytest.mark.parametrize("position", ["?", "1", "2"])
def test_two_polymer_instances_sharing_one_author_chain_do_not_prove_a_port(position):
    cif = _CIF.replace("X 1\nloop_", "X 1\nZ 1\nloop_")
    cif += f"ATOM 3 C CA . ALA Z 1 {position} A 50 8 2 1 1 20 1\n"
    assert not _facts(read_structure_string(cif, FileFormat.MMCIF))


def test_reordered_het_encoded_polymer_does_not_certify_ports_in_the_opposite_order():
    cif = (
        _CIF.replace("1 1 ALA", "1 1 MSE")
        .replace("ATOM 1 C CA . ALA", "HETATM 1 C CA . MSE")
        .replace("A 10", "A 80")
    )
    source = read_structure_string(cif, FileFormat.MMCIF)
    assert source.constitution.chain("A").residue_ids() == (
        ResidueId("A", 11),
        ResidueId("A", 80),
    )
    assert not _facts(source)
    pdb = _pdb(
        _seqres("ALA", "GLY", "SER", "THR"),
        _pdb_atom(1, "ALA", 80, het=True),
        _pdb_atom(2, "GLY", 81),
        _pdb_atom(3, "THR", 11),
    )
    assert pdb.constitution.chain("A").residue_ids() == tuple(
        ResidueId("A", n) for n in (11, 80, 81)
    )
    assert not _facts(pdb)


def test_first_model_can_be_number_two():
    cif = (
        _CIF.replace("20 1\n", "20 2\n") + "ATOM 3 C CA . SER X 1 2 A 50 2 3 4 1 20 1\n"
    )
    assert _facts(read_structure_string(cif, FileFormat.MMCIF))["A:10.C"] is MISSING


def test_unassigned_model_on_a_competing_polymer_row_withholds_the_claim():
    cif = _CIF + "ATOM 3 C CA . SER X 1 2 A 50 2 3 4 1 20 ?\n"
    assert not _facts(read_structure_string(cif, FileFormat.MMCIF))


def test_canonical_chain_selection_keeps_original_facts_but_lost_anchors_do_not_rebind():
    cif = _CIF.replace("X 1\nloop_", "X 1\nY 1\nloop_")
    cif += "ATOM 3 C CA . ALA Y 1 1 B 50 2 3 4 1 20 1\n"
    original = read_structure_string(cif, FileFormat.MMCIF)
    selected = apply_structure_normalization_policy(
        original, policy=StructureNormalizationPolicy(selected_chain_ids=("B",))
    )
    obs = original.provenance.ingress.observation
    assert obs is not None and selected.provenance.ingress.observation is obs
    rebound = {
        fact.port.display_token()
        for fact in obs.deposited_peptide_boundaries
        if fact.identifies_polymer_in(selected.constitution)
    }
    assert rebound == {"B:50.N", "B:50.C"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("5owx", {"1:133.C", "1:155.N"}),
        ("5n9y", set()),
    ],
)
def test_archived_polypeptide_instances_have_the_declared_flanks(name, expected):
    facts = _facts(read_structure(FIXTURES / "corpus" / f"{name}.cif"))
    assert {port for port, kind in facts.items() if kind is MISSING} == expected
    if name == "5owx":
        assert {
            port
            for port, kind in facts.items()
            if kind is END and port.startswith("1:")
        } == {
            "1:27.N",
            "1:208.C",
        }
    else:
        assert {port for port in facts if port.startswith("A:")} == {"A:1.N", "A:327.C"}
