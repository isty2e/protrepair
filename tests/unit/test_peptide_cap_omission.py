"""Retention custody of peptide-cap junctions, including bare PDB CONECT."""

from dataclasses import replace
from pathlib import Path

import gemmi
import pytest

from protrepair import process_structure
from protrepair.chemistry.peptide_caps import observe_linked_peptide_cap
from protrepair.io import FileFormat, write_structure_string
from protrepair.io.gemmi_ingress import read_structure_string_with_policy
from protrepair.io.ingress_policy import (
    LigandHandling,
    NonPolymerRetentionPolicy,
    RetentionAction,
    RetentionRule,
    StructureNormalizationPolicy,
)
from protrepair.io.rdkit import to_rdkit_molecule
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.cap_attachment import PeptideCapAttachment
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondProvenance, StructureTopology
from protrepair.transformer.completion.hydrogen.linked_caps import (
    stage_complete_cap_topology,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    PeptideCappingRequest,
    PeptideCapStatus,
    PeptideEnd,
    PeptidePort,
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)

DATA = Path(__file__).parents[1] / "fixtures/pdb/linked-caps/openmm-populated.pdb"
KEEP = StructureNormalizationPolicy()
DROP = StructureNormalizationPolicy(ligand_handling=LigandHandling.DROP)
DROP_ACE = StructureNormalizationPolicy(
    non_polymer_retention=NonPolymerRetentionPolicy(
        other_default=RetentionAction.KEEP,
        water_default=RetentionAction.DROP,
        rules=(RetentionRule(RetentionAction.DROP, component_ids=("ACE",)),),
    )
)
N = AtomRef(ResidueId("A", 2), "N")
C = AtomRef(ResidueId("A", 3), "C")
ACE = ResidueId("A", 1)
NME = ResidueId("A", 4)


def omitted(s: ProteinStructure) -> set[tuple[AtomRef, ResidueId]]:
    return {(cap.port, cap.cap_id) for cap in s.provenance.omitted_peptide_caps}


@pytest.mark.parametrize(
    "policy,expected",
    [(KEEP, set()), (DROP, {(N, ACE), (C, NME)}), (DROP_ACE, {(N, ACE)})],
)
def test_bare_cap_omissions_survive_refiltering(
    policy: StructureNormalizationPolicy, expected: set[tuple[AtomRef, ResidueId]]
) -> None:
    text = DATA.read_text()
    kept = read_structure_string_with_policy(text, FileFormat.PDB, policy=KEEP)
    direct = read_structure_string_with_policy(text, FileFormat.PDB, policy=policy)
    projected = apply_structure_normalization_policy(kept, policy=policy)
    assert (
        projected.provenance.ingress.observation is kept.provenance.ingress.observation
    )
    assert omitted(direct) == omitted(projected) == expected
    for item in (direct, projected):
        again = apply_structure_normalization_policy(
            apply_structure_normalization_policy(item, policy=KEEP), policy=policy
        )
        assert omitted(again) == expected
        assert not any(
            site.residue_id in {r for _, r in expected}
            for site in again.constitution.ligands
        )


def test_a_missing_or_nonadjacent_bare_atom_does_not_identify_an_omitted_cap() -> None:
    bad = "\n".join(
        line
        for line in DATA.read_text().splitlines()
        if not line.startswith("HETATM    1 ")
    )
    s = read_structure_string_with_policy(bad, FileFormat.PDB, policy=DROP)
    assert omitted(s) == {(C, NME)}
    bad = DATA.read_text().replace("ACE A   1", "ACE A  99")
    s = read_structure_string_with_policy(bad, FileFormat.PDB, policy=DROP)
    assert omitted(s) == {(C, NME)}


def test_contact_without_declared_attachment_does_not_count_as_a_cap() -> None:
    text = "\n".join(
        line for line in DATA.read_text().splitlines() if not line.startswith("CONECT")
    )
    assert not omitted(
        read_structure_string_with_policy(text, FileFormat.PDB, policy=DROP)
    )


@pytest.mark.parametrize("isotope", ["D", "T"])
def test_cap_reconciliation_keeps_deposited_isotopes(isotope: str) -> None:
    text = "\n".join(
        line[:12] + f"{isotope + '1':>4}" + line[16:76] + f"{isotope:>2}" + line[78:]
        if line.startswith("HETATM    4")
        else line
        for line in DATA.read_text().splitlines()
    )
    original = read_structure_string_with_policy(text, FileFormat.PDB, policy=KEEP)
    reconciled = stage_complete_cap_topology(original, N, ACE)
    h = AtomRef(ACE, isotope + "1")
    index = reconciled.constitution.atom_index(h)
    assert reconciled.constitution.atom_site_at(index).element == isotope
    assert reconciled.geometry is original.geometry
    edge = reconciled.topology.bond_between(
        index, reconciled.constitution.atom_index(AtomRef(ACE, "CH3"))
    )
    assert edge is not None and edge.order == 1
    observed = original.topology.bond_between(*edge.endpoint_pair())
    assert observed is not None and observed.source_metadata == edge.source_metadata


def modelled_caps() -> ProteinStructure:
    text = DATA.read_text()
    source = (
        "SEQRES   1 A    4  SER GLY ALA THR\n"
        + "\n".join(line for line in text.splitlines() if line.startswith("ATOM"))
        + "\nEND\n"
    )
    original = read_structure_string_with_policy(source, FileFormat.PDB, policy=KEEP)
    capped = read_structure_string_with_policy(text, FileFormat.PDB, policy=KEEP)
    for port, cap in ((N, ACE), (C, NME)):
        capped = stage_complete_cap_topology(capped, port, cap)
    externals = {
        edge
        for _, edge in PeptideCapAttachment.bonded_ports(
            capped.constitution,
            capped.geometry,
            capped.topology,
            adjacent=capped.provenance.ingress.pdb_adjacent_cap_links,
        )
    }
    assert len(externals) == 2
    bonds = tuple(
        replace(b, provenance=BondProvenance.REPAIR_INFERRED, source_metadata=None)
        if b in externals
        else b
        for b in capped.topology.bonds
    )
    return ProteinStructure.from_payload(
        constitution=capped.constitution,
        geometry=capped.geometry,
        topology=StructureTopology(
            constitution=capped.constitution,
            atom_topologies=capped.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=original.provenance,
    )


@pytest.mark.parametrize("fmt", tuple(FileFormat))
@pytest.mark.parametrize("complete_h", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_workflow_reread_preserves_requested_retention(
    tmp_path: Path, fmt: FileFormat, complete_h: bool, keep: bool
) -> None:
    original = modelled_caps()
    path = tmp_path / ("capped.pdb" if fmt is FileFormat.PDB else "capped.cif")
    path.write_text(write_structure_string(original, fmt))
    ingress = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP if keep else LigandPolicy.DROP
    )
    goals = (
        (
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        )
        if complete_h
        else ()
    )
    for source in (original, path):
        result = process_structure(source, ingress=ingress, requested_goals=goals)
        ids = {r.residue_id for r in result.structure.constitution.ligands}
        assert ids == ({ACE, NME} if keep else set())
        assert omitted(result.structure) == (set() if keep else {(N, ACE), (C, NME)})


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_exported_cap_graph_survives_two_round_trips(fmt: FileFormat) -> None:
    current = modelled_caps()
    original_ports = {N, C}
    for _ in range(2):
        data = write_structure_string(current, fmt)
        for policy, expected in (
            (DROP, {(N, ACE), (C, NME)}),
            (DROP_ACE, {(N, ACE)}),
            (KEEP, set()),
        ):
            reread = read_structure_string_with_policy(data, fmt, policy=policy)
            memory = apply_structure_normalization_policy(current, policy=policy)
            assert omitted(reread) == omitted(memory) == expected
            obs = reread.provenance.ingress.observation
            assert obs is not None
            assert {
                f.port
                for f in obs.deposited_peptide_boundaries
                if f.kind.value == "missing_continuation"
            } == original_ports
            assert [
                s.components
                for s in reread.provenance.ingress.deposited_peptide_sequences
            ] == [("SER", "GLY", "ALA", "THR")]
        restored = read_structure_string_with_policy(data, fmt, policy=KEEP)
        for port, cap in ((N, ACE), (C, NME)):
            before = restored
            restored = stage_complete_cap_topology(before, port, cap)
            assert (
                restored.geometry is before.geometry
                and restored.constitution is before.constitution
            )
            assert restored.provenance is before.provenance
            site = restored.constitution.residue_or_ligand(cap)
            assert site is not None
            link = observe_linked_peptide_cap(restored, site)
            assert (
                link is not None
                and link.partner == port
                and link.hydrogens_are_complete()
            )
            assert stage_complete_cap_topology(restored, port, cap) is restored
        current = restored


@pytest.mark.parametrize("contradicts", [False, True])
def test_mmcif_keeps_only_bound_sequence_addresses(contradicts: bool) -> None:
    doc = gemmi.cif.read_string(
        write_structure_string(modelled_caps(), FileFormat.MMCIF)
    )
    block = doc.sole_block()
    asym = next(iter(block.find_values("_atom_site.label_asym_id")))
    loop = block.init_loop(
        "_pdbx_poly_seq_scheme.",
        ["asym_id", "seq_id", "pdb_seq_num", "auth_seq_num", "pdb_ins_code", "mon_id"],
    )
    expected = [
        ("1", "81", "?", "."),
        ("2", "2", "2", "."),
        ("3", "3", "3", "."),
        ("4", "84", "184", "A"),
    ]
    for monomer, row in zip(("SER", "GLY", "ALA", "THR"), expected, strict=True):
        loop.add_row([asym, *row, "VAL" if contradicts and row[0] == "4" else monomer])
    current = read_structure_string_with_policy(
        doc.as_string(), FileFormat.MMCIF, policy=KEEP
    )
    written = gemmi.cif.read_string(
        write_structure_string(current, FileFormat.MMCIF)
    ).sole_block()
    stored = [
        tuple(row)
        for row in written.find(
            "_pdbx_poly_seq_scheme.",
            ["seq_id", "pdb_seq_num", "auth_seq_num", "pdb_ins_code"],
        )
    ]
    assert stored == ([] if contradicts else expected)


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_a_cap_does_not_replace_another_chains_deposited_sequence(
    fmt: FileFormat,
) -> None:
    raw = write_structure_string(modelled_caps(), FileFormat.PDB)
    other = [
        line[:6] + f"{int(line[6:11]) + 300:5}" + line[11:21] + "B" + line[22:]
        for line in raw.splitlines()
        if line.startswith("ATOM  ")
    ]
    rows = [line for line in raw.splitlines() if not line.startswith(("END", "CONECT"))]
    text = (
        "\n".join(["SEQRES   1 B    3  GLY ALA PRO", *rows, *other, "TER", "END"])
        + "\n"
    )
    source = read_structure_string_with_policy(text, FileFormat.PDB, policy=KEEP)
    output = read_structure_string_with_policy(
        write_structure_string(source, fmt), fmt, policy=KEEP
    )
    assert {
        p.chain_id: p.components
        for p in output.provenance.ingress.deposited_peptide_sequences
    } == {"A": ("SER", "GLY", "ALA", "THR"), "B": ("GLY", "ALA", "PRO")}


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_prepared_export_retains_the_native_graph(fmt: FileFormat) -> None:
    keep = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    goals = (
        requested_process_goal(
            scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
        ),
    )
    original = process_structure(
        modelled_caps(), ingress=keep, requested_goals=goals
    ).structure
    restored = read_structure_string_with_policy(
        write_structure_string(original, fmt), fmt, policy=KEEP
    )
    for port, cap in ((N, ACE), (C, NME)):
        restored = stage_complete_cap_topology(restored, port, cap)
    # PDB CONECT does not encode the order on the unrelated polymer C-H bonds.
    restored = process_structure(
        restored, ingress=keep, requested_goals=goals
    ).structure

    def graph(s: ProteinStructure):
        mol = to_rdkit_molecule(s)
        ref = s.constitution.atom_ref_at
        atoms = {
            ref(AtomIndex(a.GetIdx())): (
                a.GetAtomicNum(),
                a.GetIsotope(),
                a.GetFormalCharge(),
                a.GetNumRadicalElectrons(),
            )
            for a in mol.GetAtoms()
        }
        bonds = {
            frozenset(
                (ref(AtomIndex(b.GetBeginAtomIdx())), ref(AtomIndex(b.GetEndAtomIdx())))
            ): b.GetBondTypeAsDouble()
            for b in mol.GetBonds()
        }
        return atoms, bonds

    assert graph(restored) == graph(original)


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_cap_in_another_crystal_image_is_not_identified_as_local(
    fmt: FileFormat,
) -> None:
    text = write_structure_string(modelled_caps(), fmt)
    if fmt is FileFormat.PDB:
        text = (
            "\n".join(
                line[:66] + "  2555" + line[72:]
                if line.startswith("LINK") and "ACE" in line
                else line
                for line in text.splitlines()
            )
            + "\n"
        )
    else:
        doc = gemmi.cif.read_string(text)
        for row in doc.sole_block().find(
            "_struct_conn.",
            ["ptnr1_label_comp_id", "ptnr2_label_comp_id", "ptnr2_symmetry"],
        ):
            if "ACE" in (row[0], row[1]):
                row[2] = "2_555"
        text = doc.as_string()
    for policy in (KEEP, DROP):
        s = read_structure_string_with_policy(text, fmt, policy=policy)
        assert s.provenance.ingress.uncertain_peptide_link_ports is not None
        assert N in s.provenance.ingress.uncertain_peptide_link_ports
        if policy == KEEP:
            with pytest.raises(ValueError, match="no verified local"):
                stage_complete_cap_topology(s, N, ACE)
        assert omitted(apply_structure_normalization_policy(s, policy=DROP)) == {
            (C, NME)
        }


def test_a_missing_cap_hydrogen_cannot_be_supplied_by_bond_order_reconciliation() -> (
    None
):
    data = "\n".join(
        line
        for line in DATA.read_text().splitlines()
        if not line.startswith("HETATM    5")
    )
    structure = read_structure_string_with_policy(data, FileFormat.PDB, policy=KEEP)
    with pytest.raises(ValueError):
        stage_complete_cap_topology(structure, N, ACE)
    assert stage_complete_cap_topology(structure, C, NME).geometry is structure.geometry


def test_filtering_a_cap_on_another_chain() -> None:
    text = write_structure_string(modelled_caps(), FileFormat.PDB).replace(
        "ACE A   1", "ACE B   1"
    )
    policy = StructureNormalizationPolicy(selected_chain_ids=("A",))
    kept = read_structure_string_with_policy(text, FileFormat.PDB, policy=KEEP)
    direct = read_structure_string_with_policy(text, FileFormat.PDB, policy=policy)
    projected = apply_structure_normalization_policy(kept, policy=policy)
    assert omitted(direct) == omitted(projected) == {(N, ResidueId("B", 1))}

    rows = [
        row
        for row in text.splitlines()
        if not row.startswith(("SEQRES", "CONECT", "END")) and "NME" not in row
    ]
    nitrogen = next(
        int(row[6:11])
        for row in rows
        if row.startswith("ATOM  ") and row[12:16].strip() == "N"
    )
    malformed = "\n".join([*rows, f"CONECT{nitrogen:5} 9999", "END"]) + "\n"
    full = read_structure_string_with_policy(malformed, FileFormat.PDB, policy=KEEP)
    raw = read_structure_string_with_policy(malformed, FileFormat.PDB, policy=policy)
    assert (
        omitted(raw)
        == omitted(apply_structure_normalization_policy(full, policy=policy))
        == set()
    )


def test_a_typed_cap_link_does_not_override_an_unresolved_conect_row() -> None:
    lines = write_structure_string(modelled_caps(), FileFormat.PDB).splitlines()
    serial = next(
        int(row[6:11])
        for row in lines
        if row.startswith("ATOM  ")
        and row[12:16].strip() == "N"
        and int(row[22:26]) == 2
    )
    # Only typed external LINKs and one malformed CONECT remain.
    text = (
        "\n".join(row for row in lines if not row.startswith(("CONECT", "END")))
        + f"\nCONECT{serial:5} 9999\nEND\n"
    )
    kept = read_structure_string_with_policy(text, FileFormat.PDB, policy=KEEP)
    obs = kept.provenance.ingress.observation
    assert obs is not None and N in obs.unresolved_conect_ports
    result = process_structure(
        kept,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        transform_requests=WorkflowTransformRequests(
            peptide_capping=PeptideCappingRequest.only(
                (PeptidePort(N.residue_id, PeptideEnd.N),)
            )
        ),
    )
    assert result.peptide_capping_report is not None
    (outcome,) = result.peptide_capping_report.outcomes
    assert outcome.status is PeptideCapStatus.ATTACHED_UNVERIFIED
    assert outcome.reason == "unresolved_conect" and outcome.cap_id == ACE
    assert outcome.attachment == (N, AtomRef(ACE, "C")) and not result.repairs
    direct = read_structure_string_with_policy(text, FileFormat.PDB, policy=DROP)
    projected = apply_structure_normalization_policy(kept, policy=DROP)
    assert omitted(direct) == omitted(projected) == {(C, NME)}


def test_polymer_role_acetyl_group_occupies_the_port_without_claiming_a_nonpolymer_cap():
    rows = write_structure_string(modelled_caps(), FileFormat.PDB).splitlines()
    text = "\n".join(
        "ATOM  " + row[6:] if row.startswith("HETATM") and row[17:20] == "ACE" else row
        for row in rows
    )
    parent = read_structure_string_with_policy(text, FileFormat.PDB, policy=KEEP)
    result = process_structure(
        parent,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        transform_requests=WorkflowTransformRequests(
            peptide_capping=PeptideCappingRequest.only(
                (PeptidePort(N.residue_id, PeptideEnd.N),)
            )
        ),
    )
    assert result.peptide_capping_report is not None
    (outcome,) = result.peptide_capping_report.outcomes
    assert outcome.status is PeptideCapStatus.OCCUPIED and outcome.cap_id is None
    assert outcome.attachment == (N, AtomRef(ACE, "C")) and not result.repairs


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_export_after_an_independent_occupancy_omission(fmt: FileFormat) -> None:
    fixture = DATA.parents[1] / "span-reconstruction/1ubq-short-gaps.pdb"
    coords = [
        line
        for line in fixture.read_text().splitlines()
        if line.startswith("ATOM  ") and 3 <= int(line[22:26]) <= 5
    ]
    coords = [
        line[:54] + "  0.00" + line[60:] if int(line[22:26]) == 4 else line
        for line in coords
    ]
    ace = [
        line[:6] + f"{8000 + int(line[6:11]):5}" + line[11:]
        for line in DATA.read_text().splitlines()
        if line.startswith("HETATM") and "ACE" in line
    ]
    typed = next(
        line
        for line in write_structure_string(modelled_caps(), FileFormat.PDB).splitlines()
        if line.startswith("LINK") and "ACE" in line
    )
    text = (
        "\n".join(
            [
                "SEQRES   1 A    3  ILE PHE VAL",
                typed.replace("GLY A   2", "ILE A   3"),
                *coords,
                "TER",
                *ace,
                "END",
            ]
        )
        + "\n"
    )
    if fmt is FileFormat.MMCIF:
        raw = gemmi.read_pdb_string(text)
        raw.setup_entities()
        raw.assign_label_seq_id()
        text = raw.make_mmcif_document().as_string()
    kept = read_structure_string_with_policy(
        text,
        fmt,
        policy=StructureNormalizationPolicy(omit_zero_occupancy_polymer_residues=True),
    )
    assert kept.constitution.chain("A").residue_ids() == (
        ResidueId("A", 3),
        ResidueId("A", 5),
    )
    assert kept.provenance.ingress.deposited_peptide_sequences[0].components == (
        "ILE",
        "PHE",
        "VAL",
    )
    outcome = read_structure_string_with_policy(
        write_structure_string(kept, fmt), fmt, policy=KEEP
    )
    observation = outcome.provenance.ingress.observation
    assert observation is not None
    assert not any(
        f.kind.value == "missing_continuation"
        for f in observation.deposited_peptide_boundaries
    )
    assert outcome.constitution.residue_or_ligand(ACE) is not None
