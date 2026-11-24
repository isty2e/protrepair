"""Omission, donor reconstruction and export of consecutive zero-occupancy residues."""

from dataclasses import replace
from itertools import pairwise, product
from pathlib import Path

import gemmi
import pytest

from protrepair import process_structure
from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.errors import StructureNormalizationError, StructureSerializationError
from protrepair.io import (
    FileFormat,
    read_structure_string,
    write_structure,
    write_structure_string,
)
from protrepair.io.ingress_policy import OccupancyPolicy, StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.cap_attachment import PeptideCapAttachment
from protrepair.structure.sequence_boundary import PeptideSequenceBoundary
from protrepair.structure.topology import BondProvenance
from protrepair.transformer.completion.span_reconstruction import (
    ReconstructedSpanCandidate,
    SpanChemistryConflictError,
    SpanReconstructionFailure,
    reconstruct_donor_span,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    LigandPolicy,
    PeptideCappingRequest,
    PeptideCapStatus,
    PeptideEnd,
    PeptidePort,
    StructureIngressOptions,
    WorkflowTransformRequests,
)

DATA = Path(__file__).parents[1] / "fixtures/pdb/span-reconstruction/1ubq.pdb"
SEQUENCE = "SEQRES   1 A   12  ILE PHE VAL LYS THR LEU THR GLY LYS THR ILE ALA"
OMIT_CONSECUTIVE = StructureNormalizationPolicy(
    omit_consecutive_zero_occupancy_polymer_residues=True
)
ORDINARY = StructureNormalizationPolicy()


def rid(number: int) -> ResidueId:
    return ResidueId("A", number)


def rows(
    zero: tuple[int, ...] = (4, 5, 6), positions: range = range(3, 9)
) -> list[str]:
    return [
        line[:54] + "  0.00" + line[60:] if int(line[22:26]) in zero else line
        for line in DATA.read_text().splitlines()
        if line.startswith("ATOM  ") and int(line[22:26]) in positions
    ]


def read(
    lines: list[str],
    fmt: FileFormat = FileFormat.PDB,
    policy: StructureNormalizationPolicy = OMIT_CONSECUTIVE,
) -> ProteinStructure:
    text = "\n".join((*lines, "END", ""))
    if fmt is FileFormat.MMCIF:
        raw = gemmi.read_pdb_string(text)
        raw.setup_entities()
        raw.assign_label_seq_id()
        text = raw.make_mmcif_document().as_string()
    return read_structure_string(text, fmt, policy=policy)


def cap(source, ports):
    return process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        transform_requests=WorkflowTransformRequests(
            peptide_capping=PeptideCappingRequest.only(ports)
        ),
    )


def assert_no_artificial_gap(source):
    c = source.constitution
    ports = (PeptidePort(rid(3), PeptideEnd.C), PeptidePort(rid(7), PeptideEnd.N))
    refs = (AtomRef(rid(3), "C"), AtomRef(rid(7), "N"))
    assert all(c.residue_or_ligand(rid(n)) is None for n in (4, 5, 6))
    assert source.topology.bond_between(*(c.atom_index(ref) for ref in refs)) is None
    obs = source.provenance.ingress.observation
    assert obs is not None
    assert not any(
        b.port in refs and b.kind is PeptideSequenceBoundary.MISSING_CONTINUATION
        for b in obs.deposited_peptide_boundaries
    )
    report = cap(source, ports).peptide_capping_report
    assert report is not None
    assert [o.reason for o in report.outcomes] == ["no_gap_evidence"] * 2


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_same_chain_generated_cap_blocks_export_only_until_restoration(fmt, tmp_path):
    source = read([SEQUENCE, *rows((4, 5, 6), range(3, 14))], FileFormat.MMCIF)
    assert_no_artificial_gap(source)
    assert_no_artificial_gap(
        read_structure_string(write_structure_string(source, fmt), fmt)
    )
    port = PeptidePort(rid(13), PeptideEnd.C)
    result = cap(source, (port,))
    assert result.peptide_capping_report is not None
    assert result.peptide_capping_report.outcomes[0].status is PeptideCapStatus.APPLIED
    path = tmp_path / "coordinates"
    path.write_bytes(b"previous structure")
    with pytest.raises(StructureSerializationError, match=r"chain A:.*A:4 through A:6"):
        write_structure(result.structure, path, file_format=fmt)
    assert path.read_bytes() == b"previous structure"

    candidate = fit(result.structure, read(rows(()), policy=ORDINARY))
    assert isinstance(candidate, ReconstructedSpanCandidate), candidate
    restored = candidate.materialize(
        result.structure, build_default_component_library()
    )
    reread = read_structure_string(write_structure_string(restored, fmt), fmt)
    assert reread.constitution.chain("A").residue(rid(5)).component_id == "VAL"
    assert any(r.component_id == "NME" for r in reread.constitution.ligands)


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_omitted_residues_do_not_block_a_generated_cap_on_another_chain(fmt):
    chain_b = [
        row[:6]
        + f"{int(row[6:11]) + 1000:5d}"
        + row[11:21]
        + "B"
        + row[22:30]
        + f"{float(row[30:38]) + 100:8.3f}"
        + row[38:]
        for row in rows((), range(3, 14))
    ]
    source = read(
        [
            SEQUENCE,
            SEQUENCE.replace(" A ", " B "),
            *rows((4, 5, 6), range(3, 14)),
            "TER",
            *chain_b,
        ],
        FileFormat.MMCIF,
    )
    port = PeptidePort(ResidueId("B", 13), PeptideEnd.C)
    result = cap(source, (port,))
    assert result.peptide_capping_report is not None
    assert result.peptide_capping_report.outcomes[0].status is PeptideCapStatus.APPLIED
    restored = read_structure_string(write_structure_string(result.structure, fmt), fmt)
    assert_no_artificial_gap(restored)
    assert any(
        r.component_id == "NME" and r.residue_id.chain_id == "B"
        for r in restored.constitution.ligands
    )


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_deposited_cap_on_chain_with_omitted_residues_uses_coordinate_only_fallback(
    fmt,
):
    complete = read([SEQUENCE, *rows((), range(3, 14))], FileFormat.MMCIF)
    capped = cap(complete, (PeptidePort(rid(13), PeptideEnd.C),)).structure
    doc = gemmi.cif.read_string(write_structure_string(capped, FileFormat.MMCIF))
    block = doc.sole_block()
    # Keep the deposited cap declarations, not writer-generated target bonds.
    links = block.find("_struct_conn.", ["ptnr1_auth_seq_id", "ptnr2_auth_seq_id"])
    for i in reversed(range(len(links))):
        if any(token in ("4", "5", "6") for token in links[i]):
            links.remove_row(i)
    for row in block.find("_atom_site.", ["auth_seq_id", "occupancy"]):
        if row[0] in ("4", "5", "6"):
            row[1] = "0.00"
    source = read_structure_string(
        doc.as_string(), FileFormat.MMCIF, policy=OMIT_CONSECUTIVE
    )
    assert source.provenance.ingress.consecutive_zero_occupancy_residue_omissions
    ingress = source.provenance.ingress
    attachments = PeptideCapAttachment.bonded_ports(
        source.constitution,
        source.geometry,
        source.topology,
        adjacent=ingress.pdb_adjacent_cap_links,
        uncertain=ingress.uncertain_peptide_link_ports or frozenset(),
        observation=ingress.observation,
    )
    assert any(
        a.port == AtomRef(rid(13), "C")
        and bond.provenance is BondProvenance.SOURCE_EXPLICIT
        for a, bond in attachments
    )
    reread = read_structure_string(write_structure_string(source, fmt), fmt)
    assert_no_artificial_gap(reread)
    assert any(r.component_id == "NME" for r in reread.constitution.ligands)


@pytest.mark.parametrize("fmt", tuple(FileFormat))
@pytest.mark.parametrize("single, multi, side", product((False, True), repeat=3))
def test_each_switch_has_its_own_receipt(fmt, single, multi, side):
    policy = StructureNormalizationPolicy(
        omit_zero_occupancy_polymer_residues=single,
        omit_consecutive_zero_occupancy_polymer_residues=multi,
        omit_zero_occupancy_sidechain_heavy_atoms=side,
    )
    selected = read(rows((4, 5, 6, 7, 9, 11, 12), range(3, 14)), fmt, policy)
    ingress = selected.provenance.ingress
    assert (ingress.zero_occupancy_omissions is not None) == side
    assert (ingress.consecutive_zero_occupancy_residue_omissions is not None) == multi
    assert (ingress.zero_occupancy_residue_omissions is not None) == single
    assert [o.residue_id for o in ingress.zero_occupancy_residue_omissions or ()] == (
        [rid(9)] if single else []
    )
    expected = [(4, 5, 6, 7), (11, 12)] if multi else []
    assert [
        r.residue_ids
        for r in ingress.consecutive_zero_occupancy_residue_omissions or ()
    ] == [tuple(map(rid, seq)) for seq in expected]
    withheld = {r for gap in ingress.recorded_residue_gaps() for r in gap.residue_ids}
    assert selected.constitution.chain("A").residue_ids() == tuple(
        rid(i) for i in range(3, 14) if rid(i) not in withheld
    )
    for omission in ingress.consecutive_zero_occupancy_residue_omissions or ():
        assert all(
            atom.geometry.occupancy == 0 for group in omission.members for atom in group
        )
    if not multi:
        with pytest.raises(StructureNormalizationError, match="re-reading"):
            apply_structure_normalization_policy(selected, policy=OMIT_CONSECUTIVE)


@pytest.mark.parametrize(
    "witness",
    [
        "partial",
        "positive_altloc",
        "outer_partner",
        "uncertain_port",
        "uncertain_sidechain",
        "oxt",
        "ter",
        "terminal",
    ],
)
def test_insufficient_evidence_keeps_every_member(witness):
    lines = rows()
    atom = next(
        row for row in lines if int(row[22:26]) == 5 and row[12:16].strip() == "CB"
    )
    if witness == "partial":
        lines[lines.index(atom)] = atom[:54] + "  0.10" + atom[60:]
    elif witness == "positive_altloc":
        i = lines.index(atom)
        lines[i] = atom[:16] + "A" + atom[17:]
        lines.insert(
            i + 1,
            atom[:6] + " 9998" + atom[11:16] + "B" + atom[17:54] + "  0.02" + atom[60:],
        )
    elif witness in ("outer_partner", "uncertain_port", "uncertain_sidechain"):
        anchor = (
            atom
            if witness == "uncertain_sidechain"
            else next(
                r for r in lines if int(r[22:26]) == 3 and r[12:16].strip() == "C"
            )
        )
        other = next(
            r for r in lines if int(r[22:26]) == 8 and r[12:16].strip() == "CB"
        )
        lines.append(
            f"CONECT{anchor[6:11]}{other[6:11] if witness == 'outer_partner' else ' 9999'}"
        )
    elif witness == "oxt":
        lines.insert(
            lines.index(atom),
            atom[:6] + " 9998" + atom[11:12] + " OXT" + atom[16:76] + " O",
        )
    elif witness == "ter":
        lines.insert(next(i for i, r in enumerate(lines) if int(r[22:26]) == 5), "TER")
    else:
        lines = rows((3, 4, 5, 6))
    source = read(
        lines, policy=replace(OMIT_CONSECUTIVE, occupancy_policy=OccupancyPolicy.LOWEST)
    )
    assert source.provenance.ingress.consecutive_zero_occupancy_residue_omissions == ()
    assert source.constitution.chain("A").residue(rid(5)).component_id == "VAL"
    if witness == "positive_altloc":
        selected = source.geometry.atom_geometry(
            source.constitution.atom_index(AtomRef(rid(5), "CB"))
        )
        assert (selected.altloc, selected.occupancy) == ("A", 0.0)


def test_shared_positive_flank_and_archive_identity():
    source = read(rows((4, 5, 7, 8), range(3, 10)))
    history = source.provenance.ingress
    assert history.consecutive_zero_occupancy_residue_omissions is not None
    first, second = history.consecutive_zero_occupancy_residue_omissions
    assert first.following_residue_id == second.preceding_residue_id == rid(6)
    with pytest.raises(ValueError, match="distinct"):
        replace(history, consecutive_zero_occupancy_residue_omissions=(first, first))
    with pytest.raises(ValueError, match="flank components"):
        replace(
            history,
            consecutive_zero_occupancy_residue_omissions=(
                replace(first, preceding_component_id="ASP"),
                second,
            ),
        )
    with pytest.raises(ValueError, match="consecutive"):
        replace(first, members=first.members[::-1])


@pytest.mark.parametrize("flank,port", [(3, "C"), (7, "N")])
def test_omission_preserves_but_donor_cannot_replace_charged_attachment(flank, port):
    lines = [
        r[:78] + "1+" if int(r[22:26]) == flank and r[12:16].strip() == port else r
        for r in rows()
    ]
    source = read(lines)
    assert source.provenance.ingress.consecutive_zero_occupancy_residue_omissions
    proposal = fit(source, read(rows(()), policy=ORDINARY))
    assert isinstance(proposal, SpanReconstructionFailure)
    assert "charge" in proposal.message


@pytest.mark.parametrize("kind", ["neutral_charge", "local_conect", "local_link"])
def test_decoded_evidence_on_a_member_refuses_the_whole_read(kind):
    lines = rows()
    one, two = (
        next(
            row for row in lines if int(row[22:26]) == 5 and row[12:16].strip() == name
        )
        for name in ("CB", "CG1")
    )
    if kind == "neutral_charge":
        lines[lines.index(one)] = one[:78] + "0+"
    elif kind == "local_link":
        lines.append(
            "LINK        "
            + one[12:27]
            + "               "
            + two[12:27]
            + "  1555   1555  1.54"
        )
    else:
        lines.append(f"CONECT{one[6:11]}{two[6:11]}")
    with pytest.raises(StructureNormalizationError, match="cannot preserve"):
        read(lines)


def fit(source, donor, ids=(4, 5, 6), left=3, right=7, library=None):
    return reconstruct_donor_span(
        source,
        scope=AbsentResidueSpanScope(rid(left), rid(right), tuple(map(rid, ids))),
        donor_structure=donor,
        donor_residue_ids=tuple(map(rid, ids)),
        donor_preceding_residue_id=rid(left),
        donor_following_residue_id=rid(right),
        component_library=library,
    )


@pytest.mark.parametrize(
    "witness", ["custom_flank", "extra_sidechain", "carbonyl_oxygen_contact"]
)
def test_only_attachment_chemistry_qualifies_gap_flanks(witness):
    lines = rows()
    lib = build_default_component_library()
    if witness == "custom_flank":
        lines = [r[:17] + "XTH" + r[20:] if int(r[22:26]) == 7 else r for r in lines]
        lib = ComponentLibrary(
            lib.templates, {**lib.alias_to_component_id, "XTH": "THR"}
        )
    elif witness == "extra_sidechain":
        atom = next(
            r for r in lines if int(r[22:26]) == 3 and r[12:16].strip() == "CD1"
        )
        lines.insert(
            lines.index(atom),
            atom[:6]
            + " 9998"
            + atom[11:12]
            + " CX1"
            + atom[16:54]
            + "  0.00"
            + atom[60:],
        )
    else:
        o = next(r for r in lines if int(r[22:26]) == 3 and r[12:16].strip() == "O")
        cb = next(r for r in lines if int(r[22:26]) == 8 and r[12:16].strip() == "CB")
        lines.append(f"CONECT{o[6:11]}{cb[6:11]}")
    source = read(lines)
    donor = read(rows(()), policy=ORDINARY)
    proposal = fit(source, donor, library=lib)
    assert isinstance(proposal, ReconstructedSpanCandidate), proposal
    output = proposal.materialize(source, lib)
    assert output.constitution.chain("A").residue(rid(5)).component_id == "VAL"
    if witness == "carbonyl_oxygen_contact":
        assert (
            output.topology.bond_between(
                *(
                    output.constitution.atom_index(ref)
                    for ref in (AtomRef(rid(3), "O"), AtomRef(rid(8), "CB"))
                )
            )
            is not None
        )


@pytest.mark.parametrize("zero_on", ["none", "donor", "source"])
def test_proline_cd_is_a_fit_atom_on_both_sides(zero_on):
    def window(zero, label):
        return [
            r[:54] + "  0.00" + r[60:]
            if zero_on == label and int(r[22:26]) == 37 and r[12:16].strip() == "CD"
            else r
            for r in rows(zero, range(34, 39))
        ]

    source = read(window((35, 36), "source"))
    donor = read(window((), "donor"), policy=ORDINARY)
    candidate = fit(source, donor, ids=(35, 36), left=34, right=37)
    if zero_on == "none":
        assert isinstance(candidate, ReconstructedSpanCandidate), candidate
    else:
        assert isinstance(candidate, SpanReconstructionFailure)
        assert "CD" in candidate.message


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_only_exact_donor_reinstates_all_junctions_and_preserves_flanks(fmt):
    # The middle slot contains one deposited placeholder; its identity is sufficient.
    lines = [r for r in rows() if int(r[22:26]) != 5 or r[12:16].strip() == "CA"]
    source = read(lines, fmt)
    records = source.provenance.ingress.consecutive_zero_occupancy_residue_omissions
    assert records is not None
    (omission,) = records
    assert (omission.component_ids, len(omission.members[1])) == (
        ("PHE", "VAL", "LYS"),
        1,
    )
    donor = read(
        [r for r in rows(()) if not (int(r[22:26]) == 5 and r[12:16].strip() == "CG2")],
        fmt,
        ORDINARY,
    )
    candidate = fit(source, donor)
    assert isinstance(candidate, ReconstructedSpanCandidate), candidate
    latest = candidate.materialize(source, build_default_component_library())
    assert not latest.constitution.chain("A").residue(rid(5)).has_atom_site("CG2")
    assert latest.provenance.ingress is source.provenance.ingress
    assert (
        apply_structure_normalization_policy(latest, policy=OMIT_CONSECUTIVE) is latest
    )
    for before, after in pairwise(range(3, 8)):
        bond = latest.topology.bond_between(
            latest.constitution.atom_index(AtomRef(rid(before), "C")),
            latest.constitution.atom_index(AtomRef(rid(after), "N")),
        )
        assert bond is not None and bond.order == 1
    for site in source.constitution.residue_slots:
        for atom in site.atom_sites:
            ref = AtomRef(site.residue_id, atom.name)
            assert latest.geometry.atom_geometry(
                latest.constitution.atom_index(ref)
            ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))
    with pytest.raises(SpanChemistryConflictError, match="another structure"):
        candidate.materialize(
            source.with_provenance(source.provenance), build_default_component_library()
        )
    with pytest.raises(SpanChemistryConflictError, match="payload changed"):
        replace(candidate, residue_payloads=candidate.residue_payloads[:1]).materialize(
            source, build_default_component_library()
        )
    assert isinstance(fit(latest, donor), SpanReconstructionFailure)
    assert isinstance(fit(source, donor, (4, 5), 3, 6), SpanReconstructionFailure)
    refreshed = read_structure_string(write_structure_string(latest, fmt), fmt)
    assert refreshed.constitution.chain("A").residue(rid(5)).component_id == "VAL"


@pytest.mark.parametrize(
    "wrong", ["middle_component", "inserted_zero", "unrelated_anchor_zero"]
)
def test_public_workflow_reports_the_selected_donor_outcome(wrong):
    lines = rows(())
    for i, row in enumerate(lines):
        n, name = int(row[22:26]), row[12:16].strip()
        if wrong == "middle_component" and n == 5:
            lines[i] = row[:17] + "ALA" + row[20:]
        if (n, name) == (
            (3, "CG2") if wrong == "unrelated_anchor_zero" else (5, "CG1")
        ) and wrong != "middle_component":
            lines[i] = row[:54] + "  0.00" + row[60:]
    if wrong == "middle_component":
        lines = [
            r
            for r in lines
            if int(r[22:26]) != 5 or r[12:16].strip() not in ("CG1", "CG2")
        ]
    donor = read(lines, policy=ORDINARY)
    source = read(rows())
    scope = AbsentResidueSpanScope(rid(3), rid(7), tuple(map(rid, (4, 5, 6))))
    result = process_structure(
        source,
        ingress=StructureIngressOptions(
            omit_consecutive_zero_occupancy_polymer_residues=True
        ),
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(scope, donor, scope.absent_residue_ids),
            )
        ),
    )
    if wrong == "unrelated_anchor_zero":
        assert (
            result.structure.constitution.chain("A").residue(rid(5)).component_id
            == "VAL"
        )
    else:
        assert result.structure == source
        assert not result.repairs
        assert any(
            (
                "requires VAL"
                if wrong == "middle_component"
                else "positive donor occupancy"
            )
            in issue.message
            for issue in result.issues
        )
