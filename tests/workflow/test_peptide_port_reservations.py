"""Interactions between input-bound gap ports and independent workflow writes."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.request_builders import c_terminal_oxt_requested_goals

from protrepair import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.io import FileFormat, read_structure_string, write_structure_string
from protrepair.io.gemmi_ingress import read_structure_string_with_policy
from protrepair.io.ingress_policy import (
    LigandHandling,
    NonPolymerRetentionPolicy,
    RetentionAction,
    StructureNormalizationPolicy,
)
from protrepair.io.rdkit import to_rdkit_molecule
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.scope import AbsentResidueSpanScope, WholeStructureScope
from protrepair.state import HydrogenCoverageState, SidechainHeavyAtomCompletenessState
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.bound_peptide_ports import BoundPeptidePorts, peptide_port_run
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.peptide import are_peptide_adjacent
from protrepair.structure.provenance import StructureIngress
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    LigandPolicy,
    PolymerChemicalSite,
    SiteProtonation,
    SiteProtonationReason,
    StructureIngressOptions,
    WorkflowPlanningContext,
    WorkflowSpanDonorAvailability,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.peptide_capping import (
    PeptideCappingRequest,
    PeptideCapStatus,
    PeptideEnd,
    PeptidePort,
)
from protrepair.workflow.engine import reporting, runtime
from protrepair.workflow.engine.hydrogen import project_hydrogen_issues
from protrepair.workflow.engine.processing import run_normalized_workflow
from protrepair.workflow.engine.runtime import WorkflowPlanningSession
from protrepair.workflow.planning.peptide_capping import (
    PeptideCapSchedule,
    PeptidePortReadiness,
)

DATA = Path(__file__).parents[1] / "fixtures/pdb/linked-caps/openmm-populated.pdb"
RID = ResidueId("A", 2)
N = AtomRef(RID, "N")
C = AtomRef(RID, "C")
H_GOAL = requested_process_goal(
    scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
)


def gly(
    *, seq: str = "ALA GLY SER", ligand: str = "", remark: str = ""
) -> ProteinStructure:
    rows = [
        s
        for s in DATA.read_text().splitlines()
        if s.startswith("ATOM") and s[17:20] == "GLY" and s[76:78].strip() != "H"
    ]
    return read_structure_string(
        "\n".join(
            (
                remark,
                f"SEQRES   1 A {len(seq.split()):4}  {seq}",
                *rows,
                "TER",
                ligand,
                "END",
            )
        ),
        FileFormat.PDB,
    )


def atoms(structure: ProteinStructure) -> set[str]:
    site = structure.constitution.residue_or_ligand(RID)
    assert site is not None
    return {a.name for a in site.atom_sites}


def only(*ports: AtomRef) -> PeptideCappingRequest:
    return PeptideCappingRequest.only(
        PeptidePort(p.residue_id, PeptideEnd(p.atom_name)) for p in ports
    )


@pytest.mark.parametrize("fmt", tuple(FileFormat))
def test_generated_products_survive_two_public_roundtrips_and_respect_drop(fmt):
    source = gly()
    request = WorkflowTransformRequests(peptide_capping=only(N, C))
    keep = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    first = process_structure(source, ingress=keep, transform_requests=request)
    assert first.peptide_capping_report is not None
    outcomes = first.peptide_capping_report.outcomes
    assert [r.port.atom_ref for r in outcomes] == [C, N]
    assert all(
        r.status is PeptideCapStatus.APPLIED and r.input_boundary is not None
        for r in outcomes
    )
    assert [r.cap_id for r in outcomes] == [ResidueId("A", 6), ResidueId("A", 7)]
    assert all(
        r.attachment is not None and r.attachment[0] == r.port.atom_ref
        for r in outcomes
    )
    assert atoms(first.structure) == atoms(source) | {"H"}

    product = first.structure
    for _ in range(2):
        reread = read_structure_string(write_structure_string(product, fmt), fmt)
        for parent in (product, reread):
            repeated = process_structure(
                parent, ingress=keep, transform_requests=request
            )
            dropped = process_structure(parent, transform_requests=request)
            assert (
                repeated.peptide_capping_report is not None
                and dropped.peptide_capping_report is not None
            )
            assert repeated.structure.constitution == parent.constitution
            assert repeated.structure.geometry == parent.geometry
            assert not dropped.structure.constitution.ligands
            for expected, seen, omitted in zip(
                outcomes,
                repeated.peptide_capping_report.outcomes,
                dropped.peptide_capping_report.outcomes,
                strict=True,
            ):
                assert (
                    seen.status is PeptideCapStatus.ALREADY_CAPPED
                    and seen.attachment == expected.attachment
                )
                assert (
                    omitted.status is PeptideCapStatus.OMITTED
                    and omitted.cap_id == expected.cap_id
                )
                assert omitted.attachment is None
            assert process_structure(parent).peptide_capping_report is None
        product = repeated.structure


def test_cap_that_loses_branch_selection_gets_no_coordinates_event_or_attachment(
    monkeypatch,
):
    score = runtime.evaluate_workflow_branch_quality_score

    def penalize_cap(result, **kwargs):
        actual = score(result, **kwargs)
        return replace(
            actual,
            protein_self_clash_count=int(bool(result.structure.constitution.ligands)),
        )

    monkeypatch.setattr(runtime, "evaluate_workflow_branch_quality_score", penalize_cap)
    source = gly()
    result = process_structure(
        source, transform_requests=WorkflowTransformRequests(peptide_capping=only(N))
    )
    assert result.peptide_capping_report is not None
    (outcome,) = result.peptide_capping_report.outcomes
    assert (
        outcome.status is PeptideCapStatus.NOT_RETAINED and outcome.attachment is None
    )
    assert result.structure == source and not result.repairs


def test_an_attached_cap_with_damaged_geometry_is_identified_but_not_credited():
    request = WorkflowTransformRequests(peptide_capping=only(C))
    keep = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    first = process_structure(gly(), ingress=keep, transform_requests=request)
    structure = first.structure
    cap = structure.constitution.ligands[0].residue_id
    g = structure.residue_geometry(structure.constitution.residue_index(cap))
    methyl = g.atom_geometry("C")
    damaged = structure.with_updated_residue_geometries(
        (
            (
                cap,
                g.with_atom_geometry(
                    "C", replace(methyl, position=methyl.position.with_offset(3, 0, 0))
                ),
            ),
        )
    )
    result = process_structure(damaged, ingress=keep, transform_requests=request)
    assert result.peptide_capping_report is not None
    (outcome,) = result.peptide_capping_report.outcomes
    assert outcome.status is PeptideCapStatus.ATTACHED_UNVERIFIED
    assert outcome.attachment == (C, AtomRef(cap, "N"))
    assert result.structure.geometry == damaged.geometry and not result.repairs


def test_generated_caps_allow_strict_native_projection_after_independent_h_completion():
    pytest.importorskip("rdkit")

    result = process_structure(
        gly(),
        requested_goals=(H_GOAL,),
        transform_requests=WorkflowTransformRequests(peptide_capping=only(C, N)),
    )
    mol = to_rdkit_molecule(result.structure)
    assert (mol.GetNumAtoms(), mol.GetNumBonds()) == (19, 18)
    assert all(not atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())


@pytest.mark.parametrize(
    "following, c_status, pending",
    [
        ("SER", PeptideCapStatus.APPLIED, []),
        ("FME", PeptideCapStatus.REFUSED, [("C", "O", "OXT")]),
    ],
)
def test_hydrogen_warnings_follow_each_final_cap_junction(following, c_status, pending):
    result = process_structure(
        gly(seq=f"ALA GLY {following}"),
        requested_goals=(H_GOAL,),
        transform_requests=WorkflowTransformRequests(peptide_capping=only(C, N)),
    )
    assert result.peptide_capping_report is not None
    c, n = result.peptide_capping_report.outcomes
    assert (c.port.atom_ref, c.status) == (C, c_status)
    assert (n.port.atom_ref, n.status) == (N, PeptideCapStatus.APPLIED)
    assert [
        issue.atom_names
        for issue in result.issues
        if issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        and issue.residue_id == RID
    ] == pending


def test_linkage_warning_requires_a_verified_product_and_keeps_other_stage_warnings():
    source = gly()
    binding = BoundPeptidePorts.bind(source, only=(C,))
    verified = process_structure(
        source, transform_requests=WorkflowTransformRequests(peptide_capping=only(C))
    ).structure
    cap = verified.constitution.ligands[0].residue_id
    incomplete = verified.without_atom_refs((AtomRef(cap, "HN2"),))
    library = build_default_component_library()

    with peptide_port_run(binding):
        stage = materialize_hydrogens_core(source, library)
        (linkage,) = (
            issue
            for issue in stage.issues
            if issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
            and issue.affects_atom("C")
        )
        other = ValidationIssue.for_residue(
            kind=linkage.kind,
            severity=linkage.severity,
            message=linkage.message,
            residue_id=RID,
            component_id="GLY",
            atom_names=linkage.atom_names,
        )
        for carrier, expected in (
            (verified, (other,)),
            (incomplete, (linkage, other)),
        ):
            assert (
                project_hydrogen_issues(
                    replace(stage, structure=carrier, issues=(linkage, other)),
                    requested_goal_report=None,
                    component_library=library,
                )
                == expected
            )


@pytest.mark.parametrize(
    "shift, expected", [(-2, PeptideCapStatus.REFUSED), (20, PeptideCapStatus.APPLIED)]
)
def test_two_facing_ports_are_checked_against_the_previously_adopted_cap(
    shift, expected
):
    rows = [
        r
        for r in DATA.read_text().splitlines()
        if r.startswith("ATOM ") and r[17:20] == "GLY"
    ]
    other_rows = []
    # The two prospective cap methyl groups coincide before the x offset.
    for row in rows:
        x, y, z = (float(row[i : i + 8]) for i in (30, 38, 46))
        other_rows.append(
            row[:6]
            + f"{int(row[6:11]) + 20:5}"
            + row[11:22]
            + "   4"
            + row[26:30]
            + f"{x - 6.924562 + shift:8.3f}{y - 2.042206:8.3f}{z + 0.838319:8.3f}"
            + row[54:]
        )
    source = read_structure_string(
        "\n".join(("SEQRES   1 A    3  GLY ALA GLY", *rows, *other_rows, "END")),
        FileFormat.PDB,
    )
    right = AtomRef(ResidueId("A", 4), "N")
    separate = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(peptide_capping=only(right)),
    )
    together = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(peptide_capping=only(right, C)),
    )
    assert (
        separate.peptide_capping_report is not None
        and together.peptide_capping_report is not None
    )
    assert (
        separate.peptide_capping_report.outcomes[0].status is PeptideCapStatus.APPLIED
    )
    first, second = together.peptide_capping_report.outcomes
    assert first.port.atom_ref == C and first.status is PeptideCapStatus.APPLIED
    assert second.port.atom_ref == right and second.status is expected
    if expected is PeptideCapStatus.REFUSED:
        assert second.reason == "no_acceptable_pose" and dict(second.pose_refusals) == {
            "steric_contact": 4
        }
        assert (
            second.attachment is None
            and len(together.structure.constitution.ligands) == 1
        )


def test_a_stationary_water_can_veto_every_pose_without_moving():
    rows = [
        r
        for r in DATA.read_text().splitlines()
        if r.startswith("ATOM ") and r[17:20] == "GLY"
    ]
    water = (
        "HETATM   41  O   HOH A  41       2.914   1.876  -1.559  1.00 12.00           O"
    )
    keep = NonPolymerRetentionPolicy(RetentionAction.KEEP, RetentionAction.KEEP)
    parent = read_structure_string_with_policy(
        "\n".join(
            (
                "SEQRES   1 A    3  ALA GLY SER",
                *rows,
                "TER",
                water,
                "CONECT    9   10   11",
                "CONECT    7    8",
                "END",
            )
        ),
        FileFormat.PDB,
        policy=StructureNormalizationPolicy(non_polymer_retention=keep),
    )
    result = process_structure(
        parent,
        ingress=StructureIngressOptions(non_polymer_retention=keep),
        transform_requests=WorkflowTransformRequests(peptide_capping=only(C)),
    )
    assert result.peptide_capping_report is not None
    (outcome,) = result.peptide_capping_report.outcomes
    assert outcome.status is PeptideCapStatus.REFUSED
    assert (
        dict(outcome.pose_refusals) == {"stationary_water": 4}
        and "A:2.C" in outcome.detail
    )
    assert result.structure.geometry == parent.geometry and not result.repairs


@pytest.mark.parametrize("element", ["H", "D", "T"])
def test_an_unbonded_deposited_amide_h_is_used_without_moving_or_renaming_it(element):
    rows = [
        r
        for r in DATA.read_text().splitlines()
        if r.startswith("ATOM ") and r[17:20] == "GLY"
    ]
    rows = [
        r[:12] + f" {element}  " + r[16:76] + f"{element:>2}" + r[78:]
        if r[12:16].strip() == "H"
        else r
        for r in rows
    ]
    parent = read_structure_string(
        "\n".join(("SEQRES   1 A    3  ALA GLY SER", *rows, "TER", "END")),
        FileFormat.PDB,
    )
    result = process_structure(
        parent, transform_requests=WorkflowTransformRequests(peptide_capping=only(N))
    )
    assert result.peptide_capping_report is not None
    (outcome,) = result.peptide_capping_report.outcomes
    assert outcome.status is PeptideCapStatus.APPLIED
    ref = AtomRef(RID, element)
    for site in parent.constitution.chains[0].residues[0].atom_sites:
        old = parent.constitution.atom_index(AtomRef(RID, site.name))
        new = result.structure.constitution.atom_index(AtomRef(RID, site.name))
        assert parent.geometry.atom_geometry(
            old
        ) == result.structure.geometry.atom_geometry(new)
        assert parent.constitution.atom_site_at(
            old
        ) == result.structure.constitution.atom_site_at(new)
    n, h = (result.structure.constitution.atom_index(r) for r in (N, ref))
    bond = result.structure.topology.bond_between(n, h)
    assert (
        bond is not None
        and bond.order == 1
        and atoms(parent) == atoms(result.structure)
    )


def test_unbonded_carbonyl_deuterium_is_not_silently_reinterpreted():
    source = gly()
    oxygen = source.geometry.position(source.constitution.atom_index(AtomRef(RID, "O")))
    site = source.constitution.residue_site_at(source.constitution.residue_index(RID))

    geometry = source.residue_geometry(source.constitution.residue_index(RID))
    parent = source.with_updated_residue_facets(
        site.with_atom_site(AtomSite("D7", "D")),
        residue_geometry=geometry.with_atom_geometry(
            "D7", AtomGeometry(oxygen.with_offset(0.98, 0, 0))
        ),
        formal_charge_by_atom_name=(),
    )
    result = process_structure(
        parent, transform_requests=WorkflowTransformRequests(peptide_capping=only(C))
    )
    assert result.peptide_capping_report is not None
    (outcome,) = result.peptide_capping_report.outcomes
    assert (
        outcome.status is PeptideCapStatus.REFUSED
        and outcome.reason == "possible_carbonyl_h"
    )
    assert result.structure == parent and outcome.attachment is None


@pytest.mark.parametrize("ports", [(), (N, N)])
def test_explicit_cap_selection_must_name_distinct_ports(ports):
    with pytest.raises(ValueError, match="distinct"):
        only(*ports)


@pytest.mark.parametrize(
    "selected,free_n,oxt",
    [
        ((N,), False, True),
        ((C,), True, False),
        ((N, C), False, False),
    ],
)
def test_independent_h_and_oxt_on_a_proven_gap(selected, free_n, oxt):
    source = gly()
    goals = (H_GOAL, *c_terminal_oxt_requested_goals(RID))
    binding = BoundPeptidePorts.bind(source, only=selected)
    actual = run_normalized_workflow(
        source,
        ingress=StructureIngressOptions(),
        transform_requests=WorkflowTransformRequests(peptide_capping=only(*selected)),
        requested_goals=goals,
    )
    with peptide_port_run(binding):
        direct = augment_c_terminal_oxt(source)
        heavy = repair_heavy_atoms_core(source)
    names = atoms(actual.structure)
    ca = actual.structure.constitution.atom_index(AtomRef(RID, "CA"))
    alpha_h = {h for h in names if h.startswith("HA")}
    assert len(alpha_h) == 2
    assert all(
        actual.structure.topology.bond_between(
            ca, actual.structure.constitution.atom_index(AtomRef(RID, h))
        )
        is not None
        for h in alpha_h
    )
    assert names.intersection(("H", "H1", "H2", "H3")) == (
        {"H1", "H2", "H3"} if free_n else {"H"}
    )
    assert ("OXT" in names) == oxt
    assert ("OXT" in atoms(direct.structure)) == oxt
    assert ("OXT" in atoms(heavy.structure)) == oxt
    control = process_structure(source, requested_goals=goals)
    assert {"OXT", "H1", "H2", "H3"} <= atoms(control.structure)


def test_real_end_and_unrelated_input_do_not_inherit_the_reservation():
    gap = gly()
    end = gly(seq="GLY SER")
    result = run_normalized_workflow(
        end,
        ingress=StructureIngressOptions(),
        transform_requests=WorkflowTransformRequests(
            peptide_capping=PeptideCappingRequest()
        ),
        requested_goals=(H_GOAL,),
    )
    assert {"H1", "H2", "H3"} <= atoms(result.structure)
    assert "OXT" not in atoms(result.structure)
    with peptide_port_run(BoundPeptidePorts.bind(gap)):
        independent = process_structure(gap, requested_goals=(H_GOAL,))
    assert {"OXT", "H1", "H2", "H3"} <= atoms(independent.structure)


@pytest.mark.parametrize(
    "missing,open_n",
    [("FME", False), ("SAC", False), ("AYA", False), ("PRO", True), ("MSE", True)],
)
def test_missing_neighbors_must_have_a_linkable_end_in_the_required_direction(
    missing, open_n
):
    source = gly(seq=f"{missing} GLY {missing}")
    result = process_structure(
        source, transform_requests=WorkflowTransformRequests(peptide_capping=only(N, C))
    )
    report = result.peptide_capping_report
    assert report is not None
    c, n = report.outcomes
    assert n.port.atom_ref == N and n.status is PeptideCapStatus.APPLIED
    assert c.port.atom_ref == C
    if open_n:
        assert c.status is PeptideCapStatus.APPLIED
    else:
        assert c.status is PeptideCapStatus.REFUSED and c.reason == "unknown_neighbor"
        assert c.attachment is None


def test_deferred_report_reinstates_its_own_terminal_chemistry(monkeypatch):
    source = gly()
    reserved = run_normalized_workflow(
        source,
        ingress=StructureIngressOptions(),
        transform_requests=WorkflowTransformRequests(peptide_capping=only(C)),
    )
    legacy = process_structure(source)
    original = reporting.evaluate_workflow_phase_outcomes
    required = []

    def observe(structure, **kwargs):
        preparation = AminoAcidPreparation(
            PolymerMicrostateContext(structure), build_default_component_library()
        )
        required.append(
            any(t.requires_terminal_oxygen() for t in preparation.targets_for(RID))
        )
        return original(structure, **kwargs)

    monkeypatch.setattr(reporting, "evaluate_workflow_phase_outcomes", observe)
    for result, expected in ((reserved, False), (legacy, True)):
        assert result.branch_report is not None
        assert result.branch_report.preferred_outcome().phase_report.outcomes
        assert required[-1] is expected


def test_old_preparation_cannot_restore_free_chemistry_and_requests_conflict():
    source = gly()
    library = build_default_component_library()
    free = AminoAcidPreparation(PolymerMicrostateContext(source), library)
    binding = BoundPeptidePorts.bind(source)
    with peptide_port_run(binding):
        blocked = free.for_inputs(source, library)
        assert not free.matches_chemistry(source)
        assert not any(t.requires_terminal_oxygen() for t in blocked.targets_for(RID))
        assert blocked.for_inputs(source, library) is blocked
        result = run_normalized_workflow(
            source,
            ingress=StructureIngressOptions(),
            transform_requests=WorkflowTransformRequests(
                peptide_capping=PeptideCappingRequest(),
                site_protonations=(
                    SiteProtonation(RID, PolymerChemicalSite.BACKBONE_N, site_charge=1),
                ),
            ),
        )
    assert not blocked.matches_chemistry(source)
    assert result.site_protonation_report is not None
    (outcome,) = result.site_protonation_report.outcomes
    assert outcome.reason is SiteProtonationReason.PHYSICAL_CONSTRAINT
    assert atoms(result.structure).intersection(("H", "H1", "H2", "H3")) == {"H"}


def test_deliberately_dropped_cap_is_reserved_even_without_a_declared_sequence():
    kept = read_structure_string(DATA.read_text(), FileFormat.PDB)
    dropped = apply_structure_normalization_policy(
        kept, policy=StructureNormalizationPolicy(ligand_handling=LigandHandling.DROP)
    )
    c_port = AtomRef(ResidueId("A", 3), "C")
    binding = BoundPeptidePorts.bind(dropped, only=(c_port,))
    assert binding.reserves(dropped, c_port) and not binding.reserves(dropped, N)
    assert not binding.cap_ids
    with peptide_port_run(binding):
        assert augment_c_terminal_oxt(dropped).structure is dropped
    assert augment_c_terminal_oxt(dropped).structure is not dropped

    def canonical() -> ProteinStructure:
        return dropped.with_provenance(
            replace(dropped.provenance, ingress=StructureIngress(FileFormat.PDB))
        )

    first, other = canonical(), canonical()
    programmatic = BoundPeptidePorts.bind(first)
    assert programmatic.reserves(first, c_port)
    assert not programmatic.reserves(other, c_port)


def test_cap_numbers_exclude_raw_dropped_waters_and_requested_insertions():
    water = (
        "HETATM   40  O   HOH A   6A     99.000  99.000  99.000  1.00 10.00           O"
    )
    source = gly(ligand=water, remark="REMARK 465     SER A     8B")
    assert not source.constitution.waters
    extra = frozenset((ResidueId("A", 7),))
    for ports in (None, (N, C), (C, N)):
        binding = BoundPeptidePorts.bind(source, only=ports, graft_ids=extra)
        assert binding.cap_ids == {C: ResidueId("A", 9), N: ResidueId("A", 10)}


@pytest.mark.parametrize("next_number, has_edge", [(3, True), (5, False)])
def test_a_short_gap_requires_a_real_current_edge_for_torsions(next_number, has_edge):
    rows = [
        s.replace("ALA A   3", f"ALA A{next_number:4d}")
        for s in DATA.read_text().splitlines()
        if s.startswith("ATOM")
    ]
    source = read_structure_string(
        "\n".join(
            (
                "SEQRES   1 A    3  GLY SER ALA",
                *rows,
                "END",
            )
        ),
        FileFormat.PDB,
    )
    left, right = source.constitution.chains[0].residues
    assert are_peptide_adjacent(left, right, structure=source)
    a = source.constitution.atom_index(AtomRef(left.residue_id, "C"))
    b = source.constitution.atom_index(AtomRef(right.residue_id, "N"))
    assert (source.topology.bond_between(a, b) is not None) == has_edge
    with peptide_port_run(BoundPeptidePorts.bind(source, only=(C,))):
        assert are_peptide_adjacent(left, right, structure=source) == has_edge


@pytest.fixture
def schedules(monkeypatch) -> list[PeptideCapSchedule]:
    observed = []
    plan = WorkflowPlanningSession.plan

    def capture(session, state):
        result = plan(session, state)
        if result.peptide_cap_schedule is not None:
            observed.append(result.peptide_cap_schedule)
        return result

    monkeypatch.setattr(WorkflowPlanningSession, "plan", capture)
    return observed


@pytest.mark.parametrize("case", ["restore", "budget", "failed_anchor"])
def test_graft_controls_admission_to_new_pose_assessment(case, schedules):
    rows = [
        s
        for s in (DATA.parents[1] / "span-reconstruction/1ubq.pdb")
        .read_text()
        .splitlines()
        if s.startswith("ATOM  ") and 3 <= int(s[22:26]) <= 7
    ]

    def read(records: list[str]) -> ProteinStructure:
        return read_structure_string(
            "\n".join(("SEQRES   1 A    5  ILE PHE VAL LYS THR", *records, "END")),
            FileFormat.PDB,
        )

    donor = read(rows)
    source = read(
        [
            s
            for s in rows
            if int(s[22:26]) != 5
            and not (
                case == "failed_anchor"
                and int(s[22:26]) == 4
                and s[12:16].strip() == "CA"
            )
        ]
    )
    before, insert, after = (ResidueId("A", i) for i in (4, 5, 6))
    left = AtomRef(before, "C")
    right = AtomRef(after, "N")
    request = ExternalSpanReconstructionSpec(
        scope=AbsentResidueSpanScope(
            preceding_residue_id=before,
            following_residue_id=after,
            absent_residue_ids=(insert,),
        ),
        donor_structure=donor,
        donor_residue_ids=(insert,),
    )
    binding = BoundPeptidePorts.bind(source, graft_ids=frozenset({insert}))
    assert binding.reserves(source, left) and binding.reserves(source, right)
    with peptide_port_run(binding):
        result = run_normalized_workflow(
            source,
            ingress=StructureIngressOptions(),
            transform_requests=WorkflowTransformRequests(
                peptide_capping=PeptideCappingRequest(),
                external_span_reconstructions=(request,),
            ),
            requested_goals=(H_GOAL,) if case == "restore" else (),
            planning_context=WorkflowPlanningContext(
                span_donor_availability=WorkflowSpanDonorAvailability.AVAILABLE,
                max_speculative_nodes=1 if case == "budget" else 128,
            ),
        )
        child = result.structure
        assert result.peptide_capping_report is not None
        outcomes = {o.port.atom_ref: o for o in result.peptide_capping_report.outcomes}
        assert {
            p
            for p, _, status in schedules[0].entries
            if status is PeptidePortReadiness.PENDING_GRAFT
        } == {left, right}
        if case == "budget":
            for port in (left, right):
                assert outcomes[port].status is PeptideCapStatus.SEARCH_LIMIT
                assert outcomes[port].reason == "pending_graft"
                assert outcomes[port].attachment is None
            assert all(not list(plan.assessable_ports()) for plan in schedules)
            assert child.constitution.residue_or_ligand(insert) is None
            return
        if case == "failed_anchor":
            assert outcomes[left].reason == "missing_heavy_anchor"
            assert result.external_span_reconstruction_report is not None
            assert child.constitution.residue_or_ligand(insert) is None
            assert any(
                {p for p, _ in plan.assessable_ports()} == {left, right}
                for plan in schedules
            )
            return
        assert any(
            {
                p
                for p, _, status in plan.entries
                if status is PeptidePortReadiness.OCCUPIED_PORT
            }
            == {left, right}
            for plan in schedules
        )
        assert child.constitution.residue_or_ligand(insert) is not None
        assert outcomes[left].attachment == (left, AtomRef(insert, "N"))
        assert outcomes[right].attachment == (right, AtomRef(insert, "C"))
        assert all(
            outcomes[port].status is PeptideCapStatus.OCCUPIED for port in (left, right)
        )
        assert not binding.reserves(child, AtomRef(insert, "N"))
        assert binding.reserves(child, right)
        for a, b in (
            (left, AtomRef(insert, "N")),
            (AtomRef(insert, "C"), right),
            (right, AtomRef(after, "H")),
        ):
            bond = child.topology.bond_between(
                child.constitution.atom_index(a), child.constitution.atom_index(b)
            )
            assert bond is not None and bond.order == 1
        assert child.constitution.resolve_atom_index(AtomRef(before, "OXT")) is None
        assert child.constitution.resolve_atom_index(AtomRef(after, "H1")) is None


def test_heavy_prerequisite_defers_only_ports_on_its_residue(schedules):
    rows = [
        s
        for s in DATA.read_text().splitlines()
        if s.startswith("ATOM") and s[12:16].strip() != "CB" and s[76:78].strip() != "H"
    ]
    source = read_structure_string(
        "\n".join(("SEQRES   1 A    4  SER GLY ALA THR", *rows, "END")), FileFormat.PDB
    )
    other = AtomRef(ResidueId("A", 3), "C")
    result = run_normalized_workflow(
        source,
        ingress=StructureIngressOptions(),
        transform_requests=WorkflowTransformRequests(
            peptide_capping=PeptideCappingRequest()
        ),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(),
                value=SidechainHeavyAtomCompletenessState.COMPLETE,
            ),
        ),
    )
    assert {p: status for p, _, status in schedules[0].entries} == {
        N: PeptidePortReadiness.ASSESSABLE,
        other: PeptidePortReadiness.PENDING_HEAVY_REPAIR,
    }
    assert (
        result.structure.constitution.resolve_atom_index(
            AtomRef(other.residue_id, "CB")
        )
        is not None
    )
    assert any(
        {p for p, _ in plan.assessable_ports()} == {N, other} for plan in schedules
    )


def test_later_port_cannot_overtake_an_earlier_pending_port(schedules):
    rows = [
        s
        for s in DATA.read_text().splitlines()
        if s.startswith("ATOM") and s[12:16].strip() != "CB" and s[76:78].strip() != "H"
    ]
    a = [s for s in rows if s[17:20] == "ALA"]
    b = [
        s[:21] + "B" + s[22:30] + f"{100 + float(s[30:38]):8.3f}" + s[38:]
        for s in rows
        if s[17:20] == "GLY"
    ]
    source = read_structure_string(
        "\n".join(
            (
                "SEQRES   1 A    2  SER ALA",
                "SEQRES   1 B    2  GLY SER",
                *a,
                "TER",
                *b,
                "END",
            )
        ),
        FileFormat.PDB,
    )
    result = run_normalized_workflow(
        source,
        ingress=StructureIngressOptions(),
        transform_requests=WorkflowTransformRequests(
            peptide_capping=PeptideCappingRequest()
        ),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(),
                value=SidechainHeavyAtomCompletenessState.COMPLETE,
            ),
        ),
    )
    assert [s for _, _, s in schedules[0].entries] == [
        PeptidePortReadiness.PENDING_HEAVY_REPAIR,
        PeptidePortReadiness.ASSESSABLE,
    ]
    assert not list(schedules[0].assessable_ports())
    assert any(len(tuple(plan.assessable_ports())) == 2 for plan in schedules)
    assert (
        result.structure.constitution.resolve_atom_index(
            AtomRef(ResidueId("A", 3), "CB")
        )
        is not None
    )
