"""A site request owns a chemical choice, not a whole-residue H repair."""

from dataclasses import replace
from pathlib import Path

import gemmi
import pytest

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.bound_protonation import (
    BoundSiteProtonations,
    site_protonation_run,
)
from protrepair.chemistry.microstate.catalog import standard_microstate_candidates
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import GeometryPlacementError
from protrepair.io import read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen.protonation import (
    histidine_microstate_requests,
)
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.packing.faspr.paths import faspr_executable_path
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions import site_protonation
from protrepair.workflow.contracts import (
    LigandPolicy,
    PackingSpec,
    PolymerChemicalSite,
    PrasRatioHistidineProtonationRequest,
    RequestedGoalSet,
    SiteProtonation,
    SiteProtonationReason,
    StructureIngressOptions,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.planning import (
    WorkflowBranchPreferencePolicy,
    WorkflowBranchQualityAxis,
)
from protrepair.workflow.engine import reporting, runtime
from protrepair.workflow.planning.completion.transformer_candidates import (
    plan_hydrogen_completion_transformers,
)

ASP = ResidueId("A", 43)
SIDE = PolymerChemicalSite.SIDECHAIN


def _asp():
    rows = (
        Path("tests/fixtures/pdb/span-reconstruction/1crn.pdb").read_text().splitlines()
    )
    records = [row for row in rows if row.startswith("ATOM") and row[17:20] == "ASP"]
    return read_structure_string("\n".join([*records, "END"]), FileFormat.PDB)


def _process(structure, *requests):
    return process_structure(
        structure,
        transform_requests=WorkflowTransformRequests(site_protonations=requests),
    )


def test_site_only_request_needs_no_global_goal_or_terminal_oxygen():
    source = _asp()
    request = SiteProtonation(ASP, SIDE, site_charge=0, parent_hydrogens={"OD2": 1})
    result = _process(source, request)
    assert result.site_protonation_report is not None
    (row,) = result.site_protonation_report.outcomes
    assert row.status.value == "applied"
    before = source.constitution.residue_or_ligand(ASP)
    after = result.structure.constitution.residue_or_ligand(ASP)
    assert before is not None and after is not None
    assert set(after.atom_site_names()) - set(before.atom_site_names()) == {"HD2"}
    again = _process(result.structure, request).site_protonation_report
    assert again is not None and again.outcomes[0].status.value == "already_satisfied"


def test_ambiguous_charge_stays_unmet_and_absent_sibling_does_not_abort():
    source = _asp()
    requests = (
        SiteProtonation(ASP, SIDE, site_charge=0),
        SiteProtonation(ResidueId("A", 42), SIDE, site_charge=1),
    )
    result = _process(source, *requests)
    assert result.site_protonation_report is not None
    first, second = result.site_protonation_report.outcomes
    assert first.assessment.value == "ambiguous"
    assert len(first.classes) == 2
    assert second.assessment.value == "unsupported"
    assert second.reason is SiteProtonationReason.ABSENT_AT_INPUT
    assert result.structure.constitution == source.constitution


def test_parent_map_is_snapshotted_and_duplicate_sites_are_errors():
    parents = {"OD2": 1}
    request = SiteProtonation(ASP, SIDE, site_charge=0, parent_hydrogens=parents)
    parents.clear()
    assert request.parent_hydrogens == (("OD2", 1),)
    with pytest.raises(ValueError, match="duplicate"):
        WorkflowTransformRequests(site_protonations=(request, request))
    with pytest.raises(TypeError):
        SiteProtonation(ASP, SIDE, site_charge=False)


HIS = ResidueId("A", 42)
H_GOAL = (
    requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    ),
)


def _his(two=False):
    rows = (
        Path("tests/fixtures/pdb/refinement/1bkr_his42_local.pdb")
        .read_text()
        .splitlines()
    )
    first = [
        row
        for row in rows
        if row.startswith("ATOM") and row[17:20] == "HIS" and int(row[22:26]) == 42
    ]
    second = (
        [
            row[:6]
            + f"{int(row[6:11]) + 20:5d}"
            + row[11:22]
            + "  44"
            + row[26:30]
            + f"{float(row[30:38]) + 30:8.3f}"
            + row[38:]
            for row in first
        ]
        if two
        else []
    )
    return read_structure_string("\n".join([*first, *second, "END"]), FileFormat.PDB)


def _atoms(structure, rid):
    return set(
        structure.constitution.residue_site_at(
            structure.constitution.residue_index(rid)
        ).atom_site_names()
    )


def _outcome(result):
    assert result.site_protonation_report is not None
    return result.site_protonation_report.outcomes[0]


def test_his31_on_the_returned_branch_with_and_without_a_whole_structure_h_goal():
    source = Path("tests/fixtures/pdb/protonation/3htb_his31_asp70.pdb")
    site = ResidueId("A", 31)
    ring_h = {"HD1", "HE2"}
    request = SiteProtonation(site, SIDE, site_charge=1)
    requests = WorkflowTransformRequests(site_protonations=(request,))

    ordinary = process_structure(source, requested_goals=H_GOAL)
    assert _atoms(ordinary.structure, site) & ring_h == {"HD1"}
    standalone = process_structure(source, transform_requests=requests)
    assert _outcome(standalone).status.value == "applied"
    assert _atoms(standalone.structure, site) & ring_h == ring_h

    default = process_structure(
        source, requested_goals=H_GOAL, transform_requests=requests
    )
    assert _outcome(default).assessment.value == "resolved"
    assert _outcome(default).status.value == "not_retained"
    assert not _atoms(default.structure, site) & ring_h
    assert default.requested_goal_report is not None
    assert default.requested_goal_report.unmet_count() == 1

    axis = WorkflowBranchQualityAxis
    goals_before_clashes = WorkflowBranchPreferencePolicy(
        axes=(
            axis.HARD_ERRORS,
            axis.PARSER_COMPATIBILITY,
            axis.STEREOCHEMISTRY,
            axis.SEVERE_GEOMETRY,
            axis.REQUESTED_GOALS,
            axis.INTRINSIC_CLASH,
            axis.INTERACTION_CLASH,
            axis.ISSUE_BURDEN,
            axis.SEARCH_DEPTH,
        )
    )
    preferred = process_structure(
        source,
        requested_goals=H_GOAL,
        transform_requests=requests,
        planning_context=WorkflowPlanningContext(
            branch_preference_policy=goals_before_clashes
        ),
    )
    assert _outcome(preferred).assessment.value == "resolved"
    assert _outcome(preferred).status.value == "applied"
    assert preferred.requested_goal_report is not None
    assert preferred.requested_goal_report.satisfied_count() == 1

    pdb = gemmi.read_pdb_string(
        write_structure_string(preferred.structure, FileFormat.PDB)
    )
    his = next(residue for residue in pdb[0]["A"] if residue.seqid.num == 31)
    assert {atom.name for atom in his} & ring_h == ring_h
    assert sum(atom.charge for atom in his if atom.name in {"ND1", "NE2"}) == 1


def test_global_h_goal_leaves_the_node_budget_for_eight_site_writes():
    pdb = gemmi.read_pdb_string(write_structure_string(_his(), FileFormat.PDB))
    requests = []
    for offset, chain_id in enumerate("ABCDEFGH"):
        if offset:
            chain = pdb[0][0].clone()
            chain.name = chain_id
            for atom in chain[0]:
                atom.pos.x += 12 * offset
            pdb[0].add_chain(chain)
        requests.append(SiteProtonation(ResidueId(chain_id, 42), SIDE, site_charge=1))
    source = read_structure_string(pdb.make_pdb_string(), FileFormat.PDB)
    prepared = process_structure(source, requested_goals=H_GOAL).structure

    result = process_structure(
        prepared,
        requested_goals=H_GOAL,
        transform_requests=WorkflowTransformRequests(site_protonations=tuple(requests)),
        planning_context=WorkflowPlanningContext(
            max_speculative_nodes=1 + len(requests)
        ),
    )
    assert result.site_protonation_report is not None
    assert [row.status.value for row in result.site_protonation_report.outcomes] == [
        "applied"
    ] * len(requests)
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.satisfied_count() == 1
    for request in requests:
        assert {"HD1", "HE2"} <= _atoms(result.structure, request.residue_id)


@pytest.mark.parametrize("whole_structure", (True, False), ids=("goal", "prerequisite"))
@pytest.mark.parametrize("missing", (None, "HA", "H1"))
@pytest.mark.parametrize("charge", (1, 0), ids=("resolved", "ambiguous"))
def test_general_h_proposal_depends_on_work_outside_the_requested_site(
    whole_structure, missing, charge
):
    source = process_structure(_his(), requested_goals=H_GOAL).structure
    if missing is not None:
        source = source.without_atom_refs((AtomRef(HIS, missing),))
    request = SiteProtonation(HIS, SIDE, site_charge=charge)
    with site_protonation_run(BoundSiteProtonations.bind(source, (request,))):
        general = plan_hydrogen_completion_transformers(
            source,
            requested_goals=RequestedGoalSet(H_GOAL if whole_structure else ()),
            required_residue_ids=() if whole_structure else (HIS,),
            component_library=build_default_component_library(),
        )
    assert bool(general.transformers) is (missing is not None)


@pytest.mark.parametrize("fixed_h", (("HA",), ("HA", "HB1", "HB2")))
def test_reserved_site_keeps_independent_fixed_inventory_failure_without_an_h_step(
    fixed_h,
):
    source = _his()
    prepared = process_structure(source, requested_goals=H_GOAL).structure
    observed = gemmi.read_pdb_string(write_structure_string(source, FileFormat.PDB))
    reference = gemmi.read_pdb_string(write_structure_string(prepared, FileFormat.PDB))
    residue = observed[0][0][0]
    for name in fixed_h:
        residue.add_atom(reference[0][0][0][name][0])
    extra = reference[0][0][0]["HA"][0].clone()
    extra.name, extra.element = "DA", gemmi.Element("D")
    residue.add_atom(extra)
    source = read_structure_string(observed.make_pdb_string(), FileFormat.PDB)
    prepared = process_structure(source, requested_goals=H_GOAL).structure

    requests = WorkflowTransformRequests(
        site_protonations=(SiteProtonation(HIS, SIDE, site_charge=1),)
    )
    for goals in ((), H_GOAL):
        result = process_structure(
            prepared,
            requested_goals=goals,
            transform_requests=requests,
            planning_context=WorkflowPlanningContext(max_speculative_nodes=2),
        )
        assert _outcome(result).status.value == "applied"
        assert {"HA", "DA", "HD1", "HE2"} <= _atoms(result.structure, HIS)
        failures = [
            issue
            for issue in result.issues
            if issue.kind is ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN
            and issue.residue_id == HIS
            and "source H exceed fixed template inventory at CA" in issue.message
        ]
        assert len(failures) == int(bool(goals))
        if goals:
            assert result.requested_goal_report is not None
            assert result.requested_goal_report.satisfied_count() == int(
                "HB2" in fixed_h
            )


@pytest.mark.parametrize("ambiguous", (False, True))
def test_ratio_masks_requested_ring_without_refilling_and_leaves_siblings_working(
    ambiguous,
):
    source = _his(two=True)
    selected = SiteProtonation(
        HIS, SIDE, site_charge=0, parent_hydrogens={} if ambiguous else {"ND1": 1}
    )
    result = process_structure(
        source,
        requested_goals=H_GOAL,
        transform_requests=WorkflowTransformRequests(
            site_protonations=(selected,),
            histidine_protonation=PrasRatioHistidineProtonationRequest(0.5),
        ),
    )
    names = _atoms(result.structure, HIS)
    assert "HA" in names
    assert names.intersection({"HE2", "HD1"}) == (set() if ambiguous else {"HD1"})
    assert (
        len(_atoms(result.structure, ResidueId("A", 44)).intersection({"HD1", "HE2"}))
        == 1
    )
    assert _outcome(result).assessment.value == (
        "ambiguous" if ambiguous else "resolved"
    )


def test_ratio_recalculates_its_prefix_on_the_branch_before_masking():
    source = _his(two=True)
    request = SiteProtonation(HIS, SIDE, site_charge=0)
    ratio = PrasRatioHistidineProtonationRequest(0.5)
    atoms = [
        r
        for r in write_structure_string(source, FileFormat.PDB).splitlines()
        if r.startswith("ATOM")
    ]
    earlier = [
        r[:6] + f"{int(r[6:11]) + 50:5d}" + r[11:22] + "  40" + r[26:]
        for r in atoms
        if int(r[22:26]) == 42
    ]
    expanded = read_structure_string(
        "\n".join([*earlier, *atoms, "END"]), FileFormat.PDB
    )
    with site_protonation_run(BoundSiteProtonations.bind(source, (request,))):
        before = histidine_microstate_requests(source, ratio)
        after = histidine_microstate_requests(expanded, ratio)
    assert before == {(ResidueId("A", 44), SIDE): None}
    assert (HIS, SIDE) not in after
    assert after[(ResidueId("A", 40), SIDE)] is not None
    assert after[(ResidueId("A", 44), SIDE)] is None


def test_committed_class_replaces_old_ratio_override_but_file_reread_is_source():
    source = _his()
    hip = process_structure(
        source,
        requested_goals=H_GOAL,
        transform_requests=WorkflowTransformRequests(
            histidine_protonation=PrasRatioHistidineProtonationRequest(1.0)
        ),
    )
    assert {"HD1", "HE2"} <= _atoms(hip.structure, HIS)
    hid = SiteProtonation(HIS, SIDE, site_charge=0, parent_hydrogens={"ND1": 1})
    result = _process(hip.structure, hid)
    assert _outcome(result).status.value == "applied"
    later = process_structure(result.structure, requested_goals=H_GOAL)
    assert _atoms(later.structure, HIS).intersection({"HD1", "HE2"}) == {"HD1"}
    reread = read_structure_string(
        write_structure_string(hip.structure, FileFormat.MMCIF), FileFormat.MMCIF
    )
    refused = _process(reread, hid)
    assert _outcome(refused).assessment.value == "conflict"
    assert _outcome(refused).reason is SiteProtonationReason.SOURCE_CONFLICT
    assert {"HD1", "HE2"} <= _atoms(refused.structure, HIS)


def test_canonical_reentry_with_no_pre_edit_evidence_does_not_mislabel_a_conflict():
    source = _his()
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=None),
        )
    )
    hip = _process(source, SiteProtonation(HIS, SIDE, site_charge=1))
    assert _outcome(hip).status.value == "applied"
    fresh = _process(
        hip.structure,
        SiteProtonation(HIS, SIDE, site_charge=0, parent_hydrogens={"ND1": 1}),
    )
    assert _outcome(fresh).assessment.value == "insufficient"
    assert _outcome(fresh).reason is SiteProtonationReason.INSUFFICIENT_PROVENANCE
    assert {"HD1", "HE2"} <= _atoms(fresh.structure, HIS)


def test_declared_sequence_and_dropped_partner_cannot_authorize_free_ends():
    atoms = [
        row
        for row in Path("tests/fixtures/pdb/span-reconstruction/1crn.pdb")
        .read_text()
        .splitlines()
        if row.startswith("ATOM") and row[17:20] == "ASP"
    ]
    source = read_structure_string(
        "\n".join(["SEQRES   1 A    2  GLY ASP", *atoms, "END"]), FileFormat.PDB
    )
    request = SiteProtonation(ASP, PolymerChemicalSite.BACKBONE_N, site_charge=1)
    assert _outcome(_process(source, request)).assessment.value == "insufficient"
    ligand = (
        atoms[2][:6].replace("ATOM  ", "HETATM")
        + f"{500:5d}"
        + atoms[2][11:17]
        + "LIG B   1"
        + atoms[2][26:]
    )
    dropped = read_structure_string(
        "\n".join([*atoms, ligand, "CONECT  294  500", "END"]), FileFormat.PDB
    )
    assert _outcome(_process(dropped, request)).assessment.value == "insufficient"


def test_node_budget_does_not_credit_an_unexecuted_site():
    result = process_structure(
        _asp(),
        transform_requests=WorkflowTransformRequests(
            site_protonations=(
                SiteProtonation(ASP, SIDE, site_charge=0, parent_hydrogens={"OD1": 1}),
            )
        ),
        planning_context=WorkflowPlanningContext(max_speculative_nodes=1),
    )
    assert _outcome(result).status.value == "search_limit"
    assert "HD1" not in _atoms(result.structure, ASP)


@pytest.mark.parametrize("mutation", (None, "D"))
def test_packing_reassesses_only_the_original_site_without_a_global_h_goal(mutation):
    try:
        faspr_executable_path()
    except FileNotFoundError:
        pytest.skip("packaged FASPR executable is unavailable")
    rows = (
        Path("tests/fixtures/pdb/refinement/1bkr_his42_local.pdb")
        .read_text()
        .splitlines()
    )
    atoms = [r for r in rows if r.startswith("ATOM") and 40 <= int(r[22:26]) <= 46]
    request = SiteProtonation(HIS, SIDE, site_charge=0, parent_hydrogens={"ND1": 1})
    source = _process(
        read_structure_string("\n".join([*atoms, "END"]), FileFormat.PDB), request
    ).structure
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            committed_sidechain_packing=PackingSpec(
                "faspr", mutable_residue_ids=(HIS,), target_sequence=mutation
            ),
            site_protonations=(request,),
        ),
    )
    if mutation is not None:
        residue = result.structure.constitution.residue_or_ligand(HIS)
        assert residue is not None and residue.component_id == "ASP"
        assert _outcome(result).reason is SiteProtonationReason.SUPERSEDED_IDENTITY
        return
    assert any(
        i.kind is ValidationIssueKind.PACKING_INVALIDATED_HYDROGENS
        for i in result.issues
    )
    assert _outcome(result).status.value == "applied"
    assert _atoms(result.structure, HIS).intersection({"HD1", "HE2"}) == {"HD1"}
    assert result.structure.geometry != source.geometry


@pytest.mark.parametrize("reverse", (False, True))
def test_one_failed_placement_does_not_claim_or_block_the_other_site(
    monkeypatch, reverse
):
    place = site_protonation.place_polymer_microstate_hydrogens

    def except_first(context, rid, site, **kwargs):
        if rid == HIS:
            raise GeometryPlacementError("unplaceable ring")
        return place(context, rid, site, **kwargs)

    monkeypatch.setattr(
        site_protonation, "place_polymer_microstate_hydrogens", except_first
    )
    other = ResidueId("A", 44)
    ids = (other, HIS) if reverse else (HIS, other)
    result = _process(
        _his(two=True), *(SiteProtonation(r, SIDE, site_charge=1) for r in ids)
    )
    assert result.site_protonation_report is not None
    rows = {
        row.request.residue_id: row for row in result.site_protonation_report.outcomes
    }
    assert tuple(rows) == ids
    assert rows[HIS].status.value == "placement_failed"
    assert "unplaceable ring" in rows[HIS].details
    assert not _atoms(result.structure, HIS).intersection({"HD1", "HE2"})
    assert rows[other].status.value == "applied"
    assert {"HD1", "HE2"} <= _atoms(result.structure, other)


@pytest.mark.parametrize("isotope", ("D", "T"))
def test_observed_acid_isotope_pins_the_parent_and_survives_a_refused_request(isotope):
    acid = _process(
        _asp(), SiteProtonation(ASP, SIDE, site_charge=0, parent_hydrogens={"OD2": 1})
    )
    records = write_structure_string(acid.structure, FileFormat.PDB).splitlines()
    name = isotope + "D2"
    records = [
        line[:12] + f"{name:>4}" + line[16:76] + f"{isotope:>2}" + line[78:]
        if line.startswith("ATOM") and line[12:16].strip() == "HD2"
        else line
        for line in records
    ]
    source = read_structure_string("\n".join(records), FileFormat.PDB)
    matched = _process(source, SiteProtonation(ASP, SIDE, site_charge=0))
    assert _outcome(matched).assessment.value == "resolved"
    assert dict(_outcome(matched).classes[0][1])["OD2"] == 1
    refused = _process(matched.structure, SiteProtonation(ASP, SIDE, site_charge=-1))
    assert _outcome(refused).assessment.value == "conflict"
    ref = AtomRef(ASP, name)
    before = source.constitution.atom_index(ref)
    after = refused.structure.constitution.atom_index(ref)
    assert refused.structure.constitution.atom_site_at(
        after
    ) == source.constitution.atom_site_at(before)
    assert refused.structure.geometry.atom_geometry(
        after
    ) == source.geometry.atom_geometry(before)
    assert refused.structure.topology.bond_between(
        after, refused.structure.constitution.atom_index(AtomRef(ASP, "OD2"))
    )


def test_contrary_terminal_sequence_withholds_oxt_even_during_global_h_completion():
    rows = (
        Path("tests/fixtures/pdb/span-reconstruction/1crn.pdb").read_text().splitlines()
    )
    atoms = [row for row in rows if row.startswith("ATOM") and row[17:20] == "ASP"]
    source = read_structure_string(
        "\n".join(["SEQRES   1 A    2  ASP GLY", *atoms, "END"]), FileFormat.PDB
    )
    result = process_structure(
        source,
        requested_goals=H_GOAL,
        transform_requests=WorkflowTransformRequests(
            site_protonations=(
                SiteProtonation(ASP, PolymerChemicalSite.BACKBONE_C, site_charge=-1),
            )
        ),
    )
    assert _outcome(result).assessment.value == "insufficient"
    assert "OXT" not in _atoms(result.structure, ASP)


@pytest.mark.parametrize(
    "sequence,expected", (("ASP GLY", "applied"), ("ASP ASP", "refused"))
)
def test_declared_sequence_must_force_the_requested_end_not_a_whole_alignment(
    sequence, expected
):
    rows = (
        Path("tests/fixtures/pdb/span-reconstruction/1crn.pdb").read_text().splitlines()
    )
    atoms = [row for row in rows if row.startswith("ATOM") and row[17:20] == "ASP"]
    source = read_structure_string(
        "\n".join([f"SEQRES   1 A    2  {sequence}", *atoms, "END"]), FileFormat.PDB
    )
    result = _process(
        source, SiteProtonation(ASP, PolymerChemicalSite.BACKBONE_N, site_charge=1)
    )
    assert _outcome(result).status.value == expected


def test_metal_bound_histidine_charge_is_physically_excluded_not_uncatalogued():
    rows = (
        Path("tests/fixtures/pdb/refinement/1jd0_gln92_local.pdb")
        .read_text()
        .splitlines()
    )
    atoms = [
        r
        for r in rows
        if r.startswith(("ATOM  ", "HETATM")) and int(r[22:26]) in (94, 901)
    ]
    link = (
        "LINK         NE2 HIS A  94                ZN    ZN A 901     1555   1555  2.06"
    )
    keep = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    source = read_structure_string(
        "\n".join([link, *atoms, "END"]),
        FileFormat.PDB,
        policy=keep.structure_normalization_policy(),
    )
    request = SiteProtonation(ResidueId("A", 94), SIDE, site_charge=1)
    result = process_structure(
        source,
        ingress=keep,
        transform_requests=WorkflowTransformRequests(site_protonations=(request,)),
    )
    row = _outcome(result)
    assert row.assessment.value == "conflict"
    assert row.reason is SiteProtonationReason.PHYSICAL_CONSTRAINT
    assert result.structure.constitution == source.constitution


def test_retain_a_nonfirst_resonance_drawing_without_a_chemical_repair():
    site = standard_microstate_candidates("ASP", SIDE)
    assert site is not None
    drawings = [g for g in site.candidates if g.protonation_key()[0] == -1]
    neutral_on_first = next(
        n for n in ("OD1", "OD2") if drawings[0].atom(n).charge == 0
    )
    source = _asp()
    records = write_structure_string(source, FileFormat.PDB).splitlines()
    pinned = read_structure_string(
        "\n".join(
            line[:78].ljust(78) + "1-"
            if line.startswith("ATOM") and line[12:16].strip() == neutral_on_first
            else line
            for line in records
        ),
        FileFormat.PDB,
    )
    request = SiteProtonation(ASP, SIDE, site_charge=-1)
    alternate = _process(pinned, request).structure.with_provenance(source.provenance)
    assert (
        alternate.topology.formal_charge(
            alternate.constitution.atom_index(AtomRef(ASP, neutral_on_first))
        )
        == -1
    )
    result = _process(alternate, request)
    assert _outcome(result).status.value == "already_satisfied"
    assert not result.repairs
    assert result.structure.topology == alternate.topology
    assert (
        result.structure.provenance.microstate_choices
        != alternate.provenance.microstate_choices
    )


def test_restoring_histidine_h_keeps_its_compatible_heavy_resonance_drawing():
    catalog = standard_microstate_candidates("HIS", SIDE)
    assert catalog is not None
    first = next(g for g in catalog.candidates if g.protonation_key()[0] == 1)
    charged_n = next(n for n in ("ND1", "NE2") if first.atom(n).charge == 0)
    source = _his()
    pinned = read_structure_string(
        "\n".join(
            line[:78].ljust(78) + "1+"
            if line.startswith("ATOM") and line[12:16].strip() == charged_n
            else line
            for line in write_structure_string(source, FileFormat.PDB).splitlines()
        ),
        FileFormat.PDB,
    )
    request = SiteProtonation(HIS, SIDE, site_charge=1)
    alternate = (
        _process(pinned, request)
        .structure.without_hydrogens_in_residues((HIS,))
        .with_provenance(source.provenance)
    )
    nitrogen = AtomRef(HIS, charged_n)
    assert (
        alternate.topology.formal_charge(alternate.constitution.atom_index(nitrogen))
        == 1
    )
    result = _process(alternate, request)
    assert _outcome(result).status.value == "applied"
    assert {"HD1", "HE2"} <= _atoms(result.structure, HIS)
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(nitrogen)
        )
        == 1
    )


@pytest.mark.parametrize("move_anchor", (False, True))
def test_restore_a_stripped_input_h_only_on_its_unchanged_anchor(move_anchor):
    request = SiteProtonation(ASP, SIDE, site_charge=0, parent_hydrogens={"OD2": 1})
    acid = _process(_asp(), request).structure
    source = read_structure_string(
        write_structure_string(acid, FileFormat.MMCIF), FileFormat.MMCIF
    )
    without_h = source.without_hydrogens_in_residues((ASP,))
    if move_anchor:
        geom = without_h.residue_geometry(without_h.constitution.residue_index(ASP))
        atom = geom.atom_geometry("OD2")
        moved = geom.with_atom_geometry(
            "OD2", atom.with_position(atom.position.with_offset(1, 0, 0))
        )
        without_h = without_h.with_updated_residue_geometries(((ASP, moved),))
    result = _process(without_h, request)
    h = AtomRef(ASP, "HD2")
    if move_anchor:
        assert _outcome(result).assessment.value == "insufficient"
        assert _outcome(result).reason is SiteProtonationReason.CHANGED_HYDROGEN_ANCHOR
        assert result.structure.constitution.resolve_atom_index(h) is None
    else:
        assert _outcome(result).status.value == "applied"
        assert result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(h)
        ) == source.geometry.atom_geometry(source.constitution.atom_index(h))


def test_retained_amino_acid_address_is_not_a_polymer_request():
    keep = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    his = write_structure_string(_his(), FileFormat.PDB).splitlines()
    asp = write_structure_string(_asp(), FileFormat.PDB).splitlines()
    atoms = [r for r in his if r.startswith("ATOM")] + [
        "HETATM" + f"{int(r[6:11]) + 100:5d}" + r[11:21] + "B" + r[22:]
        for r in asp
        if r.startswith("ATOM")
    ]
    source = read_structure_string(
        "\n".join([*atoms, "END"]),
        FileFormat.PDB,
        policy=keep.structure_normalization_policy(),
    )
    requests = (
        SiteProtonation(ResidueId("B", 43), SIDE, site_charge=-1),
        SiteProtonation(HIS, SIDE, site_charge=1),
    )
    result = process_structure(
        source,
        ingress=keep,
        transform_requests=WorkflowTransformRequests(site_protonations=requests),
    )
    assert result.site_protonation_report is not None
    refused, applied = result.site_protonation_report.outcomes
    assert refused.reason is SiteProtonationReason.NON_POLYMER
    assert applied.status.value == "applied"


def test_rejected_chemical_trial_is_not_credited_to_the_selected_branch(monkeypatch):
    score = runtime.evaluate_workflow_branch_quality_score

    def penalize_acid_h(result, **kwargs):
        actual = score(result, **kwargs)
        return replace(
            actual, protein_self_clash_count=int("HD2" in _atoms(result.structure, ASP))
        )

    monkeypatch.setattr(
        runtime, "evaluate_workflow_branch_quality_score", penalize_acid_h
    )
    request = SiteProtonation(ASP, SIDE, site_charge=0, parent_hydrogens={"OD2": 1})
    source = _asp()
    result = _process(source, request)
    assert _outcome(result).assessment.value == "resolved"
    assert _outcome(result).status.value == "not_retained"
    assert not result.repairs
    assert result.structure == source


def test_local_refinement_prerequisite_cannot_break_requested_tautomer_ambiguity():
    request = SiteProtonation(HIS, SIDE, site_charge=0)
    result = process_structure(
        _his(),
        transform_requests=WorkflowTransformRequests(
            site_protonations=(request,),
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((HIS,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            ),
        ),
    )
    assert _outcome(result).assessment.value == "ambiguous"
    names = _atoms(result.structure, HIS)
    assert not names.intersection({"HD1", "HE2"})
    assert "HA" in names


def test_label_sequence_position_blocks_a_false_n_terminus_without_a_full_sequence():
    document = gemmi.cif.read_string(write_structure_string(_asp(), FileFormat.MMCIF))
    block = document.sole_block()
    assert not tuple(block.find_values("_entity_poly_seq.mon_id"))
    labels = block.find_values("_atom_site.label_seq_id")
    for index in range(len(labels)):
        labels[index] = "4"
    structure = read_structure_string(document.as_string(), FileFormat.MMCIF)
    result = _process(
        structure, SiteProtonation(ASP, PolymerChemicalSite.BACKBONE_N, site_charge=1)
    )
    assert _outcome(result).reason is SiteProtonationReason.UNVERIFIED_BOUNDARY
    assert result.structure.constitution == structure.constitution


def test_rebinding_the_same_carrier_does_not_reuse_another_runs_selection():
    source = _his()
    library = build_default_component_library()
    ordinary = AminoAcidPreparation(PolymerMicrostateContext(source), library)

    def ring(preparation):
        return next(t for t in preparation.targets_for(HIS) if t.site.kind is SIDE)

    assert ring(ordinary).resolution.graph is not None
    request = SiteProtonation(HIS, SIDE, site_charge=0)
    with site_protonation_run(BoundSiteProtonations.bind(source, (request,))):
        selected = ordinary.for_inputs(source, library)
        assert ring(selected).resolution.status.value == "ambiguous"
        assert not ordinary.matches_chemistry(source)
    assert not selected.matches_chemistry(source)
    assert ring(selected.for_inputs(source, library)).resolution.graph is not None


def test_deferred_phase_report_observes_its_own_sites_after_return(monkeypatch):
    observe = reporting.evaluate_workflow_phase_outcomes
    seen = []

    def phase(structure, **kwargs):
        prepared = AminoAcidPreparation(
            PolymerMicrostateContext(structure), build_default_component_library()
        )
        ring = next(t for t in prepared.targets_for(HIS) if t.site.kind is SIDE)
        seen.append(ring.resolution.status.value)
        return observe(structure, **kwargs)

    monkeypatch.setattr(reporting, "evaluate_workflow_phase_outcomes", phase)
    result = _process(_his(), SiteProtonation(HIS, SIDE, site_charge=0))
    assert result.branch_report is not None
    assert result.branch_report.preferred_outcome().phase_report.outcomes
    assert seen[-1] == "ambiguous"
