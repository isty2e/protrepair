from collections import Counter
from dataclasses import replace
from math import cos, pi, sin
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from tests.support.request_builders import whole_structure_requested_goals

from protrepair import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.inference.smiles_correspondence import (
    CorrespondenceLimits,
    bind_smiles_assignment,
)
from protrepair.chemistry.retained_non_polymer.definition import MatchedSmiles
from protrepair.chemistry.retained_non_polymer.evidence import (
    CorrespondenceFailure,
    RefusedSmiles,
)
from protrepair.io import (
    read_retained_component_definitions,
    read_structure,
    read_structure_string,
    write_structure_string,
)
from protrepair.sources.chemistry import (
    RetainedNonPolymerChemistryOverride,
    SmilesAssignment,
)
from protrepair.state import HydrogenCoverageState
from protrepair.structure import ResidueId
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportMode,
    resolve_local_bond_planning_support,
)
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

DATA = Path(__file__).resolve().parents[1] / "fixtures/chemistry/named-ccd"
SITE = ResidueId("A", 999)
ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC"
GOALS = whole_structure_requested_goals(
    HydrogenCoverageState.COMPLETE, include_default_heavy_completion=False
)


def test_smiles_alone_recovers_deposited_aq4_chemistry():
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[SmilesAssignment(SITE, ERLOTINIB)],
    )
    source = read_structure(
        DATA / "1m17-aq4.pdb", policy=options.structure_normalization_policy()
    )
    assert not process_structure(source, ingress=options).repairs
    result = process_structure(source, ingress=options, requested_goals=GOALS)
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied()
    prepared = result.structure
    parents = prepared.topology.residue_hydrogen_anchor_by_name(
        constitution=prepared.constitution,
        residue_index=prepared.constitution.residue_index(SITE),
    )
    reference = read_retained_component_definitions(DATA / "ligands.cif")["AQ4"]
    assert len(parents) == 23
    assert Counter(parents.values()) == {
        p: len(h) for p, h in reference.hydrogen_slots().items()
    }
    assert not process_structure(
        prepared, ingress=options, requested_goals=GOALS
    ).repairs
    original = {
        n: source.residue_geometry(
            source.constitution.residue_index(SITE)
        ).atom_geometry(n)
        for n in reference.heavy_atom_names
    }
    pose = prepared.residue_geometry(prepared.constitution.residue_index(SITE))
    assert all(pose.atom_geometry(n) == g for n, g in original.items())
    graph = bind_smiles_assignment(source, SITE, ERLOTINIB)
    assert isinstance(graph, MatchedSmiles)
    assert {
        tuple(sorted((b.first, b.second))): b.order if not b.aromatic else "AROM"
        for b in graph.definition.bonds
        if b.first in original and b.second in original
    } == {
        tuple(sorted((b.first, b.second))): b.order if not b.aromatic else "AROM"
        for b in reference.bonds
        if b.first in original and b.second in original
    }
    for fmt in (FileFormat.PDB, FileFormat.MMCIF):
        reread = read_structure_string(
            write_structure_string(prepared, fmt),
            fmt,
            policy=options.structure_normalization_policy(),
        )
        again = process_structure(reread, ingress=options, requested_goals=GOALS)
        assert (
            again.requested_goal_report is not None
            and again.requested_goal_report.outcomes[0].is_satisfied()
        )
        assert (
            again.structure.topology.residue_hydrogen_anchor_by_name(
                constitution=again.structure.constitution,
                residue_index=again.structure.constitution.residue_index(SITE),
            )
            == parents
        )


SULFATE = (
    ("S", "S", 0.0, 0.0, 0.0),
    ("O1", "O", 0.0, -1.2, -0.85),
    ("O2", "O", 0.0, 1.2, -0.85),
    ("O3", "O", -1.2, 0.0, 0.85),
    ("O4", "O", 1.2, 0.0, 0.85),
)
ACID = (
    ("C1", "C", 0.0, 0.0, 0.0),
    ("O1", "O", -1.25, 0.0, 0.0),
    ("O2", "O", 0.64, 1.08, 0.0),
    ("C2", "C", 0.76, -1.32, 0.0),
)
SULFATE_SMILES = "[O-]S(=O)(=O)[O-]"


def _text(atoms, component="LIG", charges=()):
    q = dict(charges)
    return "".join(
        f"HETATM{i:5d} {n:>4} {component:>3} A 999    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {e:>2}{q.get(n, ''):>2}\n"
        for i, (n, e, x, y, z) in enumerate(atoms, 1)
    )


def _source(text):
    return read_structure_string(
        text,
        FileFormat.PDB,
        policy=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )


def _options(smiles):
    return StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[SmilesAssignment(SITE, smiles)],
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("component", ["LIG", "SO4"])
def test_whole_sulfate_localization_is_stable_and_respects_source_charge(
    reverse, pinned, component
):
    atoms = SULFATE[::-1] if reverse else SULFATE
    source = _source(_text(atoms, component, (("O1", "1-"),) if pinned else ()))
    options = _options("O=S([O-])([O-])=O" if reverse else SULFATE_SMILES)
    out = process_structure(source, ingress=options, requested_goals=GOALS)
    c = out.structure.constitution
    observed = {
        n: out.structure.topology.formal_charge(c.atom_index(AtomRef(SITE, n)))
        for n, _, *p in atoms
    }
    assert observed == (
        {"S": 0, "O1": -1, "O2": 0, "O3": 0, "O4": -1}
        if pinned
        else {"S": 0, "O1": 0, "O2": 0, "O3": -1, "O4": -1}
    )
    for n in ("O1", "O2", "O3", "O4"):
        bond = out.structure.topology.bond_between(
            c.atom_index(AtomRef(SITE, "S")), c.atom_index(AtomRef(SITE, n))
        )
        assert bond is not None and bond.order == (1 if observed[n] == -1 else 2)
    assert any("model-assigned" in (event.details or "") for event in out.repairs)
    assert not process_structure(
        out.structure, ingress=options, requested_goals=GOALS
    ).repairs


def test_refused_assignment_is_sticky_for_a_bundled_ligand_and_passive_context():
    source = _source(_text(SULFATE, "SO4", (("O1", "0+"), ("O2", "0+"), ("O3", "0+"))))
    bound = bind_smiles_assignment(source, SITE, SULFATE_SMILES)
    assert (
        isinstance(bound, RefusedSmiles)
        and bound.cause is CorrespondenceFailure.INCOMPATIBLE
    )
    assert bound.requires_hydrogens is False
    out = process_structure(
        source, ingress=_options(SULFATE_SMILES), requested_goals=GOALS
    )
    assert not out.repairs and out.structure.topology == source.topology
    assert any(
        i.residue_id == SITE and "caller SMILES" in i.message for i in out.issues
    )
    idx = source.constitution.residue_index(SITE)
    support = resolve_local_bond_planning_support(
        ProteinStructureSnapshot.from_structure(source),
        idx,
        source.constitution.residue_site_at(idx),
        movable_atom_indices=(),
        component_library=build_default_component_library(),
        retained_non_polymer_chemistry_evidence=(bound,),
    )
    assert (
        support.mode is LocalBondPlanningSupportMode.UNSUPPORTED
        and "caller SMILES" in support.blocker_message
    )


def test_passive_context_waits_for_the_selected_bond_orders_even_if_charges_match():
    source = _source(_text(SULFATE, "SO4", (("O1", "1-"), ("O4", "1-"))))
    bound = bind_smiles_assignment(source, SITE, SULFATE_SMILES)
    assert isinstance(bound, MatchedSmiles)
    prepared = process_structure(
        source, ingress=_options(SULFATE_SMILES), requested_goals=GOALS
    )
    for structure, expected in (
        (source, LocalBondPlanningSupportMode.UNSUPPORTED),
        (prepared.structure, LocalBondPlanningSupportMode.NAMED_GRAPH_PASSIVE_CONTEXT),
    ):
        idx = structure.constitution.residue_index(SITE)
        support = resolve_local_bond_planning_support(
            ProteinStructureSnapshot.from_structure(structure),
            idx,
            structure.constitution.residue_site_at(idx),
            movable_atom_indices=(),
            component_library=build_default_component_library(),
            retained_non_polymer_chemistry_evidence=(bound,),
        )
        assert support.mode is expected


@pytest.mark.parametrize(
    "declaration", ["", "CONECT    2    3\n", "CONECT    2    3    3\n"]
)
def test_selected_alcohol_can_replace_only_an_unreported_carbonyl_order(declaration):
    atoms = (
        ("CH3", "C", -1.50, 0.0, 0.0),
        ("C", "C", 0.0, 0.0, 0.0),
        ("O", "O", 0.65, 1.08, 0.0),
    )
    source = _source(_text(atoms, "ACE") + declaration)
    before = source.topology.bond_between(
        *(source.constitution.atom_index(AtomRef(SITE, n)) for n in ("C", "O"))
    )
    assert before is not None and before.order == 2
    out = process_structure(source, ingress=_options("CCO"), requested_goals=GOALS)
    if (
        before.source_metadata is not None
        and before.source_metadata.reported_order == 2
    ):
        assert out.structure.topology == source.topology and not out.repairs
        assert any(
            i.residue_id == SITE and "caller SMILES" in i.message for i in out.issues
        )
        return
    c = out.structure.constitution
    after = out.structure.topology.bond_between(
        *(c.atom_index(AtomRef(SITE, n)) for n in ("C", "O"))
    )
    assert after is not None and after.order == 1
    assert after.source_metadata == before.source_metadata
    parents = dict(
        out.structure.topology.residue_hydrogen_anchor_by_name(
            constitution=c, residue_index=c.residue_index(SITE)
        )
    )
    assert Counter(parents.values()) == {"CH3": 3, "C": 2, "O": 1}


@pytest.mark.parametrize(
    "link,admitted",
    [
        ("", False),
        ("CONECT    1    2\n", False),
        ("CONECT    1    2    2\n", True),
        ("CONECT    3    5\n", True),
    ],
)
def test_only_reported_order_or_hydrogen_parent_can_resolve_an_acid(link, admitted):
    atoms = (*ACID, ("DOBS", "D", 1.12, 1.90, 0.0))
    s = _source(_text(atoms) + link)
    selection = bind_smiles_assignment(s, SITE, "CC(=O)O")
    if not admitted:
        assert (
            isinstance(selection, RefusedSmiles)
            and selection.cause is CorrespondenceFailure.AMBIGUOUS
        )
        return
    assert isinstance(selection, MatchedSmiles)
    out = process_structure(s, ingress=_options("CC(=O)O"), requested_goals=GOALS)
    assert (
        out.requested_goal_report is not None
        and out.requested_goal_report.outcomes[0].is_satisfied()
    )
    idx = out.structure.constitution.atom_index(AtomRef(SITE, "DOBS"))
    assert out.structure.constitution.atom_site_at(idx).element == "D"
    assert out.structure.geometry.atom_geometry(idx) == s.geometry.atom_geometry(
        s.constitution.atom_index(AtomRef(SITE, "DOBS"))
    )
    assert (
        dict(
            out.structure.topology.residue_hydrogen_anchor_by_name(
                constitution=out.structure.constitution,
                residue_index=out.structure.constitution.residue_index(SITE),
            )
        )["DOBS"]
        == "O2"
    )


def test_a_successful_prefix_and_a_no_match_prefix_do_not_prove_results():
    s = _source(_text(ACID))
    for smiles, limits in (
        ("CC(=O)O", CorrespondenceLimits(max_candidates=1)),
        ("CC(=O)O", CorrespondenceLimits(max_extensions=1)),
        ("CC(=O)O", CorrespondenceLimits(max_bytes=2)),
    ):
        result = bind_smiles_assignment(s, SITE, smiles, limits=limits)
        assert (
            isinstance(result, RefusedSmiles)
            and result.cause is CorrespondenceFailure.SEARCH_LIMIT
        )


def test_original_hydrogen_cannot_change_conformer_on_the_current_carrier():
    source = _source(
        _text((*ACID, ("DOBS", "D", 1.12, 1.90, 0.0))) + "CONECT    3    5\n"
    )
    pose = source.residue_geometry(source.constitution.residue_index(SITE))
    changed = source.with_updated_residue_geometries(
        (
            (
                SITE,
                pose.with_atom_geometry(
                    "DOBS", replace(pose.atom_geometry("DOBS"), altloc="B")
                ),
            ),
        )
    )
    result = bind_smiles_assignment(changed, SITE, "CC(=O)O")
    assert isinstance(result, RefusedSmiles) and "conformer" in result.detail


def test_canonical_input_without_original_is_identified_as_the_supplied_model():
    original = _source(_text(ACID) + "CONECT    1    2    2\n")
    manual = original.with_provenance(
        replace(
            original.provenance,
            ingress=replace(original.provenance.ingress, observation=None),
        )
    )
    result = process_structure(
        manual, ingress=_options("CC(=O)O"), requested_goals=GOALS
    )
    assert any(
        "supplied canonical input" in (event.details or "") for event in result.repairs
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "[H]OC",
        "[CH3:0]O",
        "[13CH3]O",
        "[S@](=O)(C)CC",
        "C[C@H](C)O",
        "C/C=C(/C)C",
        "C/C(O)CC",
        "[CH3]",
        "C.C",
        "CO |$foo;bar$|",
    ],
)
def test_unsupported_query_feature_is_not_silently_removed(smiles):
    s = _source(_text(ACID))
    result = bind_smiles_assignment(s, SITE, smiles)
    assert (
        isinstance(result, RefusedSmiles)
        and result.cause is CorrespondenceFailure.UNSUPPORTED
    )


def test_malformed_smiles_and_dropped_ligand_are_request_errors():
    source = _source(_text(ACID))
    with pytest.raises(ValueError, match="invalid SMILES"):
        process_structure(source, ingress=_options("C1(=O"))
    with pytest.raises(ValueError, match="invalid SMILES"):
        process_structure(source, ingress=_options("C1(=O |$a$|"))
    with pytest.raises(ValueError, match="kept ligand"):
        process_structure(
            source,
            ingress=StructureIngressOptions(
                retained_non_polymer_chemistry_overrides=[SmilesAssignment(SITE, "CCO")]
            ),
        )


def test_good_site_proceeds_beside_an_ambiguous_one():
    aq = (DATA / "1m17-aq4.pdb").read_text().split("END")[0].replace("A 999", "B 999")
    source = _source(_text(ACID) + aq)
    other = ResidueId("B", 999)
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[
            SmilesAssignment(SITE, "CC(=O)O"),
            SmilesAssignment(other, ERLOTINIB),
        ],
    )
    out = process_structure(source, ingress=options, requested_goals=GOALS)
    assert (
        sum(
            a.is_hydrogen()
            for a in out.structure.constitution.residue_site_at(
                out.structure.constitution.residue_index(other)
            ).atom_sites
        )
        == 23
    )
    assert any(i.residue_id == SITE and "ambiguous" in i.message for i in out.issues)
    assert out.structure.residue_geometry(
        out.structure.constitution.residue_index(SITE)
    ) == source.residue_geometry(source.constitution.residue_index(SITE))


def test_generated_orders_or_hydrogens_cannot_disambiguate_original_acid():
    source = _source(_text(ACID))
    named = RetainedNonPolymerChemistryOverride(
        SITE, "CC(=O)O", ("C2", "C1", "O1", "O2")
    )
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[named],
    )
    prepared = process_structure(
        source, ingress=options, requested_goals=GOALS
    ).structure
    assert any(
        a.is_hydrogen()
        for a in prepared.constitution.residue_site_at(
            prepared.constitution.residue_index(SITE)
        ).atom_sites
    )
    result = bind_smiles_assignment(prepared, SITE, "CC(=O)O")
    assert (
        isinstance(result, RefusedSmiles)
        and result.cause is CorrespondenceFailure.AMBIGUOUS
    )
    manual = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=None),
        )
    )
    result = bind_smiles_assignment(manual, SITE, "CC(=O)O")
    assert (
        isinstance(result, RefusedSmiles)
        and result.cause is CorrespondenceFailure.AMBIGUOUS
    )


def test_glycerol_symmetry_preserves_generated_h_names_across_row_and_query_order():
    rows = (DATA.parents[1] / "corpus/pdb1jyd.ent").read_text().splitlines()
    records = [
        line
        for line in rows
        if line.startswith("HETATM") and line[17:20] == "GOL" and line[21:26] == "A 600"
    ]
    assert len(records) == 6
    reference = None
    rid = ResidueId("A", 600)
    for ordered, smiles in ((records, "OCC(O)CO"), (records[::-1], "C(C(CO)O)O")):
        source = _source("\n".join(ordered) + "\n")
        options = StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=[SmilesAssignment(rid, smiles)],
        )
        out = process_structure(source, ingress=options, requested_goals=GOALS)
        parents = dict(
            out.structure.topology.residue_hydrogen_anchor_by_name(
                constitution=out.structure.constitution,
                residue_index=out.structure.constitution.residue_index(rid),
            )
        )
        assert len(parents) == 8
        if reference is not None:
            assert parents == reference
        reference = parents


def test_exhaustive_zero_solution_is_different_from_cutoff():
    # Two separated required rings; no one-ring embedding can cover both.
    atoms = [
        (
            f"C{r * 8 + i}",
            "C",
            12 * r + 1.92 * cos(i * pi / 4),
            1.92 * sin(i * pi / 4),
            0.0,
        )
        for r in range(2)
        for i in range(8)
    ]
    links = "".join(
        f"CONECT{r * 8 + i + 1:5}{r * 8 + (i + 1) % 8 + 1:5}\n"
        for r in range(2)
        for i in range(8)
    )
    s = _source(_text(atoms) + links)
    query = "C1" + "C" * 15 + "1"
    complete = bind_smiles_assignment(s, SITE, query)
    cut = bind_smiles_assignment(
        s, SITE, query, limits=CorrespondenceLimits(max_extensions=250)
    )
    assert (
        isinstance(complete, RefusedSmiles)
        and complete.cause is CorrespondenceFailure.INCOMPATIBLE
    )
    assert (
        isinstance(cut, RefusedSmiles)
        and cut.cause is CorrespondenceFailure.SEARCH_LIMIT
    )


@pytest.mark.parametrize(
    ("model", "opposite"),
    [("F[C@H](Cl)Br", "F[C@@H](Cl)Br"), ("F/C=C/Cl", "F/C=C\\Cl")],
)
def test_mapped_stereo_respects_measured_pose(model, opposite):
    mol = Chem.AddHs(Chem.MolFromSmiles(model))
    assert EmbedMolecule(mol, randomSeed=13) == 0
    sites = [
        (f"A{i}", a.GetSymbol(), *tuple(mol.GetConformer().GetAtomPosition(i)))
        for i, a in enumerate(mol.GetAtoms())
        if a.GetAtomicNum() != 1
    ]
    source = _source(_text(sites))
    assert isinstance(bind_smiles_assignment(source, SITE, model), MatchedSmiles)
    failed = bind_smiles_assignment(source, SITE, opposite)
    assert (
        isinstance(failed, RefusedSmiles)
        and failed.cause is CorrespondenceFailure.INCOMPATIBLE
    )
