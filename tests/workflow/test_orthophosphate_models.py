"""Request-selected chemistry for retained inorganic phosphate."""

from dataclasses import replace

import gemmi
import pytest
from tests.support.request_builders import whole_structure_requested_goals

from protrepair.api import process_structure
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.ingress_policy import (
    NonPolymerRetentionPolicy,
    RetentionAction,
    RetentionRule,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

_KEEP = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
_GOALS = whole_structure_requested_goals(
    HydrogenCoverageState.COMPLETE, include_default_heavy_completion=False
)
_PARENT = ResidueId("A", 1)


def _site(
    component: str,
    n: int = 1,
    *,
    observed: str | None = None,
    missing: str | None = None,
) -> str:
    atoms = [
        ("P", "P", 0, 0, 0),
        ("O3", "O", -0.9, 0.9, -0.9),
        ("O1", "O", 0.9, 0.9, 0.9),
        ("O4", "O", -0.9, -0.9, 0.9),
        ("O2", "O", 0.9, -0.9, -0.9),
    ]
    if observed:
        atoms.append((observed, observed[0], -1.87, -0.9, 0.9))
    return "".join(
        f"HETATM{n * 10 + i:5} {name:4} {component:>3} A{n:4}    "
        f"{x + n * 15:8.3f}{y:8.3f}{z:8.3f}{0.75:6.2f}{12.5:6.2f}          {element:>2}\n"
        for i, (name, element, x, y, z) in enumerate(atoms)
        if name != missing
    )


def _source(pdb: str) -> ProteinStructure:
    return read_structure_string(
        pdb, FileFormat.PDB, policy=_KEEP.structure_normalization_policy()
    )


def _parents(structure: ProteinStructure, rid: ResidueId) -> dict[str, str]:
    return dict(
        structure.topology.residue_hydrogen_anchor_by_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(rid),
        )
    )


def _charges(structure: ProteinStructure, rid: ResidueId) -> list[int | None]:
    return [
        structure.topology.formal_charge(
            structure.constitution.atom_index(AtomRef(rid, name))
        )
        for name in ("P", "O1", "O2", "O3", "O4")
    ]


def test_selected_model_resolves_each_label_without_moving_input_atoms():
    selected = replace(_KEEP, use_ccd_orthophosphate_models=True)
    original = _source(_site("PO4") + _site("PI", 2) + _site("2HP", 3))
    result = process_structure(original, ingress=selected, requested_goals=_GOALS)
    report = result.requested_goal_report
    assert report is not None and report.outcomes[0].is_satisfied()

    for n, oxygens, charges in (
        (1, [], [0, 0, -1, -1, -1]),
        (2, ["O4"], [0, 0, -1, -1, 0]),
        (3, ["O3", "O4"], [0, 0, -1, 0, 0]),
    ):
        rid = ResidueId("A", n)
        parents = _parents(result.structure, rid)
        assert sorted(parents.values()) == oxygens
        assert _charges(result.structure, rid) == charges
        for name in ("P", "O1", "O2", "O3", "O4"):
            ref = AtomRef(rid, name)
            for_structure = result.structure.constitution.atom_index(ref)
            assert result.structure.geometry.atom_geometry(
                for_structure
            ) == original.geometry.atom_geometry(original.constitution.atom_index(ref))
        added = [
            event
            for event in result.repairs
            if event.residue_id == rid and event.kind is RepairEventKind.HYDROGENS_ADDED
        ]
        assert bool(added) == bool(oxygens)
        for event in added:
            assert set(event.atom_names) == set(parents)
            assert event.details is not None and "Model-assigned" in event.details
            assert all(f"{h} on {o}" in event.details for h, o in parents.items())

    repeated = process_structure(
        result.structure, ingress=selected, requested_goals=_GOALS
    )
    assert not repeated.repairs


def test_an_h_goal_is_not_a_model_choice_and_choosing_a_model_is_not_an_h_goal():
    original = _source(_site("2HP"))
    without_model = process_structure(original, ingress=_KEEP, requested_goals=_GOALS)
    assert without_model.requested_goal_report is not None
    assert not without_model.requested_goal_report.outcomes[0].is_satisfied()
    assert _charges(without_model.structure, _PARENT) == [None] * 5
    assert not _parents(without_model.structure, _PARENT)
    no_goal = process_structure(
        original, ingress=replace(_KEEP, use_ccd_orthophosphate_models=True)
    )
    assert not no_goal.repairs
    assert no_goal.structure.topology == original.topology

    dropped = process_structure(
        original, ingress=StructureIngressOptions(use_ccd_orthophosphate_models=True)
    )
    assert not dropped.structure.constitution.ligands


@pytest.mark.parametrize("isotope", ["H", "D", "T"])
def test_original_isotope_and_its_scalars_survive_mixed_completion(isotope: str):
    name = isotope + "O4"
    original = _source(_site("2HP", observed=name))
    selected = replace(_KEEP, use_ccd_orthophosphate_models=True)
    result = process_structure(original, ingress=selected, requested_goals=_GOALS)
    parents = _parents(result.structure, _PARENT)
    assert parents[name] == "O4" and sorted(parents.values()) == ["O3", "O4"]
    ref = AtomRef(_PARENT, name)
    assert result.structure.constitution.resolve_atom_site(
        ref
    ) == original.constitution.resolve_atom_site(ref)
    assert result.structure.geometry.atom_geometry(
        result.structure.constitution.atom_index(ref)
    ) == original.geometry.atom_geometry(original.constitution.atom_index(ref))
    added = [
        event
        for event in result.repairs
        if event.kind is RepairEventKind.HYDROGENS_ADDED
    ]
    assert len(added) == 1 and set(added[0].atom_names) == set(parents) - {name}


def test_per_site_override_wins_without_changing_other_sites_or_relabelling():
    override = RetainedNonPolymerChemistryOverride(
        _PARENT, "P(=O)([O-])([O-])O", ("P", "O1", "O2", "O3", "O4")
    )
    selected = replace(
        _KEEP,
        use_ccd_orthophosphate_models=True,
        retained_non_polymer_chemistry_overrides=(override,),
    )
    original = _source(_site("PO4") + _site("PO4", 2))
    result = process_structure(original, ingress=selected, requested_goals=_GOALS)
    assert result.structure.constitution.ligands[0].component_id == "PO4"
    assert list(_parents(result.structure, _PARENT).values()) == ["O4"]
    assert _charges(result.structure, ResidueId("A", 2)) == [0, 0, -1, -1, -1]


@pytest.mark.parametrize(
    "bad",
    [
        _site("PO4", observed="DO4"),
        _site("2HP", missing="O3"),
        _site("PI").replace(" O4    PI", " OX    PI"),
        _site("PI").replace("           O\n", "           S\n", 1),
        _site("PI", observed="HO2"),
        _site("PI").replace("           P\n", "           P1+\n", 1),
        _site("PI") + "CONECT   10   13   13\n",
        _site("PI")
        + "HETATM   33 MG    MG A   3      12.600  -0.900   0.900  1.00 12.50          MG\n"
        + "CONECT   13   33\n",
    ],
    ids=(
        "excess-D",
        "missing-O",
        "misnamed-O",
        "wrong-element",
        "wrong-H-parent",
        "charge",
        "wrong-order",
        "external-bond",
    ),
)
def test_rejected_model_leaves_one_site_unchanged_but_prepares_its_sibling(bad: str):
    original = _source(bad + _site("2HP", 2))
    result = process_structure(
        original,
        ingress=replace(_KEEP, use_ccd_orthophosphate_models=True),
        requested_goals=_GOALS,
    )
    assert result.requested_goal_report is not None
    assert not result.requested_goal_report.outcomes[0].is_satisfied()
    assert not any(event.residue_id == _PARENT for event in result.repairs)
    assert any(issue.residue_id == _PARENT for issue in result.issues)
    assert sorted(_parents(result.structure, ResidueId("A", 2)).values()) == [
        "O3",
        "O4",
    ]


@pytest.mark.parametrize("fmt", (FileFormat.PDB, FileFormat.MMCIF))
def test_model_generated_hydrogens_survive_export_and_reread(fmt: FileFormat):
    opts = replace(_KEEP, use_ccd_orthophosphate_models=True)
    first = process_structure(
        _source(_site("2HP")), ingress=opts, requested_goals=_GOALS
    )
    reread = read_structure_string(
        write_structure_string(first.structure, file_format=fmt),
        fmt,
        policy=opts.structure_normalization_policy(),
    )
    second = process_structure(reread, ingress=opts, requested_goals=_GOALS)
    assert sorted(_parents(second.structure, _PARENT).values()) == ["O3", "O4"]
    assert not any(
        event.kind is RepairEventKind.HYDROGENS_ADDED for event in second.repairs
    )


def test_removed_source_deuterium_is_not_replaced_by_a_generated_hydrogen():
    ref = AtomRef(_PARENT, "DO4")
    original = _source(_site("PI", observed=ref.atom_name))
    stripped = original.without_atom_refs((ref,))
    result = process_structure(
        stripped,
        ingress=replace(_KEEP, use_ccd_orthophosphate_models=True),
        requested_goals=_GOALS,
    )
    assert (
        result.requested_goal_report is not None
        and not result.requested_goal_report.outcomes[0].is_satisfied()
    )
    assert not result.repairs and not _parents(result.structure, _PARENT)
    assert any("source H/D/T" in issue.message for issue in result.issues)


def test_detached_observed_deuterium_is_not_moved_to_fit_the_model():
    original = _source(_site("PI", observed="DO4").replace("  13.130", "  23.130"))
    result = process_structure(
        original,
        ingress=replace(_KEEP, use_ccd_orthophosphate_models=True),
        requested_goals=_GOALS,
    )
    assert result.structure.geometry == original.geometry
    assert result.structure.topology == original.topology
    assert not result.repairs
    assert any(
        issue.residue_id == _PARENT and "reconciliation" in issue.message
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "smiles", ["P(=O)([O-])([O-])[O-]", "P(=O)([O-])(O)[O-]"], ids=("no-H", "H-on-O3")
)
def test_conflicting_site_override_does_not_fall_back_to_the_deposited_model(
    smiles: str,
):
    ref = AtomRef(_PARENT, "DO4")
    original = _source(_site("PI", observed=ref.atom_name))
    override = RetainedNonPolymerChemistryOverride(
        _PARENT, smiles, ("P", "O1", "O2", "O3", "O4")
    )
    options = replace(
        _KEEP,
        use_ccd_orthophosphate_models=True,
        retained_non_polymer_chemistry_overrides=(override,),
    )
    result = process_structure(original, ingress=options, requested_goals=_GOALS)
    assert (
        result.requested_goal_report is not None
        and not result.requested_goal_report.outcomes[0].is_satisfied()
    )
    assert result.structure.constitution.resolve_atom_site(
        ref
    ) == original.constitution.resolve_atom_site(ref)
    assert _charges(result.structure, _PARENT) == [None] * 5 and not result.repairs


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize(
    "projection", ["raw-chain", "canonical-filter", "alternate-location"]
)
@pytest.mark.parametrize(
    "relationship", [gemmi.ConnectionType.MetalC, gemmi.ConnectionType.Covale]
)
def test_omitting_a_linked_partner_does_not_make_phosphate_isolated(
    override: bool, projection: str, relationship: gemmi.ConnectionType
):
    raw = gemmi.read_pdb_string(
        _site("PI")
        + "HETATM   38 MG    MG B  30      12.100  -0.900   0.900  1.00 12.50          MG\n"
    )
    altloc = "\0"
    if projection == "alternate-location":
        mg = raw[0]["B"][0]
        mg[0].altloc = "A"
        mg[0].occ = 0.75
        discarded = mg[0].clone()
        discarded.altloc = altloc = "B"
        discarded.occ = 0.25
        mg.add_atom(discarded)
    link = gemmi.Connection()
    link.name = "declared"
    link.type = relationship
    link.partner1 = gemmi.AtomAddress("A", gemmi.SeqId(1, " "), "PI", "O4", "\0")
    link.partner2 = gemmi.AtomAddress("B", gemmi.SeqId(30, " "), "MG", "MG", altloc)
    raw.connections.append(link)
    text = raw.make_mmcif_document().as_string()
    policy = _KEEP.structure_normalization_policy()
    if projection == "raw-chain":
        policy = replace(policy, selected_chain_ids=("A",))
    source = read_structure_string(text, FileFormat.MMCIF, policy=policy)
    assert bool(source.topology.bonds) == (projection == "canonical-filter")
    if projection == "alternate-location":
        mg_index = source.constitution.atom_index(AtomRef(ResidueId("B", 30), "MG"))
        assert source.geometry.atom_geometry(mg_index).altloc == "A"
    options = replace(
        _KEEP,
        use_ccd_orthophosphate_models=True,
        retained_non_polymer_chemistry_overrides=(
            RetainedNonPolymerChemistryOverride(
                _PARENT, "P(=O)([O-])([O-])O", ("P", "O1", "O2", "O3", "O4")
            ),
        )
        if override
        else (),
    )
    if projection == "canonical-filter":
        options = replace(
            options,
            ligand_policy=LigandPolicy.DROP,
            non_polymer_retention=NonPolymerRetentionPolicy(
                water_default=RetentionAction.DROP,
                other_default=RetentionAction.DROP,
                rules=(RetentionRule(RetentionAction.KEEP, component_ids=("PI",)),),
            ),
        )
    result = process_structure(source, ingress=options, requested_goals=_GOALS)
    assert (
        result.requested_goal_report is not None
        and not result.requested_goal_report.outcomes[0].is_satisfied()
    )
    assert not any(event.residue_id == _PARENT for event in result.repairs)
    assert _charges(result.structure, _PARENT) == [None] * 5
    assert any(
        "partner" in issue.message and issue.residue_id == _PARENT
        for issue in result.issues
    )
