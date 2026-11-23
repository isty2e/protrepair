"""Behavior of named component definitions on normalized retained sites."""

from dataclasses import replace
from math import cos, pi, sin
from pathlib import Path

import gemmi
import pytest
from tests.support.request_builders import whole_structure_requested_goals

from protrepair import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.io import (
    read_retained_component_definitions,
    read_structure,
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure import ResidueId
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportMode,
    resolve_local_bond_planning_support,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    WorkflowTransformRequests,
)

DATA = Path(__file__).resolve().parents[1] / "fixtures/chemistry/named-ccd"
SITE = ResidueId("A", 999)
KEEP = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
GOALS = whole_structure_requested_goals(
    HydrogenCoverageState.COMPLETE, include_default_heavy_completion=False
)


def _source(text):
    return read_structure_string(
        text, FileFormat.PDB, policy=KEEP.structure_normalization_policy()
    )


def _names(structure, rid):
    site = structure.constitution.residue_or_ligand(rid)
    assert site is not None
    return {a.name for a in site.atom_sites if a.is_hydrogen()}


def _protons(structure, rid):
    return dict(
        structure.topology.residue_hydrogen_anchor_by_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(rid),
        )
    )


def _coord(structure, rid, name):
    return structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(rid, name))
    )


def test_aq4_retains_29_heavy_atoms_and_adds_23_named_hydrogens():
    model = read_retained_component_definitions(DATA / "ligands.cif")["AQ4"]
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[model.for_residue(SITE)],
    )
    source = read_structure(
        DATA / "1m17-aq4.pdb", policy=KEEP.structure_normalization_policy()
    )
    quiet = process_structure(source, ingress=options)
    assert not quiet.repairs and quiet.structure.topology == source.topology
    result = process_structure(
        DATA / "1m17-aq4.pdb", ingress=options, requested_goals=GOALS
    )
    assert (
        result.requested_goal_report is not None
        and result.requested_goal_report.outcomes[0].is_satisfied()
    )
    assert len(model.heavy_atom_names) == 29
    for name in model.heavy_atom_names:
        assert _coord(result.structure, SITE, name) == _coord(source, SITE, name)
    expected = {h: p for p, group in model.hydrogen_slots().items() for h in group}
    assert len(expected) == 23 and _protons(result.structure, SITE) == expected
    events = [e for e in result.repairs if e.kind is RepairEventKind.HYDROGENS_ADDED]
    assert len(events) == 1 and "caller CCD AQ4" in (events[0].details or "")
    native = to_rdkit_molecule(result.structure)
    assert (native.GetNumAtoms(), native.GetNumBonds()) == (52, 54)
    assert sum(a.GetNumRadicalElectrons() for a in native.GetAtoms()) == 0
    assert sum(a.GetFormalCharge() for a in native.GetAtoms()) == 0

    assert not process_structure(
        result.structure, ingress=options, requested_goals=GOALS
    ).repairs
    for fmt in (FileFormat.PDB, FileFormat.MMCIF):
        reread = read_structure_string(
            write_structure_string(result.structure, fmt),
            fmt,
            policy=KEEP.structure_normalization_policy(),
        )
        again = process_structure(reread, ingress=options, requested_goals=GOALS)
        assert _protons(again.structure, SITE) == expected
        assert not any(e.kind is RepairEventKind.HYDROGENS_ADDED for e in again.repairs)


def _at(rid, offset=0.0, missing=None):
    records = []
    for line in (DATA / "1m17-aq4.pdb").read_text().splitlines():
        if not line.startswith("HETATM") or line[12:16].strip() == missing:
            continue
        records.append(
            line[:21]
            + f"{rid.chain_id}{rid.seq_num:4}{rid.insertion_code or ' '}"
            + line[27:30]
            + f"{float(line[30:38]) + offset:8.3f}"
            + line[38:]
        )
    return "\n".join(records) + "\n"


def test_assignment_is_per_exact_address_and_one_bad_copy_does_not_block_another():
    # These are distinct addresses even though their display tokens coincide.
    other = ResidueId("A", 99, "9")
    assert SITE.display_token() == other.display_token()
    model = read_retained_component_definitions(DATA / "ligands.cif")["AQ4"]
    bad_name = model.heavy_atom_names[0]
    s = _source(_at(SITE) + _at(other, 25.0, missing=bad_name))
    for ids in ((SITE, other), (other, SITE)):
        options = StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=(
                model.for_residue(r) for r in ids
            ),
        )
        result = process_structure(s, ingress=options, requested_goals=GOALS)
        assert len(_names(result.structure, SITE)) == 23 and not _names(
            result.structure, other
        )
        assert any(
            i.residue_id == other
            and "caller CCD AQ4" in i.message
            and "heavy atoms" in i.message
            for i in result.issues
        )
    control = process_structure(s, ingress=KEEP, requested_goals=GOALS)
    one = process_structure(
        s,
        ingress=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=[model.for_residue(SITE)],
        ),
        requested_goals=GOALS,
    )
    assert _protons(one.structure, other) == _protons(control.structure, other)


def test_address_errors_are_not_site_refusals_and_smiles_validation_stays_strict():
    defs = read_retained_component_definitions(DATA / "ligands.cif")
    original = _source(_at(SITE))
    for override, keep, pattern in (
        (defs["LAC"].for_residue(SITE), KEEP, "expected kept ligand LAC, found AQ4"),
        (
            defs["AQ4"].for_residue(SITE),
            StructureIngressOptions(),
            "expected kept ligand AQ4",
        ),
        (defs["AQ4"].for_residue(ResidueId("B", 999)), KEEP, "absent"),
    ):
        with pytest.raises(ValueError, match=pattern):
            process_structure(
                original,
                ingress=StructureIngressOptions(
                    ligand_policy=keep.ligand_policy,
                    retained_non_polymer_chemistry_overrides=[override],
                ),
            )
    other = ResidueId("B", 999)
    with pytest.raises(ValueError, match="heavy_atom_names"):
        process_structure(
            _source(_at(SITE) + _at(other, 25.0)),
            ingress=StructureIngressOptions(
                ligand_policy=LigandPolicy.KEEP,
                retained_non_polymer_chemistry_overrides=[
                    defs["AQ4"].for_residue(SITE),
                    RetainedNonPolymerChemistryOverride(other, "CC", ("A", "B")),
                ],
            ),
        )


def test_named_isotope_consumes_its_own_dictionary_slot():
    model = read_retained_component_definitions(DATA / "ligands.cif")["AQ4"]
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[model.for_residue(SITE)],
    )
    original = _source(_at(SITE))
    prepared = process_structure(
        original, ingress=options, requested_goals=GOALS
    ).structure
    proton = next(
        h for group in model.hydrogen_slots().values() if len(group) > 1 for h in group
    )
    isotope = "D" + proton[1:]
    partner = _protons(prepared, SITE)[proton]
    alternate = next(h for h in model.hydrogen_slots()[partner] if h != proton)
    p = _coord(prepared, SITE, alternate).position
    line = f"HETATM  999 {isotope:4} AQ4 A 999    {p.x:8.3f}{p.y:8.3f}{p.z:8.3f}{0.6:6.2f}{17.0:6.2f}           D\n"
    source = _source(_at(SITE) + line)
    result = process_structure(source, ingress=options, requested_goals=GOALS)
    assert _names(result.structure, SITE) == (_names(prepared, SITE) - {proton}) | {
        isotope
    }
    assert _protons(result.structure, SITE)[isotope] == _protons(prepared, SITE)[proton]
    assert _coord(result.structure, SITE, isotope) == _coord(source, SITE, isotope)
    # The observed isotope sits at the *other* generated position on this parent.
    assert _coord(result.structure, SITE, alternate).position.distance_to(p) > 1.4
    removed = source.without_atom_refs((AtomRef(SITE, isotope),))
    refused = process_structure(removed, ingress=options, requested_goals=GOALS)
    assert not _names(refused.structure, SITE)
    assert any(
        i.residue_id == SITE and "replace or reparent" in i.message
        for i in refused.issues
    )


def _named_model(tmp_path, component, atoms, bonds):
    doc = gemmi.cif.Document()
    block = doc.add_new_block(component)
    block.set_pair("_chem_comp.id", component)
    table = block.init_loop(
        "_chem_comp_atom.", ["comp_id", "atom_id", "type_symbol", "charge"]
    )
    for name, element, charge in atoms:
        table.add_row([component, name, element, str(charge)])
    edges = block.init_loop(
        "_chem_comp_bond.", ["comp_id", "atom_id_1", "atom_id_2", "value_order"]
    )
    for a, b, order in bonds:
        edges.add_row([component, a, b, order])
    path = tmp_path / (component + ".cif")
    doc.write_file(str(path))
    return read_retained_component_definitions(path)[component]


def _pdb_atom(serial, rid, comp, name, element, xyz):
    x, y, z = xyz
    return f"HETATM{serial:5} {name:4} {comp:>3} {rid.chain_id}{rid.seq_num:4}    {x:8.3f}{y:8.3f}{z:8.3f}{0.8:6.2f}{13.0:6.2f}          {element:>2}\n"


def test_isolated_dictionary_does_not_shadow_coordinated_imidazole_or_erase_filtered_partner(
    tmp_path,
):
    ring = ("N1", "C2", "N3", "C4", "C5")
    slots = {"C2": "H2", "N3": "HN3", "C4": "H4", "C5": "H5"}
    model = _named_model(
        tmp_path,
        "IMD",
        [(name, name[0], 0) for name in (*ring, *slots.values())],
        [
            (name, ring[(i + 1) % 5], "DOUB" if i in (0, 3) else "SING")
            for i, name in enumerate(ring)
        ]
        + [(p, h, "SING") for p, h in slots.items()],
    )
    free, bound, metal = ResidueId("A", 1), ResidueId("A", 2), ResidueId("B", 30)
    text = "".join(
        _pdb_atom(
            10 * j + i + 1,
            rid,
            "IMD",
            name,
            name[0],
            (
                1.18 * cos(pi + i * 2 * pi / 5) + offset,
                1.18 * sin(pi + i * 2 * pi / 5),
                0,
            ),
        )
        for j, (rid, offset) in enumerate(((free, 12), (bound, 0)))
        for i, name in enumerate(ring)
    )
    raw = gemmi.read_pdb_string(
        text + _pdb_atom(30, metal, "FE", "FE", "FE", (-3.77, 0, 0))
    )
    link = gemmi.Connection()
    link.name = "declared"
    link.type = gemmi.ConnectionType.MetalC
    link.partner1 = gemmi.AtomAddress("A", gemmi.SeqId(2, " "), "IMD", "N1", "\0")
    link.partner2 = gemmi.AtomAddress("B", gemmi.SeqId(30, " "), "FE", "FE", "\0")
    raw.connections.append(link)
    text = raw.make_mmcif_document().as_string()
    policy = KEEP.structure_normalization_policy()
    source = read_structure_string(text, FileFormat.MMCIF, policy=policy)

    def run(s, ids):
        opts = StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=[
                model.for_residue(r) for r in ids
            ],
        )
        return process_structure(s, ingress=opts, requested_goals=GOALS)

    control = run(source, ())
    selected = run(source, (free,))
    assert _protons(selected.structure, free) == {h: p for p, h in slots.items()}
    assert len(_protons(control.structure, bound)) == 4
    assert _protons(selected.structure, bound) == _protons(control.structure, bound)
    for name in ring:
        ref = AtomRef(bound, name)
        assert selected.structure.topology.formal_charge(
            selected.structure.constitution.atom_index(ref)
        ) == control.structure.topology.formal_charge(
            control.structure.constitution.atom_index(ref)
        )
    for s in (
        source,
        read_structure_string(
            text, FileFormat.MMCIF, policy=replace(policy, selected_chain_ids=("A",))
        ),
        source.without_atom_refs((AtomRef(metal, "FE"),)),
    ):
        result = run(s, (free, bound))
        assert not _protons(result.structure, bound)
        assert _protons(result.structure, free) == _protons(selected.structure, free)
        assert any(
            i.residue_id == bound
            and "caller CCD IMD" in i.message
            and ("partner" in i.message or "attachment" in i.message)
            for i in result.issues
        )


def test_ccd_smiles_and_deposited_phosphate_follow_per_site_precedence(tmp_path):
    atoms = ("P", "O1", "O2", "O3", "O4")
    xyz = (
        (0, 0, 0),
        (0.9, 0.9, 0.9),
        (0.9, -0.9, -0.9),
        (-0.9, 0.9, -0.9),
        (-0.9, -0.9, 0.9),
    )
    # At a PI-labelled site, deliberately select the explicit 3- model.
    model = _named_model(
        tmp_path,
        "PI",
        [(n, n[0], 0 if i < 2 else -1) for i, n in enumerate(atoms)],
        [("P", n, "DOUB" if i == 1 else "SING") for i, n in enumerate(atoms) if i],
    )
    sites = tuple(ResidueId("A", i) for i in (1, 2, 3))
    text = "".join(
        _pdb_atom(10 * k + i, rid, "PI", name, name[0], (p[0] + 15 * k, p[1], p[2]))
        for k, rid in enumerate(sites)
        for i, (name, p) in enumerate(zip(atoms, xyz, strict=True))
    )
    overrides = (
        model.for_residue(sites[0]),
        RetainedNonPolymerChemistryOverride(sites[1], "P(=O)([O-])(O)[O-]", atoms),
    )
    opts = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        use_ccd_orthophosphate_models=True,
        retained_non_polymer_chemistry_overrides=overrides,
    )
    good = process_structure(_source(text), ingress=opts, requested_goals=GOALS)
    assert [list(_protons(good.structure, r).values()) for r in sites] == [
        [],
        ["O3"],
        ["O4"],
    ]
    assert (
        good.structure.topology.formal_charge(
            good.structure.constitution.atom_index(AtomRef(sites[0], "O4"))
        )
        == -1
    )
    bad_source = _source(
        text + _pdb_atom(80, sites[0], "PI", "DO4", "D", (-1.87, -0.9, 0.9))
    )
    bad = process_structure(bad_source, ingress=opts, requested_goals=GOALS)
    assert (
        bad.structure.topology.formal_charge(
            bad.structure.constitution.atom_index(AtomRef(sites[0], "O4"))
        )
        is None
    )
    assert _names(bad.structure, sites[0]) == {"DO4"}
    assert any(
        i.residue_id == sites[0] and "caller CCD PI" in i.message for i in bad.issues
    )
    assert [list(_protons(bad.structure, r).values()) for r in sites[1:]] == [
        ["O3"],
        ["O4"],
    ]


@pytest.mark.parametrize("known_template", [False, True])
def test_distorted_zero_h_site_refuses_both_refinement_and_h_work(
    tmp_path, known_template
):
    code = "SO4" if known_template else "XYZ"
    atoms = ("S", "O1", "O2", "O3", "O4")
    model = _named_model(
        tmp_path,
        code,
        [(n, n[0], 0 if i < 3 else -1) for i, n in enumerate(atoms)],
        [("S", n, "DOUB" if i <= 2 else "SING") for i, n in enumerate(atoms) if i],
    )
    positions = (
        (0, 0, 0),
        (0, -1.2, -0.85),
        (0, 1.2, -0.85),
        (-1.2, 0, 0.85),
        (9.2, 0, 0.85),
    )
    s = _source(
        "".join(
            _pdb_atom(i, SITE, code, n, n[0], p)
            for i, (n, p) in enumerate(zip(atoms, positions, strict=True), 1)
        )
    )
    assignment = model.for_residue(SITE)
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[assignment],
    )
    local = WorkflowTransformRequests(
        repair_refinement=RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues((SITE,)),
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationForceField.UFF
            ),
        )
    )
    h = process_structure(s, ingress=options, requested_goals=GOALS)
    ridx = s.constitution.residue_index(SITE)
    site = s.constitution.residue_site_at(ridx)
    resolution = resolve_retained_non_polymer_chemistry(
        s,
        site,
        component_library=build_default_component_library(),
        evidence=assignment,
    )
    assert resolution.unresolved_hydrogen_requirement is False
    assert (
        resolution.failure is not None
        and resolution.failure.selected_source
        is RetainedNonPolymerChemistryEvidenceSource.CALLER_COMPONENT_DEFINITION
    )
    for out in (h, process_structure(s, ingress=options, transform_requests=local)):
        assert any(
            i.residue_id == SITE
            and f"caller CCD {code}" in i.message
            and "geometry" in i.message
            for i in out.issues
        )
        assert out.structure.topology == s.topology and not out.repairs
    support = resolve_local_bond_planning_support(
        ProteinStructureSnapshot.from_structure(s),
        ridx,
        site,
        movable_atom_indices=(),
        component_library=build_default_component_library(),
        retained_non_polymer_chemistry_evidence=(assignment,),
    )
    assert (
        support.mode is LocalBondPlanningSupportMode.UNSUPPORTED
        and "geometry" in support.blocker_message
    )


@pytest.mark.parametrize(
    ("link", "detail"),
    [
        ("CONECT 2513 2514 2514", "order"),  # AQ4 C1-C2 is a triple bond.
        ("CONECT 2513 2523", "unexpected bond"),
    ],
)
def test_incompatible_declared_internal_edge_is_not_replaced(link, detail):
    model = read_retained_component_definitions(DATA / "ligands.cif")["AQ4"]
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[model.for_residue(SITE)],
    )
    source = _source(_at(SITE) + link + "\n")
    out = process_structure(source, ingress=options, requested_goals=GOALS)
    assert out.structure.topology == source.topology and not _names(out.structure, SITE)
    assert any(
        i.residue_id == SITE and "caller CCD AQ4" in i.message and detail in i.message
        for i in out.issues
    )


def test_ambiguous_non_dictionary_deuterium_is_not_assigned_by_atom_order(tmp_path):
    names = ("CM", "NT", "H1", "H2", "H3", "HN1", "HN2")
    model = _named_model(
        tmp_path,
        "AMN",
        [
            (name, element, 0)
            for name, element in zip(
                names, ("C", "N", "H", "H", "H", "H", "H"), strict=True
            )
        ],
        [("CM", "NT", "SING")]
        + [("CM" if h in names[2:5] else "NT", h, "SING") for h in names[2:]],
    )
    options = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=[model.for_residue(SITE)],
    )
    source = _source(
        "".join(
            _pdb_atom(i, SITE, "AMN", n, e, p)
            for i, (n, e, p) in enumerate(
                (
                    ("CM", "C", (-0.73, 0, 0)),
                    ("NT", "N", (0.73, 0, 0)),
                    ("DX", "D", (0, 0.72, 0)),
                ),
                1,
            )
        )
    )
    out = process_structure(source, ingress=options, requested_goals=GOALS)
    assert out.structure.topology == source.topology and _names(
        out.structure, SITE
    ) == {"DX"}
    assert any(
        i.residue_id == SITE and "no unambiguous" in i.message for i in out.issues
    )
