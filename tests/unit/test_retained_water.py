from dataclasses import replace

import gemmi
import numpy as np
import pytest

from protrepair.chemistry import RadiusKind, prepare_radius_lookup
from protrepair.diagnostics.clashes import ClashPolicy
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.water_contacts import (
    stationary_water_refusal,
    stationary_water_run,
)
from protrepair.errors import StructureNormalizationError, StructureSerializationError
from protrepair.geometry import Vec3
from protrepair.io import (
    FileFormat,
    NonPolymerRetentionPolicy,
    ResidueRole,
    RetentionAction,
    RetentionRule,
    read_structure_string,
    write_structure_string,
)
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.topology import StructureTopology
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.transformer.packing.domain import PackingResult
from protrepair.transformer.packing.spec import PackingSpec
from protrepair.workflow.engine.packing import reference as packing_reference

KEEP, DROP, REJECT = RetentionAction.KEEP, RetentionAction.DROP, RetentionAction.REJECT
KEEP_ALL = NonPolymerRetentionPolicy(KEEP, KEEP)
POLICY = StructureNormalizationPolicy(non_polymer_retention=KEEP_ALL)
PDB = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 12.00           C
HETATM    2 ZN    ZN Z   1      14.000   0.000   0.000  1.00 14.00          ZN
HETATM    3  O   HOH W   1       5.000   0.000   0.000  0.75 18.00           O
HETATM    4  O   DOD W   2      -5.000   0.000   0.000  1.00 22.00           O
HETATM    5  D1  DOD W   2      -5.950   0.000   0.000  1.00 24.00           D
END
"""


def moved(
    structure: ProteinStructure, rid: ResidueId, atom: str, x: float
) -> ProteinStructure:
    geometry = structure.residue_geometry(structure.constitution.residue_index(rid))
    new_atom = geometry.atom_geometry(atom).with_position(Vec3(x, 0.0, 0.0))
    return structure.with_updated_residue_geometries(
        ((rid, geometry.with_atom_geometry(atom, new_atom)),)
    )


@pytest.mark.parametrize("fmt", (FileFormat.PDB, FileFormat.MMCIF))
def test_waters_and_ions_round_trip_in_separate_roles(fmt):
    structure = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    data = write_structure_string(structure, fmt)
    restored = read_structure_string(data, fmt, policy=POLICY)
    c = restored.constitution
    assert [site.component_id for site in c.waters] == ["HOH", "DOD"]
    assert [site.component_id for site in c.ligands] == ["ZN"]
    assert [c.role_at(c.residue_index(s.residue_id)) for s in c.residue_slots] == [
        ResidueRole.POLYMER,
        ResidueRole.LIGAND,
        ResidueRole.WATER,
        ResidueRole.WATER,
    ]
    assert c.waters == structure.constitution.waters
    for water in c.waters:
        assert restored.residue_geometry(
            c.residue_index(water.residue_id)
        ) == structure.residue_geometry(
            structure.constitution.residue_index(water.residue_id)
        )
    assert not read_structure_string(data, fmt).constitution.waters


@pytest.mark.parametrize("fmt", (FileFormat.PDB, FileFormat.MMCIF))
def test_tritium_on_retained_water_keeps_its_name_and_isotope(fmt):
    text = PDB.replace(" D1  DOD", " T1  DOD").replace(
        "24.00           D", "24.00           T"
    )
    retained = read_structure_string(text, FileFormat.PDB, policy=POLICY)
    restored = read_structure_string(
        write_structure_string(retained, fmt), fmt, policy=POLICY
    )
    atom = restored.constitution.waters[1].atom_site("T1")
    assert atom.element_identity.source_symbol == "T"


def test_retention_uses_last_match_and_same_rules_at_reentry():
    rules = (
        RetentionRule(REJECT, chain_ids=("W",)),
        RetentionRule(DROP, role=ResidueRole.WATER),
        RetentionRule(KEEP, component_ids=(" dod ",), residue_ids=(ResidueId("W", 2),)),
        RetentionRule(KEEP, role=ResidueRole.LIGAND, component_ids=("ZN",)),
    )
    policy = replace(
        POLICY, non_polymer_retention=NonPolymerRetentionPolicy(REJECT, REJECT, rules)
    )
    wet = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    for carrier in (
        read_structure_string(PDB, FileFormat.PDB, policy=policy),
        apply_structure_normalization_policy(wet, policy=policy),
    ):
        assert carrier.constitution.waters == (wet.constitution.waters[1],)
        assert carrier.constitution.ligands == wet.constitution.ligands
        assert carrier.constitution.chains == wet.constitution.chains
    with pytest.raises(StructureNormalizationError, match="HOH"):
        read_structure_string(
            PDB,
            FileFormat.PDB,
            policy=replace(
                policy, non_polymer_retention=NonPolymerRetentionPolicy(REJECT, KEEP)
            ),
        )


def test_stationary_guard_compares_named_pairs_and_both_baselines():
    source = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    protein = ResidueId("A", 1)
    root = moved(source, protein, "CA", 3.5)
    other_water = moved(root, protein, "CA", -3.5)
    issue = stationary_water_refusal(root, other_water)
    assert (
        issue is not None
        and issue.kind is ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED
    )
    assert "W:2" in issue.message
    assert stationary_water_refusal(root, source) is None
    with stationary_water_run(root):
        assert (
            stationary_water_refusal(source, moved(source, protein, "CA", 3.0))
            is not None
        )
    assert stationary_water_refusal(root, moved(root, protein, "CA", 3.0)) is None
    assert (
        stationary_water_refusal(source, moved(source, ResidueId("W", 1), "O", 5.1))
        is not None
    )
    dry = source.without_waters()
    assert stationary_water_refusal(dry, moved(dry, protein, "CA", -3.5)) is None
    assert stationary_water_refusal(source, dry) is not None


def test_entity_only_water_requires_mmcif_for_lossless_export():
    ordinary = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    doc = gemmi.cif.read_string(write_structure_string(ordinary, FileFormat.MMCIF))
    for key in ("label_comp_id", "auth_comp_id"):
        column = doc.sole_block().find_values("_atom_site." + key)
        for i, value in enumerate(column):
            if value == "HOH":
                column[i] = "WXY"
    water = read_structure_string(doc.as_string(), FileFormat.MMCIF, policy=POLICY)
    assert water.constitution.waters[0].component_id == "WXY"
    reread = read_structure_string(
        write_structure_string(water, FileFormat.MMCIF), FileFormat.MMCIF, policy=POLICY
    )
    assert reread.constitution.waters == water.constitution.waters
    with pytest.raises(StructureSerializationError, match="PDB cannot preserve"):
        write_structure_string(water, FileFormat.PDB)


@pytest.mark.parametrize("prepare", (add_hydrogens, add_retained_non_polymer_hydrogens))
def test_explicit_water_is_not_a_successful_hydrogen_preparation_target(prepare):
    wet = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    with pytest.raises(ValueError, match="retained water W:2"):
        prepare(wet, target_residue_ids=frozenset((ResidueId("W", 2),)))


def test_indexed_admission_matches_all_carbon_water_pairs():
    rng = np.random.default_rng(78)
    waters = np.round(rng.uniform(-7.0, 7.0, (20, 3)), 3)
    carbons = np.round(rng.uniform(-7.0, 7.0, (5, 3)), 3)
    rows = []
    for record, coordinates, element, name, component, chain in (
        ("ATOM", carbons, "C", "CA", "ALA", "A"),
        ("HETATM", waters, "O", "O", "HOH", "W"),
    ):
        for i, (x, y, z) in enumerate(coordinates, 1):
            rows.append(
                f"{record:6}{len(rows) + 1:5} {name:>4} {component} {chain}{i:4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 12.00          {element:>2}"
            )
    source = read_structure_string(
        "\n".join((*rows, "END", "")), FileFormat.PDB, policy=POLICY
    )
    lookup = prepare_radius_lookup(("C", "O"), RadiusKind.VAN_DER_WAALS)
    limit = ClashPolicy().allowed_distance_angstrom(
        left_van_der_waals_radius_angstrom=lookup.radius_angstrom("C"),
        right_van_der_waals_radius_angstrom=lookup.radius_angstrom("O"),
        left_is_hydrogen=False,
        right_is_hydrogen=False,
    )
    baseline = np.maximum(0.0, limit - np.linalg.norm(waters - carbons[0], axis=1))
    rid = ResidueId("A", 1)
    geom = source.residue_geometry(source.constitution.residue_index(rid))
    expected_outcomes = set()
    for coords in (carbons[0], *waters, *rng.uniform(-10.0, 10.0, (18, 3))):
        distance = np.linalg.norm(waters - coords, axis=1)
        worsened = bool(np.any(np.maximum(0.0, limit - distance) > baseline + 1e-6))
        expected_outcomes.add(worsened)
        moved_ca = geom.atom_geometry("CA").with_position(Vec3(*coords))
        changed = source.with_updated_residue_geometries(
            ((rid, geom.with_atom_geometry("CA", moved_ca)),)
        )
        assert (stationary_water_refusal(source, changed) is not None) is worsened
    assert expected_outcomes == {True, False}


def test_changed_local_bond_cannot_borrow_old_clash_allowance():
    extra = "ATOM      7  CB  ALA A   1       2.500   0.000   0.000  1.00 12.00           C\n"
    source = moved(
        read_structure_string(extra + PDB, FileFormat.PDB, policy=POLICY),
        ResidueId("A", 1),
        "CA",
        4.0,
    )
    c = source.constitution
    ca, cb = (c.atom_index(AtomRef(ResidueId("A", 1), name)) for name in ("CA", "CB"))
    bond = source.topology.bond_between(ca, cb)
    assert bond is not None
    topology = StructureTopology(
        constitution=c,
        atom_topologies=source.topology.atom_topologies,
        bonds=tuple(
            replace(b, order=2) if b == bond else b for b in source.topology.bonds
        ),
    )
    candidate = ProteinStructure.from_payload(
        constitution=c,
        geometry=source.geometry,
        topology=topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    assert stationary_water_refusal(source, candidate) is not None


def test_selected_altlocs_on_different_residues_coexist():
    rows: list[str] = list(PDB.splitlines())
    rows[0] = rows[0][:16] + "A" + rows[0][17:]
    rows[2] = rows[2][:16] + "B" + rows[2][17:]
    source = read_structure_string(
        "\n".join(rows) + "\n", FileFormat.PDB, policy=POLICY
    )
    parent, water = ResidueId("A", 1), ResidueId("W", 1)
    for rid, atom, label in ((parent, "CA", "A"), (water, "O", "B")):
        assert (
            source.residue_geometry(source.constitution.residue_index(rid))
            .atom_geometry(atom)
            .altloc
            == label
        )
    assert (
        stationary_water_refusal(source, moved(source, parent, "CA", 4.0)) is not None
    )


def test_normalized_input_bounds_precision_drift_and_catches_lost_custody():
    source = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    protein = ResidueId("A", 1)
    root = moved(source, protein, "CA", 4.0)
    first = moved(root, protein, "CA", 4.0 + 0.75e-6)
    next_step = moved(first, protein, "CA", 4.0 + 1.5e-6)
    dry = root.without_waters()
    assert stationary_water_refusal(first, next_step) is None
    with stationary_water_run(root):
        assert stationary_water_refusal(root, first) is None
        assert stationary_water_refusal(first, next_step) is not None
        assert stationary_water_refusal(dry, dry) is not None
    assert stationary_water_refusal(dry, dry) is None


def test_changed_donor_invalidates_polar_contact_with_fixed_endpoints():
    pdb = """ATOM      1  CB  SER A   1      -1.400   0.000   0.000  1.00 12.00           C
ATOM      2  OG  SER A   1       0.000   0.000   0.000  1.00 12.00           O
ATOM      3  HG  SER A   1       1.000   0.000   0.000  1.00 12.00           H
HETATM    4  O   HOH W   1       2.700   0.000   0.000  1.00 12.00           O
END
"""
    source = read_structure_string(pdb, FileFormat.PDB, policy=POLICY)
    shifted = moved(source, ResidueId("A", 1), "OG", -2.0)
    issue = stationary_water_refusal(source, shifted)
    assert issue is not None and "HG" in issue.message
    assert stationary_water_refusal(shifted, source) is None


def test_changed_polar_pair_with_ambiguous_water_chemistry_is_unassessed():
    rows = """ATOM      1  CB  SER A   1      -1.400   0.000   0.000  1.00 12.00           C
ATOM      2  OG  SER A   1       0.000   0.000   0.000  1.00 12.00           O
ATOM      3  HG  SER A   1       1.000   0.000   0.000  1.00 12.00           H
HETATM    4  O   HOH W   1       2.700   0.000   0.000  1.00 12.00           O
HETATM    5  O2  HOH W   1       7.000   0.000   0.000  1.00 12.00           O
END
"""
    source = read_structure_string(rows, FileFormat.PDB, policy=POLICY)
    outcome = stationary_water_refusal(
        source, moved(source, ResidueId("A", 1), "HG", 1.05)
    )
    assert outcome is not None and "chemistry is unresolved" in outcome.message


def test_water_only_selection_and_ligand_replacement_preserve_facets():
    source = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    stripped = source.with_ligand_facets(ligand_sites=(), ligand_geometries=())
    waters = stripped.select_chains(("W",))
    assert not waters.constitution.chains
    assert not waters.constitution.ligands
    assert waters.constitution.waters == source.constitution.waters
    for site in waters.constitution.waters:
        rid = site.residue_id
        assert waters.residue_geometry(
            waters.constitution.residue_index(rid)
        ) == source.residue_geometry(source.constitution.residue_index(rid))


@pytest.mark.parametrize("at_new_atom", [False, True])
def test_direct_terminal_placement_does_not_publish_a_water_clash(at_new_atom):
    ala = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 12.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 12.00           C
ATOM      3  C   ALA A   1       2.013   1.424   0.000  1.00 12.00           C
ATOM      4  O   ALA A   1       1.369   2.472   0.000  1.00 12.00           O
ATOM      5  CB  ALA A   1       1.970  -0.748  -1.222  1.00 12.00           C
"""
    dry = read_structure_string(ala + "END\n", FileFormat.PDB)
    rid = ResidueId("A", 1)
    result = augment_c_terminal_oxt(dry)
    x, y, z = (
        result.structure.residue_geometry(
            result.structure.constitution.residue_index(rid)
        ).position("OXT")
        if at_new_atom
        else (20.0, 20.0, 20.0)
    )
    row = f"HETATM    6  O   HOH W   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 11.00           O"
    wet = read_structure_string(ala + row + "\nEND\n", FileFormat.PDB, policy=POLICY)
    checked = augment_c_terminal_oxt(wet)
    assert bool(checked.repairs) != at_new_atom
    assert (checked.structure is wet) == at_new_atom
    assert (
        any(
            i.kind is ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED
            for i in checked.issues
        )
        == at_new_atom
    )


@pytest.mark.parametrize("clashing", [False, True])
def test_packing_reference_distinguishes_success_and_withheld_candidate(
    monkeypatch, clashing
):
    source = read_structure_string(PDB, FileFormat.PDB, policy=POLICY)
    target = ResidueId("A", 1)
    packed = moved(source, target, "CA", 4.5 if clashing else 0.2)
    monkeypatch.setattr(
        packing_reference,
        "execute_sidechain_packing",
        lambda *_: PackingResult(packed, (target,), (), "test"),
    )
    reference = packing_reference.prepare_workflow_packing_reference(
        source, PackingSpec("faspr")
    )
    if clashing:
        assert reference.reference_structure is None
        assert reference.packing_result.packed_structure is source
        assert not reference.packing_result.changed_residue_ids
        assert any(
            i.kind is ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED
            for i in reference.packing_result.issues
        )
    else:
        assert reference.reference_structure is packed
