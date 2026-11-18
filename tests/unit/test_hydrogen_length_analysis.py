import builtins
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.hydrogen_lengths import (
    HG,
    REFERENCE_CIF,
    THIOL,
    reference,
    structure_with_thiol,
    with_position,
    with_topology,
)

from protrepair.analysis import (
    HydrogenLengthConvention,
    HydrogenLengthReference,
    HydrogenLengthStatus,
    analyze_hydrogen_lengths,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.scope import (
    AtomSetScope,
    ChainSetScope,
    ResidueSetScope,
)
from protrepair.structure import (
    AtomRef,
    BondRelationshipType,
    FileFormat,
    ProteinStructure,
    ResidueId,
    TopologyBond,
)


@pytest.mark.parametrize(
    ("distance", "status"),
    [
        (1.0, HydrogenLengthStatus.TOO_SHORT),
        (1.12, HydrogenLengthStatus.WITHIN_REFERENCE),
        (1.20, HydrogenLengthStatus.WITHIN_REFERENCE),
        (1.29, HydrogenLengthStatus.BETWEEN_REFERENCES),
        (1.38, HydrogenLengthStatus.WITHIN_REFERENCE),
        (1.40, HydrogenLengthStatus.TOO_LONG),
    ],
)
def test_observed_thiol_uses_applicable_explicit_ranges(distance, status):
    structure = structure_with_thiol(distance)
    before = write_structure_string(structure, FileFormat.PDB)
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))

    assert result.status is status
    assert result.parent == THIOL
    assert result.distance_angstrom == pytest.approx(distance)
    assert result.comparison is not None
    assert result.comparison.bounds_angstrom == (1.12, 1.38)
    if status is HydrogenLengthStatus.BETWEEN_REFERENCES:
        assert result.comparison.matching_conventions == ()
        assert result.comparison.outer_margin_angstrom == pytest.approx(0.09)
    assert write_structure_string(structure, FileFormat.PDB) == before


def test_explicit_convention_can_disagree_where_outer_envelope_does_not():
    structure = structure_with_thiol(1.29)
    (nuclear,) = analyze_hydrogen_lengths(
        structure,
        references=(reference(),),
        convention=HydrogenLengthConvention.NUCLEAR,
    )
    assert nuclear.status is HydrogenLengthStatus.TOO_SHORT
    assert nuclear.comparison is not None
    assert nuclear.comparison.bounds_angstrom == (1.30, 1.38)


def test_unknown_convention_uses_envelope_even_for_nested_intervals():
    supplied = reference(REFERENCE_CIF.replace("1.34 0.01", "1.34 0.09"))
    (result,) = analyze_hydrogen_lengths(
        structure_with_thiol(1.0), references=(supplied,)
    )
    assert result.status is HydrogenLengthStatus.WITHIN_REFERENCE
    assert result.comparison is not None
    assert result.comparison.bounds_angstrom == (0.98, 1.7)
    assert result.comparison.matching_conventions == (HydrogenLengthConvention.NUCLEAR,)


def test_overlapping_intervals_can_both_match():
    supplied = reference(REFERENCE_CIF.replace("1.34 0.01", "1.30 0.02"))
    (result,) = analyze_hydrogen_lengths(
        structure_with_thiol(1.25), references=(supplied,)
    )
    assert result.status is HydrogenLengthStatus.WITHIN_REFERENCE
    assert result.comparison is not None
    assert result.comparison.matching_conventions == (
        HydrogenLengthConvention.ELECTRON_CLOUD,
        HydrogenLengthConvention.NUCLEAR,
    )


def test_missing_convention_profile_does_not_erase_measured_length():
    structure = structure_with_thiol(1.0)
    supplied = reference(REFERENCE_CIF.replace("1.34 0.01", "1.34 ?"))
    (unknown,) = analyze_hydrogen_lengths(structure, references=(supplied,))
    (electron,) = analyze_hydrogen_lengths(
        structure,
        references=(supplied,),
        convention=HydrogenLengthConvention.ELECTRON_CLOUD,
    )
    assert unknown.status is HydrogenLengthStatus.UNASSESSED
    assert unknown.distance_angstrom == pytest.approx(1.0)
    assert unknown.comparison is None
    assert unknown.detail == "missing reference for the requested convention or term"
    assert electron.status is HydrogenLengthStatus.TOO_SHORT


def test_name_position_conflict_does_not_reassign_or_grade_hydrogen():
    structure = structure_with_thiol(-1.3)
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    assert result.status is HydrogenLengthStatus.PARENT_POSITION_CONFLICT
    assert result.parent == THIOL
    assert result.distance_angstrom is None
    assert result.comparison is None
    assert result.position_conflict is not None
    assert result.position_conflict.nearest == (AtomRef(THIOL.residue_id, "CB"),)
    assert result.position_conflict.assigned_distance_angstrom == pytest.approx(1.3)
    assert result.position_conflict.nearest_distance_angstrom == pytest.approx(0.5)


def test_ordered_thiol_bond_keeps_measured_distance_on_position_conflict():
    raw = structure_with_thiol(-1.3)
    c = raw.constitution
    structure = with_topology(
        raw,
        bonds=(TopologyBond(c.atom_index(THIOL), c.atom_index(HG)),),
        keep_source=False,
    )
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    assert result.status is HydrogenLengthStatus.PARENT_POSITION_CONFLICT
    assert result.parent == THIOL
    assert result.distance_angstrom == pytest.approx(1.3)
    assert result.position_conflict is not None
    assert result.position_conflict.nearest == (AtomRef(THIOL.residue_id, "CB"),)
    assert result.comparison is None


def test_hand_built_coupled_site_without_observation_is_not_inferred():
    structure = with_topology(structure_with_thiol(1.0), keep_source=False)
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.distance_angstrom is None
    assert result.detail


def test_hand_built_thiol_is_assessed_with_a_compatible_ordered_bond():
    raw = structure_with_thiol(1.0)
    c = raw.constitution
    structure = with_topology(
        raw,
        bonds=(TopologyBond(c.atom_index(THIOL), c.atom_index(HG)),),
        keep_source=False,
    )
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    assert result.status is HydrogenLengthStatus.TOO_SHORT
    assert result.distance_angstrom == pytest.approx(1.0)


def test_conflicting_current_attachment_cannot_replace_source_parent():
    raw = structure_with_thiol(1.0)
    c = raw.constitution
    structure = with_topology(
        raw,
        bonds=(
            TopologyBond(
                c.atom_index(AtomRef(THIOL.residue_id, "CB")), c.atom_index(HG)
            ),
        ),
    )
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.comparison is None
    assert result.detail == "current H attachment contradicts its source parent"


def test_tied_nearest_heavy_atoms_withdraw_name_based_measurement():
    (result,) = analyze_hydrogen_lengths(
        structure_with_thiol(-0.9), references=(reference(),)
    )
    assert result.status is HydrogenLengthStatus.PARENT_POSITION_CONFLICT
    assert result.position_conflict is not None
    assert set(result.position_conflict.nearest) == {
        THIOL,
        AtomRef(THIOL.residue_id, "CB"),
    }
    assert result.distance_angstrom is None


def test_default_observer_reports_equal_nearest_parents_in_structure_order():
    source = structure_with_thiol(1.0)
    carbon = AtomRef(THIOL.residue_id, "CB")
    structure = with_position(
        source, source.constitution.atom_index(carbon), Vec3(2.0, 0, 0)
    )

    (row,) = analyze_hydrogen_lengths(structure)
    assert row.status is HydrogenLengthStatus.PARENT_POSITION_CONFLICT
    assert row.parent == THIOL and row.distance_angstrom is None
    assert row.position_conflict is not None
    assert row.position_conflict.nearest == (carbon, THIOL)
    assert row.position_conflict.assigned_distance_angstrom == 1.0
    assert row.position_conflict.nearest_distance_angstrom == 1.0


@pytest.mark.parametrize("missing_profile", [False, True])
def test_uncatalogued_charge_is_not_graded_against_neutral_thiol(missing_profile):
    raw = structure_with_thiol(1.0)
    c = raw.constitution
    structure = with_topology(
        raw,
        sulfur_charge=-1,
        bonds=(TopologyBond(c.atom_index(THIOL), c.atom_index(HG)),),
    )
    text = (
        REFERENCE_CIF.replace("1.34 0.01", "1.34 ?")
        if missing_profile
        else REFERENCE_CIF
    )
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(text),))
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.distance_angstrom == pytest.approx(1.0)
    assert result.comparison is None
    assert (
        result.detail
        == "observed chemistry does not match the supplied side-chain reference"
    )


def test_coordinated_sulfur_cannot_use_isolated_thiol_reference():
    raw = structure_with_thiol(
        1.0,
        suffix="HETATM    8 ZN    ZN B   1       0.000   2.400   0.000  1.00 20.00          ZN\n",
    )
    c = raw.constitution
    zinc = AtomRef(ResidueId("B", 1), "ZN")
    structure = with_topology(
        raw,
        bonds=(
            TopologyBond(c.atom_index(THIOL), c.atom_index(HG)),
            TopologyBond(
                c.atom_index(THIOL),
                c.atom_index(zinc),
                order=None,
                relationship_type=BondRelationshipType.METAL_COORDINATION,
            ),
        ),
    )
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.comparison is None


@pytest.mark.parametrize("declared", [False, True])
def test_source_or_geometric_disulfide_precludes_thiol_grading(declared: bool):
    raw = structure_with_thiol(
        1.0,
        suffix="ATOM      8  SG  CYS B   1       0.000   2.050   0.000  1.00 20.00           S\n",
    )
    c = raw.constitution
    second = AtomRef(ResidueId("B", 1), "SG")
    assert raw.topology.bond_between(c.atom_index(THIOL), c.atom_index(second)) is None
    structure = (
        with_topology(
            raw,
            bonds=(
                TopologyBond(
                    c.atom_index(THIOL),
                    c.atom_index(second),
                    relationship_type=BondRelationshipType.DISULFIDE,
                ),
            ),
        )
        if declared
        else raw
    )
    (result,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.comparison is None
    assert result.detail is not None and "disulfide" in result.detail


@pytest.mark.parametrize("file_format", [FileFormat.PDB, FileFormat.MMCIF])
def test_reference_and_scope_are_snapshot_bound_after_coordinate_round_trip(
    file_format,
):
    original = structure_with_thiol(1.38)
    structure = read_structure_string(
        write_structure_string(original, file_format), file_format
    )
    (result,) = analyze_hydrogen_lengths(
        structure,
        references=(reference(),),
        scope=ResidueSetScope((THIOL.residue_id,)),
    )
    assert result.status is HydrogenLengthStatus.WITHIN_REFERENCE
    assert result.comparison is not None
    assert result.comparison.outer_margin_angstrom == 0.0


def test_internal_coordinates_are_not_projected_to_pdb_precision():
    raw = structure_with_thiol(1.12)
    index = raw.constitution.atom_index(HG)
    structure = with_position(raw, index, Vec3(1.1199, 0.0, 0.0))
    (internal,) = analyze_hydrogen_lengths(structure, references=(reference(),))
    (rounded,) = analyze_hydrogen_lengths(
        read_structure_string(
            write_structure_string(structure, FileFormat.PDB), FileFormat.PDB
        ),
        references=(reference(),),
    )
    assert internal.status is HydrogenLengthStatus.TOO_SHORT
    assert rounded.status is HydrogenLengthStatus.WITHIN_REFERENCE


def test_missing_reference_and_no_selected_hydrogens_are_distinct():
    structure = structure_with_thiol(1.0)
    (ungraded,) = analyze_hydrogen_lengths(structure, references=())
    assert ungraded.status is HydrogenLengthStatus.UNASSESSED
    assert ungraded.distance_angstrom == pytest.approx(1.0)
    assert (
        analyze_hydrogen_lengths(
            structure, references=(reference(),), scope=AtomSetScope((THIOL,))
        )
        == ()
    )


def test_only_explicit_reference_loading_requires_rdkit(monkeypatch):
    structure = structure_with_thiol(1.0)
    original_import = builtins.__import__

    def without_rdkit(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("injected unavailable RDKit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rdkit)
    (ungraded,) = analyze_hydrogen_lengths(structure)
    assert ungraded.status is HydrogenLengthStatus.UNASSESSED
    assert ungraded.distance_angstrom == pytest.approx(1.0)

    with pytest.raises(RdkitUnavailableError, match="monomer reference validation"):
        reference()


def test_conflicting_reference_inputs_do_not_choose_one_by_order():
    with pytest.raises(ValueError, match="at most one"):
        analyze_hydrogen_lengths(
            structure_with_thiol(1.0), references=(reference(), reference())
        )


def test_selected_deuterium_uses_hg_row_without_changing_identity():
    structure = structure_with_thiol(1.0, name="DG", element="D")
    dg = AtomRef(THIOL.residue_id, "DG")
    (result,) = analyze_hydrogen_lengths(
        structure, references=(reference(),), scope=AtomSetScope((dg,))
    )
    assert result.hydrogen == dg
    assert result.status is HydrogenLengthStatus.TOO_SHORT
    assert (
        structure.constitution.atom_site_at(
            structure.constitution.atom_index(dg)
        ).element
        == "D"
    )


def test_bad_sibling_uses_existing_residue_attachment_failure():
    raw = structure_with_thiol(1.0, extra_sites=(("QH", "H", -1.8, 1.09, 0.0),))
    c = raw.constitution
    with_ordered_thiol = with_topology(
        raw, bonds=(TopologyBond(c.atom_index(THIOL), c.atom_index(HG)),)
    )
    results = analyze_hydrogen_lengths(with_ordered_thiol, references=(reference(),))
    assert tuple(row.hydrogen.atom_name for row in results) == ("HG", "QH")
    assert all(row.status is HydrogenLengthStatus.UNASSESSED for row in results)
    assert all(row.distance_angstrom is None for row in results)
    assert {row.detail for row in results} == {"unresolved H attachment: QH"}


@pytest.mark.parametrize("keep_source", [False, True])
def test_ordered_thiol_does_not_bypass_overfilled_hydrogen_inventory(keep_source):
    source = structure_with_thiol(
        1.0,
        extra_sites=(
            ("HB1", "H", -1.8, -1.09, 0.0),
            ("HB2", "H", -1.8, 1.09, 0.0),
            ("HB3", "H", -1.8, 0.0, 1.09),
        ),
    )
    c = source.constitution
    observed = with_topology(
        source,
        bonds=(TopologyBond(c.atom_index(THIOL), c.atom_index(HG)),),
        keep_source=keep_source,
    )
    (result,) = analyze_hydrogen_lengths(
        observed, references=(reference(),), scope=AtomSetScope((HG,))
    )
    assert result.status is HydrogenLengthStatus.UNASSESSED
    if keep_source:
        assert result.detail == "source H exceed fixed template inventory at CB"
    else:
        assert result.detail == (
            "observed chemistry does not match the supplied side-chain reference"
        )


def test_observed_thiol_does_not_inherit_name_evidence_from_a_source_serine():
    cysteine = structure_with_thiol(1.0)
    serine = read_structure_string(
        write_structure_string(cysteine, FileFormat.PDB)
        .replace(" CYS ", " SER ")
        .replace(" SG ", " OG ")
        .replace("           S\n", "           O\n"),
        FileFormat.PDB,
    )
    changed = ProteinStructure.from_payload(
        constitution=cysteine.constitution,
        geometry=cysteine.geometry,
        topology=cysteine.topology,
        polymer_blueprint=cysteine.polymer_blueprint,
        provenance=replace(cysteine.provenance, ingress=serine.provenance.ingress),
    )

    (uncertain,) = analyze_hydrogen_lengths(changed, references=(reference(),))
    (control,) = analyze_hydrogen_lengths(cysteine, references=(reference(),))
    assert control.status is HydrogenLengthStatus.TOO_SHORT
    assert uncertain.status is HydrogenLengthStatus.UNASSESSED
    assert uncertain.detail == "source and observed residue components differ"


def test_explicit_parent_wins_but_cannot_use_reference_for_another_attachment():
    source = structure_with_thiol(
        1.20,
        name="HB2",
        extra_sites=(("HG", "H", -1.8, 1.09, 0.0), ("HB3", "H", -1.8, 0.0, 1.09)),
    )
    c = source.constitution
    cb = AtomRef(THIOL.residue_id, "CB")
    hb2 = AtomRef(THIOL.residue_id, "HB2")
    observed = with_topology(
        source,
        bonds=(
            TopologyBond(c.atom_index(THIOL), c.atom_index(hb2)),
            TopologyBond(c.atom_index(cb), c.atom_index(HG)),
        ),
        keep_source=False,
    )
    (result,) = analyze_hydrogen_lengths(
        observed, references=(reference(),), scope=AtomSetScope((HG,))
    )
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.parent == cb
    assert result.distance_angstrom == pytest.approx(1.09)
    assert (
        result.detail
        == "assigned parent differs from the supplied reference attachment"
    )


def test_fixed_side_chain_measurement_needs_compatible_thiol_state_to_grade():
    supplied = reference(
        REFERENCE_CIF.replace(
            "CB HB2 SINGLE n ? ? ? ?", "CB HB2 SINGLE n 0.97 0.02 1.09 0.02"
        )
    )
    hb2 = AtomRef(THIOL.residue_id, "HB2")
    other_hydrogens = (
        ("HB2", "H", -1.8, 1.09, 0.0),
        ("HB3", "H", -1.8, 0.0, 1.09),
    )
    full = structure_with_thiol(1.20, extra_sites=other_hydrogens)
    no_thiol_h = structure_with_thiol(1.09, "HB2", extra_sites=(other_hydrogens[1],))
    index = no_thiol_h.constitution.atom_index(hb2)
    no_thiol_h = with_position(no_thiol_h, index, Vec3(-1.8, 1.09, 0.0))

    (assessed,) = analyze_hydrogen_lengths(
        full, references=(supplied,), scope=AtomSetScope((hb2,))
    )
    (unassessed,) = analyze_hydrogen_lengths(
        no_thiol_h, references=(supplied,), scope=AtomSetScope((hb2,))
    )
    assert assessed.status is HydrogenLengthStatus.WITHIN_REFERENCE
    assert unassessed.status is HydrogenLengthStatus.UNASSESSED
    assert unassessed.distance_angstrom == assessed.distance_angstrom
    assert (
        unassessed.detail
        == "observed chemistry does not match the supplied side-chain reference"
    )


def test_free_ammonium_row_does_not_grade_backbone_hydrogen():
    raw = structure_with_thiol(1.20, extra_sites=(("H", "H", -3.0, 2.02, 0.0),))
    c = raw.constitution
    nitrogen = AtomRef(THIOL.residue_id, "N")
    hydrogen = AtomRef(THIOL.residue_id, "H")
    raw = with_topology(
        raw, bonds=(TopologyBond(c.atom_index(nitrogen), c.atom_index(hydrogen)),)
    )

    (result,) = analyze_hydrogen_lengths(
        raw, references=(reference(),), scope=AtomSetScope((hydrogen,))
    )
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.parent == nitrogen
    assert result.distance_angstrom == pytest.approx(1.02)
    assert (
        result.detail == "backbone length references are outside the diagnostic domain"
    )


def test_zero_distance_is_measured_and_flagged():
    (result,) = analyze_hydrogen_lengths(
        structure_with_thiol(0.0), references=(reference(),)
    )
    assert result.status is HydrogenLengthStatus.TOO_SHORT
    assert result.distance_angstrom == 0.0
    assert result.position_conflict is None


@pytest.mark.parametrize("invalid_x", [float("nan"), float("inf")])
def test_nonfinite_heavy_context_cannot_certify_nearest_parent(invalid_x):
    source = structure_with_thiol(
        1.20,
        suffix="ATOM      8  SG  CYS B   1       0.000  20.000   0.000  1.00 20.00           S\n",
    )
    other_sulfur = source.constitution.atom_index(AtomRef(ResidueId("B", 1), "SG"))
    structure = with_position(source, other_sulfur, Vec3(invalid_x, 20.0, 0.0))

    (control,) = analyze_hydrogen_lengths(
        source, references=(reference(),), scope=ChainSetScope(("A",))
    )
    (result,) = analyze_hydrogen_lengths(
        structure, references=(reference(),), scope=ChainSetScope(("A",))
    )
    assert control.status is HydrogenLengthStatus.WITHIN_REFERENCE
    assert result.status is HydrogenLengthStatus.UNASSESSED
    assert result.detail == "unusable Cartesian heavy-atom context"


def test_nonpolymer_hydrogen_is_returned_unassessed_in_structure_order():
    residue_id = ResidueId("B", 7)
    ligand_h = AtomRef(residue_id, "H1")
    structure = structure_with_thiol(
        1.20,
        suffix=(
            "HETATM    8  C1  UNL B   7      10.000   0.000   0.000  1.00 20.00           C\n"
            "HETATM    9  H1  UNL B   7      11.000   0.000   0.000  1.00 20.00           H\n"
        ),
    )
    thiol, ligand = analyze_hydrogen_lengths(
        structure, references=(reference(),), scope=AtomSetScope((ligand_h, HG))
    )
    assert thiol.hydrogen == HG
    assert thiol.status is HydrogenLengthStatus.WITHIN_REFERENCE
    assert ligand.hydrogen == ligand_h
    assert ligand.status is HydrogenLengthStatus.UNASSESSED
    assert ligand.detail == "outside the standard polymer domain"


def test_duplicate_monomer_bond_is_invalid_configuration():
    with pytest.raises(ValueError, match="unrepeated endpoints"):
        reference(REFERENCE_CIF + "HG SG SINGLE n 1.2 0.02 1.34 0.01\n")


def test_monomer_cannot_relabel_the_elements_of_a_standard_scaffold():
    with pytest.raises(ValueError, match="standard side-chain scaffold"):
        reference(REFERENCE_CIF.replace("SG S 0", "SG O 0"))


def test_reference_accepts_explicit_kekule_encoding_of_stock_aromatic_scaffold():
    phe = (
        REFERENCE_CIF.replace("comp_CYS", "comp_PHE")
        .replace("SG S 0", "CG C 0\nCD1 C 0\nCD2 C 0\nCE1 C 0\nCE2 C 0\nCZ C 0")
        .replace("HG H 0", "HD1 H 0\nHD2 H 0\nHE1 H 0\nHE2 H 0\nHZ H 0")
        .replace("CB SG SINGLE n ? ? ? ?", "CB CG SINGLE n ? ? ? ?")
        .replace(
            "SG HG SINGLE n 1.2 0.02 1.34 0.01",
            """\
CG CD1 SINGLE n ? ? ? ?
CD1 CE1 DOUBLE n ? ? ? ?
CE1 CZ SINGLE n ? ? ? ?
CZ CE2 DOUBLE n ? ? ? ?
CE2 CD2 SINGLE n ? ? ? ?
CD2 CG DOUBLE n ? ? ? ?
CD1 HD1 SINGLE n 0.96 0.02 1.08 0.02
CD2 HD2 SINGLE n ? ? ? ?
CE1 HE1 SINGLE n ? ? ? ?
CE2 HE2 SINGLE n ? ? ? ?
CZ HZ SINGLE n ? ? ? ?""",
        )
    )
    supplied = HydrogenLengthReference(
        phe, component_id="PHE", source="synthetic Kekule"
    )
    structure = read_structure(Path(__file__).parents[1] / "fixtures/pdb/1aho.pdb")
    hd1 = AtomRef(ResidueId("A", 15), "HD1")
    (result,) = analyze_hydrogen_lengths(
        structure, references=(supplied,), scope=AtomSetScope((hd1,))
    )
    assert result.parent == AtomRef(hd1.residue_id, "CD1")
    assert result.status is HydrogenLengthStatus.WITHIN_REFERENCE


def test_histidine_requires_realized_tautomer_bond_orders():
    his = (
        REFERENCE_CIF.replace("comp_CYS", "comp_HIS")
        .replace("SG S 0", "CG C 0\nND1 N 0\nCE1 C 0\nNE2 N 0\nCD2 C 0")
        .replace("HG H 0", "HD1 H 0\nHE1 H 0\nHD2 H 0")
        .replace("CB SG SINGLE n ? ? ? ?", "CB CG SINGLE n ? ? ? ?")
        .replace(
            "SG HG SINGLE n 1.2 0.02 1.34 0.01",
            """\
CG ND1 SINGLE n ? ? ? ?
ND1 CE1 SINGLE n ? ? ? ?
CE1 NE2 DOUBLE n ? ? ? ?
NE2 CD2 SINGLE n ? ? ? ?
CD2 CG DOUBLE n ? ? ? ?
ND1 HD1 SINGLE n 0.90 0.02 1.02 0.02
CE1 HE1 SINGLE n ? ? ? ?
CD2 HD2 SINGLE n ? ? ? ?""",
        )
    )
    supplied = HydrogenLengthReference(his, component_id="HIS", source="synthetic HID")
    raw = read_structure(Path(__file__).parents[1] / "fixtures/pdb/1aho.pdb")
    rid = ResidueId("A", 54)
    hd1 = AtomRef(rid, "HD1")
    c = raw.constitution
    attachments = (
        ("ND1", "HD1", 1),
        ("CE1", "HE1", 1),
        ("CD2", "HD2", 1),
    )

    def with_bond_orders(double_order: int) -> ProteinStructure:
        return with_topology(
            raw,
            bonds=tuple(
                TopologyBond(
                    c.atom_index(AtomRef(rid, a)), c.atom_index(AtomRef(rid, b)), order
                )
                for a, b, order in (
                    *attachments,
                    ("CE1", "NE2", double_order),
                    ("CD2", "CG", double_order),
                )
            ),
        )

    (realized,) = analyze_hydrogen_lengths(
        with_bond_orders(2), references=(supplied,), scope=AtomSetScope((hd1,))
    )
    (placeholder,) = analyze_hydrogen_lengths(
        with_bond_orders(1), references=(supplied,), scope=AtomSetScope((hd1,))
    )
    assert realized.status is HydrogenLengthStatus.WITHIN_REFERENCE
    assert placeholder.status is HydrogenLengthStatus.UNASSESSED
    assert placeholder.comparison is None


@pytest.mark.parametrize("sigma", ["0", "-0.01", "nan", "inf", "0.30"])
def test_invalid_restraint_width_identifies_input_error(sigma):
    with pytest.raises(ValueError, match="interval"):
        reference(REFERENCE_CIF.replace("1.2 0.02", f"1.2 {sigma}"))
