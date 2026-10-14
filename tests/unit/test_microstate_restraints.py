"""Chemical applicability of restraints, including actual native FF construction."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.microstate.catalog import (
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateChoice,
    MicrostateSelectionBasis,
)
from protrepair.chemistry.restraint.structure import StructureRestraints
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.diagnostics.geometry import detect_heavy_geometry
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import StructureTopology
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.transformer.continuous import rdkit as native
from protrepair.transformer.local import LocalScopeSpec

HIS = ResidueId("A", 42)
FIXTURE = Path(__file__).parents[1] / "fixtures/pdb/refinement/1bkr_his42_local.pdb"


def _source():
    # Experimental heavy coordinates, without an asserted protonation state.
    text = "\n".join(
        line
        for line in FIXTURE.read_text().splitlines()
        if line.startswith("ATOM") and line[22:26].strip() == "42"
    )
    return read_structure_string(text + "\nEND\n", FileFormat.PDB)


def _graph(counts, resonance=0):
    catalog = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert catalog is not None
    return [
        graph
        for graph in catalog.candidates
        if tuple(graph.atom(name).hydrogens for name in ("ND1", "NE2")) == counts
    ][resonance]


def _prepared(counts=(0, 1), resonance=0):
    source = _source()
    choice = AppliedMicrostateChoice(
        HIS, "HIS", _graph(counts, resonance), MicrostateSelectionBasis.DEFAULT
    )
    selected = source.with_provenance(
        replace(source.provenance, microstate_choices=(choice,))
    )
    return add_hydrogens(selected).structure


def _restraints(structure, library=None, components=None):
    template = StructureRestraints(
        structure,
        build_default_component_library() if components is None else components,
        build_default_restraint_library() if library is None else library,
    ).for_residue(HIS)
    assert template is not None
    return template


def _ce_ne(template):
    return next(
        (
            target
            for target in template.bond_targets
            if {target.atom_name_1, target.atom_name_2} == {"CE1", "NE2"}
        ),
        None,
    )


@pytest.mark.parametrize(
    "counts,distance,esd,angle,angle_esd",
    (
        ((1, 0), 1.316, 0.009, 105.1, 0.5),
        ((0, 1), 1.335, 0.014, 109.1, 0.8),
        ((1, 1), 1.316, 0.009, 105.9, 0.5),
    ),
)
def test_realized_histidine_selects_published_state_specific_targets(
    counts, distance, esd, angle, angle_esd
):
    structure = _prepared(counts)
    template = _restraints(structure)
    target = _ce_ne(template)
    assert target is not None
    assert (target.target_distance_angstrom, target.esd_angstrom) == (distance, esd)
    target_angle = next(
        target
        for target in template.angle_targets
        if target.center_atom_name == "CG"
        and {target.atom_name_1, target.atom_name_2} == {"ND1", "CD2"}
    )
    assert (target_angle.target_angle_degrees, target_angle.esd_degrees) == (
        angle,
        angle_esd,
    )
    assert len(template.bond_targets) == 10
    assert len(template.angle_targets) == 12
    assert _restraints(structure.without_hydrogens()) == template


def test_unselected_or_unrealized_state_does_not_borrow_cationic_dictionary():
    source = _source()
    assert _ce_ne(_restraints(source)) is None
    assert len(_restraints(source).bond_targets) == 4
    choice = AppliedMicrostateChoice(
        HIS, "HIS", _graph((1, 1)), MicrostateSelectionBasis.DEFAULT
    )
    desired_only = source.with_provenance(
        replace(source.provenance, microstate_choices=(choice,))
    )
    assert _ce_ne(_restraints(desired_only)) is None


@pytest.mark.parametrize("prepared", (False, True))
def test_ring_angle_without_target_still_receives_broad_diagnostics(prepared):
    source = _prepared() if prepared else _source()
    nd = source.geometry.position(source.constitution.atom_index(AtomRef(HIS, "ND1")))
    ne = source.geometry.position(source.constitution.atom_index(AtomRef(HIS, "NE2")))
    ce = source.constitution.atom_index(AtomRef(HIS, "CE1"))
    position = Vec3.from_iterable(
        nd.to_array() + 0.25 * (ne.to_array() - nd.to_array())
    )
    geometry = source.geometry.with_atom_geometry(
        ce, source.geometry.atom_geometry(ce).with_position(position)
    )
    changed = type(source).from_payload(
        constitution=source.constitution,
        geometry=geometry,
        topology=source.topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    report = detect_heavy_geometry(
        changed, component_library=build_default_component_library()
    )
    ring = [
        outlier
        for outlier in report.bond_angle_outliers
        if outlier.center_atom_name == "CE1"
        and {outlier.atom_name_1, outlier.atom_name_2} == {"ND1", "NE2"}
    ]
    assert len(ring) == 1
    assert ring[0].observed_angle_degrees == pytest.approx(180.0)
    assert ring[0].restraint_backed is prepared
    if not prepared:
        assert (ring[0].minimum_angle_degrees, ring[0].maximum_angle_degrees) == (
            85.0,
            145.0,
        )
        assert _ce_ne(_restraints(changed)) is None

    custom = AngleRestraintTarget("NE2", "CE1", "ND1", 180.0, 1.0)
    library = build_default_restraint_library()
    dictionary = library.require("HIS")
    library = library.with_template(
        replace(dictionary, angle_targets=(*dictionary.angle_targets, custom))
    )
    report = detect_heavy_geometry(
        changed,
        component_library=build_default_component_library(),
        restraint_library=library,
    )
    assert not any(
        outlier.center_atom_name == "CE1"
        and {outlier.atom_name_1, outlier.atom_name_2} == {"ND1", "NE2"}
        for outlier in report.bond_angle_outliers
    )


def test_cation_resonance_encoding_has_one_restraint_identity():
    assert _restraints(_prepared((1, 1), 0)) == _restraints(_prepared((1, 1), 1))
    template = build_default_restraint_library().require("HIS")
    duplicate = replace(template.sidechain_variants[2], graph=_graph((1, 1), 1))
    with pytest.raises(ValueError, match="must not repeat"):
        replace(template, sidechain_variants=(*template.sidechain_variants, duplicate))


def test_changed_bond_order_invalidates_snapshot_applicability():
    source = _prepared()
    pair = frozenset(
        source.constitution.atom_index(AtomRef(HIS, name)) for name in ("CE1", "NE2")
    )
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=tuple(
            replace(bond, order=2) if frozenset(bond.endpoint_pair()) == pair else bond
            for bond in source.topology.bonds
        ),
    )
    changed = type(source).from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    assert _ce_ne(_restraints(source)) is not None
    assert _ce_ne(_restraints(changed)) is None


def test_static_custom_dictionary_and_partial_override_remain_authoritative():
    source = _prepared()
    library = build_default_restraint_library()
    custom = BondRestraintTarget("NE2", "CE1", 1.4, 0.02)
    static = ResidueRestraintTemplate("HIS", (custom,), ())
    assert _restraints(source, library.with_template(static)) is static
    default = library.require("HIS")
    partial = replace(default, bond_targets=(*default.bond_targets, custom))
    resolved = _restraints(source, library.with_template(partial))
    assert _ce_ne(resolved) is custom
    assert len(resolved.bond_targets) == 10


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("hydrogen_name", ("HE2", "Q17"))
def test_source_isotope_evidence_survives_mmcif_roundtrip(isotope, hydrogen_name):
    prepared = _prepared()
    text = write_structure_string(prepared, FileFormat.PDB)
    lines = [
        line[:76] + f"{isotope:>2}" + line[78:]
        if line.startswith(("ATOM", "HETATM")) and line[76:78].strip() == "H"
        else line
        for line in text.splitlines()
    ]
    reread = read_structure_string("\n".join(lines) + "\n", FileFormat.PDB)
    target = _ce_ne(_restraints(reread))
    assert target is not None
    assert target.target_distance_angstrom == 1.335
    # A nonstandard H name needs typed covalent attachment, not untyped CONECT.
    cif_lines = write_structure_string(prepared, FileFormat.MMCIF).splitlines()
    cif_text = "\n".join(
        line.replace(" H ", f" {isotope} ", 1) if line.startswith("ATOM ") else line
        for line in cif_lines
    ).replace("HE2", hydrogen_name)
    roundtrip = read_structure_string(cif_text, FileFormat.MMCIF)
    assert _restraints(roundtrip) == _restraints(reread)


@pytest.mark.parametrize("counts", ((1, 0), (0, 1), (1, 1)))
def test_native_uff_uses_same_dictionary_and_has_finite_energy(counts, monkeypatch):
    source = _prepared(counts)
    library = build_default_restraint_library()
    problem = build_continuous_relaxation_problem(
        source,
        LocalScopeSpec.from_residues((HIS,)),
        component_library=build_default_component_library(),
    )
    molecule, mapping = native.build_rdkit_molecule(problem)
    recorded = []
    add_distance = native.add_rdkit_distance_constraint

    def record(force_field, **kwargs):
        recorded.append(kwargs)
        return add_distance(force_field, **kwargs)

    monkeypatch.setattr(native, "add_rdkit_distance_constraint", record)
    force_field = native.build_rdkit_force_field(
        molecule,
        problem=problem,
        restraint_library=library,
        rdkit_atom_index_by_structure_atom_index=mapping,
    )
    force_field.Initialize()
    assert np.isfinite(force_field.CalcEnergy())
    assert np.isfinite(force_field.CalcGrad()).all()
    indices = {
        mapping[source.constitution.atom_index(AtomRef(HIS, n))] for n in ("CE1", "NE2")
    }
    constraint = next(
        c for c in recorded if {c["atom_index_1"], c["atom_index_2"]} == indices
    )
    target = _ce_ne(_restraints(source))
    assert target is not None
    assert (
        constraint["minimum_distance"] + constraint["maximum_distance"]
    ) / 2 == pytest.approx(target.target_distance_angstrom)


def _with_ce_ne_distance(source, distance):
    geometry = source.residue_geometry(source.constitution.residue_index(HIS))
    first, second = (geometry.position(name) for name in ("CE1", "NE2"))
    delta = np.asarray(tuple(second)) - np.asarray(tuple(first))
    point = Vec3.from_iterable(
        np.asarray(tuple(first)) + delta * distance / np.linalg.norm(delta)
    )
    return source.with_updated_residue_geometries(
        (
            (
                HIS,
                geometry.with_atom_geometry(
                    "NE2", replace(geometry.atom_geometry("NE2"), position=point)
                ),
            ),
        )
    )


def test_diagnostics_do_not_reject_neutral_epsilon_using_cationic_targets():
    source = _with_ce_ne_distance(_prepared(), 1.37)
    report = detect_heavy_geometry(
        source, component_library=build_default_component_library()
    )
    assert not any(
        {v.atom_name_1, v.atom_name_2} == {"CE1", "NE2"}
        for v in report.bond_length_outliers
    )
    old = ResidueRestraintTemplate(
        "HIS", (BondRestraintTarget("CE1", "NE2", 1.321, 0.01),), ()
    )
    old_report = detect_heavy_geometry(
        source,
        component_library=build_default_component_library(),
        restraint_library=build_default_restraint_library().with_template(old),
    )
    assert any(
        {v.atom_name_1, v.atom_name_2} == {"CE1", "NE2"}
        for v in old_report.bond_length_outliers
    )


def test_catastrophic_guard_uses_selected_target_without_reselecting_from_coordinates():
    source = _prepared()
    problem = build_continuous_relaxation_problem(
        source,
        LocalScopeSpec.from_residues((HIS,)),
        component_library=build_default_component_library(),
    )
    distorted = _with_ce_ne_distance(source, 2.2)
    report = native.catastrophic_bond_distortions(
        problem,
        refined_structure=distorted,
        restraint_library=build_default_restraint_library(),
    )
    target = next(v for v in report if {v.atom_name_1, v.atom_name_2} == {"CE1", "NE2"})
    assert target.target_distance_angstrom == 1.335
    assert _restraints(source) == _restraints(distorted)
