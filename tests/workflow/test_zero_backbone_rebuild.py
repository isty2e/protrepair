"""Opt-in backbone coordinate replacement through the public workflow."""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import gemmi
import pytest
from tests.support.request_builders import c_terminal_oxt_requested_goals

from protrepair import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.geometry import TetrahedralCenter, Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.ingress_policy import OccupancyPolicy, StructureNormalizationPolicy
from protrepair.scope import WholeStructureScope
from protrepair.state import BackboneHeavyAtomCompletenessState, HydrogenCoverageState
from protrepair.structure import AtomRef, FileFormat, ResidueId
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.provenance import StructureIngress
from protrepair.structure.topology import BondProvenance, StructureTopology
from protrepair.transformer.completion.backbone_rebuild import (
    rebuild_selected_backbones,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    PolymerChemicalSite,
    SiteProtonation,
    SiteProtonationReason,
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.contracts.peptide_capping import (
    PeptideCappingRequest,
    PeptideCapStatus,
    PeptideEnd,
    PeptidePort,
)

pytestmark = pytest.mark.workflow
_PDB = Path(__file__).parents[1] / "fixtures/pdb/span-reconstruction/1ubq.pdb"
_REQUEST = WorkflowTransformRequests(rebuild_zero_occupancy_backbone=True)
_MASK = MappingProxyType(
    {
        3: frozenset(("N", "CA")),
        5: frozenset(("C",)),
        8: frozenset(("CA", "C")),
        10: frozenset(("C",)),
    }
)


def _input(shift: float, *, break_after: int | None = None) -> str:
    rows: list[str] = []
    for row in _PDB.read_text().splitlines():
        if not row.startswith("ATOM  "):
            continue
        n = int(row[22:26])
        if not 3 <= n <= 13:
            continue
        if n == break_after and rows and rows[-1][22:26] != row[22:26]:
            rows.append("TER")
        atom = row[12:16].strip()
        if atom in _MASK.get(n, ()):
            xyz = "".join(f"{float(row[i : i + 8]) + shift:8.3f}" for i in (30, 38, 46))
            row = row[:30] + xyz + "  0.00" + row[60:]
        rows.append(row)
    return "\n".join([*rows, "TER", "END", ""])


def _ref(n: int, atom: str) -> AtomRef:
    return AtomRef(ResidueId("A", n), atom)


def _events(result):
    return {
        e.residue_id: set(e.atom_names)
        for e in result.repairs
        if e.kind is RepairEventKind.BACKBONE_ATOMS_REBUILT
    }


def _geometry(structure, ref):
    return structure.geometry.atom_geometry(structure.constitution.atom_index(ref))


@pytest.mark.parametrize("fmt", (FileFormat.PDB, FileFormat.MMCIF))
def test_rebuild_replaces_selected_coordinates_only_and_survives_reread(fmt):
    source = read_structure_string(_input(6), FileFormat.PDB)
    ordinary = process_structure(source)
    assert not _events(ordinary)
    assert ordinary.structure.geometry == source.geometry

    result = process_structure(source, transform_requests=_REQUEST)
    expected = {ResidueId("A", n): names for n, names in _MASK.items() if n != 3}
    assert _events(result) == expected
    assert {
        i.residue_id
        for i in result.issues
        if i.kind is ValidationIssueKind.BACKBONE_REBUILD_REFUSED
    } == {ResidueId("A", 3)}
    assert result.structure.constitution == source.constitution
    assert result.structure.topology == source.topology
    moved = {AtomRef(rid, k) for rid, names in expected.items() for k in names}
    for site in source.constitution.iter_residues():
        for atom in site.atom_sites:
            ref = AtomRef(site.residue_id, atom.name)
            before, after = _geometry(source, ref), _geometry(result.structure, ref)
            if ref in moved:
                assert before.position.distance_to(after.position) > 5
                assert after == before.with_position(after.position)
            else:
                assert before == after
    assert {
        o.source_identity.atom_ref
        for o in result.structure.provenance.original_backbone_coordinates
    } == moved
    restored = read_structure_string(write_structure_string(result.structure, fmt), fmt)
    repeated = process_structure(restored, transform_requests=_REQUEST)
    assert not _events(repeated)
    assert not any(
        i.kind is ValidationIssueKind.BACKBONE_REBUILD_REFUSED
        and i.residue_id in expected
        for i in repeated.issues
    )
    for ref in moved:
        assert _geometry(restored, ref).occupancy == 0
        assert (
            _geometry(restored, ref).position.distance_to(
                _geometry(result.structure, ref).position
            )
            <= 0.002
        )


def test_placeholder_coordinates_supply_neither_anchor_nor_completion_credit():
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=BackboneHeavyAtomCompletenessState.COMPLETE
    )
    a, b = (read_structure_string(_input(x), FileFormat.PDB) for x in (-8, 14))
    ordinary = process_structure(a, requested_goals=(goal,)).requested_goal_report
    assert ordinary is not None and ordinary.outcomes[0].is_satisfied()
    left, right = (
        process_structure(s, requested_goals=(goal,), transform_requests=_REQUEST)
        for s in (a, b)
    )
    assert left.requested_goal_report is not None
    assert not left.requested_goal_report.outcomes[0].is_satisfied()
    assert _events(left) == _events(right)
    for rid, names in _events(left).items():
        for name in names:
            ref = AtomRef(rid, name)
            assert _geometry(left.structure, ref) == _geometry(right.structure, ref)


def test_chain_break_withholds_only_the_carbonyl_that_needed_its_neighbor():
    source = read_structure_string(_input(9, break_after=6), FileFormat.PDB)
    result = process_structure(source, transform_requests=_REQUEST)
    assert ResidueId("A", 5) not in _events(result)
    assert _geometry(result.structure, _ref(5, "C")) == _geometry(source, _ref(5, "C"))
    assert _events(result)[ResidueId("A", 8)] == {"C", "CA"}


def test_rebuilt_coordinates_follow_a_rigid_frame_but_not_a_displaced_anchor():
    source = read_structure_string(_input(9), FileFormat.PDB)
    result, availability = rebuild_selected_backbones(
        source, build_default_component_library()
    )
    assert availability is not None
    s = result.structure
    rid = ResidueId("A", 8)
    transformed = []
    for site in s.constitution.iter_residues():
        geo = s.residue_geometry(s.constitution.residue_index(site.residue_id))
        rotated = []
        for atom in site.atom_sites:
            old = geo.atom_geometry(atom.name)
            x, y, z = old.position
            rotated.append((atom.name, old.with_position(Vec3(-y + 3, x + 5, z - 4))))
        transformed.append((site.residue_id, geo.with_atom_geometries(rotated)))
    same = s.with_updated_residue_geometries(transformed)
    assert not availability.unresolved(same, rid)
    neighbor = _ref(9, "N")
    geo = same.residue_geometry(same.constitution.residue_index(neighbor.residue_id))
    n = geo.atom_geometry("N")
    x, y, z = n.position
    changed = same.with_updated_residue_geometries(
        (
            (
                neighbor.residue_id,
                geo.with_atom_geometry("N", n.with_position(Vec3(x + 1, y, z))),
            ),
        )
    )
    assert availability.unresolved(changed, rid)
    assert not availability.unresolved(changed, ResidueId("A", 5))


def test_labeled_peptide_link_does_not_depend_on_the_distance_to_a_zero_carbon():
    fitted = []
    for shift in (0.0, 12.0):
        text = (
            "SEQRES   1 A   11  ILE PHE VAL LYS THR LEU THR GLY LYS THR ILE\n"
            + _input(shift)
        )
        raw = gemmi.read_pdb_string(text)
        raw.setup_entities()
        raw.assign_label_seq_id()
        doc = raw.make_mmcif_document()
        auth = doc.sole_block().find_loop("_atom_site.auth_seq_id")
        for i in range(len(auth)):
            if int(auth[i]) >= 6:
                auth[i] = str(int(auth[i]) + 20)
        source = read_structure_string(doc.as_string(), FileFormat.MMCIF)
        carbon, nitrogen = _ref(5, "C"), _ref(26, "N")
        bond = source.topology.bond_between(
            source.constitution.atom_index(carbon),
            source.constitution.atom_index(nitrogen),
        )
        assert (bond is not None) == (not shift)
        result = process_structure(source, transform_requests=_REQUEST)
        assert not any(
            i.kind is ValidationIssueKind.BACKBONE_REBUILD_REFUSED
            and i.residue_id == carbon.residue_id
            for i in result.issues
        )
        fitted.append(_geometry(result.structure, carbon).position)
    assert fitted[0].distance_to(fitted[1]) <= 0.002


def test_programmatic_carrier_needs_an_explicit_bond_if_it_has_no_source_path():
    s = read_structure_string(_input(7), FileFormat.PDB)
    bare = s.with_provenance(
        replace(s.provenance, ingress=StructureIngress(FileFormat.PDB))
    )
    assert not _events(process_structure(bare, transform_requests=_REQUEST))
    edge = bare.topology.bond_between(
        bare.constitution.atom_index(_ref(5, "C")),
        bare.constitution.atom_index(_ref(6, "N")),
    )
    assert edge is not None
    topology = StructureTopology(
        constitution=bare.constitution,
        atom_topologies=bare.topology.atom_topologies,
        bonds=tuple(
            replace(b, provenance=BondProvenance.SOURCE_EXPLICIT) if b == edge else b
            for b in bare.topology.bonds
        ),
    )
    explicit = ProteinStructure.from_payload(
        constitution=bare.constitution,
        geometry=bare.geometry,
        topology=topology,
        provenance=bare.provenance,
    )
    assert _events(process_structure(explicit, transform_requests=_REQUEST)) == {
        ResidueId("A", 5): {"C"}
    }


def test_uncertain_conect_without_seqres_and_retained_ligand_each_veto_a_single_fit():
    text = _input(6)
    line = next(
        row
        for row in _PDB.read_text().splitlines()
        if row.startswith("ATOM") and int(row[22:26]) == 5 and row[12:16].strip() == "C"
    )
    ligand = "HETATM 9000 ZN    ZN Z 501    " + line[30:54] + "  1.00 20.00          ZN"
    for record in (f"CONECT{int(line[6:11]):5d} 9999", ligand):
        source = read_structure_string(
            text.replace("END\n", record + "\nEND\n"), FileFormat.PDB
        )
        result = process_structure(
            source,
            ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
            transform_requests=_REQUEST,
        )
        assert ResidueId("A", 5) not in _events(result)
        assert ResidueId("A", 8) in _events(result)


def test_only_the_selected_alternate_is_a_rebuild_target():
    rows = _input(6).splitlines()
    idx = next(
        i
        for i, row in enumerate(rows)
        if row.startswith("ATOM") and int(row[22:26]) == 5 and row[12:16].strip() == "C"
    )
    row = rows[idx]
    positive = next(
        r
        for r in _PDB.read_text().splitlines()
        if r.startswith("ATOM") and r[6:11] == row[6:11]
    )
    rows[idx : idx + 1] = [
        row[:16] + "A" + row[17:],
        positive[:6] + " 9000" + positive[11:16] + "B" + positive[17:],
    ]
    text = "\n".join(rows) + "\n"
    for policy in (OccupancyPolicy.HIGHEST, OccupancyPolicy.LOWEST):
        source = read_structure_string(
            text,
            FileFormat.PDB,
            policy=StructureNormalizationPolicy(occupancy_policy=policy),
        )
        result = process_structure(source, transform_requests=_REQUEST)
        chosen = _geometry(result.structure, _ref(5, "C"))
        assert (ResidueId("A", 5) in _events(result)) == (
            policy is OccupancyPolicy.LOWEST
        )
        assert chosen.altloc == ("A" if policy is OccupancyPolicy.LOWEST else "B")


def test_unvalidated_oxygen_is_preserved_without_suppressing_carbon_repairs():
    text = _input(6)
    rows = text.splitlines()
    i = next(
        i
        for i, row in enumerate(rows)
        if row.startswith("ATOM")
        and int(row[22:26]) == 12
        and row[12:16].strip() == "O"
    )
    rows[i] = rows[i][:30] + "  99.000" * 3 + "  0.00" + rows[i][60:]
    source = read_structure_string("\n".join(rows) + "\n", FileFormat.PDB)
    result = process_structure(source, transform_requests=_REQUEST)
    assert ResidueId("A", 8) in _events(result)
    assert _geometry(result.structure, _ref(12, "O")) == _geometry(
        source, _ref(12, "O")
    )
    assert any(
        i.residue_id == ResidueId("A", 12)
        and i.kind is ValidationIssueKind.BACKBONE_REBUILD_REFUSED
        for i in result.issues
    )


def test_a_shared_peptide_plane_cannot_be_built_one_residue_at_a_time():
    rows = _input(6).splitlines()
    for i, row in enumerate(rows):
        if row.startswith("ATOM") and (int(row[22:26]), row[12:16].strip()) in {
            (10, "O"),
            (11, "N"),
        }:
            rows[i] = row[:30] + "  88.000" * 3 + "  0.00" + row[60:]
    source = read_structure_string("\n".join(rows) + "\n", FileFormat.PDB)
    result = process_structure(source, transform_requests=_REQUEST)
    for n, names in ((10, ("C", "O")), (11, ("N",))):
        assert any(
            i.residue_id == ResidueId("A", n) and "cross-junction" in i.message
            for i in result.issues
        )
        for name in names:
            ref = _ref(n, name)
            assert _geometry(source, ref) == _geometry(result.structure, ref)
    assert ResidueId("A", 8) in _events(result)


@pytest.mark.parametrize("charge", ["  ", "1+"])
def test_solitary_zero_glycine_isotope_is_not_replaced_by_an_invented_hydrogen(charge):
    rows = _input(6).splitlines()
    idx = next(
        i
        for i, row in enumerate(rows)
        if row.startswith("ATOM")
        and int(row[22:26]) == 10
        and row[12:16].strip() == "CA"
    )
    row = rows[idx]
    rows[idx] = row[:30] + "  55.000" * 3 + "  0.00" + row[60:]
    rows.insert(
        idx + 1,
        "ATOM   9001  DA2"
        + row[16:30]
        + "  88.000" * 3
        + "  0.00"
        + row[60:76]
        + " D"
        + charge,
    )
    source = read_structure_string("\n".join(rows) + "\n", FileFormat.PDB)
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    )
    result = process_structure(
        source, transform_requests=_REQUEST, requested_goals=(goal,)
    )
    ref = _ref(10, "DA2")
    if charge.strip():
        assert ResidueId("A", 10) not in _events(result)
        assert _geometry(result.structure, ref) == _geometry(source, ref)
        assert (
            result.structure.topology.formal_charge(
                result.structure.constitution.atom_index(ref)
            )
            == 1
        )
    else:
        assert ResidueId("A", 10) in _events(result)
        (held,) = result.structure.provenance.withheld_backbone_hydrogens
        assert held.original.source_identity.atom_ref == ref and held.parent == _ref(
            10, "CA"
        )
        assert held.original.geometry == _geometry(source, ref)
        site = result.structure.constitution.residue_or_ligand(ref.residue_id)
        assert site is not None and not any(a.is_hydrogen() for a in site.atom_sites)


def test_refused_rows_are_not_anchors_for_requested_hydrogens():
    source = read_structure_string(_input(-14), FileFormat.PDB)
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    )
    result = process_structure(
        source, requested_goals=(goal,), transform_requests=_REQUEST
    )
    refused = result.structure.constitution.residue_or_ligand(ResidueId("A", 3))
    repaired = result.structure.constitution.residue_or_ligand(ResidueId("A", 8))
    assert refused is not None and repaired is not None
    assert not any(atom.is_hydrogen() for atom in refused.atom_sites)
    assert any(atom.is_hydrogen() for atom in repaired.atom_sites)
    assert result.requested_goal_report is not None
    assert not result.requested_goal_report.outcomes[0].is_satisfied()


def test_an_unresolved_nitrogen_does_not_rotate_a_neighboring_hydroxyl():
    s = read_structure_string(_input(40), FileFormat.PDB)
    goal = requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    )

    def hydroxyl(structure):
        result = process_structure(
            structure, requested_goals=(goal,), transform_requests=_REQUEST
        )
        return _geometry(result.structure, _ref(7, "HG1")).position

    position = hydroxyl(s)
    rid = ResidueId("A", 3)
    geo = s.residue_geometry(s.constitution.residue_index(rid))
    nitrogen = geo.atom_geometry("N")
    overlapping = s.with_updated_residue_geometries(
        ((rid, geo.with_atom_geometry("N", nitrogen.with_position(position))),)
    )
    assert hydroxyl(overlapping) == position


def test_unresolved_terminal_anchors_do_not_authorize_oxt_protonation_or_caps():
    n_end, c_end = ResidueId("A", 3), ResidueId("A", 13)
    rows = _input(7).splitlines()
    for i, row in enumerate(rows):
        if (
            row.startswith("ATOM")
            and int(row[22:26]) == 13
            and row[12:16].strip() == "O"
        ):
            rows[i] = row[:30] + "  85.000" * 3 + "  0.00" + row[60:]
    text = "\n".join(rows) + "\n"
    source = read_structure_string(text, FileFormat.PDB)
    result = process_structure(
        source,
        requested_goals=c_terminal_oxt_requested_goals(c_end),
        transform_requests=WorkflowTransformRequests(
            rebuild_zero_occupancy_backbone=True,
            site_protonations=(
                SiteProtonation(n_end, PolymerChemicalSite.BACKBONE_N, site_charge=1),
            ),
        ),
    )
    assert result.requested_goal_report is not None
    assert not result.requested_goal_report.outcomes[0].is_satisfied()
    assert (
        result.structure.constitution.resolve_atom_site(AtomRef(c_end, "OXT")) is None
    )
    assert result.site_protonation_report is not None
    (outcome,) = result.site_protonation_report.outcomes
    assert outcome.reason is SiteProtonationReason.PHYSICAL_CONSTRAINT
    assert ResidueId("A", 8) in _events(result)

    seq = "SEQRES   1 A   13  ALA ILE PHE VAL LYS THR LEU THR GLY LYS THR ILE MET\n"
    capped = process_structure(
        read_structure_string(seq + text, FileFormat.PDB),
        transform_requests=WorkflowTransformRequests(
            rebuild_zero_occupancy_backbone=True,
            peptide_capping=PeptideCappingRequest.only(
                (PeptidePort(n_end, PeptideEnd.N), PeptidePort(c_end, PeptideEnd.C))
            ),
        ),
    )
    assert capped.peptide_capping_report is not None
    assert {out.port for out in capped.peptide_capping_report.outcomes} == {
        PeptidePort(n_end, PeptideEnd.N),
        PeptidePort(c_end, PeptideEnd.C),
    }
    assert all(
        out.status is PeptideCapStatus.REFUSED
        and out.attachment is None
        and out.reason == "unoccupied_anchor"
        for out in capped.peptide_capping_report.outcomes
    )
    assert not capped.structure.constitution.ligands
    assert ResidueId("A", 8) in _events(capped)


@pytest.mark.parametrize("element", ["D", "T"])
@pytest.mark.parametrize("occupied", [False, True])
def test_dependent_isotope_moves_only_if_it_was_zero(element, occupied):
    rows = _input(6).splitlines()
    index = next(
        i
        for i, row in enumerate(rows)
        if row.startswith("ATOM")
        and int(row[22:26]) == 8
        and row[12:16].strip() == "CA"
    )
    anchor = rows[index]
    isotope = (
        anchor[:6]
        + " 9000"
        + f" {element + 'A':>3} "
        + anchor[16:30]
        + f"{99.0:8.3f}" * 3
        + ("  1.00" if occupied else "  0.00")
        + anchor[60:76]
        + f"{element:>2}"
    )
    rows.insert(index + 1, isotope)
    source = read_structure_string("\n".join(rows) + "\n", FileFormat.PDB)
    result = process_structure(source, transform_requests=_REQUEST)
    ref = _ref(8, element + "A")
    before, after = _geometry(source, ref), _geometry(result.structure, ref)
    if occupied:
        assert after == before
        assert ref.residue_id not in _events(result)
    else:
        assert _events(result)[ref.residue_id] == {"CA", "C", ref.atom_name}
        assert before.with_position(after.position) == after
        assert after.position.distance_to(
            _geometry(result.structure, _ref(8, "CA")).position
        ) == pytest.approx(1.09)
    assert ResidueId("A", 5) in _events(result)


@pytest.mark.parametrize("names", [("DA2", "TA3"), ("1DA", "2TA"), ("DA1", "TA2")])
def test_glycine_pair_keeps_its_names_isotopes_and_handedness(names):
    rows = _input(6).splitlines()
    idx = next(
        i
        for i, row in enumerate(rows)
        if row.startswith("ATOM")
        and int(row[22:26]) == 10
        and row[12:16].strip() == "CA"
    )
    row = rows[idx]
    rows[idx] = row[:30] + "  55.000" * 3 + "  0.00" + row[60:]
    additions = []
    for serial, (name, element) in enumerate(zip(names, ("D", "T"), strict=True), 9001):
        additions.append(
            f"ATOM  {serial:5d} {name:>4}"
            + row[16:30]
            + "  88.000" * 3
            + "  0.00"
            + row[60:76]
            + f"{element:>2}"
        )
    rows[idx + 1 : idx + 1] = additions
    source = read_structure_string("\n".join(rows) + "\n", FileFormat.PDB)
    result = process_structure(source, transform_requests=_REQUEST)
    assert _events(result)[ResidueId("A", 10)] == {"C", "CA", *names}

    def position(name):
        return _geometry(result.structure, _ref(10, name)).position

    pair = TetrahedralCenter(position("C"), position("N"), position("CA")).pair()
    for name, vertex, element in zip(names, pair, ("D", "T"), strict=True):
        ref = _ref(10, name)
        assert _geometry(result.structure, ref) == _geometry(source, ref).with_position(
            vertex
        )
        atom = result.structure.constitution.resolve_atom_site(ref)
        assert atom is not None and atom.element == element


@pytest.mark.parametrize("use_sidechains", [False, True])
def test_deferred_omission_refuses_charge_locally_without_undoing_a_backbone(
    tmp_path, use_sidechains
):
    rows = _input(6).splitlines()
    for i, row in enumerate(rows):
        if not row.startswith("ATOM"):
            continue
        key = (int(row[22:26]), row[12:16].strip())
        if key in {(5, "CG2"), (8, "CD1")}:
            rows[i] = row[:54] + "  0.00" + row[60:]
        elif key == (8, "CD2"):
            rows[i] = row[:78] + "1+"
    path = tmp_path / "mixed.pdb"
    path.write_text("\n".join(rows) + "\n")
    ingress = StructureIngressOptions(
        omit_zero_occupancy_sidechain_heavy_atoms=use_sidechains
    )
    result = process_structure(path, ingress=ingress, transform_requests=_REQUEST)
    assert {ResidueId("A", 5), ResidueId("A", 8)} <= _events(result).keys()
    assert result.structure.constitution.resolve_atom_site(_ref(8, "CD1")) is not None
    assert (
        result.structure.constitution.resolve_atom_site(_ref(5, "CG2")) is None
    ) == use_sidechains
    retained = [
        i
        for i in result.issues
        if i.kind is ValidationIssueKind.ZERO_OCCUPANCY_SIDECHAIN_RETAINED
    ]
    assert [i.residue_id for i in retained] == (
        [ResidueId("A", 8)] if use_sidechains else []
    )
    if use_sidechains:
        assert "formal charge" in retained[0].message
        omissions = result.structure.provenance.ingress.zero_occupancy_omissions
        assert omissions is not None
        assert _ref(5, "CG2") in {r.source_identity.atom_ref for r in omissions}
        observed = result.structure.provenance.ingress.observation
        assert observed is not None
        i = observed.constitution.atom_index(_ref(5, "C"))
        assert (
            observed.geometry.position(i)
            != _geometry(result.structure, _ref(5, "C")).position
        )
