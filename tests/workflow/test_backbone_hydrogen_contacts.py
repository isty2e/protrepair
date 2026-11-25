"""Backbone placement against a fixed H in the same or next residue."""

from dataclasses import replace
from math import cos, radians, sin
from pathlib import Path

import gemmi
import pytest

from protrepair import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.clashes import detect_clashes
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure import AtomRef, FileFormat, ResidueId
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.topology import BondRelationshipType, StructureTopology
from protrepair.transformer.completion.backbone_environment import BackboneEnvironment
from protrepair.transformer.completion.backbone_rebuild import (
    rebuild_selected_backbones,
)
from protrepair.workflow.contracts import WorkflowTransformRequests

pytestmark = pytest.mark.workflow
_CORPUS = Path(__file__).parents[1] / "fixtures/corpus"


def _masked_input(
    entry, *, declared=False, pose="native", isotope="H", beta_angle=None
):
    n, h_res, parent, h_name = (
        (24, 25, "N", "H") if entry == "1div" else (40, 40, "CA", "HA")
    )
    rows = (
        row
        for row in (_CORPUS / f"pdb{entry}.ent").read_text().splitlines()
        if row.startswith("ATOM  ")
        and row[21] == "A"
        and n - 2 <= int(row[22:26]) <= n + 6
    )
    st = gemmi.read_pdb_string("\n".join((*rows, "TER", "END", "")))
    st.setup_entities()
    st.assign_label_seq_id()
    residues = {r.seqid.num: r for r in st[0]["A"]}
    carbon = next(a for a in residues[n] if a.name == "C")
    moving = carbon
    if beta_angle is not None:
        parent, h_name = "CB", "HB2"
        moving = next(a for a in residues[n] if a.name == "CA")
    heavy, h = (
        next(a for a in residues[h_res] if a.name == k) for k in (parent, h_name)
    )
    if pose == "rotated":
        ray = carbon.pos - heavy.pos
        h.pos = heavy.pos + ray * (h.pos.dist(heavy.pos) / ray.length())
    elif pose == "bent" or beta_angle is not None:
        axis = (moving.pos - heavy.pos).normalized()
        ray = gemmi.Vec3(*(h.pos - heavy.pos))
        perp = (ray - axis * axis.dot(ray)).normalized()
        angle = radians(
            beta_angle if beta_angle is not None else (85 if entry == "1div" else 82)
        )
        h.pos = heavy.pos + gemmi.Position(
            *((axis * cos(angle) + perp * sin(angle)) * ray.length())
        )
    elif pose == "coincident":
        h.pos = carbon.pos
    elif pose == "unrelated":
        next(a for a in residues[n + 3] if a.name == "H").pos = carbon.pos
    if isotope != "H":
        h.element = gemmi.Element("D")
        h.name = isotope + h_name[1:]

    def connect(i, a, j, b):
        con = gemmi.Connection()
        con.name = f"{i}{a}_{j}{b}"
        con.type, con.asu = gemmi.ConnectionType.Covale, gemmi.Asu.Same
        con.partner1, con.partner2 = (
            gemmi.AtomAddress("A", residues[k].seqid, residues[k].name, name)
            for k, name in ((i, a), (j, b))
        )
        st.connections.append(con)

    connect(n - 1, "C", n, "N")
    connect(n, "C", n + 1, "N")
    if declared:
        connect(h_res, parent, h_res, h.name)
    if pose == "other_parent":
        connect(h_res, "CB", h_res, h.name)
    if pose == "surplus":
        extra = h.clone()
        extra.name = "H2"
        extra.pos = gemmi.Position(-80, -80, -80)
        residues[h_res].add_atom(extra)
    for atom in (carbon, moving):
        atom.occ, atom.pos = 0.0, gemmi.Position(80, -80, 120)
    doc = st.make_mmcif_document()
    if isotope == "T":
        for row in doc.sole_block().find(
            "_atom_site.", ["type_symbol", "label_atom_id"]
        ):
            if row[1] == h.name:
                row[0] = "T"
    return read_structure_string(doc.as_string(), FileFormat.MMCIF)


@pytest.mark.parametrize("declared", (False, True))
@pytest.mark.parametrize(
    ("entry", "pose", "clash"),
    (
        ("1div", "native", None),
        ("1div", "rotated", "A:25.H"),
        ("1div", "bent", "A:25.H"),
        ("1div", "unrelated", "A:27.H"),
        ("1div", "other_parent", "A:25.H"),
        ("1div", "surplus", "A:25.H"),
        ("1mbe", "native", None),
        ("1mbe", "coincident", "A:40.HA"),
        ("1mbe", "rotated", "A:40.HA"),
        ("1mbe", "bent", "A:40.HA"),
    ),
)
def test_backbone_contact_uses_both_connectivity_and_geometry(
    entry, pose, clash, declared
):
    source = _masked_input(entry, declared=declared, pose=pose)
    result = process_structure(
        source,
        requested_goals=(),
        transform_requests=WorkflowTransformRequests(
            rebuild_zero_occupancy_backbone=True
        ),
    )
    rid = ResidueId("A", 24 if entry == "1div" else 40)
    edits = [
        e.atom_names
        for e in result.repairs
        if e.residue_id == rid and e.kind is RepairEventKind.BACKBONE_ATOMS_REBUILT
    ]
    refusals = [
        i.message
        for i in result.issues
        if i.residue_id == rid
        and i.kind is ValidationIssueKind.BACKBONE_REBUILD_REFUSED
    ]
    if clash is None:
        assert edits == [("C",)] and not refusals
    else:
        assert not edits and refusals == ["steric clash with " + clash]

    assert source.topology == result.structure.topology
    for site in source.constitution.iter_residues():
        for atom in site.atom_sites:
            ref = AtomRef(site.residue_id, atom.name)
            if clash is not None or ref != AtomRef(rid, "C"):
                idx = source.constitution.atom_index(ref)
                assert source.geometry.atom_geometry(
                    idx
                ) == result.structure.geometry.atom_geometry(idx)


@pytest.mark.parametrize("isotope", ("D", "T"))
def test_implicit_amide_isotope_is_not_a_steric_clash(isotope):
    source = _masked_input("1div", isotope=isotope)
    result = process_structure(
        source,
        requested_goals=(),
        transform_requests=WorkflowTransformRequests(
            rebuild_zero_occupancy_backbone=True
        ),
    )
    assert [
        (e.residue_id, e.atom_names)
        for e in result.repairs
        if e.kind is RepairEventKind.BACKBONE_ATOMS_REBUILT
    ] == [(ResidueId("A", 24), ("C",))]
    protium, _ = rebuild_selected_backbones(
        _masked_input("1div"), build_default_component_library()
    )
    carbon = AtomRef(ResidueId("A", 24), "C")
    assert result.structure.geometry.position(
        result.structure.constitution.atom_index(carbon)
    ) == (
        protium.structure.geometry.position(
            protium.structure.constitution.atom_index(carbon)
        )
    )
    again = read_structure_string(
        write_structure_string(result.structure, FileFormat.MMCIF), FileFormat.MMCIF
    )
    n, h = (
        again.constitution.atom_index(AtomRef(ResidueId("A", 25), k))
        for k in ("N", isotope)
    )
    assert again.topology.bond_between(n, h) is None


@pytest.mark.parametrize("isotope", ("D", "T"))
@pytest.mark.parametrize("rebuilt", (False, True))
def test_diagnostic_contacts_agree_with_protium_at_identical_coordinates(
    isotope, rebuilt
):
    def contacts(element):
        src = _masked_input(
            "1div", isotope=element, pose="native" if rebuilt else "unrelated"
        )
        if rebuilt:
            out, _ = rebuild_selected_backbones(src, build_default_component_library())
            assert out.repairs
            src = out.structure
        report = detect_clashes(
            src, component_library=build_default_component_library()
        )

        def address(rid, name):
            return AtomRef(
                rid, "H" if rid == ResidueId("A", 25) and name == element else name
            )

        return {
            frozenset(
                (
                    address(c.left_residue_id, c.left_atom_name),
                    address(c.right_residue_id, c.right_atom_name),
                )
            ): c.overlap_angstrom
            for c in report.clashes
        }

    observed = contacts("H")
    if not rebuilt:
        assert observed
    assert contacts(isotope) == observed


def test_refused_neighbor_does_not_change_contact_exclusions_in_either_order(
    monkeypatch,
):
    source = _masked_input("1div")
    rid = ResidueId("A", 25)
    g = source.residue_geometry(source.constitution.residue_index(rid))
    # Missing CB evidence prevents the second carbonyl fit on its own residue.
    g = g.with_atom_geometries(
        (k, replace(g.atom_geometry(k), occupancy=0)) for k in ("C", "CB")
    )
    source = source.with_updated_residue_geometries([(rid, g)])
    lib = build_default_component_library()
    forward, first = rebuild_selected_backbones(source, lib)
    original = type(source.constitution).iter_residues
    monkeypatch.setattr(
        type(source.constitution),
        "iter_residues",
        lambda self, **kwargs: iter(reversed(tuple(original(self, **kwargs)))),
    )
    backward, second = rebuild_selected_backbones(source, lib)
    assert first is not None and second is not None
    assert first.placements == second.placements
    assert first.placements[ResidueId("A", 24)].reason is None
    assert first.placements[rid].reason
    assert forward.structure == backward.structure


@pytest.mark.parametrize(
    ("comp", "name", "parent", "anchor"),
    (
        ("ALA", "1H", "N", "CA"),
        ("ALA", "1D", "N", "CA"),
        ("ASP", "HD1", "OD1", "CG"),
        ("GLU", "HE1", "OE1", "CD"),
    ),
)
def test_supported_stationary_h_names_do_not_block_carbon_rebuild(
    comp, name, parent, anchor
):
    st = gemmi.read_pdb_string((_CORPUS / "pdb1div.ent").read_text())
    chain = st[0]["A"]
    index = next(
        i for i, r in enumerate(chain) if r.name == comp and 1 < i < len(chain) - 1
    )
    r = chain[index]
    assert r.seqid.num is not None
    rid = ResidueId("A", r.seqid.num)
    kept = [r.clone(), chain[index + 1].clone()]
    if name not in {"1H", "1D"}:
        kept.insert(0, chain[index - 1].clone())
    segment = gemmi.Structure()
    output = gemmi.Chain("A")
    for res in kept:
        res.remove_hydrogens()
        if res.seqid == r.seqid:
            x, y = (next(a for a in res if a.name == k).pos for k in (parent, anchor))
            h = gemmi.Atom()
            h.name, h.element, h.occ = (
                name,
                gemmi.Element("D" if name == "1D" else "H"),
                1,
            )
            h.pos = x + (x - y) * (1.0 / x.dist(y))
            res.add_atom(h)
        output.add_residue(res)
    model = gemmi.Model(1)
    model.add_chain(output)
    segment.add_model(model)
    src = read_structure_string(segment.make_pdb_string(), FileFormat.PDB)
    g = src.residue_geometry(src.constitution.residue_index(rid))
    g = g.with_atom_geometries(
        [
            (
                "C",
                replace(
                    g.atom_geometry("C"),
                    occupancy=0,
                    position=g.position("N").with_offset(80, 80, 80),
                ),
            )
        ]
    )
    masked = src.with_updated_residue_geometries([(rid, g)])
    out, _ = rebuild_selected_backbones(masked, build_default_component_library())
    assert [(e.residue_id, e.atom_names) for e in out.repairs] == [(rid, ("C",))], (
        out.issues
    )
    assert out.structure.topology == masked.topology


def test_three_bonds_across_a_verified_peptide_exempt_heavy_atoms_only():
    source = _masked_input("1div")
    env = BackboneEnvironment(source, build_default_component_library())
    before, after = (ResidueId("A", i) for i in (24, 25))
    ca, next_ca, next_h = (
        AtomRef(before, "CA"),
        AtomRef(after, "CA"),
        AtomRef(after, "H"),
    )
    p = Vec3(0, 0, 0)
    q = Vec3(2.77, 0, 0)  # a short cis CA--C--N--CA contact
    assert not env.clash(ca, p, next_ca, q, {ca: p, next_ca: q})
    h = source.geometry.position(source.constitution.atom_index(next_h))
    close = h.with_offset(1.6, 0, 0)
    assert env.clash(ca, close, next_h, h, {ca: close})
    assert env.clash(next_h, h, ca, close, {ca: close})


@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
)
def test_template_projection_cannot_unprotect_an_untyped_backbone_bond(relationship):
    s = _masked_input("1mbe", declared=True)
    rid = ResidueId("A", 40)
    ca, ha = (s.constitution.atom_index(AtomRef(rid, n)) for n in ("CA", "HA"))
    edge = s.topology.bond_between(ca, ha)
    assert edge is not None
    topology = StructureTopology(
        constitution=s.constitution,
        atom_topologies=s.topology.atom_topologies,
        bonds=tuple(
            replace(e, relationship_type=relationship, order=1) if e == edge else e
            for e in s.topology.bonds
        ),
    )
    s = ProteinStructure.from_payload(
        constitution=s.constitution,
        geometry=s.geometry,
        topology=topology,
        provenance=s.provenance,
        polymer_blueprint=s.polymer_blueprint,
    )
    g = s.residue_geometry(s.constitution.residue_index(rid))
    s = s.with_updated_residue_geometries(
        [
            (
                rid,
                g.with_atom_geometries(
                    [("CA", replace(g.atom_geometry("CA"), occupancy=0))]
                ),
            )
        ]
    )
    out, avail = rebuild_selected_backbones(s, build_default_component_library())
    assert avail is not None
    if relationship is BondRelationshipType.UNKNOWN:
        assert (
            avail.placements[rid].reason
            == "unsupported or externally bound backbone site"
        )
        assert out.structure is s
    else:
        assert avail.placements[rid].reason is None
    assert out.structure.topology == topology


@pytest.mark.parametrize("declared", (False, True))
@pytest.mark.parametrize("angle", (82, 100))
def test_coupled_alpha_carbon_fit_checks_a_beta_hydrogen_angle(declared, angle):
    src = _masked_input("1mbe", declared=declared, beta_angle=angle)
    result, availability = rebuild_selected_backbones(
        src, build_default_component_library()
    )
    assert availability is not None
    reason = availability.placements[ResidueId("A", 40)].reason
    if angle == 82:
        assert reason == "steric clash with A:40.HB2"
        assert not result.repairs
    else:
        assert reason is None
        assert result.repairs[0].atom_names == ("C", "CA")
