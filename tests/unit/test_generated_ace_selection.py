"""Selection at the boundary after a generated ACE has passed the hard checks."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from protrepair.analysis.backbone_contours import BackboneContour
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.terminal import ace_selection
from protrepair.transformer.completion.terminal.ace_selection import (
    AceBackboneResult,
    GeneratedAceSelection,
)

ACE = ResidueId("A", 1)
GLY = ResidueId("A", 2)
ALA = ResidueId("A", 3)
DATA = Path(__file__).parents[1] / "fixtures/pdb/linked-caps/openmm-populated.pdb"


def _structure(*, capped: bool, following: bool = True) -> ProteinStructure:
    lines = []
    for line in DATA.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        rid = int(line[22:26])
        if rid == 4 or (not capped and rid == 1) or (not following and rid == 3):
            continue
        if not capped and rid == 2 and line[12:16].strip() == "H":
            continue
        lines.append(line)
    if capped:
        lines.append("CONECT    1    7")
    s = read_structure_string("\n".join([*lines, "END", ""]), FileFormat.PDB)
    if capped:
        s = _edge(s, AtomRef(ACE, "C"), AtomRef(GLY, "N"))
    return s


def _edge(
    s: ProteinStructure,
    left: AtomRef,
    right: AtomRef,
    kind: BondRelationshipType = BondRelationshipType.COVALENT,
    order: int | None = 1,
) -> ProteinStructure:
    c = s.constitution
    edge = TopologyBond(
        c.atom_index(left), c.atom_index(right), order=order, relationship_type=kind
    )
    others = tuple(
        b for b in s.topology.bonds if b.endpoint_pair() != edge.endpoint_pair()
    )
    return ProteinStructure.from_payload(
        constitution=c,
        geometry=s.geometry,
        polymer_blueprint=s.polymer_blueprint,
        provenance=s.provenance,
        topology=StructureTopology(
            constitution=c,
            atom_topologies=s.topology.atom_topologies,
            bonds=(*others, edge),
        ),
    )


def _position(
    s: ProteinStructure, rid: ResidueId, name: str, p: Vec3
) -> ProteinStructure:
    g = s.residue_geometry(s.constitution.residue_index(rid))
    return s.with_updated_residue_geometries(
        ((rid, g.with_atom_geometry(name, g.atom_geometry(name).with_position(p))),)
    )


def _select(
    parent: ProteinStructure, *candidates: ProteinStructure
) -> GeneratedAceSelection:
    return GeneratedAceSelection.from_verified_poses(
        parent, candidates, flank_id=GLY, cap_id=ACE
    )


def test_structurally_open_psi_is_not_an_evaluation_failure() -> None:
    p = _structure(capped=False, following=False)
    candidate = _structure(capped=True, following=False)
    got = _select(p, candidate)
    assert got.structure is candidate
    assert got.accepted is AceBackboneResult.NO_FOLLOWING_PEPTIDE
    assert not got.rejected


def test_open_psi_does_not_hide_an_invalid_phi() -> None:
    p = _structure(capped=False, following=False)
    s = _structure(capped=True, following=False)
    n = s.geometry.position(s.constitution.atom_index(AtomRef(GLY, "N")))
    got = _select(p, _position(s, ACE, "C", n))
    assert got.structure is p and got.accepted is None
    assert got.rejected == (AceBackboneResult.INVALID_TORSION,)


def test_losing_the_parents_opposite_bond_is_not_a_structurally_open_psi() -> None:
    p = _structure(capped=False)
    s = _structure(capped=True, following=False)
    decision = _select(p, s)
    assert decision.structure is p
    assert decision.rejected == (AceBackboneResult.INVALID_CONNECTIVITY,)


def test_a_source_declaration_with_a_missing_partner_is_not_an_open_psi() -> None:
    p = _structure(capped=False, following=False)
    s = _structure(capped=True, following=False)
    ingress = p.provenance.ingress
    assert ingress.observation is not None
    observation = replace(
        ingress.observation, unresolved_polymer_endpoints=frozenset({AtomRef(GLY, "C")})
    )
    p = p.with_provenance(
        replace(p.provenance, ingress=replace(ingress, observation=observation))
    )
    decision = _select(p, s)
    assert decision.structure is p
    assert decision.rejected == (AceBackboneResult.INVALID_CONNECTIVITY,)


def test_degenerate_psi_is_not_an_open_peptide() -> None:
    p, s = _structure(capped=False), _structure(capped=True)
    carbon = s.geometry.position(s.constitution.atom_index(AtomRef(GLY, "C")))
    got = _select(p, _position(s, ALA, "N", carbon))
    assert got.structure is p and got.rejected == (AceBackboneResult.INVALID_TORSION,)


@pytest.mark.parametrize(
    "kind,order",
    [
        (BondRelationshipType.COVALENT, None),
        (BondRelationshipType.UNKNOWN, 1),
        (BondRelationshipType.COVALENT, 2),
    ],
)
def test_unqualified_or_ambiguous_following_edge_is_not_open(
    kind: BondRelationshipType, order: int | None
) -> None:
    p, s = _structure(capped=False), _structure(capped=True)
    bad = _edge(s, AtomRef(GLY, "C"), AtomRef(ALA, "N"), kind, order)
    got = _select(p, bad)
    assert got.structure is p and got.rejected == (
        AceBackboneResult.INVALID_CONNECTIVITY,
    )
    ambiguous = _edge(s, AtomRef(GLY, "C"), AtomRef(ACE, "C"))
    assert _select(p, ambiguous).rejected == (AceBackboneResult.INVALID_CONNECTIVITY,)


def test_real_profile_accepts_a_defined_pose_without_reading_the_tail() -> None:
    p, s = _structure(capped=False), _structure(capped=True)

    def candidates() -> Iterator[ProteinStructure]:
        yield s
        pytest.fail("a qualifying first pose must not trigger further work")

    got = GeneratedAceSelection.from_verified_poses(
        p, candidates(), flank_id=GLY, cap_id=ACE
    )
    assert got.structure is s
    assert got.accepted in (AceBackboneResult.ALLOWED, AceBackboneResult.FAVORED)


def test_first_measurement_error_never_becomes_unassessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _structure(capped=False, following=False)
    s = _structure(capped=True, following=False)
    real = ace_selection.measured_dihedral_degrees
    fail = iter((True, False))

    def once(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float | None:
        if next(fail):
            raise ValueError("measurement failed")
        return real(a, b, c, d)

    monkeypatch.setattr(ace_selection, "measured_dihedral_degrees", once)
    got = _select(p, s, s)
    assert got.structure is s and got.accepted is AceBackboneResult.NO_FOLLOWING_PEPTIDE
    assert got.rejected == (AceBackboneResult.INVALID_TORSION,)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.01])
def test_failed_contour_preserves_parent_then_can_recover(
    bad: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, s = _structure(capped=False), _structure(capped=True)
    scores = iter([bad, bad, 0.01])
    monkeypatch.setattr(BackboneContour, "density", lambda *args: next(scores))
    refused = _select(p, s)
    assert refused.structure is p and refused.accepted is None
    assert refused.rejected == (AceBackboneResult.REFERENCE_UNAVAILABLE,)
    result = _select(p, s, s)
    assert result.structure is s and result.accepted is AceBackboneResult.ALLOWED
    assert result.rejected == refused.rejected


def test_sticky_outlier_and_exhaustion_are_local_to_one_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p, s = _structure(capped=False), _structure(capped=True)
    results = iter(
        [
            AceBackboneResult.OUTLIER,
            AceBackboneResult.NO_FOLLOWING_PEPTIDE,
            AceBackboneResult.OUTLIER,
            AceBackboneResult.OUTLIER,
            AceBackboneResult.NO_FOLLOWING_PEPTIDE,
            AceBackboneResult.ALLOWED,
            AceBackboneResult.NO_FOLLOWING_PEPTIDE,
        ]
    )
    monkeypatch.setattr(ace_selection, "_assess", lambda *args: next(results))
    exhausted = _select(p, s, s, s)
    assert exhausted.structure is p and exhausted.accepted is None
    assert exhausted.rejected == (
        AceBackboneResult.OUTLIER,
        AceBackboneResult.NO_FOLLOWING_PEPTIDE,
        AceBackboneResult.OUTLIER,
    )
    alternate = _select(p, s, s, s)
    assert alternate.structure is s and alternate.accepted is AceBackboneResult.ALLOWED
    assert alternate.rejected == exhausted.rejected[:2]
    assert _select(p, s).accepted is AceBackboneResult.NO_FOLLOWING_PEPTIDE
    assert _select(p).structure is p


def test_missing_reference_is_not_an_unassessed_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object) -> float:
        raise FileNotFoundError("reference grid is missing")

    monkeypatch.setattr(BackboneContour, "density", unavailable)
    p, s = _structure(capped=False), _structure(capped=True)
    got = _select(p, s)
    assert got.structure is p and got.rejected == (
        AceBackboneResult.REFERENCE_UNAVAILABLE,
    )


def test_contour_interpolates_across_both_angular_seams() -> None:
    contour = BackboneContour.GENERAL
    corners = (
        0.005560270487425575,
        0.0034230091396416297,
        0.007019335197746127,
        0.004306548647892551,
    )
    assert contour.density(-179, -179) == pytest.approx(corners[0])
    for phi, psi in ((180, 180), (-180, -180), (180, -180)):
        assert contour.density(phi, psi) == pytest.approx(sum(corners) / 4)


def test_backbone_family_controls_the_reference() -> None:
    assert BackboneContour.for_nonproline("GLY", "PRO") is BackboneContour.GLYCINE
    assert BackboneContour.for_nonproline("VAL", "PRO") is BackboneContour.PRE_PROLINE
    assert BackboneContour.for_nonproline("ILE", "NME") is BackboneContour.ILE_VAL
    assert BackboneContour.for_nonproline("ALA", "GLY") is BackboneContour.GENERAL
    with pytest.raises(ValueError, match="proline"):
        BackboneContour.for_nonproline("PRO", "GLY")


def test_first_allowed_pose_wins_without_looking_for_a_favored_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, pose = _structure(capped=False), _structure(capped=True)
    scores = iter((0.0, BackboneContour.GLYCINE.allowed_floor, 1.0))
    monkeypatch.setattr(BackboneContour, "density", lambda *args: next(scores))
    selection = _select(parent, pose, pose, pose)
    assert selection.accepted is AceBackboneResult.ALLOWED
    assert selection.rejected == (AceBackboneResult.OUTLIER,)
    assert next(scores) == 1.0
