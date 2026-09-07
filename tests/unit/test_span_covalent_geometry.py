"""Experimental-coordinate regressions for covalent span kinematics."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import pytest
from rdkit import Chem

from protrepair.api import process_structure
from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.geometry.rotation import AxisRotation
from protrepair.io import (
    FileFormat,
    read_structure,
    read_structure_string,
    write_structure_string,
)
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure import ProteinStructure
from protrepair.structure.disulfide import disulfide_atom_ref_pairs
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondProvenance
from protrepair.transformer.completion.span_reconstruction import (
    ReconstructedSpanCandidate,
    SpanReconstructionFailure,
    SpanReconstructionFailureKind,
    reconstruct_donor_span,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    WorkflowTransformRequests,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/pdb/span-reconstruction"


def _position(structure: ProteinStructure, rid: ResidueId, name: str) -> Vec3:
    return structure.geometry.position(
        structure.constitution.atom_index(AtomRef(rid, name))
    )


def _angle(
    structure: ProteinStructure, refs: tuple[AtomRef, AtomRef, AtomRef]
) -> float:
    return gemmi.calculate_angle(
        *(
            gemmi.Position(*_position(structure, ref.residue_id, ref.atom_name))
            for ref in refs
        )
    )


def _dihedral(
    structure: ProteinStructure, refs: tuple[AtomRef, AtomRef, AtomRef, AtomRef]
) -> float:
    return gemmi.calculate_dihedral(
        *(
            gemmi.Position(*_position(structure, ref.residue_id, ref.atom_name))
            for ref in refs
        )
    )


def _source(
    reference: ProteinStructure, gap: tuple[ResidueId, ...]
) -> ProteinStructure:
    numbers = {rid.seq_num for rid in gap}
    lines = write_structure_string(reference, FileFormat.PDB).splitlines()
    return read_structure_string(
        "\n".join(
            line
            for line in lines
            if not line.startswith("CONECT")
            and not (line.startswith("ATOM  ") and int(line[22:26]) in numbers)
            and not (
                line.startswith("SSBOND")
                and (int(line[17:21]) in numbers or int(line[31:35]) in numbers)
            )
        )
        + "\n",
        FileFormat.PDB,
    )


def _seed(
    reference: ProteinStructure, pivot: ResidueId, degrees: float
) -> ProteinStructure:
    origin = _position(reference, pivot, "CA").to_array()
    unit = _position(reference, pivot, "C").to_array() - origin
    unit /= np.linalg.norm(unit)
    radians = np.deg2rad(degrees)
    geometries = []
    for index, geometry in enumerate(reference.geometry.atom_geometries):
        atom_index = AtomIndex(index)
        ref = reference.constitution.atom_ref_at(atom_index)
        point = geometry.position.to_array()
        if ref.residue_id > pivot or (ref.residue_id == pivot and ref.atom_name == "O"):
            relative = point - origin
            point = (
                relative * np.cos(radians)
                + np.cross(unit, relative) * np.sin(radians)
                + unit * np.dot(unit, relative) * (1.0 - np.cos(radians))
                + origin
            )
        point = np.array([-point[1], point[0], point[2]]) + [13.0, -7.0, 4.0]
        geometries.append(geometry.with_position(Vec3.from_iterable(point)))
    return ProteinStructure.from_payload(
        constitution=reference.constitution,
        geometry=StructureGeometry(
            constitution=reference.constitution, atom_geometries=tuple(geometries)
        ),
        topology=reference.topology,
        polymer_blueprint=reference.polymer_blueprint,
        provenance=reference.provenance,
    )


@pytest.mark.parametrize(
    "pdb,start,length,perturbation",
    [
        ("1ubq", 36, 3, -90.0),
        ("1ubq", 36, 3, 0.0),
        ("2ci2", 80, 3, -135.0),
        ("2ci2", 80, 3, 0.0),
        ("1ubq", 35, 2, 0.0),
    ],
)
def test_reconstruction_preserves_covalent_geometry(
    pdb: str,
    start: int,
    length: int,
    perturbation: float,
) -> None:
    reference = read_structure(FIXTURES / f"{pdb}.pdb")
    chain_id = next(reference.constitution.iter_residues()).residue_id.chain_id
    gap = tuple(ResidueId(chain_id, number) for number in range(start, start + length))
    source = _source(reference, gap)
    donor = _seed(reference, gap[length // 2], perturbation)
    scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id, start - 1),
        absent_residue_ids=gap,
        following_residue_id=ResidueId(chain_id, start + length),
    )
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=scope,
                    donor_structure=donor,
                    donor_residue_ids=gap,
                ),
            ),
        ),
    )
    assert not result.issues, [issue.message for issue in result.issues]
    assert all(
        result.structure.constitution.residue_or_ligand(rid) is not None for rid in gap
    )
    for integer in range(source.geometry.atom_count()):
        index = AtomIndex(integer)
        ref = source.constitution.atom_ref_at(index)
        assert _position(
            result.structure, ref.residue_id, ref.atom_name
        ) == source.geometry.position(index)

    library = build_default_component_library()
    for rid in gap:
        residue = reference.constitution.residue_or_ligand(rid)
        assert residue is not None
        definition = library.require(residue.component_id).definition
        for bond in definition.bonds:
            if not residue.has_atom_site(bond.atom_name_1) or not residue.has_atom_site(
                bond.atom_name_2
            ):
                continue
            expected = _position(reference, rid, bond.atom_name_1).distance_to(
                _position(reference, rid, bond.atom_name_2)
            )
            actual = _position(result.structure, rid, bond.atom_name_1).distance_to(
                _position(result.structure, rid, bond.atom_name_2)
            )
            assert actual == pytest.approx(expected, abs=1e-8)
        for name in residue.atom_site_names():
            neighbors = sorted(
                definition.bonded_atom_names(name)
                & frozenset(residue.atom_site_names())
            )
            for left, right in combinations(neighbors, 2):
                refs = (AtomRef(rid, left), AtomRef(rid, name), AtomRef(rid, right))
                assert _angle(result.structure, refs) == pytest.approx(
                    _angle(reference, refs), abs=1e-8
                )
        if residue.component_id == "PRO":
            previous = ResidueId(chain_id, rid.seq_num - 1)
            refs = (
                AtomRef(previous, "C"),
                AtomRef(rid, "N"),
                AtomRef(rid, "CA"),
                AtomRef(rid, "CD"),
            )
            assert _dihedral(result.structure, refs) == pytest.approx(
                _dihedral(reference, refs), abs=1e-8
            )
    if perturbation:
        for rid in gap:
            residue = reference.constitution.residue_or_ligand(rid)
            assert residue is not None
            for name in residue.atom_site_names():
                assert tuple(_position(result.structure, rid, name)) == pytest.approx(
                    tuple(_position(reference, rid, name)), abs=1e-8
                )
    if start in (36, 80):
        text = write_structure_string(result.structure, FileFormat.PDB)
        assert (
            Chem.MolFromPDBBlock(
                text, sanitize=True, removeHs=False, proximityBonding=True
            )
            is not None
        )


@pytest.mark.parametrize("missing_in_source", [False, True])
def test_incomplete_amide_neighborhood_is_rejected_atomically(
    missing_in_source: bool,
) -> None:
    reference = read_structure(FIXTURES / "1ubq.pdb")
    gap = (
        (ResidueId("A", 36),)
        if missing_in_source
        else tuple(ResidueId("A", n) for n in (36, 37, 38))
    )
    source = _source(reference, gap)
    target = source if missing_in_source else reference
    text = write_structure_string(target, FileFormat.PDB)
    incomplete = read_structure_string(
        "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith("CONECT")
            and not (
                line.startswith("ATOM  ")
                and int(line[22:26]) == 37
                and line[12:16].strip() == "CD"
            )
        )
        + "\n",
        FileFormat.PDB,
    )
    if missing_in_source:
        source = incomplete
    scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId("A", 35),
        absent_residue_ids=gap,
        following_residue_id=ResidueId("A", gap[-1].seq_num + 1),
    )
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=scope,
                    donor_structure=reference if missing_in_source else incomplete,
                    donor_residue_ids=gap,
                ),
            ),
        ),
    )
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert len(result.issues) == 1
    assert "amide-N neighbor" in result.issues[0].message


@pytest.mark.parametrize("perturbation", [0.0, 45.0])
def test_fixed_carbonyl_oxygen_remains_in_the_closed_peptide_plane(
    perturbation: float,
) -> None:
    reference = read_structure(FIXTURES / "1ubq.pdb")
    gap = tuple(ResidueId("A", number) for number in range(4, 13))
    source = _source(reference, gap)
    scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId("A", 3),
        following_residue_id=ResidueId("A", 13),
        absent_residue_ids=gap,
    )
    # The experimental span itself has residue-local restraint outliers. This
    # test isolates closure, not permission to bypass the workflow's other gates.
    outcome = reconstruct_donor_span(
        source,
        scope=scope,
        donor_structure=_seed(reference, gap[4], perturbation),
        donor_residue_ids=gap,
        donor_preceding_residue_id=scope.preceding_residue_id,
        donor_following_residue_id=scope.following_residue_id,
    )
    assert isinstance(outcome, ReconstructedSpanCandidate)
    assert outcome.endpoint_rmsd_angstrom is not None
    assert outcome.endpoint_rmsd_angstrom <= 0.1
    o = _position(source, ResidueId("A", 3), "O")
    c = _position(source, ResidueId("A", 3), "C")
    n = outcome.residue_payloads[0].position("N")
    observed = gemmi.calculate_angle(
        gemmi.Position(*o), gemmi.Position(*c), gemmi.Position(*n)
    )
    assert observed == pytest.approx(
        _angle(
            reference,
            (
                AtomRef(ResidueId("A", 3), "O"),
                AtomRef(ResidueId("A", 3), "C"),
                AtomRef(gap[0], "N"),
            ),
        ),
        abs=1e-8,
    )
    for payload in outcome.residue_payloads:
        for name in ("N", "CA", "C", "O"):
            assert tuple(payload.position(name)) == pytest.approx(
                tuple(_position(reference, payload.residue_id, name)), abs=1e-8
            )


def test_inserted_cysteine_is_followed_by_state_driven_disulfide_resolution() -> None:
    reference = read_structure(FIXTURES / "1crn.pdb")
    gap = (ResidueId("A", 4),)
    source = _source(reference, gap)
    assert len(disulfide_atom_ref_pairs(source)) == 2
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=AbsentResidueSpanScope(
                        preceding_residue_id=ResidueId("A", 3),
                        absent_residue_ids=gap,
                        following_residue_id=ResidueId("A", 5),
                    ),
                    donor_structure=reference,
                    donor_residue_ids=gap,
                ),
            ),
        ),
    )
    assert not result.issues
    assert disulfide_atom_ref_pairs(result.structure) == disulfide_atom_ref_pairs(
        reference
    )
    bond = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(AtomRef(gap[0], "SG")),
        result.structure.constitution.atom_index(AtomRef(ResidueId("A", 32), "SG")),
    )
    assert bond is not None
    assert bond.provenance is BondProvenance.EVIDENCE_RESOLVED
    assert [event.kind.value for event in result.repairs] == [
        "absent_residue_span_reconstructed",
        "disulfide_topology_resolved",
    ]


@pytest.mark.parametrize("cyclic", [True, False])
def test_renamed_and_acyclic_n_substituents_use_the_active_chemistry(
    cyclic: bool,
) -> None:
    lines = (FIXTURES / "1ubq.pdb").read_text().splitlines()
    renamed = []
    for line in lines:
        if line.startswith("ATOM  ") and line[17:20] == "PRO":
            line = line[:17] + "XPR" + line[20:]
            if line[12:16].strip() == "CD":
                line = line[:12] + " CX " + line[16:]
        renamed.append(line)
    reference = read_structure_string("\n".join(renamed) + "\n", FileFormat.PDB)
    library = build_default_component_library()
    template = library.require("PRO")
    bonds = tuple(
        replace(
            bond,
            atom_name_1="CX" if bond.atom_name_1 == "CD" else bond.atom_name_1,
            atom_name_2="CX" if bond.atom_name_2 == "CD" else bond.atom_name_2,
        )
        for bond in template.definition.bonds
        if cyclic or {bond.atom_name_1, bond.atom_name_2} != {"CG", "CD"}
    )
    custom = replace(
        template,
        definition=replace(
            template.definition,
            component_id="XPR",
            atom_names=tuple(
                "CX" if name == "CD" else name
                for name in template.definition.atom_names
            ),
            bonds=bonds,
            aliases=(),
        ),
        heavy_atom_semantics=None,
        hydrogen_semantics=None,
    )
    library = library.with_template(custom)
    gap = tuple(ResidueId("A", number) for number in (36, 37, 38))
    source = _source(reference, gap)
    scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId("A", 35),
        following_residue_id=ResidueId("A", 39),
        absent_residue_ids=gap,
    )
    outcome = reconstruct_donor_span(
        source,
        scope=scope,
        donor_structure=_seed(reference, gap[1], -90.0),
        donor_residue_ids=gap,
        donor_preceding_residue_id=scope.preceding_residue_id,
        donor_following_residue_id=scope.following_residue_id,
        component_library=library,
    )
    assert isinstance(outcome, ReconstructedSpanCandidate)
    previous, current = outcome.residue_payloads[1:]
    after = gemmi.calculate_dihedral(
        *(
            gemmi.Position(*point)
            for point in (
                previous.position("C"),
                current.position("N"),
                current.position("CA"),
                current.position("CX"),
            )
        )
    )
    before = _dihedral(
        reference,
        (
            AtomRef(gap[1], "C"),
            AtomRef(gap[2], "N"),
            AtomRef(gap[2], "CA"),
            AtomRef(gap[2], "CX"),
        ),
    )
    assert after == pytest.approx(before, abs=1e-8)


@pytest.mark.parametrize("terminus", ["prefix", "suffix"])
def test_terminal_projection_rejects_pyramidal_amide_n(terminus: str) -> None:
    lines = (FIXTURES / "1ubq.pdb").read_text().splitlines()
    reference = read_structure_string(
        "\n".join(
            line
            for line in lines
            if not line.startswith("ATOM  ") or int(line[22:26]) >= 34
        )
        + "\n",
        FileFormat.PDB,
    )
    is_prefix = terminus == "prefix"
    gap = tuple(
        ResidueId("A", number)
        for number in (range(34, 37) if is_prefix else range(38, 41))
    )
    source = _source(reference, gap)
    target = source if is_prefix else reference
    proline = ResidueId("A", 37 if is_prefix else 38)
    n = _position(target, proline, "N")
    ca = _position(target, proline, "CA")
    rotation = AxisRotation.from_points(n, ca)
    cd_index = target.constitution.atom_index(AtomRef(proline, "CD"))
    geometries = list(target.geometry.atom_geometries)
    geometries[cd_index.value] = geometries[cd_index.value].with_position(
        rotation.rotate_point(
            _position(target, proline, "CD"),
            origin=n,
            theta_radians=float(np.pi / 2),
        )
    )
    malformed = ProteinStructure.from_payload(
        constitution=target.constitution,
        geometry=StructureGeometry(
            constitution=target.constitution, atom_geometries=tuple(geometries)
        ),
        topology=target.topology,
        polymer_blueprint=target.polymer_blueprint,
        provenance=target.provenance,
    )
    scope = AbsentResidueSpanScope(
        preceding_residue_id=None if is_prefix else ResidueId("A", 37),
        following_residue_id=ResidueId("A", 37) if is_prefix else None,
        absent_residue_ids=gap,
    )
    outcome = reconstruct_donor_span(
        malformed if is_prefix else source,
        scope=scope,
        donor_structure=reference if is_prefix else malformed,
        donor_residue_ids=gap,
        donor_preceding_residue_id=scope.preceding_residue_id,
        donor_following_residue_id=scope.following_residue_id,
    )
    assert isinstance(outcome, SpanReconstructionFailure)
    assert outcome.kind is SpanReconstructionFailureKind.INVALID_PEPTIDE_JUNCTION
