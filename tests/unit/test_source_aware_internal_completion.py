"""Partial standard completion preserves materialized geometry."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from protrepair.api import process_structure
from protrepair.chemistry.component.semantics import HeavyAtomSemantics
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.chemistry.standard.heavy_atom_programs import (
    STANDARD_HEAVY_ATOM_SEMANTICS,
)
from protrepair.chemistry.standard.restraints import STANDARD_RESTRAINT_TEMPLATES
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.geometry import InternalCoordinateFrame
from protrepair.io import read_structure, read_structure_string
from protrepair.scope import ResidueSetScope
from protrepair.state import SidechainHeavyAtomCompletenessState
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.atom.internal_coordinates import (
    InternalCoordinateEnvironment,
)
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.workflow.contracts import requested_process_goal

CRAMBIN = Path("tests/fixtures/pdb/span-reconstruction/1crn.pdb")
UBIQUITIN = Path("tests/fixtures/pdb/span-reconstruction/1ubq.pdb")


def _source_payload(path: Path, residue_id: ResidueId) -> CompletionResiduePayload:
    text = "\n".join(
        line
        for line in path.read_text().splitlines()
        if line.startswith("ATOM  ")
        and line[21] == residue_id.chain_id
        and int(line[22:26]) == residue_id.seq_num
    )
    structure = read_structure_string(text + "\nEND\n", FileFormat.PDB)
    index = structure.constitution.residue_index(residue_id)
    return CompletionResiduePayload(
        residue_site=structure.constitution.residue_site_at(index),
        residue_geometry=structure.residue_geometry(index),
    )


def _complete(
    source: CompletionResiduePayload, missing: tuple[str, ...]
) -> InternalCoordinateEnvironment:
    partial = source.without_atom_sites(missing)
    environment = InternalCoordinateEnvironment.from_payloads(
        residue=partial,
        next_peptide_residue=None,
        missing_atom_names=missing,
    )
    semantics = (
        build_standard_component_library()
        .require(source.component_id)
        .heavy_atom_semantics
    )
    assert isinstance(semantics, HeavyAtomSemantics)
    semantics.program.apply(environment)
    return environment


def test_connected_tyr_prefix_does_not_reset_observed_chi() -> None:
    source = _source_payload(
        Path("tests/fixtures/corpus/pdb6l54.ent"), ResidueId("A", 2088)
    )
    missing = ("CD1", "CD2", "CE1", "CE2", "CZ", "OH")
    environment = _complete(source, missing)

    for name in source.atom_names():
        assert environment.geometry.position(name) == source.position(name)
    assert environment.torsion(("N", "CA", "CB", "CG")) == pytest.approx(
        InternalCoordinateFrame.torsion(
            *(source.position(name) for name in ("N", "CA", "CB", "CG"))
        )
    )
    assert all(environment.has_atom(name) for name in missing)


def test_heavy_repair_pipeline_preserves_connected_tyr_source_prefix() -> None:
    path = Path("tests/fixtures/corpus/pdb6l54.ent")
    text = "\n".join(
        line
        for line in path.read_text().splitlines()
        if line.startswith("ATOM  ") and line[21] == "A" and int(line[22:26]) == 2088
    )
    original = read_structure_string(text + "\nEND\n", FileFormat.PDB)
    result = repair_heavy_atoms_core(original, augment_c_terminal_oxt=False)
    assert not result.has_errors()
    residue_id = ResidueId("A", 2088)
    before_index = original.constitution.residue_index(residue_id)
    after_index = result.structure.constitution.residue_index(residue_id)
    for atom in original.constitution.residue_site_at(before_index).atom_sites:
        assert result.structure.residue_geometry(after_index).atom_geometry(
            atom.name
        ) == original.residue_geometry(before_index).atom_geometry(atom.name)
    assert (
        len(result.structure.constitution.residue_site_at(after_index).atom_sites) == 12
    )


@pytest.mark.parametrize("workflow", (False, True))
def test_impossible_proline_closure_keeps_anchors_and_reports_incompleteness(workflow):
    records = []
    for line in UBIQUITIN.read_text().splitlines():
        if not line.startswith("ATOM  ") or int(line[22:26]) != 37:
            continue
        name = line[12:16].strip()
        if name == "CD":
            continue
        if name == "N":
            line = line[:30] + f"{float(line[30:38]) + 10.0:8.3f}" + line[38:]
        records.append(line)
    source = read_structure_string("\n".join(records) + "\nEND\n", FileFormat.PDB)
    residue_id = ResidueId("A", 37)
    if workflow:
        result = process_structure(
            source,
            requested_goals=(
                requested_process_goal(
                    scope=ResidueSetScope((residue_id,)),
                    value=SidechainHeavyAtomCompletenessState.COMPLETE,
                ),
            ),
        )
        assert result.requested_goals_fully_satisfied() is False
    else:
        result = repair_heavy_atoms_core(source, augment_c_terminal_oxt=False)

    before = source.residue_geometry(source.constitution.residue_index(residue_id))
    after = result.structure.residue_geometry(
        result.structure.constitution.residue_index(residue_id)
    )
    for name in ("N", "CA", "C", "O", "CB", "CG"):
        assert after.atom_geometry(name) == before.atom_geometry(name)
    residue = result.structure.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    assert not residue.has_atom_site("CD")
    if not workflow:
        assert any(
            issue.kind is ValidationIssueKind.GEOMETRY_PLACEMENT_SKIPPED
            and issue.severity is IssueSeverity.WARNING
            and "CD" in issue.atom_names
            for issue in result.issues
        )


@pytest.mark.parametrize(
    ("path", "seq_num", "missing"),
    (
        (CRAMBIN, 1, ("OG1",)),
        (CRAMBIN, 8, ("CG1",)),
        (CRAMBIN, 13, ("CD1", "CE1", "CZ")),
        (CRAMBIN, 13, ("CD2", "CE2", "CZ")),
        (UBIQUITIN, 37, ("CD",)),
        (UBIQUITIN, 38, ("CD",)),
    ),
)
def test_partial_standard_group_keeps_source_and_closes_bonds(
    path: Path, seq_num: int, missing: tuple[str, ...]
) -> None:
    source = _source_payload(path, ResidueId("A", seq_num))
    environment = _complete(source, missing)
    template = build_standard_component_library().require(source.component_id)

    for name in source.atom_names():
        if name not in missing:
            assert environment.geometry.position(name) == source.position(name)

    for bond in template.definition.bonds:
        first, second = bond.atom_name_1, bond.atom_name_2
        if not source.has_atom(first) or not source.has_atom(second):
            continue
        if first not in missing and second not in missing:
            continue
        actual = np.linalg.norm(
            environment.geometry.position(first).to_array()
            - environment.geometry.position(second).to_array()
        )
        source_length = np.linalg.norm(
            source.position(first).to_array() - source.position(second).to_array()
        )
        assert actual == pytest.approx(source_length, abs=0.15), (first, second)


@pytest.fixture(scope="module")
def standard_sources() -> dict[str, CompletionResiduePayload]:
    sources: dict[str, CompletionResiduePayload] = {}
    for path in (
        CRAMBIN,
        UBIQUITIN,
        Path("tests/fixtures/pdb/1aho.pdb"),
        Path("tests/fixtures/corpus/pdb6l54.ent"),
    ):
        structure = read_structure(path)
        for site in structure.constitution.residue_slots:
            semantics = STANDARD_HEAVY_ATOM_SEMANTICS.get(site.component_id)
            if semantics is None or site.component_id in sources:
                continue
            if not all(site.has_atom_site(name) for name in semantics.atom_order):
                continue
            index = structure.constitution.residue_index(site.residue_id)
            sources[site.component_id] = CompletionResiduePayload(
                residue_site=site,
                residue_geometry=structure.residue_geometry(index),
            )
    return sources


@pytest.mark.parametrize("component_id", tuple(STANDARD_HEAVY_ATOM_SEMANTICS))
def test_fully_missing_sidechain_retains_literal_fallback(
    component_id: str, standard_sources: dict[str, CompletionResiduePayload]
) -> None:
    source = standard_sources[component_id]
    missing = tuple(
        name
        for name in STANDARD_HEAVY_ATOM_SEMANTICS[component_id].atom_order
        if name not in {"N", "CA", "C", "O", "CB"}
    )
    completed = _complete(source, missing)
    literal = InternalCoordinateEnvironment.from_payloads(
        residue=completed.residue,
        next_peptide_residue=None,
        missing_atom_names=missing,
    )
    STANDARD_HEAVY_ATOM_SEMANTICS[component_id].program.apply(literal)
    for name in STANDARD_HEAVY_ATOM_SEMANTICS[component_id].atom_order:
        assert completed.geometry.position(name) == literal.geometry.position(name)


@pytest.mark.parametrize("component_id", tuple(STANDARD_HEAVY_ATOM_SEMANTICS))
def test_complete_standard_residue_keeps_all_materialized_coordinates(
    component_id: str, standard_sources: dict[str, CompletionResiduePayload]
) -> None:
    source = standard_sources[component_id]
    completed = _complete(source, ())
    for name in source.atom_names():
        assert completed.geometry.position(name) == source.position(name)


@pytest.mark.parametrize("component_id", tuple(STANDARD_HEAVY_ATOM_SEMANTICS))
def test_single_missing_atom_preserves_connected_materialized_neighbors(
    component_id: str, standard_sources: dict[str, CompletionResiduePayload]
) -> None:
    source = standard_sources[component_id]
    template = build_standard_component_library().require(component_id)
    semantics = STANDARD_HEAVY_ATOM_SEMANTICS[component_id]
    for missing in semantics.atom_order:
        if missing in {"N", "CA", "C", "O", "CB"}:
            continue
        retained = set(semantics.atom_order) - {missing}
        connected = {"CA"}
        while True:
            next_connected = connected | {
                neighbor
                for bond in template.definition.bonds
                for first, neighbor in (
                    (bond.atom_name_1, bond.atom_name_2),
                    (bond.atom_name_2, bond.atom_name_1),
                )
                if first in connected and neighbor in retained
            }
            if connected == next_connected:
                break
            connected = next_connected
        if connected != retained:
            # Disconnected fragments are governed by explicit orphan policy.
            continue

        completed = _complete(source, (missing,))
        assert completed.has_atom(missing)
        for name in source.atom_names():
            if name != missing:
                assert completed.geometry.position(name) == source.position(name)
        for bond in template.definition.bonds:
            if missing not in (bond.atom_name_1, bond.atom_name_2):
                continue
            if not all(
                source.has_atom(name) for name in (bond.atom_name_1, bond.atom_name_2)
            ):
                continue
            actual = np.linalg.norm(
                completed.geometry.position(bond.atom_name_1).to_array()
                - completed.geometry.position(bond.atom_name_2).to_array()
            )
            expected = np.linalg.norm(
                source.position(bond.atom_name_1).to_array()
                - source.position(bond.atom_name_2).to_array()
            )
            assert actual == pytest.approx(expected, abs=0.15), (
                component_id,
                missing,
                bond,
            )


@pytest.mark.parametrize("seq_num", (37, 38))
def test_partial_proline_selects_closure_consistent_with_nitrogen_angle(
    seq_num: int,
) -> None:
    source = _source_payload(UBIQUITIN, ResidueId("A", seq_num))
    completed = _complete(source, ("CD",))
    target = next(
        target
        for target in STANDARD_RESTRAINT_TEMPLATES["PRO"].angle_targets
        if target.center_atom_name == "N"
    )
    a = (
        completed.geometry.position("CA").to_array()
        - completed.geometry.position("N").to_array()
    )
    b = (
        completed.geometry.position("CD").to_array()
        - completed.geometry.position("N").to_array()
    )
    angle = np.degrees(
        np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    )
    assert angle == pytest.approx(target.target_angle_degrees, abs=4.0)


def test_custom_coherent_replacement_program_remains_literal() -> None:
    source = _source_payload(CRAMBIN, ResidueId("A", 13))
    partial = source.without_atom_sites(("CD1", "CE1", "CZ"))
    context = InternalCoordinateEnvironment.from_payloads(
        residue=partial,
        next_peptide_residue=None,
        missing_atom_names=("CD1", "CE1", "CZ"),
    )
    STANDARD_HEAVY_ATOM_SEMANTICS["PHE"].program.apply(context)
    assert context.geometry.position("CG") != source.position("CG")
    assert context.torsion(("N", "CA", "CB", "CG")) == pytest.approx(-64.7)


def test_partial_completion_preserves_scalar_metadata() -> None:
    source = _source_payload(CRAMBIN, ResidueId("A", 13))
    source = replace(
        source,
        residue_geometry=source.residue_geometry.with_atom_geometry(
            "CG",
            replace(
                source.atom_geometry("CG"),
                occupancy=0.75,
                b_factor=11.0,
                altloc="A",
            ),
        ),
        formal_charge_by_atom_name=(("CG", 0),),
    )
    environment = _complete(source, ("CD1", "CE1", "CZ"))
    patch = environment.build_patch(STANDARD_HEAVY_ATOM_SEMANTICS["PHE"].atom_order)
    _, geometry, charges = patch.materialize_on_payload(
        environment.residue.residue_site,
        residue_geometry=environment.residue.residue_geometry,
        formal_charge_by_atom_name=environment.residue.formal_charge_by_atom_name,
    )
    assert geometry.atom_geometry("CG") == source.atom_geometry("CG")
    assert dict(charges)["CG"] == 0


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_materialized_hydrogen_isotopes_survive_partial_heavy_patch(
    isotope: str,
) -> None:
    source = _source_payload(Path("tests/fixtures/pdb/1aho.pdb"), ResidueId("A", 6))
    hydrogen = next(atom for atom in source.atom_sites if atom.is_hydrogen())
    source = replace(
        source,
        residue_site=source.residue_site.with_atom_site(
            replace(hydrogen, element=isotope)
        ),
    )
    environment = _complete(source, ("CG2",))
    patch = environment.build_patch(STANDARD_HEAVY_ATOM_SEMANTICS["ILE"].atom_order)
    site, geometry, _ = patch.materialize_on_payload(
        environment.residue.residue_site,
        residue_geometry=environment.residue.residue_geometry,
    )
    assert site.atom_site(hydrogen.name).element == isotope
    assert geometry.atom_geometry(hydrogen.name) == source.atom_geometry(hydrogen.name)


@pytest.mark.parametrize(
    ("path", "seq_num", "missing"),
    ((CRAMBIN, 1, ("OG1",)), (Path("tests/fixtures/pdb/1aho.pdb"), 6, ("CG1", "CD1"))),
)
def test_complementary_branch_retains_measured_cb_stereochemistry(
    path: Path, seq_num: int, missing: tuple[str, ...]
) -> None:
    source = _source_payload(path, ResidueId("A", seq_num))
    completed = _complete(source, missing)
    center = next(
        center
        for center in build_standard_component_library()
        .require(source.component_id)
        .tetrahedral_centers
        if center.center_atom_name == "CB"
    )
    before = np.array(
        [
            source.position(name).to_array() - source.position("CB").to_array()
            for name in center.ordered_neighbor_atom_names
        ]
    )
    after = np.array(
        [
            completed.geometry.position(name).to_array()
            - completed.geometry.position("CB").to_array()
            for name in center.ordered_neighbor_atom_names
        ]
    )
    assert np.sign(np.linalg.det(after)) == np.sign(np.linalg.det(before))
