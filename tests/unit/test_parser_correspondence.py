"""Canonical attribution of native no-CONECT PDB observations."""

from dataclasses import replace

import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics import parser_readability as diagnostics
from protrepair.errors import ModelInvariantError
from protrepair.geometry import Vec3
from protrepair.io import pdb_projection
from protrepair.io.pdb_projection import (
    pdb_without_conect,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.state.structure_parser import StructureParserCompatibilityFacts
from protrepair.structure import AtomRef, ProteinStructure, ResidueId, StructureGeometry
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import StructureTopology, TopologyBond
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.parser_witness import (
    parser_witness_repair_candidates,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import RequestedGoalReport, WorkflowPlanningContext
from protrepair.workflow.engine.reporting import evaluate_workflow_branch_quality_score


def _failing_structure(
    residue_ids: tuple[ResidueId, ...],
    *,
    altloc: str | None = None,
    retained_indices: frozenset[int] = frozenset(),
) -> ProteinStructure:
    chains = []
    ligands = []
    for ordinal, residue_id in enumerate(residue_ids):
        atoms = tuple(
            atom_payload(name, element, Vec3(x + 20 * ordinal, y, z), altloc=altloc)
            for name, element, (x, y, z) in (
                ("N", "N", (0, 0, 0)),
                ("CA", "C", (1.45, 0, 0)),
                ("H", "H", (-1, 0, 0)),
                ("H2", "H", (0, 1, 0)),
                ("H3", "H", (0, -1, 0)),
                ("H4", "H", (0, 0, 1)),
            )
        )
        residue = residue_payload(
            component_id="ALA",
            residue_id=residue_id,
            atoms=atoms,
            is_hetero=ordinal in retained_indices,
        )
        if ordinal in retained_indices:
            ligands.append(residue)
        else:
            chains.append(chain_payload(residue_id.chain_id, (residue,)))
    return build_bonded_structure(
        chains=tuple(chains), ligands=tuple(ligands), source_format=FileFormat.MMCIF
    )


@pytest.mark.parametrize(
    "residue_id",
    [
        ResidueId("A", 1),
        ResidueId("AA", 1),
        ResidueId("BA", 1),
        ResidueId("AB", 1),
        ResidueId("AA", -1, "b"),
        ResidueId("_", 1),
    ],
)
def test_native_failure_and_known_bonds_refer_to_canonical_atoms(
    residue_id: ResidueId,
) -> None:
    source = _failing_structure((residue_id,))
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is False
    assert tuple(w.residue_id for w in probe.parser_problem_witnesses()) == (
        residue_id,
    )
    assert probe.parser_problem_witnesses()[0].atom_names == ("N",)
    heavy = tuple(
        b
        for b in probe.parser_problem_witnesses()[0].proximity_bonds
        if b.is_heavy_heavy()
    )
    assert len(heavy) == 1
    assert heavy[0].is_known_component_bond
    assert heavy[0].residue_ids() == (residue_id,)
    assert (
        diagnostics.measure_rdkit_no_conect_extra_heavy_proximity_bond_count(source)
        == 0
    )
    assert diagnostics.rdkit_no_conect_parser_failing_residue_ids(source) == (
        residue_id,
    )


def test_native_chain_collisions_and_ter_serial_gaps_keep_distinct_residues() -> None:
    residue_ids = (ResidueId("A", 1), ResidueId("AA", 1), ResidueId("BA", 1))
    source = _failing_structure(residue_ids)
    rows = [
        row
        for row in pdb_without_conect(source).splitlines()
        if row.startswith("ATOM  ")
    ]
    assert int(rows[6][6:11]) != 7
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert tuple(w.residue_id for w in probe.parser_problem_witnesses()) == residue_ids
    assert (
        diagnostics.measure_rdkit_no_conect_extra_heavy_proximity_bond_count(source)
        == 0
    )


def test_native_atom_order_is_not_the_source_write_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _failing_structure((ResidueId("AA", 1),))
    reader = Chem.MolFromPDBBlock

    def reversed_reader(
        block: str, *, sanitize: bool, removeHs: bool, proximityBonding: bool
    ) -> Mol | None:
        molecule = reader(
            block,
            sanitize=sanitize,
            removeHs=removeHs,
            proximityBonding=proximityBonding,
        )
        return (
            None
            if molecule is None
            else Chem.RenumberAtoms(
                molecule, list(reversed(range(molecule.GetNumAtoms())))
            )
        )

    monkeypatch.setattr(Chem, "MolFromPDBBlock", reversed_reader)
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert tuple(w.residue_id for w in probe.parser_problem_witnesses()) == (
        ResidueId("AA", 1),
    )
    assert probe.parser_problem_witnesses()[0].atom_names == ("N",)
    assert not probe.parser_problem_witnesses()[0].extra_heavy_proximity_bonds()


def test_interleaved_polymer_and_retained_records_keep_canonical_targets() -> None:
    residue_ids = (ResidueId("AA", 1), ResidueId("B", 1), ResidueId("AA", 2))
    source = _failing_structure(residue_ids, retained_indices=frozenset({2}))
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert tuple(w.residue_id for w in probe.parser_problem_witnesses()) == (
        residue_ids[0],
        residue_ids[2],
        residue_ids[1],
    )
    assert all(
        w.atom_names == ("N",) and not w.extra_heavy_proximity_bonds()
        for w in probe.parser_problem_witnesses()
    )


def test_native_failure_builds_a_repair_candidate_for_the_original_retained_residue() -> (
    None
):
    residue_id = ResidueId("AA", 7)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                is_hetero=True,
                atoms=(
                    atom_payload("N1", "N", Vec3(0, 0, 0)),
                    atom_payload("C1", "C", Vec3(1.45, 0, 0)),
                    atom_payload("H1", "H", Vec3(-1, 0, 0)),
                    atom_payload("H2", "H", Vec3(0, 1, 0)),
                    atom_payload("H3", "H", Vec3(0, -1, 0)),
                    atom_payload("H4", "H", Vec3(0, 0, 1)),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    candidates = parser_witness_repair_candidates(
        source, component_library=build_default_component_library()
    )
    assert len(candidates) == 1
    assert candidates[0].cluster.residue_ids == (residue_id,)
    assert candidates[0].repair_refinement.scope_spec == LocalScopeSpec.from_residues(
        (residue_id,)
    )


@pytest.mark.parametrize(
    "fault",
    [
        "foreign_serial",
        "duplicate_serial",
        "name",
        "element",
        "isotope",
        "missing_info",
    ],
)
def test_broken_native_correspondence_is_an_internal_invariant_failure(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _failing_structure((ResidueId("A", 1),))
    reader = Chem.MolFromPDBBlock

    def corrupting_reader(
        block: str, *, sanitize: bool, removeHs: bool, proximityBonding: bool
    ) -> Mol | None:
        molecule = reader(
            block,
            sanitize=sanitize,
            removeHs=removeHs,
            proximityBonding=proximityBonding,
        )
        assert molecule is not None
        atom = molecule.GetAtomWithIdx(0)
        if fault == "foreign_serial":
            atom.GetPDBResidueInfo().SetSerialNumber(0)
        elif fault == "duplicate_serial":
            molecule.GetAtomWithIdx(1).GetPDBResidueInfo().SetSerialNumber(
                atom.GetPDBResidueInfo().GetSerialNumber()
            )
        elif fault == "name":
            atom.GetPDBResidueInfo().SetName("BAD ")
        elif fault == "element":
            atom.SetAtomicNum(6)
        elif fault == "isotope":
            molecule.GetAtomWithIdx(2).SetIsotope(3)
        else:
            atom.SetMonomerInfo(Chem.AtomMonomerInfo())
        return molecule

    monkeypatch.setattr(Chem, "MolFromPDBBlock", corrupting_reader)
    with pytest.raises(ModelInvariantError, match="PDB.*correspondence"):
        diagnostics.probe_rdkit_no_conect_parser_readability(source)


@pytest.mark.parametrize("isotope", ["H", "D", "T"])
def test_retained_hydrogen_attachment_reaches_canonical_workflow_burden(
    isotope: str,
) -> None:
    residue_id = ResidueId("AA", 91)
    base = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("S1", "S", Vec3(1.8, 0, 0)),
                    atom_payload("HOBS", isotope, Vec3(1, 0.3, 0)),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    source = ProteinStructure.from_payload(
        constitution=base.constitution,
        geometry=base.geometry,
        topology=StructureTopology(
            constitution=base.constitution,
            atom_topologies=base.topology.atom_topologies,
            bonds=(
                TopologyBond(AtomIndex(0), AtomIndex(1)),
                TopologyBond(AtomIndex(0), AtomIndex(2)),
            ),
        ),
        polymer_blueprint=base.polymer_blueprint,
        provenance=base.provenance,
    )
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is True
    assert len(probe.hydrogen_attachment_witnesses) == 1
    witness = probe.hydrogen_attachment_witnesses[0]
    assert witness.atom_ref_1 == AtomRef(residue_id, "HOBS")
    assert witness.atom_ref_2 == AtomRef(residue_id, "S1")
    assert StructureParserCompatibilityFacts.from_structure(
        source
    ).has_parser_visible_proximity_burden()
    score = evaluate_workflow_branch_quality_score(
        TransformationResult(source, (), ()),
        requested_goal_report=RequestedGoalReport(()),
        planning_context=WorkflowPlanningContext(),
        component_library=None,
    )
    assert score.parser_extra_bond_count == 1


def test_cached_and_rebuilt_projection_bind_the_actual_current_source() -> None:
    source = _failing_structure((ResidueId("AA", 1),))
    projector = prepare_rdkit_no_conect_pdb_block_projector(source)
    assert projector is not None
    geometries = list(source.geometry.atom_geometries)
    geometries[0] = replace(geometries[0], position=Vec3(0.001, 0, 0))
    moved = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=StructureGeometry(
            constitution=source.constitution, atom_geometries=tuple(geometries)
        ),
        topology=source.topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    assert projector.render(moved) == pdb_without_conect(moved)
    cached = diagnostics.probe_rdkit_no_conect_parser_readability(
        moved, pdb_block_projector=projector
    )
    assert cached == diagnostics.probe_rdkit_no_conect_parser_readability(moved)
    assert cached.parser_problem_witnesses()[0].residue_id == ResidueId("AA", 1)
    changed = _failing_structure((ResidueId("BA", 1),))
    assert not projector.can_render(changed)
    actual = diagnostics.probe_rdkit_no_conect_parser_readability(
        changed, pdb_block_projector=projector
    )
    assert actual.parser_problem_witnesses()[0].residue_id == ResidueId("BA", 1)


def test_projection_rebuilds_after_h_inventory_or_pdb_metadata_changes() -> None:
    residue_id = ResidueId("AA", 1)
    source = _failing_structure((residue_id,))
    projector = prepare_rdkit_no_conect_pdb_block_projector(source)
    assert projector is not None
    residue_index = source.constitution.residue_index(residue_id)
    residue = source.constitution.residue_site_at(residue_index)
    changed_charge = source.with_updated_residue_facets(
        residue,
        residue_geometry=source.residue_geometry(residue_index),
        formal_charge_by_atom_name=(("N", 1),),
    )
    for changed in (
        source.without_atom_refs((AtomRef(residue_id, "H4"),)),
        _failing_structure((residue_id,), altloc="A"),
        changed_charge,
    ):
        assert not projector.can_render(changed)
        parser_input = pdb_projection.prepare_rdkit_no_conect_pdb_input(
            changed, projector
        )
        assert parser_input.pdb_block == pdb_without_conect(changed)
        assert diagnostics.probe_rdkit_no_conect_parser_readability(
            changed, pdb_block_projector=projector
        ) == diagnostics.probe_rdkit_no_conect_parser_readability(changed)


@pytest.mark.parametrize("fault", ["missing", "extra", "reordered", "duplicate"])
def test_broken_emission_is_not_guessed_from_native_residue_labels(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _failing_structure((ResidueId("A", 1),))
    writer = pdb_projection.write_pdb_structure_string_without_conect

    def corrupting_writer(structure: ProteinStructure) -> str:
        rows = writer(structure).splitlines()
        indices = [i for i, row in enumerate(rows) if row.startswith("ATOM  ")]
        first, second = indices[:2]
        if fault == "missing":
            del rows[first]
        elif fault == "extra":
            rows.insert(first, rows[first])
        elif fault == "reordered":
            rows[first], rows[second] = rows[second], rows[first]
        else:
            rows[second] = rows[second][:6] + rows[first][6:11] + rows[second][11:]
        return "\n".join(rows) + "\n"

    monkeypatch.setattr(
        pdb_projection, "write_pdb_structure_string_without_conect", corrupting_writer
    )
    with pytest.raises(ModelInvariantError, match="PDB.*correspondence"):
        diagnostics.probe_rdkit_no_conect_parser_readability(source)


@pytest.mark.parametrize("serial, parse_failed", [("99999", False), ("A0000", True)])
def test_native_serial_limit_does_not_renumber_the_parser_input(
    serial: str,
    parse_failed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _failing_structure((ResidueId("AA", 1),))
    writer = pdb_projection.write_pdb_structure_string_without_conect

    def supplied_writer(structure: ProteinStructure) -> str:
        text = writer(structure)
        first = True
        rows = []
        for row in text.splitlines():
            if first and row.startswith("ATOM  "):
                row = row[:6] + serial + row[11:]
                first = False
            rows.append(row)
        return "\n".join(rows) + "\n"

    monkeypatch.setattr(
        pdb_projection, "write_pdb_structure_string_without_conect", supplied_writer
    )
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is False
    if parse_failed:
        assert probe.structure_messages == ("_pdb_parse_failed_before_sanitize",)
        assert not probe.parser_problem_witnesses()
    else:
        assert probe.parser_problem_witnesses()[0].residue_id == ResidueId("AA", 1)


def test_native_filtered_cohort_semantics_are_not_changed_by_correspondence() -> None:
    source = _failing_structure((ResidueId("AA", 1),), altloc="B")
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is True
    assert probe.parser_problem_witnesses() == ()


def test_native_hybrid_residue_limit_remains_a_nonlocalized_parse_failure() -> None:
    source = _failing_structure((ResidueId("AA", 10000),))
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert probe.structure_messages == ("_pdb_parse_failed_before_sanitize",)
    assert not probe.parser_problem_witnesses()


def test_h_absent_observation_does_not_render_or_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _failing_structure((ResidueId("AA", 1),)).without_hydrogens()

    def unexpected_writer(structure: ProteinStructure) -> str:
        pytest.fail("H-absent observation must not construct parser input")

    monkeypatch.setattr(
        pdb_projection, "write_pdb_structure_string_without_conect", unexpected_writer
    )
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert probe.sanitize_readable is None
    assert not probe.parser_problem_witnesses()


def test_native_blank_chain_metadata_still_targets_the_canonical_default_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _failing_structure((ResidueId("_", 1),))
    writer = pdb_projection.write_pdb_structure_string_without_conect

    def blank_chain_writer(structure: ProteinStructure) -> str:
        return (
            "\n".join(
                row[:20] + "  " + row[22:] if row.startswith("ATOM  ") else row
                for row in writer(structure).splitlines()
            )
            + "\n"
        )

    monkeypatch.setattr(
        pdb_projection, "write_pdb_structure_string_without_conect", blank_chain_writer
    )
    probe = diagnostics.probe_rdkit_no_conect_parser_readability(source)
    assert probe.parser_problem_witnesses()[0].residue_id == ResidueId("_", 1)
