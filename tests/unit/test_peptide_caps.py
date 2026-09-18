"""Calculation-only peptide support and source projection behavior."""

from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from tests.support.correction_state_registry import CORRECTION_STATE_CASES

from protrepair.api import process_structure
from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.diagnostics import ValidationIssueKind
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import AtomTopology, StructureTopology, TopologyBond
from protrepair.transformer.artifacts import RegionTransformationResult, StructureDelta
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem
from protrepair.transformer.continuous.local_relaxation import (
    ContinuousLocalRelaxationTransformer,
)
from protrepair.transformer.continuous.peptide_boundary import PeptideBoundaryPlan
from protrepair.transformer.continuous.peptide_caps import PeptideCapTransformer
from protrepair.transformer.continuous.rdkit import (
    RdkitContinuousRelaxationBackend,
    build_rdkit_molecule,
)
from protrepair.transformer.continuous.readiness import (
    atom_scope_facts_continuous_relaxation_error,
    derive_atom_scope_continuous_relaxation_facts,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationForceField,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.continuous.shared import apply_position_updates


@pytest.fixture(scope="module")
def intact() -> ProteinStructure:
    return add_hydrogens(read_structure(Path("tests/fixtures/pdb/1aho.pdb"))).structure


@pytest.fixture(scope="module")
def cropped() -> ProteinStructure:
    return add_hydrogens(
        process_structure(
            Path(
                "tests/fixtures/pdb/refinement/3j6b_terminal_helix_misthread_local.pdb"
            )
        ).structure
    ).structure


def _problem(
    source: ProteinStructure, ids: tuple[ResidueId, ...], radius: float = 3
) -> ContinuousRelaxationProblem:
    atoms = tuple(
        index
        for rid in ids
        for index in source.constitution.atom_indices_for_residue_index(
            source.constitution.residue_index(rid)
        )
    )
    return ContinuousRelaxationProblem.from_inputs(
        ProteinStructureSnapshot.from_structure(source),
        AtomInput(atoms, AtomInputBasis.RESIDUEWISE, ResidueSetScope(ids)),
        spec=ContinuousRelaxationSettings(
            backend_name="rdkit",
            force_field=ContinuousRelaxationForceField.UFF,
            context_radius_angstrom=radius,
            max_iterations=2000,
        ),
        component_library=build_default_component_library(),
    )


def _caps(problem: ContinuousRelaxationProblem):
    plan = PeptideBoundaryPlan.from_region(
        problem.region, problem.bonds, build_default_component_library()
    )
    return PeptideCapTransformer().transform(problem, plan)


def _result(structure: ProteinStructure) -> RegionTransformationResult:
    return RegionTransformationResult(
        structure,
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
        ),
        (),
        "test",
    )


def _with_topology(
    source: ProteinStructure, topology: StructureTopology
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )


@pytest.mark.parametrize("radius", [0, 3, 100])
def test_source_backed_caps_close_native_graph_and_preserve_source(intact, radius):
    problem = _problem(intact, (ResidueId("A", 10),), radius)
    capped = _caps(problem)
    molecule, mapping = build_rdkit_molecule(capped.problem)
    Chem.SanitizeMol(molecule)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
    for boundary in capped.plan.boundaries:
        assert boundary.source_partner is not None
        if boundary.endpoint.atom_name == "N":
            index = intact.constitution.atom_index(boundary.endpoint)
            assert (
                molecule.GetAtomWithIdx(mapping[index]).GetHybridization()
                == Chem.HybridizationType.SP2
            )
    model = capped.problem.region.snapshot.structure
    projected = capped.project_result(_result(model)).refined_structure
    assert projected.constitution is intact.constitution
    assert projected.topology is intact.topology
    assert projected.provenance is intact.provenance
    assert projected.geometry == intact.geometry
    assert (
        capped.problem.region.movable_atom_indices
        == problem.region.movable_atom_indices
    )
    assert set(problem.region.included_atom_indices()) <= set(
        capped.problem.region.included_atom_indices()
    )
    if radius == 100:
        assert capped.problem is problem
    else:
        assert capped.plan.boundaries


def test_compact_caps_support_unknown_internal_sites_without_rewriting_facts(cropped):
    problem = _problem(cropped, tuple(ResidueId("9", n) for n in (149, 152, 235)), 100)
    capped = _caps(problem)
    molecule, _ = build_rdkit_molecule(capped.problem)
    Chem.SanitizeMol(molecule)
    assert len(capped.plan.boundaries) == 10
    assert all(b.source_partner is None for b in capped.plan.boundaries)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
    assert molecule.GetNumAtoms() == len(problem.region.included_atom_indices()) + 30
    facts = derive_atom_scope_continuous_relaxation_facts(
        problem.region.snapshot,
        problem.region.atom_input.observed_atom_scope(problem.region.snapshot),
        component_library=build_default_component_library(),
        context_radius_angstrom=100,
    )
    region = facts.continuous_region_readiness_facts
    assert any(
        f.has_unrealized_microstates()
        for f in region.chemistry_readiness_facts.residue_facts
    )
    assert not region.unsupported_polymer_residue_ids()
    assert atom_scope_facts_continuous_relaxation_error(facts) is None
    projected = capped.project_result(_result(capped.problem.region.snapshot.structure))
    assert projected.refined_structure.topology is cropped.topology
    assert projected.refined_structure.geometry == cropped.geometry
    assert not projected.delta.moved_atoms
    assert len(projected.issues) == 10
    assert all(
        i.kind is ValidationIssueKind.COMPUTATIONAL_PEPTIDE_CAP
        for i in projected.issues
    )


@pytest.mark.parametrize(
    "alter", ["cap", "fixed_source", "charge", "graph", "metadata"]
)
def test_projection_rejects_backend_changes_outside_coordinate_authority(intact, alter):
    problem = _problem(intact, (ResidueId("A", 10),))
    capped = _caps(problem)
    model = capped.problem.region.snapshot.structure
    if alter in {"cap", "fixed_source"}:
        index = (
            AtomIndex(len(intact.constitution.atom_slots))
            if alter == "cap"
            else problem.region.fixed_context_atom_indices[0]
        )
        changed = apply_position_updates(model, moved_positions={index: Vec3(0, 0, 0)})
    elif alter == "metadata":
        movable = problem.region.movable_atom_indices[0]
        changed = ProteinStructure.from_payload(
            constitution=model.constitution,
            geometry=StructureGeometry(
                constitution=model.constitution,
                atom_geometries=tuple(
                    replace(model.geometry.atom_geometry(AtomIndex(i)), occupancy=0.5)
                    if AtomIndex(i) == movable
                    else model.geometry.atom_geometry(AtomIndex(i))
                    for i in range(len(model.constitution.atom_slots))
                ),
            ),
            topology=model.topology,
            provenance=model.provenance,
        )
    else:
        topology = StructureTopology(
            constitution=model.constitution,
            atom_topologies=tuple(
                AtomTopology(0)
                if alter == "charge" and i == 0
                else model.topology.atom_topology(AtomIndex(i))
                for i in range(len(model.constitution.atom_slots))
            ),
            bonds=model.topology.bonds[:-1]
            if alter == "graph"
            else model.topology.bonds,
        )
        changed = _with_topology(model, topology)
    with pytest.raises(RefinementError, match="cap calculation"):
        capped.project_result(_result(changed))


def test_plan_cannot_be_reused_with_a_different_region(intact):
    problem = _problem(intact, (ResidueId("A", 10),))
    plan = PeptideBoundaryPlan.from_region(
        problem.region, problem.bonds, build_default_component_library()
    )
    other = _problem(intact, (ResidueId("A", 11),))
    with pytest.raises(RefinementError, match="different problem"):
        PeptideCapTransformer().transform(other, plan)


@pytest.mark.parametrize("alter", ["charge", "order", "external"])
def test_synthetic_cap_cannot_hide_conflicting_source_chemistry(cropped, alter):
    index = cropped.constitution.atom_index(AtomRef(ResidueId("9", 147), "N"))
    ca = cropped.constitution.atom_index(AtomRef(ResidueId("9", 147), "CA"))
    bonds = list(cropped.topology.bonds)
    if alter == "order":
        bonds = [
            replace(b, order=2) if set(b.endpoint_pair()) == {index, ca} else b
            for b in bonds
        ]
    if alter == "external":
        other = cropped.constitution.atom_index(AtomRef(ResidueId("9", 152), "CB"))
        bonds.append(TopologyBond(index, other))
    topology = StructureTopology(
        constitution=cropped.constitution,
        atom_topologies=tuple(
            AtomTopology(1)
            if alter == "charge" and AtomIndex(i) == index
            else cropped.topology.atom_topology(AtomIndex(i))
            for i in range(len(cropped.constitution.atom_slots))
        ),
        bonds=tuple(bonds),
    )
    changed = _with_topology(cropped, topology)
    problem = _problem(changed, (ResidueId("9", 149),), 100)
    facts = derive_atom_scope_continuous_relaxation_facts(
        problem.region.snapshot,
        problem.region.atom_input.observed_atom_scope(problem.region.snapshot),
        component_library=build_default_component_library(),
        context_radius_angstrom=100,
    )
    assert atom_scope_facts_continuous_relaxation_error(facts) is not None
    transformer = ContinuousLocalRelaxationTransformer(
        problem.spec,
        build_default_component_library(),
        build_default_restraint_library(),
        RdkitContinuousRelaxationBackend(),
    )
    context = ProteinTransformationContext.from_snapshot_atom_input(
        problem.region.snapshot, problem.region.atom_input
    )
    with pytest.raises(RefinementError):
        transformer.transform(context)


@pytest.mark.parametrize(
    "method", [ContinuousRelaxationForceField.UFF, ContinuousRelaxationForceField.MMFF]
)
def test_direct_transformer_uses_caps_and_returns_only_original_atoms(intact, method):
    problem = _problem(intact, (ResidueId("A", 10),))
    problem = replace(
        problem,
        spec=ContinuousRelaxationSettings(
            backend_name="rdkit",
            force_field=method,
            context_radius_angstrom=problem.spec.context_radius_angstrom,
            max_iterations=problem.spec.max_iterations,
        ),
    )
    transformer = ContinuousLocalRelaxationTransformer(
        problem.spec,
        build_default_component_library(),
        build_default_restraint_library(),
        RdkitContinuousRelaxationBackend(),
    )
    context = ProteinTransformationContext.from_snapshot_atom_input(
        problem.region.snapshot, problem.region.atom_input
    )
    result = transformer.transform(context)
    assert result.refined_structure.constitution is intact.constitution
    assert result.refined_structure.topology is intact.topology
    assert result.delta.before_constitution is intact.constitution
    assert result.delta.after_constitution is intact.constitution
    assert any(
        i.kind is ValidationIssueKind.COMPUTATIONAL_PEPTIDE_CAP for i in result.issues
    )
    movable = set(problem.region.movable_atom_indices)
    assert result.delta.moved_atoms
    for i in range(len(intact.constitution.atom_slots)):
        index = AtomIndex(i)
        if index not in movable:
            assert result.refined_structure.geometry.atom_geometry(
                index
            ) == intact.geometry.atom_geometry(index)


def test_source_backed_cap_accepts_named_h_without_repairing_source_bonds(intact):
    problem = _problem(intact, (ResidueId("A", 10),))
    boundary = next(
        b for b in _caps(problem).plan.boundaries if b.source_partner_hydrogen
    )
    hydrogen = intact.constitution.atom_index(boundary.source_partner_hydrogen)
    changed = _with_topology(
        intact,
        StructureTopology(
            constitution=intact.constitution,
            atom_topologies=intact.topology.atom_topologies,
            bonds=tuple(
                b for b in intact.topology.bonds if hydrogen not in b.endpoint_pair()
            ),
        ),
    )
    capped = _caps(_problem(changed, (ResidueId("A", 10),)))
    molecule, _ = build_rdkit_molecule(capped.problem)
    Chem.SanitizeMol(molecule)
    assert all(a.GetNumRadicalElectrons() == 0 for a in molecule.GetAtoms())
    projected = capped.project_result(_result(capped.problem.region.snapshot.structure))
    assert projected.refined_structure.topology is changed.topology
    assert projected.refined_structure.geometry == changed.geometry


def test_internal_n_missing_source_h_gets_only_calculation_h(tmp_path):
    source_path = Path(
        "tests/fixtures/pdb/refinement/3j6b_terminal_helix_misthread_local.pdb"
    )
    cropped_path = tmp_path / "missing-boundary-h.pdb"
    cropped_path.write_text(
        "".join(
            line
            for line in source_path.read_text().splitlines(keepends=True)
            if not (
                line.startswith("ATOM")
                and line[22:26].strip() == "147"
                and line[12:16].strip() == "H"
            )
        )
    )
    source = add_hydrogens(process_structure(cropped_path).structure).structure
    ref = AtomRef(ResidueId("9", 147), "N")
    assert source.constitution.resolve_atom_index(AtomRef(ref.residue_id, "H")) is None
    problem = _problem(source, (ResidueId("9", 149),), 100)
    capped = _caps(problem)
    assert any(
        b.endpoint == ref and b.supplemental_hydrogen for b in capped.plan.boundaries
    )
    molecule, _ = build_rdkit_molecule(capped.problem)
    Chem.SanitizeMol(molecule)
    assert all(a.GetNumRadicalElectrons() == 0 for a in molecule.GetAtoms())
    projected = capped.project_result(_result(capped.problem.region.snapshot.structure))
    assert projected.refined_structure.constitution is source.constitution
    assert projected.refined_structure.topology is source.topology


def test_canonical_h_parent_excludes_a_nearer_geometric_parent():
    source = CORRECTION_STATE_CASES["topology-blocked-preparation"].build_structure(
        build_default_component_library()
    )
    problem = _problem(source, (ResidueId("A", 1), ResidueId("B", 1)))
    for chain in ("A", "B"):
        h = source.constitution.atom_index(AtomRef(ResidueId(chain, 1), "HB1"))
        parents = [
            source.constitution.atom_ref_at(other).atom_name
            for bond in problem.bonds
            if h in (bond.atom_index_1, bond.atom_index_2)
            for other in (bond.atom_index_1, bond.atom_index_2)
            if other != h
        ]
        assert parents == ["CB"]


def test_caps_do_not_invent_support_for_non_peptide_cuts(intact):
    first = intact.constitution.atom_index(AtomRef(ResidueId("A", 9), "CB"))
    second = intact.constitution.atom_index(AtomRef(ResidueId("A", 1), "CB"))
    source = _with_topology(
        intact,
        StructureTopology(
            constitution=intact.constitution,
            atom_topologies=intact.topology.atom_topologies,
            bonds=(*intact.topology.bonds, TopologyBond(first, second)),
        ),
    )
    with pytest.raises(RefinementError, match="non-peptide"):
        _caps(_problem(source, (ResidueId("A", 10),), 0))


def test_degenerate_synthetic_cap_frame_fails_without_changing_source(cropped):
    n = cropped.constitution.atom_index(AtomRef(ResidueId("9", 147), "N"))
    ca = cropped.constitution.atom_index(AtomRef(ResidueId("9", 147), "CA"))
    source = apply_position_updates(
        cropped,
        moved_positions={
            ca: cropped.geometry.atom_geometry(n).position,
        },
    )
    with pytest.raises(RefinementError, match="degenerate"):
        _caps(_problem(source, (ResidueId("9", 149),), 100))
    assert source.topology is cropped.topology
