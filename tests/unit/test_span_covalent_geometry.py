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
from protrepair.diagnostics import RepairEventKind, ValidationIssueKind
from protrepair.diagnostics.events import IssueSeverity, ValidationIssue
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
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.completion import span_reconstruction as kernel
from protrepair.transformer.completion.span_reconstruction import (
    ReconstructedSpanCandidate,
    SpanReconstructionFailure,
    SpanReconstructionFailureKind,
    reconstruct_donor_span,
)
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.spec import (
    BackboneWindowRefinementSpec,
    RepairRefinementSpec,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.backbone_window_refinement import (
    BackboneWindowRefinementTransformer,
)
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    WorkflowTransformRequests,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/pdb/span-reconstruction"


@pytest.mark.parametrize("junction", [35, 36, 37, 38])
@pytest.mark.parametrize("workflow", [False, True])
def test_span_rejects_disconnected_donor_window(junction: int, workflow: bool) -> None:
    reference = read_structure(FIXTURES / "1ubq.pdb")
    gap = tuple(ResidueId("A", number) for number in (36, 37, 38))
    source = _source(reference, gap)
    scope = AbsentResidueSpanScope(ResidueId("A", 35), ResidueId("A", 39), gap)
    left = reference.constitution.atom_index(AtomRef(ResidueId("A", junction), "C"))
    right = reference.constitution.atom_index(
        AtomRef(ResidueId("A", junction + 1), "N")
    )
    removed = reference.topology.bond_between(left, right)
    assert removed is not None
    donor = _with_topology_bonds(
        reference, tuple(bond for bond in reference.topology.bonds if bond != removed)
    )
    assert donor.geometry == reference.geometry

    if workflow:
        result = process_structure(
            source,
            transform_requests=WorkflowTransformRequests(
                external_span_reconstructions=(
                    ExternalSpanReconstructionSpec(scope, donor, gap),
                ),
            ),
        )
        assert result.structure.constitution == source.constitution
        assert result.structure.geometry == source.geometry
        assert result.structure.topology == source.topology
        assert not result.repairs
        assert len(result.issues) == 1
        assert result.issues[0].kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
        assert f"A:{junction}.C" in result.issues[0].message
    else:
        outcome = reconstruct_donor_span(
            source,
            scope=scope,
            donor_structure=donor,
            donor_residue_ids=gap,
            donor_preceding_residue_id=scope.preceding_residue_id,
            donor_following_residue_id=scope.following_residue_id,
        )
        assert isinstance(outcome, SpanReconstructionFailure)
        assert outcome.kind is SpanReconstructionFailureKind.INVALID_PEPTIDE_JUNCTION
        assert f"A:{junction}.C" in outcome.message


@pytest.mark.parametrize(
    ("relationship", "order", "aromatic"),
    [
        (BondRelationshipType.HYDROGEN_BOND, 1, False),
        (BondRelationshipType.METAL_COORDINATION, 1, False),
        (BondRelationshipType.UNKNOWN, 1, False),
        (BondRelationshipType.DISULFIDE, 1, False),
        (BondRelationshipType.COVALENT, None, False),
        (BondRelationshipType.COVALENT, 2, False),
        (BondRelationshipType.COVALENT, 1, True),
    ],
)
def test_donor_junction_requires_resolved_peptide_chemistry(
    relationship: BondRelationshipType, order: int | None, aromatic: bool
) -> None:
    reference = read_structure(FIXTURES / "1ubq-short-gaps.pdb")
    gap = (ResidueId("A", 4),)
    source = _source(reference, gap)
    scope = AbsentResidueSpanScope(ResidueId("A", 3), ResidueId("A", 5), gap)
    left = reference.constitution.atom_index(AtomRef(ResidueId("A", 3), "C"))
    right = reference.constitution.atom_index(AtomRef(gap[0], "N"))
    original = reference.topology.bond_between(left, right)
    assert original is not None
    donor = _with_topology_bonds(
        reference,
        tuple(
            replace(
                bond, relationship_type=relationship, order=order, aromatic=aromatic
            )
            if bond == original
            else bond
            for bond in reference.topology.bonds
        ),
    )
    outcome = reconstruct_donor_span(
        source,
        scope=scope,
        donor_structure=donor,
        donor_residue_ids=gap,
        donor_preceding_residue_id=scope.preceding_residue_id,
        donor_following_residue_id=scope.following_residue_id,
    )
    assert isinstance(outcome, SpanReconstructionFailure)
    assert outcome.kind is SpanReconstructionFailureKind.INVALID_PEPTIDE_JUNCTION


@pytest.mark.parametrize("workflow", [False, True])
@pytest.mark.parametrize(
    "provenance", [BondProvenance.SOURCE_EXPLICIT, BondProvenance.SEQUENCE_INFERRED]
)
def test_span_rejects_existing_source_anchor_shortcut(
    workflow: bool, provenance: BondProvenance
) -> None:
    reference = read_structure(FIXTURES / "1crn.pdb")
    gap = (ResidueId("A", 15), ResidueId("A", 16))
    source = _source(reference, gap)
    scope = AbsentResidueSpanScope(ResidueId("A", 14), ResidueId("A", 17), gap)
    shortcut = TopologyBond(
        source.constitution.atom_index(AtomRef(ResidueId("A", 14), "C")),
        source.constitution.atom_index(AtomRef(ResidueId("A", 17), "N")),
        provenance=provenance,
    )
    source = _with_topology_bonds(source, (*source.topology.bonds, shortcut))
    if workflow:
        result = process_structure(
            source,
            transform_requests=WorkflowTransformRequests(
                external_span_reconstructions=(
                    ExternalSpanReconstructionSpec(scope, reference, gap),
                ),
            ),
        )
        assert result.structure.constitution == source.constitution
        assert result.structure.geometry == source.geometry
        assert result.structure.topology == source.topology
        assert not result.repairs
        assert len(result.issues) == 1
        assert result.issues[0].kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
        assert "already" in result.issues[0].message
    else:
        outcome = reconstruct_donor_span(
            source,
            scope=scope,
            donor_structure=reference,
            donor_residue_ids=gap,
            donor_preceding_residue_id=scope.preceding_residue_id,
            donor_following_residue_id=scope.following_residue_id,
        )
        assert isinstance(outcome, SpanReconstructionFailure)
        assert outcome.kind is SpanReconstructionFailureKind.INVALID_TARGET_STATE


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("junction", [3, 4])
def test_terminal_span_rejects_disconnected_donor_window(
    prefix: bool, junction: int
) -> None:
    reference = read_structure(FIXTURES / "1ubq-short-gaps.pdb")
    gap = tuple(ResidueId("A", number) for number in ((3, 4) if prefix else (4, 5)))
    removed_ids = tuple(
        residue.residue_id
        for residue in reference.constitution.chains[0].residues
        if (
            residue.residue_id.seq_num <= 4
            if prefix
            else residue.residue_id.seq_num >= 4
        )
    )
    source = _source(reference, removed_ids)
    scope = AbsentResidueSpanScope(
        None if prefix else ResidueId("A", 3),
        ResidueId("A", 5) if prefix else None,
        gap,
    )
    removed_bond = reference.topology.bond_between(
        reference.constitution.atom_index(AtomRef(ResidueId("A", junction), "C")),
        reference.constitution.atom_index(AtomRef(ResidueId("A", junction + 1), "N")),
    )
    assert removed_bond is not None
    donor = _with_topology_bonds(
        reference,
        tuple(bond for bond in reference.topology.bonds if bond != removed_bond),
    )
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(scope, donor, gap),
            ),
        ),
    )
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert not result.repairs
    assert len(result.issues) == 1
    assert f"A:{junction}.C" in result.issues[0].message


@pytest.mark.parametrize(
    ("reverse", "relationship"),
    [
        (True, BondRelationshipType.COVALENT),
        (False, BondRelationshipType.HYDROGEN_BOND),
        (False, BondRelationshipType.METAL_COORDINATION),
    ],
)
def test_source_anchor_guard_preserves_other_connection_roles(
    reverse: bool, relationship: BondRelationshipType
) -> None:
    reference = read_structure(FIXTURES / "1crn.pdb")
    gap = (ResidueId("A", 15), ResidueId("A", 16))
    source = _source(reference, gap)
    scope = AbsentResidueSpanScope(ResidueId("A", 14), ResidueId("A", 17), gap)
    left_number, right_number = (17, 14) if reverse else (14, 17)
    left_ref = AtomRef(ResidueId("A", left_number), "C")
    right_ref = AtomRef(ResidueId("A", right_number), "N")
    connection = TopologyBond(
        source.constitution.atom_index(left_ref),
        source.constitution.atom_index(right_ref),
        relationship_type=relationship,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = _with_topology_bonds(source, (*source.topology.bonds, connection))
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(scope, reference, gap),
            ),
        ),
    )
    assert not result.issues
    assert all(
        result.structure.constitution.residue_or_ligand(rid) is not None for rid in gap
    )
    carried = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(left_ref),
        result.structure.constitution.atom_index(right_ref),
    )
    assert carried is not None
    assert carried.relationship_type is relationship
    assert carried.provenance is BondProvenance.SOURCE_EXPLICIT
    for ref in (left_ref, right_ref):
        assert _position(result.structure, ref.residue_id, ref.atom_name) == _position(
            source, ref.residue_id, ref.atom_name
        )


def _with_topology_bonds(
    structure: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


@pytest.mark.parametrize("start", [4, 26, 49, 72])
@pytest.mark.parametrize("native", [False, True])
def test_real_short_gap_closure_preserves_source_and_donor_chemistry(
    start: int, native: bool
) -> None:
    reference = read_structure(FIXTURES / "1ubq-short-gaps.pdb")
    donor = (
        reference if native else read_structure(FIXTURES / "af-p0cg48-short-gaps.pdb")
    )
    gap = (ResidueId("A", start),)
    source = _source(reference, gap)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=AbsentResidueSpanScope(
                        preceding_residue_id=ResidueId("A", start - 1),
                        following_residue_id=ResidueId("A", start + 1),
                        absent_residue_ids=gap,
                    ),
                    donor_structure=donor,
                    donor_residue_ids=gap,
                ),
            ),
        ),
    )
    for index, atom in enumerate(source.geometry.atom_geometries):
        ref = source.constitution.atom_ref_at(AtomIndex(index))
        assert (
            _position(result.structure, ref.residue_id, ref.atom_name) == atom.position
        )
    if native and start == 72:
        assert result.structure.constitution == source.constitution
        assert result.structure.topology == source.topology
        assert len(result.issues) == 1
        assert "intrinsic heavy-atom geometry" in result.issues[0].message
        return
    assert not result.issues
    residue = donor.constitution.residue_or_ligand(gap[0])
    assert residue is not None
    definition = (
        build_default_component_library().require(residue.component_id).definition
    )
    for bond in definition.bonds:
        if not all(
            residue.has_atom_site(name) for name in (bond.atom_name_1, bond.atom_name_2)
        ):
            continue
        assert _position(result.structure, gap[0], bond.atom_name_1).distance_to(
            _position(result.structure, gap[0], bond.atom_name_2)
        ) == pytest.approx(
            _position(donor, gap[0], bond.atom_name_1).distance_to(
                _position(donor, gap[0], bond.atom_name_2)
            ),
            abs=1e-8,
        )
    for name in residue.atom_site_names():
        neighbors = sorted(
            definition.bonded_atom_names(name) & frozenset(residue.atom_site_names())
        )
        for left, right in combinations(neighbors, 2):
            refs = (
                AtomRef(gap[0], left),
                AtomRef(gap[0], name),
                AtomRef(gap[0], right),
            )
            assert _angle(result.structure, refs) == pytest.approx(
                _angle(donor, refs), abs=1e-8
            )
    assert (
        Chem.MolFromPDBBlock(
            write_structure_string(result.structure, FileFormat.PDB),
            sanitize=True,
            removeHs=False,
            proximityBonding=True,
        )
        is not None
    )


def _short_gap_objective(start: int = 4) -> kernel._AnchoredSpanObjective:
    reference = read_structure(FIXTURES / "1ubq-short-gaps.pdb")
    donor = read_structure(FIXTURES / "af-p0cg48-short-gaps.pdb")
    return _closure_objective(reference, donor, start)


def _closure_objective(
    reference: ProteinStructure, donor: ProteinStructure, start: int
) -> kernel._AnchoredSpanObjective:
    ids = tuple(ResidueId("A", n) for n in (start - 1, start, start + 1))
    scope = AbsentResidueSpanScope(
        preceding_residue_id=ids[0],
        absent_residue_ids=(ids[1],),
        following_residue_id=ids[2],
    )
    source = _source(reference, scope.absent_residue_ids)
    window = kernel._build_donor_window(
        donor,
        ids,
        preceding_offset=0,
        following_offset=2,
        component_library=build_default_component_library(),
    )
    assert not isinstance(window, SpanReconstructionFailure)
    seed = kernel._place_donor_window_in_source_frame(
        window,
        source_structure=source,
        scope=scope,
        preceding_offset=0,
        following_offset=2,
    )
    assert not isinstance(seed, SpanReconstructionFailure)
    return window.anchored_objective(
        source,
        source_anchor_ids=(ids[0], ids[2]),
        seed=seed,
        axes=window.internal_span_rotation_axes(
            preceding_offset=0,
            first_inserted_offset=1,
            inserted_residue_count=1,
            following_offset=2,
        ),
    )


@pytest.mark.parametrize("start", [4, 26, 72])
def test_joint_closure_resolves_failed_ccd_without_relaxing_either_anchor(
    start: int,
) -> None:
    objective = _short_gap_objective(start)
    settings = kernel.SpanClosureSettings()
    coarse = kernel._fit_endpoint_by_cyclic_coordinate_descent(
        objective.seed,
        rotation_axes=objective.axes,
        endpoint_indices=objective.anchor_indices[1],
        endpoint_targets=objective.anchor_targets[1],
        settings=settings,
    )
    assert not isinstance(coarse, SpanReconstructionFailure)
    assert coarse[1] > settings.endpoint_rmsd_tolerance_angstrom
    before = objective.seed.copy()
    result = objective.fit(settings)
    assert not isinstance(result, SpanReconstructionFailure)
    coordinates, reported, _ = result
    rmsds = objective.anchor_rmsds(coordinates)
    assert reported == max(rmsds)
    assert all(value <= settings.endpoint_rmsd_tolerance_angstrom for value in rmsds)
    np.testing.assert_array_equal(objective.seed, before)


@pytest.mark.parametrize("anchor", [0, 1])
def test_joint_closure_rejects_one_unreachable_anchor_even_if_average_is_small(
    anchor: int,
) -> None:
    objective = _short_gap_objective()
    targets = objective.seed[objective.anchor_indices].copy()
    targets[anchor] += np.array([0.15, 0.0, 0.0])
    fixed = objective.fixed_targets.copy()
    fixed[objective.anchor_rows] = targets
    fixed_objective = replace(objective, fixed_targets=fixed)
    rmsds = fixed_objective.anchor_rmsds(objective.seed)
    assert np.mean(rmsds) < 0.1 < max(rmsds)
    # An internally stretched anchor cannot be made congruent by any rigid pose.
    targets[anchor, 0] += np.array([10.0, 0.0, 0.0])
    fixed[objective.anchor_rows] = targets
    result = replace(objective, fixed_targets=fixed).fit(kernel.SpanClosureSettings())
    assert isinstance(result, SpanReconstructionFailure)
    assert result.kind is SpanReconstructionFailureKind.NON_CONVERGENT_CLOSURE


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e308])
def test_joint_closure_rejects_unusable_objective_values(value: float) -> None:
    objective = _short_gap_objective()
    targets = objective.fixed_targets.copy()
    targets[objective.anchor_rows[0, 0], 0] = value
    result = replace(objective, fixed_targets=targets).fit(kernel.SpanClosureSettings())
    assert isinstance(result, SpanReconstructionFailure)
    assert result.kind is SpanReconstructionFailureKind.NON_FINITE_COORDINATES


@pytest.mark.parametrize("start", [26, 72])
def test_joint_fit_is_rigid_frame_equivariant(start: int) -> None:
    objective = _short_gap_objective(start)
    rotation = AxisRotation.from_points(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 2.0, -1.0))

    def move(points: kernel.FloatArray) -> kernel.FloatArray:
        shape = points.shape
        return np.asarray(
            [
                rotation.rotate_point(
                    Vec3.from_iterable(point),
                    origin=Vec3(0.0, 0.0, 0.0),
                    theta_radians=1.2,
                ).to_array()
                + np.array([11.0, -8.0, 3.0])
                for point in points.reshape(-1, 3)
            ]
        ).reshape(shape)

    transformed = replace(
        objective,
        seed=move(objective.seed),
        fixed_targets=move(objective.fixed_targets),
    )
    original_fit = objective.fit(kernel.SpanClosureSettings())
    transformed_fit = transformed.fit(kernel.SpanClosureSettings())
    assert not isinstance(original_fit, SpanReconstructionFailure)
    assert not isinstance(transformed_fit, SpanReconstructionFailure)
    np.testing.assert_allclose(
        transformed_fit[0], move(original_fit[0]), atol=2e-5, rtol=0.0
    )
    assert transformed_fit[1] == pytest.approx(original_fit[1], abs=2e-5)


def test_anchor_bounds_recover_a_failed_unconstrained_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _short_gap_objective(72)
    settings = kernel.SpanClosureSettings()
    with monkeypatch.context() as context:
        context.setattr(
            kernel._AnchoredSpanObjective,
            "_fit_anchor_bounds",
            lambda self, parameters, settings: (parameters, 0),
        )
        failure = objective.fit(settings)
    assert isinstance(failure, SpanReconstructionFailure)
    assert failure.kind is SpanReconstructionFailureKind.NON_CONVERGENT_CLOSURE
    result = objective.fit(settings)
    assert not isinstance(result, SpanReconstructionFailure)
    assert max(objective.anchor_rmsds(result[0])) <= 0.1


def test_already_feasible_joint_fit_does_not_run_constrained_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError("already feasible joint fit must remain unchanged")

    monkeypatch.setattr(kernel._AnchoredSpanObjective, "_fit_anchor_bounds", forbidden)
    assert not isinstance(
        _short_gap_objective(4).fit(kernel.SpanClosureSettings()),
        SpanReconstructionFailure,
    )


@pytest.mark.parametrize("anchor", [0, 1])
def test_constrained_step_checks_both_inequalities(anchor: int) -> None:
    objective = _short_gap_objective()
    residuals = np.zeros(20)
    jacobian = np.zeros((20, 2))
    for index in range(2):
        residuals[index * 9] = np.sqrt(3) * 0.1
        jacobian[index * 9, index] = np.sqrt(3)
    residuals[18 + anchor] = -1.0
    jacobian[18 + anchor, anchor] = 1.0
    result = objective._constrained_step(residuals, jacobian, bound=0.1, damping=0.01)
    assert result is not None
    step, multiplier = result
    assert step[anchor] == pytest.approx(0.0, abs=1e-10)
    assert step[1 - anchor] < 0.0
    assert multiplier > 0.0


def test_constrained_step_enforces_two_simultaneous_active_bounds() -> None:
    objective = _short_gap_objective()
    residuals = np.zeros(20)
    jacobian = np.zeros((20, 2))
    for index in range(2):
        residuals[index * 9] = np.sqrt(3) * 0.1
        jacobian[index * 9, index] = np.sqrt(3)
        residuals[18 + index] = -1.0
        jacobian[18 + index, index] = 1.0
    result = objective._constrained_step(residuals, jacobian, bound=0.1, damping=0.01)
    assert result is not None
    np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
    assert result[1] > 0.0


def test_constrained_step_rejects_incompatible_linearized_bounds() -> None:
    objective = _short_gap_objective()
    residuals = np.zeros(18)
    jacobian = np.zeros((18, 1))
    residuals[[0, 9]] = np.sqrt(3) * 0.2
    jacobian[0, 0], jacobian[9, 0] = np.sqrt(3), -np.sqrt(3)
    assert (
        objective._constrained_step(residuals, jacobian, bound=0.1, damping=0.01)
        is None
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("failure", ["derivative", "step", "merit"])
def test_constrained_fallback_failures_preserve_the_seed(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    objective = _short_gap_objective(72)
    before = objective.seed.copy()
    if failure == "derivative":
        monkeypatch.setattr(
            kernel._AnchoredSpanObjective, "_jacobian", lambda self, parameters: None
        )
        result = objective._fit_anchor_bounds(
            np.zeros(6 + len(objective.axes)), kernel.SpanClosureSettings()
        )
    else:

        def broken_step(self: object, *args: object, **kwargs: object):
            if failure == "step":
                return None
            return np.zeros(6 + len(objective.axes)), np.inf

        monkeypatch.setattr(
            kernel._AnchoredSpanObjective, "_constrained_step", broken_step
        )
        result = objective.fit(kernel.SpanClosureSettings())
    assert isinstance(result, SpanReconstructionFailure)
    assert result.kind is (
        SpanReconstructionFailureKind.NON_CONVERGENT_CLOSURE
        if failure == "step"
        else SpanReconstructionFailureKind.NON_FINITE_COORDINATES
    )
    np.testing.assert_array_equal(objective.seed, before)


def test_constrained_anchor_limits_remain_user_configurable() -> None:
    objective = _short_gap_objective(72)
    result = objective.fit(
        kernel.SpanClosureSettings(endpoint_rmsd_tolerance_angstrom=0.09)
    )
    assert not isinstance(result, SpanReconstructionFailure)
    assert max(objective.anchor_rmsds(result[0])) <= 0.09


def test_junction_residuals_use_actual_fixed_source_atoms() -> None:
    objective = _short_gap_objective()
    parameters = np.zeros(6 + len(objective.axes))
    residuals = objective.residuals(parameters)
    assert residuals is not None
    actual = objective.seed.copy()
    actual[objective.fixed_indices] = objective.fixed_targets
    pairs = objective.junction_pairs
    expected = (
        np.linalg.norm(actual[pairs[:, 0]] - actual[pairs[:, 1]], axis=1)
        - objective.junction_distances
    )
    assert np.linalg.norm(expected) > 0.1
    np.testing.assert_allclose(residuals[18:], expected, atol=1e-12)


def test_joint_objective_includes_source_proline_substituent_and_preserves_ring() -> (
    None
):
    reference = read_structure(FIXTURES / "1ubq.pdb")
    objective = _closure_objective(reference, reference, 36)
    cd = _position(reference, ResidueId("A", 37), "CD").to_array()
    cd_row = int(np.flatnonzero(np.all(objective.fixed_targets == cd, axis=1))[0])
    cd_index = objective.fixed_indices[cd_row]
    assert np.count_nonzero(objective.junction_pairs[:, 1] == cd_index) == 3
    nitrogen_index = objective.anchor_indices[1, 0]
    assert all(axis.start_index != nitrogen_index for axis in objective.axes)
    moved_targets = objective.fixed_targets.copy()
    moved_targets[min(objective.anchor_rows[1]) :] += np.array([0.1, 0.0, 0.0])
    result = replace(objective, fixed_targets=moved_targets).fit(
        kernel.SpanClosureSettings()
    )
    assert not isinstance(result, SpanReconstructionFailure)
    coordinates = result[0]
    assert np.linalg.norm(
        coordinates[nitrogen_index] - coordinates[cd_index]
    ) == pytest.approx(
        np.linalg.norm(objective.seed[nitrogen_index] - objective.seed[cd_index]),
        abs=1e-10,
    )


def test_joint_solver_failure_is_atomic_and_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _short_gap_objective()
    before = objective.seed.copy()

    def singular(
        matrix: kernel.FloatArray, rhs: kernel.FloatArray
    ) -> kernel.FloatArray:
        raise np.linalg.LinAlgError("singular numerical system")

    monkeypatch.setattr(np.linalg, "solve", singular)
    result = objective.fit(kernel.SpanClosureSettings())
    assert isinstance(result, SpanReconstructionFailure)
    assert result.kind is SpanReconstructionFailureKind.NON_CONVERGENT_CLOSURE
    np.testing.assert_array_equal(objective.seed, before)


def test_joint_candidate_still_passes_through_actual_junction_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = read_structure(FIXTURES / "1ubq-short-gaps.pdb")
    donor = read_structure(FIXTURES / "af-p0cg48-short-gaps.pdb")
    source = _source(reference, (ResidueId("A", 4),))

    def broken_fit(
        self: kernel._AnchoredSpanObjective, settings: kernel.SpanClosureSettings
    ) -> tuple[kernel.FloatArray, float, int]:
        coordinates = self.seed.copy()
        inserted_oxygen = self.junction_pairs[-1, 0]
        coordinates[inserted_oxygen] = self.anchor_targets[1, 0]
        return coordinates, 0.0, 1

    monkeypatch.setattr(kernel._AnchoredSpanObjective, "fit", broken_fit)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=AbsentResidueSpanScope(
                        preceding_residue_id=ResidueId("A", 3),
                        absent_residue_ids=(ResidueId("A", 4),),
                        following_residue_id=ResidueId("A", 5),
                    ),
                    donor_structure=donor,
                    donor_residue_ids=(ResidueId("A", 4),),
                ),
            ),
        ),
    )
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert len(result.issues) == 1
    assert "peptide-junction gate" in result.issues[0].message


@pytest.mark.parametrize("local", [False, True])
def test_refinement_request_survives_absent_span_until_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    local: bool,
) -> None:
    reference = read_structure(FIXTURES / "1ubq.pdb")
    gap = tuple(ResidueId("A", number) for number in (36, 37, 38))
    source = _source(reference, gap)
    calls = []

    def record_execution(
        self: WorkflowStructureTransformer,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        assert all(
            projected_domain.state.constitution.residue_or_ligand(rid) for rid in gap
        )
        assert any(
            event.kind is RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED
            for event in carrier.repairs
        )
        calls.append(self.workflow_scope)
        return ProjectedCodomainState(
            scope=self.workflow_scope,
            state=projected_domain.state,
            issues=(
                ValidationIssue(
                    kind=ValidationIssueKind.REFINEMENT_REJECTED,
                    severity=IssueSeverity.WARNING,
                    message="test backend rejected candidate",
                ),
            ),
        )

    monkeypatch.setattr(
        LocalRefinementTransformer if local else BackboneWindowRefinementTransformer,
        "transform_projected_domain",
        record_execution,
    )
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=AbsentResidueSpanScope(
                        preceding_residue_id=ResidueId("A", 35),
                        absent_residue_ids=gap,
                        following_residue_id=ResidueId("A", 39),
                    ),
                    donor_structure=reference,
                    donor_residue_ids=gap,
                ),
            ),
            backbone_window_refinements=()
            if local
            else (BackboneWindowRefinementSpec(gap),),
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues(gap),
                binding=RecommendedContinuousRelaxationBinding(),
            )
            if local
            else None,
        ),
    )
    assert calls
    assert any(
        issue.message == "test backend rejected candidate" for issue in result.issues
    )
    for index, atom in enumerate(source.geometry.atom_geometries):
        ref = source.constitution.atom_ref_at(AtomIndex(index))
        assert (
            _position(result.structure, ref.residue_id, ref.atom_name) == atom.position
        )


@pytest.mark.parametrize("local", [False, True])
def test_unmaterializable_refinement_request_reports_rejection(local: bool) -> None:
    reference = read_structure(FIXTURES / "1ubq.pdb")
    gap = tuple(ResidueId("A", number) for number in (36, 37, 38))
    source = _source(reference, gap)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            backbone_window_refinements=()
            if local
            else (BackboneWindowRefinementSpec(gap),),
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues(gap),
                binding=RecommendedContinuousRelaxationBinding(),
            )
            if local
            else None,
        ),
    )
    assert result.structure == source
    assert not result.repairs
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
        and "unknown residue A:" in issue.message
        for issue in result.issues
    )


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
            and not (
                line.startswith(("ATOM  ", "HETATM")) and int(line[22:26]) in numbers
            )
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
    assert outcome.maximum_anchor_rmsd_angstrom is not None
    assert outcome.maximum_anchor_rmsd_angstrom <= 0.1
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
