"""Local candidate selection observes retained contacts independently of FF mode."""

import numpy as np
import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import (
    ComponentLibrary,
    RestraintLibrary,
    build_default_component_library,
)
from protrepair.chemistry.component.graph import (
    BondDefinition,
    ChemicalComponentDefinition,
)
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.radii import UnknownElementRadiusError
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    detect_clashes_from_context,
    prepare_clash_detection_basis,
)
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope
from protrepair.structure import ProteinStructure, StructureTopology
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import TopologyBond
from protrepair.transformer.artifacts import (
    MovedAtomDelta,
    RegionTransformationResult,
    StructureDelta,
)
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.binding_policy import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationForceField,
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.continuous.shared import apply_position_updates
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    profile_transform_local_region,
    transform_local_region,
)
from protrepair.transformer.refinement.acceptance import (
    assess_refinement_result,
    measure_refinement_acceptance_metrics_for_scope,
)
from protrepair.transformer.refinement.local_pipeline.assessment import (
    execute_and_assess_refinement_candidate_batch,
    select_refinement_candidate,
)
from protrepair.transformer.refinement.local_pipeline.candidates import (
    ExecutedRefinementCandidate,
    RefinementExecutionBatch,
    RefinementExecutionCandidate,
)
from protrepair.transformer.refinement.local_pipeline.lineage import (
    RefinementCandidateLineage,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
)
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.transformer.refinement.speculative_planning import SpeculativeExecution
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.engine.runtime import execute_workflow_transformer


def _fragments(
    *,
    element: str = "C",
    polymer_count: int = 0,
    hydrogen_element: str = "H",
    centers: tuple[float, ...] = (0.0, 2.65),
    reverse_ligands: bool = False,
) -> tuple[ProteinStructure, ComponentLibrary]:
    # Complete tetrahedral CH4/NH4+ graphs isolate contact assessment from FF
    # optimization. Polymer-labelled slots test role coverage, not protein physics.
    library = build_default_component_library()
    payloads = []
    edges = []
    rays = np.asarray(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))
    for slot, center in enumerate(centers):
        heavy_element = element if slot == 0 else "C"
        charge = 1 if heavy_element == "N" else 0
        length = 1.03 if heavy_element == "N" else 1.09
        positions = (
            Vec3(0, 0, 0),
            *(Vec3.from_iterable(row * length / np.sqrt(3)) for row in rays),
        )
        names = ("X0", "H1", "H2", "H3", "H4")
        bonds = tuple(BondDefinition("X0", name) for name in names[1:])
        component_id = f"R{slot}"
        rid = ResidueId("A" if slot < polymer_count else "L", slot + 1)
        ideal = IdealizedComponent(
            component_id,
            None,
            tuple(
                IdealizedComponentAtom(
                    name,
                    heavy_element if i == 0 else "H",
                    charge if i == 0 else 0,
                    ideal_position=(position.x, position.y, position.z),
                )
                for i, (name, position) in enumerate(zip(names, positions, strict=True))
            ),
            bonds,
        )
        library = library.with_template(
            ResidueTemplate(
                ChemicalComponentDefinition(component_id, ("X0",), (), {"X0": charge}),
                hydrogen_semantics=IdealGeometryHydrogenSemantics(ideal),
            )
        )
        payloads.append(
            residue_payload(
                component_id=component_id,
                residue_id=rid,
                is_hetero=slot >= polymer_count,
                atoms=tuple(
                    atom_payload(
                        name,
                        heavy_element if i == 0 else hydrogen_element,
                        position.with_offset(center, 0, 0),
                        formal_charge=charge if i == 0 else 0,
                    )
                    for i, (name, position) in enumerate(
                        zip(names, positions, strict=True)
                    )
                ),
            )
        )
        edges.extend((AtomRef(rid, "X0"), AtomRef(rid, name)) for name in names[1:])
    source = build_structure(
        chains=(chain_payload("A", tuple(payloads[:polymer_count])),)
        if polymer_count
        else (),
        ligands=tuple(reversed(payloads[polymer_count:]))
        if reverse_ligands
        else tuple(payloads[polymer_count:]),
        source_format=FileFormat.MMCIF,
    )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(
                TopologyBond(
                    source.constitution.atom_index(left),
                    source.constitution.atom_index(right),
                )
                for left, right in edges
            ),
        ),
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    ), library


def _request(
    source: ProteinStructure, library: ComponentLibrary
) -> LocalRefinementRequest:
    rid = source.constitution.residue_slots[1].residue_id
    context = ProteinTransformationContext.from_snapshot_atom_input(
        ProteinStructureSnapshot.from_structure(source),
        AtomInput(
            tuple(source.constitution.atom_indices_for_residue(rid)),
            AtomInputBasis.RESIDUEWISE,
            ResidueSetScope((rid,)),
        ),
    )
    return LocalRefinementRequest.from_context(
        context,
        spec=ContinuousRelaxationSettings(
            profile=ContinuousRelaxationProfile.RDKIT_UFF
        ),
        component_library=library,
        restraint_library=RestraintLibrary(),
    )


def _frame(source: ProteinStructure, offset: float) -> RegionTransformationResult:
    rid = source.constitution.residue_slots[1].residue_id
    indices = source.constitution.atom_indices_for_residue(rid)
    return RegionTransformationResult(
        apply_position_updates(
            source,
            moved_positions={
                index: source.geometry.position(index).with_offset(offset, 0, 0)
                for index in indices
            },
        ),
        StructureDelta(
            source.constitution,
            source.constitution,
            moved_atoms=tuple(MovedAtomDelta(index, index) for index in indices),
        ),
        (),
        "rdkit",
    )


@pytest.fixture(params=[("C", 0), ("N", 0), ("C", 1), ("N", 1)])
def contact_pair(
    request: pytest.FixtureRequest,
) -> tuple[ProteinStructure, ComponentLibrary]:
    element, polymer_count = request.param
    return _fragments(element=element, polymer_count=polymer_count)


def test_standalone_acceptance_rejects_retained_contact_regression(
    contact_pair,
) -> None:
    source, library = contact_pair
    request = _request(source, library)
    worse = _frame(source, -0.30)
    assessment = assess_refinement_result(
        ProteinStructureSnapshot.from_structure(source),
        result=worse,
        selected_scope=request.context.atom_input.selected_scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
    )
    assert assessment.before_metrics.whole_structure_rdkit_sanitize_readable is True
    assert assessment.after_metrics.whole_structure_rdkit_sanitize_readable is True
    assert (
        assessment.before_metrics.whole_structure_parser_extra_proximity_bond_count == 0
    )
    assert (
        assessment.after_metrics.whole_structure_parser_extra_proximity_bond_count == 0
    )
    assert (
        assessment.after_metrics.focus_clash_count
        > assessment.before_metrics.focus_clash_count
        > 0
    )
    assert not assessment.is_accepted()


def test_candidate_batch_selects_contact_relief_over_an_adverse_endpoint(
    contact_pair, monkeypatch
) -> None:
    source, library = contact_pair
    request = _request(source, library)
    frames = (_frame(source, -0.30), _frame(source, 2.35))
    endpoints = iter(frames)
    visited = []

    def replay(
        self: RefinementExecutionCandidate, *, request: LocalRefinementRequest
    ) -> ExecutedRefinementCandidate:
        frame = next(endpoints)
        visited.append(frame)
        return SpeculativeExecution(self, frame)

    monkeypatch.setattr(RefinementExecutionCandidate, "execute", replay)
    batch = RefinementExecutionBatch(
        (
            RefinementExecutionCandidate(
                request.context, RefinementCandidateLineage(), source
            ),
            RefinementExecutionCandidate(
                request.context, RefinementCandidateLineage(), source
            ),
        )
    )
    assessed, _, _ = execute_and_assess_refinement_candidate_batch(
        batch, request=request
    )
    selected = select_refinement_candidate(assessed).require_candidate()
    assert visited == list(frames)
    assert selected.execution.outcome.refined_structure is frames[1].refined_structure
    assert selected.evaluation.is_accepted()
    assert selected.evaluation.accepted_metrics().focus_clash_count == 0


@pytest.mark.parametrize("runner", ["direct", "profile", "action"])
@pytest.mark.parametrize("offset", [-0.30, 2.35])
def test_user_paths_deliver_the_better_retained_contact_candidate(
    contact_pair, runner, offset, monkeypatch
) -> None:
    source, library = contact_pair
    endpoint = _frame(source, offset)

    def replay(
        self: RefinementExecutionCandidate, *, request: LocalRefinementRequest
    ) -> ExecutedRefinementCandidate:
        return SpeculativeExecution(self, endpoint)

    monkeypatch.setattr(RefinementExecutionCandidate, "execute", replay)
    rid = source.constitution.residue_slots[1].residue_id
    scope_spec = LocalScopeSpec.from_residues((rid,))
    if runner in {"direct", "profile"}:
        spec = DirectRegionTransformationSpec(
            scope_spec, ContinuousRelaxationForceField.UFF
        )
        if runner == "direct":
            result = transform_local_region(source, spec, component_library=library)
        else:
            result, _ = profile_transform_local_region(
                source, spec, component_library=library
            )
        delivered = result.refined_structure
    else:
        repair = RepairRefinementSpec(
            scope_spec, binding=RecommendedContinuousRelaxationBinding()
        )
        delivered = execute_workflow_transformer(
            TransformationResult(source, (), ()),
            transformer=LocalRefinementTransformer.from_repair_refinement(repair),
            execution_context=TransformerExecutionContext(
                component_library=library,
                original_structure=source,
                orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
            ),
        ).structure
    if offset > 0:
        assert delivered is endpoint.refined_structure
    else:
        assert delivered.geometry == source.geometry
    assert delivered.constitution is source.constitution
    assert delivered.topology is source.topology
    for index in source.constitution.atom_indices_for_residue(
        source.constitution.residue_slots[0].residue_id
    ):
        assert delivered.geometry.atom_geometry(index) == source.geometry.atom_geometry(
            index
        )


@pytest.mark.parametrize("hydrogen_element", ["H", "D", "T"])
def test_bound_and_standalone_metrics_share_retained_hydrogen_coverage(
    hydrogen_element,
) -> None:
    source, library = _fragments(hydrogen_element=hydrogen_element)
    request = _request(source, library)
    metrics = measure_refinement_acceptance_metrics_for_scope(
        _frame(source, -0.30).refined_structure,
        selected_scope=request.context.atom_input.selected_scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
    )
    cached = measure_refinement_acceptance_metrics_for_scope(
        _frame(source, -0.30).refined_structure,
        selected_scope=request.context.atom_input.selected_scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
        clash_basis=request.clash_basis,
        pdb_block_projector=request.pdb_block_projector,
    )
    assert metrics == cached
    assert metrics.focus_clash_count == 5
    heavy_basis = prepare_clash_detection_basis(
        source,
        component_library=library,
        policy=ClashPolicy(include_ligands=True, include_hydrogens=False),
    )
    heavy = measure_refinement_acceptance_metrics_for_scope(
        _frame(source, -0.30).refined_structure,
        selected_scope=request.context.atom_input.selected_scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
        clash_basis=heavy_basis,
    )
    assert heavy.focus_clash_count == 1


def test_explicit_contact_exclusion_remains_supported(contact_pair) -> None:
    source, library = contact_pair
    request = _request(source, library)
    basis = prepare_clash_detection_basis(
        source, component_library=library, policy=ClashPolicy(include_ligands=False)
    )
    assessment = assess_refinement_result(
        ProteinStructureSnapshot.from_structure(source),
        result=_frame(source, -0.30),
        selected_scope=request.context.atom_input.selected_scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
        clash_basis=basis,
    )
    assert assessment.before_metrics.focus_clash_count == 0
    assert assessment.after_metrics.focus_clash_count == 0
    assert assessment.is_accepted()


def test_no_template_contacts_are_not_reported_as_zero() -> None:
    source, _ = _fragments()
    metrics = measure_refinement_acceptance_metrics_for_scope(
        source,
        selected_scope=ResidueSetScope((ResidueId("L", 2),)),
        component_library=ComponentLibrary({}),
        restraint_library=RestraintLibrary(),
    )
    assert metrics.focus_clash_count == 1


def test_retained_contacts_outside_selected_focus_do_not_add_local_burden() -> None:
    source, library = _fragments(centers=(0, 2.65, 100))
    basis = prepare_clash_detection_basis(
        source, component_library=library, policy=ClashPolicy(include_ligands=True)
    )
    assert not detect_clashes_from_context(basis.bind_context(source)).is_empty()
    metrics = measure_refinement_acceptance_metrics_for_scope(
        source,
        selected_scope=ResidueSetScope((ResidueId("L", 3),)),
        component_library=library,
        restraint_library=RestraintLibrary(),
    )
    assert metrics.focus_clash_count == 0


def test_polymer_only_assessment_keeps_existing_contact_results() -> None:
    source, library = _fragments(polymer_count=2)
    scope = ResidueSetScope((ResidueId("A", 2),))
    original_basis = prepare_clash_detection_basis(source, component_library=library)
    actual = measure_refinement_acceptance_metrics_for_scope(
        source,
        selected_scope=scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
    )
    original = measure_refinement_acceptance_metrics_for_scope(
        source,
        selected_scope=scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
        clash_basis=original_basis,
    )
    assert actual == original
    assert actual.focus_clash_count == 1


def test_adverse_endpoint_retains_the_parent(contact_pair, monkeypatch) -> None:
    source, library = contact_pair
    worse = _frame(source, -0.30)

    def replay(
        self: RefinementExecutionCandidate, *, request: LocalRefinementRequest
    ) -> ExecutedRefinementCandidate:
        return SpeculativeExecution(self, worse)

    monkeypatch.setattr(RefinementExecutionCandidate, "execute", replay)
    delivered = transform_local_region(
        source,
        DirectRegionTransformationSpec(
            LocalScopeSpec.from_residues(
                (source.constitution.residue_slots[1].residue_id,)
            ),
            ContinuousRelaxationForceField.UFF,
        ),
        component_library=library,
    )
    assert delivered.refined_structure.geometry == source.geometry
    assert delivered.refined_structure.topology is source.topology


def test_candidate_batch_keeps_healthy_endpoint_over_a_later_adverse_one(
    monkeypatch,
) -> None:
    source, library = _fragments()
    request = _request(source, library)
    frames = (_frame(source, 2.35), _frame(source, -0.30))
    endpoints = iter(frames)
    visited = []

    def replay(
        self: RefinementExecutionCandidate, *, request: LocalRefinementRequest
    ) -> ExecutedRefinementCandidate:
        frame = next(endpoints)
        visited.append(frame)
        return SpeculativeExecution(self, frame)

    monkeypatch.setattr(RefinementExecutionCandidate, "execute", replay)
    batch = RefinementExecutionBatch(
        tuple(
            RefinementExecutionCandidate(
                request.context, RefinementCandidateLineage(), source
            )
            for _ in frames
        )
    )
    assessed, _, _ = execute_and_assess_refinement_candidate_batch(
        batch, request=request
    )
    assert visited == list(frames)
    assert (
        select_refinement_candidate(assessed).require_candidate().execution.outcome
        is frames[0]
    )


def test_unknown_retained_radius_is_not_silently_zero_coverage() -> None:
    source, library = _fragments(element="X")
    with pytest.raises(UnknownElementRadiusError):
        measure_refinement_acceptance_metrics_for_scope(
            source,
            selected_scope=ResidueSetScope((ResidueId("L", 2),)),
            component_library=library,
            restraint_library=RestraintLibrary(),
        )


def test_assessment_rejects_a_basis_for_different_chemistry() -> None:
    source, library = _fragments()
    request = _request(source, library)
    other, other_library = _fragments(element="N")
    basis = prepare_clash_detection_basis(
        other, component_library=other_library, policy=ClashPolicy(include_ligands=True)
    )
    with pytest.raises(ValueError, match="clash.*basis"):
        measure_refinement_acceptance_metrics_for_scope(
            source,
            selected_scope=request.context.atom_input.selected_scope,
            component_library=library,
            restraint_library=RestraintLibrary(),
            clash_basis=basis,
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_retained_contact_metrics_do_not_depend_on_ligand_slot_order(reverse) -> None:
    source, library = _fragments(reverse_ligands=reverse)
    metrics = measure_refinement_acceptance_metrics_for_scope(
        source,
        selected_scope=ResidueSetScope((ResidueId("L", 2),)),
        component_library=library,
        restraint_library=RestraintLibrary(),
    )
    assert metrics.focus_clash_count == 1
    assert metrics.focus_clash_overlap_sum_angstrom == pytest.approx(0.75)


def test_rigid_motion_preserves_retained_contact_assessment() -> None:
    source, library = _fragments()
    request = _request(source, library)
    transformed = apply_position_updates(
        source,
        moved_positions={
            index: Vec3(-point.y + 1000, point.x - 500, point.z + 125)
            for index in (
                AtomIndex(i) for i in range(len(source.constitution.atom_slots))
            )
            for point in (source.geometry.position(index),)
        },
    )
    expected = measure_refinement_acceptance_metrics_for_scope(
        source,
        selected_scope=request.context.atom_input.selected_scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
        clash_basis=request.clash_basis,
    )
    actual = measure_refinement_acceptance_metrics_for_scope(
        transformed,
        selected_scope=request.context.atom_input.selected_scope,
        component_library=library,
        restraint_library=RestraintLibrary(),
        clash_basis=request.clash_basis,
    )
    assert actual.focus_clash_count == expected.focus_clash_count == 1
    assert actual.focus_clash_overlap_sum_angstrom == pytest.approx(
        expected.focus_clash_overlap_sum_angstrom
    )


@pytest.mark.parametrize("offset, expected", [(-1e-7, 1), (1e-7, 0)])
def test_supplied_heavy_policy_keeps_its_contact_boundary(offset, expected) -> None:
    source, library = _fragments()
    policy = ClashPolicy(include_ligands=True, include_hydrogens=False)
    basis = prepare_clash_detection_basis(
        source, component_library=library, policy=policy
    )
    threshold = basis.bind_context(source).allowed_distance_by_element_pair[("C", "C")]
    candidate = _frame(source, threshold - 2.65 + offset)
    metrics = measure_refinement_acceptance_metrics_for_scope(
        candidate.refined_structure,
        selected_scope=ResidueSetScope((ResidueId("L", 2),)),
        component_library=library,
        restraint_library=RestraintLibrary(),
        clash_basis=basis,
    )
    assert metrics.focus_clash_count == expected
