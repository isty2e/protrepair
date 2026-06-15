"""Unit tests for correction-phase planning over intrinsic and interaction facts."""

from pathlib import Path

from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.correction_state_fixtures import (
    load_refinement_fixture,
)
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.scenario_fixture_matrix import SCENARIO_FIXTURE_MATRIX

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.state import StructureChemistryReadinessFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts import (
    StructureIngressOptions,
    WorkflowLigandContextMode,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
)
from protrepair.workflow.planning.correction import plan_correction_transformers

SERINE_RESIDUE_ID = ResidueId("A", 17)


def test_correction_planner_emits_intrinsic_geometry_refinement_for_matrix_case(
) -> None:
    """Ligand-free intrinsic geometry cases should plan correction explicitly."""

    assert "3g8l-asn182" in SCENARIO_FIXTURE_MATRIX
    structure = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    residue_id = ResidueId("A", 182)
    component_library = build_default_component_library()

    outcome = plan_correction_transformers(
        structure,
        transform_requests=WorkflowTransformRequests(
            repair_refinement=_repair_refinement(residue_id)
        ),
        planning_context=WorkflowPlanningContext(),
        component_library=component_library,
        chemistry_readiness_facts=StructureChemistryReadinessFacts.from_structure(
            structure,
            component_library=component_library,
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], LocalRefinementTransformer)


def test_correction_planner_emits_ligand_aware_refinement_only_in_holo_context() -> (
    None
):
    """Ligand-aware clashes should plan correction only when ligands are kept."""

    assert "1jd0-gln92" in SCENARIO_FIXTURE_MATRIX
    structure = load_refinement_fixture("1jd0_gln92_local.pdb")
    residue_id = ResidueId("A", 92)
    component_library = build_default_component_library()
    chemistry_facts = StructureChemistryReadinessFacts.from_structure(
        structure,
        component_library=component_library,
    )

    holo_outcome = plan_correction_transformers(
        structure,
        transform_requests=WorkflowTransformRequests(
            repair_refinement=_repair_refinement(residue_id)
        ),
        planning_context=WorkflowPlanningContext(
            ligand_context_mode=WorkflowLigandContextMode.CONSIDER_IF_PRESENT,
        ),
        component_library=component_library,
        chemistry_readiness_facts=chemistry_facts,
    )
    apo_outcome = plan_correction_transformers(
        structure,
        transform_requests=WorkflowTransformRequests(
            repair_refinement=_repair_refinement(residue_id)
        ),
        planning_context=WorkflowPlanningContext(),
        component_library=component_library,
        chemistry_readiness_facts=chemistry_facts,
    )

    assert len(holo_outcome.transformers) == 1
    assert isinstance(holo_outcome.transformers[0], LocalRefinementTransformer)
    assert apo_outcome.transformers == ()


def test_correction_planner_emits_post_hydrogen_interaction_refinement_for_matrix_case(
) -> None:
    """Post-hydrogen interaction burden should plan correction explicitly."""

    assert "synthetic-ligand-clashing-serine-hydrogen" in SCENARIO_FIXTURE_MATRIX
    component_library = build_default_component_library()
    structure = _hydrogenated_holo_serine_clash_structure()

    outcome = plan_correction_transformers(
        structure,
        transform_requests=WorkflowTransformRequests(
            repair_refinement=_repair_refinement(SERINE_RESIDUE_ID)
        ),
        planning_context=WorkflowPlanningContext(
            ligand_context_mode=WorkflowLigandContextMode.CONSIDER_IF_PRESENT,
        ),
        component_library=component_library,
        chemistry_readiness_facts=StructureChemistryReadinessFacts.from_structure(
            structure,
            component_library=component_library,
        ),
    )

    assert len(outcome.transformers) == 1
    assert isinstance(outcome.transformers[0], LocalRefinementTransformer)


def _repair_refinement(residue_id: ResidueId) -> RepairRefinementSpec:
    """Return one canonical repair-refinement spec for local planning tests."""

    return RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        binding=ManualContinuousRelaxationBinding(
            ContinuousRelaxationForceField.UFF
        ),
    )


def _hydrogenated_holo_serine_clash_structure(
    ) -> ProteinStructure:
    """Return one hydrogenated holo serine window with a ligand clash on HG."""

    library = build_default_component_library()
    protein_only = _build_serine_window_structure()
    hydrogenated = add_hydrogens(
        protein_only,
        component_library=library,
    ).structure
    residue = _residue_payload_from_structure(hydrogenated, SERINE_RESIDUE_ID)
    ligand = _build_ligand_residue(
        "L",
        1,
        (_resolvable_ligand_atom(residue, component_library=library),),
    )
    return hydrogenated.with_ligand_facets(
        ligand_sites=(ligand[0],),
        ligand_geometries=(ligand[1],),
        ligand_formal_charge_payloads=(ligand[2],),
    )


def _build_serine_window_structure() -> ProteinStructure:
    """Return the same serine window fixture used by hydrogen cleanup tests."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    chain_site = structure.chain_site("A")
    center_index = next(
        index
        for index, residue_site in enumerate(chain_site.residues)
        if residue_site.residue_id == SERINE_RESIDUE_ID
    )
    window_residue_sites = chain_site.residues[
        max(0, center_index - 1) : center_index + 2
    ]
    window_payloads = tuple(
        _canonical_residue_payload_from_structure(
            structure,
            residue_site.residue_id,
        )
        for residue_site in window_residue_sites
    )
    return build_structure(
        chains=(chain_payload("A", window_payloads),),
        source_format=structure.provenance.ingress.source_format,
        source_name="pdb1afc-ser-window",
    )


def _build_ligand_residue(
    chain_id: str,
    seq_num: int,
    atoms: tuple[CanonicalAtomPayload, ...],
) -> CanonicalResiduePayload:
    """Return one ligand residue payload from prebuilt atoms."""

    return residue_payload(
        component_id="LIG",
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=True,
    )


def _canonical_residue_payload_from_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CanonicalResiduePayload:
    """Return one canonical residue payload from a canonical structure."""

    residue_site = structure.constitution.chain(residue_id.chain_id).residue(
        residue_id
    )
    residue_index = structure.constitution.residue_index(residue_id)
    return (
        residue_site,
        structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
        structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
    )


def _residue_payload_from_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CompletionResiduePayload:
    """Return one completion residue payload from a canonical structure."""

    residue_site = structure.constitution.chain(residue_id.chain_id).residue(
        residue_id
    )
    residue_index = structure.constitution.residue_index(residue_id)
    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
    )


def _resolvable_ligand_atom(
    residue: CompletionResiduePayload,
    *,
    component_library: ComponentLibrary,
) -> CanonicalAtomPayload:
    """Return one ligand atom positioned to clash with the serine HG hydrogen."""

    from protrepair.transformer.completion.hydrogen.rotatable import (
        build_rotatable_hydrogen_search,
        rotatable_hydrogen_placement_spec,
    )

    template = component_library.require("SER")
    spec = rotatable_hydrogen_placement_spec(template.hydrogen_semantics)
    assert spec is not None

    search = build_rotatable_hydrogen_search(residue, spec=spec)
    assert search is not None

    candidate = next(iter(search.candidate_positions()))
    offset = Vec3.from_iterable(candidate).with_offset(0.05, 0.0, 0.0)
    return atom_payload("C1", "C", offset)
