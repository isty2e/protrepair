"""Unit tests for joint local-correction scope proposals."""

from functools import lru_cache
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

import protrepair.transformer.continuous.rdkit as continuous_rdkit
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics import ClashPolicy
from protrepair.diagnostics.near_covalent import detect_near_covalent_contacts
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure, write_structure_string
from protrepair.scope import ResidueSetScope
from protrepair.state import StructureChemistryReadinessFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.local.models import LocalScopeLowering
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.contracts import (
    RequestedGoalSet,
    StructureIngressOptions,
    WorkflowPlanningContext,
    WorkflowTransformRequests,
)
from protrepair.workflow.planning.joint_correction_scopes import (
    JointCorrectionBatchingPolicy,
    JointCorrectionMotionClass,
    JointCorrectionScopeProposal,
    batch_joint_correction_scope_proposals,
    execution_scope_spec_for_joint_correction,
    propose_joint_correction_scopes,
)
from protrepair.workflow.planning.planner import plan_workflow_actions

RDKIT_AVAILABLE = continuous_rdkit.Chem is not None
REPRESENTATIVE_RESIDUE_IDS = (
    ResidueId("C", 11),
    ResidueId("C", 45),
)


@lru_cache(maxsize=1)
def _hydrogenated_1afc_structure() -> ProteinStructure:
    """Return the canonical hydrogenated 1AFC representative structure."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    return add_hydrogens(
        structure,
        component_library=build_default_component_library(),
        protonate_histidines=True,
    ).structure


def test_joint_correction_scope_proposals_cluster_representative_contact() -> None:
    """Representative severe contacts should expand into one joint residue scope."""

    component_library = build_default_component_library()
    structure = _hydrogenated_1afc_structure()
    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure,
        component_library=component_library,
    )

    proposals = propose_joint_correction_scopes(
        structure,
        focus_residue_ids=(ResidueId("C", 45),),
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )

    assert proposals
    assert proposals[0].residue_ids == REPRESENTATIVE_RESIDUE_IDS
    assert proposals[0].contact_pair_count >= 1
    assert proposals[0].worst_overlap_angstrom > 0.0


def test_joint_correction_scope_keeps_sidechain_only_contacts_sidechain_local() -> None:
    """Pure sidechain severe contacts should stay on sidechain-local lowering."""

    component_library = build_default_component_library()
    structure = _toy_sidechain_contact_structure()
    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure,
        component_library=component_library,
    )

    proposals = propose_joint_correction_scopes(
        structure,
        focus_residue_ids=(ResidueId("A", 1),),
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )

    assert proposals
    assert proposals[0].motion_class is JointCorrectionMotionClass.RESIDUE_SIDECHAINS
    assert (
        proposals[0].as_local_scope_spec().lowering
        is LocalScopeLowering.RESIDUE_SIDECHAIN_ATOMS
    )


def test_joint_correction_scope_promotes_backbone_contacts_to_residue_atoms() -> None:
    """Backbone-involving severe contacts should widen lowering to residue atoms."""

    component_library = build_default_component_library()
    structure = _toy_backbone_contact_structure()
    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure,
        component_library=component_library,
    )

    proposals = propose_joint_correction_scopes(
        structure,
        focus_residue_ids=(ResidueId("A", 1),),
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )

    assert proposals
    assert proposals[0].residue_ids == (ResidueId("A", 1), ResidueId("B", 2))
    assert proposals[0].contact_pair_count >= 1
    assert proposals[0].worst_overlap_angstrom > 1.0
    assert proposals[0].motion_class is JointCorrectionMotionClass.RESIDUE_ATOMS
    assert (
        proposals[0].as_local_scope_spec().lowering
        is LocalScopeLowering.RESIDUE_ATOMS
    )


def test_backbone_joint_execution_scope_widens_to_peptide_neighbors() -> None:
    """Backbone motion should widen execution scope without widening semantics."""

    component_library = build_default_component_library()
    structure = _toy_same_chain_backbone_contact_structure()
    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure,
        component_library=component_library,
    )

    proposals = propose_joint_correction_scopes(
        structure,
        focus_residue_ids=(ResidueId("A", 2),),
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )

    assert proposals
    assert proposals[0].residue_ids == (ResidueId("A", 2), ResidueId("A", 4))
    execution_scope_spec = execution_scope_spec_for_joint_correction(
        structure,
        proposal=proposals[0],
    )
    assert execution_scope_spec is not None
    assert execution_scope_spec.lowering is LocalScopeLowering.RESIDUE_ATOMS
    assert execution_scope_spec.scope == ResidueSetScope(
        residue_ids=(
            ResidueId("A", 1),
            ResidueId("A", 2),
            ResidueId("A", 3),
            ResidueId("A", 4),
        )
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_representative_no_conect_failure_is_covered_by_joint_scope_proposal() -> None:
    """The representative no-CONECT sanitize failure should expose a joint scope."""

    assert continuous_rdkit.Chem is not None
    component_library = build_default_component_library()
    structure = _hydrogenated_1afc_structure()
    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure,
        component_library=component_library,
    )
    pdb_text = write_structure_string(structure, FileFormat.PDB)
    pdb_without_conect = "\n".join(
        line for line in pdb_text.splitlines() if not line.startswith("CONECT")
    )
    parsed = continuous_rdkit.Chem.MolFromPDBBlock(
        pdb_without_conect + "\n",
        sanitize=True,
        removeHs=False,
        proximityBonding=True,
    )

    proposals = propose_joint_correction_scopes(
        structure,
        focus_residue_ids=(ResidueId("C", 45),),
        component_library=component_library,
        chemistry_readiness_facts=chemistry_readiness_facts,
    )

    assert parsed is None
    assert proposals
    assert proposals[0].residue_ids == REPRESENTATIVE_RESIDUE_IDS


def test_workflow_planner_emits_joint_refinement_scope_before_fallback() -> None:
    """Workflow planning should surface joint local-refinement proposals first."""

    repair_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId("C", 45),)),
        binding=ManualContinuousRelaxationBinding(
            ContinuousRelaxationForceField.UFF
        ),
    )

    outcome = plan_workflow_actions(
        _hydrogenated_1afc_structure(),
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=repair_refinement
        ),
        planning_context=WorkflowPlanningContext(),
    )

    assert outcome.transformers
    assert all(
        isinstance(transformer, LocalRefinementTransformer)
        for transformer in outcome.transformers
    )
    residue_scopes = tuple(
        transformer.scope.residue_ids
        for transformer in outcome.transformers
        if isinstance(transformer, LocalRefinementTransformer)
        and isinstance(transformer.scope, ResidueSetScope)
    )
    assert residue_scopes[0] == REPRESENTATIVE_RESIDUE_IDS
    assert residue_scopes[-1] == (ResidueId("C", 45),)


def test_batch_joint_correction_scope_proposals_merges_compatible_clusters() -> None:
    """Nearby severe clusters should batch into one merged correction proposal."""

    structure = _toy_batching_structure()
    proposals = (
        JointCorrectionScopeProposal(
            residue_ids=(ResidueId("A", 1), ResidueId("A", 2)),
            contact_pair_count=2,
            worst_overlap_angstrom=1.9,
            total_overlap_angstrom=5.2,
        ),
        JointCorrectionScopeProposal(
            residue_ids=(ResidueId("A", 3), ResidueId("A", 4)),
            contact_pair_count=2,
            worst_overlap_angstrom=1.8,
            total_overlap_angstrom=5.1,
        ),
    )

    batched = batch_joint_correction_scope_proposals(
        structure,
        proposals=proposals,
    )

    assert batched == (
        JointCorrectionScopeProposal(
            residue_ids=(
                ResidueId("A", 1),
                ResidueId("A", 2),
                ResidueId("A", 3),
                ResidueId("A", 4),
            ),
            contact_pair_count=4,
            worst_overlap_angstrom=1.9,
            total_overlap_angstrom=10.3,
        ),
    )


def test_batch_joint_correction_scope_proposals_keeps_incompatible_clusters_separate(
) -> None:
    """Distant severe clusters should remain separate proposals after batching."""

    structure = _toy_batching_structure()
    proposals = (
        JointCorrectionScopeProposal(
            residue_ids=(ResidueId("A", 1), ResidueId("A", 2)),
            contact_pair_count=2,
            worst_overlap_angstrom=1.9,
            total_overlap_angstrom=5.2,
        ),
        JointCorrectionScopeProposal(
            residue_ids=(ResidueId("A", 5), ResidueId("A", 6)),
            contact_pair_count=2,
            worst_overlap_angstrom=1.85,
            total_overlap_angstrom=5.0,
        ),
    )

    batched = batch_joint_correction_scope_proposals(
        structure,
        proposals=proposals,
    )

    assert tuple(proposal.residue_ids for proposal in batched) == (
        (ResidueId("A", 1), ResidueId("A", 2)),
        (ResidueId("A", 5), ResidueId("A", 6)),
    )


def test_batch_joint_correction_scope_proposals_filters_below_threshold() -> None:
    """Automatic batching should drop low-severity clusters before proposal."""

    structure = _toy_batching_structure()
    proposals = (
        JointCorrectionScopeProposal(
            residue_ids=(ResidueId("A", 1), ResidueId("A", 2)),
            contact_pair_count=2,
            worst_overlap_angstrom=1.4,
            total_overlap_angstrom=3.8,
        ),
        JointCorrectionScopeProposal(
            residue_ids=(ResidueId("A", 3), ResidueId("A", 4)),
            contact_pair_count=2,
            worst_overlap_angstrom=1.8,
            total_overlap_angstrom=5.1,
        ),
    )

    batched = batch_joint_correction_scope_proposals(
        structure,
        proposals=proposals,
        batching_policy=JointCorrectionBatchingPolicy(
            minimum_cluster_worst_overlap_angstrom=1.75,
            minimum_cluster_total_overlap_angstrom=5.0,
        ),
    )

    assert tuple(proposal.residue_ids for proposal in batched) == (
        (ResidueId("A", 3), ResidueId("A", 4)),
    )

def test_representative_joint_scope_contact_is_detected_as_near_covalent() -> None:
    """Representative sanitize failure should surface as one near-covalent contact."""

    component_library = build_default_component_library()
    structure = _hydrogenated_1afc_structure()
    near_covalent_contacts = detect_near_covalent_contacts(
        structure,
        component_library=component_library,
        focus_residue_ids=frozenset((ResidueId("C", 45),)),
        pair_policy=ClashPolicy(include_hydrogens=True),
    )

    assert near_covalent_contacts
    assert any(
        frozenset((clash.left_residue_id, clash.right_residue_id))
        == frozenset(REPRESENTATIVE_RESIDUE_IDS)
        for clash in near_covalent_contacts
    )


def _toy_batching_structure() -> ProteinStructure:
    """Return one toy structure with two nearby and one distant residue pair."""

    residues = (
        residue_payload(
            component_id="ALA",
            residue_id=ResidueId("A", 1),
            atoms=(atom_payload("CB", "C", Vec3(0.0, 0.0, 0.0)),),
        ),
        residue_payload(
            component_id="ALA",
            residue_id=ResidueId("A", 2),
            atoms=(atom_payload("CB", "C", Vec3(1.5, 0.0, 0.0)),),
        ),
        residue_payload(
            component_id="ALA",
            residue_id=ResidueId("A", 3),
            atoms=(atom_payload("CB", "C", Vec3(5.0, 0.0, 0.0)),),
        ),
        residue_payload(
            component_id="ALA",
            residue_id=ResidueId("A", 4),
            atoms=(atom_payload("CB", "C", Vec3(6.5, 0.0, 0.0)),),
        ),
        residue_payload(
            component_id="ALA",
            residue_id=ResidueId("A", 5),
            atoms=(atom_payload("CB", "C", Vec3(20.0, 0.0, 0.0)),),
        ),
        residue_payload(
            component_id="ALA",
            residue_id=ResidueId("A", 6),
            atoms=(atom_payload("CB", "C", Vec3(21.5, 0.0, 0.0)),),
        ),
    )

    return build_structure(
        chains=(chain_payload("A", residues),),
        source_format=FileFormat.PDB,
    )


def _toy_backbone_contact_structure() -> ProteinStructure:
    """Return one two-residue structure with one severe backbone-local contact."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(-1.2, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(-0.2, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.2, 1.5, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("B", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(2.0, 1.5, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _toy_same_chain_backbone_contact_structure() -> ProteinStructure:
    """Return one same-chain backbone contact with one peptide-side neighbor."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(-6.8, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(-5.8, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(-4.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(-4.0, 0.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(-1.2, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(-0.2, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.2, 1.5, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 3),
                        atoms=(
                            atom_payload("N", "N", Vec3(8.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(9.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(10.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(10.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(9.0, 1.5, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 4),
                        atoms=(
                            atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(2.0, 1.5, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _toy_sidechain_contact_structure() -> ProteinStructure:
    """Return one two-residue structure with one severe sidechain-local contact."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(-2.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(-1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(0.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(0.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="LEU",
                        residue_id=ResidueId("B", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(4.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(5.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(6.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(6.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(0.9, 2.4, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
