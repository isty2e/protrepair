"""Registry of correction-state cases over synthetic and extracted inputs."""

from protrepair.state.domain import (
    BackboneHeavyAtomCompletenessState,
    ClashState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    TopologyAvailabilityAspect,
    TopologyAvailabilityState,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.planning.transformation.legality import (
    LocalTransformationStratum,
)
from protrepair.workflow.planning.transformation.runtime import (
    LocalTransformationFamily,
    TransformationTerminationReason,
)
from tests.support.correction_state_dsl import (
    CorrectionCoverageTag,
    CorrectionStateCase,
    LocalExpectation,
    TopologyExpectation,
    WorkflowExpectation,
)
from tests.support.correction_state_fixtures import (
    build_chain,
    build_residue,
    build_structure,
    hydrogenated_refinement_fixture,
    residue_bond_specs,
    with_topology_bonds,
)

REQUIRED_CORRECTION_COVERAGE: frozenset[CorrectionCoverageTag] = frozenset(
    {
        CorrectionCoverageTag.UNSUPPORTED_STOP,
        CorrectionCoverageTag.HEAVY_INCOMPLETENESS,
        CorrectionCoverageTag.HYDROGEN_ONLY_WORKFLOW,
        CorrectionCoverageTag.HETEROGENEOUS_WORKFLOW,
        CorrectionCoverageTag.RELAXATION_READY,
        CorrectionCoverageTag.CHEMISTRY_PREPARATION,
        CorrectionCoverageTag.CANDIDATE_CONSTRUCTION,
    }
)


CORRECTION_STATE_CASES: dict[str, CorrectionStateCase] = {
    "unsupported-component-local-stop": CorrectionStateCase(
        case_id="unsupported-component-local-stop",
        description="Unsupported residue blocks local correction planning.",
        coverage_tags=(CorrectionCoverageTag.UNSUPPORTED_STOP,),
        structure_factory=lambda _library: build_structure(
            "unsupported-local-stop",
            (
                build_chain(
                    "A",
                    (
                        build_residue(
                            "UNK",
                            "A",
                            1,
                            ("N", "CA", "C", "O"),
                        ),
                    ),
                ),
            ),
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
            component_support_state=(
                ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
            ),
            backbone_heavy_atom_completeness_state=(
                BackboneHeavyAtomCompletenessState.COMPLETE
            ),
            sidechain_heavy_atom_completeness_state=(
                SidechainHeavyAtomCompletenessState.COMPLETE
            ),
            continuous_relaxation_ready=False,
            termination_reason=(
                TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
            ),
            topology_expectations=(
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                    state=TopologyAvailabilityState.UNSUPPORTED,
                ),
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    state=TopologyAvailabilityState.UNSUPPORTED,
                ),
            ),
        ),
    ),
    "backbone-incomplete-local-stop": CorrectionStateCase(
        case_id="backbone-incomplete-local-stop",
        description="Backbone-incomplete residues block continuous relaxation.",
        coverage_tags=(CorrectionCoverageTag.HEAVY_INCOMPLETENESS,),
        structure_factory=lambda _library: build_structure(
            "backbone-incomplete-local-stop",
            (
                build_chain(
                    "A",
                    (build_residue("ALA", "A", 1, ("N", "CA")),),
                ),
            ),
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
            backbone_heavy_atom_completeness_state=(
                BackboneHeavyAtomCompletenessState.INCOMPLETE
            ),
            sidechain_heavy_atom_completeness_state=(
                SidechainHeavyAtomCompletenessState.INCOMPLETE
            ),
            continuous_relaxation_ready=False,
            termination_reason=(
                TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
            ),
            topology_expectations=(
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                    state=TopologyAvailabilityState.ABSENT,
                ),
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    state=TopologyAvailabilityState.ABSENT,
                ),
            ),
        ),
    ),
    "hydrogen-only-workflow-continuation": CorrectionStateCase(
        case_id="hydrogen-only-workflow-continuation",
        description=(
            "Heavy-complete residues with no H need hydrogen-only workflow completion."
        ),
        coverage_tags=(CorrectionCoverageTag.HYDROGEN_ONLY_WORKFLOW,),
        structure_factory=lambda _library: with_topology_bonds(
            build_structure(
                "hydrogen-only-workflow",
                (
                    build_chain(
                        "A",
                        (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
                    ),
                ),
            ),
            *residue_bond_specs(
                ResidueId("A", 1),
                (("N", "CA"), ("CA", "C"), ("C", "O")),
            ),
        ),
        workflow=WorkflowExpectation(
            requests_hydrogen_population=True,
            execution_stage_values=("hydrogen_completion",),
            partition_kind_values=("hydrogen_only",),
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
            hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
            hydrogen_coverage_state=HydrogenCoverageState.NONE,
            continuous_relaxation_ready=False,
            termination_reason=(
                TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
            ),
            topology_expectations=(
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                    state=TopologyAvailabilityState.PRESENT,
                ),
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    state=TopologyAvailabilityState.ABSENT,
                ),
            ),
        ),
    ),
    "heterogeneous-workflow-partition": CorrectionStateCase(
        case_id="heterogeneous-workflow-partition",
        description=(
            "Mixed supported and unsupported residues partition whole-workflow "
            "completion."
        ),
        coverage_tags=(
            CorrectionCoverageTag.HETEROGENEOUS_WORKFLOW,
            CorrectionCoverageTag.HYDROGEN_ONLY_WORKFLOW,
        ),
        structure_factory=lambda _library: build_structure(
            "heterogeneous-workflow",
            (
                build_chain(
                    "A",
                    (
                        build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                        build_residue("UNK", "A", 2, ("N", "CA", "C", "O")),
                    ),
                ),
            ),
        ),
        workflow=WorkflowExpectation(
            requests_hydrogen_population=True,
            execution_stage_values=("hydrogen_completion",),
            partition_kind_values=("hydrogen_only", "unsupported_stop"),
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 1),)),
            hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
            hydrogen_coverage_state=HydrogenCoverageState.NONE,
            continuous_relaxation_ready=False,
            termination_reason=(
                TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
            ),
        ),
    ),
    "hydrogenated-template-ready": CorrectionStateCase(
        case_id="hydrogenated-template-ready",
        description=(
            "Hydrogenated Thr local fixture with clash-free coordinates should "
            "be relaxation-ready."
        ),
        coverage_tags=(CorrectionCoverageTag.RELAXATION_READY,),
        structure_factory=lambda component_library: hydrogenated_refinement_fixture(
            "1bkr-thr101",
            component_library=component_library,
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 101),)),
            clash_state=ClashState.NONE,
            hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
            hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
            continuous_relaxation_ready=True,
            legal_families=(
                LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION,
            ),
            legal_strata=(LocalTransformationStratum.RELAXATION,),
            topology_expectations=(
                TopologyExpectation(
                    residue_id=ResidueId("A", 101),
                    aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                    state=TopologyAvailabilityState.PRESENT,
                ),
                TopologyExpectation(
                    residue_id=ResidueId("A", 101),
                    aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    state=TopologyAvailabilityState.PRESENT,
                ),
            ),
        ),
    ),
    "topology-blocked-preparation": CorrectionStateCase(
        case_id="topology-blocked-preparation",
        description=(
            "Clashy hydrogenated coordinates should preserve graph-realizable "
            "topology and keep joint local correction legal."
        ),
        coverage_tags=(
            CorrectionCoverageTag.CHEMISTRY_PREPARATION,
            CorrectionCoverageTag.RELAXATION_READY,
        ),
        structure_factory=lambda _library: with_topology_bonds(
            build_structure(
                "topology-coordinate-blocked",
                (
                    build_chain(
                        "A",
                        (
                            build_residue(
                                "SER",
                                "A",
                                1,
                                (
                                    "N",
                                    "CA",
                                    "C",
                                    "O",
                                    "CB",
                                    "OG",
                                    "H1",
                                    "H2",
                                    "H3",
                                    "HA",
                                    "HB1",
                                    "HB2",
                                    "HG",
                                ),
                            ),
                        ),
                    ),
                    build_chain(
                        "B",
                        (
                            build_residue(
                                "SER",
                                "B",
                                1,
                                (
                                    "N",
                                    "CA",
                                    "C",
                                    "O",
                                    "CB",
                                    "OG",
                                    "H1",
                                    "H2",
                                    "H3",
                                    "HA",
                                    "HB1",
                                    "HB2",
                                    "HG",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            *residue_bond_specs(
                ResidueId("A", 1),
                (
                    ("N", "CA"),
                    ("CA", "C"),
                    ("C", "O"),
                    ("CA", "CB"),
                    ("CB", "OG"),
                    ("CA", "HA"),
                    ("CB", "HB1"),
                    ("CB", "HB2"),
                    ("OG", "HG"),
                ),
            ),
            *residue_bond_specs(
                ResidueId("B", 1),
                (
                    ("N", "CA"),
                    ("CA", "C"),
                    ("C", "O"),
                    ("CA", "CB"),
                    ("CB", "OG"),
                    ("CA", "HA"),
                    ("CB", "HB1"),
                    ("CB", "HB2"),
                    ("OG", "HG"),
                ),
            ),
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_residues(
                (ResidueId("A", 1), ResidueId("B", 1))
            ),
            clash_state=ClashState.PRESENT,
            continuous_relaxation_ready=True,
            discrete_preparation_applicable=True,
            legal_families=(
                LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION,
                LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION,
            ),
            legal_strata=(
                LocalTransformationStratum.PREPARATION,
                LocalTransformationStratum.RELAXATION,
            ),
            topology_expectations=(
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                    state=TopologyAvailabilityState.PRESENT,
                ),
                TopologyExpectation(
                    residue_id=ResidueId("B", 1),
                    aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                    state=TopologyAvailabilityState.PRESENT,
                ),
                TopologyExpectation(
                    residue_id=ResidueId("A", 1),
                    aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    state=TopologyAvailabilityState.PRESENT,
                ),
                TopologyExpectation(
                    residue_id=ResidueId("B", 1),
                    aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    state=TopologyAvailabilityState.PRESENT,
                ),
            ),
        ),
    ),
    "chemistry-preparation-amide-flip": CorrectionStateCase(
        case_id="chemistry-preparation-amide-flip",
        description=(
            "Hydrogenated Asn amide-flip fixture with clash-free coordinates "
            "should expose chemistry preparation before relaxation."
        ),
        coverage_tags=(CorrectionCoverageTag.CHEMISTRY_PREPARATION,),
        structure_factory=lambda component_library: hydrogenated_refinement_fixture(
            "1ywr-asn155",
            component_library=component_library,
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_atoms_with_attached_hydrogens(
                (
                    AtomRef(ResidueId("A", 155), "CB"),
                    AtomRef(ResidueId("A", 155), "CG"),
                    AtomRef(ResidueId("A", 155), "OD1"),
                    AtomRef(ResidueId("A", 155), "ND2"),
                )
            ),
            clash_state=ClashState.NONE,
            continuous_relaxation_ready=True,
            discrete_preparation_applicable=True,
            validate_discrete_preparation_detector=True,
            legal_families=(
                LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION,
                LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION,
            ),
            legal_strata=(
                LocalTransformationStratum.PREPARATION,
                LocalTransformationStratum.RELAXATION,
            ),
        ),
    ),
    "candidate-construction-branched-sidechain": CorrectionStateCase(
        case_id="candidate-construction-branched-sidechain",
        description="Hydrogenated Leu seed case should expose candidate construction.",
        coverage_tags=(CorrectionCoverageTag.CANDIDATE_CONSTRUCTION,),
        structure_factory=lambda component_library: hydrogenated_refinement_fixture(
            "1xgo-leu253",
            component_library=component_library,
        ),
        local=LocalExpectation(
            scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 253),)),
            continuous_relaxation_ready=True,
            discrete_seeding_applicable=True,
            validate_discrete_seeding_detector=True,
            legal_families=(
                LocalTransformationFamily.BRANCHED_SIDECHAIN_SEED,
                LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION,
            ),
            legal_strata=(
                LocalTransformationStratum.CANDIDATE_CONSTRUCTION,
                LocalTransformationStratum.RELAXATION,
            ),
        ),
    ),
}


def collected_correction_coverage_tags() -> frozenset[CorrectionCoverageTag]:
    """Return the route-precondition coverage tags exercised by the corpus."""

    return frozenset(
        coverage_tag
        for case in CORRECTION_STATE_CASES.values()
        for coverage_tag in case.coverage_tags
    )
