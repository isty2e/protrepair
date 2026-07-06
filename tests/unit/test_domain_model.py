"""Unit tests for the redesigned ProtRepair domain model."""

from typing import cast

import numpy as np
import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.request_builders import (
    ingress_options,
    transform_requests,
    whole_structure_requested_goals,
)

from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    EventScope,
    EventScopeKind,
    IssueSeverity,
    RepairEvent,
    RepairEventKind,
    ResidueAtomImpact,
    ValidationIssue,
    ValidationIssueKind,
)
from protrepair.errors import ModelInvariantError
from protrepair.geometry import Vec3
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    ClashPresenceState,
    ClashState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.structure import (
    AtomSite,
    ChainSite,
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
    ProteinStructure,
    ResidueSite,
)
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import (
    AtomIndex,
)
from protrepair.transformer.continuous.binding_policy import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.packing import PackingScope, PackingSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import (
    AnalysisBundle,
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
    OrphanFragmentPolicy,
    ProcessResult,
    RequestedGoalCompletionVerdict,
    RequestedGoalOutcome,
    RequestedGoalReport,
    RequestedGoalSet,
    RequestedGoalStatus,
    WorkflowTransformRequests,
    requested_process_goal,
)


def test_vec3_supports_distance_and_array_conversion() -> None:
    left = Vec3(0.0, 0.0, 0.0)
    right = Vec3.from_iterable([1.0, 2.0, 2.0])

    assert left.distance_to(right) == 3.0
    assert np.array_equal(
        right.to_array(),
        np.asarray([1.0, 2.0, 2.0], dtype=np.float64),
    )


def test_residue_site_supports_lookup_and_rejects_duplicate_atom_names() -> None:
    residue_site = ResidueSite(
        component_id="ALA",
        residue_id=ResidueId(chain_id="A", seq_num=10),
        atom_sites=(
            AtomSite(" N ", "N"),
            AtomSite("ca", "C"),
            AtomSite("C", "C"),
            AtomSite("XX", "C"),
        ),
    )

    assert residue_site.atom_site_names() == ("N", "CA", "C", "XX")
    assert residue_site.has_atom_site("ca")
    assert residue_site.atom_site(" CA ").name == "CA"

    with pytest.raises(ValueError, match="atom-site name must not be blank"):
        AtomSite(" ", "C")

    with pytest.raises(Exception, match="duplicate atom-site names"):
        ResidueSite(
            component_id="GLY",
            residue_id=ResidueId(chain_id="A", seq_num=1),
            atom_sites=(AtomSite("CA", "C"), AtomSite("CA", "C")),
        )


def test_residue_site_with_atom_site_replaces_existing_name() -> None:
    residue_site = ResidueSite(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atom_sites=(AtomSite("N", "N"), AtomSite("CA", "C")),
    )

    updated = residue_site.with_atom_site(AtomSite("CA", "N"))

    assert updated.atom_site("CA").element == "N"
    assert residue_site.atom_site("CA").element == "C"


def test_component_library_normalizes_aliases() -> None:
    definition = ChemicalComponentDefinition(
        component_id="HIS",
        atom_names=("N", "CA", "C"),
        aliases=("HSD", "HSE"),
    )
    template = ResidueTemplate(definition=definition)
    library = ComponentLibrary(templates={"HIS": template})

    assert library.normalize_component_id("hsd") == "HIS"
    assert library.require("HSE") == template


def test_standard_templates_expose_semantics_and_forcefield_data() -> None:
    library = build_standard_component_library()
    histidine_template = library.require("HSE")
    serine_template = library.require("SER")
    threonine_template = library.require("THR")

    assert histidine_template.component_id == "HIS"
    assert histidine_template.can_add_hydrogens()
    assert serine_template.can_repair_heavy_atoms()
    assert serine_template.has_forcefield_params("OG")
    assert threonine_template.has_tetrahedral_stereochemistry()
    assert serine_template.missing_atom_names(
        ("N", "CA", "C", "O"),
        exclude_atom_names=("OXT",),
    ) == ("CB", "OG")


def test_standard_templates_cover_heavy_repair_for_all_standard_residues() -> None:
    library = build_standard_component_library()
    standard_component_ids = (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    )

    assert all(
        library.require(component_id).can_repair_heavy_atoms()
        for component_id in standard_component_ids
    )


def test_chain_site_and_structure_constitution_support_navigation() -> None:
    residue_1 = residue_payload(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),),
    )
    residue_2 = residue_payload(
        component_id="ALA",
        residue_id=ResidueId(chain_id="A", seq_num=2),
        atoms=(atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),),
    )
    residue_3 = residue_payload(
        component_id="SER",
        residue_id=ResidueId(chain_id="B", seq_num=1),
        atoms=(atom_payload("N", "N", Vec3(2.0, 0.0, 0.0)),),
    )
    chain_a = ChainSite(
        chain_id="A",
        residues=(residue_1[0], residue_2[0]),
    )
    structure = build_structure(
        chains=(
            chain_payload("A", (residue_1, residue_2)),
            chain_payload("B", (residue_3,)),
        ),
        source_format=FileFormat.PDB,
        source_name="fixture",
    )

    assert chain_a.residue_ids() == (
        ResidueId(chain_id="A", seq_num=1),
        ResidueId(chain_id="A", seq_num=2),
    )
    assert chain_a.residue(ResidueId(chain_id="A", seq_num=2)) == residue_2[0]
    assert structure.constitution.chain_ids() == ("A", "B")
    assert structure.select_chains(("B",)).constitution.chain_ids() == ("B",)
    assert tuple(
        structure.constitution.atom_ref_at(AtomIndex(atom_index)).atom_name
        for atom_index in range(len(structure.constitution.atom_slots))
    ) == (
        "N",
        "N",
        "N",
    )


def test_protein_structure_requires_blueprint_chain_ids_to_match_constitution() -> None:
    """Canonical structure roots should reject mismatched polymer blueprints."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    with pytest.raises(ModelInvariantError, match="polymer blueprint"):
        ProteinStructure.from_payload(
            constitution=structure.constitution,
            geometry=structure.geometry,
            topology=structure.topology,
            polymer_blueprint=PolymerBlueprint(
                chains=(
                    PolymerChainBlueprint(
                        chain_id="X",
                        residue_slots=(
                            PolymerResidueSlot(sequence_position=1, token="G"),
                        ),
                    ),
                ),
            ),
            provenance=structure.provenance,
        )


def test_protein_structure_rejects_unassigned_reference_blueprint() -> None:
    """Structure-attached blueprints require concrete constitution chain ids."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    with pytest.raises(ModelInvariantError, match="structure-attached"):
        ProteinStructure.from_payload(
            constitution=structure.constitution,
            geometry=structure.geometry,
            topology=structure.topology,
            polymer_blueprint=PolymerBlueprint(
                chains=(
                    PolymerChainBlueprint(
                        chain_id=None,
                        residue_slots=(
                            PolymerResidueSlot(sequence_position=1, token="G"),
                        ),
                    ),
                ),
            ),
            provenance=structure.provenance,
        )


def test_protein_structure_chain_selection_subsets_polymer_blueprint() -> None:
    """Selecting chains should subset the attached polymer blueprint."""

    blueprint = PolymerBlueprint(
        chains=(
            PolymerChainBlueprint(
                chain_id="A",
                residue_slots=(PolymerResidueSlot(sequence_position=1, token="G"),),
            ),
            PolymerChainBlueprint(
                chain_id="B",
                residue_slots=(PolymerResidueSlot(sequence_position=1, token="A"),),
            ),
        ),
    )
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId(chain_id="B", seq_num=1),
                        atoms=(atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        polymer_blueprint=blueprint,
    )

    selected = structure.select_chains(("B",))

    assert selected.constitution.chain_ids() == ("B",)
    assert selected.polymer_blueprint == PolymerBlueprint(
        chains=(blueprint.chain("B"),)
    )


def test_structure_normalization_preserves_selected_polymer_blueprint() -> None:
    """Canonical normalization should preserve blueprint alignment.

    This should remain true when normalization selects a subset of chains.
    """

    blueprint = PolymerBlueprint(
        chains=(
            PolymerChainBlueprint(
                chain_id="A",
                residue_slots=(PolymerResidueSlot(sequence_position=1, token="G"),),
            ),
            PolymerChainBlueprint(
                chain_id="B",
                residue_slots=(PolymerResidueSlot(sequence_position=1, token="A"),),
            ),
        ),
    )
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId(chain_id="B", seq_num=1),
                        atoms=(atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        polymer_blueprint=blueprint,
    )

    normalized = apply_structure_normalization_policy(
        structure,
        policy=StructureNormalizationPolicy(selected_chain_ids=("B",)),
    )

    assert normalized.constitution.chain_ids() == ("B",)
    assert normalized.polymer_blueprint == PolymerBlueprint(
        chains=(blueprint.chain("B"),)
    )


def test_request_contracts_normalize_requested_states_and_transforms() -> None:
    repair_refinement = RepairRefinementSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        binding=RecommendedContinuousRelaxationBinding(),
    )
    ingress = ingress_options(
        occupancy_policy=OccupancyPolicy.HIGHEST,
        mutation_policy=MutationPolicy.HIGHEST_OCCUPANCY,
        ligand_policy=LigandPolicy.DROP,
    )
    ingress_normalization_policy = ingress.structure_normalization_policy()
    normalization_policy = StructureNormalizationPolicy(
        selected_chain_ids=("A", "A", "B"),
    )
    requested_goals = RequestedGoalSet(
        whole_structure_requested_goals(
            HydrogenCoverageState.NONE,
        )
    )
    requests = WorkflowTransformRequests(
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
        reference_sidechain_packing=PackingSpec(
            backend_name="faspr",
            scope=PackingScope.FULL,
        ),
        repair_refinement=repair_refinement,
    )

    preserved = transform_requests(
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
        reference_sidechain_packing=requests.reference_sidechain_packing,
        repair_refinement=requests.repair_refinement,
    )

    assert ingress_normalization_policy.drops_ligands()
    assert ingress_normalization_policy.occupancy_policy is OccupancyPolicy.HIGHEST
    assert (
        ingress_normalization_policy.mutation_policy is MutationPolicy.HIGHEST_OCCUPANCY
    )
    assert normalization_policy.selected_chain_ids == ("A", "B")
    assert normalization_policy.selects_chain("A")
    assert not normalization_policy.selects_chain("C")
    assert requests.requests_reference_sidechain_packing()
    assert not requests.requests_committed_sidechain_packing()
    assert requests.repair_refinement is repair_refinement
    assert not requested_goals.requests_whole_structure_hydrogen_population()
    assert preserved.orphan_fragment_policy is OrphanFragmentPolicy.PRESERVE
    assert preserved.repair_refinement is requests.repair_refinement
    assert (
        requested_goals.whole_structure_goal_state_value(
            BackboneHeavyAtomCompletenessState,
        )
        is BackboneHeavyAtomCompletenessState.COMPLETE
    )
    assert (
        requested_goals.whole_structure_goal_state_value(
            SidechainHeavyAtomCompletenessState,
        )
        is SidechainHeavyAtomCompletenessState.COMPLETE
    )


def test_requested_goal_set_separates_clash_scope_axes() -> None:
    """Workflow clash goals should keep local and structure-level axes distinct."""

    local_clash_goal = requested_process_goal(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
        value=ClashState.NONE,
    )
    whole_structure_clash_goal = requested_process_goal(
        scope=WholeStructureScope(),
        value=ClashPresenceState.NONE,
    )
    requested_goals = RequestedGoalSet(
        (local_clash_goal, whole_structure_clash_goal)
    )

    assert requested_goals.requests_whole_structure_clash_absence()
    assert (
        requested_goals.whole_structure_goal_state_value(ClashPresenceState)
        is ClashPresenceState.NONE
    )

    with pytest.raises(ValueError, match="whole-structure clash goals"):
        RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=ClashState.NONE,
                ),
            )
        )

    with pytest.raises(ValueError, match="ClashPresenceState goals"):
        RequestedGoalSet(
            (
                requested_process_goal(
                    scope=ResidueSetScope(
                        residue_ids=(ResidueId(chain_id="A", seq_num=1),)
                    ),
                    value=ClashPresenceState.NONE,
                ),
            )
        )


def test_transform_requests_reject_mixed_reference_and_committed_packing() -> None:
    """Transform requests should reject ambiguous mixed packing modes."""

    with pytest.raises(ValueError, match="must not both be set"):
        WorkflowTransformRequests(
            reference_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
            ),
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.FULL,
            ),
        )


def test_transform_requests_validate_refinement_type() -> None:
    """Transform requests should reject invalid refinement configuration."""

    with pytest.raises(TypeError, match="RepairRefinementSpec"):
        WorkflowTransformRequests(
            repair_refinement=cast(RepairRefinementSpec, "not-a-config")
        )


def test_transform_requests_validate_retained_non_polymer_fallback_policy() -> None:
    """Retained non-polymer fallback policy should stay explicit and typed."""

    assert WorkflowTransformRequests().allow_retained_non_polymer_rdkit_fallback
    assert not WorkflowTransformRequests(
        allow_retained_non_polymer_rdkit_fallback=False
    ).allow_retained_non_polymer_rdkit_fallback

    with pytest.raises(TypeError, match="retained_non_polymer_rdkit_fallback"):
        WorkflowTransformRequests(
            allow_retained_non_polymer_rdkit_fallback=cast(bool, "false")
        )


def test_repair_refinement_spec_validate_binding_type() -> None:
    """RepairRefinementSpec should reject invalid canonical binding variants."""

    with pytest.raises(TypeError, match="continuous-relaxation binding"):
        RepairRefinementSpec(
            scope_spec=LocalScopeSpec.from_residues(
                (ResidueId(chain_id="A", seq_num=1),)
            ),
            binding=cast(
                RecommendedContinuousRelaxationBinding,
                "recommended",
            ),
        )


def test_process_result_tracks_repairs_and_issues() -> None:
    structure = build_structure(
        chains=(),
        source_format=FileFormat.MMCIF,
    )
    repair = RepairEvent.for_residue(
        kind=RepairEventKind.HEAVY_ATOMS_ADDED,
        residue_id=ResidueId(chain_id="A", seq_num=10),
        component_id="ALA",
        atom_names=("CB",),
    )
    issue = ValidationIssue(
        kind=ValidationIssueKind.UNEXPECTED_ATOMS,
        severity=IssueSeverity.WARNING,
        message="unexpected atom",
    )
    result = ProcessResult(
        structure=structure,
        repairs=(repair,),
        issues=(issue,),
        analyses=AnalysisBundle(),
    )

    assert result.repair_count() == 1
    assert result.issue_count() == 1
    assert not result.has_errors()
    assert result.has_warnings()


def test_process_result_exposes_requested_goal_completion_verdict() -> None:
    """Workflow result should expose requested-goal completion on the result."""

    structure = build_structure(
        chains=(),
        source_format=FileFormat.MMCIF,
    )
    requested_state = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    result = ProcessResult(
        structure=structure,
        repairs=(),
        issues=(),
        requested_goal_report=RequestedGoalReport(
            outcomes=(
                RequestedGoalOutcome(
                    requested_goal=requested_state,
                    status=RequestedGoalStatus.UNSUPPORTED,
                ),
            )
        ),
    )

    assert (
        result.requested_goal_completion_verdict()
        is RequestedGoalCompletionVerdict.UNACHIEVED
    )
    assert result.requested_goals_fully_satisfied() is False
    assert result.with_appended_issues(()) is result


def test_process_result_marks_mixed_requested_state_outcomes_as_partial() -> None:
    """Workflow result should expose partial achievement across requested goals."""

    structure = build_structure(
        chains=(),
        source_format=FileFormat.MMCIF,
    )
    hydrogen_request = requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    )
    heavy_request = requested_process_goal(
        scope=WholeStructureScope(),
        value=BackboneHeavyAtomCompletenessState.COMPLETE,
    )
    result = ProcessResult(
        structure=structure,
        repairs=(),
        issues=(),
        requested_goal_report=RequestedGoalReport(
            outcomes=(
                RequestedGoalOutcome(
                    requested_goal=hydrogen_request,
                    status=RequestedGoalStatus.SATISFIED,
                ),
                RequestedGoalOutcome(
                    requested_goal=heavy_request,
                    status=RequestedGoalStatus.BLOCKED,
                ),
            )
        ),
    )

    assert (
        result.requested_goal_completion_verdict()
        is RequestedGoalCompletionVerdict.PARTIALLY_ACHIEVED
    )
    assert result.requested_goals_fully_satisfied() is False


def test_process_result_marks_missing_requested_states_as_not_requested() -> None:
    """Workflow result should expose missing request context explicitly."""

    result = ProcessResult(
        structure=build_structure(
            chains=(),
            source_format=FileFormat.MMCIF,
        ),
        repairs=(),
        issues=(),
    )

    assert (
        result.requested_goal_completion_verdict()
        is RequestedGoalCompletionVerdict.NOT_REQUESTED
    )
    assert result.requested_goals_fully_satisfied() is None
    assert result.with_appended_issues(()) is result


def test_event_scope_supports_cross_residue_provenance() -> None:
    """Repair and issue ontology should support pair-scoped provenance."""

    pair_scope = EventScope.for_residue_pair(
        ResidueId(chain_id="A", seq_num=10),
        ResidueId(chain_id="A", seq_num=11),
    )
    repair = RepairEvent(
        kind=RepairEventKind.LOCAL_REFINEMENT_APPLIED,
        scope=pair_scope,
        residue_impacts=(
            ResidueAtomImpact(
                residue_id=ResidueId(chain_id="A", seq_num=10),
                component_id="ALA",
                atom_names=("C",),
            ),
            ResidueAtomImpact(
                residue_id=ResidueId(chain_id="A", seq_num=11),
                component_id="GLY",
                atom_names=("N",),
            ),
        ),
    )
    issue = ValidationIssue(
        kind=ValidationIssueKind.STERIC_CLASH,
        severity=IssueSeverity.WARNING,
        message="pair clash",
        scope=pair_scope,
    )

    assert pair_scope.kind is EventScopeKind.RESIDUE_PAIR
    assert repair.residue_id is None
    assert repair.component_id is None
    assert repair.atom_names == ()
    assert repair.affects_atom("N")
    assert issue.residue_id is None
