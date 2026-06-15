"""Scenario-classified fixture matrix for capability-aware workflow planning.

This module is the canonical metadata registry for scenario-planning coverage.
It deliberately owns only fixture classification and planner expectation tags.
It must not grow benchmark execution logic, planner runtime policy, or ad hoc
builders.
"""

from dataclasses import dataclass
from enum import Enum

from protrepair.workflow.planning.capability import (
    WorkflowActionCapability,
    WorkflowActionDonorRequirement,
    WorkflowActionEffectClass,
    WorkflowActionLocality,
    WorkflowCapabilityDeficitFamily,
)


class ScenarioFixtureKind(str, Enum):
    """Source kind for one scenario-classified fixture entry."""

    LITERATURE_LOCAL = "literature_local"
    REPRESENTATIVE_WHOLE = "representative_whole"
    SYNTHETIC_UNIT = "synthetic_unit"


class ScenarioFixtureStatus(str, Enum):
    """Intended use status for one scenario-classified fixture entry."""

    ACTIVE = "active"
    EXPLORATORY = "exploratory"
    NEGATIVE_CONTROL = "negative_control"


class ScenarioDefectTag(str, Enum):
    """Orthogonal defect families used by scenario planning."""

    MISSING_SIDECHAIN_ATOMS = "missing_sidechain_atoms"
    MISSING_MAINCHAIN_ATOMS = "missing_mainchain_atoms"
    MISSING_INTERNAL_RESIDUE_SPAN = "missing_internal_residue_span"
    MISSING_PREFIX_TERMINAL_SPAN = "missing_prefix_terminal_span"
    MISSING_SUFFIX_TERMINAL_SPAN = "missing_suffix_terminal_span"
    MISSING_COMPONENT_SUPPORT = "missing_component_support"
    MISSING_TOPOLOGY_SUPPORT = "missing_topology_support"
    MISSING_HYDROGENS = "missing_hydrogens"
    PROTEIN_SELF_CLASH = "protein_self_clash"
    SIDECHAIN_ORIENTATION_ANOMALY = "sidechain_orientation_anomaly"
    STEREOCHEMISTRY_SENSITIVE_PATHOLOGY = "stereochemistry_sensitive_pathology"
    LIGAND_AWARE_INTERACTION_CLASH = "ligand_aware_interaction_clash"
    POST_HYDROGEN_INTERACTION_CLASH = "post_hydrogen_interaction_clash"


class ScenarioContextTag(str, Enum):
    """Non-defect planning contexts carried by fixture metadata."""

    APO = "apo"
    HOLO_KEEP_LIGAND = "holo_keep_ligand"
    PROTONATE_HISTIDINES = "protonate_histidines"
    PREFIX_TERMINAL_GAP_OPT_IN = "prefix_terminal_gap_opt_in"
    SUFFIX_TERMINAL_GAP_OPT_IN = "suffix_terminal_gap_opt_in"


class ScenarioPlanningPhase(str, Enum):
    """Planner phases that should consume one scenario entry."""

    COVERAGE = "coverage"
    CHEMISTRY_AUGMENTATION = "chemistry_augmentation"
    INTRINSIC_GEOMETRY_CORRECTION = "intrinsic_geometry_correction"
    INTERACTION_AWARE_CORRECTION = "interaction_aware_correction"


@dataclass(frozen=True, slots=True)
class ScenarioCapabilityPlannerExpectation:
    """Capability-planner expectation attached to one scenario fixture entry."""

    reducible_deficit_families: tuple[WorkflowCapabilityDeficitFamily, ...]
    preferred_localities: tuple[WorkflowActionLocality, ...]
    admissible_effect_classes: tuple[WorkflowActionEffectClass, ...]
    admissible_donor_requirements: tuple[WorkflowActionDonorRequirement, ...]
    requires_holo_interaction_policy: bool = False
    expects_post_augmentation_correction: bool = False
    expects_no_action_by_default: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reducible_deficit_families",
            tuple(self.reducible_deficit_families),
        )
        object.__setattr__(
            self,
            "preferred_localities",
            tuple(self.preferred_localities),
        )
        object.__setattr__(
            self,
            "admissible_effect_classes",
            tuple(self.admissible_effect_classes),
        )
        object.__setattr__(
            self,
            "admissible_donor_requirements",
            tuple(self.admissible_donor_requirements),
        )

    def matches_capability(
        self,
        capability: WorkflowActionCapability,
    ) -> bool:
        """Return whether one workflow capability fits this scenario expectation."""

        return (
            any(
                capability.can_reduce_deficit_family(deficit_family)
                for deficit_family in self.reducible_deficit_families
            )
            and capability.effect_class in self.admissible_effect_classes
            and any(
                capability.supports_locality(locality)
                for locality in self.preferred_localities
            )
            and capability.donor_requirement in self.admissible_donor_requirements
        )


@dataclass(frozen=True, slots=True)
class ScenarioFixtureMatrixEntry:
    """One concrete fixture or synthetic scenario in the planning matrix."""

    fixture_id: str
    title: str
    kind: ScenarioFixtureKind
    status: ScenarioFixtureStatus
    fixture_refs: tuple[str, ...]
    defect_tags: tuple[ScenarioDefectTag, ...]
    context_tags: tuple[ScenarioContextTag, ...]
    planning_phases: tuple[ScenarioPlanningPhase, ...]
    description: str
    capability_expectation: ScenarioCapabilityPlannerExpectation
    evidence_reference: str | None = None


def _planner_expectation(
    *,
    reducible_deficit_families: tuple[WorkflowCapabilityDeficitFamily, ...],
    preferred_localities: tuple[WorkflowActionLocality, ...],
    admissible_effect_classes: tuple[WorkflowActionEffectClass, ...],
    admissible_donor_requirements: tuple[WorkflowActionDonorRequirement, ...],
    requires_holo_interaction_policy: bool = False,
    expects_post_augmentation_correction: bool = False,
    expects_no_action_by_default: bool = False,
) -> ScenarioCapabilityPlannerExpectation:
    """Return one normalized capability-planner expectation for a matrix entry."""

    return ScenarioCapabilityPlannerExpectation(
        reducible_deficit_families=reducible_deficit_families,
        preferred_localities=preferred_localities,
        admissible_effect_classes=admissible_effect_classes,
        admissible_donor_requirements=admissible_donor_requirements,
        requires_holo_interaction_policy=requires_holo_interaction_policy,
        expects_post_augmentation_correction=expects_post_augmentation_correction,
        expects_no_action_by_default=expects_no_action_by_default,
    )


LOCAL_INTRINSIC_CORRECTION_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY,),
    preferred_localities=(
        WorkflowActionLocality.LOCAL_SCOPE,
        WorkflowActionLocality.RESIDUE_SET,
    ),
    admissible_effect_classes=(
        WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    ),
    admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
)

HOLO_INTERACTION_CORRECTION_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.INTERACTION,),
    preferred_localities=(
        WorkflowActionLocality.LOCAL_SCOPE,
        WorkflowActionLocality.RESIDUE_SET,
    ),
    admissible_effect_classes=(
        WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    ),
    admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
    requires_holo_interaction_policy=True,
)

ATOM_COVERAGE_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.ATOM_COVERAGE,),
    preferred_localities=(WorkflowActionLocality.RESIDUE_SET,),
    admissible_effect_classes=(WorkflowActionEffectClass.AUGMENTS_ABSENCE,),
    admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
)

CHEMISTRY_AUGMENTATION_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,),
    preferred_localities=(WorkflowActionLocality.RESIDUE_SET,),
    admissible_effect_classes=(WorkflowActionEffectClass.AUGMENTS_ABSENCE,),
    admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
)

SPAN_DONOR_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(WorkflowCapabilityDeficitFamily.COVERAGE_GAP,),
    preferred_localities=(WorkflowActionLocality.RESIDUE_SPAN,),
    admissible_effect_classes=(WorkflowActionEffectClass.AUGMENTS_ABSENCE,),
    admissible_donor_requirements=(
        WorkflowActionDonorRequirement.EXTERNAL_DONOR,
    ),
)

CHEMISTRY_AND_INTRINSIC_INTERPLAY_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(
        WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
        WorkflowCapabilityDeficitFamily.INTRINSIC_GEOMETRY,
    ),
    preferred_localities=(
        WorkflowActionLocality.RESIDUE_SET,
        WorkflowActionLocality.LOCAL_SCOPE,
    ),
    admissible_effect_classes=(
        WorkflowActionEffectClass.AUGMENTS_ABSENCE,
        WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    ),
    admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
)

POST_HYDROGEN_HOLO_INTERACTION_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(
        WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
        WorkflowCapabilityDeficitFamily.INTERACTION,
    ),
    preferred_localities=(
        WorkflowActionLocality.RESIDUE_SET,
        WorkflowActionLocality.LOCAL_SCOPE,
    ),
    admissible_effect_classes=(
        WorkflowActionEffectClass.AUGMENTS_ABSENCE,
        WorkflowActionEffectClass.REVISES_PRESENT_GEOMETRY,
    ),
    admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
    requires_holo_interaction_policy=True,
    expects_post_augmentation_correction=True,
)

NO_ACTION_BASELINE_EXPECTATION = _planner_expectation(
    reducible_deficit_families=(),
    preferred_localities=(),
    admissible_effect_classes=(),
    admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
    expects_no_action_by_default=True,
)


SCENARIO_FIXTURE_MATRIX: dict[str, ScenarioFixtureMatrixEntry] = {
    "1bkr-thr101": ScenarioFixtureMatrixEntry(
        fixture_id="1bkr-thr101",
        title="1BKR Thr101 local side-chain misfit",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]',
        ),
        defect_tags=(ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description=(
            "Literature-backed local side-chain misfit with clashes and wrong "
            "branch assignment."
        ),
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1bkr-his42": ScenarioFixtureMatrixEntry(
        fixture_id="1bkr-his42",
        title="1BKR His42 histidine-flip neighborhood",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_BENCHMARK_CASES["1bkr-his42"]',
        ),
        defect_tags=(ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description="Hydrogen-sensitive histidine orientation cleanup case.",
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1lxa-his125": ScenarioFixtureMatrixEntry(
        fixture_id="1lxa-his125",
        title="1LXA His125 apo LpxA histidine flip",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_BENCHMARK_CASES["1lxa-his125"]',
        ),
        defect_tags=(ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,),
        context_tags=(ScenarioContextTag.APO,),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description="Ligand-free histidine orientation cleanup benchmark.",
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1uaq-gln55": ScenarioFixtureMatrixEntry(
        fixture_id="1uaq-gln55",
        title="1UAQ Gln55 ligand-free amide flip",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_BENCHMARK_CASES["1uaq-gln55"]',
        ),
        defect_tags=(ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,),
        context_tags=(ScenarioContextTag.APO,),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description="Ligand-free glutamine amide orientation cleanup benchmark.",
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1xgo-leu253": ScenarioFixtureMatrixEntry(
        fixture_id="1xgo-leu253",
        title="1XGO Leu253 distorted helix neighborhood",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_BENCHMARK_CASES["1xgo-leu253"]',
            'tests/support/refinement_cases.py:REFINEMENT_FIXTURE_SOURCES["1xgs-leu253-reference"]',
        ),
        defect_tags=(
            ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,
            ScenarioDefectTag.PROTEIN_SELF_CLASH,
        ),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description=(
            "Low-resolution local rotamer and helix distortion case with paired "
            "higher-quality reference neighborhood."
        ),
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "3g8l-asn182": ScenarioFixtureMatrixEntry(
        fixture_id="3g8l-asn182",
        title="3G8L Asn182 helix-cap contact pathology",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_BENCHMARK_CASES["3g8l-asn182"]',
        ),
        defect_tags=(ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description="Literature-backed local contact improvement benchmark.",
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1ywr-his148": ScenarioFixtureMatrixEntry(
        fixture_id="1ywr-his148",
        title="1YWR His148 active-site histidine flip",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_BENCHMARK_CASES["1ywr-his148"]',
        ),
        defect_tags=(
            ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,
            ScenarioDefectTag.STEREOCHEMISTRY_SENSITIVE_PATHOLOGY,
        ),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description="Active-site histidine orientation cleanup benchmark.",
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1jd0-gln92": ScenarioFixtureMatrixEntry(
        fixture_id="1jd0-gln92",
        title="1JD0 Gln92 ligand-adjacent amide flip",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.EXPLORATORY,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_FIXTURE_SOURCES["1jd0-gln92"]',
        ),
        defect_tags=(
            ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,
            ScenarioDefectTag.LIGAND_AWARE_INTERACTION_CLASH,
        ),
        context_tags=(ScenarioContextTag.HOLO_KEEP_LIGAND,),
        planning_phases=(ScenarioPlanningPhase.INTERACTION_AWARE_CORRECTION,),
        capability_expectation=HOLO_INTERACTION_CORRECTION_EXPECTATION,
        description=(
            "Ligand-adjacent exploratory amide-flip case currently blocked by "
            "unsupported local ligand chemistry."
        ),
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1ywr-asn155": ScenarioFixtureMatrixEntry(
        fixture_id="1ywr-asn155",
        title="1YWR Asn155 ligand-adjacent amide flip",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.EXPLORATORY,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_FIXTURE_SOURCES["1ywr-asn155"]',
            'tests/support/correction_state_registry.py:CORRECTION_STATE_CASES["chemistry-preparation-amide-flip"]',
        ),
        defect_tags=(
            ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,
            ScenarioDefectTag.LIGAND_AWARE_INTERACTION_CLASH,
        ),
        context_tags=(ScenarioContextTag.HOLO_KEEP_LIGAND,),
        planning_phases=(
            ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,
            ScenarioPlanningPhase.INTERACTION_AWARE_CORRECTION,
        ),
        capability_expectation=POST_HYDROGEN_HOLO_INTERACTION_EXPECTATION,
        description=(
            "Ligand-adjacent exploratory amide-flip case that already anchors "
            "chemistry-preparation state coverage."
        ),
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "2q6f-cys143-pje": ScenarioFixtureMatrixEntry(
        fixture_id="2q6f-cys143-pje",
        title="2Q6F covalent-link negative control",
        kind=ScenarioFixtureKind.LITERATURE_LOCAL,
        status=ScenarioFixtureStatus.NEGATIVE_CONTROL,
        fixture_refs=(
            'tests/support/refinement_cases.py:REFINEMENT_FIXTURE_SOURCES["2q6f-cys143-pje"]',
        ),
        defect_tags=(ScenarioDefectTag.MISSING_TOPOLOGY_SUPPORT,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,),
        capability_expectation=CHEMISTRY_AUGMENTATION_EXPECTATION,
        description=(
            "Negative control for missing linkage chemistry and unsupported local "
            "topology."
        ),
        evidence_reference="tests/fixtures/pdb/refinement/README.md",
    ),
    "1aho-heavy-default": ScenarioFixtureMatrixEntry(
        fixture_id="1aho-heavy-default",
        title="1AHO apo whole-structure baseline",
        kind=ScenarioFixtureKind.REPRESENTATIVE_WHOLE,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/representative_cases.py:REPRESENTATIVE_CASES["1aho-heavy-default"]',
        ),
        defect_tags=(),
        context_tags=(ScenarioContextTag.APO,),
        planning_phases=(),
        capability_expectation=NO_ACTION_BASELINE_EXPECTATION,
        description="Representative apo whole-structure baseline with no ligand.",
        evidence_reference="tests/support/representative_cases.py",
    ),
    "1aho-hydrogen-default": ScenarioFixtureMatrixEntry(
        fixture_id="1aho-hydrogen-default",
        title="1AHO apo whole-structure hydrogenation",
        kind=ScenarioFixtureKind.REPRESENTATIVE_WHOLE,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/representative_cases.py:REPRESENTATIVE_CASES["1aho-hydrogen-default"]',
        ),
        defect_tags=(ScenarioDefectTag.MISSING_HYDROGENS,),
        context_tags=(ScenarioContextTag.APO,),
        planning_phases=(ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,),
        capability_expectation=CHEMISTRY_AUGMENTATION_EXPECTATION,
        description="Representative apo whole-structure hydrogenation path.",
        evidence_reference="tests/support/representative_cases.py",
    ),
    "1cjc-heavy-keep-ligand": ScenarioFixtureMatrixEntry(
        fixture_id="1cjc-heavy-keep-ligand",
        title="1CJC holo whole-structure keep-ligand baseline",
        kind=ScenarioFixtureKind.REPRESENTATIVE_WHOLE,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/representative_cases.py:REPRESENTATIVE_CASES["1cjc-heavy-keep-ligand"]',
        ),
        defect_tags=(),
        context_tags=(ScenarioContextTag.HOLO_KEEP_LIGAND,),
        planning_phases=(),
        capability_expectation=NO_ACTION_BASELINE_EXPECTATION,
        description="Representative holo whole-structure baseline with ligand kept.",
        evidence_reference="tests/support/representative_cases.py",
    ),
    "1cjc-hydrogen-keep-ligand": ScenarioFixtureMatrixEntry(
        fixture_id="1cjc-hydrogen-keep-ligand",
        title="1CJC holo whole-structure hydrogenation with ligand kept",
        kind=ScenarioFixtureKind.REPRESENTATIVE_WHOLE,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/representative_cases.py:REPRESENTATIVE_CASES["1cjc-hydrogen-keep-ligand"]',
        ),
        defect_tags=(ScenarioDefectTag.MISSING_HYDROGENS,),
        context_tags=(ScenarioContextTag.HOLO_KEEP_LIGAND,),
        planning_phases=(ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,),
        capability_expectation=_planner_expectation(
            reducible_deficit_families=(
                WorkflowCapabilityDeficitFamily.CHEMISTRY_READINESS,
            ),
            preferred_localities=(WorkflowActionLocality.RESIDUE_SET,),
            admissible_effect_classes=(
                WorkflowActionEffectClass.AUGMENTS_ABSENCE,
            ),
            admissible_donor_requirements=(WorkflowActionDonorRequirement.NONE,),
            requires_holo_interaction_policy=True,
        ),
        description=(
            "Representative holo whole-structure hydrogenation path under "
            "keep-ligand policy."
        ),
        evidence_reference="tests/support/representative_cases.py",
    ),
    "1afc-hydrogen-his-protonated": ScenarioFixtureMatrixEntry(
        fixture_id="1afc-hydrogen-his-protonated",
        title="1AFC representative histidine-protonation workflow",
        kind=ScenarioFixtureKind.REPRESENTATIVE_WHOLE,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/representative_cases.py:REPRESENTATIVE_CASES["1afc-hydrogen-his-protonated"]',
        ),
        defect_tags=(ScenarioDefectTag.MISSING_HYDROGENS,),
        context_tags=(ScenarioContextTag.PROTONATE_HISTIDINES,),
        planning_phases=(ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,),
        capability_expectation=CHEMISTRY_AUGMENTATION_EXPECTATION,
        description=(
            "Representative whole-structure hydrogenation path with explicit "
            "histidine protonation request."
        ),
        evidence_reference="tests/support/representative_cases.py",
    ),
    "synthetic-backbone-gap-existing-residue": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-backbone-gap-existing-residue",
        title="Synthetic existing-residue backbone gap",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/correction_state_registry.py:CORRECTION_STATE_CASES["backbone-incomplete-local-stop"]',
        ),
        defect_tags=(ScenarioDefectTag.MISSING_MAINCHAIN_ATOMS,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.COVERAGE,),
        capability_expectation=ATOM_COVERAGE_EXPECTATION,
        description=(
            "Synthetic existing-residue case with missing backbone atoms and no "
            "donor-based span semantics."
        ),
        evidence_reference="tests/support/correction_state_registry.py",
    ),
    "synthetic-sidechain-gap-existing-residue": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-sidechain-gap-existing-residue",
        title="Synthetic existing-residue side-chain gap",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            "tests/unit/test_workflow_planning.py::test_plan_workflow_actions_emits_heavy_action_before_hydrogen",
        ),
        defect_tags=(ScenarioDefectTag.MISSING_SIDECHAIN_ATOMS,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.COVERAGE,),
        capability_expectation=ATOM_COVERAGE_EXPECTATION,
        description=(
            "Synthetic existing-residue case with intact backbone and missing "
            "side-chain heavy atoms only."
        ),
        evidence_reference="tests/unit/test_workflow_planning.py",
    ),
    "synthetic-unsupported-component-stop": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-unsupported-component-stop",
        title="Synthetic unsupported-component chemistry blocker",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/correction_state_registry.py:CORRECTION_STATE_CASES["unsupported-component-local-stop"]',
        ),
        defect_tags=(ScenarioDefectTag.MISSING_COMPONENT_SUPPORT,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,),
        capability_expectation=CHEMISTRY_AUGMENTATION_EXPECTATION,
        description=(
            "Synthetic chemistry-readiness blocker for unsupported residue "
            "templates."
        ),

        evidence_reference="tests/support/correction_state_registry.py",
    ),
    "synthetic-topology-coordinate-blocked": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-topology-coordinate-blocked",
        title="Synthetic clashy but topology-ready local correction",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            'tests/support/correction_state_registry.py:CORRECTION_STATE_CASES["topology-blocked-preparation"]',
        ),
        defect_tags=(ScenarioDefectTag.PROTEIN_SELF_CLASH,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,),
        capability_expectation=LOCAL_INTRINSIC_CORRECTION_EXPECTATION,
        description=(
            "Synthetic local clash case where graph-realizable topology remains "
            "available and intrinsic correction stays legal."
        ),
        evidence_reference="tests/support/correction_state_registry.py",
    ),
    "synthetic-internal-blueprint-gap": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-internal-blueprint-gap",
        title="Synthetic internal missing-residue span",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            "tests/unit/test_external_reference_workflow.py::test_build_alphafold_span_reconstruction_specs_infers_internal_deletions",
        ),
        defect_tags=(ScenarioDefectTag.MISSING_INTERNAL_RESIDUE_SPAN,),
        context_tags=(),
        planning_phases=(ScenarioPlanningPhase.COVERAGE,),
        capability_expectation=SPAN_DONOR_EXPECTATION,
        description=(
            "Synthetic blueprint-coverage gap case for donor-based internal span "
            "reconstruction."
        ),
        evidence_reference="tests/unit/test_external_reference_workflow.py",
    ),
    "synthetic-prefix-terminal-blueprint-gap": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-prefix-terminal-blueprint-gap",
        title="Synthetic prefix-terminal missing-residue span",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            "tests/unit/test_external_reference_workflow.py::test_build_alphafold_span_reconstruction_specs_can_select_prefix_terminal_gap",
        ),
        defect_tags=(ScenarioDefectTag.MISSING_PREFIX_TERMINAL_SPAN,),
        context_tags=(ScenarioContextTag.PREFIX_TERMINAL_GAP_OPT_IN,),
        planning_phases=(ScenarioPlanningPhase.COVERAGE,),
        capability_expectation=SPAN_DONOR_EXPECTATION,
        description=(
            "Synthetic terminal-gap case for prefix opt-in donor-based span "
            "reconstruction."
        ),
        evidence_reference="tests/unit/test_external_reference_workflow.py",
    ),
    "synthetic-suffix-terminal-blueprint-gap": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-suffix-terminal-blueprint-gap",
        title="Synthetic suffix-terminal missing-residue span",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            "tests/unit/test_external_reference_workflow.py::test_build_alphafold_span_reconstruction_specs_can_select_suffix_terminal_gap",
        ),
        defect_tags=(ScenarioDefectTag.MISSING_SUFFIX_TERMINAL_SPAN,),
        context_tags=(ScenarioContextTag.SUFFIX_TERMINAL_GAP_OPT_IN,),
        planning_phases=(ScenarioPlanningPhase.COVERAGE,),
        capability_expectation=SPAN_DONOR_EXPECTATION,
        description=(
            "Synthetic terminal-gap case for suffix opt-in donor-based span "
            "reconstruction."
        ),
        evidence_reference="tests/unit/test_external_reference_workflow.py",
    ),
    "synthetic-ligand-clashing-serine-hydrogen": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-ligand-clashing-serine-hydrogen",
        title="Synthetic ligand-clashing post-hydrogen cleanup case",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.ACTIVE,
        fixture_refs=(
            "tests/unit/test_hydrogen_cleanup.py::test_cleanup_residual_hydrogen_clashes_reorients_ligand_clashing_serine_hydrogen",
            "tests/unit/test_hydrogen_cleanup.py::test_add_hydrogens_runs_residual_cleanup_for_ligand_clashes",
        ),
        defect_tags=(
            ScenarioDefectTag.LIGAND_AWARE_INTERACTION_CLASH,
            ScenarioDefectTag.POST_HYDROGEN_INTERACTION_CLASH,
        ),
        context_tags=(ScenarioContextTag.HOLO_KEEP_LIGAND,),
        planning_phases=(
            ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,
            ScenarioPlanningPhase.INTERACTION_AWARE_CORRECTION,
        ),
        capability_expectation=POST_HYDROGEN_HOLO_INTERACTION_EXPECTATION,
        description=(
            "Synthetic ligand-aware interaction case for post-hydrogen cleanup and "
            "interaction-aware correction."
        ),
        evidence_reference="tests/unit/test_hydrogen_cleanup.py",
    ),
    "synthetic-unresolved-ligand-clashing-serine-hydrogen": ScenarioFixtureMatrixEntry(
        fixture_id="synthetic-unresolved-ligand-clashing-serine-hydrogen",
        title="Synthetic unresolved post-hydrogen ligand clash negative control",
        kind=ScenarioFixtureKind.SYNTHETIC_UNIT,
        status=ScenarioFixtureStatus.NEGATIVE_CONTROL,
        fixture_refs=(
            "tests/unit/test_hydrogen_cleanup.py::test_cleanup_residual_hydrogen_clashes_reports_unresolved_targets",
        ),
        defect_tags=(ScenarioDefectTag.POST_HYDROGEN_INTERACTION_CLASH,),
        context_tags=(ScenarioContextTag.HOLO_KEEP_LIGAND,),
        planning_phases=(ScenarioPlanningPhase.INTERACTION_AWARE_CORRECTION,),
        capability_expectation=HOLO_INTERACTION_CORRECTION_EXPECTATION,
        description=(
            "Synthetic negative-control interaction case where every hydrogen "
            "orientation still clashes with the ligand environment."
        ),
        evidence_reference="tests/unit/test_hydrogen_cleanup.py",
    ),
}

REQUIRED_SCENARIO_DEFECT_COVERAGE: frozenset[ScenarioDefectTag] = frozenset(
    {
        ScenarioDefectTag.MISSING_SIDECHAIN_ATOMS,
        ScenarioDefectTag.MISSING_MAINCHAIN_ATOMS,
        ScenarioDefectTag.MISSING_INTERNAL_RESIDUE_SPAN,
        ScenarioDefectTag.MISSING_PREFIX_TERMINAL_SPAN,
        ScenarioDefectTag.MISSING_SUFFIX_TERMINAL_SPAN,
        ScenarioDefectTag.MISSING_COMPONENT_SUPPORT,
        ScenarioDefectTag.MISSING_TOPOLOGY_SUPPORT,
        ScenarioDefectTag.MISSING_HYDROGENS,
        ScenarioDefectTag.PROTEIN_SELF_CLASH,
        ScenarioDefectTag.SIDECHAIN_ORIENTATION_ANOMALY,
        ScenarioDefectTag.STEREOCHEMISTRY_SENSITIVE_PATHOLOGY,
        ScenarioDefectTag.LIGAND_AWARE_INTERACTION_CLASH,
        ScenarioDefectTag.POST_HYDROGEN_INTERACTION_CLASH,
    }
)

REQUIRED_SCENARIO_CONTEXT_COVERAGE: frozenset[ScenarioContextTag] = frozenset(
    {
        ScenarioContextTag.APO,
        ScenarioContextTag.HOLO_KEEP_LIGAND,
        ScenarioContextTag.PROTONATE_HISTIDINES,
        ScenarioContextTag.PREFIX_TERMINAL_GAP_OPT_IN,
        ScenarioContextTag.SUFFIX_TERMINAL_GAP_OPT_IN,
    }
)

REQUIRED_SCENARIO_PHASE_COVERAGE: frozenset[ScenarioPlanningPhase] = frozenset(
    {
        ScenarioPlanningPhase.COVERAGE,
        ScenarioPlanningPhase.CHEMISTRY_AUGMENTATION,
        ScenarioPlanningPhase.INTRINSIC_GEOMETRY_CORRECTION,
        ScenarioPlanningPhase.INTERACTION_AWARE_CORRECTION,
    }
)


def collected_scenario_defect_tags(
    *,
    allowed_statuses: frozenset[ScenarioFixtureStatus] | None = None,
) -> frozenset[ScenarioDefectTag]:
    """Return the defect tags covered by the scenario fixture matrix."""

    statuses = (
        frozenset(ScenarioFixtureStatus)
        if allowed_statuses is None
        else allowed_statuses
    )
    return frozenset(
        defect_tag
        for entry in SCENARIO_FIXTURE_MATRIX.values()
        if entry.status in statuses
        for defect_tag in entry.defect_tags
    )


def collected_scenario_context_tags(
    *,
    allowed_statuses: frozenset[ScenarioFixtureStatus] | None = None,
) -> frozenset[ScenarioContextTag]:
    """Return the planning-context tags covered by the scenario fixture matrix."""

    statuses = (
        frozenset(ScenarioFixtureStatus)
        if allowed_statuses is None
        else allowed_statuses
    )
    return frozenset(
        context_tag
        for entry in SCENARIO_FIXTURE_MATRIX.values()
        if entry.status in statuses
        for context_tag in entry.context_tags
    )


def collected_scenario_phase_tags(
    *,
    allowed_statuses: frozenset[ScenarioFixtureStatus] | None = None,
) -> frozenset[ScenarioPlanningPhase]:
    """Return the planner phases covered by the scenario fixture matrix."""

    statuses = (
        frozenset(ScenarioFixtureStatus)
        if allowed_statuses is None
        else allowed_statuses
    )
    return frozenset(
        phase_tag
        for entry in SCENARIO_FIXTURE_MATRIX.values()
        if entry.status in statuses
        for phase_tag in entry.planning_phases
    )
